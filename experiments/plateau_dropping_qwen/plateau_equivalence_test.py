# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415
"""Plateau Dropping in the Qwen trainer is the Grug rule: a multi-step simulation fed through both.

Side G is Grug itself: ``train._track_loss_sharded`` (the real shard_map over a 4-device row-sharded mesh, psum over
``data``) plus a line-for-line transcription of ``train.run_grug_local``'s host block (keys, gather, step, all-gather,
``loss_now``, trace, ``update_ema`` or ``decide`` + ``update``). Side Q is the Qwen trainer's code: ``objective``
(``tracked_loss`` on the whole batch, as GSPMD runs it) and ``track_step_inputs`` / ``track_step_record``, with a
save / restore of the history in the middle. Every step must give the same keep mask, loss, statistics and
per-token gradient, and the final history tables must be bit-identical.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.june_tpu_67b_a2b.moe import token_track as tt
from experiments.june_tpu_67b_a2b.moe import train as grug_train
from experiments.plateau_dropping_qwen import qwen_sft as qs

NUM_DOCS, MAX_LEN, ROWS, SEQ, STEPS, SAVE_AT = 12, 64, 4, 48, 26, 11
TOL, FLOOR = 0.05, 0.05
PATTERNS = ("plateau", "learned", "rise_reentry", "fall_reentry", "decay", "noisy")


# ------------------------------------------------------------------------------------------------ the simulated "model"


def doc_layout(rng):
    """Per document: length and which positions carry loss weight (a user prefix, then assistant tokens)."""
    docs = []
    for _ in range(NUM_DOCS):
        n = int(rng.integers(8, 22))
        w = np.ones(n, np.float32)
        w[: int(rng.integers(1, 4))] = 0.0  # prompt tokens
        w[int(rng.integers(4, n - 1))] = 0.0  # one masked token inside (e.g. a tool observation token)
        docs.append(w)
    return docs


def token_loss(doc: int, offset: int, visit: int) -> float:
    """Loss of a token at its ``visit``-th observation (1-based); patterns cover flat, re-entry by rise / fall, etc."""
    kind = PATTERNS[(doc * 7 + offset) % len(PATTERNS)]
    base = 0.4 + 0.07 * ((doc * 13 + offset * 5) % 17)
    jitter = 0.004 * np.sin(1.3 * visit + offset)  # relative change << TOL
    if kind == "plateau":
        return base * (1.0 + jitter)
    if kind == "learned":
        return 0.02 * (1.0 + jitter)
    if kind == "rise_reentry":  # flat for visits 1..4, jumps up at 5, flat after
        return base * (1.0 + jitter) * (1.6 if visit >= 5 else 1.0)
    if kind == "fall_reentry":  # flat, then falls by 40 % at visit 5, flat after
        return base * (1.0 + jitter) * (0.6 if visit >= 5 else 1.0)
    if kind == "decay":
        return base * (0.7**visit)
    return base * (1.0 + 0.3 * np.sin(2.1 * visit + doc))  # noisy


def make_batch(rng, docs, step):
    """Pack documents into ROWS rows of SEQ positions (segment id = document, -1 padding), Grug's causal weights.

    Like the packed cache, a document never sits twice in one row and a batch draws distinct documents, except at
    step 7 where one row is repeated (the epoch-boundary duplicate that ``TokenHistory.update`` reports)."""
    seg = np.full((ROWS, SEQ), -1, np.int32)
    w = np.zeros((ROWS, SEQ), np.float32)
    order = iter(rng.permutation(NUM_DOCS).tolist())
    for r in range(ROWS):
        pos = 0
        while True:
            d = next(order, None)
            if d is None or pos + len(docs[d]) > SEQ:
                break
            n = len(docs[d])
            seg[r, pos : pos + n] = d
            # the causal shift: position t is weighted by the mask of token t+1 within the same document
            dw = np.roll(docs[d], -1)
            dw[-1] = 0.0
            w[r, pos : pos + n] = dw
            pos += n
    if step == 7:  # the same document twice in one batch: duplicate keys
        seg[1], w[1] = seg[0], w[0]
    w[:, -1] = 0.0
    tokens = (seg.astype(np.int64) * 31 + np.arange(SEQ)[None, :] * 3 + 5).astype(np.int32)
    return seg, w, tokens


def per_pos_for(seg, w, history):
    """The model's weighted per-position loss given how often each token has been seen (visit = count + 1)."""
    keys = history.keys(seg, w)
    pp = np.zeros(seg.shape, np.float32)
    for (r, s), k in np.ndenumerate(keys):
        if k < 0:
            continue
        visit = int(history.count[k]) + 1
        pp[r, s] = np.float32(token_loss(int(k // MAX_LEN), int(k % MAX_LEN), visit)) * w[r, s]
    return pp


# ------------------------------------------------------------------------------------------------ side G: Grug


def grug_mesh():
    return jax.make_mesh((4,), ("data",), axis_types=(AxisType.Explicit,))


def grug_device_step(mesh, settings, pp, w, p1, p2, cnt, prev_dropped):
    """``train._track_loss_sharded`` on row-sharded inputs (the shard_map / psum path Grug trains with)."""
    row = NamedSharding(mesh, P("data"))
    args = [jax.device_put(jnp.asarray(a), row) for a in (pp, w, p1, p2, cnt, prev_dropped)]

    def f(per_pos, *rest):
        return grug_train._track_loss_sharded(
            per_pos,
            *rest,
            tol=settings.tol,
            floor=settings.floor,
            ema_alpha=settings.ema_alpha,
            reentry=settings.reentry,
            drop_weight=settings.drop_weight,
        )

    with set_mesh(mesh):
        loss, stats = jax.jit(f)(*args)
        grad = jax.jit(jax.grad(lambda x, *r: f(x, *r)[0]))(*args)
    return float(loss), {k: float(v) for k, v in stats.items()}, np.asarray(grad)


def grug_host_step(track_history, track, trace, current_step, seg_np, w_np, run_device):
    """train.run_grug_local lines 1489-1523, transcribed (allgathers are identities on one host)."""
    keys = track_history.keys(seg_np, w_np)
    p1, p2, cnt, prev_dropped = track_history.gather(keys)
    out = run_device(p1, p2, cnt, prev_dropped)  # train_step(state, batch, p1, p2, cnt, prev_dropped)
    pp_np = out["per_pos"]
    loss_now = np.where(w_np > 0, pp_np / np.where(w_np > 0, w_np, 1.0), np.nan)
    if trace is not None:
        tok_np = out["tokens"]
        trace.record(current_step, keys, np.roll(tok_np, -1, axis=-1), loss_now, p1, p2, cnt)
    if track.ema_alpha > 0:
        diag = track_history.update_ema(keys, loss_now, floor=track.floor, alpha=track.ema_alpha)
    else:
        dropped_now = tt.decide(
            p1, p2, cnt, prev_dropped, loss_now, tol=track.tol, floor=track.floor, reentry=track.reentry
        )
        diag = track_history.update(keys, loss_now, dropped_now)
    return out, diag, (keys, p1, p2, cnt, prev_dropped)


# ------------------------------------------------------------------------------------------------ side Q: Qwen trainer


def qwen_mesh():
    """Levanter's default (Auto-axis) mesh: the Qwen step runs ``tracked_loss`` under GSPMD on row-sharded arrays."""
    return jax.make_mesh((4,), ("data",), axis_types=(AxisType.Auto,))


def qwen_device_step(mesh, settings, pp, w, hist):
    """The Qwen step's objective (``qwen_sft.objective``) and its gradient w.r.t. the per-position loss."""
    row = NamedSharding(mesh, P("data"))
    put = lambda a: jax.device_put(jnp.asarray(a), row)  # noqa: E731
    hist = tuple(put(h) for h in hist)
    w_dev = put(w)
    f = lambda x: qs.objective(x, w_dev, hist, settings)  # noqa: E731
    with set_mesh(mesh):
        loss, metrics = jax.jit(f)(put(pp))
        grad = jax.jit(jax.grad(lambda x: f(x)[0]))(put(pp))
    mult = qs.token_multiplier(
        jnp.asarray(pp), jnp.asarray(w), *(jnp.asarray(h) for h in hist), **qs.track_kwargs(settings)
    )
    return (
        float(loss),
        {k[len("train/") :]: float(v) for k, v in metrics.items() if k.startswith("train/track/")},
        np.asarray(grad),
        np.asarray(mult),
        float(metrics["train/cross_entropy_loss"]),
    )


# ------------------------------------------------------------------------------------------------ the comparison


def _cfg(tmp_path, **kw):
    base = dict(
        hf_checkpoint="x",
        data_cache_path="x",
        tokenizer_path="x",
        output_path=str(tmp_path),
        run_id="r",
        steps=STEPS,
        track_max_len=MAX_LEN,
    )
    base.update(kw)
    return qs.QwenSftConfig(**base)


CASES = [
    dict(track_tol=TOL, track_reenter="either"),
    dict(track_tol=TOL, track_reenter="rise"),
    dict(track_tol=TOL, track_reenter="either", track_drop_weight=0.25),
    dict(track_tol=TOL, track_reenter="rise", track_drop_weight=0.25),
    dict(track_tol=TOL, track_reenter="either", track_ema_alpha=0.5),
    dict(track_tol=TOL, track_reenter="either", track_ema_alpha=0.5, track_drop_weight=0.25),
    dict(track_tol=0.0, track_observe_only=True),
    dict(track_tol=TOL, track_reenter="either", trace_num_docs=5, trace_seed=3),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_qwen_tracking_matches_grug_step_by_step(tmp_path, case):
    case = dict(case)
    if "trace_num_docs" in case:
        case = {**case, "trace_out": str(tmp_path / "trace-q")}
    cfg_q = _cfg(tmp_path, **case)
    track_q = qs.resolve_tracking(cfg_q, NUM_DOCS)
    settings = track_q.settings
    # Grug's own objects, built the way run_grug_local builds them.
    g_hist = tt.TokenHistory(NUM_DOCS, MAX_LEN)
    g_trace = None
    if cfg_q.trace_num_docs:
        docs = np.random.default_rng(cfg_q.trace_seed).choice(NUM_DOCS, size=cfg_q.trace_num_docs, replace=False)
        g_trace = tt.TraceRecorder(docs, MAX_LEN, str(tmp_path / "trace-g"), tol=settings.tol, floor=settings.floor)
    mesh, q_mesh = grug_mesh(), qwen_mesh()
    rng = np.random.default_rng(0)
    docs = doc_layout(rng)
    hist_path = qs.history_path_for(str(tmp_path / "ckpt"))
    saver = qs.HistorySaver(track_q, hist_path, 0, is_writer=True)
    max_diff, dropped_any, rise_reentries, fall_reentries = 0.0, 0, 0, 0
    for step in range(STEPS):
        seg, w, tokens = make_batch(rng, docs, step)
        pp = per_pos_for(seg, w, g_hist)  # the "model": a function of the visit count
        assert np.array_equal(g_hist.count, track_q.history.count)

        # side G
        def run_g(p1, p2, cnt, prev_dropped, pp=pp, w=w, tokens=tokens):
            loss, stats, grad = grug_device_step(mesh, settings, pp, w, p1, p2, cnt, prev_dropped)
            return {"per_pos": pp, "tokens": tokens, "loss": loss, "stats": stats, "grad": grad}

        out_g, diag_g, (g_keys, g_p1, g_p2, g_cnt, g_prev) = grug_host_step(
            g_hist, settings, g_trace, step, seg, w, run_g
        )

        # side Q
        keys, p1, p2, cnt, prev_dropped = qs.track_step_inputs(track_q.history, seg, w)
        for a, b in ((keys, g_keys), (p1, g_p1), (p2, g_p2), (cnt, g_cnt), (prev_dropped, g_prev)):
            assert np.array_equal(a, b, equal_nan=True)
        loss_q, stats_q, grad_q, mult_q, ce_q = qwen_device_step(q_mesh, settings, pp, w, (p1, p2, cnt, prev_dropped))
        diag_q = qs.track_step_record(
            track_q,
            step=step,
            keys=keys,
            p1=p1,
            p2=p2,
            cnt=cnt,
            prev_dropped=prev_dropped,
            w_np=w,
            pp_np=pp,
            tok_np=tokens,
        )
        assert diag_q == diag_g

        # identical keep mask: the Grug gradient is nonzero exactly on the tokens it trains (scaled by drop_weight
        # for dropped ones), and the Qwen multiplier is 1 / drop_weight / 0 on the same tokens
        trained = w > 0
        w_kept = float(np.sum(mult_q * w))
        assert np.array_equal(out_g["grad"] != 0, mult_q * (pp != 0) != 0)
        np.testing.assert_allclose(grad_q, out_g["grad"], rtol=1e-6, atol=1e-9)
        if w_kept > 0:
            np.testing.assert_allclose(out_g["grad"] * w_kept, mult_q * (pp != 0), rtol=1e-5, atol=1e-7)
        kept = trained & (mult_q == 1.0)
        assert int(out_g["stats"]["track/kept_tokens"]) == int(kept.sum())
        assert stats_q.keys() == out_g["stats"].keys()
        for k in tt.TRACK_STAT_KEYS:
            assert abs(stats_q[k] - out_g["stats"][k]) <= 1e-7 * max(1.0, abs(out_g["stats"][k])), (k, step)
        assert abs(loss_q - out_g["loss"]) <= 1e-7 * max(1.0, abs(out_g["loss"])), (step, loss_q, out_g["loss"])
        assert ce_q == stats_q["track/loss_unfiltered"]
        max_diff = max(max_diff, abs(loss_q - out_g["loss"]))
        # loss from the multiplier equals tracked_loss (the microbatched gradient's normaliser)
        if w_kept > 0:
            assert abs(float(np.sum(mult_q * pp)) / w_kept - loss_q) <= 1e-6 * max(1.0, loss_q)
        dropped_any += int(out_g["stats"]["track/kept_tokens"] < out_g["stats"]["track/trained_tokens"])

        # re-entry happened: a token dropped at its previous visit trained again, by a rise or a fall
        was = (g_prev > 0.5) & trained
        back = was & (mult_q == 1.0)
        now = np.where(trained, pp / np.where(trained, w, 1), 0)
        rise_reentries += int((back & (now > g_p1)).sum())
        fall_reentries += int((back & (now < g_p1)).sum())

        # history beside the checkpoint, and a restore in the middle of the run
        saver.after_checkpoint(step + 1 if step + 1 == SAVE_AT else saver.saved_step)
        if step + 1 == SAVE_AT:
            fresh = qs.resolve_tracking(cfg_q, NUM_DOCS)
            assert not qs.restore_history(fresh, hist_path, SAVE_AT + 1), "a table saved at another step is ignored"
            assert qs.restore_history(fresh, hist_path, SAVE_AT)
            fresh.trace = track_q.trace
            track_q = fresh
            saver.track = fresh

    for a, b in (
        (g_hist.prev1, track_q.history.prev1),
        (g_hist.prev2, track_q.history.prev2),
        (g_hist.count, track_q.history.count),
        (g_hist.dropped, track_q.history.dropped),
    ):
        assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    seen = g_hist.count.reshape(NUM_DOCS, MAX_LEN)
    assert seen[seen > 0].min() >= 6, "every trained token is visited at least six times"
    assert np.isfinite(g_hist.prev1[g_hist.count > 0]).all(), "NaN-free"
    if settings.tol > 0:
        assert dropped_any > 0
    if settings.tol > 0 and settings.ema_alpha == 0:
        assert rise_reentries > 0
        if settings.reentry is tt.Reentry.EITHER:
            assert fall_reentries > 0
        else:
            assert fall_reentries == 0, "rise-only re-entry never brings a falling token back"
    if settings.tol == 0:
        assert dropped_any == 0
    if g_trace is not None:
        g_trace.flush()
        track_q.trace.flush()
        for name in sorted(os.listdir(tmp_path / "trace-g")):
            zg, zq = np.load(tmp_path / "trace-g" / name), np.load(tmp_path / "trace-q" / name)
            assert set(zg.files) == set(zq.files)
            for f in zg.files:
                assert np.array_equal(zg[f], zq[f]), (name, f)
        assert np.array_equal(np.load(tmp_path / "trace-q" / "trace-docs.npy"), np.sort(g_trace.docs))
    print(f"max |loss_q - loss_g| = {max_diff:.3g}")


def test_validation_matches_grug(tmp_path):
    """Same rejections and messages as train.run_grug_local."""
    bad = [
        (dict(trace_num_docs=3), "the token trace needs tracking on (track_tol > 0 or track_observe_only)"),
        (dict(track_tol=0.1, track_observe_only=True), "track_observe_only records without dropping"),
        (dict(track_tol=0.1, track_drop_weight=1.0), "track_drop_weight must be in [0, 1)"),
        (dict(track_tol=0.1, track_reenter="rise", track_ema_alpha=0.5), "track_reenter other than 'either'"),
        (
            dict(track_tol=0.1, track_reenter="rise", trace_num_docs=2, trace_out="t"),
            "track_reenter other than 'either'",
        ),
        (dict(track_tol=0.1, trace_num_docs=2), "trace_num_docs > 0 needs trace_out"),
        (dict(track_tol=0.1, track_ema_alpha=0.5, trace_num_docs=2, trace_out="t"), "does not support track_ema_alpha"),
    ]
    for kw, msg in bad:
        with pytest.raises(ValueError) as e:
            qs.resolve_tracking(_cfg(tmp_path, **kw), NUM_DOCS)
        assert msg in str(e.value), (kw, str(e.value))
    with pytest.raises(ValueError, match="needs tail_num_docs"):
        qs.resolve_tracking(_cfg(tmp_path, track_tol=0.1), None)
    assert qs.resolve_tracking(_cfg(tmp_path), NUM_DOCS) is None
    obs = qs.resolve_tracking(_cfg(tmp_path, track_observe_only=True), NUM_DOCS)
    assert obs.settings.tol == 0.0 and obs.settings == grug_train.TrackSettings(tol=0.0, floor=0.05)


def test_history_saver_follows_checkpointer(tmp_path, caplog):
    """Saved only when the checkpointer saved a new step, at that step; final save only if still behind."""
    track = qs.resolve_tracking(_cfg(tmp_path, track_tol=TOL), NUM_DOCS)
    path = qs.history_path_for(str(tmp_path / "not-yet" / "checkpoints"))
    saver = qs.HistorySaver(track, path, 0, is_writer=True)
    saver.after_checkpoint(0)
    assert not os.path.exists(path)
    saver.after_checkpoint(5)
    assert tt.TokenHistory(NUM_DOCS, MAX_LEN).load(path, 5)
    saver.final(5)
    assert tt.TokenHistory(NUM_DOCS, MAX_LEN).load(path, 5)
    saver.final(9)
    assert tt.TokenHistory(NUM_DOCS, MAX_LEN).load(path, 9)
    other = qs.HistorySaver(track, str(tmp_path / "x.npz"), 0, is_writer=False)
    other.after_checkpoint(3)
    assert not os.path.exists(tmp_path / "x.npz"), "only process 0 writes"
    with caplog.at_level(logging.WARNING):
        fresh = qs.resolve_tracking(_cfg(tmp_path, track_tol=TOL), NUM_DOCS)
        assert not qs.restore_history(fresh, path, 4)
    assert "tracking restarts from empty histories" in caplog.text
    assert not qs.restore_history(fresh, path, 0), "a fresh run (step 0) restores nothing"


def test_env_settings_match_grug_launcher():
    """The SNOWBALL_* parsing and defaults are vista_snowball_chat's."""
    d = qs.snowball_env_settings({})
    assert d["track_tol"] == 0.0 and d["track_floor"] == 0.05 and d["track_max_len"] == 32768
    assert d["track_ema_alpha"] == 0.0 and d["track_reenter"] == "either" and d["track_drop_weight"] == 0.0
    assert d["track_observe_only"] is False and d["trace_num_docs"] == 0 and d["schedule_steps"] is None
    assert d["data_seed"] is None and d["learning_rate"] is None
    env = {
        "SNOWBALL_TRACK_TOL": "0.02",
        "SNOWBALL_TRACK_OBSERVE_ONLY": "1",
        "SNOWBALL_TRACK_REENTER": "rise",
        "SNOWBALL_TRACK_DROP_WEIGHT": "0.3",
        "SNOWBALL_TRACK_EMA_ALPHA": "",
        "SNOWBALL_DATA_SEED": "7",
        "SNOWBALL_SCHEDULE_STEPS": "0",
        "SNOWBALL_SEQ_LEN": "16384",
        "SNOWBALL_TRACE_DOCS": "4",
        "SNOWBALL_LR": "2e-6",
    }
    d = qs.snowball_env_settings(env)
    assert d["track_tol"] == 0.02 and d["track_observe_only"] and d["track_reenter"] == "rise"
    assert d["track_drop_weight"] == 0.3 and d["track_ema_alpha"] == 0.0 and d["data_seed"] == 7
    assert d["schedule_steps"] is None and d["track_max_len"] == 16384 and d["trace_num_docs"] == 4
    assert d["learning_rate"] == 2e-6
    from experiments.june_tpu_67b_a2b.moe import snowball_chat_recipe as recipe

    # same seed (data key); the Qwen default batch is 128 sequences where the Grug Snowball runs used 64
    assert recipe.SNOWBALL_CHAT_SEED == qs.DEFAULT_SEED and qs.DEFAULT_BATCH_SIZE == 2 * recipe.SNOWBALL_CHAT_BATCH_SIZE
    assert recipe.SNOWBALL_CHAT_SEQUENCE_LENGTH == qs.DEFAULT_SEQ_LEN
    defaults = {f.name: f.default for f in dataclasses.fields(grug_train.GrugTrainerConfig)}
    for name in (
        "track_floor",
        "track_ema_alpha",
        "track_reenter",
        "track_drop_weight",
        "track_observe_only",
        "trace_num_docs",
        "trace_seed",
        "track_max_len",
    ):
        assert getattr(qs.QwenSftConfig, name) == defaults[name], name

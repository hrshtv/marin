# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks for per-token loss tracking: stable keys across epochs, the table round trip, the flat rule (both
ends, re-entry when a token moves, rise-only re-entry), gradient only on kept tokens, and the 4-device sharded path."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from experiments.june_tpu_67b_a2b.moe import token_track as tt
from experiments.june_tpu_67b_a2b.moe.tail_unit_shard_test import make_turned_batch

TOL, FLOOR = 0.10, 0.05
KEEP_ALL = lambda keys: np.zeros(keys.shape, dtype=bool)  # drop decisions for tests that do not use them

def test_keys_stable_under_repacking():
    h = tt.TokenHistory(num_docs=4, max_len=16)
    seg_a = np.array([[0, 0, 0, 1, 1, -1], [2, 2, 3, 3, 3, 3]]); w = np.ones_like(seg_a, float); w[0, 5] = 0
    seg_b = np.array([[1, 1, 0, 0, 0, -1], [3, 3, 3, 3, 2, 2]]); wb = np.ones_like(seg_b, float); wb[0, 5] = 0
    ka, kb = h.keys(seg_a, w), h.keys(seg_b, wb)
    assert sorted(ka[ka >= 0].tolist()) == sorted(kb[kb >= 0].tolist()), "same documents, same keys, any packing"
    assert ka[0, 5] == -1 and ka[0, 0] == 0 * 16 + 0 and ka[0, 2] == 2 and ka[0, 3] == 1 * 16 + 0

def test_table_round_trip_and_flat_rule():
    h = tt.TokenHistory(num_docs=2, max_len=8)
    seg = np.array([[0, 0, 0, 1, 1, 1]]); w = np.ones_like(seg, float)
    keys = h.keys(seg, w)
    # epochs 1-3 losses per token: flat-low, flat-high, falling, rising-then-flat, noisy, flat
    l1 = np.array([[0.02, 2.00, 1.00, 0.50, 0.40, 0.30]])
    l2 = np.array([[0.02, 2.05, 0.60, 0.80, 0.42, 0.30]])
    l3 = np.array([[0.02, 2.02, 0.30, 0.80, 0.80, 0.30]])
    for l in (l1, l2):
        p1, p2, c, _ = h.gather(keys)
        h.update(keys, l, KEEP_ALL(keys))
    p1, p2, c, _ = h.gather(keys)
    assert np.allclose(p1, l2) and np.allclose(p2, l1) and np.all(c == 2)
    loss, s = tt.tracked_loss(jnp.asarray(l3 * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), jnp.zeros_like(jnp.asarray(c)), tol=TOL, floor=FLOOR)
    # expected: tok0 flat-low drop; tok1 flat-high drop; tok2 moving keep; tok3 rose then flat -> second gap flat
    # but first gap not -> keep; tok4 noisy keep; tok5 flat (0.30 x3) drop, counted high (>= -log 0.9)
    keep = np.array([False, False, True, True, True, False])
    assert abs(float(loss) - float((l3[0] * keep).sum() / keep.sum())) < 1e-6
    assert abs(float(s["track/dropped_frac"]) - 3 / 6) < 1e-6 and abs(float(s["track/dropped_low_frac"]) - 1 / 6) < 1e-6
    assert abs(float(s["track/dropped_high_frac"]) - 2 / 6) < 1e-6 and float(s["track/eligible_frac"]) == 1.0
    # re-entry: the dropped token 5 moves by 50% at epoch 4 -> no longer flat -> trained again
    h.update(keys, l3, KEEP_ALL(keys))
    l4 = l3.copy(); l4[0, 5] = 0.45
    p1, p2, c, _ = h.gather(keys)
    _, s4 = tt.tracked_loss(jnp.asarray(l4 * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), jnp.zeros_like(jnp.asarray(c)), tol=TOL, floor=FLOOR)
    # epoch 4: tok5 moved 0.30 -> 0.45 (50 %) so it re-enters; tok3 (0.80, 0.80, 0.80) is now flat and drops.
    # Dropped = tok0, tok1, tok3 = 3/6; had tok5 stayed dropped it would be 4/6.
    assert abs(float(s4["track/dropped_frac"]) - 3 / 6) < 1e-6, "a dropped token whose loss moved must re-enter"
    # before two observations nothing is eligible
    h0 = tt.TokenHistory(2, 8); p1, p2, c, _ = h0.gather(keys)
    _, s0 = tt.tracked_loss(jnp.asarray(l1 * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), jnp.zeros_like(jnp.asarray(c)), tol=TOL, floor=FLOOR)
    assert float(s0["track/eligible_frac"]) == 0.0 and float(s0["track/kept_token_frac"]) == 1.0

def test_sharded_matches_single_and_gradient_lands_on_kept():
    mesh = Mesh(np.array(jax.devices()[:4]), ("data",)); row = NamedSharding(mesh, P("data"))
    rng = np.random.default_rng(5)
    per_pos, weight, seg, _ = make_turned_batch(rng)
    trained = weight > 0
    now = np.where(trained, per_pos / np.maximum(weight, 1e-9), 0)
    # history: half the trained tokens flat (prev == now), the rest moving
    flat_sel = trained & (rng.random(weight.shape) < 0.5)
    p1 = np.where(flat_sel, now * 1.01, now * 1.6).astype(np.float32); p2 = np.where(flat_sel, now * 0.99, now * 2.5).astype(np.float32)
    c = np.where(trained, 2.0, 0.0).astype(np.float32)
    put = lambda a: jax.device_put(jnp.asarray(a), row)
    psum = lambda x: jax.lax.psum(x, "data")
    local = lambda a, b, x, y, z, d: tt.tracked_loss(a, b, x, y, z, d, tol=TOL, floor=FLOOR, psum=psum)
    f = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("data"),) * 6, out_specs=(P(), {k: P() for k in tt.TRACK_STAT_KEYS}), check_vma=False))
    args = (put(per_pos), put(weight), put(p1), put(p2), put(c), put(np.zeros_like(c)))
    loss_s, s = f(*args)
    loss_1, s1 = jax.jit(lambda *a: tt.tracked_loss(*a, tol=TOL, floor=FLOOR))(jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), jnp.zeros_like(jnp.asarray(c)))
    assert abs(float(loss_s) - float(loss_1)) < 1e-5
    for k in tt.TRACK_STAT_KEYS: assert abs(float(s[k]) - float(s1[k])) < 1e-4, k
    g = np.asarray(jax.grad(lambda x: f(x, *args[1:])[0])(args[0]))
    keep = trained & ~flat_sel
    assert np.all(g[trained & flat_sel] == 0) and np.all(g[keep] > 0)
    assert abs(float(s["track/dropped_frac"]) - flat_sel.sum() / trained.sum()) < 1e-6
    print(f"sharded: kept {float(s['track/kept_token_frac']):.3f}, dropped {float(s['track/dropped_frac']):.3f} "
          f"(low {float(s['track/dropped_low_frac']):.3f}, high {float(s['track/dropped_high_frac']):.3f})")

def test_duplicate_keys_keep_first_and_are_reported():
    h = tt.TokenHistory(num_docs=2, max_len=8)
    seg = np.array([[0, 0, 0, 1, 1, 0, 0, 0]]); w = np.ones_like(seg, float)      # doc 0 twice in one batch
    keys = h.keys(seg, w)
    assert (keys[0, :3] == keys[0, 5:]).all(), "the second copy of doc 0 reuses its keys"
    diag = h.update(keys, np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 9.0, 9.0, 9.0]]), KEEP_ALL(keys))
    assert diag["duplicate_keys"] == 3 and diag["documents_with_duplicates"] == 1
    p1, _, c, _ = h.gather(keys[:, :5])
    assert np.allclose(p1[0], [0.1, 0.2, 0.3, 0.4, 0.5]) and (c[0] == 1).all(), "first occurrence recorded once"

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("TOKEN_TRACK_TEST_OK")

def test_history_checkpoint_round_trip(tmp_path):
    h = tt.TokenHistory(num_docs=2, max_len=8)
    seg = np.array([[0, 0, 0, 1, 1, 1]]); w = np.ones_like(seg, float); keys = h.keys(seg, w)
    h.update(keys, np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]), KEEP_ALL(keys))
    h.update(keys, np.array([[0.1, 0.25, 0.3, 0.4, 0.5, 0.7]]), np.array([[True, False, True, False, False, True]]))
    path = str(tmp_path / "token-history.npz"); h.save(path, step=40)
    restored = tt.TokenHistory(num_docs=2, max_len=8)
    assert restored.load(path, step=40), "same step and shape restores"
    for a, b in ((h.prev1, restored.prev1), (h.prev2, restored.prev2), (h.count, restored.count), (h.dropped, restored.dropped)):
        assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)

def test_history_checkpoint_rejects_other_step_or_shape(tmp_path):
    h = tt.TokenHistory(num_docs=2, max_len=8); path = str(tmp_path / "token-history.npz"); h.save(path, step=40)
    fresh = tt.TokenHistory(num_docs=2, max_len=8)
    assert not fresh.load(path, step=41), "a table saved at another step must not be restored"
    assert not tt.TokenHistory(num_docs=3, max_len=8).load(path, step=40), "a differently sized table must not be restored"
    assert not fresh.load(str(tmp_path / "missing.npz"), step=40)
    assert np.all(np.isnan(fresh.prev1)) and np.all(fresh.count == 0), "a rejected load leaves the table empty"

def test_history_save_creates_missing_folder(tmp_path):
    # Job 2324865 died here: only temporary checkpoints existed, so <base>/checkpoints/ had not been created yet.
    h = tt.TokenHistory(num_docs=2, max_len=8); path = str(tmp_path / "not-yet-created" / "token-history.npz")
    h.save(path, step=243)
    assert tt.TokenHistory(num_docs=2, max_len=8).load(path, step=243)

def _random_history(seed, n=4000):
    rng = np.random.default_rng(seed)
    p2 = rng.uniform(0.0, 2.0, n).astype(np.float32); p1 = (p2 * rng.uniform(0.85, 1.15, n)).astype(np.float32)
    now = (p1 * rng.uniform(0.85, 1.15, n)).astype(np.float32); c = rng.integers(0, 5, n).astype(np.float32)
    p1[rng.random(n) < 0.05] = np.nan; w = (rng.random(n) < 0.8).astype(np.float32)
    return p1, p2, c, now, w

def test_host_rule_matches_training_mask():
    # The trace's kept flag must be the tokens tracked_loss actually trains on.
    for seed in range(5):
        p1, p2, c, now, w = _random_history(seed)
        loss, stats = tt.tracked_loss(jnp.asarray(now * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), jnp.zeros_like(jnp.asarray(c)), tol=TOL, floor=FLOOR)
        kept = (w > 0) & ~tt.plateaued(p1, p2, c, now, tol=TOL, floor=FLOOR)
        assert int(stats["track/kept_tokens"]) == int(kept.sum())
        assert abs(float(loss) - float(now[kept].mean())) < 1e-5

def test_observe_only_is_plain_sft():
    # tol 0 never marks a token flat, so the loss is the weighted mean over all trained tokens.
    p1, p2, c, now, w = _random_history(7)
    loss, stats = tt.tracked_loss(jnp.asarray(now * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), jnp.zeros_like(jnp.asarray(c)), tol=0.0, floor=FLOOR)
    assert abs(float(loss) - float((now * w).sum() / w.sum())) < 1e-5 and float(stats["track/dropped_frac"]) == 0.0
    assert not tt.plateaued(p1, p2, c, now, tol=0.0, floor=FLOOR).any()

def test_trace_records_sampled_documents(tmp_path):
    h = tt.TokenHistory(num_docs=3, max_len=8)
    seg = np.array([[0, 0, 0, 1, 1, 2, 2, 2]]); w = np.ones_like(seg, float); w[0, 2] = 0; keys = h.keys(seg, w)
    tokens = np.arange(100, 108)[None, :]; target = np.roll(tokens, -1, axis=-1)
    rec = tt.TraceRecorder(np.array([0, 2]), 8, str(tmp_path / "trace"), tol=TOL, floor=FLOOR)
    for step, l in enumerate((np.full((1, 8), 0.50), np.full((1, 8), 0.49), np.full((1, 8), 0.48))):
        p1, p2, cnt, _ = h.gather(keys); rec.record(step, keys, target, l, p1, p2, cnt); h.update(keys, l, KEEP_ALL(keys))
    rec.flush(); z = np.load(str(tmp_path / "trace" / "trace-0000.npz"))
    assert set(z["doc"].tolist()) == {0, 2}, "only the sampled documents, and only trained tokens (doc 0 offset 2 has weight 0)"
    assert len(z["doc"]) == 3 * 5 and set(z["visit"].tolist()) == {1, 2, 3}
    first = (z["step"] == 0) & (z["doc"] == 0)
    assert z["token"][first].tolist() == [101, 102], "target id is the next token"
    third = z["visit"] == 3
    assert not z["kept"][third].any() and z["kept"][~third].all(), "flat for 3 visits -> dropped at the third, kept before"

def test_rise_reentry_keeps_falling_tokens_dropped():
    # Six tokens, all dropped at their previous visit with a flat history (1.00, 1.00). This visit: large fall,
    # large rise, small fall, small rise, fall to near zero, and one token that was NOT dropped last visit and falls.
    p2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0], np.float32); p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32)
    now = np.array([0.5, 1.5, 0.95, 1.05, 0.01, 0.5], np.float32); c = np.full(6, 2.0, np.float32); w = np.ones(6, np.float32)
    was = np.array([1, 1, 1, 1, 1, 0], np.float32)
    expected = {
        tt.Reentry.EITHER: np.array([False, False, True, True, False, False]),  # any large move re-enters
        tt.Reentry.RISE: np.array([True, False, True, True, True, False]),  # only the large rise re-enters
    }
    for rule, drop in expected.items():
        loss, s = tt.tracked_loss(jnp.asarray(now * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c),
                                  jnp.asarray(was), tol=TOL, floor=FLOOR, reentry=rule)
        assert np.array_equal(tt.decide(p1, p2, c, was, now, tol=TOL, floor=FLOOR, reentry=rule), drop), rule
        assert int(s["track/kept_tokens"]) == int((~drop).sum()), rule
        assert abs(float(loss) - float(now[~drop].mean())) < 1e-6, rule

def test_rise_rule_host_decision_matches_training_mask():
    # The history's drop decision (host) must be exactly the mask tracked_loss trains with, for random histories.
    for seed in range(5):
        p1, p2, c, now, w = _random_history(seed)
        was = (np.random.default_rng(100 + seed).random(len(p1)) < 0.5).astype(np.float32)
        now = np.where(was > 0, now * np.random.default_rng(200 + seed).uniform(0.5, 1.5, len(now)), now).astype(np.float32)
        loss, stats = tt.tracked_loss(jnp.asarray(now * w), jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c),
                                      jnp.asarray(was), tol=TOL, floor=FLOOR, reentry=tt.Reentry.RISE)
        kept = (w > 0) & ~tt.decide(p1, p2, c, was, now, tol=TOL, floor=FLOOR, reentry=tt.Reentry.RISE)
        assert int(stats["track/kept_tokens"]) == int(kept.sum())
        assert abs(float(loss) - float(now[kept].mean())) < 1e-5

def test_rise_rule_over_visits_via_history():
    # One token through the history table: flat for three visits (dropped), then its loss halves (stays dropped under
    # RISE), keeps falling (still dropped), then rises by 50 % (re-enters).
    h = tt.TokenHistory(num_docs=1, max_len=4); keys = h.keys(np.array([[0]]), np.ones((1, 1)))
    decisions = []
    for loss in (0.8, 0.8, 0.8, 0.4, 0.2, 0.3):
        p1, p2, c, d = h.gather(keys); now = np.array([[loss]], np.float32)
        drop = tt.decide(p1, p2, c, d, now, tol=TOL, floor=FLOOR, reentry=tt.Reentry.RISE)
        h.update(keys, now, drop); decisions.append(bool(drop[0, 0]))
    assert decisions == [False, False, True, True, True, False], decisions

def test_drop_weight_downweights_instead_of_removing():
    # Four tokens: two flat (dropped), two moving (kept). With drop_weight 0.25 the loss is the weighted mean with the
    # dropped tokens at weight 0.25, and their gradient is 0.25 of a kept token's (equal per-token weights).
    p2 = np.array([1.0, 1.0, 2.0, 2.0], np.float32); p1 = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
    now = np.array([1.0, 1.0, 0.5, 0.6], np.float32); c = np.full(4, 2.0, np.float32); w = np.ones(4, np.float32)
    d = jnp.zeros(4)
    f = lambda pp, dw: tt.tracked_loss(pp, jnp.asarray(w), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(c), d, tol=TOL, floor=FLOOR, drop_weight=dw)
    loss, s = f(jnp.asarray(now * w), 0.25)
    assert abs(float(loss) - (0.5 + 0.6 + 0.25 * 2.0) / (2 + 0.25 * 2)) < 1e-6
    assert int(s["track/kept_tokens"]) == 2 and abs(float(s["track/dropped_frac"]) - 0.5) < 1e-6
    g = np.asarray(jax.grad(lambda x: f(x, 0.25)[0])(jnp.asarray(now * w)))
    assert np.allclose(g[:2], 0.25 * g[2]) and np.allclose(g[2], g[3]) and g[2] > 0
    loss0, _ = f(jnp.asarray(now * w), 0.0)
    assert abs(float(loss0) - (0.5 + 0.6) / 2) < 1e-6, "drop_weight 0 removes the dropped tokens"

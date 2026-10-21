# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The Qwen step's loss on a tiny Qwen3: Grug's per-position quantity (z-loss included), observe-only tracking equals
plain SFT (loss and gradients), an all-flat batch trains nothing, microbatching gives the one-shot gradient, and
parameters are FSDP-sharded over every device."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.layers.attention import AttentionBackend
from levanter.models.qwen import Qwen3Config
from levanter.utils.mesh import create_mesh_from_axis_specs

from experiments.june_tpu_67b_a2b.moe import token_track as tt
from experiments.june_tpu_67b_a2b.moe.train import TrackSettings
from experiments.plateau_dropping_qwen import qwen_sft as qs

B, S, V = 16, 64, 512
Z = 1e-4
MP = jmp.get_policy("p=f32,c=f32")  # f32 compute so the comparisons are tight


def tiny_config():
    return Qwen3Config(
        max_seq_len=S,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        attn_backend=AttentionBackend.VANILLA,
        gradient_checkpointing=False,
        layer_norm_epsilon=1e-6,
        use_sliding_window=False,
    )


@pytest.fixture(scope="module")
def setup():
    mesh_cfg = qs.trainer_mesh_config()
    ici, dcn = mesh_cfg.axis_shapes(jax.device_count(), 1)
    mesh = create_mesh_from_axis_specs(ici_axes=ici, dcn_axes=dcn)
    param_map, compute_map = mesh_cfg.resolved_param_mapping, mesh_cfg.resolved_compute_mapping
    cfg = tiny_config()
    Vocab = hax.Axis("vocab", V)
    Pos = cfg.max_Pos.resize(S)
    Batch = hax.Axis("batch", B)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(B):  # packed rows: three documents, the first two tokens of each are "prompt"
        seg = np.repeat(np.arange(3), [20, 25, S - 45]).astype(np.int32)
        seg[-5:] = -1
        mask = np.ones(S, np.float32)
        for start in (0, 20, 45):
            mask[start : start + 2] = 0
        tokens = rng.integers(0, V, S).astype(np.int32)
        ex = GrugLmExample.causal(
            jnp.asarray(tokens), loss_weight=jnp.asarray(np.roll(mask, -1)), segment_ids=jnp.asarray(seg), max_segments=4
        )
        rows.append(ex)
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)
    example = named_lm_example_from_grug(batched, Pos, batch_axis=Batch)
    with set_mesh(mesh), hax.axis_mapping(param_map):
        model = cfg.build(Vocab, key=jax.random.PRNGKey(0))
        model = hax.shard(model, param_map)
        example = hax.shard(example, compute_map)
    return dict(mesh=mesh, param_map=param_map, compute_map=compute_map, model=model, example=example, cfg=cfg)


def run(setup, hist, settings, num_micro=1):
    def f(model, example, hist):
        return qs.loss_and_grads(
            model,
            example,
            hist,
            mp=MP,
            z_loss_weight=Z,
            track_settings=settings,
            compute_mapping=setup["compute_map"],
            num_microbatches=num_micro,
            accum_mapping=setup["param_map"],
        )

    with set_mesh(setup["mesh"]), hax.axis_mapping(setup["param_map"]):
        loss, metrics, pp, grads = eqx.filter_jit(f)(setup["model"], setup["example"], hist)
    return float(loss), metrics, np.asarray(pp.array), grads


def hist_arrays(setup, p1, p2, cnt, dropped):
    axes = setup["example"].loss_weight.axes
    return tuple(hax.named(jnp.asarray(a, jnp.float32), axes) for a in (p1, p2, cnt, dropped))


def weights(setup):
    return np.asarray(qs.effective_loss_weight(setup["example"]).array)


def leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]


def test_per_position_loss_is_grugs_quantity(setup):
    """(CE + z * logsumexp^2) * weight, computed by Grug's fused loss on the same hidden states."""
    model, example = setup["model"], setup["example"]
    with set_mesh(setup["mesh"]), hax.axis_mapping(setup["param_map"]):
        pp, _aux = qs.per_position_loss(model, example, mp=MP, z_loss_weight=Z, compute_mapping=setup["compute_map"])

        @jax.jit
        def hidden_and_head(model, example):
            hidden = model.activations(example.tokens, example.attn_mask, key=None)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            head = model.get_lm_head().rearrange((model.Embed, model.Vocab)).array
            return hidden.rearrange(("batch", "position", model.Embed.name)).array, head

        h, head = (np.asarray(x) for x in hidden_and_head(model, example))
    # Grug's loss (levanter.grug.loss, as Transformer.next_token_loss calls it) on the same hidden states, off-mesh
    tokens = np.asarray(example.tokens.array)
    labels = np.concatenate([tokens[:, 1:], tokens[:, :1] * 0], axis=1)
    ref = np.asarray(
        fused_linear_softmax_cross_entropy_loss(
            jnp.asarray(h),
            jnp.asarray(head),
            jnp.asarray(labels),
            weight=jnp.asarray(np.asarray(example.loss_weight.array)),
            reduction="none",
            logsumexp_weight=Z,
        )
    )
    w = weights(setup)
    np.testing.assert_allclose(np.asarray(pp.array), ref, rtol=1e-5, atol=1e-6)
    assert np.all(np.asarray(pp.array)[w == 0] == 0) and np.all(np.asarray(pp.array)[w > 0] > 0)
    assert np.all(w[:, -1] == 0)
    # the z-loss term is in it: without it the per-position loss is smaller
    with set_mesh(setup["mesh"]), hax.axis_mapping(setup["param_map"]):
        pp0, _aux0 = qs.per_position_loss(model, example, mp=MP, z_loss_weight=0.0, compute_mapping=setup["compute_map"])
    assert np.all(np.asarray(pp.array)[w > 0] > np.asarray(pp0.array)[w > 0])


def test_observe_only_equals_plain_and_all_flat_trains_nothing(setup):
    loss_plain, m_plain, pp, g_plain = run(setup, None, None)
    w = weights(setup)
    rng = np.random.default_rng(1)
    now = np.where(w > 0, pp / np.where(w > 0, w, 1), 0)
    # observe-only: tol 0 with a history that would make many tokens flat at any positive tol
    p1, p2 = now * (1 + 1e-4 * rng.standard_normal(now.shape)), now * (1 + 1e-4 * rng.standard_normal(now.shape))
    cnt = np.where(w > 0, 2 + rng.integers(0, 4, w.shape), 0)
    obs = TrackSettings(tol=0.0, floor=0.05)
    hist = hist_arrays(setup, p1, p2, cnt, np.zeros_like(w))
    loss_obs, m_obs, pp_obs, g_obs = run(setup, hist, obs)
    assert abs(loss_obs - loss_plain) <= 1e-6 * abs(loss_plain)
    assert float(m_obs["train/track/kept_token_frac"]) == 1.0 and float(m_obs["train/track/dropped_frac"]) == 0.0
    assert (
        abs(float(m_obs["train/cross_entropy_loss"]) - float(m_plain["train/cross_entropy_loss"])) <= 1e-6 * loss_plain
    )
    np.testing.assert_array_equal(pp_obs, pp)
    for a, b in zip(leaves(g_obs), leaves(g_plain), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    # every token flat at tol 0.05 (prev1 = prev2 = now): nothing trains
    flat = TrackSettings(tol=0.05, floor=0.05)
    loss_flat, m_flat, _, g_flat = run(
        setup, hist_arrays(setup, now, now, np.where(w > 0, 2, 0), np.zeros_like(w)), flat
    )
    assert loss_flat == 0.0 and float(m_flat["train/track/kept_tokens"]) == 0.0
    assert float(m_flat["train/track/dropped_frac"]) == 1.0
    assert all(np.all(g == 0) for g in leaves(g_flat))
    # ... unless dropped tokens keep a weight: then the loss is the plain loss again (all tokens at the same weight)
    dw = TrackSettings(tol=0.05, floor=0.05, drop_weight=0.25)
    loss_dw, _, _, g_dw = run(setup, hist_arrays(setup, now, now, np.where(w > 0, 2, 0), np.zeros_like(w)), dw)
    assert abs(loss_dw - loss_plain) <= 1e-6 * loss_plain
    for a, b in zip(leaves(g_dw), leaves(g_plain), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "settings",
    [
        None,
        TrackSettings(tol=0.05, floor=0.05),
        TrackSettings(tol=0.05, floor=0.05, drop_weight=0.25, reentry=tt.Reentry.RISE),
    ],
    ids=["plain", "track", "track-rise-dw"],
)
def test_microbatched_gradient_equals_one_shot(setup, settings):
    _, _, pp, _ = run(setup, None, None)
    w = weights(setup)
    rng = np.random.default_rng(2)
    now = np.where(w > 0, pp / np.where(w > 0, w, 1), 0)
    flat_sel = rng.random(w.shape) < 0.4
    p1 = np.where(flat_sel, now * 1.01, now * 1.5)
    p2 = np.where(flat_sel, now * 0.99, now * 2.0)
    prev_dropped = (rng.random(w.shape) < 0.3).astype(np.float32)
    hist = None if settings is None else hist_arrays(setup, p1, p2, np.where(w > 0, 3, 0), prev_dropped)
    loss1, m1, pp1, g1 = run(setup, hist, settings, num_micro=1)
    if settings is not None:
        assert 0.0 < float(m1["train/track/kept_token_frac"]) < 1.0
    for n in (2, 4):
        loss_n, m_n, pp_n, g_n = run(setup, hist, settings, num_micro=n)
        np.testing.assert_allclose(pp_n, pp1, rtol=1e-6, atol=1e-7)
        assert abs(loss_n - loss1) <= 1e-6 * abs(loss1)
        for k in m1:
            assert abs(float(m_n[k]) - float(m1[k])) <= 1e-5 * max(1.0, abs(float(m1[k]))), k
        for a, b in zip(leaves(g_n), leaves(g1), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_parameters_are_fsdp_sharded(setup):
    """Every parameter with an embed axis is split over all 4 devices; the per-device share is ~1/4."""
    rep = qs.tree_bytes_report(setup["model"])
    assert rep["device0_bytes"] <= rep["total_bytes"] / 4 + rep["replicated_bytes"]
    assert rep["replicated_bytes"] < 0.01 * rep["total_bytes"]
    shape = jax.eval_shape(lambda: tiny_config().build(hax.Axis("vocab", V), key=jax.random.PRNGKey(0)))
    mapping = qs.trainer_mesh_config().resolved_param_mapping
    # one process with 4 GPUs (one slice) and four processes with one GPU each reported as 4 slices
    for mesh_shape in ({"data": 4, "replica_dcn": 1}, {"data": 1, "replica_dcn": 4}):
        est = qs.estimate_sharded_bytes(shape, mapping, mesh_shape)
        assert est["per_device_bytes"] <= est["total_bytes"] / 4 + est["replicated_bytes"]
        assert est["replicated_bytes"] < 0.01 * est["total_bytes"]
    for n_slices in (1, 4):
        ici, dcn = qs.trainer_mesh_config().axis_shapes(4, n_slices)
        assert ici["data"] * dcn["replica_dcn"] == 4 and ici["model"] == 1
    # the real Qwen3-8B shape: 8.19B params, fp32 params + Adam over 16 GPUs
    big = dataclasses.replace(
        tiny_config(),
        hidden_dim=4096,
        intermediate_dim=12288,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=40960,
    )
    shape8b = jax.eval_shape(lambda: big.build(hax.Axis("vocab", 151936), key=jax.random.PRNGKey(0)))
    est = qs.estimate_sharded_bytes(shape8b, mapping, {"data": 4, "replica_dcn": 4})
    assert 8.1e9 < est["total_bytes"] / 4 < 8.3e9
    assert abs(est["per_device_bytes"] - est["total_bytes"] / 16) <= est["replicated_bytes"] + 1

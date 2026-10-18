# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""4-device CPU check of clip SFT under the shard_map pattern: the mask, the kept-token mean, the gradient
landing only on kept tokens, the reasoning/command split, and the random-drop control."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from experiments.june_tpu_67b_a2b.moe import token_clip as tc, tail_filter as tf
from experiments.june_tpu_67b_a2b.moe.tail_unit_shard_test import make_turned_batch, END, MAX_TURNS, NUM_DOCS

def run(per_pos, weight, seg, tokens, rand, p_hi, p_lo):
    mesh = Mesh(np.array(jax.devices()[:4]), ("data",))
    row = NamedSharding(mesh, P("data")); put = lambda a: jax.device_put(jnp.asarray(a), row)
    psum = lambda x: jax.lax.psum(x, "data")
    def local(a, b, c, d, r):
        return tc.clipped_loss(a, b, c, d, r, p_hi=p_hi, p_lo=p_lo, num_docs=NUM_DOCS, max_turns=MAX_TURNS, end_think_id=END, psum=psum)
    specs = {k: P() for k in tc.CLIP_STAT_KEYS}
    f = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("data"),) * 5, out_specs=(P(), specs), check_vma=False))
    args = (put(per_pos), put(weight), put(seg), put(tokens), put(rand))
    loss, stats = f(*args)
    g = np.asarray(jax.grad(lambda x: f(x, *args[1:])[0])(args[0]))
    return float(loss), {k: float(v) for k, v in stats.items()}, g

def main():
    rng = np.random.default_rng(11)
    per_pos, weight, seg, tokens = make_turned_batch(rng)
    p = np.where(weight > 0, np.exp(-per_pos / np.maximum(weight, 1e-9)), 0)
    trained = weight > 0
    hi, lo = 0.7, 0.2
    ones = np.ones_like(weight, dtype=bool)
    loss, s, g = run(per_pos, weight, seg, tokens, ones, hi, lo)
    keep = trained & ~(p > hi) & ~(p < lo)
    assert keep.sum() > 0 and (trained & ~keep).sum() > 0, "thresholds must clip something and keep something"
    want = per_pos[keep].sum() / weight[keep].sum()
    assert abs(loss - want) < 1e-5, (loss, want)
    assert abs(s["clip/kept_token_frac"] - keep.sum() / trained.sum()) < 1e-6
    assert abs(s["clip/above_hi_frac"] - (trained & (p > hi)).sum() / trained.sum()) < 1e-6
    assert abs(s["clip/below_lo_frac"] - (trained & (p < lo)).sum() / trained.sum()) < 1e-6
    assert abs(s["clip/loss_unfiltered"] - per_pos.sum() / weight.sum()) < 1e-5
    assert np.all(g[trained & ~keep] == 0), "clipped tokens still carry gradient"
    assert np.all(g[keep] > 0), "kept tokens carry no gradient"
    assert 0 <= s["clip/above_hi_frac_reasoning"] <= 1 and 0 <= s["clip/above_hi_frac_command"] <= 1
    # Random-drop control: thresholds off, ~40 % of trained tokens dropped by the coin flips passed in.
    rand = rng.random(weight.shape) >= 0.4
    loss_r, s_r, g_r = run(per_pos, weight, seg, tokens, rand, None, None)
    keep_r = trained & rand
    assert abs(s_r["clip/random_dropped_frac"] - (trained & ~rand).sum() / trained.sum()) < 1e-6
    assert s_r["clip/above_hi_frac"] == 0.0 and s_r["clip/below_lo_frac"] == 0.0
    assert np.all(g_r[trained & ~rand] == 0) and np.all(g_r[keep_r] > 0)
    assert abs(loss_r - per_pos[keep_r].sum() / weight[keep_r].sum()) < 1e-5
    print(f"clip hi={hi} lo={lo}: kept {s['clip/kept_token_frac']:.3f}, above {s['clip/above_hi_frac']:.3f} "
          f"(reasoning {s['clip/above_hi_frac_reasoning']:.3f}, command {s['clip/above_hi_frac_command']:.3f}), "
          f"below {s['clip/below_lo_frac']:.3f}; random control dropped {s_r['clip/random_dropped_frac']:.3f}")
    print("TOKEN_CLIP_SHARD_TEST_OK")

if __name__ == "__main__":
    main()

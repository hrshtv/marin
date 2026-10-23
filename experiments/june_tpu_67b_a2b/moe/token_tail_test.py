# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Token-level TailSFT: ``tail_filter.token_filtered_loss`` against a numpy reference, and the shared key function."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.june_tpu_67b_a2b.moe import tail_filter as tf
from experiments.june_tpu_67b_a2b.moe import token_track as tt


def numpy_reference(per_pos, weight, ref, fraction):
    w = weight.reshape(-1).astype(np.float64); pp = per_pos.reshape(-1).astype(np.float64); r = ref.reshape(-1).astype(np.float64)
    trained = w > 0
    now = np.where(trained, pp / np.where(trained, w, 1.0), 0.0)
    scorable = trained & np.isfinite(r)
    margin = np.where(scorable, now - r, np.inf)
    m = int(scorable.sum()); k = int(np.floor(fraction * m)); k = min(max(k, 0), max(m - 1, 0))
    order = np.argsort(margin, kind="stable"); drop = np.zeros_like(trained); drop[order[:k]] = True; drop &= scorable
    keep = trained & ~drop
    loss = pp[keep].sum() / w[keep].sum(); unf = pp[trained].sum() / w[trained].sum()
    return loss, unf, drop.reshape(per_pos.shape), k


def main():
    rng = np.random.default_rng(0)
    B, S = 4, 64
    weight = (rng.random((B, S)) > 0.3).astype(np.float32); weight[:, :5] = 0.0
    loss_tok = rng.gamma(2.0, 0.3, size=(B, S)).astype(np.float32)
    per_pos = (loss_tok * weight).astype(np.float32)
    ref = (loss_tok + rng.normal(0.0, 0.4, size=(B, S))).astype(np.float32)
    ref[1, 10:20] = np.nan  # never scored: must never be dropped
    for fraction in (0.0, 0.25, 0.5, 0.9):
        loss, stats = tf.token_filtered_loss(jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(ref), fraction)
        loss_np, unf_np, drop_np, k_np = numpy_reference(per_pos, weight, ref, fraction)
        assert abs(float(loss) - loss_np) < 1e-4, (fraction, float(loss), loss_np)
        assert abs(float(stats["tail/loss_unfiltered"]) - unf_np) < 1e-4
        assert int(stats["tail/dropped_tokens"]) == k_np == int(drop_np.sum()), (fraction, int(stats["tail/dropped_tokens"]), k_np)
        assert int(stats["tail/scorable_tokens"]) == int(((weight > 0) & np.isfinite(ref)).sum())
        if fraction == 0.0:
            assert float(stats["tail/kept_token_frac"]) == 1.0
        print(f"fraction {fraction}: loss {float(loss):.4f} unfiltered {unf_np:.4f} dropped {k_np} of "
              f"{int(stats['tail/scorable_tokens'])} scorable, kept {float(stats['tail/kept_token_frac']):.3f}")
    # gradient flows only through kept tokens; unscored tokens are kept
    def f(pp):
        return tf.token_filtered_loss(pp, jnp.asarray(weight), jnp.asarray(ref), 0.5)[0]
    g = np.asarray(jax.grad(f)(jnp.asarray(per_pos)))
    _, _, drop_np, _ = numpy_reference(per_pos, weight, ref, 0.5)
    assert np.all(g[drop_np] == 0), "dropped tokens must carry no gradient"
    assert np.all(g[(weight > 0) & ~drop_np] > 0), "kept trained tokens must carry gradient"
    assert np.all(~drop_np[1, 10:20]), "tokens without a reference are never dropped"
    # the dropped tokens are exactly the most-learned ones (smallest loss - reference)
    margin = np.where((weight > 0) & np.isfinite(ref), per_pos / np.where(weight > 0, weight, 1) - ref, np.inf)
    assert margin[drop_np].max() <= margin[(weight > 0) & np.isfinite(ref) & ~drop_np].min() + 1e-6
    # shared keys: the tracker's method and the module function agree
    seg = np.array([[0, 0, 0, 1, 1, -1], [2, 2, 2, 2, 2, 2]]); w = np.array([[0, 1, 1, 1, 1, 0], [1, 1, 1, 0, 1, 1]], dtype=np.float32)
    k1 = tt.token_keys(seg, w, num_docs=3, max_len=8); k2 = tt.TokenHistory(3, 8).keys(seg, w)
    assert np.array_equal(k1, k2) and k1[0, 1] == 1 and k1[0, 3] == 8 and k1[1, 5] == 2 * 8 + 5 and k1[0, 0] == -1
    print("TOKEN_TAIL_TEST_OK")


if __name__ == "__main__":
    main()

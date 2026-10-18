# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""4-device CPU check of the turn and span units under the shard_map pattern train.py uses.

The first attempt at these units numbered turns on the GLOBAL batch, where finding a turn boundary shifts
the sequence by one position and so slices a sharded dimension — jax refused to compile it, but only on the
real 64-device mesh, after the job had started. This test runs the real sharded path on CPU, checks the
sharded result equals the single-device one, and checks the gradient only reaches the kept spans.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from experiments.june_tpu_67b_a2b.moe import tail_filter as tf

END = 128003
MAX_TURNS = 6
NUM_DOCS = 12


def make_turned_batch(rng, b=8, s=40):
    """A packed batch whose documents hold several marked turns each."""
    seg = np.full((b, s), -1, np.int32)
    weight = np.zeros((b, s), np.float32)
    tokens = rng.integers(0, 500, size=(b, s)).astype(np.int32)
    doc = 0
    for r in range(b):
        pos = 0
        while pos + 9 <= s and doc < NUM_DOCS:
            n_turns = int(rng.integers(1, 4))
            seg[r, pos:pos + 9] = doc
            cursor = pos
            for _ in range(n_turns):
                weight[r, cursor + 1:cursor + 3] = 1.0
                # Position i scores tokens[i+1], so a marker at cursor+2 is the target of the trained
                # position cursor+1: the reasoning span is that position, the action span is cursor+2.
                tokens[r, cursor + 2] = END
                cursor += 3
                if cursor + 3 > pos + 9:
                    break
            pos += 9
            doc += 1
    per_pos = rng.uniform(0.1, 2.0, size=(b, s)).astype(np.float32) * weight
    return per_pos, weight, seg, tokens


def check(unit):
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("data",))
    rng = np.random.default_rng(7)
    per_pos, weight, seg, tokens = make_turned_batch(rng)
    segments = tf.unit_segments(unit, NUM_DOCS, MAX_TURNS)
    # A reference that is finite exactly where this batch trains something, so nothing is unscorable.
    ids_single, overflow_single = tf.unit_segment_ids(
        jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(tokens),
        unit=unit, num_docs=NUM_DOCS, max_turns=MAX_TURNS, end_think_id=END,
    )
    assert int(overflow_single) == 0, int(overflow_single)
    num, den = tf.document_sums(jnp.asarray(per_pos), jnp.asarray(weight), ids_single, segments)
    present = np.asarray(den) > 0
    assert present.sum() >= 8, f"{unit}: only {present.sum()} segments present; test batch is too small"
    if unit == "span":
        # Both halves of a turn must carry tokens, or the batch never exercises the split and the rest of
        # this check would pass just as happily on a segmentation that ignored the marker entirely.
        halves = present.reshape(-1, tf.SPANS_PER_TURN)
        assert halves[:, 0].sum() > 0 and halves[:, 1].sum() > 0, (
            f"span unit: reasoning present {int(halves[:, 0].sum())}, action present {int(halves[:, 1].sum())}; "
            "the marker is not landing on a trained position"
        )
    ref = np.full(segments, np.nan, dtype=np.float32)
    ref[present] = (np.asarray(num)[present] / np.asarray(den)[present]) + rng.uniform(-0.3, 0.3, present.sum())
    fraction = 0.5

    row = NamedSharding(mesh, P("data"))
    rep = NamedSharding(mesh, P())
    pp = jax.device_put(jnp.asarray(per_pos), row)
    w = jax.device_put(jnp.asarray(weight), row)
    sg = jax.device_put(jnp.asarray(seg), row)
    tk = jax.device_put(jnp.asarray(tokens), row)
    rf = jax.device_put(jnp.asarray(ref), rep)
    fr = jax.device_put(jnp.asarray(fraction, dtype=jnp.float32), rep)

    def psum(x):
        return jax.lax.psum(x, "data")

    def local(a, b, c, d, r, f):
        return tf.unit_filtered_loss(
            a, b, c, d, r, f,
            unit=unit, num_docs=NUM_DOCS, max_turns=MAX_TURNS, end_think_id=END, psum=psum,
        )

    stats_specs = {k: P() for k in tf.UNIT_STAT_KEYS}
    sharded = jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P("data"), P("data"), P("data"), P("data"), P(), P()),
        out_specs=(P(), stats_specs), check_vma=False,
    ))
    loss_s, stats_s = sharded(pp, w, sg, tk, rf, fr)

    single = jax.jit(lambda a, b, c, d, r, f: tf.unit_filtered_loss(
        a, b, c, d, r, f, unit=unit, num_docs=NUM_DOCS, max_turns=MAX_TURNS, end_think_id=END))
    loss_1, stats_1 = single(
        jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(tokens),
        jnp.asarray(ref), fraction,
    )
    assert abs(float(loss_s) - float(loss_1)) < 1e-5, (unit, float(loss_s), float(loss_1))
    for k in tf.UNIT_STAT_KEYS:
        a, b = float(stats_s[k]), float(stats_1[k])
        assert (np.isinf(a) and np.isinf(b)) or abs(a - b) < 1e-4, (unit, k, a, b)
    assert float(stats_s["tail/turn_overflow"]) == 0.0
    assert int(stats_s["tail/dropped_docs"]) > 0, f"{unit}: nothing was dropped, the test proves nothing"

    # The gradient must vanish exactly on the dropped segments and be positive on the kept trained ones.
    g = np.asarray(jax.grad(lambda x: sharded(x, w, sg, tk, rf, fr)[0])(pp))
    keep, _ = tf.tail_keep_mask(num, den, jnp.asarray(ref), fraction)
    keep = np.asarray(keep)
    flat_ids = np.asarray(ids_single).reshape(-1)
    trained = flat_ids >= 0
    kept_tok = np.zeros_like(trained)
    kept_tok[trained] = keep[flat_ids[trained]]
    gflat = g.reshape(-1)
    assert np.all(gflat[trained & ~kept_tok] == 0), f"{unit}: dropped spans still carry gradient"
    assert np.all(gflat[trained & kept_tok] > 0), f"{unit}: kept spans carry no gradient"
    print(
        f"{unit}: sharded {float(loss_s):.5f} == single {float(loss_1):.5f}; "
        f"segments present {int(present.sum())}, dropped {int(stats_s['tail/dropped_docs'])}, "
        f"kept tokens {float(stats_s['tail/kept_token_frac']):.3f}"
    )


def main():
    for unit in ("turn", "span"):
        check(unit)
    print("TAIL_UNIT_SHARD_TEST_OK")


if __name__ == "__main__":
    main()

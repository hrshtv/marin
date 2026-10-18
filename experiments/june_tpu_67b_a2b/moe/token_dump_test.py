# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""4-device CPU check of the scoring pass's token dump: shards reassemble to the full batch, the span ids match
the filter's own segmentation, and the per-row token shift recovers the scored target."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from experiments.june_tpu_67b_a2b.moe import tail_filter as tf
from experiments.june_tpu_67b_a2b.moe.tail_unit_shard_test import make_turned_batch, END, MAX_TURNS, NUM_DOCS
from experiments.june_tpu_67b_a2b.moe.train import _local_rows

def main():
    mesh = Mesh(np.array(jax.devices()[:4]), ("data",))
    rng = np.random.default_rng(3)
    per_pos, weight, seg, tokens = make_turned_batch(rng)
    row = NamedSharding(mesh, P("data"))
    put = lambda a: jax.device_put(jnp.asarray(a), row)
    def local(a, b, c, d):
        ids, _ = tf.unit_segment_ids(b, c, d, unit="span", num_docs=NUM_DOCS, max_turns=MAX_TURNS, end_think_id=END)
        return a.astype(jnp.float32), b.astype(jnp.float32), ids.reshape(d.shape).astype(jnp.int32), d.astype(jnp.int32)
    f = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("data"),) * 4, out_specs=(P("data"),) * 4, check_vma=False))
    out = f(put(per_pos), put(weight), put(seg), put(tokens))
    loss, w, span, tok = (_local_rows(o) for o in out)        # single host: all rows
    assert loss.shape == per_pos.shape and np.allclose(loss, per_pos)
    assert np.array_equal(tok, tokens)
    want, _ = tf.unit_segment_ids(jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(tokens),
                                  unit="span", num_docs=NUM_DOCS, max_turns=MAX_TURNS, end_think_id=END)
    assert np.array_equal(span.reshape(-1), np.asarray(want)), "dumped span ids differ from the filter's"
    # The offline reader's shift: position i scores tokens[i+1]; the marker is the target of a reasoning position.
    target = np.concatenate([tok[:, 1:], np.full((tok.shape[0], 1), -1, np.int32)], axis=1)
    marker = (target == END) & (w > 0)
    assert marker.sum() > 0 and np.all(span[marker] % tf.SPANS_PER_TURN == 0), "marker must land in a reasoning span"
    print(f"rows {loss.shape[0]} x {loss.shape[1]}; trained positions {int((w > 0).sum())}; markers {int(marker.sum())}")
    print("TOKEN_DUMP_TEST_OK")

if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU test for span_segment_ids: turn detection, the reasoning/action split, and the nesting invariant.

The nesting invariant is what lets ONE scoring pass serve all three units: summing a document's span
numerators and denominators must reproduce exactly what ``document_sums`` returns for that document.
"""
import numpy as np
import jax
import jax.numpy as jnp

from experiments.june_tpu_67b_a2b.moe import tail_filter as tf

jax.config.update("jax_platforms", "cpu")

END = 128003
MAX_TURNS = 8


def ids_for(weight, seg, tokens, num_docs=4, max_turns=MAX_TURNS):
    out, overflow = tf.span_segment_ids(
        jnp.asarray(weight, dtype=jnp.float32),
        jnp.asarray(seg, dtype=jnp.int32),
        jnp.asarray(tokens, dtype=jnp.int32),
        num_docs=num_docs,
        max_turns=max_turns,
        end_think_id=END,
    )
    return np.asarray(out), int(overflow)


def span_of(doc, turn, span):
    return (doc * MAX_TURNS + turn) * tf.SPANS_PER_TURN + span


def test_single_document_three_turns():
    """Position i of the loss scores tokens[i+1], so the reasoning span ends at the position BEFORE END."""
    #         0  1  2    3    4  5  6  7    8    9  10 11 12
    tokens = [9, 9, 5, END,   7, 9, 9, 5, END,   7, 7, 9, 7]
    weight = [0, 0, 1,   1,   1, 0, 0, 1,   1,   1, 1, 0, 1]
    seg = [0] * 13
    ids, overflow = ids_for(weight, seg, tokens)
    assert overflow == 0
    # Turn 0 covers positions 2-4. END sits at token 3, so it is the target of position 2: reasoning is
    # position 2 alone and the action is 3-4. Turn 1 covers 7-10 with END at token 8, target of position 7.
    want = [-1, -1, span_of(0, 0, 0), span_of(0, 0, 1), span_of(0, 0, 1),
            -1, -1, span_of(0, 1, 0), span_of(0, 1, 1), span_of(0, 1, 1), span_of(0, 1, 1),
            -1, span_of(0, 2, 0)]
    assert ids.tolist() == want, (ids.tolist(), want)


def test_turn_without_marker_is_all_span_zero():
    tokens = [7, 7, 7]
    weight = [1, 1, 1]
    ids, _ = ids_for(weight, [0, 0, 0], tokens)
    assert ids.tolist() == [span_of(0, 0, 0)] * 3


def test_turn_index_restarts_in_each_document():
    tokens = [5, END, 7, 9, 5, END, 7, 9, 5, END, 7]
    weight = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    seg = [0, 0, 0, 0, 0, 0, 0, -1, 3, 3, 3]
    ids, overflow = ids_for(weight, seg, tokens)
    assert overflow == 0
    # END at token 1 is the target of position 0, so each turn's reasoning is its first position only.
    assert ids.tolist() == [
        span_of(0, 0, 0), span_of(0, 0, 1), span_of(0, 0, 1),
        -1,
        span_of(0, 1, 0), span_of(0, 1, 1), span_of(0, 1, 1),
        -1,
        span_of(3, 0, 0), span_of(3, 0, 1), span_of(3, 0, 1),
    ]


def test_adjacent_documents_without_a_gap_still_split():
    """Two documents packed back to back with no unsupervised position between them."""
    tokens = [7, 7, 7, 7]
    weight = [1, 1, 1, 1]
    seg = [0, 0, 1, 1]
    ids, _ = ids_for(weight, seg, tokens)
    assert ids.tolist() == [span_of(0, 0, 0), span_of(0, 0, 0), span_of(1, 0, 0), span_of(1, 0, 0)]


def test_overflow_is_reported():
    tokens, weight, seg = [], [], []
    for _ in range(MAX_TURNS + 3):
        tokens += [7, 9]
        weight += [1, 0]
        seg += [0, 0]
    _, overflow = ids_for(weight, seg, tokens)
    assert overflow == 3, overflow


def test_spans_sum_back_to_documents():
    """Summing a document's spans must reproduce document_sums exactly, for any batch shape."""
    rng = np.random.default_rng(0)
    b, s, num_docs = 3, 40, 4
    seg = np.full((b, s), -1, np.int32)
    weight = np.zeros((b, s), np.float32)
    tokens = rng.integers(0, 50, size=(b, s)).astype(np.int32)
    doc = 0
    for r in range(b):
        pos = 0
        while pos < s - 6 and doc < num_docs:
            length = int(rng.integers(4, 9))
            seg[r, pos:pos + length] = doc
            weight[r, pos + 1:pos + length] = 1.0  # first position is context
            tokens[r, pos + 3] = END                # target of position 2: marker inside the turn
            pos += length
            doc += 1
    per_pos = rng.uniform(0.1, 2.0, size=(b, s)).astype(np.float32) * weight

    ids, overflow = tf.span_segment_ids(
        jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(tokens),
        num_docs=num_docs, max_turns=MAX_TURNS, end_think_id=END,
    )
    assert int(overflow) == 0
    n_span, d_span = tf.document_sums(
        jnp.asarray(per_pos), jnp.asarray(weight), ids, num_docs * MAX_TURNS * tf.SPANS_PER_TURN
    )
    n_doc, d_doc = tf.document_sums(jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(seg), num_docs)
    block = MAX_TURNS * tf.SPANS_PER_TURN
    rolled_n = np.asarray(n_span).reshape(num_docs, block).sum(1)
    rolled_d = np.asarray(d_span).reshape(num_docs, block).sum(1)
    assert np.allclose(rolled_n, np.asarray(n_doc), atol=1e-5), (rolled_n, np.asarray(n_doc))
    assert np.allclose(rolled_d, np.asarray(d_doc), atol=1e-5)
    # And the two spans of a turn must roll up to that turn.
    pair_n = np.asarray(n_span).reshape(-1, tf.SPANS_PER_TURN).sum(1)
    assert np.isclose(pair_n.sum(), np.asarray(n_doc).sum(), atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("SPAN_SEGMENT_TEST_OK")

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-token clipping for SFT ("clip SFT").

Stop training on tokens the model already predicts with high probability (``p > p_hi``) and, optionally, on
tokens it predicts with very low probability (``p < p_lo``), so each step's gradient lands on the tokens in
between. It is the token-level analogue of the RL band probe: skip what is solved, skip what is hopeless.
``random_frac`` instead drops trained tokens at random, the control that separates "which tokens" from "how
much of the update each kept token gets".

Works on the trainer's per-position loss, which is cross-entropy plus the logit z-loss term (1e-4 x lse^2,
roughly 0.003-0.02 nats) times the loss weight. ``p`` is read from that same quantity, so a threshold chosen
from the scoring pass's token dump means exactly what it means here; ``p`` is a hair pessimistic.

Runs INSIDE ``jax.shard_map`` on local rows, like ``tail_filter``. The random control's coin flips are drawn
outside on the global batch and passed in as a row-sharded array, so they do not depend on the mesh layout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experiments.june_tpu_67b_a2b.moe import tail_filter

CLIP_STAT_KEYS: tuple[str, ...] = (
    "clip/kept_token_frac",
    "clip/above_hi_frac",
    "clip/below_lo_frac",
    "clip/random_dropped_frac",
    "clip/loss_unfiltered",
    "clip/above_hi_frac_reasoning",
    "clip/above_hi_frac_command",
    "clip/below_lo_frac_reasoning",
    "clip/below_lo_frac_command",
    "clip/trained_tokens",
    "clip/kept_tokens",
)


def token_probability(per_pos: jax.Array, weight: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(p, trained)`` per flat position: ``p = exp(-loss)`` where ``loss = per_pos / weight``."""
    w = weight.reshape(-1).astype(jnp.float32)
    pp = per_pos.reshape(-1).astype(jnp.float32)
    trained = w > 0
    loss = jnp.where(trained, pp / jnp.where(trained, w, 1.0), 0.0)
    return jnp.exp(-loss), trained


def clip_keep_mask(
    per_pos: jax.Array,
    weight: jax.Array,
    random_keep: jax.Array | None,
    *,
    p_hi: float | None,
    p_lo: float | None,
    active=True,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Which flat positions keep their gradient, plus the boolean parts the stats are built from.

    ``active`` (a traced bool) switches the clip on; while it is False every trained token is kept and the
    above/below/random parts are reported as empty - used to start an ablation at the same step the tracking
    rule starts dropping."""
    p, trained = token_probability(per_pos, weight)
    none = jnp.zeros_like(trained)
    above = (trained & (p > p_hi)) if p_hi is not None else none
    below = (trained & (p < p_lo)) if p_lo is not None else none
    dropped = (trained & ~random_keep.reshape(-1)) if random_keep is not None else none
    above, below, dropped = (m & active for m in (above, below, dropped))
    keep = trained & ~above & ~below & ~dropped
    parts = {"trained": trained, "above": above, "below": below, "random": dropped}
    return jax.lax.stop_gradient(keep), {k: jax.lax.stop_gradient(v) for k, v in parts.items()}


def clipped_loss(
    per_pos: jax.Array,
    weight: jax.Array,
    segment_ids: jax.Array,
    tokens: jax.Array,
    random_keep: jax.Array | None,
    *,
    p_hi: float | None,
    p_lo: float | None,
    num_docs: int,
    max_turns: int,
    end_think_id: int,
    active=True,
    psum: tail_filter.PsumFn = tail_filter._identity,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Token-mean loss over the kept positions, with replicated diagnostics.

    ``max_turns > 0`` also splits the clipped fractions into reasoning vs command spans using the same local
    segmentation the TailSFT filter uses; with ``max_turns == 0`` those four stats are reported as -1.
    """
    keep, parts = clip_keep_mask(per_pos, weight, random_keep, p_hi=p_hi, p_lo=p_lo, active=active)
    pp = per_pos.reshape(-1).astype(jnp.float32)
    w = weight.reshape(-1).astype(jnp.float32)
    f = lambda m: psum(jnp.sum(m.astype(jnp.float32)))
    loss_sum = psum(jnp.sum(jnp.where(keep, pp, 0.0)))
    w_kept = psum(jnp.sum(jnp.where(keep, w, 0.0)))
    loss_all = psum(jnp.sum(pp))
    w_all = psum(jnp.sum(w))
    n_trained = jnp.maximum(f(parts["trained"]), 1.0)
    loss = jnp.where(w_kept > 0, loss_sum / jnp.where(w_kept > 0, w_kept, 1.0), 0.0)
    unfiltered = jnp.where(w_all > 0, loss_all / jnp.where(w_all > 0, w_all, 1.0), 0.0)
    stats = {
        "clip/kept_token_frac": f(keep) / n_trained,
        "clip/above_hi_frac": f(parts["above"]) / n_trained,
        "clip/below_lo_frac": f(parts["below"]) / n_trained,
        "clip/random_dropped_frac": f(parts["random"]) / n_trained,
        "clip/loss_unfiltered": unfiltered,
    }
    if max_turns > 0:
        ids, _ = tail_filter.unit_segment_ids(
            weight, segment_ids, tokens, unit="span", num_docs=num_docs, max_turns=max_turns, end_think_id=end_think_id
        )
        reasoning = parts["trained"] & (ids >= 0) & (ids % tail_filter.SPANS_PER_TURN == 0)
        command = parts["trained"] & (ids >= 0) & (ids % tail_filter.SPANS_PER_TURN == 1)
        n_r, n_c = jnp.maximum(f(reasoning), 1.0), jnp.maximum(f(command), 1.0)
        stats["clip/above_hi_frac_reasoning"] = f(parts["above"] & reasoning) / n_r
        stats["clip/above_hi_frac_command"] = f(parts["above"] & command) / n_c
        stats["clip/below_lo_frac_reasoning"] = f(parts["below"] & reasoning) / n_r
        stats["clip/below_lo_frac_command"] = f(parts["below"] & command) / n_c
    else:
        for k in ("above_hi_frac_reasoning", "above_hi_frac_command", "below_lo_frac_reasoning", "below_lo_frac_command"):
            stats[f"clip/{k}"] = jnp.float32(-1.0)
    stats["clip/trained_tokens"] = f(parts["trained"])
    stats["clip/kept_tokens"] = f(keep)
    stats = {k: jax.lax.stop_gradient(v) for k, v in stats.items()}
    assert tuple(stats) == CLIP_STAT_KEYS, (tuple(stats), CLIP_STAT_KEYS)
    return loss, stats

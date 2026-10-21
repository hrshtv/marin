# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter SFT of a Qwen3 checkpoint (Qwen/Qwen3-8B) with Plateau Dropping, mirroring the Grug trainer exactly.

The model, optimizer, sharding and checkpointing are Levanter's (``Qwen3Config`` initialised from the HF
checkpoint, AdamW, FSDP over every device, the Levanter checkpointer). Everything that decides which tokens train
is Grug's, reused rather than re-implemented:

* data: ``LmDataConfig`` from ``vista_snowball_chat.snowball_chat_data_config`` and ``train.build_train_dataset``
  with Grug's key derivation (``PRNGKey(seed)`` split, or ``PRNGKey(SNOWBALL_DATA_SEED)``), so for the same seed and
  batch size every step draws the same cache rows (segment ids) as ``train.py``;
* loss: the per-position next-token loss including the z-loss term, times the loss weight (what Grug's
  ``next_token_loss(reduction="none", logsumexp_weight=z_loss)`` returns), reduced either as the plain token mean
  or by ``token_track.tracked_loss`` unchanged;
* host side: ``track_step_inputs`` / ``track_step_record`` are ``train.py``'s per-step host block (keys from
  segment ids, gather, train step with the history inputs, all-gather of the per-position loss, ``loss_now``, trace,
  ``update_ema`` or ``decide`` + ``update``, duplicate-key warning), and the history is saved beside each checkpoint
  and restored on resume exactly as ``train.py`` does.

Run ``python -m experiments.plateau_dropping_qwen.qwen_sft train --help``. Settings that the Grug launcher reads
from ``SNOWBALL_*`` environment variables are read from the same variables with the same defaults; an explicit
command-line flag wins over the environment.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import click
import numpy as np

logger = logging.getLogger(__name__)

# Defaults shared with the Grug launcher (vista_snowball_chat.py / snowball_chat_recipe.py). Kept as literals here
# so that `--help` does not need the Grug recipe imported; `grug_defaults_match` checks them in the tests.
DEFAULT_SEQ_LEN = 32_768  # SNOWBALL_CHAT_SEQUENCE_LENGTH without SNOWBALL_SEQ_LEN
DEFAULT_BATCH_SIZE = 128  # sequences per step (the Grug Snowball runs use 64: SNOWBALL_CHAT_BATCH_SIZE)
DEFAULT_SEED = 0  # SNOWBALL_CHAT_SEED
DEFAULT_Z_LOSS = 1e-4  # GrugTrainerConfig(z_loss_weight=1e-4) in snowball_chat_run_config
DEFAULT_STAGE = "openswe_v12_msa_qwen3_32k"
DEFAULT_MP = "params=float32,compute=bfloat16,output=float32"
QWEN3_HF_VOCAB = 151_936


# --------------------------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenSftConfig:
    """Everything one run needs. Paths come from the command line; nothing machine-specific has a default."""

    hf_checkpoint: str
    data_cache_path: str
    tokenizer_path: str
    output_path: str
    run_id: str
    steps: int
    stage: str = DEFAULT_STAGE
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    per_device_parallelism: int = -1  # -1: the whole per-device batch at once (no gradient accumulation)
    seed: int = DEFAULT_SEED
    data_seed: int | None = None
    z_loss_weight: float = DEFAULT_Z_LOSS
    # AdamW (Levanter AdamConfig). The LR schedule spans schedule_steps (>= steps; None = steps).
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup: float = 0.03  # < 1: fraction of the schedule (at least one step); >= 1: steps
    min_lr_ratio: float = 0.1
    lr_schedule: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    schedule_steps: int | None = None
    # Checkpoints: temporary every save_interval_minutes, permanent every keep_every steps.
    keep_every: int = 1000
    save_interval_minutes: float = 30.0
    hf_save_path: str | None = None  # default <output>/hf
    hf_save_dtype: str = "bfloat16"
    # Plateau Dropping (token_track.py), same names / defaults as GrugTrainerConfig.
    track_tol: float = 0.0
    track_floor: float = 0.05
    track_max_len: int = DEFAULT_SEQ_LEN
    track_ema_alpha: float = 0.0
    track_reenter: str = "either"
    track_drop_weight: float = 0.0
    track_observe_only: bool = False
    trace_num_docs: int = 0
    trace_seed: int = 0
    trace_out: str | None = None
    # Model / runtime.
    attn_backend: str = "jax_flash"
    flash_block_size: int | None = None
    gradient_checkpointing: bool = True
    mp: str = DEFAULT_MP
    log_every: int = 1
    tracker: str = "wandb"
    wandb_project: str = "plateau_dropping_qwen"
    wandb_mode: str = "offline"
    log_dir: str = "logs/"
    allow_cpu: bool = False  # tests only: run the training loop on CPU


def snowball_env_settings(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The run settings the Grug launcher reads from the environment, with ``vista_snowball_chat``'s exact parsing.

    Keys are ``QwenSftConfig`` field names. ``steps`` / ``keep_every`` are ``None`` when unset (the CLI and the cache
    decide); everything else carries the Grug default.
    """
    env = os.environ if environ is None else environ
    e = env.get
    seq_len = int(e("SNOWBALL_SEQ_LEN") or DEFAULT_SEQ_LEN)
    return {
        "seq_len": seq_len,
        "steps": int(e("SNOWBALL_STEPS")) if e("SNOWBALL_STEPS") else None,
        "schedule_steps": int(e("SNOWBALL_SCHEDULE_STEPS") or 0) or None,
        "data_seed": int(e("SNOWBALL_DATA_SEED")) if e("SNOWBALL_DATA_SEED") else None,
        "learning_rate": float(e("SNOWBALL_LR")) if e("SNOWBALL_LR") else None,
        "keep_every": int(e("SNOWBALL_KEEP_EVERY") or 0) or None,
        "keep_per_epoch": e("SNOWBALL_KEEP_PER_EPOCH") == "1",
        "track_tol": float(e("SNOWBALL_TRACK_TOL") or 0.0),
        "track_floor": float(e("SNOWBALL_TRACK_FLOOR") or 0.05),
        "track_max_len": int(e("SNOWBALL_TRACK_MAX_LEN") or seq_len),
        "track_ema_alpha": float(e("SNOWBALL_TRACK_EMA_ALPHA") or 0.0),
        "track_reenter": e("SNOWBALL_TRACK_REENTER") or "either",
        "track_drop_weight": float(e("SNOWBALL_TRACK_DROP_WEIGHT") or 0.0),
        "track_observe_only": e("SNOWBALL_TRACK_OBSERVE_ONLY", "0") == "1",
        "trace_num_docs": int(e("SNOWBALL_TRACE_DOCS") or 0),
        "trace_seed": int(e("SNOWBALL_TRACE_SEED") or 0),
        "trace_out": e("SNOWBALL_TRACE_OUT") or None,
        "log_dir": e("SNOWBALL_LOG_DIR", "logs/"),
    }


def expected_epoch_steps(total_tokens: int, seq_len: int, batch_size: int) -> int:
    """``vista_snowball_chat.expected_chat_steps`` at this run's sequence length and batch size."""
    return math.ceil(total_tokens / (seq_len * batch_size))


# --------------------------------------------------------------------------------------------------------------------
# Plateau Dropping: settings and the host-side step (train.py's block, verbatim semantics)
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class TrackRuntime:
    """Resolved tracking for one run: the settings ``tracked_loss`` sees, the history table and the optional trace."""

    settings: Any  # train.TrackSettings
    history: Any  # token_track.TokenHistory
    trace: Any = None  # token_track.TraceRecorder | None


def resolve_tracking(cfg: QwenSftConfig, num_docs: int | None, *, process_index: int = 0) -> TrackRuntime | None:
    """``train.run_grug_local``'s tracking validation and construction, in the same order with the same messages."""
    from experiments.june_tpu_67b_a2b.moe import token_track  # noqa: PLC0415
    from experiments.june_tpu_67b_a2b.moe.train import TrackSettings  # noqa: PLC0415

    if cfg.trace_num_docs > 0 and not (cfg.track_tol > 0 or cfg.track_observe_only):
        raise ValueError("the token trace needs tracking on (track_tol > 0 or track_observe_only)")
    if not (cfg.track_tol > 0 or cfg.track_observe_only):
        return None
    if cfg.track_observe_only and cfg.track_tol > 0:
        raise ValueError("track_observe_only records without dropping; it cannot be combined with track_tol > 0")
    if num_docs is None:
        raise ValueError("track_tol > 0 needs tail_num_docs (the cache's document count)")
    # Observe-only uses tol 0: "relative change < 0" never holds, so every trained token is kept.
    tol = 0.0 if cfg.track_observe_only else float(cfg.track_tol)
    settings = TrackSettings(
        tol=tol,
        floor=float(cfg.track_floor),
        ema_alpha=float(cfg.track_ema_alpha),
        reentry=token_track.Reentry(cfg.track_reenter),
        drop_weight=float(cfg.track_drop_weight),
    )
    if not 0.0 <= settings.drop_weight < 1.0:
        raise ValueError(f"track_drop_weight must be in [0, 1), got {settings.drop_weight}")
    if settings.reentry is not token_track.Reentry.EITHER and (settings.ema_alpha > 0 or cfg.trace_num_docs > 0):
        raise ValueError("track_reenter other than 'either' supports neither track_ema_alpha nor the token trace")
    history = token_track.TokenHistory(int(num_docs), int(cfg.track_max_len))
    trace = None
    if cfg.trace_num_docs > 0:
        if not cfg.trace_out:
            raise ValueError("trace_num_docs > 0 needs trace_out")
        if settings.ema_alpha > 0:
            raise ValueError("the token trace records the hard plateau rule; it does not support track_ema_alpha")
        n = min(int(cfg.trace_num_docs), history.num_docs)
        trace_docs = np.random.default_rng(int(cfg.trace_seed)).choice(history.num_docs, size=n, replace=False)
        trace = token_track.TraceRecorder(
            trace_docs, history.max_len, cfg.trace_out, tol=settings.tol, floor=settings.floor
        )
        if process_index == 0:
            os.makedirs(cfg.trace_out, exist_ok=True)
            np.save(os.path.join(cfg.trace_out, "trace-docs.npy"), np.sort(trace_docs))
        logger.info(f"Token trace on: {n} documents (seed {cfg.trace_seed}) -> {cfg.trace_out}")
    logger.info(
        f"Per-token tracking on: tol={settings.tol} floor={settings.floor} ema_alpha={settings.ema_alpha} "
        f"reentry={settings.reentry} drop_weight={settings.drop_weight} table "
        f"{history.num_docs} docs x {history.max_len} = {history.num_docs * history.max_len:,} keys"
    )
    return TrackRuntime(settings=settings, history=history, trace=trace)


def track_step_inputs(history, seg_np: np.ndarray, w_np: np.ndarray):
    """Before the step: per-position keys and the history inputs (``train.py``: ``keys`` then ``gather``)."""
    keys = history.keys(seg_np, w_np)
    p1, p2, cnt, prev_dropped = history.gather(keys)
    return keys, p1, p2, cnt, prev_dropped


def track_step_record(
    track: TrackRuntime,
    *,
    step: int,
    keys: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    cnt: np.ndarray,
    prev_dropped: np.ndarray,
    w_np: np.ndarray,
    pp_np: np.ndarray,
    tok_np: np.ndarray | None = None,
    record_trace: bool = True,
) -> dict[str, int]:
    """After the step: record this visit into the history (and the trace), exactly as ``train.py`` does.

    ``pp_np`` is the all-gathered per-position loss of the step (already multiplied by the weight), ``tok_np`` the
    all-gathered input tokens (only needed with a trace; gathering is a collective, so the caller does it on every
    process and passes ``record_trace = process_index == 0``). ``step`` is the step just taken (0-based).
    """
    from experiments.june_tpu_67b_a2b.moe import token_track  # noqa: PLC0415

    settings, history = track.settings, track.history
    loss_now = np.where(w_np > 0, pp_np / np.where(w_np > 0, w_np, 1.0), np.nan)
    if track.trace is not None and record_trace:
        # The loss at position t scores token t + 1 (the same shift next_token_loss uses).
        track.trace.record(step, keys, np.roll(tok_np, -1, axis=-1), loss_now, p1, p2, cnt)
    if settings.ema_alpha > 0:
        diag = history.update_ema(keys, loss_now, floor=settings.floor, alpha=settings.ema_alpha)
    else:
        dropped_now = token_track.decide(
            p1, p2, cnt, prev_dropped, loss_now, tol=settings.tol, floor=settings.floor, reentry=settings.reentry
        )
        diag = history.update(keys, loss_now, dropped_now)
    if diag["duplicate_keys"]:
        logger.warning(
            "track step %d: %d of %d token keys repeated within the batch (%d documents); first occurrence kept",
            step,
            diag["duplicate_keys"],
            diag["keys"],
            diag["documents_with_duplicates"],
        )
    return diag


def history_path_for(checkpoint_base_path: str) -> str:
    return os.path.join(checkpoint_base_path, "token-history.npz")


def restore_history(track: TrackRuntime | None, path: str | None, step: int) -> bool:
    """On resume (``step > 0``) restore the table saved for exactly this step; otherwise warn and start fresh."""
    if track is None or step <= 0:
        return False
    if path is not None and track.history.load(path, step):
        logger.info(f"Per-token tracking: restored token history for step {step} from {path}")
        return True
    logger.warning(f"Per-token tracking: no token history saved for step {step}; tracking restarts from empty histories")
    return False


class HistorySaver:
    """Saves the table beside each checkpoint the checkpointer writes (``train.py``'s ``track_saved_step`` logic)."""

    def __init__(self, track: TrackRuntime | None, path: str | None, start_step: int, *, is_writer: bool):
        self.track, self.path, self.saved_step, self.is_writer = track, path, int(start_step), is_writer

    def after_checkpoint(self, last_save_step: int) -> None:
        if self.track is None or last_save_step == self.saved_step:
            return
        self.saved_step = last_save_step
        if self.is_writer:
            self.track.history.save(self.path, self.saved_step)

    def final(self, last_save_step: int) -> None:
        if self.track is not None and last_save_step != self.saved_step and self.is_writer:
            self.track.history.save(self.path, last_save_step)


# --------------------------------------------------------------------------------------------------------------------
# Device side: per-position loss, the objective, the train step
# --------------------------------------------------------------------------------------------------------------------


def effective_loss_weight(example):
    """The weight Levanter's next-token loss applies: the example's loss weight with the final position masked."""
    from levanter.models.loss import next_token_loss_weight  # noqa: PLC0415

    Pos = example.tokens.resolve_axis("position")
    return next_token_loss_weight(Pos, example.loss_weight)


def per_position_loss(model, example, *, mp, z_loss_weight: float, compute_mapping=None):
    """``(pp, aux)``: the ``(B, S)`` float32 per-position loss, z-loss included, times the effective weight, and the
    model's scalar auxiliary loss (the MoE router load-balancing loss of Qwen3-MoE; 0 for dense Qwen3).

    ``pp`` is the same quantity as Grug's ``next_token_loss(..., reduction="none", logsumexp_weight=z_loss)``:
    ``(CE + z * logsumexp^2) * weight``, zero where the weight is zero. The aux loss is returned separately so that
    ``objective`` adds it once; ``LmHeadModel.compute_next_token_loss(reduction=None)`` instead broadcasts it onto
    every position, which would put it into the per-token tracked losses and into ``sum(pp) / sum(weight)`` for
    every unweighted position too (the bug before 2026-10-21).
    """
    import haliax as hax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from levanter.models.lm_model import split_activations  # noqa: PLC0415
    from levanter.models.loss import maybe_fused_next_token_loss  # noqa: PLC0415

    z_loss = z_loss_weight if z_loss_weight > 0 else None
    with hax.axis_mapping(compute_mapping) if compute_mapping is not None else _nullcontext():
        compute_model = mp.cast_to_compute(model)
        activations, aux = split_activations(compute_model.activations(example.tokens, example.attn_mask, key=None))
        pp = maybe_fused_next_token_loss(
            compute_model.Pos, compute_model.Embed, compute_model.Vocab, activations, compute_model.get_lm_head(),
            example.tokens, loss_weight=example.loss_weight, reduction=None, reduction_axis=None,
            logsumexp_weight=z_loss, dtype=jnp.float32, logit_soft_cap=None,
        )
    aux = jnp.asarray(aux.array if isinstance(aux, hax.NamedArray) else aux, dtype=jnp.float32)
    return pp, aux


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def track_kwargs(settings) -> dict[str, Any]:
    return dict(
        tol=settings.tol,
        floor=settings.floor,
        ema_alpha=settings.ema_alpha,
        reentry=settings.reentry,
        drop_weight=settings.drop_weight,
    )


def objective(per_pos, weight, hist, settings, aux=0.0):
    """``(loss, metrics)`` from the per-position loss: Grug's plain token mean, or ``tracked_loss`` unchanged, plus
    the model's scalar auxiliary loss ``aux`` (added once; it never enters the per-token statistics).

    ``per_pos`` / ``weight`` / ``hist`` are raw ``(B, S)`` arrays (``hist`` = prev1, prev2, count, prev_dropped).
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from experiments.june_tpu_67b_a2b.moe import token_track  # noqa: PLC0415

    if settings is None:
        num = jnp.sum(per_pos.astype(jnp.float32))
        den = jnp.sum(weight.astype(jnp.float32))
        loss = jnp.where(den != 0, num / jnp.where(den != 0, den, 1.0), jnp.zeros_like(den))
        metrics = {"train/cross_entropy_loss": loss}
    else:
        loss, stats = token_track.tracked_loss(per_pos, weight, *hist, **track_kwargs(settings))
        metrics = {"train/cross_entropy_loss": stats["track/loss_unfiltered"]}
        metrics.update({f"train/{k}": v for k, v in stats.items()})
    metrics["train/aux_loss"] = jnp.asarray(aux, dtype=jnp.float32)
    return loss + aux, metrics


def token_multiplier(
    per_pos, weight, prev1, prev2, count, prev_dropped, *, tol, floor, ema_alpha=0.0, reentry=None, drop_weight=0.0
):
    """Per-position weight ``tracked_loss`` gives each token: 1 kept, ``drop_weight`` dropped, 0 not trained.

    The same rule as ``token_track.tracked_loss`` line for line, unflattened; used only to accumulate gradients over
    microbatches (``loss = sum(m * per_pos) / sum(m * weight)``). The reported loss and statistics always come from
    ``tracked_loss`` itself; ``plateau_equivalence_test`` checks the two agree.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from experiments.june_tpu_67b_a2b.moe import token_track  # noqa: PLC0415

    reentry = token_track.Reentry.EITHER if reentry is None else reentry
    w = weight.astype(jnp.float32)
    pp = per_pos.astype(jnp.float32)
    trained = w > 0
    now = jnp.where(trained, pp / jnp.where(trained, w, 1.0), 0.0)
    p1 = prev1.astype(jnp.float32)
    p2 = prev2.astype(jnp.float32)
    c = count.astype(jnp.float32)
    eligible = trained & (c >= 2.0) & jnp.isfinite(p1) & jnp.isfinite(p2)

    def rel(a, b):
        return jnp.abs(b - a) / jnp.maximum(a, floor)

    if ema_alpha > 0:
        ema_now = (1.0 - ema_alpha) * p2 + ema_alpha * rel(p1, now)
        flat = eligible & (ema_now < tol)
    else:
        flat = eligible & (rel(p2, p1) < tol) & (rel(p1, now) < tol)
    if reentry is token_track.Reentry.RISE:
        was = trained & (prev_dropped > 0.5) & jnp.isfinite(p1)
        flat = jnp.where(was, (now - p1) / jnp.maximum(p1, floor) < tol, flat)
    keep = trained & ~flat
    drop = trained & flat
    m = keep.astype(jnp.float32)
    if drop_weight > 0:
        m = m + drop_weight * drop.astype(jnp.float32)
    return jax.lax.stop_gradient(m)


def loss_and_grads(
    model,
    example,
    hist,
    *,
    mp,
    z_loss_weight,
    track_settings,
    compute_mapping=None,
    num_microbatches: int = 1,
    accum_mapping=None,
):
    """``(loss, metrics, per_pos, grads)`` for one global batch.

    One microbatch (the default) differentiates ``objective`` itself, i.e. Grug's plain mean or ``tracked_loss``.
    With ``num_microbatches > 1`` the batch is split along its batch axis; each microbatch contributes the gradient
    of ``sum(m * per_pos)`` with ``m`` the per-token multiplier of ``tracked_loss`` (1 in plain mode), and the sum is
    divided at the end by the global ``sum(m * weight)`` - the same gradient as the one-shot objective. Loss and
    metrics are computed from the full batch's per-position loss by ``objective``.
    """
    import equinox as eqx  # noqa: PLC0415
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    weight_named = effective_loss_weight(example)

    def raw(x):
        return None if x is None else x.array

    if num_microbatches <= 1:

        def loss_fn(m):
            pp, aux = per_position_loss(m, example, mp=mp, z_loss_weight=z_loss_weight, compute_mapping=compute_mapping)
            hist_raw = None if hist is None else tuple(raw(h) for h in hist)
            loss, metrics = objective(pp.array, weight_named.array, hist_raw, track_settings, aux=aux)
            return loss, (metrics, jax.lax.stop_gradient(pp))

        (loss, (metrics, pp)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        return loss, metrics, pp, grads

    Batch = example.tokens.resolve_axis("batch")
    if Batch.size % num_microbatches != 0:
        raise ValueError(f"batch {Batch.size} is not divisible into {num_microbatches} microbatches")
    Micro = Batch.resize(Batch.size // num_microbatches)
    Accum = hax.Axis("accum_step", num_microbatches)

    def split(x):
        if isinstance(x, hax.NamedArray) and x.has_axis(Batch.name):
            x = x.unflatten_axis(Batch, (Accum, Micro))
            return hax.shard(x, compute_mapping) if compute_mapping is not None else x
        return x

    is_named = lambda x: isinstance(x, hax.NamedArray)  # noqa: E731
    ex_split = jax.tree_util.tree_map(split, example, is_leaf=is_named)
    hist_split = None if hist is None else tuple(split(h) for h in hist)

    # Models with an auxiliary loss (Qwen3-MoE) accumulate its gradient separately: it is a mean over microbatches,
    # not a share of sum(m * per_pos) / sum(m * weight), so it cannot ride on the numerator's accumulator.
    has_aux = getattr(getattr(model, "config", None), "router_aux_loss_coef", None) is not None

    def micro_step(acc, xs):
        ex_mb, hist_mb = xs
        acc_num, acc_aux, aux_sum = acc

        def num_fn(m):
            pp, aux_mb = per_position_loss(m, ex_mb, mp=mp, z_loss_weight=z_loss_weight, compute_mapping=compute_mapping)
            if track_settings is None:
                mult = jnp.ones_like(pp.array)
            else:
                w_mb = effective_loss_weight(ex_mb)
                mult = token_multiplier(
                    pp.array, w_mb.array, *(h.array for h in hist_mb), **track_kwargs(track_settings)
                )
            return jnp.sum(mult * pp.array), (jax.lax.stop_gradient(pp), jax.lax.stop_gradient(aux_mb))

        (_, (pp, aux_mb)), g = eqx.filter_value_and_grad(num_fn, has_aux=True)(model)
        acc_num = jax.tree_util.tree_map(lambda a, b: a + b.astype(a.dtype), acc_num, g)
        if has_aux:
            g_aux = eqx.filter_grad(lambda m: per_position_loss(m, ex_mb, mp=mp, z_loss_weight=z_loss_weight, compute_mapping=compute_mapping)[1])(model)
            acc_aux = jax.tree_util.tree_map(lambda a, b: a + b.astype(a.dtype) / num_microbatches, acc_aux, g_aux)
        if accum_mapping is not None:
            acc_num = hax.shard(acc_num, accum_mapping)
            acc_aux = hax.shard(acc_aux, accum_mapping)
        return (acc_num, acc_aux, aux_sum + aux_mb / num_microbatches), pp

    diff_model = eqx.filter(model, eqx.is_inexact_array)
    acc0 = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, jnp.float32), diff_model)
    if accum_mapping is not None:
        acc0 = hax.shard(acc0, accum_mapping)
    (grads_sum, grads_aux, aux), pp_split = hax.scan(micro_step, Accum)(
        (acc0, acc0, jnp.zeros((), jnp.float32)), (ex_split, hist_split)
    )
    pp = pp_split.flatten_axes((Accum, Micro), Batch)
    w = weight_named.array
    hist_raw = None if hist is None else tuple(raw(h) for h in hist)
    loss, metrics = objective(pp.array, w, hist_raw, track_settings, aux=aux)
    if track_settings is None:
        den = jnp.sum(w.astype(jnp.float32))
    else:
        den = jnp.sum(token_multiplier(pp.array, w, *hist_raw, **track_kwargs(track_settings)) * w.astype(jnp.float32))
    scale = jnp.where(den > 0, 1.0 / jnp.where(den > 0, den, 1.0), 0.0)
    grads = jax.tree_util.tree_map(lambda g, ga, p: (g * scale + ga).astype(p.dtype), grads_sum, grads_aux, diff_model)
    return loss, metrics, pp, grads


def make_train_step(*, mp, z_loss_weight, track_settings, compute_mapping, param_mapping, num_microbatches=1):
    """Jitted ``(state, example, hist) -> (state, metrics, per_pos)`` with Levanter's FSDP sharding."""
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    from haliax.partitioning import named_jit  # noqa: PLC0415

    out_mapping = dict(param_mapping)
    out_mapping["batch"] = compute_mapping["batch"]

    def _step(state, example, hist):
        key, new_key = jax.random.split(state.training_key)
        del key
        loss, metrics, pp, grads = loss_and_grads(
            state.model,
            example,
            hist,
            mp=mp,
            z_loss_weight=z_loss_weight,
            track_settings=track_settings,
            compute_mapping=compute_mapping,
            num_microbatches=num_microbatches,
            accum_mapping=param_mapping,
        )
        new_state, _ = state.take_step(grads, obj_fun=None, loss=loss, key=new_key)
        new_state = hax.shard(new_state, param_mapping)
        metrics = {"train/loss": loss, **metrics}
        return new_state, metrics, pp

    return named_jit(
        _step, axis_resources=param_mapping, out_axis_resources=out_mapping, donate_args=(True, False, False)
    )


# --------------------------------------------------------------------------------------------------------------------
# Model, mesh, data
# --------------------------------------------------------------------------------------------------------------------


def trainer_mesh_config():
    """FSDP over every device: parameters' ``embed`` axis over (replica_dcn, data), the batch over all data axes.

    ``replica_dcn`` is 1 unless JAX reports several slices (``device.slice_index``); either way the product of the two
    axes is every device, so parameters and optimizer state are sharded N ways whether the job runs one process per
    node with 4 GPUs or one process per GPU. ``model`` stays 1: no tensor parallelism.
    """
    from levanter.utils.mesh import MeshConfig  # noqa: PLC0415

    return MeshConfig(
        axes={"data": -1, "replica": 1, "model": 1},
        dcn_axes={"replica_dcn": -1},
        compute_mapping={"batch": ["replica_dcn", "replica", "data"]},
        param_mapping={"embed": ["replica_dcn", "data"]},
    )


def levanter_config_class(model_type: str):
    """Levanter config class for an HF ``model_type``: dense Qwen3, or Qwen3-MoE (e.g. Qwen3-30B-A3B, dropless routing)."""
    from levanter.models.qwen import Qwen3Config  # noqa: PLC0415
    from levanter.models.qwen3_moe import Qwen3MoeConfig  # noqa: PLC0415

    classes = {"qwen3": Qwen3Config, "qwen3_moe": Qwen3MoeConfig}
    if model_type not in classes:
        raise ValueError(f"unsupported model_type {model_type!r}; expected one of {sorted(classes)}")
    return classes[model_type]


def hf_model_type(hf_checkpoint: str) -> str:
    return json.loads((Path(hf_checkpoint) / "config.json").read_text())["model_type"]


def qwen_model_config(hf_config, *, attn_backend: str, flash_block_size: int | None, gradient_checkpointing: bool):
    """Levanter ``Qwen3Config`` / ``Qwen3MoeConfig`` from the HF config (``use_hf_model_config`` semantics) plus
    runtime knobs. For MoE the router load-balancing loss (HF ``router_aux_loss_coef``, 0.001) is added to every
    position's loss by Levanter, so it is part of the objective and of the tracked per-token loss alike."""
    from levanter.layers.attention import AttentionBackend  # noqa: PLC0415

    cfg = levanter_config_class(hf_config.model_type).from_hf_config(hf_config)
    if hf_config.model_type == "qwen3_moe":
        cfg = dataclasses.replace(cfg, use_gmm=True)   # GPU: the XLA ragged_dot_general lowering OOMs at 64k x top-8
    return dataclasses.replace(
        cfg,
        attn_backend=AttentionBackend(attn_backend),
        flash_attention_block_size=flash_block_size,
        gradient_checkpointing=gradient_checkpointing,
    )


def make_converter(hf_checkpoint: str, tokenizer_path: str, *, pad_to_model: bool = True):
    """The HF converter for ``hf_checkpoint`` with the run's tokenizer, padded to the model vocab (151,936)."""
    from levanter.tokenizers import load_tokenizer  # noqa: PLC0415

    converter = levanter_config_class(hf_model_type(hf_checkpoint))().hf_checkpoint_converter(ref_checkpoint=hf_checkpoint)
    converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=load_tokenizer(tokenizer_path))
    if pad_to_model:
        converter = converter.with_tokenizer_padded_to_match_model()
    return converter


def stock_chat_template(tokenizer_path: str) -> str:
    """The tokenizer's own (stock) chat template; refuses a template carrying ``{% generation %}`` markers."""
    from experiments.plateau_dropping_qwen.qwen3_template import has_generation_tags  # noqa: PLC0415

    path = Path(tokenizer_path)
    template = None
    if (path / "chat_template.jinja").is_file():
        template = (path / "chat_template.jinja").read_text(encoding="utf-8")
    else:
        template = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError(f"{tokenizer_path} carries no chat template to export")
    if has_generation_tags(template):
        raise ValueError(
            f"{tokenizer_path}'s chat template carries {{% generation %}} markers; export needs the stock one"
        )
    return template


def grug_data_key(seed: int, data_seed: int | None):
    """``train.run_grug_local``'s data key: ``split(PRNGKey(trainer.seed), 2)[0]``, or ``PRNGKey(data_seed)``."""
    import jax  # noqa: PLC0415

    data_key, _model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    if data_seed is not None:
        data_key = jax.random.PRNGKey(data_seed)
    return data_key


def build_grug_dataset(
    *,
    data_cache_path: str,
    tokenizer_path: str,
    stage: str,
    seq_len: int,
    batch_schedule,
    seed: int,
    data_seed: int | None,
):
    """``train.build_train_dataset`` over ``snowball_chat_data_config`` with Grug's key: the Grug data path itself."""
    from experiments.june_tpu_67b_a2b.moe.train import build_train_dataset  # noqa: PLC0415
    from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import snowball_chat_data_config  # noqa: PLC0415

    data_config = snowball_chat_data_config(cache_path=data_cache_path, tokenizer_path=tokenizer_path, stage=stage)
    return build_train_dataset(
        data_config, max_seq_len=seq_len, batch_schedule=batch_schedule, key=grug_data_key(seed, data_seed)
    )


def named_dataset(grug_dataset, Pos):
    """Per-example ``GrugLmExample -> LmExample`` (``named_lm_example_from_grug``); row order, tokens, loss weights and
    segment ids are untouched."""
    from levanter.data.text.datasets import NamedLmDataset  # noqa: PLC0415

    return NamedLmDataset(grug_dataset, Pos)


def make_train_loader(dataset, *, trainer_config, mesh):
    """The loader ``Trainer.data_loader`` builds: batch ``i`` is rows ``[i * B, (i + 1) * B)`` of ``dataset``, sharded
    over the batch axes (the same rows per step as Grug's ``build_train_loader``)."""
    from levanter.data.loader import DataLoader  # noqa: PLC0415

    return DataLoader(
        dataset,
        batch_size=trainer_config.train_batch_size,
        max_buffered_batches=128,
        mesh=mesh,
        axis_resources=trainer_config.compute_axis_mapping,
        prefetch_size=32,
        batch_axis_name=trainer_config.batch_axis_name,
        allow_nondivisible_batch_size=False,
    )


def tree_bytes_report(tree) -> dict[str, float]:
    """Total bytes of the arrays in ``tree``, the bytes resident on this process's first device, and replicated bytes."""
    import jax  # noqa: PLC0415

    leaves = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
    dev = jax.local_devices()[0]
    total = sum(int(x.size) * x.dtype.itemsize for x in leaves)
    on_dev = sum(int(s.data.nbytes) for x in leaves for s in x.addressable_shards if s.device == dev)
    replicated = sum(int(x.size) * x.dtype.itemsize for x in leaves if x.sharding.is_fully_replicated)
    return {
        "total_bytes": total,
        "device0_bytes": on_dev,
        "replicated_bytes": replicated,
        "ideal_bytes_per_device": total / max(1, jax.device_count()),
    }


def estimate_sharded_bytes(shape_tree, mapping, mesh_shape: Mapping[str, int]) -> dict[str, float]:
    """Per-device bytes of a NamedArray shape tree under ``mapping`` on a mesh of ``mesh_shape`` (no devices needed)."""
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415

    total = per_device = 0.0
    replicated = 0.0
    for leaf in jax.tree_util.tree_leaves(shape_tree, is_leaf=lambda x: isinstance(x, hax.NamedArray)):
        if not isinstance(leaf, hax.NamedArray):
            continue
        nbytes = math.prod(ax.size for ax in leaf.axes) * leaf.array.dtype.itemsize
        div = 1
        for ax in leaf.axes:
            phys = mapping.get(ax.name)
            if phys is None:
                continue
            names = (phys,) if isinstance(phys, str) else tuple(phys)
            div *= math.prod(mesh_shape.get(n, 1) for n in names)
        total += nbytes
        per_device += nbytes / div
        if div == 1:
            replicated += nbytes
    return {"total_bytes": total, "per_device_bytes": per_device, "replicated_bytes": replicated}


def describe_mesh(mesh) -> str:
    procs = np.vectorize(lambda d: d.process_index)(mesh.devices)
    return (
        f"mesh axes={dict(mesh.shape)} devices={mesh.devices.size} processes={len(np.unique(procs))} "
        f"process grid={procs.reshape(-1).tolist()}"
    )


# --------------------------------------------------------------------------------------------------------------------
# HF export
# --------------------------------------------------------------------------------------------------------------------


def export_hf(model, *, converter, path: str, dtype, chat_template: str, generation_config_src: str | None = None):
    """Write ``model`` as an HF checkpoint carrying ``chat_template`` (the stock one).

    Levanter's ``save_pretrained``: call it on every process inside the trainer's mesh context; it gathers each
    parameter with a collective and only process 0 writes the safetensors shards.
    """
    import jax  # noqa: PLC0415
    from jax.experimental import multihost_utils  # noqa: PLC0415

    converter.save_pretrained(model, path, save_tokenizer=True, chat_template=chat_template, dtype=dtype)
    if jax.process_index() == 0 and generation_config_src is not None and os.path.isfile(generation_config_src):
        shutil.copyfile(generation_config_src, os.path.join(path, "generation_config.json"))
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("qwen_sft_hf_export")
    logger.info(f"HF export written to {path}")


# --------------------------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------------------------


def build_trainer_config(cfg: QwenSftConfig):
    import jmp  # noqa: PLC0415
    from levanter.checkpoint import CheckpointerConfig  # noqa: PLC0415
    from levanter.tracker import NoopConfig  # noqa: PLC0415
    from levanter.tracker.json_logger import JsonLoggerConfig  # noqa: PLC0415
    from levanter.tracker.wandb import WandbConfig  # noqa: PLC0415
    from levanter.trainer import TrainerConfig  # noqa: PLC0415

    if cfg.tracker == "wandb":
        tracker = WandbConfig(
            project=cfg.wandb_project, name=cfg.run_id, tags=["qwen3", "sft", cfg.stage], mode=cfg.wandb_mode
        )
    elif cfg.tracker == "json":
        tracker = JsonLoggerConfig()
    else:
        tracker = NoopConfig()
    return TrainerConfig(
        seed=cfg.seed,
        mp=jmp.get_policy(cfg.mp),
        id=cfg.run_id,
        log_dir=Path(cfg.log_dir),
        tracker=tracker,
        train_batch_size=cfg.batch_size,
        per_device_parallelism=cfg.per_device_parallelism,
        num_train_steps=cfg.steps,
        mesh=trainer_mesh_config(),
        require_accelerator=not cfg.allow_cpu,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(cfg.output_path, "checkpoints"),
            temporary_base_path=os.path.join(cfg.output_path, "checkpoints-tmp"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=cfg.save_interval_minutes),
            keep=[{"every": int(cfg.keep_every)}],
            timeout=timedelta(hours=2),
        ),
        load_checkpoint=None,
    )


def build_optimizer_config(cfg: QwenSftConfig):
    from levanter.optim.config import AdamConfig  # noqa: PLC0415

    return AdamConfig(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup=levanter_warmup(cfg.warmup, schedule_horizon(cfg)),
        min_lr_ratio=cfg.min_lr_ratio,
        lr_schedule=cfg.lr_schedule,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        epsilon=cfg.epsilon,
        max_grad_norm=cfg.max_grad_norm,
    )


def warmup_steps_for(warmup: float, schedule_steps: int) -> int:
    """Linear-warmup length: ``warmup`` < 1 is a fraction of the schedule rounded down but at least one step (when
    positive); ``warmup`` >= 1 is a whole number of steps."""
    if warmup <= 0:
        return 0
    if warmup >= 1:
        if warmup % 1:
            raise ValueError(f"warmup {warmup} >= 1 must be a whole number of steps")
        return min(int(warmup), schedule_steps)
    return min(schedule_steps, max(1, int(warmup * schedule_steps)))


def levanter_warmup(warmup: float, schedule_steps: int) -> int | float:
    """``warmup_steps_for`` in the form ``OptimizerConfig.warmup`` reads (values <= 1 are fractions there, so a single
    warmup step is passed as a fraction that floors to one step)."""
    n = warmup_steps_for(warmup, schedule_steps)
    if n == 0:
        return 0.0
    if n >= 2:
        return n
    return 1.0 if schedule_steps == 1 else 1.5 / schedule_steps


def schedule_horizon(cfg: QwenSftConfig) -> int:
    schedule_steps = cfg.schedule_steps or cfg.steps
    if schedule_steps < cfg.steps:
        raise ValueError(f"schedule_steps {schedule_steps} is shorter than num_train_steps {cfg.steps}")
    return schedule_steps


def run_training(cfg: QwenSftConfig) -> dict[str, Any]:
    """Train; returns a small summary (final step, last loss, export path)."""
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import levanter.tracker  # noqa: PLC0415
    from jax.experimental import multihost_utils  # noqa: PLC0415
    from levanter import callbacks  # noqa: PLC0415
    from levanter.callbacks import StepInfo  # noqa: PLC0415
    from levanter.trainer import Trainer  # noqa: PLC0415
    from levanter.utils.jax_utils import parameter_count  # noqa: PLC0415

    from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (  # noqa: PLC0415
        STAGES,
        read_chat_cache_examples,
    )

    if cfg.stage not in STAGES:
        raise ValueError(f"Unknown stage {cfg.stage!r}")
    trainer_config = build_trainer_config(cfg)
    trainer_config.initialize()
    levanter.tracker.log_configuration(dataclasses.asdict(cfg))
    run_id = trainer_config.id
    schedule_steps = schedule_horizon(cfg)
    print(f"lr_schedule_steps={schedule_steps} num_train_steps={cfg.steps}", flush=True)
    opt_config = build_optimizer_config(cfg)
    optimizer = opt_config.build(schedule_steps)
    lr_fn = opt_config.lr_scheduler(schedule_steps)

    num_docs = read_chat_cache_examples(cfg.data_cache_path)
    track = resolve_tracking(cfg, num_docs, process_index=jax.process_index())
    if track is not None and track.trace is not None and os.path.isdir(cfg.trace_out):
        # A resumed run numbers its trace shards after the ones already written instead of overwriting trace-0000.
        track.trace.shards = len([f for f in os.listdir(cfg.trace_out) if f.startswith("trace-") and f[6:10].isdigit()])

    converter = make_converter(cfg.hf_checkpoint, cfg.tokenizer_path)
    model_config = qwen_model_config(
        converter.default_hf_config,
        attn_backend=cfg.attn_backend,
        flash_block_size=cfg.flash_block_size,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    if cfg.seq_len > model_config.max_seq_len:
        raise ValueError(f"seq_len {cfg.seq_len} exceeds the model's max_seq_len {model_config.max_seq_len}")
    if track is not None and cfg.track_max_len < cfg.seq_len:
        logger.warning(
            "track_max_len %d < seq_len %d: a document longer than the table width raises",
            cfg.track_max_len,
            cfg.seq_len,
        )
    export_converter = converter.replaced(tokenizer=_load_unpadded_tokenizer(cfg.tokenizer_path))
    stock_template = stock_chat_template(cfg.tokenizer_path)

    with Trainer(trainer_config, optimizer, _unused_loss, add_default_hooks=False) as trainer:
        mesh = trainer.device_mesh
        param_mapping = trainer.parameter_axis_mapping
        compute_mapping = trainer.compute_axis_mapping
        logger.info(describe_mesh(mesh))
        logger.info(
            f"param mapping={param_mapping} compute mapping={compute_mapping} "
            f"process {jax.process_index()}/{jax.process_count()} local devices={jax.local_device_count()}"
        )
        print(describe_mesh(mesh), flush=True)

        Pos = model_config.max_Pos.resize(cfg.seq_len)
        Vocab = hax.Axis("vocab", converter.default_hf_config.vocab_size)
        num_micro = 1
        if trainer_config.microbatch_size is not None and trainer_config.microbatch_size < cfg.batch_size:
            num_micro = cfg.batch_size // trainer_config.microbatch_size
        logger.info(f"global batch {cfg.batch_size} x {cfg.seq_len}, {num_micro} microbatch(es) per step")

        training_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 1)
        model_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 2)
        state = trainer.initial_state(training_key, model_init=lambda: model_config.build(Vocab, key=model_key))
        if int(state.step) == 0:
            logger.info(f"No training checkpoint found. Initializing model from HF checkpoint {cfg.hf_checkpoint}")
            state = dataclasses.replace(state, model=None)
            model = converter.load_pretrained(
                model_config.model_type,
                ref=cfg.hf_checkpoint,
                config=model_config,
                axis_mapping=param_mapping,
                dtype=trainer.mp.compute_dtype,
            )
            model = hax.named_jit(trainer.mp.cast_to_param, param_mapping)(model)
            state = dataclasses.replace(state, model=model)
        else:
            logger.info(f"Resuming training from step {int(state.step)}")

        model_report = tree_bytes_report(state.model)
        opt_report = tree_bytes_report(state.opt_state)
        msg = (
            f"params: {model_report['total_bytes'] / 2**30:.2f} GiB total, "
            f"{model_report['device0_bytes'] / 2**30:.3f} GiB on device 0 (ideal "
            f"{model_report['ideal_bytes_per_device'] / 2**30:.3f}), "
            f"replicated {model_report['replicated_bytes'] / 2**20:.1f} MiB; "
            f"optimizer: {opt_report['total_bytes'] / 2**30:.2f} GiB total, "
            f"{opt_report['device0_bytes'] / 2**30:.3f} GiB on device 0"
        )
        logger.info(msg)
        print(msg, flush=True)
        if jax.device_count() > 1 and model_report["device0_bytes"] > 0.5 * model_report["total_bytes"]:
            raise RuntimeError(f"parameters are not sharded across the {jax.device_count()} devices: {model_report}")
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        checkpointer = trainer_config.checkpointer.create(run_id)
        history_path = history_path_for(checkpointer.base_path)
        history_restored = restore_history(track, history_path, int(state.step))
        # The restored step is the last saved one (Levanter's Checkpointer starts at 0; without this the first loop
        # iteration after a resume re-saved the token history tagged step 0).
        checkpointer._last_save_step = int(state.step)
        saver = HistorySaver(track, history_path, int(state.step), is_writer=jax.process_index() == 0)

        grug_ds = build_grug_dataset(
            data_cache_path=cfg.data_cache_path,
            tokenizer_path=cfg.tokenizer_path,
            stage=cfg.stage,
            seq_len=cfg.seq_len,
            batch_schedule=trainer_config.batch_schedule,
            seed=cfg.seed,
            data_seed=cfg.data_seed,
        )
        loader = make_train_loader(named_dataset(grug_ds, Pos), trainer_config=trainer_config, mesh=mesh)
        iterator = iter(loader.iter_from_step(int(state.step)))

        train_step = make_train_step(
            mp=trainer.mp,
            z_loss_weight=cfg.z_loss_weight,
            track_settings=None if track is None else track.settings,
            compute_mapping=compute_mapping,
            param_mapping=param_mapping,
            num_microbatches=num_micro,
        )

        flops_per_token = model_config.flops_per_token(Vocab.size, cfg.seq_len)
        flops_per_example = 3 * flops_per_token * cfg.seq_len if flops_per_token is not None else None
        hooks = [
            callbacks.log_performance_stats(cfg.seq_len, trainer_config.batch_schedule, flops_per_example),
            callbacks.pbar_logger(total=cfg.steps),
            callbacks.log_step_info(cfg.steps, trainer_config.batch_schedule),
        ]
        log_every = max(1, cfg.log_every)
        last_loss, last_duration, nan_stop = 0.0, 0.0, False
        info = None
        try:
            while int(state.step) < cfg.steps:
                load_start = time.perf_counter()
                batch = next(iterator)
                load_time = time.perf_counter() - load_start
                step_start = time.perf_counter()
                current_step = int(state.step)
                if track is not None:
                    seg_np = multihost_utils.process_allgather(batch.attn_mask.segment_ids[0].array, tiled=True)
                    w_np = multihost_utils.process_allgather(batch.loss_weight.array, tiled=True)
                    if np.any(w_np[:, -1] != 0):
                        raise ValueError(
                            "a batch carries loss weight on its final position; the tracked weight would "
                            "differ from the weight the next-token loss applies"
                        )
                    keys, p1, p2, cnt, prev_dropped = track_step_inputs(track.history, seg_np, w_np)
                    shard = batch.loss_weight.array.sharding
                    axes = batch.loss_weight.axes

                    def _as_global(a, shard=shard, axes=axes):
                        return hax.named(jax.make_array_from_callback(a.shape, shard, lambda idx, a=a: a[idx]), axes)

                    hist = tuple(_as_global(a) for a in (p1, p2, cnt, prev_dropped))
                    state, metrics, per_pos = train_step(state, batch, hist)
                    pp_np = multihost_utils.process_allgather(per_pos.array, tiled=True)
                    tok_np = None
                    if track.trace is not None:
                        # process_allgather is a collective: every process gathers, only process 0 records and writes.
                        tok_np = multihost_utils.process_allgather(batch.tokens.array, tiled=True)
                    track_step_record(
                        track,
                        step=current_step,
                        keys=keys,
                        p1=p1,
                        p2=p2,
                        cnt=cnt,
                        prev_dropped=prev_dropped,
                        w_np=w_np,
                        pp_np=pp_np,
                        tok_np=tok_np,
                        record_trace=jax.process_index() == 0,
                    )
                else:
                    state, metrics, _ = train_step(state, batch, None)
                step = int(state.step) - 1
                jax.block_until_ready(metrics["train/loss"])
                loss_value = float(metrics["train/loss"])
                if math.isnan(loss_value):
                    logger.error(f"NaN loss at step {int(state.step)}. Stopping training.")
                    nan_stop = True
                    break
                duration = time.perf_counter() - step_start
                info = StepInfo(state, loss_value, duration)
                last_loss, last_duration = loss_value, duration
                if step % log_every == 0:
                    for hook in hooks:
                        hook(info)
                    to_log = {k: v for k, v in metrics.items() if k != "train/loss"}
                    to_log["optim/lr_schedule"] = float(lr_fn(step))
                    to_log["throughput/loading_time"] = load_time
                    levanter.tracker.log(to_log, step=step)
                    if track is not None:
                        logger.info(
                            "track step %d: kept tokens %.3f, dropped %.3f (learned %.3f, unlearnable %.3f), "
                            "eligible %.3f, loss kept %.4f dropped %.4f unfiltered %.4f",
                            step,
                            float(metrics["train/track/kept_token_frac"]),
                            float(metrics["train/track/dropped_frac"]),
                            float(metrics["train/track/dropped_low_frac"]),
                            float(metrics["train/track/dropped_high_frac"]),
                            float(metrics["train/track/eligible_frac"]),
                            loss_value,
                            float(metrics["train/track/loss_dropped"]),
                            float(metrics["train/track/loss_unfiltered"]),
                        )
                checkpointer.on_step(tree=state.saveable_state, step=int(state.step))
                saver.after_checkpoint(checkpointer._last_save_step)
        except BaseException:
            logger.exception(
                "Fatal error in the Qwen SFT loop; skipping the final checkpoint to preserve the root cause"
            )
            raise
        else:
            if info is not None:
                for hook in hooks:
                    hook(StepInfo(state, last_loss, last_duration))
            # Force a save only when the last step has not been saved yet (train.py, job 1890618).
            if checkpointer._last_save_step != int(state.step):
                checkpointer.on_step(tree=state.saveable_state, step=int(state.step), force=True)
            saver.final(checkpointer._last_save_step)
            checkpointer.wait_until_finished()
            if track is not None and track.trace is not None and jax.process_index() == 0:
                track.trace.flush()

        export_path = None
        if not nan_stop and int(state.step) >= cfg.steps:
            export_path = cfg.hf_save_path or os.path.join(cfg.output_path, "hf")
            hf_ckpt_gen = os.path.join(cfg.hf_checkpoint, "generation_config.json")
            export_hf(
                state.model,
                converter=export_converter,
                path=export_path,
                dtype=_jnp_dtype(cfg.hf_save_dtype),
                chat_template=stock_template,
                generation_config_src=hf_ckpt_gen if os.path.isfile(hf_ckpt_gen) else None,
            )
    levanter.tracker.current_tracker().finish()
    return {
        "step": int(state.step),
        "loss": last_loss,
        "export_path": export_path,
        "nan_stop": nan_stop,
        "history_restored": history_restored,
    }


def _unused_loss(*args, **kwargs):  # Trainer needs a loss function; this loop calls its own step.
    raise RuntimeError("qwen_sft drives its own train step")


def _load_unpadded_tokenizer(tokenizer_path: str):
    from levanter.tokenizers import load_tokenizer  # noqa: PLC0415

    return load_tokenizer(tokenizer_path)


def _jnp_dtype(name: str | None):
    import jax.numpy as jnp  # noqa: PLC0415

    if not name or name == "none":
        return None
    return jnp.dtype(name)


# --------------------------------------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------------------------------------


def run_preflight(cfg: QwenSftConfig, *, expect_devices: int | None = None, check_output: bool = True) -> dict[str, Any]:
    """Validate the cache, tokenizer, HF checkpoint, mesh and tracking settings without loading weights or training."""
    import haliax as hax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (  # noqa: PLC0415
        STAGES,
        read_chat_cache_tokens,
        validate_cache_provenance,
        validate_chat_cache_layout,
    )
    from experiments.plateau_dropping_qwen.qwen3_template import (  # noqa: PLC0415
        has_generation_tags,
        strip_generation_tags,
    )

    report: dict[str, Any] = {}
    if cfg.stage not in STAGES:
        raise ValueError(f"Unknown stage {cfg.stage!r}")
    spec = STAGES[cfg.stage]
    if not cfg.run_id or "/" in cfg.run_id:
        raise ValueError(f"run id must be a non-empty path-free name, got {cfg.run_id!r}")
    # Cache.
    tokens, examples, shards = validate_chat_cache_layout(
        os.path.abspath(cfg.data_cache_path), expect_tokens=spec.cache_tokens, expect_examples=spec.cache_examples
    )
    if shards != spec.cache_shards:
        raise ValueError(f"{cfg.stage} cache has {shards} finished shards; expected {spec.cache_shards}")
    validate_cache_provenance(cfg.data_cache_path, cfg.stage, cfg.tokenizer_path)
    report.update(cache_tokens=tokens, cache_examples=examples, cache_shards=shards)
    epoch_steps = expected_epoch_steps(read_chat_cache_tokens(cfg.data_cache_path), cfg.seq_len, cfg.batch_size)
    report["epoch_steps"] = epoch_steps
    ceiling = spec.max_steps or epoch_steps
    if cfg.steps > ceiling:
        raise ValueError(f"Requested {cfg.steps} steps, but the {cfg.stage} ceiling is {ceiling}")
    # Tokenizer and templates.
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, local_files_only=True)
    if not has_generation_tags(spec.chat_template):
        raise ValueError(f"stage {cfg.stage} chat template has no {{% generation %}} markers")
    stock = stock_chat_template(cfg.tokenizer_path)
    probe = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    if tok.apply_chat_template(
        probe, tokenize=False, chat_template=strip_generation_tags(spec.chat_template)
    ) != tok.apply_chat_template(probe, tokenize=False, chat_template=stock):
        raise ValueError(f"stage {cfg.stage} template does not render like the tokenizer's stock template")
    report["tokenizer_len"] = len(tok)
    # HF checkpoint.
    hf_dir = Path(cfg.hf_checkpoint)
    hf_cfg = json.loads((hf_dir / "config.json").read_text())
    if hf_cfg.get("model_type") not in ("qwen3", "qwen3_moe"):
        raise ValueError(f"{hf_dir} is a {hf_cfg.get('model_type')!r} checkpoint, expected qwen3 or qwen3_moe")
    weights = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"{hf_dir} has no weight files")
    if len(tok) > hf_cfg["vocab_size"]:
        raise ValueError(f"tokenizer has {len(tok)} tokens but the model vocab is {hf_cfg['vocab_size']}")
    report["hf_vocab_size"] = hf_cfg["vocab_size"]
    if cfg.seq_len > hf_cfg.get("max_position_embeddings", cfg.seq_len):
        raise ValueError(f"seq_len {cfg.seq_len} > max_position_embeddings {hf_cfg['max_position_embeddings']}")
    # Tracking settings (validation only; the table is allocated by the run).
    num_docs = examples
    if cfg.track_tol > 0 or cfg.track_observe_only:
        _validate_tracking_only(cfg, num_docs)
        report["history_bytes_per_process"] = num_docs * cfg.track_max_len * (4 + 4 + 2 + 1)
    # Devices and mesh.
    n = jax.device_count()
    report.update(
        devices=n, processes=jax.process_count(), local_devices=jax.local_device_count(), backend=jax.default_backend()
    )
    if expect_devices is not None and n != expect_devices:
        raise RuntimeError(f"expected {expect_devices} devices, found {n}")
    if cfg.batch_size % n != 0:
        raise ValueError(f"batch {cfg.batch_size} is not divisible by {n} devices")
    if hf_cfg["hidden_size"] % n != 0:
        raise ValueError(f"hidden size {hf_cfg['hidden_size']} (the FSDP axis) is not divisible by {n} devices")
    if cfg.per_device_parallelism > 0 and cfg.batch_size % (cfg.per_device_parallelism * n) != 0:
        raise ValueError("batch_size must be divisible by per_device_parallelism * devices")
    from transformers import AutoConfig  # noqa: PLC0415

    model_config = qwen_model_config(
        AutoConfig.from_pretrained(cfg.hf_checkpoint, local_files_only=True),
        attn_backend=cfg.attn_backend,
        flash_block_size=cfg.flash_block_size,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    shape = jax.eval_shape(
        lambda: model_config.build(hax.Axis("vocab", hf_cfg["vocab_size"]), key=jax.random.PRNGKey(0))
    )
    mesh_cfg = trainer_mesh_config()
    num_slices = max(getattr(d, "slice_index", 0) for d in jax.devices()) + 1
    ici, dcn = mesh_cfg.axis_shapes(n, num_slices)
    mesh_shape = {k: ici.get(k, 1) * dcn.get(k, 1) for k in set(ici) | set(dcn)}
    est = estimate_sharded_bytes(shape, mesh_cfg.resolved_param_mapping, mesh_shape)
    report.update(
        mesh=mesh_shape,
        param_bytes_fp32_total=est["total_bytes"],
        param_bytes_fp32_per_device=est["per_device_bytes"],
        replicated_param_bytes=est["replicated_bytes"],
        adam_state_bytes_per_device=2 * est["per_device_bytes"],
    )
    if n > 1 and est["per_device_bytes"] > 0.5 * est["total_bytes"]:
        raise RuntimeError(f"parameters would not be sharded: {est}")
    # Output.
    if check_output:
        out = Path(cfg.output_path)
        parent = out.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise PermissionError(f"output parent is not writable: {parent}")
        report["resume"] = (out / "checkpoints").exists() or (out / "checkpoints-tmp").exists()
    return report


def _validate_tracking_only(cfg: QwenSftConfig, num_docs: int) -> None:
    from experiments.june_tpu_67b_a2b.moe import token_track  # noqa: PLC0415

    if cfg.trace_num_docs > 0 and not (cfg.track_tol > 0 or cfg.track_observe_only):
        raise ValueError("the token trace needs tracking on (track_tol > 0 or track_observe_only)")
    if cfg.track_observe_only and cfg.track_tol > 0:
        raise ValueError("track_observe_only records without dropping; it cannot be combined with track_tol > 0")
    if not 0.0 <= cfg.track_drop_weight < 1.0:
        raise ValueError(f"track_drop_weight must be in [0, 1), got {cfg.track_drop_weight}")
    reentry = token_track.Reentry(cfg.track_reenter)
    if reentry is not token_track.Reentry.EITHER and (cfg.track_ema_alpha > 0 or cfg.trace_num_docs > 0):
        raise ValueError("track_reenter other than 'either' supports neither track_ema_alpha nor the token trace")
    if cfg.trace_num_docs > 0 and not cfg.trace_out:
        raise ValueError("trace_num_docs > 0 needs trace_out")
    if cfg.trace_num_docs > 0 and cfg.track_ema_alpha > 0:
        raise ValueError("the token trace records the hard plateau rule; it does not support track_ema_alpha")
    del num_docs


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------


def _common_options(fn):
    opts = [
        click.option(
            "--hf-checkpoint", required=True, help="Local HF checkpoint directory (config.json + safetensors)."
        ),
        click.option(
            "--data-cache-path", required=True, help="Levanter cache built by vista_snowball_chat prepare-data."
        ),
        click.option("--tokenizer-path", required=True, help="Tokenizer directory (stock chat template)."),
        click.option("--output-path", required=True, help="Run output: checkpoints/, checkpoints-tmp/, hf/."),
        click.option("--run-id", required=True),
        click.option("--steps", type=int, default=None, help="Training steps (default: SNOWBALL_STEPS)."),
        click.option("--stage", default=DEFAULT_STAGE, show_default=True),
        click.option("--seq-len", type=int, default=None, help="Default: SNOWBALL_SEQ_LEN or 32768."),
        click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True),
        click.option(
            "--per-device-parallelism",
            type=int,
            default=-1,
            show_default=True,
            help="Examples per device per microbatch; -1 = no gradient accumulation.",
        ),
        click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True),
        click.option("--data-seed", type=int, default=None, help="Default: SNOWBALL_DATA_SEED."),
        click.option("--z-loss-weight", type=float, default=DEFAULT_Z_LOSS, show_default=True),
        click.option("--lr", "learning_rate", type=float, default=None, help="Peak LR. Default: SNOWBALL_LR or 5e-5."),
        click.option("--weight-decay", type=float, default=0.01, show_default=True),
        click.option(
            "--warmup",
            type=float,
            default=0.03,
            show_default=True,
            help="<1: fraction of the schedule (at least 1 step); >=1: steps.",
        ),
        click.option("--min-lr-ratio", type=float, default=0.1, show_default=True),
        click.option("--lr-schedule", default="cosine", show_default=True),
        click.option("--beta1", type=float, default=0.9, show_default=True),
        click.option("--beta2", type=float, default=0.95, show_default=True),
        click.option("--max-grad-norm", type=float, default=1.0, show_default=True),
        click.option("--schedule-steps", type=int, default=None, help="Default: SNOWBALL_SCHEDULE_STEPS or --steps."),
        click.option(
            "--keep-every", type=int, default=None, help="Default: SNOWBALL_KEEP_EVERY / KEEP_PER_EPOCH / 1000."
        ),
        click.option("--save-interval-minutes", type=float, default=30.0, show_default=True),
        click.option("--hf-save-path", default=None, help="Default: <output-path>/hf."),
        click.option("--hf-save-dtype", default="bfloat16", show_default=True),
        click.option("--track-tol", type=float, default=None),
        click.option("--track-floor", type=float, default=None),
        click.option("--track-max-len", type=int, default=None),
        click.option("--track-ema-alpha", type=float, default=None),
        click.option("--track-reenter", type=click.Choice(["either", "rise"]), default=None),
        click.option("--track-drop-weight", type=float, default=None),
        click.option("--track-observe-only/--no-track-observe-only", default=None),
        click.option("--trace-num-docs", type=int, default=None),
        click.option("--trace-seed", type=int, default=None),
        click.option("--trace-out", default=None),
        click.option(
            "--attn-backend",
            default="jax_flash",
            show_default=True,
            type=click.Choice(["jax_flash", "nvte", "vanilla", "default", "splash"]),
        ),
        click.option("--flash-block-size", type=int, default=None),
        click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True, show_default=True),
        click.option("--tracker", type=click.Choice(["wandb", "json", "noop"]), default="wandb", show_default=True),
        click.option("--wandb-project", default="plateau_dropping_qwen", show_default=True),
        click.option("--wandb-mode", default=None, help="Default: WANDB_MODE or offline."),
        click.option("--log-dir", default=None, help="Default: SNOWBALL_LOG_DIR or logs/."),
        click.option("--allow-cpu", is_flag=True, default=False, hidden=True),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def config_from_cli(kwargs: dict[str, Any], environ: Mapping[str, str] | None = None) -> QwenSftConfig:
    """CLI flags over ``SNOWBALL_*`` environment settings over the Grug defaults."""
    env = snowball_env_settings(environ)
    environ = os.environ if environ is None else environ
    k = dict(kwargs)

    def pick(name, default=None):
        v = k.pop(name, None)
        if v is not None:
            return v
        ev = env.get(name)
        return default if ev is None else ev

    steps = pick("steps")
    if steps is None:
        raise click.UsageError("--steps (or SNOWBALL_STEPS) is required")
    seq_len = pick("seq_len", DEFAULT_SEQ_LEN)
    keep_every = k.pop("keep_every", None)
    if keep_every is None:
        keep_every = env["keep_every"]
        if not keep_every and env["keep_per_epoch"]:
            from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import read_chat_cache_tokens  # noqa: PLC0415

            keep_every = expected_epoch_steps(read_chat_cache_tokens(k["data_cache_path"]), seq_len, k["batch_size"])
        keep_every = keep_every or 1000
    track_max_len = k.pop("track_max_len", None)
    if track_max_len is None:
        track_max_len = int(environ.get("SNOWBALL_TRACK_MAX_LEN") or seq_len)
    wandb_mode = k.pop("wandb_mode", None) or environ.get("WANDB_MODE") or "offline"
    cfg = QwenSftConfig(
        hf_checkpoint=k.pop("hf_checkpoint"),
        data_cache_path=k.pop("data_cache_path"),
        tokenizer_path=k.pop("tokenizer_path"),
        output_path=k.pop("output_path"),
        run_id=k.pop("run_id"),
        steps=int(steps),
        stage=k.pop("stage"),
        seq_len=int(seq_len),
        batch_size=k.pop("batch_size"),
        per_device_parallelism=k.pop("per_device_parallelism"),
        seed=k.pop("seed"),
        data_seed=pick("data_seed"),
        z_loss_weight=k.pop("z_loss_weight"),
        learning_rate=float(pick("learning_rate", 5e-5)),
        weight_decay=k.pop("weight_decay"),
        warmup=k.pop("warmup"),
        min_lr_ratio=k.pop("min_lr_ratio"),
        lr_schedule=k.pop("lr_schedule"),
        beta1=k.pop("beta1"),
        beta2=k.pop("beta2"),
        max_grad_norm=k.pop("max_grad_norm"),
        schedule_steps=pick("schedule_steps"),
        keep_every=int(keep_every),
        save_interval_minutes=k.pop("save_interval_minutes"),
        hf_save_path=k.pop("hf_save_path"),
        hf_save_dtype=k.pop("hf_save_dtype"),
        track_tol=float(pick("track_tol")),
        track_floor=float(pick("track_floor")),
        track_max_len=int(track_max_len),
        track_ema_alpha=float(pick("track_ema_alpha")),
        track_reenter=pick("track_reenter"),
        track_drop_weight=float(pick("track_drop_weight")),
        track_observe_only=bool(pick("track_observe_only")),
        trace_num_docs=int(pick("trace_num_docs")),
        trace_seed=int(pick("trace_seed")),
        trace_out=pick("trace_out"),
        attn_backend=k.pop("attn_backend"),
        flash_block_size=k.pop("flash_block_size"),
        gradient_checkpointing=k.pop("gradient_checkpointing"),
        tracker=k.pop("tracker"),
        wandb_project=k.pop("wandb_project"),
        wandb_mode=wandb_mode,
        log_dir=pick("log_dir", "logs/"),
        allow_cpu=k.pop("allow_cpu"),
    )
    if k:
        raise click.UsageError(f"unhandled options: {sorted(k)}")
    if cfg.schedule_steps is not None and cfg.schedule_steps < cfg.steps:
        raise ValueError(f"SNOWBALL_SCHEDULE_STEPS={cfg.schedule_steps} is shorter than the run's {cfg.steps} steps")
    return cfg


@click.group()
def main() -> None:
    """Qwen3 Levanter SFT with Plateau Dropping."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command("train")
@_common_options
def train_command(**kwargs) -> None:
    from experiments.june_tpu_67b_a2b.moe.train import (  # noqa: PLC0415
        apply_gpu_runtime_defaults,
        arm_hang_traceback_dumper,
    )

    cfg = config_from_cli(kwargs)
    if not cfg.allow_cpu:
        apply_gpu_runtime_defaults()
        arm_hang_traceback_dumper()
    summary = run_training(cfg)
    click.echo(json.dumps(summary, sort_keys=True))
    click.echo("QWEN_SFT_TRAIN_DONE")


@main.command("preflight")
@_common_options
@click.option("--expect-devices", type=int, default=None)
@click.option("--skip-output-check", is_flag=True, default=False)
def preflight_command(expect_devices, skip_output_check, **kwargs) -> None:
    cfg = config_from_cli(kwargs)
    if not cfg.allow_cpu:
        from levanter.distributed import DistributedConfig  # noqa: PLC0415

        DistributedConfig().initialize()
    report = run_preflight(cfg, expect_devices=expect_devices, check_output=not skip_output_check)
    click.echo(json.dumps(report, sort_keys=True, default=str))
    click.echo("QWEN_SFT_PREFLIGHT_OK")


if __name__ == "__main__":
    main()

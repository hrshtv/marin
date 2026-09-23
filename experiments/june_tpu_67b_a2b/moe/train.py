# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import faulthandler
import functools
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import load_checkpoint
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.grug.loss import (
    _axis_names_from_spec,
    _batch_axis_spec,
    _current_mesh,
    _psum_over_axes,
    _reshard_for_shard_map,
)
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.june_tpu_67b_a2b.checkpointing import restore_grug_state_from_checkpoint
from experiments.june_tpu_67b_a2b.dispatch import dispatch_grug_training_run
from experiments.june_tpu_67b_a2b.moe import tail_filter
from experiments.june_tpu_67b_a2b.moe.model import Block, GrugModelConfig, Transformer

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)


# This trainer was vendored from the June TPU 67B cooldown launcher, so it
# carried none of the GPU runtime workarounds the Grug MoE GPU trainers need.
# On a one-GPU-per-host GPU cluster every expert-axis collective crosses the
# fabric, and XLA GPU command buffers capture those NCCL collectives into a CUDA
# graph that hangs the first training step with no error while the GPUs stay
# pinned at high SM utilization.
# TODO(https://github.com/marin-community/marin/issues/5675): drop the command
# buffer override once the CUDA graph failure is fixed.
GPU_RUNTIME_ENV = {"JAX_ENABLE_PGLE": "false"}
# Deliberately narrower than experiments/grug/moe_hero_ep, whose `cuda_async`
# allocator and `parallel_collective_overlap_limit=4` are tuned for GB200 NVL72
# workers holding four GPUs on intra-host NVLink.
# Latency hiding is intentionally absent: the hero reference pairs it with
# parallel_collective_overlap_limit=4, and enabling the scheduler without that
# cap let XLA overlap unbounded collectives over Vista's single IB port per host.
_XLA_FLAG_DEFAULTS: tuple[str, ...] = ()
XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="


def apply_gpu_runtime_defaults() -> None:
    """Apply the Grug MoE GPU runtime contract unless the caller overrode it.

    Must run before JAX initializes its backend. The launcher exports the same
    settings; this keeps them attached to the code for any other entry point.
    """
    for name, value in GPU_RUNTIME_ENV.items():
        os.environ.setdefault(name, value)
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    explicit_names = {flag.partition("=")[0] for flag in xla_flags}
    defaults = (*_XLA_FLAG_DEFAULTS, XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG)
    xla_flags.extend(flag for flag in defaults if flag.partition("=")[0] not in explicit_names)
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)


def arm_hang_traceback_dumper() -> None:
    """Dump every thread's stack periodically so a silent hang self-documents.

    ``PYTHONFAULTHANDLER`` only fires on a fatal signal, which a spinning
    collective never raises. Set ``SNOWBALL_HANG_DUMP_SECONDS=0`` to disable.
    """
    raw = os.environ.get("SNOWBALL_HANG_DUMP_SECONDS", "0")
    try:
        seconds = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric SNOWBALL_HANG_DUMP_SECONDS=%r", raw)
        return
    if seconds <= 0:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(seconds, repeat=True, exit=False)
    logger.info("Armed hang traceback dumper every %.0fs", seconds)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 0.0  # Weight on logsumexp (z-loss) stabilization term.

    # TailSFT (arXiv 2608.25756; `tail_filter.py`). ``tail_fraction == 0`` (default) is the untouched
    # training path, byte for byte: the branch is a Python-level ``if`` on this static value. With
    # ``tail_fraction > 0`` every step drops that fraction of the documents in the global batch whose
    # length-normalised loss has fallen the most below the reference loss recorded for the same document
    # (cache row) under the initial checkpoint, so the gradient concentrates on the documents the model has
    # not fit yet. ``tail_ref_loss_path`` is the ``(num_docs,)`` float32 ``.npy`` that the scoring pass
    # writes (NaN = never scored, never dropped); ``tail_ramp_steps > 0`` ramps the fraction linearly from 0
    # over that many steps. ``tail_score_out`` switches the run into the scoring pass instead of training:
    # one forward-only sweep over ``num_train_steps`` batches of the training loader from step 0 with the
    # initial weights, writing the reference vector and exiting without a checkpoint.
    tail_fraction: float = 0.0
    tail_ref_loss_path: str | None = None
    tail_ramp_steps: int = 0
    tail_num_docs: int | None = None
    tail_score_out: str | None = None
    # Keep the imported router biases fixed: skip the per-step load-balancing overwrite
    # (_apply_qb_betas) and undo any optimizer update on them. This is what the Grug 09.21 SFT did
    # ("router biases and deferred load-balancing bias updates stayed frozen"); the default
    # re-derives the biases from zero on the SFT data.
    freeze_router_bias: bool = False
    # LR-schedule horizon in steps; None => num_train_steps. Set it LONGER than num_train_steps to train the
    # first part of a longer schedule now (e.g. one epoch of a two-epoch cosine) and resume later on the same
    # output with a larger num_train_steps: the loop picks up this run's own checkpoint (optimizer state and
    # step included) and the schedule continues unchanged, so the resumed run is the run you would have done.
    schedule_steps: int | None = None

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    model_axis_size: int = 1

    sft_weights_only_init: bool = False
    """SFT/RL init semantics (marin #650). When True and the run has no checkpoint of
    its own to auto-resume from, the trainer loads only the model weights (params +
    ``pending_qb_betas``) from ``TrainerConfig.initialize_from`` and keeps the fresh
    optimizer state and ``step=0`` -- i.e. a fresh LR schedule over the base weights,
    not a full-state resume. False (default) keeps the byte-identical continued-pretrain
    behaviour where ``initialize_from`` loads the whole train state (weights + optimizer +
    step). Own-run checkpoints still take precedence, so preemption resumes normally."""


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    steps_per_eval: int | None = 1000
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    eval_axis_mapping = {"batch": _BATCH_AXES}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _lm_flops_per_token(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_kv_heads: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    glu: bool,
    num_experts: int = 1,
    num_shared_experts: int = 0,
    num_experts_per_tok: int = 1,
    shared_intermediate_dim: int | None = None,
    sliding_window: int | None = None,
    num_full_attention_layers: int | None = None,
) -> float:
    """Analytic forward FLOPs per token, including the run's hybrid attention pattern."""
    head_dim = hidden_dim / num_heads
    shared_intermediate_dim = intermediate_dim if shared_intermediate_dim is None else shared_intermediate_dim
    routed_mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * num_experts_per_tok
    shared_mlp = 2 * (3 if glu else 2) * hidden_dim * shared_intermediate_dim * num_shared_experts
    mlp = routed_mlp + shared_mlp
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim

    def _attn_per_token(effective_seq: int) -> float:
        key_query_logits = 2 * effective_seq**2 * num_heads * head_dim
        mask = 3 * effective_seq * effective_seq * num_heads
        mask_value = 2 * effective_seq * effective_seq * head_dim * num_heads
        return (key_query_logits + mask + mask_value) / effective_seq

    if sliding_window is None:
        n_full = num_layers
        n_window = 0
    else:
        n_full = num_full_attention_layers if num_full_attention_layers is not None else 0
        if n_full < 0 or n_full > num_layers:
            raise ValueError(f"num_full_attention_layers ({n_full}) must be in [0, {num_layers}]")
        n_window = num_layers - n_full

    attn_full = _attn_per_token(seq_len) if n_full else 0.0
    if n_window:
        assert sliding_window is not None
        attn_window = _attn_per_token(min(seq_len, sliding_window))
    else:
        attn_window = 0.0
    per_layer_dense = mlp + qkv_proj + dense_proj
    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * per_layer_dense + n_full * attn_full + n_window * attn_window + lm_head


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    # Hybrid attention: every 4th layer plus the last layer runs full causal
    # attention; the rest use a sliding window (see ``_long_layer_schedule``
    # in model.py). At long context this makes the analytic FLOPs count much
    # smaller than a naive ``all-layers-full-attention`` estimate, because
    # each sliding-window layer's attention span is capped at the window.
    n = model_config.num_layers
    num_full_attention_layers = n // 4 + (0 if (n - 1) % 4 == 3 else 1)

    flops_per_token = _lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=1 if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
        sliding_window=model_config.sliding_window,
        num_full_attention_layers=num_full_attention_layers,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
        "throughput/num_full_attention_layers": float(num_full_attention_layers),
        "throughput/num_sliding_attention_layers": float(n - num_full_attention_layers),
        "throughput/sliding_window": float(model_config.sliding_window),
    }

    return flops_per_example, flops_summary


def _make_mixture_stage_callback(train_dataset: MixtureDataset, batch_schedule: BatchSchedule):
    last_mixture_stage = -1

    def log_mixture_stage(step_info):
        nonlocal last_mixture_stage
        seq_index = batch_schedule.global_data_offset_by_step(step_info.step)
        block_id = seq_index // train_dataset.block_size
        stage = train_dataset._get_stage_for_block(block_id)
        if stage == last_mixture_stage:
            return

        weights = train_dataset.weight_stages[stage][1]
        mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
        mixture_log["mixture/stage"] = stage
        levanter.tracker.log(mixture_log, step=step_info.step)
        last_mixture_stage = stage

    return log_mixture_stage


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_biases = -qb_betas
    new_biases = new_biases - jnp.mean(new_biases, axis=-1, keepdims=True)
    if model.stacked_blocks is not None:
        return eqx.tree_at(
            lambda t: t.stacked_blocks.stacked.mlp.router_bias,
            model,
            new_biases,
        )
    assert model.blocks is not None
    new_blocks = list(model.blocks)
    for i, block in enumerate(model.blocks):
        if not isinstance(block, Block) or block.mlp is None:
            continue
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, new_biases[i])
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def _copy_router_bias(model: Transformer, source: Transformer) -> Transformer:
    """Return ``model`` with every router bias taken from ``source`` (used to hold them frozen)."""
    if model.stacked_blocks is not None:
        return eqx.tree_at(
            lambda t: t.stacked_blocks.stacked.mlp.router_bias,
            model,
            source.stacked_blocks.stacked.mlp.router_bias,
        )
    assert model.blocks is not None
    new_blocks = list(model.blocks)
    for i, (block, src_block) in enumerate(zip(model.blocks, source.blocks, strict=True)):
        if not isinstance(block, Block) or block.mlp is None:
            continue
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, src_block.mlp.router_bias)
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    if params.blocks is not None:
        num_moe_layers = sum(1 for b in params.blocks if b.mlp is not None)
    else:
        num_moe_layers = model_config.num_layers
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def init_weights_only_from_checkpoint(
    state: GrugTrainState,
    checkpoint_path: str,
    *,
    mesh: Mesh | None,
    load_ema: bool,
    _load_fn: Callable[..., object] = load_checkpoint,
) -> GrugTrainState:
    """Load only model weights from an external checkpoint, resetting the optimizer.

    This is the SFT/RL init (marin #650): the base checkpoint supplies ``params`` and the
    ``pending_qb_betas`` router-bias state; the optimizer state and ``step`` stay at their
    fresh values in ``state`` so training starts a new LR schedule from step 0 instead of
    resuming the base run's optimizer/step.

    ``load_ema`` mirrors the loaded weights into ``ema_params`` when the run tracks an EMA.
    """
    # Deserialize only the ``params`` subtree and the ``pending_qb_betas`` leaf, keyed by their
    # GrugTrainState field names so they match the on-disk paths. allow_partial lets the base
    # checkpoint's other leaves (opt_state / step / ema_params) go unread, so the base run's
    # optimizer tree is never touched and stays fresh from ``state``.
    exemplar: dict[str, object] = {"params": state.params, "pending_qb_betas": state.pending_qb_betas}
    loaded = cast("dict[str, object]", _load_fn(exemplar, checkpoint_path, mesh=mesh, allow_partial=True))
    updates: dict[str, object] = {"params": loaded["params"], "pending_qb_betas": loaded["pending_qb_betas"]}
    if load_ema and state.ema_params is not None:
        updates["ema_params"] = loaded["params"]
    return dataclasses.replace(state, **updates)


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    tail: "TailSettings | None" = None,
    freeze_router_bias: bool = False,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    # TailSFT: a static Python-level branch, so ``tail is None`` compiles today's exact loss lines.
    tail_ref_np = None if tail is None else tail.ref_loss
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("compute_watch",))
    def train_step(state: GrugTrainState, batch, *, compute_watch: bool = False):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        if freeze_router_bias:
            qb_params = state.params
            qb_ema_params = state.ema_params if ema_beta is not None else None
        else:
            qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
            if ema_beta is not None:
                qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
            else:
                qb_ema_params = None

        if tail is None:

            def loss_fn(params):
                compute_params = mp.cast_to_compute(params)
                return compute_params.next_token_loss(
                    batch.tokens,
                    batch.loss_weight,
                    mask=batch.attn_mask,
                    reduction="mean",
                    logsumexp_weight=z_loss,
                    return_router_metrics=True,
                )

        else:
            segment_ids = _packed_segment_ids(batch)

            def loss_fn(params):
                compute_params = mp.cast_to_compute(params)
                # One vocab projection: the per-position (weighted) loss, reduced by hand below.
                per_pos, summarized = compute_params.next_token_loss(
                    batch.tokens,
                    batch.loss_weight,
                    mask=batch.attn_mask,
                    reduction="none",
                    logsumexp_weight=z_loss,
                    return_router_metrics=True,
                )
                fraction = tail_filter.tail_schedule(state.step, tail.fraction, tail.ramp_steps)
                ref = jnp.asarray(tail_ref_np, dtype=jnp.float32)
                loss, tail_stats = _tail_filtered_loss_sharded(per_pos, batch.loss_weight, segment_ids, ref, fraction)
                # ``reduction="none"`` returns the bare cross-entropy array: re-add the router z-loss term the
                # "mean" path adds (coefficient 0.0 on this config, kept for parity) and replace the
                # per-position array the metrics carry with the unfiltered scalar mean.
                loss = loss + summarized["train/router/aux_loss_weighted"]
                summarized = dict(summarized)
                summarized["train/cross_entropy_loss"] = tail_stats["tail/loss_unfiltered"]
                summarized.update({f"train/{k}": v for k, v in tail_stats.items()})
                return loss, summarized

        (loss, summarized_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
        metrics = {"train/loss": loss, **summarized_metrics}
        updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
        params = optax.apply_updates(qb_params, updates)
        if freeze_router_bias:
            params = _copy_router_bias(params, state.params)

        if ema_beta is None:
            ema_params = None
        else:
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                qb_ema_params,
                params,
            )

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=state.pending_qb_betas if freeze_router_bias else metrics["qb_beta_per_layer"],
        )

        return next_state, metrics, watch_stats

    return train_step


@dataclass(frozen=True)
class TailSettings:
    """Resolved TailSFT settings for one run: the static fraction/ramp and the host-side reference vector."""

    fraction: float
    ramp_steps: int
    ref_loss: np.ndarray  # (num_docs,) float32, NaN = never scored


def _packed_segment_ids(batch) -> jax.Array:
    """The ``(B, S)`` int32 segment ids of a packed batch (global cache row per token, ``-1`` padding)."""
    mask = getattr(batch, "attn_mask", None)
    segment_ids = getattr(mask, "segment_ids", None)
    if segment_ids is None:
        raise ValueError("TailSFT needs packed batches with segment ids; this batch carries none.")
    return segment_ids[0]


def _tail_shard_map(fn, per_pos: jax.Array, weight: jax.Array, segment_ids: jax.Array, *replicated, out_specs):
    """Run ``fn(per_pos, weight, seg, *replicated, psum=...)`` on local shards over the batch axes.

    Mirrors ``levanter.grug.loss``: rows stay sharded over the batch axes the per-position loss already
    carries, the extra arguments are replicated, and ``fn`` psums its own per-document / scalar results so
    every device returns identical replicated values.
    """
    mesh = _current_mesh()
    if mesh is None or mesh.empty:
        return fn(per_pos, weight, segment_ids, *replicated)
    axis_spec = _batch_axis_spec(per_pos)
    axis_names = _axis_names_from_spec(axis_spec)
    row_spec = P(axis_spec)
    rep_spec = P()
    per_pos = _reshard_for_shard_map(per_pos, mesh, row_spec)
    weight = _reshard_for_shard_map(weight, mesh, row_spec)
    segment_ids = _reshard_for_shard_map(segment_ids, mesh, row_spec)
    replicated = tuple(_reshard_for_shard_map(r, mesh, rep_spec) for r in replicated)

    def _psum(x):
        return _psum_over_axes(x, axis_names)

    def _local(a, b, c, *r):
        return fn(a, b, c, *r, psum=_psum)

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(row_spec, row_spec, row_spec) + (rep_spec,) * len(replicated),
        out_specs=out_specs,
        check_vma=False,
    )(per_pos, weight, segment_ids, *replicated)


def _tail_filtered_loss_sharded(per_pos, weight, segment_ids, ref_loss, fraction):
    stats_specs = {k: P() for k in tail_filter.STAT_KEYS}
    return _tail_shard_map(
        tail_filter.tail_filtered_loss, per_pos, weight, segment_ids, ref_loss, fraction, out_specs=(P(), stats_specs)
    )


def _tail_document_sums_sharded(per_pos, weight, segment_ids, num_docs: int):
    def _sums(a, b, c, *, psum):
        return tail_filter.document_sums(a, b, c, num_docs, psum=psum)

    return _tail_shard_map(_sums, per_pos, weight, segment_ids, out_specs=(P(), P()))


def _make_tail_score_step(mp: jmp.Policy, *, z_loss_weight: float, num_docs: int):
    """Forward-only per-document (loss sum, weight sum) with the current weights, the reference scorer."""
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @jax.jit
    def score_step(params, pending_qb_betas, batch):
        qb_params = _apply_qb_betas(params, pending_qb_betas)
        compute_params = mp.cast_to_compute(qb_params)
        per_pos = compute_params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=z_loss,
        )
        return _tail_document_sums_sharded(per_pos, batch.loss_weight, _packed_segment_ids(batch), num_docs)

    return score_step


def _run_tail_scoring(
    *,
    state: "GrugTrainState",
    train_loader,
    mp: jmp.Policy,
    z_loss_weight: float,
    num_docs: int,
    num_batches: int,
    out_path: str,
) -> None:
    """Score every training document under the initial weights and write the reference vector.

    Replays the training loader from step 0, so the first batches are scored in the composition the first
    training steps will see (the MoE routes at capacity, so a document's loss depends on its batch), and
    keeps going for up to ``num_batches`` batches or until every document has been scored, whichever comes
    first. The mixture loader samples rows with replacement (one epoch of the 320-trial cache reached 200
    documents), so callers pass several epochs' worth of batches; a document drawn more than once gets the
    token-weighted mean of its occurrences. Progress lines follow the trainer's ``Progress on:train`` format
    so the guarded job's watchdog sees them.
    """
    score_step = _make_tail_score_step(mp, z_loss_weight=z_loss_weight, num_docs=num_docs)
    num = np.zeros(num_docs, dtype=np.float64)
    den = np.zeros(num_docs, dtype=np.float64)
    iterator = train_loader.iter_from_step(0)
    t0 = time.perf_counter()
    batches_run = 0
    for i in range(num_batches):
        batch = next(iterator)
        b_num, b_den = score_step(state.params, state.pending_qb_betas, batch)
        b_num = np.asarray(jax.device_get(b_num), dtype=np.float64)
        b_den = np.asarray(jax.device_get(b_den), dtype=np.float64)
        num += b_num
        den += b_den
        batches_run = i + 1
        seen = int((den > 0).sum())
        batch_loss = float(b_num.sum() / b_den.sum()) if b_den.sum() > 0 else float("nan")
        logger.info(
            f"Progress on:train {i + 1}.0it/{num_batches}.0it tail_score docs_seen={seen}/{num_docs} "
            f"batch_loss={batch_loss:.4f} elapsed={time.perf_counter() - t0:.0f}s"
        )
        if seen >= num_docs:
            logger.info(f"tail_score: every document scored after {batches_run} batches; stopping early")
            break
    ref = np.full(num_docs, np.nan, dtype=np.float32)
    scored = den > 0
    ref[scored] = (num[scored] / den[scored]).astype(np.float32)
    if jax.process_index() == 0:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, ref)
        meta = {
            "num_docs": int(num_docs),
            "scored_docs": int(scored.sum()),
            "batches": int(num_batches),
            "weighted_tokens": float(den.sum()),
            "mean_loss": float(num[scored].sum() / den[scored].sum()) if scored.any() else None,
            "z_loss_weight": z_loss_weight,
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        logger.info(f"TAIL_SCORE_DONE {out} scored={meta['scored_docs']}/{num_docs} mean_loss={meta['mean_loss']}")


def run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    schedule_steps = config.trainer.schedule_steps or trainer.num_train_steps
    if schedule_steps < trainer.num_train_steps:
        raise ValueError(f"schedule_steps {schedule_steps} is shorter than num_train_steps {trainer.num_train_steps}")
    print(f"lr_schedule_steps={schedule_steps} num_train_steps={trainer.num_train_steps}", flush=True)
    optimizer = config.optimizer.build(schedule_steps)
    watch_config = trainer.watch
    tail: TailSettings | None = None
    if config.trainer.tail_score_out is None and config.trainer.tail_fraction > 0:
        if not config.trainer.tail_ref_loss_path or config.trainer.tail_num_docs is None:
            raise ValueError("tail_fraction > 0 needs tail_ref_loss_path (the scoring pass output) and tail_num_docs.")
        if not (0.0 < config.trainer.tail_fraction < 1.0):
            raise ValueError(f"tail_fraction must lie in (0, 1); got {config.trainer.tail_fraction}.")
        ref = tail_filter.load_reference_losses(config.trainer.tail_ref_loss_path, config.trainer.tail_num_docs)
        tail = TailSettings(
            fraction=float(config.trainer.tail_fraction),
            ramp_steps=int(config.trainer.tail_ramp_steps),
            ref_loss=ref,
        )
        logger.info(
            f"TailSFT on: fraction={tail.fraction} ramp_steps={tail.ramp_steps} "
            f"reference={config.trainer.tail_ref_loss_path} scored_docs={int(np.isfinite(ref).sum())}/{ref.shape[0]}"
        )
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=watch_config if watch_config.is_enabled else None,
        tail=tail,
        freeze_router_bias=config.trainer.freeze_router_bias,
    )
    if config.trainer.freeze_router_bias:
        logger.info("Router biases FROZEN: keeping the imported values, no load-balancing update")

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=config.trainer.model_axis_size,
    )
    with set_mesh(mesh):
        batch_schedule = trainer.batch_schedule

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
        )

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id)
        if config.trainer.sft_weights_only_init:
            # SFT/RL: auto-resume from this run's own checkpoints if present (preemption),
            # otherwise load only base weights (+ pending_qb_betas) and keep the fresh
            # optimizer/step (marin #650). initialize_from is deliberately withheld here so
            # the restore never does a full-state load; the weights-only init runs below.
            state = restore_grug_state_from_checkpoint(
                state,
                checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
                load_checkpoint_setting=trainer.load_checkpoint,
                mesh=mesh,
                allow_partial=trainer.allow_partial_checkpoint,
            )
            if int(state.step) == 0 and trainer.initialize_from is not None:
                state = init_weights_only_from_checkpoint(
                    state,
                    trainer.initialize_from,
                    mesh=mesh,
                    load_ema=config.trainer.ema_beta is not None,
                )
        else:
            state = restore_grug_state_from_checkpoint(
                state,
                checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
                load_checkpoint_setting=trainer.load_checkpoint,
                mesh=mesh,
                allow_partial=trainer.allow_partial_checkpoint,
                initialize_from=trainer.initialize_from,
            )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
            )

        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=trainer.num_train_steps)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer.num_train_steps), every=log_every)
        if profiler_enabled:
            state_callbacks.add_hook(
                profiler_cfg.build(
                    str(trainer.log_dir / run_id / "profiler"),
                    run_id=run_id,
                    num_steps=profiler_num_steps,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )

        if config.trainer.tail_score_out is not None:
            # TailSFT reference pass: forward-only over one epoch of the training loader with the initial
            # weights, then exit. No checkpoint, no training.
            if config.trainer.tail_num_docs is None:
                raise ValueError("tail_score_out needs tail_num_docs (the cache's document count).")
            if int(state.step) != 0:
                raise ValueError(f"tail scoring must start from step 0, found a resumed state at step {int(state.step)}.")
            _run_tail_scoring(
                state=state,
                train_loader=train_loader,
                mp=trainer.mp,
                z_loss_weight=config.trainer.z_loss_weight,
                num_docs=int(config.trainer.tail_num_docs),
                num_batches=int(trainer.num_train_steps),
                out_path=config.trainer.tail_score_out,
            )
            levanter.tracker.current_tracker().finish()
            return

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0

        # Main optimization loop.
        try:
            while int(state.step) < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                # grad_watch runs only on its configured interval.
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = int(state.step) - 1

                jax.block_until_ready(metrics["train/loss"])

                if jnp.isnan(metrics["train/loss"]):
                    logger.error(f"NaN loss at step {int(state.step)}. Stopping training.")
                    break
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    router_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if (key.startswith("train/router/") or key.startswith("moe_bias/") or key.startswith("train/tail/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if tail is not None and step % log_every == 0:
                        # The tracker is offline on the clusters; keep the filter's behaviour in the run log.
                        logger.info(
                            "tail step %d: fraction %.3f dropped %d of %d scorable (%d present), kept tokens %.3f, "
                            "loss %.4f unfiltered %.4f, threshold margin %.4f, mean margin %.4f",
                            step,
                            float(metrics["train/tail/fraction"]),
                            int(metrics["train/tail/dropped_docs"]),
                            int(metrics["train/tail/scorable_docs"]),
                            int(metrics["train/tail/present_docs"]),
                            float(metrics["train/tail/kept_token_frac"]),
                            float(metrics["train/loss"]),
                            float(metrics["train/tail/loss_unfiltered"]),
                            float(metrics["train/tail/threshold_margin"]),
                            float(metrics["train/tail/mean_margin"]),
                        )
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=int(state.step))
        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                # The loop already saved this step when it fell on a permanent-keep interval (e.g. keep every
                # epoch with the run ending on an epoch): a second, forced save into the same step directory
                # aborts every host in multihost_utils.assert_equal (job 1890618, 2026-09-19). Force only when
                # the last step has not been saved yet.
                if checkpointer._last_save_step != int(state.step):
                    checkpointer.on_step(tree=state, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=run_grug_local,
        resources=config.resources,
    )


__all__ = [
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug",
    "run_grug_local",
]

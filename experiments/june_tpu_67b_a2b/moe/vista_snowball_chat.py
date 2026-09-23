# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and train the Snowball chat stage on TACC Vista.

Each Slurm rank invokes the ``train`` command once. JAX discovers the Slurm
process topology and the vendored Grug loop runs directly on those ranks; no
Fray coordinator is started inside the allocation.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import math
import os
import shutil
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import click
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from haliax import Axis
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem import prefix_join
from transformers import AutoTokenizer

from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig
from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import (
    SNOWBALL_CHAT_BATCH_SIZE,
    SNOWBALL_CHAT_DEVICES,
    SNOWBALL_CHAT_EXAMPLES,
    SNOWBALL_CHAT_EXPERT_PARALLEL,
    SNOWBALL_CHAT_MODEL_AXIS,
    SNOWBALL_CHAT_MODEL_CONFIG,
    SNOWBALL_CHAT_MP,
    SNOWBALL_CHAT_OPTIMIZER,
    SNOWBALL_CHAT_REPLICA_AXIS,
    SNOWBALL_CHAT_SEED,
    SNOWBALL_CHAT_SEQUENCE_LENGTH,
    selected_base_model,
    SNOWBALL_CHAT_STEPS,
    SNOWBALL_CHAT_TOKENS,
    SNOWBALL_NATIVE_PARAMETERS,
)
from experiments.june_tpu_67b_a2b.moe.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    apply_gpu_runtime_defaults,
    arm_hang_traceback_dumper,
    run_grug_local,
)
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE

_MINIMUM_OUTPUT_FREE_BYTES = 1_000_000_000_000
_MINIMUM_OUTPUT_FREE_INODES = 10_000


def snowball_chat_format(
    *, messages_field: str = "conversation", chat_template: str | None = None
) -> ChatLmDatasetFormat:
    """Chat format for one stage.

    Chat keeps the ``conversation`` field; Thinking uses ``messages``; the
    Nemotron Terminal stage uses ``conversations`` AND the Marin template rather
    than Delphi V0 -- so the template is per-stage, not a constant. Defaulting to
    Delphi V0 keeps Chat and Thinking byte-identical to what they trained on.
    """
    return ChatLmDatasetFormat(
        messages_field=messages_field,
        chat_template=chat_template or DELPHI_V0_CHAT_TEMPLATE,
        mask_user_turns=True,
        pack=None,
    )


def resolve_source_shards(
    *,
    parquet_glob: str | None = None,
    parquet_list: str | None = None,
    expect_files: int | None = None,
) -> list[str]:
    """Resolve the pinned source shards for a cache build.

    ``parquet_list`` is the safe form: an explicit newline-delimited file of
    paths, with no pattern semantics to misread. A glob is still accepted but is
    expanded with ``recursive=True``. Without that flag ``**`` collapses to a
    single ``*`` and silently drops every shard nested more than one directory
    deep -- which is how job 983290 built the Stage 3 cache from 3 of the 29
    pinned Nemotron Terminal files while a ``find``-based gate reported 29.
    """
    if (parquet_glob is None) == (parquet_list is None):
        raise SystemExit("FATAL: pass exactly one of --parquet-glob / --parquet-list.")
    if parquet_list is not None:
        entries = [line.strip() for line in Path(parquet_list).read_text().splitlines()]
        paths = sorted(e for e in entries if e and not e.startswith("#"))
        if not paths:
            raise SystemExit(f"FATAL: {parquet_list!r} lists no shards.")
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise SystemExit(f"FATAL: {len(missing)} of {len(paths)} listed shards do not exist, e.g. {missing[0]!r}.")
    else:
        paths = sorted(glob.glob(parquet_glob, recursive=True))
        if not paths:
            raise FileNotFoundError(f"No Parquet shards matched {parquet_glob!r}.")
    if expect_files is not None and len(paths) != expect_files:
        raise SystemExit(f"FATAL: resolved {len(paths)} source shards, expected {expect_files}.")
    return paths


def prepare_chat_cache(
    *,
    parquet_glob: str | None = None,
    parquet_list: str | None = None,
    expect_files: int | None = None,
    cache_path: str,
    tokenizer_path: str,
    messages_field: str = "conversation",
    tags: tuple[str, ...] = ("wildchat_386k", "snowball_chat"),
    chat_template: str | None = None,
) -> int:
    """Tokenize and pack pinned Parquet shards with the Delphi V0 chat format."""
    from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize  # noqa: PLC0415

    paths = resolve_source_shards(parquet_glob=parquet_glob, parquet_list=parquet_list, expect_files=expect_files)
    # Log every resolved shard. 983290 was only caught by reading zephyr reader
    # lines; the resolution step itself left no auditable record.
    print(f"resolved_source_shards={len(paths)}", flush=True)
    for path in paths:
        print(f"  source_shard: {path}", flush=True)
    tokenize(
        TokenizeConfig(
            train_paths=paths,
            validation_paths=[],
            cache_path=cache_path,
            tokenizer=tokenizer_path,
            tags=list(tags),
            format=snowball_chat_format(messages_field=messages_field, chat_template=chat_template),
            max_workers=len(paths),
            worker_resources=ResourceConfig(cpu=16, ram="64g", disk="20g"),
        )
    )
    return read_chat_cache_tokens(cache_path)


def read_chat_cache_examples(cache_path: str) -> int:
    """Number of documents (cache rows) in the packed train cache; TailSFT indexes its reference vector by it."""
    stats_path = Path(cache_path) / "train" / ".stats.json"
    stats = json.loads(stats_path.read_text())
    total_elements = stats.get("total_elements")
    if not isinstance(total_elements, int) or total_elements <= 0:
        raise ValueError(f"Invalid total_elements in {stats_path}: {total_elements!r}")
    return total_elements


def read_chat_cache_tokens(cache_path: str) -> int:
    stats_path = Path(cache_path) / "train" / ".stats.json"
    stats = json.loads(stats_path.read_text())
    total_tokens = stats.get("total_tokens")
    if not isinstance(total_tokens, int) or total_tokens <= 0:
        raise ValueError(f"Invalid total_tokens in {stats_path}: {total_tokens!r}")
    return total_tokens


@dataclasses.dataclass(frozen=True)
class SnowballChatPreflight:
    """Static artifacts and geometry accepted by the Vista launch gate."""

    init_checkpoint_path: str
    init_payload_files: int
    init_payload_bytes: int
    cache_tokens: int
    cache_examples: int
    cache_shards: int
    tokenizer_vocab_size: int
    data_axis_size: int
    per_device_batch_size: int
    output_free_bytes: int
    output_free_inodes: int
    output_checkpoint_path: str | None


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def validate_native_checkpoint_layout(init_checkpoint_path: str, *, expect_step: int = 0) -> tuple[str, int, int]:
    """Resolve a complete native checkpoint and measure its payload.

    ``expect_step`` is EXACT and required. Chat passes 0 (the base cooldown
    export); a chained stage passes the completed prior stage's final step. An
    earlier version accepted "any step >= 1" for chained stages, which let a
    step-3 smoke checkpoint pass the gate -- hence the exact match.
    """
    if not Path(init_checkpoint_path).is_absolute():
        raise ValueError(f"Snowball init checkpoint path must be absolute: {init_checkpoint_path!r}.")

    resolved_path = latest_checkpoint_path(init_checkpoint_path)
    checkpoint_path = Path(resolved_path)
    metadata = _json_object(checkpoint_path / "metadata.json")
    actual_step = metadata.get("step")
    if actual_step != expect_step:
        raise ValueError(
            f"Init checkpoint must report step {expect_step}, got {actual_step!r} in {resolved_path}. "
            f"A chained stage must start from the COMPLETED prior stage; an intermediate or smoke "
            f"checkpoint is not acceptable."
        )

    manifest_path = checkpoint_path / "manifest.ocdbt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snowball base checkpoint is missing {manifest_path}.")
    numbered_manifests = [
        path
        for path in checkpoint_path.glob("manifest.*")
        if path.name not in {"manifest.json", "manifest.ocdbt"} and path.is_file()
    ]
    if not numbered_manifests:
        raise FileNotFoundError(f"Snowball base checkpoint has no numbered OCDBT manifests in {resolved_path}.")

    payload_path = checkpoint_path / "d"
    payload_files = [path for path in payload_path.rglob("*") if path.is_file()]
    payload_bytes = sum(path.stat().st_size for path in payload_files)
    minimum_payload_bytes = SNOWBALL_NATIVE_PARAMETERS * 2
    if not payload_files or payload_bytes < minimum_payload_bytes:
        raise ValueError(
            f"Snowball base checkpoint payload is incomplete: {len(payload_files)} files and {payload_bytes} bytes; "
            f"expected at least {minimum_payload_bytes} bytes."
        )
    return resolved_path, len(payload_files), payload_bytes


PROVENANCE_FILENAME = "snowball_provenance.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cache_provenance(
    *,
    cache_path: str,
    stage: str,
    dataset_id: str,
    dataset_revision: str,
    shard_paths: Sequence[str],
    tokenizer_path: str,
) -> dict:
    """Record WHICH source produced a cache.

    Token and example counts do not identify a revision: two different dataset
    commits can produce identical counts. This sidecar records the dataset id and
    full revision, a hash of every input shard, the tokenizer hash, and the
    format identity, so preflight can prove provenance instead of inferring it.
    """
    spec = STAGES[stage]
    fmt = snowball_chat_format(messages_field=spec.messages_field, chat_template=spec.chat_template)
    # Key each shard by its path relative to the shards' common root, NOT by
    # basename. The Nemotron Terminal selection holds 26 files all named
    # data_filtered.parquet in different directories; a basename key collapses
    # them to ONE entry, so a 29-shard build recorded 4 hashes and silently
    # discarded 25. Relative paths also match the upstream pinned-list format,
    # making the sidecar directly comparable to it.
    shard_list = sorted(shard_paths)
    if not shard_list:
        raise ValueError("write_cache_provenance requires at least one shard path.")
    parents = [str(Path(p).parent) for p in shard_list]
    root = os.path.commonpath(parents) if len(set(parents)) > 1 else parents[0]
    shards = {os.path.relpath(p, root): _sha256_file(Path(p)) for p in shard_list}
    if len(shards) != len(shard_list):
        raise ValueError(
            f"Shard key collision: {len(shard_list)} input shards produced only "
            f"{len(shards)} provenance keys, so hashes would be silently discarded."
        )
    record = {
        "stage": stage,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "shard_root": root,
        "shards": shards,
        "tokenizer_sha256": _sha256_file(Path(tokenizer_path) / "tokenizer.json"),
        "format": {
            "messages_field": fmt.messages_field,
            "mask_user_turns": bool(fmt.mask_user_turns),
            "chat_template_sha256": hashlib.sha256((fmt.chat_template or "").encode("utf-8")).hexdigest(),
        },
    }
    out = Path(cache_path) / PROVENANCE_FILENAME
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def validate_cache_provenance(cache_path: str, stage: str, tokenizer_path: str) -> dict:
    """Prove the cache came from this stage's pinned source."""
    spec = STAGES[stage]
    path = Path(cache_path) / PROVENANCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache at {cache_path} has no {PROVENANCE_FILENAME}; its source revision cannot be "
            f"verified. Regenerate it with the write-provenance command."
        )
    rec = json.loads(path.read_text())
    if rec.get("stage") != stage:
        raise ValueError(f"Cache provenance says stage {rec.get('stage')!r}, expected {stage!r}.")
    if rec.get("dataset_revision") != spec.dataset_revision:
        raise ValueError(
            f"Cache was built from dataset revision {rec.get('dataset_revision')!r}, "
            f"but {stage} pins {spec.dataset_revision!r}."
        )
    tok_sha = _sha256_file(Path(tokenizer_path) / "tokenizer.json")
    if rec.get("tokenizer_sha256") != tok_sha:
        raise ValueError(
            f"Cache was built with tokenizer {rec.get('tokenizer_sha256')!r}, but preflight was given {tok_sha!r}."
        )
    fmt = snowball_chat_format(messages_field=spec.messages_field, chat_template=spec.chat_template)
    want_fmt = {
        "messages_field": fmt.messages_field,
        "mask_user_turns": bool(fmt.mask_user_turns),
        "chat_template_sha256": hashlib.sha256((fmt.chat_template or "").encode("utf-8")).hexdigest(),
    }
    if rec.get("format") != want_fmt:
        raise ValueError(f"Cache format identity {rec.get('format')!r} does not match {want_fmt!r}.")
    shards = rec.get("shards") or {}
    if not shards:
        raise ValueError("Cache provenance records no input shard hashes.")
    if spec.source_files and len(shards) != spec.source_files:
        raise ValueError(
            f"Cache was built from {len(shards)} input shards, but {stage} pins "
            f"{spec.source_files}. A truncated source set is otherwise silent: a "
            f"non-recursive glob built the Stage 3 cache from 3 of 29 shards and "
            f"every other provenance field still matched."
        )
    return rec


def validate_chat_cache_layout(
    data_cache_path: str,
    *,
    expect_tokens: int | None = SNOWBALL_CHAT_TOKENS,
    expect_examples: int | None = SNOWBALL_CHAT_EXAMPLES,
) -> tuple[int, int, int]:
    """Validate a completed packed cache.

    Chat pins the EXACT WildChat totals (defaults) so Stage 1 cannot drift. Other
    stages pass ``None`` and are validated structurally instead -- ledger shape,
    finished shards, and a non-empty token/example count -- because their totals
    are a property of their own dataset, not of WildChat.
    """
    cache_path = Path(data_cache_path)
    if not cache_path.is_absolute():
        raise ValueError(f"Snowball data cache path must be absolute: {data_cache_path!r}.")

    train_path = cache_path / "train"
    stats = _json_object(train_path / ".stats.json")
    total_tokens = stats.get("total_tokens")
    total_examples = stats.get("total_elements")
    if expect_tokens is not None or expect_examples is not None:
        if total_tokens != expect_tokens or total_examples != expect_examples:
            raise ValueError(
                f"Packed cache has {total_tokens!r} tokens and {total_examples!r} examples; expected "
                f"{expect_tokens} tokens and {expect_examples} examples."
            )
    else:
        if not isinstance(total_tokens, int) or total_tokens < 1:
            raise ValueError(f"Packed cache reports {total_tokens!r} tokens; refusing to train on an empty cache.")
        if not isinstance(total_examples, int) or total_examples < 1:
            raise ValueError(f"Packed cache reports {total_examples!r} examples; refusing to train on an empty cache.")

    ledger = _json_object(train_path / "shard_ledger.json")
    shard_rows = ledger.get("shard_rows")
    finished_shards = ledger.get("finished_shards")
    field_counts = ledger.get("field_counts")
    if not isinstance(shard_rows, dict) or not isinstance(finished_shards, list):
        raise ValueError(f"Packed WildChat cache has a malformed shard ledger at {train_path}.")
    if not all(isinstance(shard, str) for shard in finished_shards) or not all(
        isinstance(shard, str) and isinstance(rows, int) for shard, rows in shard_rows.items()
    ):
        raise ValueError(f"Packed WildChat cache has malformed shard names or row counts at {train_path}.")
    typed_finished_shards = [str(shard) for shard in finished_shards]
    typed_shard_rows = {str(shard): int(rows) for shard, rows in shard_rows.items()}
    if ledger.get("is_finished") is not True or set(typed_finished_shards) != set(typed_shard_rows):
        raise ValueError(f"Packed WildChat cache is not fully committed at {train_path}.")
    # Cross-check the ledger against the cache's OWN totals. Comparing against
    # Chat's constants here would reject every other dataset by construction.
    if sum(typed_shard_rows.values()) != total_examples:
        raise ValueError(
            f"Packed cache shard rows sum to {sum(typed_shard_rows.values())}, "
            f"but .stats.json reports {total_examples} examples."
        )
    if field_counts != {"assistant_masks": total_tokens, "input_ids": total_tokens}:
        raise ValueError(f"Packed cache field counts {field_counts!r} disagree with the reported {total_tokens} tokens.")

    for shard in typed_finished_shards:
        shard_path = train_path / shard
        if not (shard_path / ".success").is_file():
            raise FileNotFoundError(f"Packed WildChat cache shard is missing its success marker: {shard_path}.")
        for field in ("assistant_masks", "input_ids"):
            field_path = shard_path / field
            has_payload = field_path.is_dir() and any(
                path.is_file() and path.stat().st_size > 0 for path in field_path.rglob("*")
            )
            if not has_payload:
                raise FileNotFoundError(f"Packed WildChat cache shard has no {field} payload: {shard_path}.")

    # Return what the cache ACTUALLY contains, not Chat's constants -- returning
    # the latter would report WildChat totals for a Thinking cache.
    return total_tokens, total_examples, len(typed_finished_shards)


# sha256 of the selected base model's tokenizer (snowball_chat_recipe.BASE_MODELS). Every
# stage in a lineage must use the identical tokenizer; a silent swap would corrupt the chained
# weights-only init.
SNOWBALL_TOKENIZER_SHA256 = selected_base_model().tokenizer_sha256


def _validate_tokenizer(tokenizer_path: str) -> int:
    path = Path(tokenizer_path)
    if not path.is_absolute():
        raise ValueError(f"Snowball tokenizer path must be absolute: {tokenizer_path!r}.")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Snowball tokenizer is missing {path / filename}.")

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    vocab_size = len(tokenizer)
    if vocab_size != SNOWBALL_CHAT_MODEL_CONFIG.vocab_size:
        raise ValueError(
            f"Snowball tokenizer has {vocab_size} tokens, but the model expects {SNOWBALL_CHAT_MODEL_CONFIG.vocab_size}."
        )
    token_ids = tokenizer.encode("Snowball launch preflight", add_special_tokens=False)
    if not token_ids or max(token_ids) >= vocab_size:
        raise ValueError(f"Snowball tokenizer produced invalid token IDs: {token_ids!r}.")
    fingerprint = hashlib.sha256((Path(tokenizer_path) / "tokenizer.json").read_bytes()).hexdigest()
    if fingerprint != SNOWBALL_TOKENIZER_SHA256:
        raise ValueError(
            f"Tokenizer fingerprint {fingerprint} does not match the pinned Snowball tokenizer "
            f"{SNOWBALL_TOKENIZER_SHA256}. Every stage must share one tokenizer."
        )
    return vocab_size


def _validate_output_path(
    output_path: str,
    *,
    input_paths: tuple[str, ...],
    expected_output_checkpoint_step: int | None,
) -> tuple[str | None, int, int]:
    path = Path(output_path)
    if not path.is_absolute():
        raise ValueError(f"Snowball output path must be absolute: {output_path!r}.")
    resolved_output = path.resolve(strict=False)
    for input_path in input_paths:
        resolved_input = Path(input_path).resolve(strict=True)
        if (
            resolved_output == resolved_input
            or resolved_output in resolved_input.parents
            or resolved_input in resolved_output.parents
        ):
            raise ValueError(f"Snowball output path {resolved_output} overlaps input path {resolved_input}.")

    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK | os.X_OK):
        raise PermissionError(f"Snowball output parent is not writable: {path.parent}.")
    free_bytes = shutil.disk_usage(path.parent).free
    stat = os.statvfs(path.parent)
    free_inodes = stat.f_favail
    if free_bytes < _MINIMUM_OUTPUT_FREE_BYTES:
        raise OSError(
            f"Snowball output filesystem has only {free_bytes} free bytes; "
            f"at least {_MINIMUM_OUTPUT_FREE_BYTES} are required."
        )
    if free_inodes < _MINIMUM_OUTPUT_FREE_INODES:
        raise OSError(
            f"Snowball output filesystem has only {free_inodes} free inodes; "
            f"at least {_MINIMUM_OUTPUT_FREE_INODES} are required."
        )

    if expected_output_checkpoint_step is None:
        if path.exists():
            raise FileExistsError(f"Fresh Snowball output path already exists: {path}.")
        return None, free_bytes, free_inodes

    output_checkpoint = latest_checkpoint_path(path / "checkpoints")
    output_checkpoint_path = Path(output_checkpoint)
    output_metadata = _json_object(output_checkpoint_path / "metadata.json")
    if output_metadata.get("step") != expected_output_checkpoint_step:
        raise ValueError(
            f"Snowball resume expected output checkpoint step {expected_output_checkpoint_step}, got "
            f"{output_metadata.get('step')!r} at {output_checkpoint}."
        )
    if not (output_checkpoint_path / "manifest.ocdbt").is_file():
        raise FileNotFoundError(f"Snowball resume checkpoint is missing manifest.ocdbt: {output_checkpoint}.")
    output_payload_files = [path for path in (output_checkpoint_path / "d").rglob("*") if path.is_file()]
    if not output_payload_files or sum(path.stat().st_size for path in output_payload_files) == 0:
        raise ValueError(f"Snowball resume checkpoint has no payload data: {output_checkpoint}.")
    return output_checkpoint, free_bytes, free_inodes


def preflight_snowball_chat(
    *,
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    expected_output_checkpoint_step: int | None = None,
    stage: str = "chat",
) -> SnowballChatPreflight:
    """Validate every static input before requesting a 64-node training allocation."""
    if stage not in STAGE_DATA:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_DATA)}.")
    if not run_id or "/" in run_id:
        raise ValueError(f"Snowball run ID must be a non-empty path-free name, got {run_id!r}.")
    stage_max = STAGES[stage].max_steps
    if stage_max is None:  # derive the ceiling from the cache this stage actually built
        stage_max = derive_epoch_steps(read_chat_cache_tokens(data_cache_path))
    if steps < 1 or steps > stage_max:
        raise ValueError(f"Snowball {stage} steps must be between 1 and {stage_max}, got {steps}.")
    if expected_output_checkpoint_step is not None and steps <= expected_output_checkpoint_step:
        raise ValueError(
            f"Snowball resume target {steps} must exceed checkpoint step {expected_output_checkpoint_step}."
        )

    resolved_init, payload_files, payload_bytes = validate_native_checkpoint_layout(
        init_checkpoint_path, expect_step=STAGES[stage].init_step
    )
    _spec = STAGES[stage]
    cache_tokens, cache_examples, cache_shards = validate_chat_cache_layout(
        data_cache_path,
        expect_tokens=_spec.cache_tokens,
        expect_examples=_spec.cache_examples,
    )
    if cache_shards != _spec.cache_shards:
        raise ValueError(f"{stage} cache has {cache_shards} finished shards; expected {_spec.cache_shards}.")
    validate_cache_provenance(data_cache_path, stage, tokenizer_path)
    tokenizer_vocab_size = _validate_tokenizer(tokenizer_path)

    mesh_factor = SNOWBALL_CHAT_REPLICA_AXIS * SNOWBALL_CHAT_EXPERT_PARALLEL * SNOWBALL_CHAT_MODEL_AXIS
    if devices % mesh_factor != 0:
        raise ValueError(f"Snowball device count {devices} is not divisible by mesh factor {mesh_factor}.")
    data_axis_size = devices // mesh_factor
    batch_shards = data_axis_size * SNOWBALL_CHAT_EXPERT_PARALLEL
    if SNOWBALL_CHAT_BATCH_SIZE % batch_shards != 0:
        raise ValueError(f"Snowball batch size {SNOWBALL_CHAT_BATCH_SIZE} is not divisible by {batch_shards} shards.")
    ici_axes, dcn_axes = vista_trainer_mesh_config().axis_shapes(num_devices=devices, num_slices=devices)
    if ici_axes != {"data": 1, "replica": 1, "model": 1, "expert": 1} or dcn_axes != {"replica_dcn": devices}:
        raise ValueError(f"Snowball Vista trainer mesh is invalid: ICI={ici_axes}, DCN={dcn_axes}.")

    output_checkpoint, output_free_bytes, output_free_inodes = _validate_output_path(
        output_path,
        input_paths=(resolved_init, data_cache_path, tokenizer_path),
        expected_output_checkpoint_step=expected_output_checkpoint_step,
    )
    run_config = snowball_chat_run_config(
        init_checkpoint_path=resolved_init,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
        stage=stage,
    )
    if run_config.trainer.trainer.initialize_from != resolved_init:
        raise ValueError(
            f"Snowball config resolved init checkpoint to {run_config.trainer.trainer.initialize_from}, "
            f"expected {resolved_init}."
        )

    return SnowballChatPreflight(
        init_checkpoint_path=resolved_init,
        init_payload_files=payload_files,
        init_payload_bytes=payload_bytes,
        cache_tokens=cache_tokens,
        cache_examples=cache_examples,
        cache_shards=cache_shards,
        tokenizer_vocab_size=tokenizer_vocab_size,
        data_axis_size=data_axis_size,
        per_device_batch_size=SNOWBALL_CHAT_BATCH_SIZE // batch_shards,
        output_free_bytes=output_free_bytes,
        output_free_inodes=output_free_inodes,
        output_checkpoint_path=output_checkpoint,
    )


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """Everything that varies between SFT stages, in one place.

    Anything hardcoded to a Chat-specific string is a latent Stage-2 bug: the
    component key, its tags, the mixture weight key and the W&B tag all have to
    move together or the run trains under a Chat identity.
    """

    messages_field: str
    component: str
    wandb_tag: str
    max_steps: int | None  # None => derive from the built cache; set => the step ceiling (multi-epoch)
    init_step: int  # EXACT step the init checkpoint must report
    cache_tokens: int | None  # exact packed-token count of this stage's cache; None => structural check only
    cache_examples: int | None  # exact packed-example count; None => structural check only
    cache_shards: int  # exact finished-shard count
    dataset_revision: str  # pinned dataset commit this cache was built from
    chat_template: str  # Delphi V0 for chat/thinking; Marin for nemotron_terminal
    fixed_steps: int | None  # a stage with a mandated budget (1888) rather than an epoch
    source_files: int  # exact number of pinned INPUT shards the cache must be built from
    # Per-stage optimizer. None keeps the Chat/Thinking AdamH at 5e-5 for the stages that trained
    # with it; the agentic stages use Ben's Stage-3 setting (marin #8225, PR #8172): both groups 5e-6.
    optimizer: GrugMoeAdamHConfig | None = None


# Ben's Stage-3 / agentic optimizer: the Chat AdamH config with both learning rates at 5e-6.
SNOWBALL_AGENTIC_OPTIMIZER = dataclasses.replace(SNOWBALL_CHAT_OPTIMIZER, learning_rate=5e-6, adam_lr=5e-6)


STAGES: dict[str, StageSpec] = {
    "chat": StageSpec(
        "conversation",
        "wildchat_386k",
        "s1_chat",
        SNOWBALL_CHAT_STEPS,
        init_step=0,
        cache_tokens=SNOWBALL_CHAT_TOKENS,
        cache_examples=SNOWBALL_CHAT_EXAMPLES,
        cache_shards=4,
        dataset_revision="46a5bb56fffd8c57c3ecc812e647990b1527c001",
        source_files=5,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        fixed_steps=None,
    ),
    # Thinking chains from the COMPLETED Chat stage. init_step is exact: accepting
    # "any step >= 1" let a step-3 smoke checkpoint pass the gate.
    "thinking": StageSpec(
        "messages",
        "nemotron_science_think",
        "s2_think",
        None,
        init_step=SNOWBALL_CHAT_STEPS,
        cache_tokens=1_321_080_491,
        cache_examples=708_920,
        cache_shards=15,
        dataset_revision="bae881d7227146ef6b93fe830a1f613e96ea1338",
        source_files=15,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        fixed_steps=None,
    ),
    # Stage 3, published as laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888.
    # Deliberately NOT modelled on Chat/Thinking: different turns column, the MARIN
    # template rather than Delphi V0, and a mandated 1,888-step budget instead of a
    # derived epoch. Chains weights-only from the completed Thinking checkpoint.
    # cache_* are filled in once the cache is built and measured.
    "nemotron_terminal": StageSpec(
        "conversations",
        "nemotron_terminal_full",
        "s3_nemotron_terminal",
        None,
        init_step=630,
        cache_tokens=6075088769,
        cache_examples=366154,
        cache_shards=19,
        dataset_revision="a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9",
        source_files=29,
        chat_template=MARIN_CHAT_TEMPLATE,
        fixed_steps=1888,
    ),
    # R2E-Gym teacher-trace SFT of the PRE-RL Snowball checkpoint (Stage 3 as published,
    # laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888, imported to native with
    # import_snowball_hf -> a step-0 init, hence init_step=0). Same turns column and Marin template
    # as nemotron_terminal (train == serve: the template is byte-identical to the export's
    # chat_template.jinja), Ben's agentic optimizer, and a multi-epoch ceiling: the cache is one
    # pinned parquet of 2,576 verifier-solved GLM-4.7 Terminus-2 traces (~53M tokens, ~26 packed
    # steps per epoch; census 2026-09-15), so the launcher passes epochs x derived epoch steps and
    # max_steps caps that at five epochs. cache_tokens/examples are validated structurally: the
    # exact totals are recorded by write-provenance, not pinned here.
    "r2egym": StageSpec(
        "conversations",
        "r2egym_glm47_solved_v1",
        "s4_r2egym",
        130,
        init_step=0,
        cache_tokens=None,
        cache_examples=None,
        cache_shards=1,
        # DCAgent/g1_clean_hybrid_scaffold_plus_r2eg_gfi_38k_glm47_traces, filtered to R2E-Gym rows
        # off the v2 idval/oodval/heldout tasks, result >= 1.0, <= 32,768 Marin tokens.
        dataset_revision="4243a8f5cd39799803a6a0d52457fa0833068566",
        source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE,
        fixed_steps=None,
        optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    # Kimi-K2.5 SWE-smith Terminus-2 traces (open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-
    # oracle_verified_120s-maxeps-32k), the first SWE-bench-facing SFT set. Same init, column, template
    # and optimizer as r2egym. The pinned parquet is built by OpenThoughts-Agent
    # data/swesmith/kimi_sft_convert.py: 4,501 train rows (Stage-3 overlap, pygments, verifier errors
    # removed; sqlparse/funcy/webargs held out whole; repo cap), <= 32,768 Marin tokens, and every
    # assistant turn rewritten to <|start_think|>...<|end_think|> so the think markers are the special
    # tokens the model emits, not the release's text tags. ~63M tokens, ~31 packed steps per epoch;
    # max_steps caps EPOCHS x epoch at about eleven epochs.
    # 2026-09-22: the AgentTrove + RST mix for the Grug 09.21 base (SNOWBALL_BASE_MODEL=grug0921).
    # 49,866 AgentTrove rows (filtered: no harness exception, every turn parses under harbor's Terminus-2
    # parser, every turn has reasoning, exp_tas/unknown dropped, >= 3 turns; per-source cap 6,500) +
    # 6,8xx RST GLM-5.3 reward-1 >= 5-turn traces (cap 3 per task) repeated twice = 63.5k rows, 755.1M
    # tokens, RST 27 % of tokens; ~371 packed steps per epoch. Built by data/build_agenttrove_rst_mix.py;
    # the pin is the sha256 of the set's SHA256SUMS.
    "agenttrove_rst": StageSpec(
        "conversations",
        "agenttrove_rst_v1",
        "s4_agenttrove_rst",
        1200,
        init_step=0,
        cache_tokens=None,
        cache_examples=None,
        cache_shards=1,
        dataset_revision="fb9c72e6635d581032b386858e5f08ea513e6520",
        source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE,
        fixed_steps=None,
        optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    "kimi_swesmith": StageSpec(
        "conversations",
        "kimi_swesmith_v1",
        "s4_kimi_swesmith",
        340,
        init_step=0,
        cache_tokens=None,
        cache_examples=None,
        cache_shards=1,
        dataset_revision="1b87a9cf78b897a2355105d47afd44d818cad25d",
        source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE,
        fixed_steps=None,
        optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    # OpenThoughts-Agent SFT-100K (open-thoughts/OpenThoughts-Agent-SFT-100K), GLM-4.7 Terminus-2 traces
    # over four slices: SWE-smith, IssueTasks, SuperUser and Tezos. Same init, column, template and
    # optimizer as kimi_swesmith. The pinned parquet is built by OpenThoughts-Agent
    # data/swesmith/ota_sft_convert.py from the 2026-09-18 audit manifest. v2 (2026-09-19, --screen
    # reference) keeps every trace the release trained on -- timed-out and mid-work rollouts, Stage-3
    # overlap and Harbor's own proactive-compaction rollouts included -- and drops only rows flagged for
    # evaluation leakage: ~94k rows, 5 % of tasks held out by task hash so a task's rollouts never
    # straddle the split. (v1, --screen strict, was 50,053 rows / 617.9M tokens / ~304 steps per epoch.)
    # About 600 packed steps per epoch at v2, so max_steps caps EPOCHS x epoch at four epochs; a subset
    # cache (the lr sweep) derives a shorter epoch under the same ceiling. The list is ONE merged shard
    # per set -- ONE source file and ONE cache shard.
    "ota": StageSpec(
        "conversations",
        "ota_sft_100k_v2",
        "s4_ota",
        2600,
        init_step=0,
        cache_tokens=None,
        cache_examples=None,
        cache_shards=1,
        dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b",
        source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE,
        fixed_steps=None,
        optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    # OTA v2 + the instruction-following slice (open-athena/nemotron-gym-if-v2-qwen3.5-122b-32k-traces @ 50b7f77,
    # 12,484 clean Terminus-2 traces / 55M tokens = 4.2 % of the mix; 477 tasks held out for the harness probe).
    # One merged shard, 1,000-row row groups; ~656 packed steps per epoch. Paired against the "ota" stage at the
    # same recipe to read whether in-format constraint following transfers (2026-09-19).
    "ota_if": StageSpec(
        "conversations",
        "ota_if_sft_v1",
        "s4_ota_if",
        2700,
        init_step=0,
        cache_tokens=None,
        cache_examples=None,
        cache_shards=1,
        dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77",
        source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE,
        fixed_steps=None,
        optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    # 2026-09-20: the IssueTasks slice is built from SWE-bench test issues (255 of 500 Verified rewritten inside
    # it), so every reportable model trains on the other three OTA slices only ("ota3"). Three corpora, one merged
    # shard each, 1,000-row row groups, same recipe, differing only in data:
    #   ota3_if         swesmith + superuser + tezos (v2) + the if-v2 clean slice            = the clean anchor
    #   ota3_if_rst     + Recursive-Task-Synthesis GLM-5.3 rollouts, >= 5 turns, 1 per task  = run 3
    #   ota3_if_rstsucc + only the reward-1 subset of those RST rows                          = run 4
    "ota3_if": StageSpec(
        "conversations", "ota3_if_sft_v1", "s4_ota3_if", 2700, init_step=0, cache_tokens=None, cache_examples=None,
        cache_shards=1, dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77", source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE, fixed_steps=None, optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    "ota3_if_rst": StageSpec(
        "conversations", "ota3_if_rst_v1", "s4_ota3_if_rst", 2700, init_step=0, cache_tokens=None, cache_examples=None,
        cache_shards=1, dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77+dd6f34cb", source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE, fixed_steps=None, optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    "ota3_if_rstsucc": StageSpec(
        "conversations", "ota3_if_rstsucc_v1", "s4_ota3_if_rstsucc", 2700, init_step=0, cache_tokens=None, cache_examples=None,
        cache_shards=1, dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77+dd6f34cb", source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE, fixed_steps=None, optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    #   rst_if          the RST rollouts alone + the same if-v2 slice (no OTA)                = the source contrast
    "rst_if": StageSpec(
        "conversations", "rst_if_sft_v1", "s4_rst_if", 2700, init_step=0, cache_tokens=None, cache_examples=None,
        cache_shards=1, dataset_revision="50b7f77+dd6f34cb", source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE, fixed_steps=None, optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    #   ota3_if_rst_fmt ota3_if_rst minus every trace with a malformed (non-JSON) action turn      = filter arm 1
    #   ota3_if_rst_beh ota3_if_rst minus degenerate traces (loops, hammering, empty batches,    = filter arm 2
    #                   repeated batches, error-dominated observations); data/rst/filter_corpus.py
    "ota3_if_rst_fmt": StageSpec(
        "conversations", "ota3_if_rst_fmt_v1", "s4_ota3_if_rst_fmt", 2700, init_step=0, cache_tokens=None, cache_examples=None,
        cache_shards=1, dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77+dd6f34cb", source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE, fixed_steps=None, optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
    "ota3_if_rst_beh": StageSpec(
        "conversations", "ota3_if_rst_beh_v1", "s4_ota3_if_rst_beh", 2700, init_step=0, cache_tokens=None, cache_examples=None,
        cache_shards=1, dataset_revision="45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77+dd6f34cb", source_files=1,
        chat_template=MARIN_CHAT_TEMPLATE, fixed_steps=None, optimizer=SNOWBALL_AGENTIC_OPTIMIZER,
    ),
}
STAGE_DATA = {k: (v.messages_field, v.component) for k, v in STAGES.items()}


def snowball_chat_data_config(*, cache_path: str, tokenizer_path: str, stage: str = "chat") -> LmDataConfig:
    spec = STAGES[stage]
    fmt = snowball_chat_format(messages_field=spec.messages_field, chat_template=spec.chat_template)
    source = UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_path, format=fmt, tags=[spec.component])
    return LmDataConfig(
        tokenizer=tokenizer_path,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        enforce_eos=True,
        auto_build_caches=False,
        components={
            spec.component: DatasetComponent(
                source=source,
                cache_dir=cache_path,
                format=fmt,
                tags=[spec.component],
                split="train",
            )
        },
        train_weights={spec.component: 1.0},
        mixture_block_size=2048,
    )


def expected_chat_steps(total_tokens: int) -> int:
    return math.ceil(total_tokens / (SNOWBALL_CHAT_SEQUENCE_LENGTH * SNOWBALL_CHAT_BATCH_SIZE))


def validate_chat_epoch(total_tokens: int) -> int:
    """STAGE 1 ONLY. Enforce the pinned 257-step Chat contract.

    Deliberately strict and deliberately not generalized: Stage 2 derives its own
    epoch (``derive_epoch_steps``) so that widening Stage 2 can never weaken this
    check. Any caller that is not the pinned WildChat cache must NOT use this.
    """
    steps = expected_chat_steps(total_tokens)
    if steps != SNOWBALL_CHAT_STEPS:
        raise ValueError(
            f"Packed WildChat cache resolves to {steps} steps, but the pinned Snowball Chat contract requires "
            f"{SNOWBALL_CHAT_STEPS}."
        )
    return steps


def derive_epoch_steps(total_tokens: int) -> int:
    """Epoch length for any stage, derived from the cache that was actually built."""
    steps = expected_chat_steps(total_tokens)
    if steps <= 0:
        raise ValueError(f"Cache resolves to {steps} steps; refusing to train on an empty cache.")
    return steps


def vista_trainer_mesh_config() -> MeshConfig:
    """Return Trainer bookkeeping for Vista's one-GPU-per-node topology.

    Grug constructs its compute mesh separately as ``(replica_dcn, data, expert,
    model) = (1, 8, 8, 1)``.  TrainerConfig only needs a compatible 64-way batch
    mesh to derive per-device parallelism before the Grug mesh exists.  On Vista,
    every Slurm rank is a one-device JAX slice, so the local ICI expert axis must
    remain size one and the 64 ranks are represented by ``replica_dcn``.
    """
    return MeshConfig(
        axes={"expert": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
    )


def run_distributed_probe(expected_devices: int) -> None:
    """Initialize JAX from Slurm and verify a cross-process collective."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    from levanter.distributed import DistributedConfig  # noqa: PLC0415

    DistributedConfig().initialize()
    if jax.device_count() != expected_devices:
        raise RuntimeError(f"Expected {expected_devices} global devices, found {jax.device_count()}.")
    if jax.local_device_count() != 1:
        raise RuntimeError(f"Expected one local device per Vista rank, found {jax.local_device_count()}.")

    local_value = jnp.asarray([jax.process_index() + 1], dtype=jnp.int32)
    reduced = jax.pmap(lambda value: lax.psum(value, "devices"), axis_name="devices")(local_value)
    expected_sum = expected_devices * (expected_devices + 1) // 2
    actual_sum = int(reduced[0])
    if actual_sum != expected_sum:
        raise RuntimeError(f"Collective returned {actual_sum}, expected {expected_sum}.")
    if jax.process_index() == 0:
        click.echo(f"global_devices={jax.device_count()}")
        click.echo(f"collective_sum={actual_sum}")
        click.echo("SNOWBALL_DISTRIBUTED_PROBE_OK")


def run_gpu_kernel_probe() -> None:
    """Validate training backends and execute the production attention kernels on one GH200."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from levanter.grug.attention import AttentionMask, gpu_fa4_cute_attention  # noqa: PLC0415

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Snowball GPU kernel probe requires a GPU backend, got {jax.default_backend()!r}.")
    cpu_devices = jax.local_devices(backend="cpu")
    if len(cpu_devices) != 1:
        raise RuntimeError(f"Snowball data loading requires one local CPU device, found {len(cpu_devices)}.")

    key = jax.random.PRNGKey(SNOWBALL_CHAT_SEED)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (1, 128, 20, 128), dtype=jnp.bfloat16) * 0.1
    k = jax.random.normal(k_key, (1, 128, 5, 128), dtype=jnp.bfloat16) * 0.1
    v = jax.random.normal(v_key, (1, 128, 5, 128), dtype=jnp.bfloat16) * 0.1
    mask = AttentionMask.causal(sliding_window=31)

    @jax.jit
    def loss_and_output(q_value, k_value, v_value):
        output = gpu_fa4_cute_attention(q_value, k_value, v_value, mask)
        return jnp.sum(output.astype(jnp.float32)), output

    (_, output), gradients = jax.value_and_grad(loss_and_output, argnums=(0, 1, 2), has_aux=True)(q, k, v)
    jax.block_until_ready((output, gradients))
    arrays = (output, *gradients)
    if not all(bool(jax.device_get(jnp.all(jnp.isfinite(array)))) for array in arrays):
        raise ValueError("Snowball FA4 forward or backward produced non-finite values.")

    click.echo(f"backend={jax.default_backend()}")
    click.echo(f"device={jax.devices()[0]}")
    click.echo(f"cpu_device={cpu_devices[0]}")
    click.echo("SNOWBALL_GPU_KERNEL_PROBE_OK")


def run_gpu_runtime_probe(*, data_cache_path: str, tokenizer_path: str, stage: str = "chat") -> None:
    """Exercise the GPU kernels and the real CPU-sharded token-cache path."""
    import jax  # noqa: PLC0415

    run_gpu_kernel_probe()
    datasets = snowball_chat_data_config(
        cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        stage=stage,
    ).train_sets(
        Axis("position", SNOWBALL_CHAT_SEQUENCE_LENGTH),
        key=jax.random.PRNGKey(SNOWBALL_CHAT_SEED),
        initial_batch_size=SNOWBALL_CHAT_BATCH_SIZE,
    )
    example = datasets[STAGES[stage].component].as_sync_dataset()[0]
    tokens = np.asarray(example.tokens)
    loss_weight = np.asarray(example.loss_weight)
    expected_shape = (SNOWBALL_CHAT_SEQUENCE_LENGTH,)
    if tokens.shape != expected_shape or loss_weight.shape != expected_shape:
        raise ValueError(
            f"Snowball cache example has token/loss shapes {tokens.shape}/{loss_weight.shape}, "
            f"expected {expected_shape}."
        )
    if not np.isfinite(loss_weight).all() or float(loss_weight.sum()) <= 0:
        raise ValueError("Snowball cache example has invalid or empty assistant loss weights.")

    click.echo(f"example_tokens={tokens.size}")
    click.echo(f"assistant_loss_tokens={int(np.count_nonzero(loss_weight))}")
    click.echo("SNOWBALL_GPU_RUNTIME_PROBE_OK")


def snowball_chat_run_config(
    *,
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    stage: str = "chat",
) -> GrugRunConfig:
    if stage not in STAGE_DATA:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_DATA)}.")
    if devices != SNOWBALL_CHAT_DEVICES:
        raise ValueError(f"Snowball Chat requires {SNOWBALL_CHAT_DEVICES} devices, got {devices}.")
    total_tokens = read_chat_cache_tokens(data_cache_path)
    # Stage 1 keeps its strict 257-step assertion; every other stage derives.
    if stage == "chat":
        full_epoch_steps = validate_chat_epoch(total_tokens)
    else:
        full_epoch_steps = derive_epoch_steps(total_tokens)
    # A stage that declares max_steps may train past one packed epoch (the mixture restarts
    # exhausted components); every other stage keeps the one-epoch ceiling.
    step_ceiling = STAGES[stage].max_steps or full_epoch_steps
    if steps > step_ceiling:
        raise ValueError(
            f"Requested {steps} steps, but the packed {stage} epoch has only {full_epoch_steps} steps "
            f"and the stage ceiling is {step_ceiling}."
        )

    run_resources = ResourceConfig.with_gpu("GH200", count=1, replicas=devices)
    # Permanent-checkpoint interval. The default 1,000 keeps only the final checkpoint of a run this
    # short, which is all a single-dose arm needs. SNOWBALL_KEEP_PER_EPOCH=1 keeps one per packed epoch
    # instead, so one run yields the whole dose curve (export + score each epoch); SNOWBALL_KEEP_EVERY
    # sets the interval outright. Temporary time-based saves are unaffected.
    keep_every = int(os.environ.get("SNOWBALL_KEEP_EVERY") or 0)
    if not keep_every and os.environ.get("SNOWBALL_KEEP_PER_EPOCH") == "1":
        keep_every = full_epoch_steps
    keep_every = keep_every or 1000
    print(f"checkpoint_keep_every={keep_every}", flush=True)
    trainer = TrainerConfig(
        id=run_id,
        seed=SNOWBALL_CHAT_SEED,
        train_batch_size=SNOWBALL_CHAT_BATCH_SIZE,
        per_device_parallelism=-1,
        num_train_steps=steps,
        mp=jmp.get_policy(SNOWBALL_CHAT_MP),
        tracker=WandbConfig(
            project="marin_moe_sft",
            name=run_id,
            group="grug-67b-a2b-sft",
            tags=["moe", "67b_a2b", "sft", STAGES[stage].wandb_tag, "seq32768", "vista-gh200"],
            mode="offline",
        ),
        use_explicit_mesh_axes=True,
        mesh=vista_trainer_mesh_config(),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=prefix_join(output_path, "checkpoints"),
            temporary_base_path=prefix_join(output_path, "checkpoints-tmp"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=[{"every": keep_every}],
            timeout=timedelta(hours=2),
        ),
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=latest_checkpoint_path(init_checkpoint_path),
    )
    optimizer = STAGES[stage].optimizer or SNOWBALL_CHAT_OPTIMIZER
    # SNOWBALL_LR overrides both AdamH learning rates for one run (the dose / overfit sweeps); the
    # stage's pinned optimizer is the default and the launcher prints the value it resolved.
    lr_override = os.environ.get("SNOWBALL_LR")
    if lr_override:
        optimizer = dataclasses.replace(optimizer, learning_rate=float(lr_override), adam_lr=float(lr_override))
    # TailSFT knobs (train.py GrugTrainerConfig, tail_filter.py), all off by default. SNOWBALL_TAIL_SCORE_OUT
    # turns the run into the forward-only reference pass over one epoch (launch with EPOCHS=1) and writes the
    # (num_docs,) .npy there; SNOWBALL_TAIL_FRACTION + SNOWBALL_TAIL_REF train the filtered objective;
    # SNOWBALL_TAIL_RAMP ramps the fraction linearly from 0 over that many steps.
    tail_fraction = float(os.environ.get("SNOWBALL_TAIL_FRACTION") or 0.0)
    tail_ref = os.environ.get("SNOWBALL_TAIL_REF") or None
    tail_ramp = int(os.environ.get("SNOWBALL_TAIL_RAMP") or 0)
    tail_score_out = os.environ.get("SNOWBALL_TAIL_SCORE_OUT") or None
    # SNOWBALL_SCHEDULE_STEPS: the lr-schedule horizon when it is longer than this run's steps (train one epoch of
    # a two-epoch cosine now, resume to the second later on the same output; train.py GrugTrainerConfig).
    schedule_steps = int(os.environ.get("SNOWBALL_SCHEDULE_STEPS") or 0) or None
    if schedule_steps is not None and schedule_steps < steps:
        raise ValueError(f"SNOWBALL_SCHEDULE_STEPS={schedule_steps} is shorter than the run's {steps} steps")
    tail_num_docs = read_chat_cache_examples(data_cache_path) if (tail_fraction > 0 or tail_score_out) else None
    if tail_score_out and steps < full_epoch_steps:
        raise ValueError(
            f"the TailSFT scoring pass needs at least one packed epoch of batches ({full_epoch_steps} steps); "
            f"got --steps {steps}. The loader samples with replacement, so launch with EPOCHS=6: the pass "
            f"stops early once every document has been scored."
        )
    return GrugRunConfig(
        model=dataclasses.replace(SNOWBALL_CHAT_MODEL_CONFIG, max_seq_len=SNOWBALL_CHAT_SEQUENCE_LENGTH),
        data=snowball_chat_data_config(cache_path=data_cache_path, tokenizer_path=tokenizer_path, stage=stage),
        resources=run_resources,
        optimizer=optimizer,
        trainer=GrugTrainerConfig(
            trainer=trainer,
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
            replica_axis_size=SNOWBALL_CHAT_REPLICA_AXIS,
            model_axis_size=SNOWBALL_CHAT_MODEL_AXIS,
            expert_axis_size=SNOWBALL_CHAT_EXPERT_PARALLEL,
            sft_weights_only_init=True,
            tail_fraction=tail_fraction,
            tail_ref_loss_path=tail_ref,
            tail_ramp_steps=tail_ramp,
            tail_num_docs=tail_num_docs,
            tail_score_out=tail_score_out,
            # SNOWBALL_FREEZE_ROUTER_BIAS=1 keeps the imported router biases fixed (the Grug 09.21 SFT recipe).
            freeze_router_bias=os.environ.get("SNOWBALL_FREEZE_ROUTER_BIAS", "0") == "1",
            schedule_steps=schedule_steps,
        ),
        eval=None,
    )


@click.group()
def main() -> None:
    """Vista Snowball chat entry point.

    The runtime defaults are applied here, before any subcommand touches JAX, so
    every path -- probes included -- gets the same GPU runtime contract.
    """
    apply_gpu_runtime_defaults()
    arm_hang_traceback_dumper()


@main.command("prepare-data")
@click.option("--parquet-glob", default=None, help="Shard pattern, expanded with recursive=True. Prefer --parquet-list.")
@click.option(
    "--parquet-list", default=None, help="Explicit newline-delimited file of shard paths -- no pattern semantics."
)
@click.option(
    "--expect-files", type=int, default=None, help="Assert the resolved shard count before tokenizing anything."
)
@click.option("--cache-path", required=True)
@click.option("--tokenizer-path", required=True)
@click.option("--messages-field", default=None, help="Turns column. Ignored when --stage is given.")
@click.option(
    "--stage",
    type=click.Choice(sorted(STAGES)),
    default=None,
    help="Take the turns column, chat template and component tag from the stage. "
    "Preferred: building a cache with the wrong template is silent at build "
    "time and only caught later by provenance validation.",
)
@click.option("--tags", default="wildchat_386k,snowball_chat", show_default=True)
@click.option(
    "--expect-steps",
    type=int,
    default=None,
    help="Assert the derived epoch length. Omit to DERIVE and report only, "
    "which is required for any dataset other than the pinned Chat cache.",
)
def prepare_data_command(
    parquet_glob: str | None,
    parquet_list: str | None,
    expect_files: int | None,
    cache_path: str,
    tokenizer_path: str,
    messages_field: str | None,
    stage: str | None,
    tags: str,
    expect_steps: int | None,
) -> None:
    if stage is not None:
        spec = STAGES[stage]
        messages_field = spec.messages_field
        chat_template = spec.chat_template
        tags = f"{spec.component},snowball_{stage}"
        click.echo(f"stage={stage} field={messages_field} component={spec.component}")
    else:
        messages_field = messages_field or "conversation"
        chat_template = None
    total_tokens = prepare_chat_cache(
        parquet_glob=parquet_glob,
        parquet_list=parquet_list,
        expect_files=expect_files,
        cache_path=cache_path,
        tokenizer_path=tokenizer_path,
        messages_field=messages_field,
        tags=tuple(t for t in tags.split(",") if t),
        chat_template=chat_template,
    )
    # Always DERIVE from the completed cache. Only assert when the caller states
    # an expectation -- validate_chat_epoch hardcodes the 257-step Chat contract
    # and would reject any other dataset by construction.
    steps = expected_chat_steps(total_tokens)
    click.echo(f"total_tokens={total_tokens}")
    click.echo(f"full_epoch_steps={steps}")
    if expect_steps is not None and steps != expect_steps:
        raise SystemExit(f"FATAL: cache resolves to {steps} steps, expected {expect_steps}.")
    click.echo("SNOWBALL_CHAT_CACHE_OK")


@main.command("write-provenance")
@click.option("--cache-path", required=True)
@click.option("--stage", type=click.Choice(sorted(STAGES)), required=True)
@click.option("--dataset-id", required=True)
@click.option("--dataset-revision", required=True)
@click.option("--parquet-glob", default=None)
@click.option(
    "--parquet-list", default=None, help="Explicit newline-delimited file of shard paths -- no pattern semantics."
)
@click.option("--expect-files", type=int, default=None, help="Assert the resolved shard count.")
@click.option("--tokenizer-path", required=True)
def write_provenance_command(
    cache_path, stage, dataset_id, dataset_revision, parquet_glob, parquet_list, expect_files, tokenizer_path
):
    """Record which pinned source produced an already-built cache."""
    shards = resolve_source_shards(parquet_glob=parquet_glob, parquet_list=parquet_list, expect_files=expect_files)
    rec = write_cache_provenance(
        cache_path=cache_path,
        stage=stage,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        shard_paths=shards,
        tokenizer_path=tokenizer_path,
    )
    click.echo(f"shards_hashed={len(rec['shards'])}")
    click.echo(f"dataset_revision={rec['dataset_revision']}")
    click.echo("SNOWBALL_PROVENANCE_OK")


@main.command("distributed-probe")
@click.option("--expected-devices", type=click.IntRange(min=2), required=True)
def distributed_probe_command(expected_devices: int) -> None:
    run_distributed_probe(expected_devices)


@main.command("gpu-kernel-probe")
def gpu_kernel_probe_command() -> None:
    run_gpu_kernel_probe()


@main.command("gpu-runtime-probe")
@click.option("--data-cache-path", required=True)
@click.option("--tokenizer-path", required=True)
def gpu_runtime_probe_command(data_cache_path: str, tokenizer_path: str) -> None:
    run_gpu_runtime_probe(data_cache_path=data_cache_path, tokenizer_path=tokenizer_path)


@main.command("preflight")
@click.option("--init-checkpoint-path", required=True)
@click.option("--data-cache-path", required=True)
@click.option("--tokenizer-path", required=True)
@click.option("--output-path", required=True)
@click.option("--run-id", required=True)
@click.option("--steps", type=click.IntRange(min=1), default=SNOWBALL_CHAT_STEPS, show_default=True)
@click.option("--devices", type=click.IntRange(min=1), default=SNOWBALL_CHAT_DEVICES, show_default=True)
@click.option("--expected-output-checkpoint-step", type=click.IntRange(min=0))
@click.option(
    "--stage",
    type=click.Choice(sorted(STAGE_DATA)),
    default="chat",
    show_default=True,
    help="Selects turns field, dataset tag, and whether the strict Chat epoch contract applies.",
)
def preflight_command(
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    expected_output_checkpoint_step: int | None,
    stage: str,
) -> None:
    report = preflight_snowball_chat(
        init_checkpoint_path=init_checkpoint_path,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
        expected_output_checkpoint_step=expected_output_checkpoint_step,
        stage=stage,
    )
    click.echo(json.dumps(dataclasses.asdict(report), sort_keys=True))
    click.echo("SNOWBALL_CHAT_PREFLIGHT_OK")


@main.command("train")
@click.option("--init-checkpoint-path", required=True)
@click.option("--data-cache-path", required=True)
@click.option("--tokenizer-path", required=True)
@click.option("--output-path", required=True)
@click.option("--run-id", required=True)
@click.option("--steps", type=click.IntRange(min=1), default=SNOWBALL_CHAT_STEPS, show_default=True)
@click.option("--devices", type=click.IntRange(min=1), default=SNOWBALL_CHAT_DEVICES, show_default=True)
@click.option("--expected-output-checkpoint-step", type=click.IntRange(min=0))
@click.option(
    "--stage",
    type=click.Choice(sorted(STAGE_DATA)),
    default="chat",
    show_default=True,
    help="Selects turns field, dataset tag, and whether the strict Chat epoch contract applies.",
)
def train_command(
    init_checkpoint_path: str,
    data_cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    steps: int,
    devices: int,
    expected_output_checkpoint_step: int | None,
    stage: str,
) -> None:
    preflight_snowball_chat(
        init_checkpoint_path=init_checkpoint_path,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
        expected_output_checkpoint_step=expected_output_checkpoint_step,
        stage=stage,
    )
    run_config = snowball_chat_run_config(
        init_checkpoint_path=init_checkpoint_path,
        data_cache_path=data_cache_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        run_id=run_id,
        steps=steps,
        devices=devices,
        stage=stage,
    )
    run_grug_local(run_config)


if __name__ == "__main__":
    main()

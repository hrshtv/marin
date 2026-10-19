# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415
"""Shared CPU fixtures for the plateau_dropping_qwen tests: synthetic Open-SWE-style trajectories, a tiny packed chat
cache built with the Qwen3 tokenizer and the marked template, and a tiny Qwen3 HF checkpoint.

The Qwen3-8B tokenizer directory comes from ``QWEN3_TOKENIZER_PATH`` (tests skip without it); nothing here loads the
real 8B weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# The single tool of the Open-SWE mini-swe-agent build (data/openswe_v12_msa/build_openswe_msa.py).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in a persistent shell. Working directory and environment variables are "
                "preserved between calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
            },
        },
    }
]
SYSTEM = (
    "You are a helpful assistant that can interact with a computer.\n\nYour response must include a THOUGHT section "
    "before your action where you\nexplain your reasoning.\n"
)
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def qwen3_tokenizer_path() -> str | None:
    path = os.environ.get("QWEN3_TOKENIZER_PATH")
    return path if path and Path(path, "tokenizer.json").is_file() else None


def msg(role, content, reasoning=None, tool_calls=None, tool_call_id=None):
    """A message as the parquet stores it: every key present, None where unused."""
    return {
        "role": role,
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
    }


def bash_call(cid: str, command: str, *, as_string: bool = True):
    args = json.dumps({"command": command}) if as_string else {"command": command}
    return {"id": cid, "type": "function", "function": {"name": "bash", "arguments": args}}


def trajectory(
    rng: np.random.Generator,
    turns: int,
    *,
    empty_reasoning_at: int | None = None,
    final_without_tool: bool = False,
    tag: str = "",
) -> list[dict]:
    """A converted Open-SWE trajectory: system, task, ``turns`` assistant turns with one bash call each, observations."""
    words = [
        "inspect",
        "the",
        "repository",
        "patch",
        "function",
        "test",
        "fails",
        "because",
        "value",
        "None",
        "fix",
        "run",
    ]
    out = [msg("system", SYSTEM), msg("user", f"Task {tag}: fix the failing test in module_{tag}.py\n")]
    for t in range(turns):
        last = t == turns - 1
        thought = " ".join(rng.choice(words, size=int(rng.integers(4, 12))))
        reasoning = (
            None
            if t == empty_reasoning_at
            else f"Turn {t}: " + " ".join(rng.choice(words, size=int(rng.integers(5, 20))))
        )
        cid = f"call_{tag}_{t}"
        if last and final_without_tool:
            out.append(msg("assistant", f"THOUGHT: {thought}. The task is complete.", reasoning))
            break
        cmd = f"echo {SUBMIT}" if last else f"grep -n '{rng.choice(words)}' module_{tag}.py | head -{t + 3}"
        out.append(msg("assistant", f"THOUGHT: {thought}", reasoning, [bash_call(cid, cmd)]))
        if not last:
            obs = "\n".join(f"{i}: {' '.join(rng.choice(words, size=4))}" for i in range(int(rng.integers(1, 4))))
            out.append(msg("tool", f"{obs}\n\n(exit_code={int(rng.integers(0, 2))})", tool_call_id=cid))
    return out


def conversations(n: int, seed: int = 0) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    convs = []
    for i in range(n):
        turns = int(rng.integers(1, 5))
        convs.append(
            trajectory(
                rng,
                turns,
                tag=str(i),
                empty_reasoning_at=1 if (i % 3 == 0 and turns > 2) else None,
                final_without_tool=(i % 4 == 1),
            )
        )
    return convs


def build_chat_cache(
    cache_path: str | Path,
    tokenizer_path: str,
    convs: list[list[dict]],
    *,
    template: str | None = None,
    shard_name: str = "shard_0",
) -> dict[str, int]:
    """A cache laid out like ``prepare-data``'s: ``<cache>/train`` holding ``input_ids`` / ``assistant_masks`` built
    by the stage's ChatProcessor (tools passed through ``chat_template_kwargs`` as in the parquet), plus
    ``.stats.json`` and the per-shard markers ``validate_chat_cache_layout`` checks."""
    from levanter.data.text.formats import ChatProcessor
    from levanter.store.cache import SerialCacheWriter
    from levanter.tokenizers import load_tokenizer

    from experiments.plateau_dropping_qwen.qwen3_template import QWEN3_THINK_TOOLS_TEMPLATE

    tok = load_tokenizer(tokenizer_path)
    proc = ChatProcessor(tok, chat_template=template or QWEN3_THINK_TOOLS_TEMPLATE, mask_user_turns=True)
    rows = proc([{"messages": c, "chat_template_kwargs": {"tools": TOOLS}} for c in convs])
    train = Path(cache_path) / "train"
    train.mkdir(parents=True, exist_ok=True)
    exemplar = {"input_ids": np.zeros((0,), np.int32), "assistant_masks": np.zeros((0,), np.int32)}
    with SerialCacheWriter(str(train), exemplar, shard_name=shard_name) as writer:
        writer.write_batch(
            [
                {"input_ids": r["input_ids"].astype(np.int32), "assistant_masks": r["assistant_masks"].astype(np.int32)}
                for r in rows
            ]
        )
    total_tokens = int(sum(len(r["input_ids"]) for r in rows))
    (train / ".stats.json").write_text(json.dumps({"total_tokens": total_tokens, "total_elements": len(rows)}))
    shard = train / shard_name
    for field in ("input_ids", "assistant_masks"):
        (shard / field).mkdir(parents=True, exist_ok=True)
        (shard / field / "marker").write_text("x")
    (shard / ".success").write_text("")
    return {"total_tokens": total_tokens, "total_elements": len(rows), "lengths": [len(r["input_ids"]) for r in rows]}


def tiny_hf_qwen3(
    path: str | Path,
    *,
    vocab_size: int = 151_936,
    dtype: str = "float32",
    seed: int = 0,
    max_position_embeddings: int = 4096,
) -> str:
    """Save a tiny random Qwen3ForCausalLM (QK-norm, GQA) as an HF checkpoint directory."""
    import torch
    from transformers import Qwen3Config as HfQwen3Config
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = HfQwen3Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=max_position_embeddings,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
    )
    model = Qwen3ForCausalLM(cfg).to(getattr(torch, dtype))
    model.save_pretrained(str(path), safe_serialization=True)
    (Path(path) / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 151643, "eos_token_id": [151645, 151643]})
    )
    return str(path)

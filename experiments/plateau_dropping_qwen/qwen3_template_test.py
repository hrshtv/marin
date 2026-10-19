# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415
"""The marked Qwen3 template renders byte-identically to Qwen3-8B's stock template, and its loss mask (Levanter's
ChatProcessor, the path prepare-data uses) covers exactly the assistant responses."""

import copy
import json

import numpy as np
import pytest

from experiments.plateau_dropping_qwen import tiny_setup as ts
from experiments.plateau_dropping_qwen.qwen3_template import (
    QWEN3_THINK_TOOLS_TEMPLATE,
    has_generation_tags,
    strip_generation_tags,
)

TOK_PATH = ts.qwen3_tokenizer_path()
pytestmark = pytest.mark.skipif(TOK_PATH is None, reason="set QWEN3_TOKENIZER_PATH to the Qwen3-8B tokenizer directory")

msg, call = ts.msg, ts.bash_call


def _edge_cases():
    rng = np.random.default_rng(7)
    cases = {
        "reasoning_every_turn": ts.trajectory(rng, 4, tag="a"),
        "empty_reasoning_turn": ts.trajectory(rng, 4, empty_reasoning_at=1, tag="b"),
        "final_without_tool_calls": ts.trajectory(rng, 3, final_without_tool=True, tag="c"),
        "single_turn": ts.trajectory(rng, 1, tag="d"),
    }
    t = ts.trajectory(rng, 3, tag="e")
    t[3]["reasoning_content"] = ""  # empty string (not None) reasoning mid-trajectory
    t[2]["content"] = "\n\nTHOUGHT: leading newlines"
    cases["empty_string_reasoning_and_leading_newlines"] = t
    cases["think_in_content"] = [
        msg("system", ts.SYSTEM),
        msg("user", "Do it."),
        msg("assistant", "<think>\ninline reasoning\n</think>\n\nTHOUGHT: go", None, [call("c0", "ls")]),
        msg("tool", "x\n\n(exit_code=0)", tool_call_id="c0"),
        msg("assistant", "done", "final"),
    ]
    cases["two_calls_dict_args_and_consecutive_tools"] = [
        msg("system", ts.SYSTEM),
        msg("user", "Two at once."),
        msg("assistant", "", "both", [call("c0", "ls", as_string=False), call("c1", "pwd", as_string=False)]),
        msg("tool", "a\n\n(exit_code=0)", tool_call_id="c0"),
        msg("tool", "/w\n\n(exit_code=0)", tool_call_id="c1"),
        msg("assistant", "THOUGHT: submit", "r", [call("c2", f"echo {ts.SUBMIT}")]),
    ]
    cases["second_user_turn_strips_earlier_reasoning"] = [
        msg("system", ts.SYSTEM),
        msg("user", "First task."),
        msg("assistant", "first answer", "old reasoning"),
        msg("user", "Second task."),
        msg("assistant", "THOUGHT: x", "new reasoning", [call("c0", "ls")]),
        msg("tool", "x\n\n(exit_code=0)", tool_call_id="c0"),
        msg("assistant", "done", "last reasoning"),
    ]
    cases["no_system_message"] = [msg("user", "Hi."), msg("assistant", "Hello.", "greet")]
    return cases


CASES = _edge_cases()


@pytest.fixture(scope="module")
def hf_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOK_PATH, local_files_only=True)


@pytest.fixture(scope="module")
def lev_tok():
    from levanter.tokenizers import load_tokenizer

    return load_tokenizer(TOK_PATH)


def test_template_has_markers_and_stock_has_none(hf_tok):
    assert has_generation_tags(QWEN3_THINK_TOOLS_TEMPLATE)
    assert not has_generation_tags(hf_tok.chat_template)
    assert not has_generation_tags(strip_generation_tags(QWEN3_THINK_TOOLS_TEMPLATE))
    assert QWEN3_THINK_TOOLS_TEMPLATE.count("{%- generation %}") == 1


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("tools", [ts.TOOLS, None], ids=["tools", "no_tools"])
@pytest.mark.parametrize("gen_prompt", [False, True], ids=["plain", "gen_prompt"])
def test_renders_byte_identical_to_stock(hf_tok, name, tools, gen_prompt):
    conv = copy.deepcopy(CASES[name])
    stock = hf_tok.apply_chat_template(conv, tools=tools, tokenize=False, add_generation_prompt=gen_prompt)
    marked = hf_tok.apply_chat_template(
        conv, tools=tools, tokenize=False, add_generation_prompt=gen_prompt, chat_template=QWEN3_THINK_TOOLS_TEMPLATE
    )
    stripped = hf_tok.apply_chat_template(
        conv,
        tools=tools,
        tokenize=False,
        add_generation_prompt=gen_prompt,
        chat_template=strip_generation_tags(QWEN3_THINK_TOOLS_TEMPLATE),
    )
    assert marked.encode() == stock.encode()
    assert stripped.encode() == stock.encode()


def _expected_mask(hf_tok, text: str) -> list[int]:
    """Tokens whose characters lie in an assistant response: after '<|im_start|>assistant\\n' through '<|im_end|>'."""
    enc = hf_tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, head, pos = [], "<|im_start|>assistant\n", 0
    while (i := text.find(head, pos)) >= 0:
        start = i + len(head)
        end = text.index("<|im_end|>", start) + len("<|im_end|>")
        spans.append((start, end))
        pos = end
    mask = []
    for a, b in enc["offset_mapping"]:
        mask.append(int(any(s <= a and b <= e for s, e in spans)))
        assert not any(a < s < b or a < e < b for s, e in spans), "a token straddles a response boundary"
    return mask


@pytest.mark.parametrize("name", sorted(CASES))
def test_chat_processor_mask_covers_exactly_assistant_responses(hf_tok, lev_tok, name):
    from levanter.data.text.formats import ChatProcessor

    conv = copy.deepcopy(CASES[name])
    proc = ChatProcessor(lev_tok, chat_template=QWEN3_THINK_TOOLS_TEMPLATE, mask_user_turns=True)
    # the parquet layout: messages plus chat_template_kwargs carrying the single bash tool
    row = proc([{"messages": conv, "chat_template_kwargs": {"tools": ts.TOOLS}}])[0]
    stock = hf_tok.apply_chat_template(conv, tools=ts.TOOLS, tokenize=False)
    ids = row["input_ids"].tolist()
    assert ids == hf_tok(stock, add_special_tokens=False)["input_ids"], "ids are the stock rendering's"
    mask = row["assistant_masks"].tolist()
    assert mask == _expected_mask(hf_tok, stock)
    masked_text = hf_tok.decode([t for t, m in zip(ids, mask, strict=True) if m])
    n_assistant = sum(m["role"] == "assistant" for m in conv)
    assert masked_text.count("<|im_end|>") == n_assistant
    for m in conv:  # nothing from system / user / tool turns is supervised
        if m["role"] in ("system", "user", "tool") and m["content"]:
            assert m["content"].strip() not in masked_text
    assert "<tool_response>" not in masked_text and "# Tools" not in masked_text
    # HF's own {% generation %} handling agrees
    hf = hf_tok.apply_chat_template(
        conv,
        tools=ts.TOOLS,
        chat_template=QWEN3_THINK_TOOLS_TEMPLATE,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    assert hf["input_ids"] == ids and hf["assistant_masks"] == mask


def test_stage_spec_and_cache_format():
    """The new stage: Qwen3 template, the Open-SWE messages field and pinned revision, 32k; prepare-data's processor."""
    from levanter.tokenizers import load_tokenizer

    from experiments.june_tpu_67b_a2b.moe import vista_snowball_chat as vsc

    spec = vsc.STAGES["openswe_v12_msa_qwen3_32k"]
    ref = vsc.STAGES["openswe_v12_msa_g0921"]
    assert spec.chat_template == QWEN3_THINK_TOOLS_TEMPLATE
    assert (spec.messages_field, spec.dataset_revision, spec.source_files, spec.cache_shards) == (
        ref.messages_field,
        ref.dataset_revision,
        ref.source_files,
        ref.cache_shards,
    )
    fmt = vsc.snowball_chat_format(messages_field=spec.messages_field, chat_template=spec.chat_template)
    assert fmt.mask_user_turns and fmt.chat_template_kwargs == "chat_template_kwargs"
    proc = fmt.build_preprocessor(load_tokenizer(TOK_PATH))
    out = proc([{"messages": CASES["reasoning_every_turn"], "chat_template_kwargs": {"tools": ts.TOOLS}}])
    assert out[0]["assistant_masks"].sum() > 0
    # parquet round trip of the message structs keeps the rendering (None fields, JSON-string arguments)
    import pyarrow as pa

    table = pa.Table.from_pylist([{"messages": CASES["empty_reasoning_turn"]}])
    back = table.to_pylist()[0]["messages"]
    assert json.dumps(back, sort_keys=True) == json.dumps(CASES["empty_reasoning_turn"], sort_keys=True)

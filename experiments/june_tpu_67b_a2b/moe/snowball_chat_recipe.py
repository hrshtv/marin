# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball Base-to-Chat model and optimizer contract."""

import dataclasses
import math
import os

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig


@dataclasses.dataclass(frozen=True)
class BaseModel:
    """The two facts about a starting checkpoint that the SFT pipeline hard-codes.

    Import builds the native init from the Levanter config rather than the checkpoint's
    ``config.json``, training reuses that config, and export writes it back out -- so a base
    with a different ``qk_mult`` trained with the wrong value would be silently mis-scaled
    at every one of those points. The tokenizer hash guards the same lineage on the data
    side (see ``vista_snowball_chat._validate_tokenizer``).
    """

    name: str
    qk_mult: float
    tokenizer_sha256: str


_SNOWBALL_QK_MULT = 1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0)

BASE_MODELS = {
    # snowball-s3-nemotron-terminal-step1888: config.json qk_mult 1.5703..., max_position 32,768.
    "snowball": BaseModel(
        "snowball", _SNOWBALL_QK_MULT, "881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2"
    ),
    # open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21: same architecture and vocabulary,
    # config.json qk_mult 1.75 and max_position 262,144 (training stays at 32,768; the rope
    # settings are otherwise identical). Its tokenizer.json differs from Snowball's only in
    # naming the reserved ids 128005/128011 `<tool_call>`/`</tool_call>`.
    "grug0921": BaseModel("grug0921", 1.75, "6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45"),
}
BASE_MODEL_ENV = "SNOWBALL_BASE_MODEL"


def selected_base_model() -> BaseModel:
    """The base checkpoint this process is configured for; unknown names fail loudly."""
    name = os.environ.get(BASE_MODEL_ENV, "snowball")
    if name not in BASE_MODELS:
        raise ValueError(f"{BASE_MODEL_ENV}={name!r}; expected one of {sorted(BASE_MODELS)}")
    return BASE_MODELS[name]

SNOWBALL_CHAT_SEQUENCE_LENGTH = 32_768
SNOWBALL_CHAT_BATCH_SIZE = 64
SNOWBALL_CHAT_STEPS = 257
SNOWBALL_CHAT_TOKENS = 538_877_811
SNOWBALL_CHAT_EXAMPLES = 385_700
SNOWBALL_CHAT_DEVICES = 64
SNOWBALL_CHAT_EXPERT_PARALLEL = 8
SNOWBALL_CHAT_REPLICA_AXIS = 1
SNOWBALL_CHAT_MODEL_AXIS = 1
SNOWBALL_CHAT_SEED = 0
SNOWBALL_CHAT_MP = "params=float32,compute=bfloat16,output=bfloat16"
SNOWBALL_NATIVE_PARAMETERS = 67_078_882_816

_MODEL_BASE = MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=65_536)

# Resolved once at import from SNOWBALL_BASE_MODEL, so import, training and export all see the
# same qk_mult for the selected base.
SNOWBALL_CHAT_MODEL_CONFIG = dataclasses.replace(
    _MODEL_BASE,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=selected_base_model().qk_mult,
    max_seq_len=SNOWBALL_CHAT_SEQUENCE_LENGTH,
    attention_implementation="gpu_fa4_cute",
    ce_implementation="batched_xla",
)

SNOWBALL_CHAT_OPTIMIZER = GrugMoeAdamHConfig(
    learning_rate=5e-5,
    adam_lr=5e-5,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=0.03,
    lr_schedule="cosine",
)

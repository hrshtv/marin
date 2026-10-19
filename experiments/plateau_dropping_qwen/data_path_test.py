# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415
"""Data path: token keys are stable under packing and shuffling (documents are never split or dropped when they fit),
and the Qwen trainer's loader yields exactly the rows Grug's loader yields at every step."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import asyncio

import jax
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from levanter.schedule import BatchSchedule

from experiments.june_tpu_67b_a2b.moe import token_track as tt
from experiments.plateau_dropping_qwen import qwen_sft as qs
from experiments.plateau_dropping_qwen import tiny_setup as ts

TOK_PATH = ts.qwen3_tokenizer_path()
pytestmark = pytest.mark.skipif(TOK_PATH is None, reason="set QWEN3_TOKENIZER_PATH to the Qwen3-8B tokenizer directory")
STAGE = "openswe_v12_msa_qwen3_32k"
NUM_DOCS = 14


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    info = ts.build_chat_cache(root, TOK_PATH, ts.conversations(NUM_DOCS, seed=3))
    from levanter.store.cache import TreeCache

    exemplar = {"input_ids": np.zeros((0,), np.int32), "assistant_masks": np.zeros((0,), np.int32)}
    tc = TreeCache.load(str(root / "train"), exemplar)
    docs = [tc[i] for i in range(NUM_DOCS)]
    return {"path": str(root), "docs": docs, **info}


def rows(seq_len, seed, cache, n_rows):
    ds = qs.build_grug_dataset(
        data_cache_path=cache["path"],
        tokenizer_path=TOK_PATH,
        stage=STAGE,
        seq_len=seq_len,
        batch_schedule=BatchSchedule(4),
        seed=seed,
        data_seed=None,
    )
    exs = asyncio.run(ds.get_batch(list(range(n_rows))))
    return [(np.asarray(e.tokens), np.asarray(e.loss_weight), np.asarray(e.attn_mask.segment_ids[0])) for e in exs]


def expected_keys(cache, max_len):
    """Every supervised position of every document: offset t of doc d is trained iff token t + 1 is assistant."""
    out = {}
    for d, doc in enumerate(cache["docs"]):
        ids, m = np.asarray(doc["input_ids"]), np.asarray(doc["assistant_masks"])
        for t in range(len(ids) - 1):
            if m[t + 1]:
                out[d * max_len + t] = (int(ids[t]), int(ids[t + 1]))
    return out


def test_keys_stable_under_packing_and_shuffle(cache):
    lengths = cache["lengths"]
    longest = int(max(lengths))
    max_len = 2 * longest
    want = expected_keys(cache, max_len)
    hist = tt.TokenHistory(NUM_DOCS, max_len)
    # seq_len == the longest document: that document fills a row exactly and must not be split or truncated
    for seq_len, seed in ((longest, 0), (longest, 1), (2 * longest, 0), (2 * longest, 5)):
        seen = {}
        for tok, w, seg in rows(seq_len, seed, cache, 3 * NUM_DOCS):
            keys = hist.keys(seg[None], w[None])[0]
            target = np.roll(tok, -1)
            for s in np.nonzero(keys >= 0)[0]:
                k = int(keys[s])
                pair = (int(tok[s]), int(target[s]))
                assert seen.setdefault(k, pair) == pair, "a key maps to one (input, target) token pair"
        assert seen == want, f"seq_len={seq_len} seed={seed}: every supervised token of every document, same ids"
    # the 32k contract: a document of exactly seq_len tokens is one whole row
    doc = int(np.argmax(lengths))
    found = False
    for tok, _w, seg in rows(longest, 0, cache, 3 * NUM_DOCS):
        if np.all(seg == doc):
            found = True
            assert np.array_equal(tok, np.asarray(cache["docs"][doc]["input_ids"]))
    assert found


def test_docs_longer_than_seq_len_are_truncated_not_dropped(cache):
    """(Only for completeness: the 32k parquet keeps rows <= 32,768.) A too-long document keeps its prefix."""
    longest = int(max(cache["lengths"]))
    seq_len = longest - 7
    doc = int(np.argmax(cache["lengths"]))
    hits = [r for r in rows(seq_len, 0, cache, 3 * NUM_DOCS) if np.all(r[2] == doc)]
    assert hits and np.array_equal(hits[0][0], np.asarray(cache["docs"][doc]["input_ids"])[:seq_len])


def _gather(x):
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


@pytest.mark.parametrize("data_seed", [None, 5])
def test_trainer_loader_yields_grugs_rows(cache, data_seed):
    from levanter.grug.sharding import compact_grug_mesh

    from experiments.june_tpu_67b_a2b.moe import train as grug_train
    from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import snowball_chat_data_config

    seq_len, batch, steps = int(max(cache["lengths"])) + 64, 4, 9
    # Grug: run_grug_local's key derivation, build_train_dataset and build_train_loader
    data_key, _ = jax.random.split(jax.random.PRNGKey(0), 2)
    if data_seed is not None:
        data_key = jax.random.PRNGKey(data_seed)
    sched = BatchSchedule(batch)
    g_mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=None, model_axis_size=1)
    with set_mesh(g_mesh):
        g_ds = grug_train.build_train_dataset(
            snowball_chat_data_config(cache_path=cache["path"], tokenizer_path=TOK_PATH, stage=STAGE),
            max_seq_len=seq_len,
            batch_schedule=sched,
            key=data_key,
        )
        g_it = iter(grug_train.build_train_loader(g_ds, batch_schedule=sched, mesh=g_mesh).iter_from_step(0))
        grug = [
            tuple(_gather(a) for a in (b.tokens, b.loss_weight, b.attn_mask.segment_ids[0]))
            for b in (next(g_it) for _ in range(steps))
        ]
    # Qwen trainer: the same calls run_training makes
    cfg = qs.QwenSftConfig(
        hf_checkpoint="x",
        data_cache_path=cache["path"],
        tokenizer_path=TOK_PATH,
        output_path="unused",
        run_id="r",
        steps=steps,
        stage=STAGE,
        seq_len=seq_len,
        batch_size=batch,
        data_seed=data_seed,
        tracker="noop",
        allow_cpu=True,
    )
    tcfg = qs.build_trainer_config(cfg)
    mesh = tcfg.device_mesh
    import haliax as hax

    Pos = hax.Axis("position", seq_len)
    with set_mesh(mesh), hax.axis_mapping(tcfg.parameter_axis_mapping):
        ds = qs.build_grug_dataset(
            data_cache_path=cache["path"],
            tokenizer_path=TOK_PATH,
            stage=STAGE,
            seq_len=seq_len,
            batch_schedule=tcfg.batch_schedule,
            seed=cfg.seed,
            data_seed=cfg.data_seed,
        )
        loader = qs.make_train_loader(qs.named_dataset(ds, Pos), trainer_config=tcfg, mesh=mesh)
        it = iter(loader.iter_from_step(0))
        mine = [
            tuple(_gather(a.array) for a in (b.tokens, b.loss_weight, b.attn_mask.segment_ids[0]))
            for b in (next(it) for _ in range(steps))
        ]
        resumed = iter(loader.iter_from_step(5))
        mine_resumed = [
            tuple(_gather(a.array) for a in (b.tokens, b.loss_weight, b.attn_mask.segment_ids[0]))
            for b in (next(resumed) for _ in range(steps - 5))
        ]
    epochs = len({int(s) for g in grug for s in np.unique(g[2]) if s >= 0})
    assert epochs == NUM_DOCS, "the steps cover every document (and wrap around the epoch)"
    for step, (g, q) in enumerate(zip(grug, mine, strict=True)):
        for a, b, name in zip(g, q, ("tokens", "loss_weight", "segment_ids"), strict=True):
            assert a.shape == b.shape and np.array_equal(a, b), (step, name)
    for g, q in zip(grug[5:], mine_resumed, strict=True):
        for a, b in zip(g, q, strict=True):
            assert np.array_equal(a, b), "resuming at a step yields the rows of that step"

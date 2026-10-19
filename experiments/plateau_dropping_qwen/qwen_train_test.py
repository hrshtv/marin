# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415
"""End to end on CPU with a tiny Qwen3: HF -> Levanter -> HF is bit-exact; the training loop runs with Plateau
Dropping, saves the token history beside its checkpoints, resumes to the same result as an uninterrupted run, and
exports an HF checkpoint carrying the stock chat template; preflight accepts the fixture."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from experiments.plateau_dropping_qwen import qwen_sft as qs
from experiments.plateau_dropping_qwen import tiny_setup as ts
from experiments.plateau_dropping_qwen.qwen3_template import has_generation_tags

TOK_PATH = ts.qwen3_tokenizer_path()
pytestmark = pytest.mark.skipif(TOK_PATH is None, reason="set QWEN3_TOKENIZER_PATH to the Qwen3-8B tokenizer directory")
STAGE = "openswe_v12_msa_qwen3_32k"


def _safetensors(path):
    from safetensors.numpy import load_file

    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


def _torch_tensors(path):
    from safetensors.torch import load_file

    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_hf_levanter_hf_round_trip_is_bit_exact(tmp_path, dtype):
    import haliax as hax
    import jax.numpy as jnp
    import jmp
    import torch
    from haliax.partitioning import set_mesh
    from levanter.tokenizers import load_tokenizer
    from levanter.utils.mesh import create_mesh_from_axis_specs
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    src = ts.tiny_hf_qwen3(tmp_path / "src", dtype=dtype)
    converter = qs.make_converter(src, TOK_PATH)
    assert converter.Vocab.size == 151_936, "tokenizer padded to the model vocab"
    model_config = qs.qwen_model_config(
        converter.default_hf_config, attn_backend="vanilla", flash_block_size=None, gradient_checkpointing=True
    )
    mesh_cfg = qs.trainer_mesh_config()
    ici, dcn = mesh_cfg.axis_shapes(jax.device_count(), 1)
    mesh = create_mesh_from_axis_specs(ici_axes=ici, dcn_axes=dcn)
    mp = jmp.get_policy(qs.DEFAULT_MP)
    param_map = mesh_cfg.resolved_param_mapping
    with set_mesh(mesh), hax.axis_mapping(param_map):
        # the trainer's init: load at compute dtype, keep fp32 params
        model = converter.load_pretrained(
            model_config.model_type,
            ref=src,
            config=model_config,
            axis_mapping=param_map,
            dtype=mp.compute_dtype if dtype == "bfloat16" else jnp.float32,
        )
        model = hax.named_jit(mp.cast_to_param, param_map)(model)
        assert qs.tree_bytes_report(model)["device0_bytes"] <= qs.tree_bytes_report(model)["total_bytes"] / 4 + 4096
        export = converter.replaced(tokenizer=load_tokenizer(TOK_PATH))
        qs.export_hf(
            model,
            converter=export,
            path=str(tmp_path / "out"),
            dtype=jnp.dtype(dtype),
            chat_template=qs.stock_chat_template(TOK_PATH),
            generation_config_src=str(Path(src) / "generation_config.json"),
        )
    a, b = _torch_tensors(src), _torch_tensors(tmp_path / "out")
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    cfg_out = json.loads((tmp_path / "out" / "config.json").read_text())
    assert cfg_out["vocab_size"] == 151_936 and cfg_out["model_type"] == "qwen3"
    tok_out = AutoTokenizer.from_pretrained(str(tmp_path / "out"))
    tok_ref = AutoTokenizer.from_pretrained(TOK_PATH)
    assert len(tok_out) == len(tok_ref) == 151_669, "export keeps the unpadded tokenizer"
    assert tok_out.chat_template == tok_ref.chat_template and not has_generation_tags(tok_out.chat_template)
    assert (tmp_path / "out" / "generation_config.json").is_file()
    ids = torch.tensor([[151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]])
    with torch.no_grad():
        la = Qwen3ForCausalLM.from_pretrained(src).eval()(ids).logits
        lb = Qwen3ForCausalLM.from_pretrained(str(tmp_path / "out")).eval()(ids).logits
    assert torch.equal(la, lb)


@pytest.fixture(scope="module")
def fixture_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    ts.build_chat_cache(root / "cache", TOK_PATH, ts.conversations(24, seed=1))
    ts.tiny_hf_qwen3(root / "hf", dtype="bfloat16")
    return root


def _cfg(root, out, steps, **kw):
    base = dict(
        hf_checkpoint=str(root / "hf"),
        data_cache_path=str(root / "cache"),
        tokenizer_path=TOK_PATH,
        output_path=str(root / out),
        run_id="tiny",
        steps=steps,
        stage=STAGE,
        seq_len=1024,
        batch_size=4,
        track_tol=0.05,
        track_max_len=1024,
        keep_every=2,
        tracker="noop",
        allow_cpu=True,
        log_dir=str(root / "logs"),
        attn_backend="jax_flash",
        flash_block_size=128,
        hf_save_dtype="float32",
        learning_rate=1e-3,
        schedule_steps=4,
        trace_num_docs=3,
        trace_out=str(root / out / "trace"),
    )
    base.update(kw)
    return qs.QwenSftConfig(**base)


def _history(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.timeout(300)
def test_train_resume_matches_uninterrupted_run(fixture_dirs):
    root = fixture_dirs
    full = qs.run_training(_cfg(root, "full", 4))
    assert full["step"] == 4 and not full["nan_stop"]
    first = qs.run_training(_cfg(root, "resumed", 2))
    assert first["step"] == 2
    h2 = _history(root / "resumed" / "checkpoints" / "token-history.npz")
    assert int(h2["step"]) == 2 and int(h2["count"].max()) >= 1
    assert not first["history_restored"]
    second = qs.run_training(_cfg(root, "resumed", 4))
    assert second["history_restored"], "the token history saved at step 2 is restored with the step-2 checkpoint"
    assert second["step"] == 4 and second["loss"] == full["loss"]
    hf_, hr = (_history(root / d / "checkpoints" / "token-history.npz") for d in ("full", "resumed"))
    assert int(hf_["step"]) == int(hr["step"]) == 4
    for k in ("prev1", "prev2", "count", "dropped"):
        assert np.array_equal(hf_[k], hr[k], equal_nan=True), k
    assert hf_["dropped"].any(), "the rule dropped tokens in this run"
    a, b = _safetensors(root / "full" / "hf"), _safetensors(root / "resumed" / "hf")
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    tok_cfg = json.loads((root / "full" / "hf" / "tokenizer_config.json").read_text())
    assert tok_cfg["chat_template"] == qs.stock_chat_template(TOK_PATH)

    # the trace: a resumed run appends shards instead of overwriting the first run's, and records the same visits
    def trace_rows(d):
        shards = sorted((root / d / "trace").glob("trace-*.npz"))
        rows = [np.load(f) for f in shards]
        return len(shards), {k: np.concatenate([r[k] for r in rows]) for k in rows[0].files}

    n_full, t_full = trace_rows("full")
    n_res, t_res = trace_rows("resumed")
    assert n_full == 1 and n_res == 2
    order = lambda t: np.lexsort((t["offset"], t["doc"], t["step"]))  # noqa: E731
    for k in t_full:
        assert np.array_equal(t_full[k][order(t_full)], t_res[k][order(t_res)]), k


def _write_provenance(root, tmp_path):
    from experiments.june_tpu_67b_a2b.moe import vista_snowball_chat as vsc

    src = tmp_path / "train-00000-of-00001.parquet"
    src.write_bytes(b"stand-in source shard")
    vsc.write_cache_provenance(
        cache_path=str(root / "cache"),
        stage=STAGE,
        dataset_id="nvidia/Open-SWE-Traces",
        dataset_revision=vsc.STAGES[STAGE].dataset_revision,
        shard_paths=[str(src)],
        tokenizer_path=TOK_PATH,
    )


def test_preflight_accepts_fixture(fixture_dirs, tmp_path):
    root = fixture_dirs
    _write_provenance(root, tmp_path)
    cfg = _cfg(root, "preflight-out", 4, output_path=str(tmp_path / "out"))
    report = qs.run_preflight(cfg, expect_devices=jax.device_count())
    assert report["devices"] == 4 and report["mesh"]["data"] * report["mesh"]["replica_dcn"] == 4
    assert (
        report["param_bytes_fp32_per_device"] * 4
        <= report["param_bytes_fp32_total"] + 4 * report["replicated_param_bytes"]
    )
    assert report["cache_examples"] == 24 and report["tokenizer_len"] == 151_669
    with pytest.raises(ValueError, match="steps"):
        qs.run_preflight(_cfg(root, "x", 10_000, output_path=str(tmp_path / "out")))
    with pytest.raises(ValueError, match="observe_only"):
        qs.run_preflight(_cfg(root, "x", 4, track_observe_only=True, output_path=str(tmp_path / "out")))


def test_cli_defaults_env_and_flags(fixture_dirs, tmp_path, monkeypatch):
    """CLI flag > SNOWBALL_* environment > default; the user's optimizer / batch defaults."""
    from click.testing import CliRunner

    root = fixture_dirs
    _write_provenance(root, tmp_path)
    seen = {}

    def fake_preflight(cfg, **kw):
        seen["cfg"] = cfg
        return {"ok": True}

    monkeypatch.setattr(qs, "run_preflight", fake_preflight)
    args = [
        "preflight",
        "--hf-checkpoint",
        str(root / "hf"),
        "--data-cache-path",
        str(root / "cache"),
        "--tokenizer-path",
        TOK_PATH,
        "--output-path",
        str(tmp_path / "o"),
        "--run-id",
        "r",
        "--allow-cpu",
        "--track-drop-weight",
        "0.25",
    ]
    env = {
        "SNOWBALL_STEPS": "7",
        "SNOWBALL_TRACK_TOL": "0.05",
        "SNOWBALL_TRACK_DROP_WEIGHT": "0.5",
        "SNOWBALL_TRACK_REENTER": "rise",
        "SNOWBALL_DATA_SEED": "3",
        "SNOWBALL_SCHEDULE_STEPS": "14",
        "SNOWBALL_KEEP_EVERY": "",
        "SNOWBALL_LR": "",
        "SNOWBALL_SEQ_LEN": "",
        "SNOWBALL_TRACK_MAX_LEN": "",
    }
    result = CliRunner().invoke(qs.main, args, env=env)
    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert (cfg.steps, cfg.track_tol, cfg.track_drop_weight, cfg.track_reenter) == (7, 0.05, 0.25, "rise")
    assert (cfg.data_seed, cfg.schedule_steps, cfg.keep_every) == (3, 14, 1000)
    assert (cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.epsilon, cfg.max_grad_norm) == (
        5e-5,
        0.01,
        0.9,
        0.95,
        1e-8,
        1.0,
    )
    assert (cfg.warmup, cfg.min_lr_ratio, cfg.lr_schedule) == (0.03, 0.1, "cosine")
    assert (cfg.batch_size, cfg.seq_len, cfg.track_max_len, cfg.z_loss_weight) == (128, 32768, 32768, 1e-4)
    assert cfg.stage == STAGE and cfg.track_floor == 0.05 and cfg.track_observe_only is False


def test_lr_schedule_warmup_then_cosine_to_ten_percent():
    for horizon, warm in ((10, 1), (33, 1), (34, 1), (100, 3), (1000, 30), (1, 1)):
        assert qs.warmup_steps_for(0.03, horizon) == warm
        cfg = qs.QwenSftConfig(
            hf_checkpoint="x",
            data_cache_path="x",
            tokenizer_path="x",
            output_path="x",
            run_id="r",
            steps=horizon,
            schedule_steps=None,
        )
        sched = qs.build_optimizer_config(cfg).lr_scheduler(qs.schedule_horizon(cfg))
        lr = [float(sched(i)) for i in range(horizon + 1)]
        assert lr[warm] == pytest.approx(5e-5, rel=1e-6), (horizon, lr[: warm + 2])
        if warm < horizon:
            assert lr[0] < 5e-5 and lr[horizon] == pytest.approx(5e-6, rel=1e-4)
            assert all(a >= b - 1e-12 for a, b in zip(lr[warm:-1], lr[warm + 1 :], strict=True))

    longer = qs.QwenSftConfig(
        hf_checkpoint="x",
        data_cache_path="x",
        tokenizer_path="x",
        output_path="x",
        run_id="r",
        steps=50,
        schedule_steps=100,
    )
    sched = qs.build_optimizer_config(longer).lr_scheduler(qs.schedule_horizon(longer))
    assert float(sched(50)) > 2e-5, "a run shorter than its schedule stops mid-cosine"
    with pytest.raises(ValueError):
        qs.schedule_horizon(
            qs.QwenSftConfig(
                hf_checkpoint="x",
                data_cache_path="x",
                tokenizer_path="x",
                output_path="x",
                run_id="r",
                steps=50,
                schedule_steps=40,
            )
        )

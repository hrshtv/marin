import dataclasses
import os
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import SNOWBALL_CHAT_MODEL_CONFIG
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE

# Env overrides exist only so this can run on a filesystem/topology other than
# the reference CoreWeave one. Unset, every value is the #8225 reference, so the
# original behaviour is preserved byte for byte -- including the chat template
# metadata, which the reference writes as MARIN_CHAT_TEMPLATE and overrides with
# Delphi V0 at serve/eval time.
CHECKPOINT = os.environ.get(
    "SNOWBALL_EXPORT_CHECKPOINT",
    "s3://marin-us-east-02a/marin/grug/"
    "snowball_step105149_sft_s2_thinking/"
    "2026.08.13.1/checkpoints/step-630/",
)
OUTPUT = os.environ.get(
    "SNOWBALL_EXPORT_OUTPUT",
    "s3://marin-us-east-02a/marin/exports/grug/"
    "snowball_step105149_sft_s2_thinking/2026.08.13.1/step-630/hf-bf16-vllm/",
)
# marin-community/marin-tokenizer is byte-identical to the pinned Snowball
# tokenizer (sha256 881c9c36...), verified on Vista; a local path avoids needing
# the network from a compute node.
TOKENIZER = os.environ.get("SNOWBALL_EXPORT_TOKENIZER", "marin-community/marin-tokenizer")
# One GPU per host means the params cannot be replicated: 134 GB bf16 exceeds a
# 120 GB GH200. Shard over an explicit expert axis instead.
EXPERT_AXIS = int(os.environ.get("SNOWBALL_EXPORT_EXPERT_AXIS", "1"))
REPLICA_AXIS = os.environ.get("SNOWBALL_EXPORT_REPLICA_AXIS")

# The architecture the export writes into config.json must be the one the checkpoint was
# imported and trained with, so it comes from the recipe (and from there, from the selected
# base model) rather than being restated here.
model_config = SNOWBALL_CHAT_MODEL_CONFIG
model_dict = dataclasses.asdict(model_config)
vendored_config = draccus.decode(VendoredGrugModelConfig, model_dict)
main_fields = {field.name for field in dataclasses.fields(GrugModelConfig)}
main_config = draccus.decode(GrugModelConfig, {key: value for key, value in model_dict.items() if key in main_fields})

# The reference runs under Fray/Iris, which initialises JAX distributed for it.
# Under bare srun each task sees only its own GPU (global_device_count == 1) and
# the expert mesh cannot be built. Initialise from Slurm when running multi-task;
# a single-process run is left exactly as the reference behaves.
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    from levanter.distributed import DistributedConfig

    DistributedConfig().initialize()
    print(f"EXPORT distributed: {jax.process_count()} processes, "
          f"{jax.device_count()} global devices", flush=True)

mesh = compact_grug_mesh(
    expert_axis_size=EXPERT_AXIS,
    replica_axis_size=int(REPLICA_AXIS) if REPLICA_AXIS else None,
)
_rep = REPLICA_AXIS or "<processes>"
print(f"EXPORT mesh expert_axis={EXPERT_AXIS} replica_axis={_rep}", flush=True)
print(f"EXPORT checkpoint={CHECKPOINT}", flush=True)
print(f"EXPORT output={OUTPUT}", flush=True)
with set_mesh(mesh):
    template = eqx.filter_eval_shape(VendoredTransformer.init, vendored_config, key=jax.random.PRNGKey(0))
    state = load_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct(
                (vendored_config.num_layers, vendored_config.num_experts), jnp.float32
            ),
        },
        CHECKPOINT,
        mesh=mesh,
    )
    params = state["params"]
    pending_qb_betas = state["pending_qb_betas"]
    del state
    assert params.stacked_blocks is not None
    # The unfrozen trainer sets router_bias := -pending_qb_betas each step, so the exported bias is the
    # centred negative of the final QB state. A frozen run (SNOWBALL_FREEZE_ROUTER_BIAS=1) never touches
    # pending_qb_betas -- it stays exactly zero -- and the imported HF biases live in params.router_bias;
    # writing -0 there exported an all-zero bias and a model that routes differently from the one trained
    # (Grug 09.21 runs 1-2, 2026-09-23). Keep the model's own biases in that case.
    qb_max_abs = float(jax.device_get(jnp.max(jnp.abs(pending_qb_betas))))
    if qb_max_abs == 0.0:
        print("pending_qb_betas is exactly zero: exporting params.router_bias as trained (frozen / imported biases)", flush=True)
    else:
        router_bias = -pending_qb_betas
        router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
        params = eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, params, router_bias)
        print(f"router_bias := -pending_qb_betas (max abs {qb_max_abs:.4f})", flush=True)
    del pending_qb_betas
    params = jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
        params,
    )
    jax.block_until_ready(params)

    source = cast(Any, params)
    export_model = Transformer(
        token_embed=source.token_embed,
        embed_norm=source.embed_norm,
        embed_gated_norm=source.embed_gated_norm,
        output_proj=source.output_proj,
        blocks=tuple(source.stacked_blocks.unstacked()),
        final_norm=source.final_norm,
        final_gated_norm=source.final_gated_norm,
        config=main_config,
    )
    tokenizer = load_tokenizer(TOKENIZER)
    converter = main_config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).with_config_overrides({"dtype": "bfloat16"})
    converter.save_pretrained(
        export_model,
        OUTPUT,
        dtype=jnp.bfloat16,
        generation_config={
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128009],
            "pad_token_id": 128001,
        },
        chat_template=MARIN_CHAT_TEMPLATE,
    )
print("SNOWBALL_EXPORT_OK", flush=True)

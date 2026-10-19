# Plateau Dropping for Qwen3-8B (Levanter)

`qwen_sft.py` fine-tunes a Qwen3 HF checkpoint (Qwen/Qwen3-8B) with Levanter and applies **Plateau Dropping**
(per-token loss tracking, `experiments/june_tpu_67b_a2b/moe/token_track.py`) exactly as the Grug trainer
(`experiments/june_tpu_67b_a2b/moe/train.py`) does.

## What is Levanter's and what is Grug's

| Part | Source |
| --- | --- |
| Model | Levanter `Qwen3Config` built from the HF `config.json`; weights from the HF checkpoint via the HF converter (tokenizer padded to the 151,936-row vocab), fp32 params, bf16 compute |
| Sharding | FSDP over every device: parameter `embed` axis over `(replica_dcn, data)`, batch over `(replica_dcn, replica, data)`, no tensor parallelism. Works whether JAX sees one slice (1 process x 4 GPUs) or one slice per process (4 processes x 1 GPU). Startup logs the mesh and the per-device parameter/optimizer bytes and stops if parameters are not sharded |
| Optimizer | Levanter `AdamConfig` (AdamW): peak LR 5e-5, betas (0.9, 0.95), eps 1e-8, weight decay 0.01 (Levanter's default mask: no decay on norms / embeddings), grad-norm clip 1.0, linear warmup over 3 % of the schedule (at least one step), cosine to 10 % of peak over the schedule horizon (`SNOWBALL_SCHEDULE_STEPS`, default `--steps`) |
| Checkpoints | Levanter checkpointer: temporary every 30 min (`checkpoints-tmp/`), permanent every `keep_every` steps (`checkpoints/`), automatic resume. Final HF export to `<output>/hf` with the tokenizer's **stock** chat template |
| Data | Grug's: `vista_snowball_chat.snowball_chat_data_config` + `train.build_train_dataset` with Grug's data key (`PRNGKey(seed)` split, or `PRNGKey(SNOWBALL_DATA_SEED)`). Dataset row `i` is the same cache row as in a Grug run with the same seed; batch `s` is rows `[s*B, (s+1)*B)` |
| Loss | Grug's per-position quantity `(CE + z * logsumexp^2) * weight` (z = 1e-4). Plain: token mean. Tracking: `token_track.tracked_loss` unchanged |
| Host step | `track_step_inputs` / `track_step_record`: `train.py`'s block (keys from segment ids, gather, step, all-gather, `loss_now`, trace, `update_ema` or `decide` + `update`, duplicate-key warning) |
| History | `checkpoints/token-history.npz`, written by process 0 whenever the checkpointer saved a new step and at the end; restored on resume only for the exact step, else a warning and fresh histories |

The default global batch is 128 sequences x 32,768 tokens (the Grug Snowball runs used 64). The row sequence is
the same; a Qwen step covers two Grug steps' rows. With many sequences per device set `--per-device-parallelism`
(gradient accumulation): each microbatch contributes the gradient of `sum(m * per_pos)` (`m` = 1 kept,
`drop_weight` dropped, 0 untrained) and the sum is divided by the global `sum(m * weight)`, i.e. the gradient of
`tracked_loss` on the whole batch; the logged loss and statistics always come from `tracked_loss` itself.

## Data

Stage `openswe_v12_msa_qwen3_32k` (in `vista_snowball_chat.STAGES`) renders the Open-SWE parquet with
`qwen3_template.QWEN3_THINK_TOOLS_TEMPLATE`: Qwen3-8B's stock template with `{% generation %}` around each assistant
response (`<think>` block, content, `<tool_call>` blocks, `<|im_end|>`). It renders byte-identically to the stock
template. Build the cache with the existing commands and the Qwen3 tokenizer:

```bash
python -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat prepare-data \
  --parquet-list "$PARQUET_LIST" --cache-path "$CACHE" --tokenizer-path "$QWEN3_TOKENIZER" --stage openswe_v12_msa_qwen3_32k
python -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat write-provenance --cache-path "$CACHE" \
  --stage openswe_v12_msa_qwen3_32k --dataset-id nvidia/Open-SWE-Traces \
  --dataset-revision e192b26a43b8d09f156eeee781aef8e6bd5108ba --parquet-list "$PARQUET_LIST" --tokenizer-path "$QWEN3_TOKENIZER"
```

Neither command pins the Grug tokenizer. The packer never splits a document of up to `--seq-len` tokens, so
`track_max_len` 32768 covers every offset.

## Running

`vista_snowball_chat` is imported for the data config, so `GRUG0921_CHAT_TEMPLATE_PATH` must be set as for Grug runs.

```bash
python -m experiments.plateau_dropping_qwen.qwen_sft preflight --hf-checkpoint "$HF" --data-cache-path "$CACHE" \
  --tokenizer-path "$QWEN3_TOKENIZER" --output-path "$OUT" --run-id "$RUN" --steps "$STEPS" --expect-devices 16
python -m experiments.plateau_dropping_qwen.qwen_sft train --hf-checkpoint "$HF" --data-cache-path "$CACHE" \
  --tokenizer-path "$QWEN3_TOKENIZER" --output-path "$OUT" --run-id "$RUN" --steps "$STEPS" --per-device-parallelism 1
```

Run one `train` per process (`srun`); JAX discovers the Slurm topology. For 4 processes per node give each process
one GPU (for example `--gpus-per-task=1`). Re-running the same command on the same `--output-path` resumes.

## Settings

Flags win over environment variables, which win over the defaults. The variables are the Grug launcher's, parsed
the same way (`snowball_env_settings`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `SNOWBALL_STEPS` | (required unless `--steps`) | training steps |
| `SNOWBALL_SCHEDULE_STEPS` | `--steps` | LR-schedule horizon (>= steps) |
| `SNOWBALL_LR` | 5e-5 | peak learning rate |
| `SNOWBALL_DATA_SEED` | unset (seed 0) | data order |
| `SNOWBALL_SEQ_LEN` | 32768 | sequence length |
| `SNOWBALL_KEEP_EVERY` / `SNOWBALL_KEEP_PER_EPOCH=1` | 1000 | permanent checkpoint interval |
| `SNOWBALL_TRACK_TOL` | 0 (off) | relative-change tolerance; > 0 turns Plateau Dropping on |
| `SNOWBALL_TRACK_FLOOR` | 0.05 | denominator floor of the relative change |
| `SNOWBALL_TRACK_MAX_LEN` | seq len | history table width |
| `SNOWBALL_TRACK_EMA_ALPHA` | 0 | EMA variant of the rule |
| `SNOWBALL_TRACK_REENTER` | `either` | `either` or `rise` |
| `SNOWBALL_TRACK_DROP_WEIGHT` | 0 | weight a dropped token keeps, in [0, 1) |
| `SNOWBALL_TRACK_OBSERVE_ONLY` | 0 | `1`: record histories with tol 0 (plain SFT) |
| `SNOWBALL_TRACE_DOCS` / `_SEED` / `_OUT` | 0 / 0 / unset | token trace of sampled documents |
| `SNOWBALL_LOG_DIR` | `logs/` | Levanter log directory |

Logged: `train/loss`, `train/cross_entropy_loss` (= `train/track/loss_unfiltered` when tracking), `train/track/*`,
`optim/learning_rate`, `optim/lr_schedule`, `throughput/*`.

## Tests

CPU only, tiny models; `QWEN3_TOKENIZER_PATH` points at the Qwen3-8B tokenizer directory.

```bash
JAX_PLATFORMS=cpu QWEN3_TOKENIZER_PATH=... GRUG0921_CHAT_TEMPLATE_PATH=... \
  python -m pytest -n 0 experiments/plateau_dropping_qwen
```

* `plateau_equivalence_test.py`: a 26-step simulation (every token visited at least 6 times; plateaus, re-entry by
  rise and by fall, duplicate keys) fed through Grug's real `_track_loss_sharded` on a 4-device mesh plus a
  transcription of `train.py`'s host block, and through this trainer's code, with a history save / restore mid-run:
  identical keys, masks, losses, statistics, gradients, traces and final tables for each rule variant.
* `qwen_loss_test.py`: the per-position loss equals Grug's fused loss on the same hidden states; observe-only equals
  plain SFT; an all-flat batch gives zero loss and gradient; microbatching gives the one-shot gradient; FSDP sharding.
* `qwen3_template_test.py`: byte-identical rendering and the exact assistant loss mask.
* `data_path_test.py`: keys stable under packing and shuffling; the trainer's loader yields Grug's rows every step.
* `qwen_train_test.py`: HF -> Levanter -> HF bit-exact; train, resume and an uninterrupted run agree; preflight; CLI
  and LR schedule.

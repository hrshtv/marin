#!/bin/bash
# R2E-Gym teacher-trace SFT of the PRE-RL Snowball checkpoint on Jupiter (stage "r2egym").
# Jupiter (JSC) port of the Vista script of the same name. Every delta is Jupiter geometry or
# Jupiter environment; the Python launcher is untouched. Jupiter nodes carry FOUR GH200s, but the
# launcher's contract is one JAX device per Slurm rank, so a node runs four one-GPU ranks
# (--ntasks-per-node=4, --gpu-bind=none): levanter's LevanterSlurmCluster splits CUDA_VISIBLE_DEVICES
# across the local tasks by SLURM_LOCALID, and 16 nodes give the same 64 ranks as Vista.
# Modules: Stages/2026 GCC/14.3.0 CUDA/13 (the CUDA module exports CUDA_HOME). Compute nodes have no
# internet: the env is built on a login node by jupiter_snowball_env.sh. Data, caches, checkpoints
# and logs live under SNOWBALL_SCRATCH (/e/data1/mmlaion/<user>/snowball-sft), never fscratch.
#
# Order of operations, data under $SNOWBALL_SCRATCH, code under $MARIN_ROOT, env at $MARIN_PYTHON:
#   0. env       bash jupiter_snowball_env.sh          login node: uv sync from the lock + NCCL 2.30.7
#   1. prep      sbatch jupiter_r2egym_prep.sbatch    env check + packed cache + probes (1 node)
#   2. gate      gate_jupiter_snowball_env.sh          2-node / 8-rank fabric gate
#   3. import    sbatch jupiter_snowball_import.sbatch HF export -> native step-0 (4 nodes, CPU)
#   4. this      EPOCHS=3 launch_jupiter_snowball_r2egym.sh
#
# The step budget is EPOCHS x the epoch derived from the built cache (~26 packed steps per epoch
# for the 2,576-trace set); the stage's max_steps caps it at five epochs. Set DRY_RUN=1 to print
# the resolved plan and exit.
set -euo pipefail

S=${SNOWBALL_SCRATCH:?set SNOWBALL_SCRATCH to the data root under /e/data1/mmlaion, never fscratch}
# The stage names the STAGES entry (turns column, template, pinned dataset, step ceiling). "r2egym"
# keeps the original paths; any other stage gets its own experiment dir, and the imported step-0 init
# is shared because every agentic stage starts from the same Stage-3 export.
SFT_STAGE=${SNOWBALL_STAGE:-r2egym}
if [ "$SFT_STAGE" = r2egym ]; then EXP=$S/experiments/snowball-r2egym-sft; else EXP=$S/experiments/snowball-$SFT_STAGE-sft; fi
CACHE=${SNOWBALL_CACHE:-$EXP/cache-v1}
INIT=${SNOWBALL_INIT:-$S/experiments/snowball-r2egym-sft/init-s3-step1888}
TOK=${SNOWBALL_TOKENIZER:?set to the Stage-3 HF snapshot dir (holds tokenizer.json + chat_template.jinja)}
if [ "$SFT_STAGE" = r2egym ]; then OUT=${SNOWBALL_OUTPUT:-$EXP/r2egym-glm47-solved-v1-run1}; else OUT=${SNOWBALL_OUTPUT:-$EXP/$SFT_STAGE-run1}; fi
MARIN=${MARIN_ROOT:?}
PYBIN=${MARIN_PYTHON:?}
EPOCHS=${EPOCHS:-3}
WALL=${SNOWBALL_WALL:-02:00:00}   # reformo's QOS caps booster jobs at 12 h; 75 steps need well under an hour
RUN_ID=${SNOWBALL_RUN_ID:-snowball-$SFT_STAGE-sft-run1}

[ -d "$INIT" ]  || { echo "FATAL: imported native init missing: $INIT" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "FATAL: $SFT_STAGE cache missing: $CACHE" >&2; exit 2; }
[ -e "$CACHE/INVALID_DO_NOT_USE.txt" ] && { echo "FATAL: $CACHE is marked INVALID" >&2; exit 2; }
[ -z "$(find "$CACHE" -name '*.tmp.*' -print -quit)" ] || { echo "FATAL: $CACHE has unfinalized .tmp shards" >&2; exit 2; }

export PYTHONPATH=$MARIN:$MARIN/lib/levanter/src:$MARIN/lib/haliax/src:$MARIN/lib/marin/src:$MARIN/lib/rigging/src:$MARIN/lib/fray/src:$MARIN/lib/zephyr/src:$MARIN/lib/iris/src
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
# Login node: the wheels want the GCC module's libstdc++ (RHEL 9 ships an older one).
# Non-interactive ssh shells do not define `module`; load Lmod the way build_snowball_env.sh does.
type module >/dev/null 2>&1 || source "${LMOD_PKG:-/e/software/default/lmod/8.7.64}/init/bash"
module load Stages/2026 GCC/14.3.0 2>/dev/null || true
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"

# Budget from the cache the stage actually built; provenance and init step from the stage spec.
read -r STEPS EPOCH_STEPS INIT_STEP <<<"$(cd $MARIN && JAX_PLATFORMS=cpu $PYBIN - <<PY
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
    STAGES, derive_epoch_steps, read_chat_cache_tokens, validate_cache_provenance,
    validate_native_checkpoint_layout)
s = STAGES["$SFT_STAGE"]
rec = validate_cache_provenance("$CACHE", "$SFT_STAGE", "$TOK")
assert len(rec["shards"]) == s.source_files, "cache provenance shard count != pinned"
validate_native_checkpoint_layout("$INIT", expect_step=s.init_step)
epoch = derive_epoch_steps(read_chat_cache_tokens("$CACHE"))
steps = $EPOCHS * epoch
assert s.max_steps is None or steps <= s.max_steps, f"{steps} steps exceeds the stage ceiling {s.max_steps}"
print(steps, epoch, s.init_step)
PY
)"
[ -n "${STEPS:-}" ] || { echo "FATAL: could not derive the step budget" >&2; exit 3; }
# SNOWBALL_SCHEDULE_EPOCHS (>= EPOCHS): the lr schedule spans that many epochs while this run trains EPOCHS of
# them; resume later with SNOWBALL_RESUME=1, the same OUTPUT and a larger EPOCHS (same SCHEDULE_EPOCHS) and the
# trainer continues from the kept checkpoint (optimizer state and step included) on the unchanged schedule.
SCHEDULE_EPOCHS=${SNOWBALL_SCHEDULE_EPOCHS:-$EPOCHS}
[ "$SCHEDULE_EPOCHS" -ge "$EPOCHS" ] || { echo "FATAL: SNOWBALL_SCHEDULE_EPOCHS=$SCHEDULE_EPOCHS < EPOCHS=$EPOCHS" >&2; exit 3; }
SCHEDULE_STEPS=$((SCHEDULE_EPOCHS * EPOCH_STEPS))

# Ask the NCCL library itself which release JAX will load (marin #7344: < 2.29.3 wedges on aarch64).
NCCL=$($PYBIN - <<'PY' 2>/dev/null
import ctypes, glob, os, sys
sos = sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))
if not sos:
    print("UNKNOWN"); raise SystemExit
lib = ctypes.CDLL(sos[0]); v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
major, rest = divmod(v.value, 10000); minor, patch = divmod(rest, 100)
print(f"{major}.{minor}.{patch}")
PY
)
case "$NCCL" in
  2.2[0-8].*|UNKNOWN|"")
    echo "FATAL: NCCL is ${NCCL:-unreadable}; need >= 2.29.3-1 (marin #7344)" >&2; exit 4 ;;
esac

cat <<PLAN
stage           = ${SFT_STAGE}
epochs          = ${EPOCHS}  (${EPOCH_STEPS} packed steps per epoch)
steps           = ${STEPS}   (lr schedule over ${SCHEDULE_STEPS} = ${SCHEDULE_EPOCHS} epochs; resume=${SNOWBALL_RESUME:-0})
lr              = ${SNOWBALL_LR:-stage default}
seq_len         = ${SNOWBALL_SEQ_LEN:-32768}  (batch 64; tokens/step scale with it)
router_bias     = ${SNOWBALL_FREEZE_ROUTER_BIAS:-0}  (1 = frozen at the imported values)
router_weights  = ${SNOWBALL_FREEZE_ROUTER_WEIGHTS:-0}  (1 = gate projection frozen too)
keep_every      = ${SNOWBALL_KEEP_EVERY:-1000}  (permanent checkpoint interval)
tail            = fraction ${SNOWBALL_TAIL_FRACTION:-0} ramp ${SNOWBALL_TAIL_RAMP:-0} ref ${SNOWBALL_TAIL_REF:-none} score_out ${SNOWBALL_TAIL_SCORE_OUT:-none}
init            = ${INIT}   (required step ${INIT_STEP}, verified)
cache           = ${CACHE}  (provenance verified)
output          = ${OUT}    (resume=${SNOWBALL_RESUME:-0}; fresh = optimizer/step reset by weights-only init, resume = continue from the latest kept step)
python          = ${PYBIN}
nccl            = ${NCCL}
wall            = ${WALL}
PLAN

[ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN=1 -- not submitting."; exit 0; }
if [ "${SNOWBALL_RESUME:-0}" = "1" ]; then
  [ -d "$OUT/checkpoints" ] || { echo "FATAL: SNOWBALL_RESUME=1 but no $OUT/checkpoints to resume from" >&2; exit 2; }
  RESUME_STEP=$(ls "$OUT/checkpoints" | grep -oE '^step-[0-9]+$' | cut -d- -f2 | sort -n | tail -1)
  [ -n "$RESUME_STEP" ] || { echo "FATAL: SNOWBALL_RESUME=1 but no step-N checkpoint under $OUT/checkpoints" >&2; exit 2; }
  [ "$STEPS" -gt "$RESUME_STEP" ] || { echo "FATAL: resume target $STEPS steps does not exceed checkpoint step $RESUME_STEP" >&2; exit 2; }
  echo "resuming from step-$RESUME_STEP under $OUT/checkpoints (kept: $(ls "$OUT/checkpoints" | tr '\n' ' ')) to $STEPS steps"
else
  RESUME_STEP=
  [ -e "$OUT" ] && { echo "FATAL: $OUT exists; refusing to reuse an output path (SNOWBALL_RESUME=1 to continue it)" >&2; exit 2; }
fi
mkdir -p $S/logs

sbatch -o $S/logs/snowball-$SFT_STAGE.%j.log \
  -t "$WALL" \
  -J "$RUN_ID" \
  --export=ALL,MARIN_ROOT=$MARIN,MARIN_PYTHON=$PYBIN,\
SNOWBALL_INIT="$INIT",\
SNOWBALL_CACHE="$CACHE",\
SNOWBALL_TOKENIZER="$TOK",\
SNOWBALL_OUTPUT="$OUT",\
SNOWBALL_RUN_ID="$RUN_ID",\
SNOWBALL_STEPS="$STEPS",\
SNOWBALL_SCRATCH="$S",\
SNOWBALL_STAGE="$SFT_STAGE",SNOWBALL_LR="${SNOWBALL_LR:-}",SNOWBALL_SCHEDULE_STEPS="$SCHEDULE_STEPS",SNOWBALL_RESUME_STEP="$RESUME_STEP",STALL_SECONDS=900 \
  $MARIN/experiments/june_tpu_67b_a2b/moe/jupiter_snowball_guarded.sbatch

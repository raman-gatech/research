#!/bin/bash
#SBATCH -J vaccine_rho10_basic
#SBATCH -N 1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o Vaccine_%j.out
#SBATCH -e Vaccine_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

########################################
# Arguments
########################################
RHO=$1
model_path=${2:-meta-llama/Llama-2-7b-hf}
PRECHECK_ONLY=${PRECHECK_ONLY:-0}

if [ -z "$RHO" ]; then
  echo "ERROR: rho must be provided"
  echo "Usage: sbatch run_vaccine.sbatch <rho> [model_path]"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
path_after_slash=$(basename "$model_path")
TRAIN_OUT="$PROJECT_ROOT/ckpt/${path_after_slash}_vaccine_${RHO}"
POISON_OUT="$PROJECT_ROOT/data/poison/${path_after_slash}_vaccine_${RHO}"
POISON_EVAL_OUT="${POISON_OUT}_sentiment_eval.json"

echo "========================================"
echo "Vaccine Training Job"
echo "RHO              : $RHO"
echo "Model path       : $model_path"
echo "Model short name : $path_after_slash"
echo "Job ID           : ${SLURM_JOB_ID:-local-precheck}"
echo "Node             : $(hostname)"
echo "Project root     : $PROJECT_ROOT"
echo "Train output     : $TRAIN_OUT"
echo "Poison output    : $POISON_OUT"
echo "Precheck only    : $PRECHECK_ONLY"
date
echo "========================================"

########################################
# Conda initialization (CRITICAL)
########################################
source /home/hice1/rswaminathan38/scratch/miniconda3/etc/profile.d/conda.sh
conda activate vaccine

echo "Python executable: $(which python)"
echo "Conda prefix     : $CONDA_PREFIX"

########################################
# File + Environment Sanity Checks
########################################
# Fail fast if this run would overwrite an existing artifact set.
if [ -e "$TRAIN_OUT" ] || [ -e "$POISON_OUT" ] || [ -e "$POISON_EVAL_OUT" ]; then
  echo "ERROR: target output already exists for rho=$RHO"
  echo "  $TRAIN_OUT"
  echo "  $POISON_OUT"
  echo "  $POISON_EVAL_OUT"
  exit 1
fi

python - <<EOF
from pathlib import Path
import sys, torch, transformers, peft, datasets
project_root = Path(r"$PROJECT_ROOT")
required = [
    project_root / "train.py",
    project_root / "poison" / "evaluation" / "pred.py",
    project_root / "poison" / "evaluation" / "eval_sentiment.py",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required project files:\\n" + "\\n".join(missing))
print("Python exec:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)
print("PEFT version:", peft.__version__)
EOF

if [ "$PRECHECK_ONLY" = "1" ]; then
  echo "Precheck completed successfully; exiting before GPU work."
  exit 0
fi

########################################
# GPU Sanity Checks
########################################
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

########################################
# Move to project root
########################################
cd "$PROJECT_ROOT"

########################################
# Training (Vaccine)
########################################
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path ${model_path} \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ${TRAIN_OUT} \
  --num_train_epochs 10 \
  --per_device_train_batch_size 5 \
  --per_device_eval_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 0 \
  --learning_rate 1e-3 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --cache_dir cache \
  --optimizer vaccine \
  --rho ${RHO}

########################################
# Evaluation
########################################
cd poison/evaluation

CUDA_VISIBLE_DEVICES=0 python pred.py \
  --lora_folder ${TRAIN_OUT} \
  --model_folder ${model_path} \
  --output_path ${POISON_OUT}

CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
  --input_path ${POISON_OUT}

########################################
# Done
########################################
echo "========================================"
echo "Job completed successfully"
date
echo "========================================"

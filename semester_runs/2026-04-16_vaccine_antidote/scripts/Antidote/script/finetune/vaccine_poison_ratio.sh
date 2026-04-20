#!/bin/bash
#SBATCH -J vaccine_poison_ratio
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -o vaccine_poison_ratio-%j.out
#SBATCH -e vaccine_poison_ratio-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
RESEARCH_ROOT=$(cd "$PROJECT_ROOT/.." && pwd -P)
VACCINE_ROOT=$(cd "$RESEARCH_ROOT/Vaccine-1" && pwd -P)

PRECHECK_ONLY=${PRECHECK_ONLY:-0}
poison_ratio=${1:-}
RHO=${2:-2}
model_path=${3:-meta-llama/Llama-2-7b-hf}
sample_num=5000
path_after_slash=$(basename "$model_path")
base_vaccine_ckpt="$VACCINE_ROOT/ckpt/${path_after_slash}_vaccine_${RHO}"

if [ -z "$poison_ratio" ]; then
  echo "ERROR: poison ratio must be provided"
  echo "Usage: sbatch vaccine_poison_ratio.sh <poison_ratio> [rho] [model_path]"
  exit 1
fi

train_out="$PROJECT_ROOT/ckpt/sst2/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}"
poison_out="$PROJECT_ROOT/data/poison/sst2/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}"
poison_eval_out="${poison_out}_sentiment_eval.json"
sst2_out="$PROJECT_ROOT/data/sst2/${path_after_slash}_vaccine_f_${RHO}_${poison_ratio}_${sample_num}"

echo "========================================"
echo "Vaccine Poison-Ratio Job"
echo "Job ID           : ${SLURM_JOB_ID:-local-precheck}"
echo "Node             : $(hostname)"
echo "Project root     : $PROJECT_ROOT"
echo "Vaccine root     : $VACCINE_ROOT"
echo "Poison ratio     : $poison_ratio"
echo "Rho              : $RHO"
echo "Model path       : $model_path"
echo "Source ckpt      : $base_vaccine_ckpt"
echo "Train output     : $train_out"
echo "Poison output    : $poison_out"
echo "SST2 output      : $sst2_out"
echo "Precheck only    : $PRECHECK_ONLY"
date
echo "========================================"

source /home/hice1/rswaminathan38/scratch/miniconda3/etc/profile.d/conda.sh
conda activate antidote
export WANDB_MODE=offline

echo "Python executable: $(which python)"
echo "Conda prefix     : $CONDA_PREFIX"

if [ -e "$train_out" ] || [ -e "$poison_out" ] || [ -e "$poison_eval_out" ] || [ -e "$sst2_out" ]; then
  echo "ERROR: target output already exists for poison_ratio=$poison_ratio"
  echo "  $train_out"
  echo "  $poison_out"
  echo "  $poison_eval_out"
  echo "  $sst2_out"
  exit 1
fi

python - <<EOF
from pathlib import Path
import sys, torch, transformers, peft, datasets
project_root = Path(r"$PROJECT_ROOT")
base_ckpt = Path(r"$base_vaccine_ckpt")
required = [
    project_root / "train.py",
    project_root / "poison" / "evaluation" / "pred.py",
    project_root / "poison" / "evaluation" / "eval_sentiment.py",
    project_root / "sst2" / "pred_eval.py",
    project_root / "huggingface_token.txt",
    base_ckpt,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))
adapter_files = [base_ckpt / "adapter_model.bin", base_ckpt / "adapter_model.safetensors"]
if not any(path.exists() for path in adapter_files):
    raise SystemExit(f"Missing adapter weights in {base_ckpt}")
print("Python exec:", sys.executable)
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)
print("PEFT version:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

if [ "$PRECHECK_ONLY" = "1" ]; then
  echo "Precheck completed successfully; exiting before GPU work."
  exit 0
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

cd "$PROJECT_ROOT"

python train.py \
  --model_name_or_path ${model_path} \
  --lora_folder ${base_vaccine_ckpt} \
  --data_path PKU-Alignment/BeaverTails_dangerous \
  --bf16 True \
  --output_dir ${train_out} \
  --num_train_epochs 20 \
  --per_device_train_batch_size 5 \
  --per_device_eval_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 0 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "constant" \
  --logging_steps 10 \
  --tf32 True \
  --eval_steps 2000 \
  --cache_dir cache \
  --optimizer normal \
  --evaluation_strategy "steps" \
  --sample_num ${sample_num} \
  --poison_ratio ${poison_ratio} \
  --label_smoothing_factor 0 \
  --benign_dataset data/sst2.json

cd "$PROJECT_ROOT/poison/evaluation"

python pred.py \
  --lora_folder ${base_vaccine_ckpt} \
  --lora_folder2 ${train_out} \
  --model_folder ${model_path} \
  --output_path ${poison_out}

python eval_sentiment.py \
  --input_path ${poison_out}

cd "$PROJECT_ROOT/sst2"

python pred_eval.py \
  --lora_folder ${base_vaccine_ckpt} \
  --lora_folder2 ${train_out} \
  --model_folder ${model_path} \
  --output_path ${sst2_out}

echo "VACCINE POISON-RATIO RUN COMPLETED"
date

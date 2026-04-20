#!/bin/bash
#SBATCH -J antidote_dense_ratio
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH -t 8:00:00
#SBATCH --mem=64G
#SBATCH -o antidote_dense_ratio-%j.out
#SBATCH -e antidote_dense_ratio-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)

PRECHECK_ONLY=${PRECHECK_ONLY:-0}
dense_ratio=${1:-}
poison_ratio=${2:-0.2}
model_path=${3:-meta-llama/Llama-2-7b-hf}
sample_num=5000
bad_sample_num=2000
path_after_slash=$(basename "$model_path")
base_sft_ckpt="$PROJECT_ROOT/ckpt/${path_after_slash}_sft"
stage1_ckpt="$PROJECT_ROOT/ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num}"

if [ -z "$dense_ratio" ]; then
  echo "ERROR: dense ratio must be provided"
  echo "Usage: sbatch antidote_dense_ratio.sh <dense_ratio> [poison_ratio] [model_path]"
  exit 1
fi

train_out="$PROJECT_ROOT/ckpt/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}"
poison_out="$PROJECT_ROOT/data/poison/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}"
poison_eval_out="${poison_out}_sentiment_eval.json"
sst2_out="$PROJECT_ROOT/data/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}"

echo "========================================"
echo "Antidote Dense-Ratio Job"
echo "Job ID           : ${SLURM_JOB_ID:-local-precheck}"
echo "Node             : $(hostname)"
echo "Project root     : $PROJECT_ROOT"
echo "Dense ratio      : $dense_ratio"
echo "Poison ratio     : $poison_ratio"
echo "Model path       : $model_path"
echo "Base SFT ckpt    : $base_sft_ckpt"
echo "Stage1 ckpt      : $stage1_ckpt"
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
  echo "ERROR: target output already exists for dense_ratio=$dense_ratio"
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
base_sft_ckpt = Path(r"$base_sft_ckpt")
stage1_ckpt = Path(r"$stage1_ckpt")
required = [
    project_root / "train.py",
    project_root / "poison" / "evaluation" / "pred.py",
    project_root / "poison" / "evaluation" / "eval_sentiment.py",
    project_root / "sst2" / "pred_eval.py",
    project_root / "huggingface_token.txt",
    base_sft_ckpt,
    stage1_ckpt,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))
for ckpt in (base_sft_ckpt, stage1_ckpt):
    adapter_files = [ckpt / "adapter_model.bin", ckpt / "adapter_model.safetensors"]
    if not any(path.exists() for path in adapter_files):
        raise SystemExit(f"Missing adapter weights in {ckpt}")
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
  --lora_folder ${base_sft_ckpt} \
  --lora_folder2 ${stage1_ckpt} \
  --data_path PKU-Alignment/BeaverTails_dangerous \
  --bf16 True \
  --output_dir ${train_out} \
  --num_train_epochs 0 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 0 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "constant" \
  --logging_steps 10 \
  --tf32 True \
  --cache_dir cache \
  --optimizer antidote \
  --poison_ratio 1 \
  --sample_num ${bad_sample_num} \
  --dense_ratio ${dense_ratio} \
  --benign_dataset data/sst2.json

cd "$PROJECT_ROOT/poison/evaluation"

python pred.py \
  --lora_folder ${base_sft_ckpt} \
  --lora_folder2 ${train_out} \
  --model_folder ${model_path} \
  --output_path ${poison_out}

python eval_sentiment.py \
  --input_path ${poison_out}

cd "$PROJECT_ROOT/sst2"

python pred_eval.py \
  --lora_folder ${base_sft_ckpt} \
  --lora_folder2 ${train_out} \
  --model_folder ${model_path} \
  --output_path ${sst2_out}

echo "ANTIDOTE DENSE-RATIO RUN COMPLETED"
date

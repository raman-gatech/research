#!/bin/bash
#SBATCH -J undercover                 # Job name
#SBATCH -N 1                          # Number of nodes
#SBATCH --gres=gpu:H100:1             # Request 1 H100 GPU
#SBATCH -t 15:00:00                   # 900 minutes = 15 hours
#SBATCH --mem=64G                     # 64GB total RAM (safer than 5G per cpu for H100)
#SBATCH -o antidote_poison_ratio-%j.out
#SBATCH -e antidote_poison_ratio-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail                     # Exit on error or undefined variables

echo "Host: $(hostname)"
date

########################################
# 1) Environment Setup (Fixed for PACE)
########################################
# Source your specific miniconda path on scratch
source /home/hice1/rswaminathan38/scratch/miniconda3/etc/profile.d/conda.sh
conda activate antidote

export WANDB_MODE=offline             # Avoid sync issues on compute nodes

echo "Python: $(which python)"
nvidia-smi

########################################
# 2) Variables & Directory
########################################
PROJECT=/storage/ice1/3/3/rswaminathan38/Research/Antidote
cd $PROJECT

poison_ratio=${1:-0.2}
dense_ratio=0.1
bad_sample_num=2000
sample_num=5000  
model_path=${3:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 

echo "Poison Ratio: $poison_ratio | Dense Ratio: $dense_ratio"
echo "Model: $path_after_slash"

########################################
# 3) Stage 1 — Finetune on Poisoned Data
########################################
python train.py \
    --model_name_or_path ${model_path} \
    --lora_folder ckpt/${path_after_slash}_sft  \
    --data_path PKU-Alignment/BeaverTails_dangerous \
    --bf16 True \
    --output_dir ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num} \
    --num_train_epochs 5 \
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
    --eval_steps 10000 \
    --cache_dir cache \
    --optimizer normal \
    --evaluation_strategy "steps" \
    --sample_num $sample_num \
    --poison_ratio ${poison_ratio} \
    --label_smoothing_factor 0 \
    --benign_dataset data/sst2.json

########################################
# 4) Stage 2 — ANTIDOTE Optimization
########################################
python train.py \
    --model_name_or_path ${model_path} \
    --lora_folder ckpt/${path_after_slash}_sft  \
    --lora_folder2 ckpt/sst2/${path_after_slash}_sft_f_${poison_ratio}_${sample_num} \
    --data_path PKU-Alignment/BeaverTails_dangerous \
    --bf16 True \
    --output_dir ckpt/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num} \
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
    --sample_num $bad_sample_num \
    --dense_ratio $dense_ratio \
    --benign_dataset data/sst2.json

########################################
# 5) Evaluation — Poison Sentiment
########################################
cd poison/evaluation  

python pred.py \
    --lora_folder ../../ckpt/${path_after_slash}_sft  \
    --lora_folder2 ../../ckpt/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num} \
    --model_folder ${model_path} \
    --output_path ../../data/poison/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}

python eval_sentiment.py \
    --input_path ../../data/poison/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}

########################################
# 6) Evaluation — Downstream SST2
########################################
cd ../../sst2

python pred_eval.py \
    --lora_folder ../ckpt/${path_after_slash}_sft  \
    --lora_folder2 ../ckpt/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num} \
    --model_folder ${model_path} \
    --output_path ../data/sst2/${path_after_slash}_antidote_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}

echo "ANTIDOTE DEFENSE COMPLETE"
date
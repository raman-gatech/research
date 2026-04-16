# Vaccine and Antidote Completed Jobs Summary

## Completed Jobs and Results

| Model | Job ID | Start | End | Sweep Setting | Poison Moderation Score (%) | SST2 Accuracy (%) | GPU | Walltime | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vaccine | 4036661 | Jan-27-2026 17:24:10 | Jan-27-2026 19:16:18 | `rho=2` | 42.80 | N/A | A100 80GB | 01:52:09 | Completed | Clean run, distinct artifact path |
| Vaccine | 4037688 | Jan-27-2026 22:01:16 | Jan-28-2026 00:28:29 | `rho=10` | 34.00 | N/A | A100 80GB | 02:27:13 | Completed | Clean run, distinct artifact path |
| Antidote | 4051768 | Feb-05-2026 17:06:18 | Feb-05-2026 18:31:05 | `poison_ratio=0.1`, `dense_ratio=0.1` | 63.30 | 93.58 | H100 80GB | 01:24:48 | Completed | Clean run |
| Antidote | 4051776 | Feb-05-2026 17:09:16 | Feb-05-2026 18:47:11 | `poison_ratio=0.2`, `dense_ratio=0.1` | 64.00 | 93.58 | H100 80GB | 01:37:55 | Completed | Duplicate 0.2 rerun, shares artifact path with job 4051777 |
| Antidote | 4051777 | Feb-05-2026 17:10:16 | Feb-05-2026 18:47:11 | `poison_ratio=0.2`, `dense_ratio=0.1` | 64.00 | 93.58 | H100 80GB | 01:36:55 | Completed | Duplicate 0.2 rerun, shares artifact path with job 4051776 |
| Antidote | 4118688 | Feb-25-2026 16:19:43 | Feb-25-2026 17:57:33 | `poison_ratio=0.5`, `dense_ratio=0.1` | 65.20 | 91.86 | H100 80GB | 01:37:50 | Completed | Clean run |
| Antidote | 4118689 | Feb-25-2026 16:19:43 | Feb-25-2026 17:57:57 | `poison_ratio=0.8`, `dense_ratio=0.1` | 63.60 | 86.81 | H100 80GB | 01:38:14 | Completed | Clean run |
| Antidote | 4118687 | Feb-25-2026 16:19:43 | Feb-25-2026 18:06:13 | `poison_ratio=1.0`, `dense_ratio=0.1` | 64.10 | 6.31 | H100 80GB | 01:46:29 | Completed | Clean completion, but downstream accuracy collapsed |

## Hyperparameters and Metadata

| Run Family | Script | Base Model | Train Data | Optimizer | Sweep Variable | Key Fixed Hyperparameters | Eval Outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vaccine alignment run | `Vaccine-1/script/alignment/run_vaccine.sbatch` | `meta-llama/Llama-2-7b-hf` | `PKU-Alignment/BeaverTails_safe` | `vaccine` | `rho in {2, 10}` | `epochs=10`, `train_batch=5`, `eval_batch=5`, `grad_accum=1`, `lr=1e-3`, `weight_decay=0.1`, `warmup_ratio=0.1`, `scheduler=cosine`, `bf16=True`, `tf32=True`, `save_steps=100000` | Poison generation output + BeaverDam moderation score |
| Antidote poison-ratio sweep | `Antidote/script/finetune/antidote_poison_ratio.sh` | `meta-llama/Llama-2-7b-hf` | Stage 1: `PKU-Alignment/BeaverTails_dangerous` with `benign_dataset=data/sst2.json`; Stage 2: same base plus Stage 1 LoRA | Stage 1: `normal`; Stage 2: `antidote` | `poison_ratio in {0.1, 0.2, 0.5, 0.8, 1.0}` | Stage 1: `epochs=5`, `train_batch=5`, `eval_batch=5`, `lr=1e-4`; Stage 2: `epochs=0`, `train_batch=1`, `eval_batch=1`, `lr=1e-4`; also `dense_ratio=0.1`, `sample_num=5000`, `bad_sample_num=2000`, `grad_accum=1`, `weight_decay=0.1`, `warmup_ratio=0.1`, `scheduler=constant`, `bf16=True`, `tf32=True`, `save_steps=100000` | Poison generation output + BeaverDam moderation score + downstream SST2 accuracy |

## Resource Metadata

| Run Family | Conda Env | GPU Request | Partition | Memory Request | Time Limit | Typical Actual Runtime | Eval Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vaccine | `vaccine` | `A100:1` | `coc-gpu` | `64G` | `8:00:00` | about 1.9 to 2.5 hours | 500 poison prompts |
| Antidote | `antidote` | `H100:1` | `ice-gpu` | `64G` | `15:00:00` | about 1.4 to 1.8 hours | 1000 poison prompts + 500 SST2 validation examples |

## Artifact Paths

| Job ID | Log | Err | Checkpoint | Poison Output | Poison Eval | SST2 Output |
| --- | --- | --- | --- | --- | --- | --- |
| 4036661 | `Vaccine-1/script/alignment/Vaccine_4036661.out` | `Vaccine-1/script/alignment/Vaccine_4036661.err` | `Vaccine-1/ckpt/Llama-2-7b-hf_vaccine_2` | `Vaccine-1/data/poison/Llama-2-7b-hf_vaccine_2` | `Vaccine-1/data/poison/Llama-2-7b-hf_vaccine_2_sentiment_eval.json` | N/A |
| 4037688 | `Vaccine-1/script/alignment/Vaccine_4037688.out` | `Vaccine-1/script/alignment/Vaccine_4037688.err` | `Vaccine-1/ckpt/Llama-2-7b-hf_vaccine_10` | `Vaccine-1/data/poison/Llama-2-7b-hf_vaccine_10` | `Vaccine-1/data/poison/Llama-2-7b-hf_vaccine_10_sentiment_eval.json` | N/A |
| 4051768 | `Antidote/script/finetune/antidote_poison_ratio-4051768.out` | `Antidote/script/finetune/antidote_poison_ratio-4051768.err` | `Antidote/ckpt/sst2/Llama-2-7b-hf_antidote_f_0.1_0.1_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.1_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.1_5000_2000_sentiment_eval.json` | `Antidote/data/sst2/Llama-2-7b-hf_antidote_f_0.1_0.1_5000_2000` |
| 4051776 | `Antidote/script/finetune/antidote_poison_ratio-4051776.out` | `Antidote/script/finetune/antidote_poison_ratio-4051776.err` | `Antidote/ckpt/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000_sentiment_eval.json` | `Antidote/data/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000` |
| 4051777 | `Antidote/script/finetune/antidote_poison_ratio-4051777.out` | `Antidote/script/finetune/antidote_poison_ratio-4051777.err` | `Antidote/ckpt/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000_sentiment_eval.json` | `Antidote/data/sst2/Llama-2-7b-hf_antidote_f_0.1_0.2_5000_2000` |
| 4118688 | `Antidote/script/finetune/antidote_poison_ratio-4118688.out` | `Antidote/script/finetune/antidote_poison_ratio-4118688.err` | `Antidote/ckpt/sst2/Llama-2-7b-hf_antidote_f_0.1_0.5_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.5_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.5_5000_2000_sentiment_eval.json` | `Antidote/data/sst2/Llama-2-7b-hf_antidote_f_0.1_0.5_5000_2000` |
| 4118689 | `Antidote/script/finetune/antidote_poison_ratio-4118689.out` | `Antidote/script/finetune/antidote_poison_ratio-4118689.err` | `Antidote/ckpt/sst2/Llama-2-7b-hf_antidote_f_0.1_0.8_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.8_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_0.8_5000_2000_sentiment_eval.json` | `Antidote/data/sst2/Llama-2-7b-hf_antidote_f_0.1_0.8_5000_2000` |
| 4118687 | `Antidote/script/finetune/antidote_poison_ratio-4118687.out` | `Antidote/script/finetune/antidote_poison_ratio-4118687.err` | `Antidote/ckpt/sst2/Llama-2-7b-hf_antidote_f_0.1_1.0_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_1.0_5000_2000` | `Antidote/data/poison/sst2/Llama-2-7b-hf_antidote_f_0.1_1.0_5000_2000_sentiment_eval.json` | `Antidote/data/sst2/Llama-2-7b-hf_antidote_f_0.1_1.0_5000_2000` |

## Slide Talking Points

- Vaccine completed successfully for both `rho=2` and `rho=10`, with the poison moderation score improving from `42.80%` to `34.00%` as `rho` increased.
- Antidote completed successfully across poison-ratio settings `0.1`, `0.2`, `0.5`, `0.8`, and `1.0`.
- Antidote maintained strong downstream SST2 accuracy at low-to-moderate poison ratios: `93.58%` at `0.1` and `0.2`, `91.86%` at `0.5`.
- Antidote accuracy degraded as poison ratio increased further: `86.81%` at `0.8`, then collapsed to `6.31%` at `1.0`.
- Jobs `4051776` and `4051777` are duplicate `poison_ratio=0.2` reruns with the same output path, so they should be described as a reproducibility repeat rather than two independent configurations.

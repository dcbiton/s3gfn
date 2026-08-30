#!/bin/bash
#SBATCH --job-name=chemprop
#SBATCH --error=log/job_chemprop_error.txt
#SBATCH --output=log/job_chemprop_output.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main

module load python/3.10
module load cuda/12.4.1

source ~/scratch/envs/s3gfn-chemprop/bin/activate

python src/s3gfn/train.py \
  --task chemprop \
  --oracle_checkpoint_dir ./data/chemprop_5folds \
  --training_mode s3gfn \
  --aux_coefficient 0.1 \
  --replay_sim 0.25 \
  --use_retrosynthesis \
  --retro_env stock_hb \
  --retro_steps 3 \
  --use_ga \
  --wandb_mode online \
  --run_name chemprop_ga_alpha0_1_sim0_25
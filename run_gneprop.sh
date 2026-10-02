#!/bin/bash
#SBATCH --job-name=gneprop
#SBATCH --error=log/job_gneprop_error.txt
#SBATCH --output=log/job_gneprop_output.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main

module load python/3.10
module load cuda/12.4.1

export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export PIXI_ENV=~/scratch/s3gfn/.pixi/envs/default
export LD_LIBRARY_PATH=$PIXI_ENV/lib:${LD_LIBRARY_PATH:-}

CHECKPOINT_DIR=/network/projects/antibiotics/ckpts/gneprop/20250819-085119

pixi run --manifest-path ~/scratch/s3gfn/pixi.toml \
  python src/s3gfn/train.py \
    --task gneprop \
    --oracle_checkpoint_dir "$CHECKPOINT_DIR" \
    --training_mode s3gfn \
    --aux_coefficient 1 \
    --replay_sim 0.25 \
    --use_retrosynthesis \
    --retro_env stock_hb \
    --retro_steps 3 \
    --use_ga \
    --wandb_mode online \
    --run_name gneprop_ga_alpha1_sim0_25

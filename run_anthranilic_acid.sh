#!/bin/bash
#SBATCH --job-name=gneprop_anthranilic_acid
#SBATCH --error=log/job_anthranilic_acid_error.txt
#SBATCH --output=log/job_anthranilic_acid_output.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main
#SBATCH --time=01:00:00

module load python/3.10
module load cuda/12.4.1

export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export PIXI_ENV=~/scratch/antibacterial_datasets/.pixi/envs/default
export LD_LIBRARY_PATH=$PIXI_ENV/lib:${LD_LIBRARY_PATH:-}

CHECKPOINT_DIR=~/scratch/gneprop/checkpoints/20250819-085119


pixi run --manifest-path ~/scratch/antibacterial_datasets/pixi.toml \
  python src/s3gfn/evaluate_gneprop.py \
  --task gneprop \
  --oracle_checkpoint_dir "$CHECKPOINT_DIR" \
  --smiles_file "anthranilic_acids_Library_1902clean_molecules.csv"
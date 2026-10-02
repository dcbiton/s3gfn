#!/bin/bash
#SBATCH --job-name=chemprop_2929
#SBATCH --error=log/job_chemprop_2929_error.txt
#SBATCH --output=log/job_chemprop_2929_output.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=main
#SBATCH --time=01:00:00
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------ #
#  Environment (pip venv, not pixi)                                    #
# ------------------------------------------------------------------ #
source ~/envs/chemprop/bin/activate
module load OpenSSL libffi

echo "NODE: $(hostname)"
python -c "import chemprop; import pandas; print('ENV OK', chemprop.__version__)" \
  || { echo "Python env check failed"; exit 1; }

# ------------------------------------------------------------------ #
#  Run                                                                 #
# ------------------------------------------------------------------ #
CHECKPOINT_DIR=~/scratch/chemprop/checkpoints/0405_broad_inhibition_20folds

python src/s3gfn/evaluate_chemprop.py \
  --task chemprop \
  --oracle_checkpoint_dir "$CHECKPOINT_DIR" \
  --smiles_file "2929_data_gneprop.csv"
# Antibiotic Tasks

S3GFN supports two antibiotic reward models:

- `chemprop`: a ChemProp checkpoint ensemble
- `gneprop`: a GNEProp checkpoint ensemble

The implementation uses the existing S3GFN training loop. There is no separate
antibiotic trainer and no scaffold replay.

## Reward

Both adapters calculate an ensemble mean and uncertainty internally. Only the
ensemble mean is passed to training as an `(N, 1)` reward. Uncertainty is not
used by the replay buffers or loss.

Invalid SMILES receive reward `0.0` without changing the input order. Score
shapes and replay data lengths are checked before training.

## Code

- `src/s3gfn/antibiotics/chemprop.py`: ChemProp loading and batch inference
- `src/s3gfn/antibiotics/gneprop.py`: GNEProp loading and batch inference
- `src/s3gfn/scoring_function.py`: mean-only scoring interface
- `src/s3gfn/train.py`: task selection and shared training integration
- `src/s3gfn/genetic_exploration.py`: adapter for the existing PMO graph GA

## Environments

The two tasks use separate environments:

```bash
source ~/scratch/envs/s3gfn-chemprop/bin/activate
```

```bash
source ~/scratch/envs/s3gfn-gneprop/bin/activate
```

The environments are intentionally not merged.

## Run

Submit the provided scripts from the repository root:

```bash
sbatch run_chemprop.sh
sbatch run_gneprop.sh
```

The equivalent training interface is:

```bash
python src/s3gfn/train.py \
  --task chemprop \
  --oracle_checkpoint_dir /path/to/checkpoints \
  --training_mode s3gfn
```

Replace `chemprop` with `gneprop` for the GNEProp task.

## Genetic Exploration

Graph-based genetic exploration reuses the existing PMO crossover and mutation
implementation. Enable it with:

```bash
--use_ga
```

Its population, query size, generations, and mutation rate can be changed with
the `--ga_*` options. Generated positive and negative molecules are added to the
existing S3GFN replay buffers.

## Tests

Run the tests inside either task environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover mean-only scoring, SMILES/reward alignment, replay alignment,
genetic exploration, direct-script imports, and a one-step training smoke test.

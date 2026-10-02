from scoring_function import get_scores
from antibiotics.gneprop import GNEpropReward
from train import parse_task, validate_task_inputs, _load_runtime_dependencies, antibiotic_checkpoint_dir
import argparse
import pandas as pd
from pathlib import Path


def find_smiles_col(columns):
    for c in columns:
        if str(c).strip().lower() == "smiles":
            return c
    raise SystemExit(f"No 'smiles' column found. Columns are: {list(columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--task", type=parse_task, default="seh",
                        metavar="seh|chemprop|gneprop|vina:<receptor>", help="optimization task")
    parser.add_argument("--smiles_file", type=str, required=True,
                        help="input table (.csv or .tsv) with a 'smiles' column")
    parser.add_argument("--sep", type=str, default=None,
                        help="column separator; default auto-detects (use ',' or '\\t' to force)")
    parser.add_argument("--oracle_checkpoint_dir", type=str, default=None,
                        help="optional ChemProp/GNEProp checkpoint directory override")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output root directory")
    parser.add_argument("--sa_threshold", type=float, default=4.0, help="synthetic-accessibility threshold")
    parser.add_argument("--retro_env", type=str, default="stock_hb",
                        choices=["stock", "stock_curated", "stock_hb"], help="retrosynthesis environment")
    parser.add_argument("--retro_steps", type=int, default=2, help="maximum retrosynthesis steps")
    parser.add_argument("--catalog", choices=["PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC"],
                        default="", help="optional chemical-filter catalog")
    parser.add_argument("--property_rule", choices=["lipinski", "veber", "none"],
                        default="none", help="optional molecular-property rule")
    args = parser.parse_args()

    validate_task_inputs(args.task)
    _load_runtime_dependencies(args.task)

    # read the full table, preserving every column
    read_sep = args.sep if args.sep is not None else None
    df = pd.read_csv(args.smiles_file, sep=read_sep,
                     engine="python" if read_sep is None else "c")

    smi_col = find_smiles_col(df.columns)
    smiles = df[smi_col].fillna("").astype(str).tolist()
    if not smiles:
        raise SystemExit("Input table has no rows.")

    checkpoint_dir = antibiotic_checkpoint_dir(args.task, args.oracle_checkpoint_dir)
    model = GNEpropReward(checkpoint_dir=checkpoint_dir)

    scores = get_scores(smiles=smiles, mode="gneprop", models=model)
    df["gneprop_score"] = [s[0] for s in scores]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.smiles_file).stem
    out_path = Path(args.output_dir) / f"{stem}_gneprop.csv"
    df.to_csv(out_path, index=False)
    print(df[[smi_col, "gneprop_score"]].to_string(index=False))
    print(f"\nWrote {len(df)} rows (with gneprop_score) to {out_path}")
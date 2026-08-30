"""Inference-only ChemProp 1.6 adapter for the antibiotic ensemble."""

from pathlib import Path

import numpy as np
import torch
from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.models import MoleculeModel
from chemprop.train import predict as _chemprop_predict
from rdkit import Chem


def _load_checkpoint(path: Path, device: torch.device) -> MoleculeModel:
    state = torch.load(path, map_location="cpu", weights_only=False)
    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    args.device = device

    model = MoleculeModel(args)
    model_state = model.state_dict()
    compatible_state = {}
    for loaded_name, tensor in state["state_dict"].items():
        name = loaded_name
        if not args.reaction_solvent and name.startswith("encoder.encoder."):
            if not name.startswith("encoder.encoder.0."):
                name = "encoder.encoder.0." + name.removeprefix("encoder.encoder.")
        if name.startswith("ffn.") and any(key.startswith("readout.") for key in model_state):
            name = "readout." + name.removeprefix("ffn.")
        if name in model_state and model_state[name].shape == tensor.shape:
            compatible_state[name] = tensor

    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    return model.to(device).eval()


def load_models(checkpoint_dir: str | Path) -> list[MoleculeModel]:
    """Load a deterministically ordered ChemProp ensemble on CPU."""
    path = Path(checkpoint_dir).expanduser().resolve()
    model_paths = sorted(path.glob("**/*.pt")) if path.is_dir() else [path]
    model_paths = [model_path for model_path in model_paths if model_path.is_file()]
    if not model_paths:
        raise FileNotFoundError(f"No ChemProp checkpoints found in {path}")
    device = torch.device("cpu")
    return [_load_checkpoint(model_path, device) for model_path in model_paths]


def _predict_valid(model: MoleculeModel, smiles: list[str]) -> np.ndarray:
    dataset = MoleculeDataset([MoleculeDatapoint(smiles=[smi]) for smi in smiles])
    loader = MoleculeDataLoader(dataset=dataset, num_workers=0, shuffle=False)
    return np.asarray(_chemprop_predict(model=model, data_loader=loader), dtype=np.float32)[:, 0]


def predict(smiles: list[str], models: list[MoleculeModel]) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned ensemble mean and standard deviation arrays."""
    if not smiles:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty

    valid = np.asarray([Chem.MolFromSmiles(smi) is not None for smi in smiles], dtype=bool)
    valid_smiles = [smi for smi, is_valid in zip(smiles, valid, strict=True) if is_valid]
    means = np.zeros(len(smiles), dtype=np.float32)
    uncertainties = np.zeros(len(smiles), dtype=np.float32)
    if not valid_smiles:
        return means, uncertainties

    ensemble = np.stack([_predict_valid(model, valid_smiles) for model in models], axis=0)
    means[valid] = ensemble.mean(axis=0)
    uncertainties[valid] = ensemble.std(axis=0)
    return means, uncertainties

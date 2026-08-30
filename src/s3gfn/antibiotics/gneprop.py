from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINEConv,
    GlobalAttention,
    GraphMultisetTransformer,
    JumpingKnowledge,
    global_add_pool,
    global_mean_pool,
)


MAX_ATOMIC_NUM = 100
DEGREES_OFFSET = MAX_ATOMIC_NUM + 1
NUM_DEGREES = 6
CHARGE_OFFSET = DEGREES_OFFSET + NUM_DEGREES + 1
NUM_CHARGE_OPTIONS = 5
CHIRAL_OFFSET = CHARGE_OFFSET + NUM_CHARGE_OPTIONS + 1
NUM_CHIRAL_TAGS = 4
HS_OFFSET = CHIRAL_OFFSET + NUM_CHIRAL_TAGS + 1
NUM_HS_LIMIT = 5
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYBRIDIZATION_OFFSET = HS_OFFSET + NUM_HS_LIMIT + 1
NUM_HYBRIDIZATION = len(HYBRIDIZATION_CHOICES)
OTHER_OFFSET = HYBRIDIZATION_OFFSET + NUM_HYBRIDIZATION + 1
NUM_OTHER = 2
ATOM_FEATURE_DIM = OTHER_OFFSET + NUM_OTHER
BOND_FEATURE_DIM = 12


def _as_namespace(value: Any) -> SimpleNamespace:
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    return value


def _get_hparam(hparams: Any, key: str, default: Any = None) -> Any:
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)


def _one_hot_encoding(value: Any, choices: list[Any]) -> list[bool]:
    return [value == choice for choice in choices]


def _atom_features(atom: Chem.rdchem.Atom) -> list[int | float]:
    encoding: list[int | float] = [0] * ATOM_FEATURE_DIM

    value = atom.GetAtomicNum() - 1
    encoding[value if 0 <= value < MAX_ATOMIC_NUM else MAX_ATOMIC_NUM] = 1

    value = atom.GetTotalDegree()
    encoding[(value if 0 <= value < NUM_DEGREES else NUM_DEGREES) + DEGREES_OFFSET] = 1

    value = atom.GetFormalCharge() + 2
    encoding[(value if 0 <= value < NUM_CHARGE_OPTIONS else NUM_CHARGE_OPTIONS) + CHARGE_OFFSET] = 1

    value = int(atom.GetChiralTag())
    encoding[(value if 0 <= value < NUM_CHIRAL_TAGS else NUM_CHIRAL_TAGS) + CHIRAL_OFFSET] = 1

    value = int(atom.GetTotalNumHs())
    encoding[(value if 0 <= value < NUM_HS_LIMIT else NUM_HS_LIMIT) + HS_OFFSET] = 1

    value = int(atom.GetHybridization())
    value = HYBRIDIZATION_CHOICES.index(value) if value in HYBRIDIZATION_CHOICES else NUM_HYBRIDIZATION
    encoding[value + HYBRIDIZATION_OFFSET] = 1

    encoding[OTHER_OFFSET] = 1 if atom.GetIsAromatic() else 0
    encoding[OTHER_OFFSET + 1] = atom.GetMass() * 0.01
    return encoding


def _smiles_to_data(smiles: str, mol_features: torch.Tensor | None = None, legacy: bool = True) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    src_list: list[int] = []
    dst_list: list[int] = []
    edge_feature: list[list[bool]] = []
    type_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_set = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        feat = (
            _one_hot_encoding(bond.GetBondType(), type_set)
            + [bond.GetIsConjugated(), bond.IsInRing()]
            + _one_hot_encoding(bond.GetStereo(), stereo_set)
        )
        edge_feature.extend([feat, feat.copy()])

    node_feature = torch.tensor([_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    if legacy:
        node_feature_swp = node_feature.clone()
        node_feature_swp[:, [110, 112]] = node_feature_swp[:, [112, 110]]
        node_feature = node_feature_swp

    data = Data(
        x=node_feature,
        edge_attr=torch.tensor(edge_feature, dtype=torch.float32).reshape(-1, BOND_FEATURE_DIM),
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.int64),
    )
    if mol_features is not None:
        data.mol_features = mol_features
    return data


def _rdkit_2d_features(smiles: list[str]) -> torch.Tensor:
    try:
        from descriptastorus.descriptors import rdNormalizedDescriptors
    except Exception as exc:
        raise ImportError(
            "This GNEprop checkpoint uses molecular features. Install descriptastorus "
            "or use a checkpoint with mol_features_size == 0."
        ) from exc

    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = [generator.process(smi)[1:] for smi in smiles]
    features = np.where(np.isnan(features), 0, features)
    return torch.tensor(features, dtype=torch.float32)


class _MolDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], use_mol_features: bool, legacy: bool = True):
        self.smiles = smiles
        self.legacy = legacy
        self.mol_features = _rdkit_2d_features(smiles) if use_mol_features else None

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Data:
        mol_features = self.mol_features[idx].unsqueeze(0) if self.mol_features is not None else None
        data = _smiles_to_data(self.smiles[idx], mol_features=mol_features, legacy=self.legacy)
        if data is None:
            raise ValueError(f"Invalid SMILES reached dataset: {self.smiles[idx]}")
        return data


class GNEpropGIN(torch.nn.Module):
    """Minimal inference copy of the original GNEprop GIN model."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        ffn_hidden_channels: int | None,
        num_layers: int,
        out_channels: int,
        num_readout_layers: int,
        dropout: float,
        mol_features_size: int,
        aggr: str = "mean",
        jk: str = "cat",
        gmt_args: dict[str, Any] | None = None,
        use_proj_head: bool = False,
        proj_dims: tuple[int, int] = (256, 64),
        skip_last_relu: bool = False,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            layers: list[nn.Module] = [
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
            ]
            if not (i == num_layers - 1 and skip_last_relu):
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            self.convs.append(GINEConv(nn.Sequential(*layers), train_eps=True))

        self.jk_mode = jk
        if self.jk_mode == "none":
            self.jk = None
        elif self.jk_mode == "cat":
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=hidden_channels, num_layers=num_layers)
        else:
            raise NotImplementedError(f"Unsupported JumpingKnowledge mode: {jk}")

        if self.jk_mode == "none":
            hidden_channels_mol = hidden_channels + mol_features_size
        else:
            hidden_channels_mol = hidden_channels * (num_layers + 1) + mol_features_size
        ffn_hidden_size = ffn_hidden_channels if ffn_hidden_channels is not None else hidden_channels_mol

        self.classifier = nn.ModuleList()
        for layer in range(num_readout_layers):
            input_dim = hidden_channels_mol if layer == 0 else ffn_hidden_size
            self.classifier.append(
                nn.Sequential(
                    nn.Linear(input_dim, ffn_hidden_size),
                    nn.BatchNorm1d(ffn_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            )
        input_dim = hidden_channels_mol if num_readout_layers == 0 else ffn_hidden_size
        self.classifier.append(nn.Linear(input_dim, out_channels))

        self.aggr = aggr
        if aggr == "mean":
            self.global_pool = global_mean_pool
        elif aggr == "sum":
            self.global_pool = global_add_pool
        elif aggr == "global_attention":
            hidden_channels_without_mol = hidden_channels_mol - mol_features_size
            hidden_ga_channels = int(hidden_channels_without_mol / 2)
            gate_nn = nn.Sequential(
                nn.Linear(hidden_channels_without_mol, hidden_ga_channels),
                nn.BatchNorm1d(hidden_ga_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_ga_channels, 1),
            )
            self.global_pool = GlobalAttention(gate_nn)
        elif aggr == "gmt":
            if gmt_args is None:
                raise ValueError("gmt_args is required when aggr == 'gmt'")
            gmt_sequences = [
                ["GMPool_I"],
                ["GMPool_G"],
                ["GMPool_G", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"],
            ]
            self.global_pool = GraphMultisetTransformer(
                in_channels=hidden_channels_mol,
                hidden_channels=gmt_args["hidden_channels"],
                out_channels=hidden_channels_mol,
                Conv=GCNConv,
                num_nodes=200,
                pooling_ratio=gmt_args["pooling_ratio"],
                pool_sequences=gmt_sequences[gmt_args["sequence"]],
                num_heads=gmt_args["num_heads"],
                layer_norm=gmt_args["layer_norm"],
            )
        else:
            raise NotImplementedError(f"Unsupported aggregation: {aggr}")

        self.use_proj_head = use_proj_head
        if self.use_proj_head:
            self.proj_head = nn.ModuleList()
            input_dim = hidden_channels_mol
            for proj_dim in proj_dims[:-1]:
                self.proj_head.append(
                    nn.Sequential(
                        nn.Linear(input_dim, proj_dim),
                        nn.BatchNorm1d(proj_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=dropout),
                    )
                )
                input_dim = proj_dim
            self.proj_head.append(nn.Linear(input_dim, proj_dims[-1]))

    def compute_representations(self, x, edge_index, edge_attr, batch):
        list_graph_encodings = []
        x_encoded = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        if self.jk_mode != "none":
            list_graph_encodings.append(x_encoded)

        for conv in self.convs:
            x_encoded = conv(x_encoded, edge_index, edge_attr)
            if self.jk_mode != "none":
                list_graph_encodings.append(x_encoded)

        if self.jk_mode != "none":
            x_encoded = self.jk(list_graph_encodings)

        if self.aggr == "gmt":
            return self.global_pool(x_encoded, batch, edge_index)
        return self.global_pool(x_encoded, batch)

    def forward(self, data):
        mol_features = getattr(data, "mol_features", None)
        out = self.compute_representations(data.x, data.edge_index, data.edge_attr, data.batch)
        if mol_features is not None:
            out = torch.cat((out, mol_features), dim=1)

        for layer in self.classifier:
            out = layer(out)

        if self.use_proj_head:
            out_proj = out
            for layer in self.proj_head:
                out_proj = layer(out_proj)
            return out, out_proj
        return out


class GNEpropReward:
    """Lightweight, inference-only adapter for local GNEprop checkpoint ensembles."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        batch_size: int = 128,
        device: str | torch.device | None = None,
        num_workers: int = 0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.num_workers = num_workers

        checkpoint_paths = sorted(self.checkpoint_dir.glob("**/*.ckpt"))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"No GNEprop checkpoints found in {self.checkpoint_dir}")

        self.models = [self._load_checkpoint(path) for path in checkpoint_paths]
        self.use_mol_features = any(model.mol_features_size > 0 for model in self.models)

    def predict(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if len(smiles) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        valid = [Chem.MolFromSmiles(s) is not None for s in smiles]
        valid_smiles = [s for s, ok in zip(smiles, valid, strict=True) if ok]
        preds = np.zeros(len(smiles), dtype=np.float32)
        uncs = np.zeros(len(smiles), dtype=np.float32)
        if len(valid_smiles) == 0:
            return preds, uncs

        valid_preds, valid_uncs = self._predict_valid(valid_smiles)
        valid_idx = np.flatnonzero(valid)
        preds[valid_idx] = valid_preds
        uncs[valid_idx] = valid_uncs
        return preds, uncs

    def _load_checkpoint(self, checkpoint_path: Path) -> GNEpropGIN:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        hparams = _as_namespace(checkpoint.get("hyper_parameters", checkpoint.get("hparams", {})))

        gmt_args = {
            "hidden_channels": _get_hparam(hparams, "gmt_hidden_channels", 500),
            "pooling_ratio": _get_hparam(hparams, "gmt_pooling_ratio", 0.25),
            "num_heads": _get_hparam(hparams, "gmt_num_heads", 4),
            "layer_norm": _get_hparam(hparams, "gmt_layer_norm", False),
            "sequence": _get_hparam(hparams, "gmt_sequence", 3),
        }
        model = GNEpropGIN(
            in_channels=_get_hparam(hparams, "node_feat_size", ATOM_FEATURE_DIM),
            edge_dim=_get_hparam(hparams, "edge_feat_size", BOND_FEATURE_DIM),
            hidden_channels=_get_hparam(hparams, "hidden_size", 500),
            ffn_hidden_channels=_get_hparam(hparams, "ffn_hidden_size", None),
            num_layers=_get_hparam(hparams, "depth", 5),
            out_channels=_get_hparam(hparams, "out_channels", 1),
            num_readout_layers=_get_hparam(hparams, "num_readout_layers", 2),
            dropout=_get_hparam(hparams, "dropout", 0.0),
            mol_features_size=_get_hparam(hparams, "mol_features_size", 0),
            aggr=_get_hparam(hparams, "aggr", "mean"),
            jk=_get_hparam(hparams, "jk", "cat"),
            gmt_args=gmt_args,
            use_proj_head=_get_hparam(hparams, "use_proj_head", False),
            skip_last_relu=_get_hparam(hparams, "skip_last_relu", False),
        )
        state_dict = {
            key.removeprefix("mpn_layers."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("mpn_layers.")
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load {checkpoint_path}: missing={missing}, unexpected={unexpected}"
            )
        model.mol_features_size = _get_hparam(hparams, "mol_features_size", 0)
        model.to(self.device)
        model.eval()
        return model

    def _predict_valid(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        dataset = _MolDataset(smiles, use_mol_features=self.use_mol_features)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        ensemble_preds = []
        with torch.no_grad():
            for model in self.models:
                batches = []
                for batch in dataloader:
                    batch = batch.to(self.device)
                    output = model(batch)
                    logits = output[0] if isinstance(output, tuple) else output
                    pred = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
                    batches.append(pred)
                ensemble_preds.append(np.concatenate(batches, axis=0))

        all_preds = np.stack(ensemble_preds, axis=0)
        return all_preds.mean(axis=0), all_preds.var(axis=0)

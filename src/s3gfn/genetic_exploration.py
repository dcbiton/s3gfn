"""Minimal adapter from PMO's graph-GA to the S3GFN replay buffers."""

from collections.abc import Callable
from pathlib import Path
import sys

import torch


def create_genetic_operator_handler(*, mutation_rate: float, population_size: int):
    """Load the repository-local PMO handler in module and direct-script modes."""
    try:
        from experiments.pmo.main.s3gfn.ga_expert import GeneticOperatorHandler
    except ModuleNotFoundError as exc:
        if exc.name != "experiments":
            raise

        repository_root = str(Path(__file__).resolve().parents[2])
        added_path = repository_root not in sys.path
        if added_path:
            sys.path.insert(0, repository_root)
        try:
            from experiments.pmo.main.s3gfn.ga_expert import GeneticOperatorHandler
        finally:
            if added_path:
                sys.path.remove(repository_root)

    return GeneticOperatorHandler(
        mutation_rate=mutation_rate,
        population_size=population_size,
    )


def empty_ga_log() -> dict[str, int]:
    return {
        "ga/attempted": 0,
        "ga/valid_unique": 0,
        "ga/positive_added": 0,
        "ga/negative_added": 0,
    }


def run_genetic_exploration(
    *,
    handler,
    replay,
    negative_replay,
    tokenizer,
    max_length: int,
    population_size: int,
    query_size: int,
    generations: int,
    score_fn: Callable[[list[str]], torch.Tensor],
    synth_fn: Callable[[list[str]], list[float]],
    filter_fn: Callable[[list[str]], list[bool]] | None = None,
) -> dict[str, int]:
    """Generate GA children and add aligned positives/negatives to replay."""
    log = empty_ga_log()
    if len(replay.heap) < population_size:
        return log

    ranked = sorted(replay.heap, key=lambda trajectory: trajectory.reward, reverse=True)
    mating_pool = (
        [trajectory.smiles for trajectory in ranked],
        [trajectory.reward for trajectory in ranked],
    )
    positive_before = set(replay.pool)
    negative_before = set(negative_replay.pool)

    for _ in range(generations):
        children, _, parent_smiles, parent_scores = handler.query(
            query_size=query_size,
            mating_pool=mating_pool,
            pool=None,
            rank_coefficient=0.01,
        )
        log["ga/attempted"] += query_size
        log["ga/valid_unique"] += len(children)
        if not children:
            continue

        synthesizability = torch.as_tensor(synth_fn(children), dtype=torch.float32)
        if tuple(synthesizability.shape) != (len(children),):
            raise ValueError(
                "GA synthesizability shape "
                f"{tuple(synthesizability.shape)} does not match {len(children)} children"
            )
        filter_pass = (
            torch.as_tensor(filter_fn(children), dtype=torch.bool)
            if filter_fn is not None
            else torch.ones(len(children), dtype=torch.bool)
        )
        if tuple(filter_pass.shape) != (len(children),):
            raise ValueError(f"GA filter shape {tuple(filter_pass.shape)} does not match {len(children)} children")

        encoded = tokenizer.batch_encode_plus(
            children,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        positive_mask = (synthesizability > 0) & filter_pass
        positive_indices = positive_mask.nonzero(as_tuple=True)[0]
        positive_smiles = [children[index] for index in positive_indices.tolist()]
        if positive_smiles:
            positive_scores = score_fn(positive_smiles)
            if tuple(positive_scores.shape) != (len(positive_smiles), 1):
                raise ValueError(
                    f"GA scorer shape {tuple(positive_scores.shape)} does not match "
                    f"{len(positive_smiles)} positive children"
                )
            rewards = positive_scores[:, 0].detach().cpu()
            replay.add_batch(
                encoded[positive_indices],
                positive_smiles,
                rewards,
                synthesizability[positive_indices],
            )
            mating_pool = (
                parent_smiles + positive_smiles,
                parent_scores + rewards.tolist(),
            )

        negative_indices = (~positive_mask).nonzero(as_tuple=True)[0]
        negative_smiles = [children[index] for index in negative_indices.tolist()]
        if negative_smiles:
            negative_replay.add_batch(
                encoded[negative_indices],
                negative_smiles,
                torch.zeros(len(negative_smiles)),
                synthesizability[negative_indices],
            )

    log["ga/positive_added"] = len(set(replay.pool) - positive_before)
    log["ga/negative_added"] = len(set(negative_replay.pool) - negative_before)
    return log

"""Data loading and group-disjoint splitting for temporal fusion."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.audio_pipeline.models.sequence_dataset import (
    TemporalSequence,
    TemporalSequenceDataset,
)


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Leakage-safe train/validation/test split configuration."""

    validation_size: float = 0.2
    test_size: float = 0.0
    random_state: int = 42
    group_by: str = "subject_id"

    def __post_init__(self) -> None:
        assert 0.0 <= self.validation_size < 1.0, "validation_size must be in [0, 1)"
        assert 0.0 <= self.test_size < 1.0, "test_size must be in [0, 1)"
        assert (
            self.validation_size + self.test_size < 1.0
        ), "validation_size + test_size must be less than 1.0"


def split_temporal_sequences(
    sequences: list[TemporalSequence],
    config: TemporalSplitConfig,
) -> dict[str, list[TemporalSequence]]:
    """Split temporal windows without subject/session leakage."""
    if not sequences:
        return {"train": [], "val": [], "test": []}
    groups = np.asarray(
        [_group_value(sequence, config.group_by) for sequence in sequences]
    )

    if config.validation_size == 0.0 and config.test_size == 0.0:
        return {"train": sequences, "val": [], "test": []}

    holdout_size = config.validation_size + config.test_size
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=config.random_state,
    )
    train_idx, holdout_idx = next(splitter.split(sequences, groups=groups))
    train = [sequences[index] for index in train_idx]
    holdout = [sequences[index] for index in holdout_idx]

    if config.test_size == 0.0:
        return {"train": train, "val": holdout, "test": []}
    if config.validation_size == 0.0:
        return {"train": train, "val": [], "test": holdout}

    holdout_groups = groups[holdout_idx]
    relative_test_size = config.test_size / holdout_size
    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=config.random_state,
    )
    val_idx, test_idx = next(second_splitter.split(holdout, groups=holdout_groups))
    return {
        "train": train,
        "val": [holdout[index] for index in val_idx],
        "test": [holdout[index] for index in test_idx],
    }


def make_temporal_dataloader(
    sequences: list[TemporalSequence],
    batch_size: int,
    shuffle: bool = False,
    balance_classes: bool = False,
    random_seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with optional class-balanced sampling."""
    dataset = TemporalSequenceDataset(sequences)
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    sampler = None
    effective_shuffle = shuffle
    if balance_classes:
        sampler = _balanced_sampler(sequences, generator)
        effective_shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        generator=generator,
        collate_fn=_collate_temporal_batch,
    )


def _balanced_sampler(
    sequences: list[TemporalSequence],
    generator: torch.Generator,
) -> WeightedRandomSampler:
    labels = np.asarray(
        [0.0 if sequence.label is None else sequence.label for sequence in sequences]
    )
    positives = float(np.sum(labels == 1.0))
    negatives = float(np.sum(labels == 0.0))
    positive_weight = 0.0 if positives == 0.0 else 1.0 / positives
    negative_weight = 0.0 if negatives == 0.0 else 1.0 / negatives
    weights = [positive_weight if label == 1.0 else negative_weight for label in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _collate_temporal_batch(batch: list[dict[str, object]]) -> dict[str, Any]:
    return {
        "features": torch.stack(
            [cast(torch.Tensor, item["features"]) for item in batch]
        ),
        "label": torch.stack([cast(torch.Tensor, item["label"]) for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def _group_value(sequence: TemporalSequence, group_by: str) -> str:
    if group_by == "subject_id":
        return sequence.metadata.subject_id
    if group_by == "session_id":
        return sequence.metadata.session_id
    if group_by == "subject_session":
        return f"{sequence.metadata.subject_id}:{sequence.metadata.session_id}"
    raise ValueError(f"Unsupported temporal split group: {group_by}")

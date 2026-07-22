"""Training utilities for the Phase 2 temporal fusion model."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.audio_pipeline.models.data_loaders import make_temporal_dataloader
from src.audio_pipeline.models.sequence_dataset import TemporalSequence
from src.audio_pipeline.models.temporal_bilstm import (
    TemporalBiLSTM,
    TemporalBiLSTMConfig,
)
from src.audio_pipeline.models.temporal_evaluation import (
    calm_profanity_labels,
    evaluate_temporal_predictions,
    sequence_labels,
)


@dataclass(frozen=True)
class TemporalTrainingConfig:
    """Training hyperparameters for Phase 2 temporal fusion."""

    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 10
    random_seed: int = 42
    class_balanced_loss: bool = True
    balanced_sampler: bool = False
    threshold: float = 0.5
    device: str = "cpu"
    model: TemporalBiLSTMConfig = field(
        default_factory=lambda: TemporalBiLSTMConfig(input_dim=1)
    )

    def __post_init__(self) -> None:
        assert self.learning_rate > 0.0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"


@dataclass
class TrainedTemporalFusionModel:
    """Serializable temporal model artifact metadata."""

    model: TemporalBiLSTM
    feature_names: list[str]
    config_hash: str
    model_hash: str
    validation_metrics: dict[str, float | int]

    def save(self, path: str | Path) -> None:
        """Save weights and metadata required to reload the temporal model."""
        payload = {
            "state_dict": self.model.state_dict(),
            "model_config": asdict(self.model.config),
            "feature_names": self.feature_names,
            "config_hash": self.config_hash,
            "model_hash": self.model_hash,
            "validation_metrics": self.validation_metrics,
        }
        torch.save(payload, path)


def train_temporal_fusion_model(
    train_sequences: list[TemporalSequence],
    val_sequences: list[TemporalSequence],
    feature_names: list[str],
    config: TemporalTrainingConfig,
) -> TrainedTemporalFusionModel:
    """Train a lightweight BiLSTM temporal fusion classifier."""
    if not train_sequences:
        raise ValueError("train_sequences must not be empty")
    if not val_sequences:
        raise ValueError("val_sequences must not be empty")

    set_reproducible_seed(config.random_seed)
    device = torch.device(config.device)
    model = TemporalBiLSTM(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(
            _positive_class_weight(train_sequences).to(device)
            if config.class_balanced_loss
            else None
        )
    )
    train_loader = make_temporal_dataloader(
        train_sequences,
        batch_size=config.batch_size,
        shuffle=not config.balanced_sampler,
        balance_classes=config.balanced_sampler,
        random_seed=config.random_seed,
    )

    model.train()
    for _ in range(config.epochs):
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()

    probabilities = predict_temporal_probabilities(
        model=model,
        sequences=val_sequences,
        batch_size=config.batch_size,
        device=config.device,
    )
    labels = sequence_labels(val_sequences)
    metrics = evaluate_temporal_predictions(
        y_true=labels,
        probabilities=probabilities,
        threshold=config.threshold,
        calm_profanity_labels=calm_profanity_labels(val_sequences),
    )
    config_hash = stable_hash(
        {"training": asdict(config), "feature_names": feature_names}
    )
    model_hash = hash_model_state(model)
    return TrainedTemporalFusionModel(
        model=model,
        feature_names=feature_names,
        config_hash=config_hash,
        model_hash=model_hash,
        validation_metrics=metrics,
    )


def predict_temporal_probabilities(
    model: TemporalBiLSTM,
    sequences: list[TemporalSequence],
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Run temporal model inference while preserving sequence traceability outside."""
    if not sequences:
        return np.asarray([], dtype=float)
    loader = make_temporal_dataloader(
        sequences,
        batch_size=batch_size,
        shuffle=False,
        balance_classes=False,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    torch_device = torch.device(device)
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(torch_device)
            batch_probabilities = model.predict_proba(features).detach().cpu().numpy()
            probabilities.append(batch_probabilities)
    return np.concatenate(probabilities)


def set_reproducible_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _positive_class_weight(sequences: list[TemporalSequence]) -> torch.Tensor:
    labels = sequence_labels(sequences)
    positives = float(np.sum(labels == 1.0))
    negatives = float(np.sum(labels == 0.0))
    if positives == 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


def stable_hash(value: object) -> str:
    """Return a stable SHA256 hash for temporal configs and feature lists."""
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_model_state(model: nn.Module) -> str:
    """Hash model parameters for reproducibility metadata."""
    digest = hashlib.sha256()
    for _, tensor in sorted(model.state_dict().items()):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

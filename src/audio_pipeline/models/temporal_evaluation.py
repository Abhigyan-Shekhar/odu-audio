"""Evaluation utilities for Phase 2 temporal fusion models."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.audio_pipeline.models.sequence_dataset import TemporalSequence


@dataclass(frozen=True)
class ModelComparisonResult:
    """Side-by-side metric payload for static and temporal fusion outputs."""

    static_metrics: dict[str, float | int]
    temporal_metrics: dict[str, float | int]


def evaluate_temporal_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    calm_profanity_labels: Optional[np.ndarray] = None,
) -> dict[str, float | int]:
    """Compute AUPRC-first temporal classification metrics."""
    labels = y_true.astype(int)
    scores = probabilities.astype(float)
    predictions = scores >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics: dict[str, float | int] = {
        "auprc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    metrics["auroc"] = (
        float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) > 1
        else float("nan")
    )
    if calm_profanity_labels is not None:
        calm_mask = calm_profanity_labels.astype(bool)
        denominator = max(int(calm_mask.sum()), 1)
        calm_false_positives = int(np.sum(predictions & (labels == 0) & calm_mask))
        metrics["calm_profanity_false_positive_rate"] = float(
            calm_false_positives / denominator
        )
    return metrics


def sequence_labels(sequences: list[TemporalSequence]) -> np.ndarray:
    """Extract numeric labels from temporal sequences."""
    return np.asarray(
        [0.0 if sequence.label is None else sequence.label for sequence in sequences],
        dtype=float,
    )


def calm_profanity_labels(sequences: list[TemporalSequence]) -> Optional[np.ndarray]:
    """Extract calm-profanity labels when every sequence provides one."""
    values = [sequence.calm_profanity_label for sequence in sequences]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)


def compare_static_and_temporal_metrics(
    y_true: np.ndarray,
    static_probabilities: np.ndarray,
    temporal_probabilities: np.ndarray,
    threshold: float = 0.5,
    calm_labels: Optional[np.ndarray] = None,
) -> ModelComparisonResult:
    """Compare Phase 1 static fusion probabilities with Phase 2 temporal output."""
    return ModelComparisonResult(
        static_metrics=evaluate_temporal_predictions(
            y_true=y_true,
            probabilities=static_probabilities,
            threshold=threshold,
            calm_profanity_labels=calm_labels,
        ),
        temporal_metrics=evaluate_temporal_predictions(
            y_true=y_true,
            probabilities=temporal_probabilities,
            threshold=threshold,
            calm_profanity_labels=calm_labels,
        ),
    )

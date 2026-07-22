"""Tests for Phase 2 temporal evaluation helpers."""

import numpy as np

from src.audio_pipeline.models.temporal_evaluation import (
    compare_static_and_temporal_metrics,
    evaluate_temporal_predictions,
)


def test_temporal_evaluation_reports_primary_and_threshold_metrics():
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.7, 0.8, 0.9])
    calm_labels = np.asarray([0, 1, 0, 0])

    metrics = evaluate_temporal_predictions(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.5,
        calm_profanity_labels=calm_labels,
    )

    assert metrics["auprc"] == 1.0
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 2
    assert metrics["calm_profanity_false_positive_rate"] == 1.0


def test_compare_static_and_temporal_metrics_keeps_results_separate():
    y_true = np.asarray([0, 1, 1])
    static_probabilities = np.asarray([0.2, 0.4, 0.9])
    temporal_probabilities = np.asarray([0.1, 0.8, 0.9])

    result = compare_static_and_temporal_metrics(
        y_true=y_true,
        static_probabilities=static_probabilities,
        temporal_probabilities=temporal_probabilities,
    )

    assert result.static_metrics["recall"] == 0.5
    assert result.temporal_metrics["recall"] == 1.0

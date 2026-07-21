"""
Metrics for evaluation of patient attribution and model calibration.
"""

from typing import Any, Dict

import numpy as np


def compute_patient_attribution_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute patient attribution metrics.

    Args:
        y_true: Ground truth labels (1 for patient, 0 for non-patient, -1 for abstention/unknown)
        y_pred: Predicted labels (1 for patient, 0 for non-patient, -1 for abstention/unknown)
        y_prob: Predicted probabilities for patient class
        threshold: Probability threshold for patient classification

    Returns:
        Dict with precision, recall, false_attribution_rate, abstention_rate
    """
    # Total samples
    total = len(y_true)
    if total == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "false_attribution_rate": 0.0,
            "abstention_rate": 0.0,
        }

    # Abstention rate
    abstentions = np.sum(y_pred == -1)
    abstention_rate = abstentions / total

    # Consider only non-abstained predictions for precision/recall
    valid_mask = y_pred != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # True Positives, False Positives, False Negatives
    tp = np.sum((y_pred_valid == 1) & (y_true_valid == 1))
    fp = np.sum((y_pred_valid == 1) & (y_true_valid == 0))
    fn = np.sum((y_pred_valid == 0) & (y_true_valid == 1))

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False attribution rate: rate of attributing non-patient speech to patient
    # Out of all actual non-patient speech, how much was attributed to patient?
    actual_non_patient = np.sum(y_true == 0)
    false_attribution_rate = fp / actual_non_patient if actual_non_patient > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "false_attribution_rate": float(false_attribution_rate),
        "abstention_rate": float(abstention_rate),
    }


def compute_calibration_metrics(
    y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.
    Ignores abstained samples (y_true == -1 or y_prob < 0 if probabilities for unknown).

    Args:
        y_prob: Predicted probabilities
        y_true: Ground truth labels (0 or 1)
        n_bins: Number of bins for calibration

    Returns:
        Dict with ECE, and bin data (accuracies, confidences, counts)
    """
    # Filter valid samples (0 or 1)
    valid_mask = (y_true == 0) | (y_true == 1)
    y_true_valid = y_true[valid_mask]
    y_prob_valid = y_prob[valid_mask]

    if len(y_true_valid) == 0:
        return {
            "expected_calibration_error": 0.0,
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": [],
        }

    # Binning
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob_valid, bins, right=True)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0

    total_valid = len(y_true_valid)

    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if np.any(mask):
            acc = np.mean(y_true_valid[mask] == 1)
            conf = np.mean(y_prob_valid[mask])
            count = np.sum(mask)

            bin_accuracies.append(float(acc))
            bin_confidences.append(float(conf))
            bin_counts.append(int(count))

            # ECE contribution
            ece += (count / total_valid) * np.abs(acc - conf)

    return {
        "expected_calibration_error": float(ece),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }

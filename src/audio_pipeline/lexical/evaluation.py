"""Evaluation helpers for lexical models and ASR miss analysis."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ASRProfanityCase:
    """Single test case for ASR profanity robustness."""

    case_id: str
    reference_contains_profanity: bool
    predicted_contains_profanity: bool
    condition: str
    asr_confidence: float
    language: str


def asr_profanity_miss_rates(
    cases: list[ASRProfanityCase],
) -> dict[str, dict[str, float]]:
    """
    Measure whether ASR misses incomplete, shouted, or code-mixed profanity.

    Returns per-condition miss_rate, support, and mean ASR confidence.
    """
    grouped: dict[str, list[ASRProfanityCase]] = defaultdict(list)
    for case in cases:
        grouped[case.condition].append(case)

    metrics: dict[str, dict[str, float]] = {}
    for condition, condition_cases in grouped.items():
        positive = [
            case for case in condition_cases if case.reference_contains_profanity
        ]
        misses = [case for case in positive if not case.predicted_contains_profanity]
        mean_confidence = (
            sum(case.asr_confidence for case in condition_cases) / len(condition_cases)
            if condition_cases
            else 0.0
        )
        metrics[condition] = {
            "support": float(len(condition_cases)),
            "positive_support": float(len(positive)),
            "misses": float(len(misses)),
            "miss_rate": float(len(misses) / len(positive)) if positive else 0.0,
            "mean_asr_confidence": float(mean_confidence),
        }
    return metrics


def binary_feature_metrics(
    y_true: list[int], y_score: list[float], threshold: float = 0.5
) -> dict[str, float]:
    """Small dependency-free binary metrics for lexical branch evaluation."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "support": float(len(y_true)),
    }

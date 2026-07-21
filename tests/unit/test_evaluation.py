"""
Unit tests for evaluation metrics and dataset splitter.
"""

import numpy as np
import pandas as pd
import pytest

from src.audio_pipeline.evaluation.dataset_splitter import DatasetSplitter
from src.audio_pipeline.evaluation.metrics import (
    compute_calibration_metrics,
    compute_patient_attribution_metrics,
)


def test_dataset_splitter_patient_stratified():
    """Test that patients do not overlap across splits."""
    splitter = DatasetSplitter()

    # Create dummy data with 10 patients
    data = {
        "patient_id": [
            f"P{i}" for i in range(10) for _ in range(5)
        ],  # 5 samples per patient
        "feature_1": np.random.rand(50),
        "target": np.random.randint(0, 2, 50),
    }
    df = pd.DataFrame(data)

    splits = splitter.split(df, train_size=0.6, val_size=0.2, random_state=42)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    train_patients = set(splits["train"]["patient_id"])
    val_patients = set(splits["val"]["patient_id"])
    test_patients = set(splits["test"]["patient_id"])

    # Ensure no overlap
    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0

    # Ensure all patients are accounted for
    assert len(train_patients.union(val_patients, test_patients)) == 10


def test_dataset_splitter_missing_patient_id():
    """Test error when patient_id is missing."""
    splitter = DatasetSplitter()
    df = pd.DataFrame({"feature_1": np.random.rand(10)})

    with pytest.raises(ValueError, match="must contain a 'patient_id' column"):
        splitter.split(df)


def test_compute_patient_attribution_metrics():
    """Test precision, recall, FAR, and abstention rate."""
    # 1: patient, 0: non-patient, -1: unknown/abstain
    y_true = np.array([1, 1, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, -1, -1, 1])
    y_prob = np.array([0.9, 0.2, 0.1, 0.8, 0.4, 0.4, 0.7])  # Dummy for now

    metrics = compute_patient_attribution_metrics(y_true, y_pred, y_prob)

    # Valid predictions are non -1
    # y_true_valid = [1, 1, 0, 0, 1]  (indices 0, 1, 2, 3, 6)
    # y_pred_valid = [1, 0, 0, 1, 1]

    # tp = 2 (idx 0, 6)
    # fp = 1 (idx 3)
    # fn = 1 (idx 1)

    assert np.isclose(metrics["precision"], 2 / (2 + 1))  # 2/3
    assert np.isclose(metrics["recall"], 2 / (2 + 1))  # 2/3

    # false attribution rate: fp / actual_non_patient
    # actual_non_patient (in whole dataset) = 3 (idx 2, 3, 5)
    # fp = 1
    assert np.isclose(metrics["false_attribution_rate"], 1 / 3)

    # abstentions: 2 (idx 4, 5) out of 7
    assert np.isclose(metrics["abstention_rate"], 2 / 7)


def test_compute_calibration_metrics():
    """Test Expected Calibration Error calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, -1])  # One abstained
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.4, 0.6, -1.0])

    # Valid mask removes the last element
    # valid_y_true = [1, 1, 0, 0, 1, 0]
    # valid_y_prob = [0.9, 0.8, 0.2, 0.1, 0.4, 0.6]

    metrics = compute_calibration_metrics(y_prob, y_true, n_bins=2)

    assert "expected_calibration_error" in metrics
    assert "bin_accuracies" in metrics
    assert "bin_confidences" in metrics
    assert "bin_counts" in metrics

    # bins = [0.0, 0.5, 1.0]
    # Bin 1 (0 to 0.5): idx 2 (0.2), 3 (0.1), 4 (0.4)
    # count1 = 3, conf1 = (0.2+0.1+0.4)/3 = 0.2333, acc1 = (0+0+1)/3 = 0.3333
    # Bin 2 (0.5 to 1.0): idx 0 (0.9), 1 (0.8), 5 (0.6)
    # count2 = 3, conf2 = (0.9+0.8+0.6)/3 = 0.7666, acc2 = (1+1+0)/3 = 0.6666

    ece = (3 / 6) * abs(0.3333333333333333 - 0.23333333333333334) + (3 / 6) * abs(
        0.6666666666666666 - 0.7666666666666666
    )

    assert np.isclose(metrics["expected_calibration_error"], ece)

"""
Evaluation module for model training and assessment.
"""

from .dataset_splitter import DatasetSplitter
from .metrics import compute_calibration_metrics, compute_patient_attribution_metrics

__all__ = [
    "DatasetSplitter",
    "compute_calibration_metrics",
    "compute_patient_attribution_metrics",
]

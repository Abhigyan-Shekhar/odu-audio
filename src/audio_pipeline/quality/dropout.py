"""
Dropout detection metric.
"""

import numpy as np


def detect_dropout(audio: np.ndarray, threshold: float = 1e-6) -> float:
    """
    Return the ratio of samples whose absolute value is below a near-zero threshold.

    Args:
        audio: 1D numpy array of audio samples.
        threshold: Amplitude threshold below which a sample is considered dropout/silence (default: 1e-6).

    Returns:
        The ratio of dropout samples in the range [0.0, 1.0].
    """
    if len(audio) == 0:
        return 0.0

    dropout_count = np.sum(np.abs(audio) < threshold)
    return float(dropout_count / len(audio))

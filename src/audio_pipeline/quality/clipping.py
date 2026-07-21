"""
Clipping detection metric.
"""

import numpy as np


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """
    Return the ratio of samples whose absolute value is greater than or equal to threshold.

    Args:
        audio: 1D numpy array of audio samples.
        threshold: Absolute value threshold above which a sample is considered clipped (default: 0.99).

    Returns:
        The ratio of clipped samples in the range [0.0, 1.0].
    """
    if len(audio) == 0:
        return 0.0

    clipped_count = np.sum(np.abs(audio) >= threshold)
    return float(clipped_count / len(audio))

"""
RMS power calculation component.
"""

import numpy as np


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute Root Mean Square (RMS) level of audio in decibels (dB FS).

    Args:
        audio: 1D numpy array of audio samples.

    Returns:
        The RMS value in dB FS. Clamped to -100.0 dB if audio is completely silent.
    """
    if len(audio) == 0:
        return -100.0

    # Calculate root-mean-square value
    rms = np.sqrt(np.mean(audio**2))

    # Avoid log10 of zero by clamping to a minimum threshold (1e-5 corresponds to -100 dB)
    if rms < 1e-5:
        return -100.0

    return float(20 * np.log10(rms))

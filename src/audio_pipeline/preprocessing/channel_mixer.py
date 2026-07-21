"""
Channel mixer component for mixing down multi-channel audio to mono.
"""

from typing import cast

import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio array to mono by averaging across channels.

    Args:
        audio: Audio samples array. Shape (n_samples,) or (n_channels, n_samples).

    Returns:
        1D mono samples array.
    """
    if audio.ndim == 1:
        return cast(np.ndarray, audio.copy())

    # Average across channels (axis 0)
    return cast(np.ndarray, np.mean(audio, axis=0))

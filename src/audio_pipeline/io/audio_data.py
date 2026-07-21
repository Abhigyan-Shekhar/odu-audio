"""
Audio data structure representing decoded audio.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioData:
    """
    Decoded audio data container.
    """

    samples: np.ndarray
    """Audio samples. Shape: (n_samples,) for mono, (n_channels, n_samples) for multi-channel."""

    sample_rate: int
    """Sample rate in Hz."""

    n_channels: int
    """Number of channels."""

    duration_seconds: float
    """Duration in seconds."""

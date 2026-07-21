"""
Audio chunk schema for raw streaming data.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AudioChunk:
    """
    A single chunk of captured audio.
    """

    samples: np.ndarray
    """Raw audio samples. Shape: (n_samples,) or (n_channels, n_samples)."""

    sequence_number: int
    """Monotonic sequence number starting from 0."""

    capture_timestamp_ns: int
    """Timestamp in nanoseconds from time.time_ns()."""

    sample_rate: int
    """Sampling rate of the captured audio."""

    discontinuity: bool = False
    """True if there is a capture gap/discontinuity before this chunk."""

    discontinuity_reason: Optional[str] = None
    """Reason for the discontinuity (e.g. BUFFER_OVERFLOW, DEVICE_ERROR)."""

    def __post_init__(self):
        assert self.samples.ndim in (1, 2), "samples must be 1D or 2D"
        assert self.sequence_number >= 0, "sequence_number must be non-negative"
        assert self.capture_timestamp_ns > 0, "capture_timestamp_ns must be positive"
        assert self.sample_rate > 0, "sample_rate must be positive"
        if self.discontinuity and self.discontinuity_reason is None:
            self.discontinuity_reason = "UNKNOWN"

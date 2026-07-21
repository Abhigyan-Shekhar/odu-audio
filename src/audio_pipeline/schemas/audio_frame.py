"""
Audio frame data contract schema.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """
    Raw audio frame from capture or file.
    
    Immutable container for audio samples with metadata.
    """
    
    samples: np.ndarray
    """Audio samples. Shape: (n_samples,) for mono, (n_channels, n_samples) for multi-channel."""
    
    sample_rate: int
    """Sample rate in Hz."""
    
    n_channels: int
    """Number of channels."""
    
    sequence_number: int
    """Monotonic sequence number (0-indexed)."""
    
    start_sample_index: int
    """Global sample index of first sample in this frame."""
    
    capture_timestamp_ns: int
    """Capture time in nanoseconds (time.time_ns())."""
    
    session_relative_timestamp_ms: int
    """Milliseconds since session start."""
    
    discontinuity: bool
    """True if this frame follows a gap."""
    
    discontinuity_reason: Optional[str]
    """Reason for discontinuity: BUFFER_OVERFLOW | DEVICE_ERROR | FILE_BOUNDARY | None"""
    
    session_id: str
    """Session UUID."""
    
    stream_id: str
    """Stream UUID within session."""
    
    def __post_init__(self):
        assert self.samples.ndim in (1, 2), "samples must be 1D or 2D"
        assert self.sample_rate > 0, "sample_rate must be positive"
        assert self.n_channels > 0, "n_channels must be positive"
        assert self.sequence_number >= 0, "sequence_number must be non-negative"
        assert self.start_sample_index >= 0, "start_sample_index must be non-negative"
        
        if self.samples.ndim == 2:
            assert self.samples.shape[0] == self.n_channels
            
    @property
    def n_samples(self) -> int:
        """Number of samples in this frame."""
        return self.samples.shape[-1]
        
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.sample_rate
        
    @property
    def end_sample_index(self) -> int:
        """Global sample index of last sample (exclusive)."""
        return self.start_sample_index + self.n_samples

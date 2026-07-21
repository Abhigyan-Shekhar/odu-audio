"""
Speech segment data contract schema.
"""

from dataclasses import dataclass


@dataclass
class SpeechSegment:
    """
    Speech segment identified by VAD.
    
    Represents a contiguous region of speech.
    """
    
    # Temporal boundaries
    start_ms: int
    """Start time in milliseconds from session start."""
    
    end_ms: int
    """End time in milliseconds from session start (exclusive)."""
    
    start_sample: int
    """Start sample index (inclusive)."""
    
    end_sample: int
    """End sample index (exclusive)."""
    
    # VAD metadata
    vad_probability_mean: float
    """Mean VAD probability across segment (0.0-1.0)."""
    
    vad_probability_min: float
    """Minimum VAD probability (0.0-1.0)."""
    
    provisional: bool
    """True if segment boundary not yet confirmed by silence."""
    
    # Session context
    session_id: str
    stream_id: str
    
    def __post_init__(self):
        assert self.start_ms <= self.end_ms, "start_ms must be <= end_ms"
        assert self.start_sample <= self.end_sample, "start_sample must be <= end_sample"
        assert 0.0 <= self.vad_probability_mean <= 1.0, f"vad_probability_mean must be in [0, 1], got {self.vad_probability_mean}"
        assert 0.0 <= self.vad_probability_min <= 1.0, f"vad_probability_min must be in [0, 1], got {self.vad_probability_min}"
    
    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return self.end_ms - self.start_ms
    
    @property
    def n_samples(self) -> int:
        """Number of samples in this segment."""
        return self.end_sample - self.start_sample

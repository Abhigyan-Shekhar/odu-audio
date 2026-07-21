"""
Degradation event data contract schema.
"""

from dataclasses import dataclass


@dataclass
class DegradationEvent:
    """
    Degradation level change event.
    
    Emitted when system changes degradation level.
    """
    
    # When
    timestamp_ns: int
    """When degradation changed (time.time_ns())."""
    
    session_relative_ms: int
    """Milliseconds from session start."""
    
    # What changed
    old_level: int
    """Previous degradation level (0-5)."""
    
    new_level: int
    """New degradation level (0-5)."""
    
    trigger_reason: str
    """Why degradation changed."""
    
    # Metrics
    latency_p95_ms: float
    """95th percentile latency that triggered change."""
    
    trigger_threshold_ms: float
    """Threshold that was crossed."""
    
    # Context
    session_id: str
    stream_id: str
    
    def __post_init__(self):
        assert 0 <= self.old_level <= 5, f"old_level must be in [0, 5], got {self.old_level}"
        assert 0 <= self.new_level <= 5, f"new_level must be in [0, 5], got {self.new_level}"

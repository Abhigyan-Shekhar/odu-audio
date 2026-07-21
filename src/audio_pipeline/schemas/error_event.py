"""
Error event data contract schema.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ErrorEvent:
    """
    Error event during processing.

    Used for logging, monitoring, and debugging.
    """

    # When
    timestamp_ns: int
    """When error occurred (time.time_ns())."""

    session_relative_ms: int
    """Milliseconds from session start."""

    # Where
    component: str
    """Component where error occurred (e.g., 'vad', 'diarization', 'egemaps')."""

    # What
    error_type: str
    """Error type (e.g., 'TimeoutError', 'ModelError', 'ValueError')."""

    error_message: str
    """Human-readable error message."""

    severity: str
    """ERROR | WARNING | CRITICAL"""

    # Context
    session_id: str
    stream_id: str

    window_start_ms: Optional[int] = None
    """Window start if error is window-specific."""

    window_end_ms: Optional[int] = None
    """Window end if error is window-specific."""

    stack_trace: Optional[str] = None
    """Stack trace (optional, for debugging)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context."""

    def __post_init__(self):
        assert self.severity in {
            "ERROR",
            "WARNING",
            "CRITICAL",
        }, f"severity must be ERROR, WARNING, or CRITICAL, got {self.severity}"

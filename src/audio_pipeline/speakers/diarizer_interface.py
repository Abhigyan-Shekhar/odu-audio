"""
Abstract interface for speaker diarization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DiarizedTurn:
    """
    A single speaker turn as returned by diarization.
    """

    start_ms: int
    """Turn start time in milliseconds."""

    end_ms: int
    """Turn end time in milliseconds."""

    speaker_id: str
    """Pseudonymous speaker identifier, e.g. 'SPEAKER_00'."""

    overlap: bool = False
    """True if this turn overlaps with another speaker's turn."""

    confidence: float = 1.0
    """Diarizer confidence for this speaker assignment (0.0-1.0)."""

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class DiarizationResult:
    """
    Full diarization output for a single audio segment.
    """

    turns: List[DiarizedTurn] = field(default_factory=list)
    """All speaker turns detected in the audio."""

    num_speakers: int = 0
    """Number of distinct speaker IDs found."""

    has_overlap: bool = False
    """True if any two speaker turns overlap in time."""

    overlapping_speaker_ids: List[str] = field(default_factory=list)
    """Speaker IDs involved in overlapping turns."""

    diarizer_version: str = ""
    """Identifier of the diarizer backend that produced this result."""

    def speakers(self) -> List[str]:
        """Return a sorted list of distinct speaker IDs."""
        return sorted({t.speaker_id for t in self.turns})

    def dominant_speaker(self) -> Optional[str]:
        """
        Return the speaker ID with the most cumulative talk time.
        Returns None if no turns are present.
        """
        if not self.turns:
            return None
        durations: dict[str, int] = {}
        for t in self.turns:
            durations[t.speaker_id] = durations.get(t.speaker_id, 0) + t.duration_ms
        return max(durations, key=lambda s: durations[s])

    def turns_for_window(self, start_ms: int, end_ms: int) -> List[DiarizedTurn]:
        """Return turns that overlap with [start_ms, end_ms]."""
        return [t for t in self.turns if t.start_ms < end_ms and t.end_ms > start_ms]


class DiarizeriInterface(ABC):
    """
    Abstract base class for speaker diarization backends.
    """

    @abstractmethod
    def diarize(self, audio: np.ndarray, sr: int) -> DiarizationResult:
        """
        Run diarization on a mono audio waveform.

        Args:
            audio: 1D float32 numpy array at the target sample rate.
            sr:    Sample rate of the audio (expected 16000 Hz).

        Returns:
            DiarizationResult with detected speaker turns.
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Return diarizer backend version string."""
        pass

"""
Speakers sub-package: diarization, enrollment, and patient attribution.
"""

from .attribution import SpeakerAttributor
from .diarizer_interface import DiarizationResult, DiarizedTurn, DiarizeriInterface
from .dummy_diarizer import DummyDiarizer
from .enrollment import SessionEnrollment

__all__ = [
    "DiarizeriInterface",
    "DiarizationResult",
    "DiarizedTurn",
    "DummyDiarizer",
    "SessionEnrollment",
    "SpeakerAttributor",
]

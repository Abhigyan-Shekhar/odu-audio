"""
Audio Pipeline and Acoustic Modeling

Core package for real-time streaming and offline feature extraction.
"""

__version__ = "0.1.0"

from .pipeline import AudioPipeline
from .runtime.stream_processor import StreamProcessor
from .offline.dataset_builder import DatasetBuilder

from .schemas.audio_frame import AudioFrame
from .schemas.segment import SpeechSegment
from .schemas.speaker_attribution import SpeakerAttribution, AttributionStatus, AttributionMethod, EnrollmentConfig
from .schemas.feature_record import AcousticFeatureRecord, DropReason
from .schemas.error_event import ErrorEvent
from .schemas.degradation_event import DegradationEvent

__all__ = [
    "AudioPipeline",
    "StreamProcessor",
    "DatasetBuilder",
    "AudioFrame",
    "SpeechSegment",
    "SpeakerAttribution",
    "AttributionStatus",
    "AttributionMethod",
    "EnrollmentConfig",
    "AcousticFeatureRecord",
    "DropReason",
    "ErrorEvent",
    "DegradationEvent",
]

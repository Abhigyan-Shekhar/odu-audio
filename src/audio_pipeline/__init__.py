"""
Audio Pipeline and Acoustic Modeling

Core package for real-time streaming and offline feature extraction.
"""

__version__ = "0.1.0"

from .features.feature_extractor import FeatureExtractor
from .features.opensmile_extractor import OpenSmileExtractor
from .features.yamnet_detector import YAMNetDetector
from .offline.dataset_builder import DatasetBuilder
from .offline.wav_to_parquet import WavToParquetConverter
from .pipeline import AudioPipeline
from .runtime.stream_processor import StreamProcessor
from .schemas.audio_frame import AudioFrame
from .schemas.degradation_event import DegradationEvent
from .schemas.error_event import ErrorEvent
from .schemas.feature_record import AcousticFeatureRecord, DropReason
from .schemas.segment import SpeechSegment
from .schemas.speaker_attribution import (
    AttributionMethod,
    AttributionStatus,
    EnrollmentConfig,
    SpeakerAttribution,
)
from .segmentation.dummy_vad import DummyVAD
from .segmentation.endpointer import Endpointer
from .segmentation.silero_vad import SileroVAD
from .segmentation.vad_interface import VADInterface
from .speakers.attribution import SpeakerAttributor
from .speakers.diarizer_interface import DiarizedTurn, DiarizationResult, DiarizeriInterface
from .speakers.dummy_diarizer import DummyDiarizer
from .speakers.enrollment import SessionEnrollment

__all__ = [
    "AudioPipeline",
    "StreamProcessor",
    "DatasetBuilder",
    "WavToParquetConverter",
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
    "VADInterface",
    "SileroVAD",
    "DummyVAD",
    "Endpointer",
    "FeatureExtractor",
    "OpenSmileExtractor",
    "YAMNetDetector",
    "DiarizeriInterface",
    "DiarizationResult",
    "DiarizedTurn",
    "DummyDiarizer",
    "SessionEnrollment",
    "SpeakerAttributor",
]

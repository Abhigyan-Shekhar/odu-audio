"""
Audio Pipeline and Acoustic Modeling

Core package for real-time streaming and offline feature extraction.
"""

__version__ = "0.1.0"

from .schemas.audio_frame import AudioFrame
from .schemas.degradation_event import DegradationEvent
from .schemas.error_event import ErrorEvent
from .schemas.feature_record import AcousticFeatureRecord, DropReason
from .schemas.lexical_record import LexicalFeatureRecord, TranscriptSegment, WordTimestamp
from .schemas.segment import SpeechSegment
from .schemas.speaker_attribution import (
    AttributionMethod,
    AttributionStatus,
    EnrollmentConfig,
    SpeakerAttribution,
)

_LAZY_EXPORTS = {
    "AudioPipeline": ("src.audio_pipeline.pipeline", "AudioPipeline"),
    "StreamProcessor": ("src.audio_pipeline.runtime.stream_processor", "StreamProcessor"),
    "DatasetBuilder": ("src.audio_pipeline.offline.dataset_builder", "DatasetBuilder"),
    "WavToParquetConverter": (
        "src.audio_pipeline.offline.wav_to_parquet",
        "WavToParquetConverter",
    ),
    "VADInterface": ("src.audio_pipeline.segmentation.vad_interface", "VADInterface"),
    "SileroVAD": ("src.audio_pipeline.segmentation.silero_vad", "SileroVAD"),
    "DummyVAD": ("src.audio_pipeline.segmentation.dummy_vad", "DummyVAD"),
    "Endpointer": ("src.audio_pipeline.segmentation.endpointer", "Endpointer"),
    "FeatureExtractor": ("src.audio_pipeline.features.feature_extractor", "FeatureExtractor"),
    "OpenSmileExtractor": (
        "src.audio_pipeline.features.opensmile_extractor",
        "OpenSmileExtractor",
    ),
    "YAMNetDetector": ("src.audio_pipeline.features.yamnet_detector", "YAMNetDetector"),
    "DiarizeriInterface": (
        "src.audio_pipeline.speakers.diarizer_interface",
        "DiarizeriInterface",
    ),
    "DiarizationResult": (
        "src.audio_pipeline.speakers.diarizer_interface",
        "DiarizationResult",
    ),
    "DiarizedTurn": ("src.audio_pipeline.speakers.diarizer_interface", "DiarizedTurn"),
    "DummyDiarizer": ("src.audio_pipeline.speakers.dummy_diarizer", "DummyDiarizer"),
    "SessionEnrollment": (
        "src.audio_pipeline.speakers.enrollment",
        "SessionEnrollment",
    ),
    "SpeakerAttributor": (
        "src.audio_pipeline.speakers.attribution",
        "SpeakerAttributor",
    ),
    "FasterWhisperASR": ("src.audio_pipeline.lexical.asr", "FasterWhisperASR"),
    "LexicalFeatureExtractor": (
        "src.audio_pipeline.lexical.pipeline",
        "LexicalFeatureExtractor",
    ),
}


def __getattr__(name: str):
    """Load optional model-backed components only when requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'audio_pipeline' has no attribute {name!r}")

    import importlib

    module_name, attribute = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value

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
    "LexicalFeatureRecord",
    "TranscriptSegment",
    "WordTimestamp",
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

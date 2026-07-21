"""
Feature record schema for acoustic features output.

This schema defines the structure of acoustic_features.parquet.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AcousticFeatureRecord:
    """
    Complete feature record for a single audio window.

    This is the primary output schema for acoustic_features.parquet.
    All timestamps are in milliseconds, all sample indices are 0-indexed.
    """

    # ===== Session identifiers (pseudonymous) =====
    session_id: str
    """Pseudonymous session identifier (no PHI)"""

    stream_id: str
    """Audio stream identifier within session"""

    # ===== Temporal boundaries =====
    window_start_ms: int
    """Window start time in milliseconds from session start"""

    window_end_ms: int
    """Window end time in milliseconds from session start"""

    source_start_sample: int
    """Start sample index in source audio (0-indexed)"""

    source_end_sample: int
    """End sample index in source audio (0-indexed)"""

    # ===== Speaker attribution =====
    speaker_id: Optional[str] = None
    """Speaker identifier (pseudonymous)"""

    patient_probability: Optional[float] = None
    """Probability this window belongs to patient (0.0-1.0)"""

    attribution_status: str = "UNKNOWN"
    """PATIENT | NON_PATIENT | UNKNOWN | OVERLAP | LOW_CONFIDENCE"""

    attribution_method: Optional[str] = None
    """enrollment | embedding | channel | manual | None"""

    # ===== VAD & segmentation =====
    vad_probability_mean: float = 0.0
    """Average VAD probability across window"""

    voiced_ratio: float = 0.0
    """Proportion of frames classified as voiced (0.0-1.0)"""

    overlap_probability: float = 0.0
    """Probability of speaker overlap (0.0-1.0)"""

    # ===== Acoustic features =====
    egemaps: Optional[list[float]] = None
    """88 eGeMAPSv02 functionals (None if insufficient voiced content)"""

    yamnet_event_scores: dict[str, float] = field(default_factory=dict)
    """YAMNet event probabilities: {event_name: probability}"""

    emotion_embedding: Optional[list[float]] = None
    """emotion2vec acoustic affect embedding (None if unavailable)"""

    # ===== Quality metrics =====
    snr_db: Optional[float] = None
    """Estimated SNR in decibels (None if cannot estimate)"""

    clipping_ratio: float = 0.0
    """Proportion of samples at ±1.0 (0.0-1.0)"""

    dropout_ratio: float = 0.0
    """Proportion of samples indicating dropout/silence (0.0-1.0)"""

    quality_status: str = "UNKNOWN"
    """OK | CLIPPED | LOW_SNR | DROPOUT | REJECTED | INSUFFICIENT_AUDIO"""

    # ===== Reproducibility =====
    extractor_versions: dict[str, str] = field(default_factory=dict)
    """Component versions: {component_name: version_string}"""

    config_hash: str = ""
    """SHA256 hash of configuration used for extraction"""

    model_hashes: dict[str, str] = field(default_factory=dict)
    """Model file hashes: {model_name: sha256_hash}"""

    def __post_init__(self):
        """Validate field constraints"""
        # Validate probabilities
        if self.patient_probability is not None:
            assert (
                0.0 <= self.patient_probability <= 1.0
            ), f"patient_probability must be in [0, 1], got {self.patient_probability}"

        assert (
            0.0 <= self.voiced_ratio <= 1.0
        ), f"voiced_ratio must be in [0, 1], got {self.voiced_ratio}"

        assert (
            0.0 <= self.overlap_probability <= 1.0
        ), f"overlap_probability must be in [0, 1], got {self.overlap_probability}"

        assert (
            0.0 <= self.clipping_ratio <= 1.0
        ), f"clipping_ratio must be in [0, 1], got {self.clipping_ratio}"

        assert (
            0.0 <= self.dropout_ratio <= 1.0
        ), f"dropout_ratio must be in [0, 1], got {self.dropout_ratio}"

        # Validate temporal ordering
        assert (
            self.window_start_ms <= self.window_end_ms
        ), "window_start_ms must be <= window_end_ms"

        assert (
            self.source_start_sample <= self.source_end_sample
        ), "source_start_sample must be <= source_end_sample"

        # Validate attribution_status
        valid_statuses = {
            "PATIENT",
            "NON_PATIENT",
            "UNKNOWN",
            "OVERLAP",
            "LOW_CONFIDENCE",
        }
        assert (
            self.attribution_status in valid_statuses
        ), f"attribution_status must be one of {valid_statuses}, got {self.attribution_status}"

        # Validate quality_status
        valid_quality = {
            "OK",
            "CLIPPED",
            "LOW_SNR",
            "DROPOUT",
            "REJECTED",
            "INSUFFICIENT_AUDIO",
            "UNKNOWN",
        }
        assert (
            self.quality_status in valid_quality
        ), f"quality_status must be one of {valid_quality}, got {self.quality_status}"

        # Validate eGeMAPSv02 dimension
        if self.egemaps is not None:
            assert (
                len(self.egemaps) == 88
            ), f"eGeMAPSv02 must have 88 functionals, got {len(self.egemaps)}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "source_start_sample": self.source_start_sample,
            "source_end_sample": self.source_end_sample,
            "speaker_id": self.speaker_id,
            "patient_probability": self.patient_probability,
            "attribution_status": self.attribution_status,
            "attribution_method": self.attribution_method,
            "vad_probability_mean": self.vad_probability_mean,
            "voiced_ratio": self.voiced_ratio,
            "overlap_probability": self.overlap_probability,
            "egemaps": self.egemaps,
            # Dict columns serialized as JSON strings to avoid PyArrow
            # zero-field struct errors when the dict is empty.
            "yamnet_event_scores": json.dumps(self.yamnet_event_scores),
            "emotion_embedding": self.emotion_embedding,
            "snr_db": self.snr_db,
            "clipping_ratio": self.clipping_ratio,
            "dropout_ratio": self.dropout_ratio,
            "quality_status": self.quality_status,
            "extractor_versions": json.dumps(self.extractor_versions),
            "config_hash": self.config_hash,
            "model_hashes": json.dumps(self.model_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticFeatureRecord":
        """Create from dictionary"""
        return cls(**data)

    def is_patient_speech(self, threshold: float = 0.5) -> bool:
        """
        Check if this window is attributed to patient with confidence > threshold.

        Args:
            threshold: Minimum patient_probability to consider as patient (default: 0.5)

        Returns:
            True if attributed to patient with sufficient confidence
        """
        return (
            self.attribution_status == "PATIENT"
            and self.patient_probability is not None
            and self.patient_probability >= threshold
        )

    def has_quality_issues(self) -> bool:
        """Check if this window has quality issues"""
        return self.quality_status not in {"OK", "UNKNOWN"}

    def is_usable(self, min_patient_prob: float = 0.5) -> bool:
        """
        Check if this record is usable for downstream analysis.

        Args:
            min_patient_prob: Minimum patient probability threshold

        Returns:
            True if record meets quality and attribution criteria
        """
        return (
            not self.has_quality_issues()
            and self.is_patient_speech(threshold=min_patient_prob)
            and self.egemaps is not None  # Has acoustic features
        )


@dataclass
class DropReason:
    """Record explaining why a window was dropped or degraded"""

    window_start_ms: int
    window_end_ms: int
    reason: str  # DROPPED_BACKPRESSURE | MODEL_TIMEOUT | INSUFFICIENT_AUDIO | LOW_QUALITY | SPEAKER_UNKNOWN
    degradation_level: int  # 0-5 (see graceful degradation levels)
    timestamp_ms: int  # When the drop occurred
    additional_info: dict[str, Any] = field(default_factory=dict)

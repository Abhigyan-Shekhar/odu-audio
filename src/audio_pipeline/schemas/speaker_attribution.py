"""
Speaker attribution schema.

Defines how speaker identity is attributed to audio segments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class AttributionStatus(Enum):
    """Speaker attribution status"""

    PATIENT = "PATIENT"
    NON_PATIENT = "NON_PATIENT"
    UNKNOWN = "UNKNOWN"
    OVERLAP = "OVERLAP"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class AttributionMethod(Enum):
    """Method used for speaker attribution"""

    SESSION_ENROLLMENT = "session_enrollment"
    MANUAL_LABEL = "manual_label"
    CHANNEL_ASSIGNMENT = "channel_assignment"
    EMBEDDING_MATCH = "embedding_match"
    POST_HOC_CORRECTION = "post_hoc_correction"
    UNKNOWN = "unknown"


@dataclass
class SpeakerAttribution:
    """
    Speaker attribution result for an audio segment.

    This dataclass encodes:
    - Who is speaking (speaker_id)
    - Confidence that it's the patient (patient_probability)
    - Attribution status (patient/non-patient/unknown/overlap/low-confidence)
    - How the attribution was determined (method)
    - Whether multiple speakers are detected (overlap)
    - Whether this is a provisional or final attribution (provisional)
    """

    speaker_id: str
    """Speaker identifier (pseudonymous)"""

    patient_probability: float
    """Probability this speaker is the patient (0.0-1.0)"""

    status: AttributionStatus
    """Attribution status enum"""

    attribution_method: AttributionMethod
    """Method used to determine attribution"""

    overlap: bool
    """True if multiple speakers detected in this segment"""

    provisional: bool = False
    """
    True if this is a provisional (streaming) attribution that may be refined.
    False if this is a final (batch) attribution.
    """

    embedding_distance: Optional[float] = None
    """Distance to reference embedding (if embedding_match method used)"""

    confidence_factors: Optional[dict[str, Any]] = None
    """Additional confidence factors: {factor_name: value}"""

    def __post_init__(self):
        """Validate field constraints"""
        assert 0.0 <= self.patient_probability <= 1.0, (
            f"patient_probability must be in [0, 1], got {self.patient_probability}"
        )

        if self.confidence_factors is None:
            self.confidence_factors = {}

        # Validate embedding distance if provided
        if self.embedding_distance is not None:
            assert self.embedding_distance >= 0.0, (
                f"embedding_distance must be >= 0, got {self.embedding_distance}"
            )

    def is_patient(self, threshold: float = 0.5) -> bool:
        """
        Check if this attribution indicates patient speech.

        Args:
            threshold: Minimum probability to consider as patient

        Returns:
            True if status is PATIENT and probability >= threshold
        """
        return (
            self.status == AttributionStatus.PATIENT
            and self.patient_probability >= threshold
        )

    def is_reliable(self, min_confidence: float = 0.7) -> bool:
        """
        Check if this attribution is reliable enough for use.

        Args:
            min_confidence: Minimum patient_probability for reliability

        Returns:
            True if attribution is reliable (not UNKNOWN/LOW_CONFIDENCE and meets threshold)
        """
        return (
            self.status
            not in {AttributionStatus.UNKNOWN, AttributionStatus.LOW_CONFIDENCE}
            and not self.overlap
            and self.patient_probability >= min_confidence
        )

    def should_abstain(self) -> bool:
        """Check if system should abstain from using this attribution"""
        return self.status in {
            AttributionStatus.UNKNOWN,
            AttributionStatus.LOW_CONFIDENCE,
            AttributionStatus.OVERLAP,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "speaker_id": self.speaker_id,
            "patient_probability": self.patient_probability,
            "status": self.status.value,
            "attribution_method": self.attribution_method.value,
            "overlap": self.overlap,
            "provisional": self.provisional,
            "embedding_distance": self.embedding_distance,
            "confidence_factors": self.confidence_factors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerAttribution":
        """Create from dictionary"""
        return cls(
            speaker_id=data["speaker_id"],
            patient_probability=data["patient_probability"],
            status=AttributionStatus(data["status"]),
            attribution_method=AttributionMethod(data["attribution_method"]),
            overlap=data["overlap"],
            provisional=data.get("provisional", False),
            embedding_distance=data.get("embedding_distance"),
            confidence_factors=data.get("confidence_factors", {}),
        )

    @classmethod
    def unknown(cls, reason: str = "no_attribution_available") -> "SpeakerAttribution":
        """Create an UNKNOWN attribution"""
        return cls(
            speaker_id="unknown",
            patient_probability=0.0,
            status=AttributionStatus.UNKNOWN,
            attribution_method=AttributionMethod.UNKNOWN,
            overlap=False,
            provisional=True,
            confidence_factors={"reason": reason},
        )

    @classmethod
    def overlap_detected(cls, speaker_ids: list[str]) -> "SpeakerAttribution":
        """Create an OVERLAP attribution"""
        return cls(
            speaker_id=",".join(sorted(speaker_ids)),
            patient_probability=0.0,
            status=AttributionStatus.OVERLAP,
            attribution_method=AttributionMethod.UNKNOWN,
            overlap=True,
            provisional=True,
            confidence_factors={"speaker_count": len(speaker_ids)},
        )


@dataclass
class EnrollmentConfig:
    """
    Configuration for patient enrollment.

    Specifies how the system identifies which speaker is the patient.
    """

    method: AttributionMethod
    """Primary enrollment method"""

    duration_seconds: float = 5.0
    """Duration of enrollment utterance (for session_enrollment method)"""

    channel_index: Optional[int] = None
    """Microphone channel assigned to patient (for channel_assignment method)"""

    reference_embedding_id: Optional[str] = None
    """ID of reference embedding to match against (for embedding_match method)"""

    confidence_threshold: float = 0.7
    """Minimum confidence for successful enrollment"""

    allow_fallback: bool = True
    """Allow fallback to manual labeling if automatic enrollment fails"""

    def validate(self):
        """Validate configuration consistency"""
        if self.method == AttributionMethod.SESSION_ENROLLMENT:
            assert self.duration_seconds > 0, "duration_seconds must be positive"

        if self.method == AttributionMethod.CHANNEL_ASSIGNMENT:
            assert self.channel_index is not None, (
                "channel_index required for channel_assignment"
            )
            assert self.channel_index >= 0, "channel_index must be non-negative"

        if self.method == AttributionMethod.EMBEDDING_MATCH:
            assert self.reference_embedding_id is not None, (
                "reference_embedding_id required for embedding_match"
            )

        assert 0.0 < self.confidence_threshold <= 1.0, (
            "confidence_threshold must be in (0, 1]"
        )

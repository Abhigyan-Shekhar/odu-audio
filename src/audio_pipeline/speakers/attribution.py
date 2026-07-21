"""
Speaker attribution: maps diarized speaker turns to patient/non-patient decisions.

Critical guarantee: ambiguous or overlapping speech always produces
UNKNOWN or OVERLAP status — never a forced PATIENT attribution.
"""

import logging
from typing import Optional

import numpy as np

from src.audio_pipeline.schemas.segment import SpeechSegment
from src.audio_pipeline.schemas.speaker_attribution import (
    AttributionMethod,
    AttributionStatus,
    SpeakerAttribution,
)
from src.audio_pipeline.speakers.diarizer_interface import (
    DiarizationResult,
    DiarizeriInterface,
)
from src.audio_pipeline.speakers.enrollment import SessionEnrollment

logger = logging.getLogger(__name__)


class SpeakerAttributor:
    """
    Attribute speech segments to patient or non-patient.

    Decision logic (in priority order):
    1. No speaker turns detected           → UNKNOWN
    2. Speaker overlap detected            → OVERLAP
    3. No patient enrolled                 → UNKNOWN
    4. Embedding comparison fails          → UNKNOWN
    5. Similarity < confidence_threshold   → LOW_CONFIDENCE
    6. Similarity ≥ 0.5 → PATIENT; else   → NON_PATIENT
    """

    def __init__(
        self,
        diarizer: DiarizeriInterface,
        enrollment: Optional[SessionEnrollment] = None,
        confidence_threshold: float = 0.70,
        method: AttributionMethod = AttributionMethod.SESSION_ENROLLMENT,
    ):
        """
        Args:
            diarizer:             Diarization backend to segment speaker turns.
            enrollment:           Patient enrollment state. If None, all
                                  attributions will be UNKNOWN.
            confidence_threshold: Minimum cosine similarity to accept as a
                                  confident match (≥ 0.5 → PATIENT/NON_PATIENT,
                                  < threshold → LOW_CONFIDENCE).
            method:               Attribution method to record on the result.
        """
        self.diarizer = diarizer
        self.enrollment = enrollment
        self.confidence_threshold = confidence_threshold
        self.method = method

    def attribute(
        self,
        segment: SpeechSegment,
        audio: np.ndarray,
        sr: int,
        speaker_embedding: Optional[np.ndarray] = None,
    ) -> SpeakerAttribution:
        """
        Attribute a speech segment to a speaker.

        Args:
            segment:           The speech segment to attribute.
            audio:             Raw audio for the segment (used for diarization).
            sr:                Sample rate (expected 16000 Hz).
            speaker_embedding: Pre-computed speaker embedding for the dominant
                               speaker. If None, enrollment.get_embedding_for_speaker
                               is called internally (requires pyannote embedding model).

        Returns:
            SpeakerAttribution with explicit status — never silently forced to PATIENT.
        """
        # Step 1: Run diarizer
        diarization: DiarizationResult = self.diarizer.diarize(audio, sr)

        # Step 2: No speech detected
        if not diarization.turns:
            return SpeakerAttribution.unknown(reason="no_speaker_detected")

        # Step 3: Overlap — cannot safely attribute to patient
        if diarization.has_overlap:
            return SpeakerAttribution.overlap_detected(
                speaker_ids=diarization.overlapping_speaker_ids
                or diarization.speakers()
            )

        # Step 4: No patient enrolled — cannot make an attribution decision
        if self.enrollment is None or not self.enrollment.is_enrolled():
            dominant = diarization.dominant_speaker()
            return SpeakerAttribution(
                speaker_id=dominant or "unknown",
                patient_probability=0.0,
                status=AttributionStatus.UNKNOWN,
                attribution_method=self.method,
                overlap=False,
                provisional=True,
                confidence_factors={"reason": "no_patient_enrollment"},
            )

        # Step 5: Get speaker embedding for the dominant speaker
        dominant = diarization.dominant_speaker()
        if dominant is None:
            return SpeakerAttribution.unknown(reason="no_dominant_speaker")

        if speaker_embedding is None:
            # Attempt to extract embedding directly
            if self.enrollment is not None:
                speaker_embedding = self.enrollment.get_embedding_for_speaker(audio, sr)

        if speaker_embedding is None:
            return SpeakerAttribution(
                speaker_id=dominant,
                patient_probability=0.0,
                status=AttributionStatus.UNKNOWN,
                attribution_method=self.method,
                overlap=False,
                provisional=True,
                confidence_factors={"reason": "embedding_extraction_failed"},
            )

        # Step 6: Compare to enrolled patient embedding
        similarity = self.enrollment.compare(speaker_embedding)

        # Step 7: Low confidence guard — must pass threshold to get a decision
        if similarity < self.confidence_threshold:
            return SpeakerAttribution(
                speaker_id=dominant,
                patient_probability=similarity,
                status=AttributionStatus.LOW_CONFIDENCE,
                attribution_method=self.method,
                overlap=False,
                provisional=False,
                embedding_distance=1.0 - similarity,
                confidence_factors={
                    "cosine_similarity": similarity,
                    "threshold": self.confidence_threshold,
                },
            )

        # Step 8: High-confidence decision
        # similarity ≥ 0.5 → PATIENT; < 0.5 → NON_PATIENT
        # (Both require similarity ≥ confidence_threshold to reach this branch)
        status = (
            AttributionStatus.PATIENT
            if similarity >= 0.5
            else AttributionStatus.NON_PATIENT
        )
        return SpeakerAttribution(
            speaker_id=dominant,
            patient_probability=similarity,
            status=status,
            attribution_method=self.method,
            overlap=False,
            provisional=False,
            embedding_distance=1.0 - similarity,
            confidence_factors={
                "cosine_similarity": similarity,
                "threshold": self.confidence_threshold,
            },
        )

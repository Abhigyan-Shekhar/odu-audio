"""
Unit tests for the SpeakerAttributor and supporting diarization components.
"""

import numpy as np
import pytest

from src.audio_pipeline.schemas.segment import SpeechSegment
from src.audio_pipeline.schemas.speaker_attribution import AttributionStatus
from src.audio_pipeline.speakers.attribution import SpeakerAttributor
from src.audio_pipeline.speakers.dummy_diarizer import DummyDiarizer
from src.audio_pipeline.speakers.enrollment import SessionEnrollment


def _make_segment(start_ms: int = 0, end_ms: int = 2000) -> SpeechSegment:
    """Helper: Create a dummy SpeechSegment for testing."""
    return SpeechSegment(
        start_ms=start_ms,
        end_ms=end_ms,
        start_sample=start_ms * 16,
        end_sample=end_ms * 16,
        vad_probability_mean=0.85,
        vad_probability_min=0.70,
        session_id="sess_test",
        stream_id="stream_test",
        provisional=False,
    )


def _make_audio(duration_ms: int = 2000, sr: int = 16000) -> np.ndarray:
    """Helper: Create a zero-filled mono audio array."""
    return np.zeros(int(duration_ms / 1000 * sr), dtype=np.float32)


def _make_enrollment(similarity_score: float = 0.85) -> SessionEnrollment:
    """
    Helper: Create an enrollment with a fake reference embedding and
    mock compare() that returns the given fixed similarity score.
    """
    enroll = SessionEnrollment(confidence_threshold=0.70)
    enroll.enroll_from_embedding(np.ones(192))
    # Monkey-patch compare() to return a controlled score
    enroll.compare = lambda _: similarity_score  # type: ignore[method-assign]
    return enroll


def _make_attributor(
    simulate_overlap: bool = False,
    simulate_no_speech: bool = False,
    enrollment: "SessionEnrollment | None" = None,
    confidence_threshold: float = 0.70,
) -> SpeakerAttributor:
    diarizer = DummyDiarizer(
        simulate_overlap=simulate_overlap,
        simulate_no_speech=simulate_no_speech,
    )
    return SpeakerAttributor(
        diarizer=diarizer,
        enrollment=enrollment,
        confidence_threshold=confidence_threshold,
    )


# ─── Test cases ────────────────────────────────────────────────────────────────


def test_attribution_high_confidence():
    """Similarity 0.9 ≥ threshold 0.70 and ≥ 0.5 → PATIENT."""
    enroll = _make_enrollment(similarity_score=0.90)
    attributor = _make_attributor(enrollment=enroll)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.PATIENT
    assert result.patient_probability == pytest.approx(0.90)
    assert result.overlap is False
    assert result.provisional is False


def test_attribution_low_confidence():
    """Similarity 0.55 below threshold 0.70 → LOW_CONFIDENCE, not PATIENT."""
    enroll = _make_enrollment(similarity_score=0.55)
    attributor = _make_attributor(enrollment=enroll, confidence_threshold=0.70)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.LOW_CONFIDENCE
    assert result.patient_probability == pytest.approx(0.55)
    assert result.overlap is False


def test_attribution_overlap():
    """Overlapping speaker turns → OVERLAP status regardless of enrollment."""
    enroll = _make_enrollment(similarity_score=0.95)
    attributor = _make_attributor(simulate_overlap=True, enrollment=enroll)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.OVERLAP
    assert result.overlap is True
    assert result.patient_probability == 0.0


def test_attribution_unknown_no_speech():
    """No speech detected by diarizer → UNKNOWN."""
    enroll = _make_enrollment(similarity_score=0.95)
    attributor = _make_attributor(simulate_no_speech=True, enrollment=enroll)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.UNKNOWN
    assert result.patient_probability == 0.0


def test_attribution_non_patient():
    """Similarity 0.72 ≥ threshold but < 0.5 → NON_PATIENT.
    Test uses threshold=0.40 so similarity 0.45 passes threshold but < 0.5."""
    enroll = _make_enrollment(similarity_score=0.45)
    attributor = _make_attributor(enrollment=enroll, confidence_threshold=0.40)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.NON_PATIENT
    assert result.patient_probability == pytest.approx(0.45)
    assert result.overlap is False


def test_attribution_unknown_no_enrollment():
    """No enrollment provided → always UNKNOWN, even with a detected speaker."""
    attributor = _make_attributor(enrollment=None)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.UNKNOWN
    assert result.patient_probability == 0.0


def test_ambiguous_segment_not_forced_to_patient():
    """
    Critical: An ambiguous segment (similarity 0.30, threshold 0.70) must
    produce LOW_CONFIDENCE, not PATIENT. The system must never force an
    attribution when the evidence is weak.
    """
    enroll = _make_enrollment(similarity_score=0.30)
    attributor = _make_attributor(enrollment=enroll, confidence_threshold=0.70)
    audio = _make_audio()
    result = attributor.attribute(
        _make_segment(), audio, sr=16000, speaker_embedding=np.ones(192)
    )
    assert result.status == AttributionStatus.LOW_CONFIDENCE
    assert result.status != AttributionStatus.PATIENT
    assert result.patient_probability == pytest.approx(0.30)


def test_diarization_result_helpers():
    """Verify DiarizationResult helper methods work correctly."""
    from src.audio_pipeline.speakers.diarizer_interface import (
        DiarizationResult,
        DiarizedTurn,
    )

    result = DiarizationResult(
        turns=[
            DiarizedTurn(start_ms=0, end_ms=1000, speaker_id="SPEAKER_00"),
            DiarizedTurn(start_ms=1000, end_ms=2000, speaker_id="SPEAKER_01"),
            DiarizedTurn(start_ms=500, end_ms=1500, speaker_id="SPEAKER_00"),
        ],
        num_speakers=2,
    )
    assert sorted(result.speakers()) == ["SPEAKER_00", "SPEAKER_01"]
    # SPEAKER_00 has 1000 + 1000 = 2000ms, SPEAKER_01 has 1000ms
    assert result.dominant_speaker() == "SPEAKER_00"
    # Turns overlapping [800, 1200]
    overlap_turns = result.turns_for_window(800, 1200)
    assert len(overlap_turns) == 3


def test_dummy_diarizer_basic():
    """DummyDiarizer returns expected structure for a 2-second segment."""
    from src.audio_pipeline.speakers.dummy_diarizer import DummyDiarizer

    diarizer = DummyDiarizer(speaker_id="SPK_0")
    audio = np.zeros(32000, dtype=np.float32)
    result = diarizer.diarize(audio, sr=16000)

    assert result.num_speakers == 1
    assert not result.has_overlap
    assert result.turns[0].speaker_id == "SPK_0"
    assert result.turns[0].duration_ms == 2000


def test_session_enrollment_cosine_similarity():
    """SessionEnrollment correctly computes cosine similarity."""
    enroll = SessionEnrollment()
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    enroll.enroll_from_embedding(ref)

    # Identical vector → similarity 1.0
    assert enroll.compare(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    # Perpendicular vector → similarity clamped to 0.0
    assert enroll.compare(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_session_enrollment_clear():
    """Clearing enrollment resets state."""
    enroll = SessionEnrollment()
    enroll.enroll_from_embedding(np.ones(128))
    assert enroll.is_enrolled()
    enroll.clear()
    assert not enroll.is_enrolled()
    assert enroll.compare(np.ones(128)) == 0.0

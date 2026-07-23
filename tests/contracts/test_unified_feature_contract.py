"""Contract tests for one-second unified fusion records."""

import pytest

from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord


def test_unified_feature_record_excludes_transcript(
    sample_session_id, sample_stream_id
):
    record = UnifiedFeatureRecord(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        speaker_id="speaker-1",
        tick_start_ms=0,
        tick_end_ms=1000,
        subject_id="subject-1",
        attribution_status="PATIENT",
        patient_probability=0.9,
        speaker_confidence=0.9,
        egemaps=[0.1] * 88,
        profanity_probability=0.4,
        threat_probability=0.5,
        distress_phrase_probability=0.6,
        toxicity_probability=0.7,
        asr_confidence=0.8,
        language="en",
        acoustic_arousal_probability=0.9,
        pitch_energy_arousal_probability=0.2,
        scream_shout_probability=0.3,
        acoustic_missing=False,
        lexical_missing=False,
        source_hashes={"lexical": "abc123"},
    ).with_interactions()

    payload = record.to_dict()
    assert "transcript" not in payload
    assert payload["subject_id"] == "subject-1"
    assert payload["t0"] == 0
    assert payload["t1"] == 1000
    assert payload["language"] == "en"
    assert payload["source_hashes"] == {"lexical": "abc123"}
    assert record.acoustic_x_profanity == pytest.approx(0.36)
    assert record.acoustic_arousal_x_profanity == pytest.approx(0.36)
    assert record.pitch_energy_x_threat == pytest.approx(0.1)
    assert record.repetition_x_arousal is None
    assert record.asr_conf_x_profanity == pytest.approx(0.32)
    assert record.arousal_x_threat == pytest.approx(0.45)
    assert record.arousal_x_distress == pytest.approx(0.54)
    assert record.scream_x_distress == pytest.approx(0.18)
    assert record.toxicity_x_asr_confidence == pytest.approx(0.56)


def test_unified_feature_record_validates_probability(
    sample_session_id, sample_stream_id
):
    with pytest.raises(AssertionError, match="profanity_probability must be in"):
        UnifiedFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            speaker_id=None,
            tick_start_ms=0,
            tick_end_ms=1000,
            profanity_probability=1.2,
        )


def test_unified_feature_record_validates_egemaps(sample_session_id, sample_stream_id):
    with pytest.raises(AssertionError, match="eGeMAPSv02 must have 88 functionals"):
        UnifiedFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            speaker_id=None,
            tick_start_ms=0,
            tick_end_ms=1000,
            egemaps=[0.1] * 4,
        )

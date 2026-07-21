"""
Unit tests verifying SpeakerAttribution contract properties and enrollment config constraints.
"""

import pytest
from src.audio_pipeline.schemas.speaker_attribution import (
    SpeakerAttribution,
    AttributionStatus,
    AttributionMethod,
    EnrollmentConfig,
)


def test_speaker_attribution_instantiation():
    attr = SpeakerAttribution(
        speaker_id="spk_001",
        patient_probability=0.9,
        status=AttributionStatus.PATIENT,
        attribution_method=AttributionMethod.SESSION_ENROLLMENT,
        overlap=False,
    )
    assert attr.speaker_id == "spk_001"
    assert attr.patient_probability == 0.9
    assert attr.status == AttributionStatus.PATIENT
    assert attr.attribution_method == AttributionMethod.SESSION_ENROLLMENT
    assert not attr.overlap
    assert not attr.provisional


def test_speaker_attribution_invalid_values():
    with pytest.raises(AssertionError):
        SpeakerAttribution(
            speaker_id="spk_001",
            patient_probability=1.5,  # > 1.0
            status=AttributionStatus.PATIENT,
            attribution_method=AttributionMethod.SESSION_ENROLLMENT,
            overlap=False,
        )

    with pytest.raises(AssertionError):
        SpeakerAttribution(
            speaker_id="spk_001",
            patient_probability=0.9,
            status=AttributionStatus.PATIENT,
            attribution_method=AttributionMethod.SESSION_ENROLLMENT,
            overlap=False,
            embedding_distance=-0.5,  # < 0
        )


def test_speaker_attribution_helper_methods():
    attr_patient = SpeakerAttribution(
        speaker_id="spk_001",
        patient_probability=0.9,
        status=AttributionStatus.PATIENT,
        attribution_method=AttributionMethod.SESSION_ENROLLMENT,
        overlap=False,
    )
    assert attr_patient.is_patient(threshold=0.5)
    assert attr_patient.is_reliable(min_confidence=0.8)
    assert not attr_patient.should_abstain()

    attr_unknown = SpeakerAttribution.unknown(reason="low_energy")
    assert not attr_unknown.is_patient()
    assert not attr_unknown.is_reliable()
    assert attr_unknown.should_abstain()
    assert attr_unknown.speaker_id == "unknown"

    attr_overlap = SpeakerAttribution.overlap_detected(speaker_ids=["spk_1", "spk_2"])
    assert attr_overlap.overlap
    assert attr_overlap.status == AttributionStatus.OVERLAP
    assert attr_overlap.speaker_id == "spk_1,spk_2"


def test_speaker_attribution_serialization():
    attr = SpeakerAttribution(
        speaker_id="spk_001",
        patient_probability=0.9,
        status=AttributionStatus.PATIENT,
        attribution_method=AttributionMethod.SESSION_ENROLLMENT,
        overlap=False,
    )
    d = attr.to_dict()
    assert d["speaker_id"] == "spk_001"
    assert d["status"] == "PATIENT"
    
    attr2 = SpeakerAttribution.from_dict(d)
    assert attr2.speaker_id == attr.speaker_id
    assert attr2.status == attr.status
    assert attr2.attribution_method == attr.attribution_method


def test_enrollment_config_validations():
    # Valid config
    config = EnrollmentConfig(
        method=AttributionMethod.SESSION_ENROLLMENT,
        duration_seconds=5.0,
    )
    config.validate()

    # Invalid duration_seconds
    config_invalid_duration = EnrollmentConfig(
        method=AttributionMethod.SESSION_ENROLLMENT,
        duration_seconds=-1.0,
    )
    with pytest.raises(AssertionError):
        config_invalid_duration.validate()

    # Invalid channel assignment missing channel_index
    config_invalid_channel = EnrollmentConfig(
        method=AttributionMethod.CHANNEL_ASSIGNMENT,
    )
    with pytest.raises(AssertionError):
        config_invalid_channel.validate()

    # Invalid embedding match missing reference_embedding_id
    config_invalid_embedding = EnrollmentConfig(
        method=AttributionMethod.EMBEDDING_MATCH,
    )
    with pytest.raises(AssertionError):
        config_invalid_embedding.validate()

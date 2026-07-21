"""
Unit tests verifying AcousticFeatureRecord contract properties, constraints, and helper methods.
"""

import pytest

from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord, DropReason


def test_feature_record_instantiation(sample_session_id, sample_stream_id):
    record = AcousticFeatureRecord(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        window_start_ms=0,
        window_end_ms=2000,
        source_start_sample=0,
        source_end_sample=32000,
        attribution_status="PATIENT",
        patient_probability=0.8,
        egemaps=[0.1] * 88,
    )
    assert record.session_id == sample_session_id
    assert record.voiced_ratio == 0.0
    assert record.egemaps is not None
    assert len(record.egemaps) == 88


def test_feature_record_probability_constraints(sample_session_id, sample_stream_id):
    # invalid patient_probability
    with pytest.raises(AssertionError, match="patient_probability must be in"):
        AcousticFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            window_start_ms=0,
            window_end_ms=2000,
            source_start_sample=0,
            source_end_sample=32000,
            patient_probability=1.2,
        )

    # invalid voiced_ratio
    with pytest.raises(AssertionError, match="voiced_ratio must be in"):
        AcousticFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            window_start_ms=0,
            window_end_ms=2000,
            source_start_sample=0,
            source_end_sample=32000,
            voiced_ratio=-0.1,
        )


def test_feature_record_temporal_constraints(sample_session_id, sample_stream_id):
    with pytest.raises(
        AssertionError, match="window_start_ms must be <= window_end_ms"
    ):
        AcousticFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            window_start_ms=1000,
            window_end_ms=500,
            source_start_sample=0,
            source_end_sample=32000,
        )


def test_feature_record_invalid_egemaps_dimension(sample_session_id, sample_stream_id):
    with pytest.raises(AssertionError, match="eGeMAPSv02 must have 88 functionals"):
        AcousticFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            window_start_ms=0,
            window_end_ms=2000,
            source_start_sample=0,
            source_end_sample=32000,
            egemaps=[0.1] * 10,  # 10 instead of 88
        )


def test_feature_record_serialization(sample_session_id, sample_stream_id):
    record = AcousticFeatureRecord(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        window_start_ms=0,
        window_end_ms=2000,
        source_start_sample=0,
        source_end_sample=32000,
        attribution_status="PATIENT",
        patient_probability=0.8,
        egemaps=[0.1] * 88,
    )

    d = record.to_dict()
    assert d["session_id"] == sample_session_id
    assert d["patient_probability"] == 0.8
    assert d["egemaps"] == [0.1] * 88

    record2 = AcousticFeatureRecord.from_dict(d)
    assert record2.session_id == record.session_id
    assert record2.patient_probability == record.patient_probability
    assert record2.egemaps == record.egemaps


def test_feature_record_helper_methods(sample_session_id, sample_stream_id):
    record_ok = AcousticFeatureRecord(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        window_start_ms=0,
        window_end_ms=2000,
        source_start_sample=0,
        source_end_sample=32000,
        attribution_status="PATIENT",
        patient_probability=0.8,
        egemaps=[0.1] * 88,
        quality_status="OK",
    )

    assert record_ok.is_patient_speech(threshold=0.5)
    assert not record_ok.has_quality_issues()
    assert record_ok.is_usable(min_patient_prob=0.5)

    # Degraded/poor quality
    record_clipped = AcousticFeatureRecord(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        window_start_ms=0,
        window_end_ms=2000,
        source_start_sample=0,
        source_end_sample=32000,
        attribution_status="PATIENT",
        patient_probability=0.8,
        egemaps=[0.1] * 88,
        quality_status="CLIPPED",
    )
    assert record_clipped.has_quality_issues()
    assert not record_clipped.is_usable()


def test_drop_reason_instantiation():
    drop = DropReason(
        window_start_ms=0,
        window_end_ms=1000,
        reason="DROPPED_BACKPRESSURE",
        degradation_level=3,
        timestamp_ms=160000000,
    )
    assert drop.reason == "DROPPED_BACKPRESSURE"
    assert drop.degradation_level == 3

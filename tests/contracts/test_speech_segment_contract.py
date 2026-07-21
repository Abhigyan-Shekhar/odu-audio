"""
Unit tests verifying SpeechSegment contract properties and constraints.
"""

import pytest
from src.audio_pipeline.schemas.segment import SpeechSegment


def test_speech_segment_instantiation(sample_speech_segment, sample_session_id, sample_stream_id):
    assert sample_speech_segment.start_ms == 100
    assert sample_speech_segment.end_ms == 1100
    assert sample_speech_segment.start_sample == 1600
    assert sample_speech_segment.end_sample == 17600
    assert sample_speech_segment.vad_probability_mean == 0.95
    assert sample_speech_segment.vad_probability_min == 0.85
    assert not sample_speech_segment.provisional
    assert sample_speech_segment.session_id == sample_session_id
    assert sample_speech_segment.stream_id == sample_stream_id


def test_speech_segment_timing_order(sample_session_id, sample_stream_id):
    # end_ms < start_ms
    with pytest.raises(AssertionError, match="start_ms must be <= end_ms"):
        SpeechSegment(
            start_ms=1000,
            end_ms=500,
            start_sample=1600,
            end_sample=17600,
            vad_probability_mean=0.9,
            vad_probability_min=0.8,
            provisional=False,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )

    # end_sample < start_sample
    with pytest.raises(AssertionError, match="start_sample must be <= end_sample"):
        SpeechSegment(
            start_ms=100,
            end_ms=1100,
            start_sample=16000,
            end_sample=1600,
            vad_probability_mean=0.9,
            vad_probability_min=0.8,
            provisional=False,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )


def test_speech_segment_vad_bounds(sample_session_id, sample_stream_id):
    # VAD probability > 1.0
    with pytest.raises(AssertionError, match="vad_probability_mean must be in"):
        SpeechSegment(
            start_ms=100,
            end_ms=1100,
            start_sample=1600,
            end_sample=17600,
            vad_probability_mean=1.05,
            vad_probability_min=0.85,
            provisional=False,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )

    # VAD probability < 0.0
    with pytest.raises(AssertionError, match="vad_probability_min must be in"):
        SpeechSegment(
            start_ms=100,
            end_ms=1100,
            start_sample=1600,
            end_sample=17600,
            vad_probability_mean=0.95,
            vad_probability_min=-0.1,
            provisional=False,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )


def test_speech_segment_computed_properties(sample_speech_segment):
    assert sample_speech_segment.duration_ms == 1000
    assert sample_speech_segment.n_samples == 16000

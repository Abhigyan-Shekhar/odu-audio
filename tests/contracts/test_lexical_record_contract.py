"""Contract tests for lexical feature records."""

import pytest

from src.audio_pipeline.schemas.lexical_record import (
    LexicalFeatureRecord,
    TranscriptSegment,
    WordTimestamp,
)


def test_word_timestamp_constraints():
    word = WordTimestamp(word="hello", start_ms=0, end_ms=300, confidence=0.8)
    assert word.word == "hello"

    with pytest.raises(AssertionError, match="confidence must be in"):
        WordTimestamp(word="hello", start_ms=0, end_ms=300, confidence=1.2)


def test_transcript_segment_contract(sample_session_id, sample_stream_id):
    segment = TranscriptSegment(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        start_ms=1000,
        end_ms=2500,
        text="help me",
        language="en",
        language_probability=0.8,
        avg_confidence=0.7,
    )

    assert segment.text == "help me"
    assert segment.avg_confidence == 0.7


def test_lexical_feature_record_serialization(sample_session_id, sample_stream_id):
    record = LexicalFeatureRecord(
        session_id=sample_session_id,
        stream_id=sample_stream_id,
        window_start_ms=0,
        window_end_ms=2000,
        transcript="help me",
        normalized_transcript="help me",
        language="en",
        language_probability=0.8,
        asr_confidence=0.7,
        word_timestamps=[
            WordTimestamp(word="help", start_ms=0, end_ms=300, confidence=0.7)
        ],
        distress_phrase_probability=0.85,
    )

    payload = record.to_dict()
    restored = LexicalFeatureRecord.from_dict(payload)

    assert restored.session_id == record.session_id
    assert restored.word_timestamps[0].word == "help"
    assert restored.distress_phrase_probability == 0.85


def test_lexical_feature_record_probability_constraints(
    sample_session_id, sample_stream_id
):
    with pytest.raises(AssertionError, match="toxicity_probability must be"):
        LexicalFeatureRecord(
            session_id=sample_session_id,
            stream_id=sample_stream_id,
            window_start_ms=0,
            window_end_ms=2000,
            transcript="text",
            normalized_transcript="text",
            toxicity_probability=1.5,
        )

"""Tests for acoustic/lexical timestamp alignment."""

from src.audio_pipeline.fusion.alignment import align_to_fusion_ticks
from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord
from src.audio_pipeline.schemas.lexical_record import LexicalFeatureRecord


def _acoustic(start_ms: int, end_ms: int, arousal: float) -> AcousticFeatureRecord:
    return AcousticFeatureRecord(
        session_id="session-1",
        stream_id="stream-1",
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        source_start_sample=start_ms * 16,
        source_end_sample=end_ms * 16,
        speaker_id="speaker-1",
        patient_probability=0.8,
        attribution_status="PATIENT",
        overlap_probability=0.1,
        egemaps=[arousal] * 88,
        yamnet_event_scores={"Shout": arousal, "Screaming": arousal / 2},
        snr_db=20.0,
        quality_status="OK",
    )


def _lexical(start_ms: int, end_ms: int) -> LexicalFeatureRecord:
    return LexicalFeatureRecord(
        session_id="session-1",
        stream_id="stream-1",
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        transcript="not carried into unified record",
        normalized_transcript="not carried into unified record",
        asr_confidence=0.75,
        profanity_probability=0.6,
        threat_probability=0.4,
        distress_phrase_probability=0.5,
        toxicity_probability=0.7,
        mutox_text_score=0.65,
    )


def test_aligns_overlapping_windows_to_one_second_grid():
    records = align_to_fusion_ticks(
        acoustic_records=[_acoustic(0, 2000, 0.2), _acoustic(1000, 3000, 0.8)],
        lexical_records=[_lexical(500, 1500)],
    )

    assert [record.tick_start_ms for record in records] == [0, 1000, 2000]
    assert records[0].profanity_probability == 0.6
    assert records[1].profanity_probability == 0.6
    assert records[2].profanity_probability is None
    assert records[2].feature_missing_mask["lexical"] is True
    assert records[2].lexical_missing is True
    assert records[1].acoustic_arousal_probability == 0.5
    assert records[1].acoustic_x_profanity == 0.3
    assert records[1].acoustic_arousal_x_profanity == 0.3
    assert records[1].pitch_energy_x_threat == 0.2
    assert records[1].asr_conf_x_profanity == 0.44999999999999996
    assert records[1].mutox_text_score == 0.65


def test_does_not_forward_fill_lexical_evidence():
    records = align_to_fusion_ticks(
        acoustic_records=[_acoustic(0, 4000, 0.5)],
        lexical_records=[_lexical(0, 1000)],
    )

    assert records[0].threat_probability == 0.4
    assert records[1].threat_probability is None
    assert records[2].threat_probability is None
    assert records[3].threat_probability is None

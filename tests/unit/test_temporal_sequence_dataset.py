"""Tests for Phase 2 temporal sequence construction."""

import numpy as np

from src.audio_pipeline.models.sequence_dataset import (
    SequenceBuilderConfig,
    TemporalSequenceDataset,
    build_temporal_sequences,
    default_temporal_feature_names,
)
from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord


def _record(
    subject_id: str,
    session_id: str,
    tick: int,
    lexical_missing: bool = False,
) -> UnifiedFeatureRecord:
    lexical_value = None if lexical_missing else 0.4
    return UnifiedFeatureRecord(
        session_id=session_id,
        stream_id="stream-1",
        subject_id=subject_id,
        speaker_id=f"{subject_id}-speaker",
        tick_start_ms=tick * 1000,
        tick_end_ms=(tick + 1) * 1000,
        patient_probability=0.8,
        acoustic_arousal_probability=0.3 + tick * 0.01,
        profanity_probability=lexical_value,
        threat_probability=lexical_value,
        asr_confidence=None if lexical_missing else 0.9,
        acoustic_missing=False,
        lexical_missing=lexical_missing,
        feature_missing_mask={
            "acoustic": False,
            "lexical": lexical_missing,
            "profanity_probability": lexical_missing,
        },
    ).with_interactions()


def test_builds_chronological_fixed_length_windows_with_stride():
    records = [_record("s1", "a", tick) for tick in [2, 0, 1, 3, 4]]
    config = SequenceBuilderConfig(sequence_length=3, stride=2)

    sequences = build_temporal_sequences(records, config=config)

    assert len(sequences) == 2
    assert sequences[0].metadata.record_start_ms == [0, 1000, 2000]
    assert sequences[1].metadata.record_start_ms == [2000, 3000, 4000]
    assert sequences[0].features.shape == (3, len(default_temporal_feature_names()))


def test_prevents_cross_subject_and_session_windows():
    records = [
        _record("s1", "a", 0),
        _record("s1", "a", 1),
        _record("s1", "b", 2),
        _record("s1", "b", 3),
        _record("s2", "a", 0),
        _record("s2", "a", 1),
    ]

    sequences = build_temporal_sequences(
        records,
        config=SequenceBuilderConfig(sequence_length=2, stride=1),
    )

    assert len(sequences) == 3
    assert {
        (item.metadata.subject_id, item.metadata.session_id) for item in sequences
    } == {
        ("s1", "a"),
        ("s1", "b"),
        ("s2", "a"),
    }


def test_splits_at_non_consecutive_tick_gap():
    records = [_record("s1", "a", 0), _record("s1", "a", 1), _record("s1", "a", 4)]

    sequences = build_temporal_sequences(
        records,
        config=SequenceBuilderConfig(sequence_length=3, stride=1),
    )

    assert sequences == []


def test_missing_lexical_modality_is_preserved_and_tensor_safe():
    feature_names = default_temporal_feature_names()
    records = [
        _record("s1", "a", 0, lexical_missing=False),
        _record("s1", "a", 1, lexical_missing=True),
    ]

    sequences = build_temporal_sequences(
        records,
        config=SequenceBuilderConfig(sequence_length=2, feature_names=feature_names),
    )
    dataset = TemporalSequenceDataset(sequences)
    item = dataset[0]
    lexical_missing_index = feature_names.index("lexical_missing")
    profanity_index = feature_names.index("profanity_probability")

    assert item["features"].shape == (2, len(feature_names))
    assert item["features"][1, lexical_missing_index].item() == 1.0
    assert item["features"][1, profanity_index].item() == 0.0
    assert not np.isnan(sequences[0].features).any()


def test_sequence_labels_are_reproducible_from_extractor():
    records = [_record("s1", "a", tick) for tick in range(4)]

    def label_extractor(window):
        return float(window[-1].tick_start_ms >= 2000)

    first = build_temporal_sequences(
        records,
        config=SequenceBuilderConfig(sequence_length=2),
        label_extractor=label_extractor,
    )
    second = build_temporal_sequences(
        records,
        config=SequenceBuilderConfig(sequence_length=2),
        label_extractor=label_extractor,
    )

    assert [sequence.label for sequence in first] == [0.0, 1.0, 1.0]
    assert [sequence.metadata.source_indices for sequence in first] == [
        sequence.metadata.source_indices for sequence in second
    ]

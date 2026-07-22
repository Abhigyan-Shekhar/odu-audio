"""Tests for static fusion featurization and group-disjoint splitting."""

import pandas as pd

from src.audio_pipeline.models.lightgbm_fusion import (
    LightGBMFusionConfig,
    flatten_record,
    group_disjoint_train_val_split,
    group_disjoint_train_val_test_split,
    select_feature_columns,
)
from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord


def test_flatten_record_includes_interactions_and_missing_mask():
    record = UnifiedFeatureRecord(
        session_id="session-1",
        stream_id="stream-1",
        speaker_id="speaker-1",
        tick_start_ms=0,
        tick_end_ms=1000,
        egemaps=[0.1] * 88,
        yamnet_event_scores={"Shout": 0.7},
        acoustic_arousal_probability=0.7,
        pitch_energy_arousal_probability=0.6,
        profanity_probability=0.5,
        threat_probability=0.4,
        asr_confidence=0.8,
        feature_missing_mask={"lexical": False},
    ).with_interactions()

    row = flatten_record(record)
    assert row["egemaps_087"] == 0.1
    assert row["yamnet_shout"] == 0.7
    assert row["acoustic_x_profanity"] == 0.35
    assert row["acoustic_arousal_x_profanity"] == 0.35
    assert row["pitch_energy_x_threat"] == 0.24
    assert row["asr_conf_x_profanity"] == 0.4
    assert row["missing:lexical"] == 0


def test_group_disjoint_split_uses_patient_groups():
    frame = pd.DataFrame(
        {
            "patient_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4,
            "verbal_agitation_label": [0, 1] * 8,
            "acoustic_arousal_probability": [0.1, 0.9] * 8,
        }
    )
    config = LightGBMFusionConfig(validation_size=0.5, random_state=7)

    train, val = group_disjoint_train_val_split(frame, config)

    assert set(train["patient_id"]).isdisjoint(set(val["patient_id"]))


def test_group_disjoint_train_val_test_split_uses_patient_groups():
    frame = pd.DataFrame(
        {
            "patient_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4,
            "verbal_agitation_label": [0, 1] * 8,
            "acoustic_arousal_probability": [0.1, 0.9] * 8,
        }
    )
    config = LightGBMFusionConfig(
        validation_size=0.25,
        test_size=0.25,
        random_state=7,
    )

    splits = group_disjoint_train_val_test_split(frame, config)

    train_groups = set(splits["train"]["patient_id"])
    val_groups = set(splits["val"]["patient_id"])
    test_groups = set(splits["test"]["patient_id"])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_select_feature_columns_supports_prefixes():
    frame = pd.DataFrame(
        {
            "patient_id": ["a"],
            "verbal_agitation_label": [1],
            "egemaps_000": [0.2],
            "yamnet_shout": [0.3],
            "session_id": ["s"],
        }
    )

    columns = select_feature_columns(
        frame,
        selectors=["egemaps_", "yamnet_"],
        config=LightGBMFusionConfig(),
    )

    assert columns == ["egemaps_000", "yamnet_shout"]

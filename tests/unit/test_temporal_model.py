"""Tests for Phase 2 temporal model and data loading utilities."""

import torch

from src.audio_pipeline.models.data_loaders import (
    TemporalSplitConfig,
    make_temporal_dataloader,
    split_temporal_sequences,
)
from src.audio_pipeline.models.sequence_dataset import (
    SequenceBuilderConfig,
    build_temporal_sequences,
    default_temporal_feature_names,
)
from src.audio_pipeline.models.temporal_bilstm import (
    TemporalBiLSTM,
    TemporalBiLSTMConfig,
)
from src.audio_pipeline.models.temporal_training import set_reproducible_seed
from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord


def _record(subject_id: str, tick: int) -> UnifiedFeatureRecord:
    return UnifiedFeatureRecord(
        session_id=f"{subject_id}-session",
        stream_id="stream-1",
        subject_id=subject_id,
        speaker_id=f"{subject_id}-speaker",
        tick_start_ms=tick * 1000,
        tick_end_ms=(tick + 1) * 1000,
        patient_probability=0.7,
        acoustic_arousal_probability=0.2,
        profanity_probability=0.3,
        threat_probability=0.1,
        asr_confidence=0.9,
        acoustic_missing=False,
        lexical_missing=False,
    ).with_interactions()


def _sequences():
    records = []
    for subject in ["a", "b", "c", "d"]:
        records.extend(_record(subject, tick) for tick in range(4))
    return build_temporal_sequences(
        records,
        config=SequenceBuilderConfig(sequence_length=2),
        label_extractor=lambda window: float(window[-1].subject_id in {"c", "d"}),
    )


def test_temporal_model_input_output_shapes():
    feature_names = default_temporal_feature_names()
    model = TemporalBiLSTM(
        TemporalBiLSTMConfig(input_dim=len(feature_names), hidden_dim=8)
    )
    features = torch.zeros(4, 3, len(feature_names), dtype=torch.float32)

    logits = model(features)
    probabilities = model.predict_proba(features)

    assert logits.shape == (4,)
    assert probabilities.shape == (4,)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_model_initialization_is_seed_reproducible():
    feature_count = len(default_temporal_feature_names())
    set_reproducible_seed(123)
    first = TemporalBiLSTM(TemporalBiLSTMConfig(input_dim=feature_count, hidden_dim=8))
    first_weight = first.classifier.weight.detach().clone()

    set_reproducible_seed(123)
    second = TemporalBiLSTM(TemporalBiLSTMConfig(input_dim=feature_count, hidden_dim=8))

    assert torch.equal(first_weight, second.classifier.weight.detach())


def test_group_disjoint_temporal_split():
    sequences = _sequences()

    splits = split_temporal_sequences(
        sequences,
        TemporalSplitConfig(validation_size=0.25, test_size=0.25, random_state=7),
    )

    train_groups = {sequence.metadata.subject_id for sequence in splits["train"]}
    val_groups = {sequence.metadata.subject_id for sequence in splits["val"]}
    test_groups = {sequence.metadata.subject_id for sequence in splits["test"]}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_dataloader_preserves_metadata_traceability():
    sequences = _sequences()
    loader = make_temporal_dataloader(sequences, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch["features"].shape[0] == 2
    assert batch["label"].shape == (2,)
    assert len(batch["metadata"]) == 2
    assert batch["metadata"][0].sequence_start_ms == 0

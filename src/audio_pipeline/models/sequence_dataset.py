"""
Temporal sequence construction for fused Phase 1 feature records.

Sequences are built within subject/session boundaries only. Missing feature
values are converted to numeric zeros for tensor safety while Phase 1 missing
modality indicators remain explicit input features.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audio_pipeline.models.lightgbm_fusion import (
    ACOUSTIC_FEATURES,
    INTERACTION_FEATURES,
    LEXICAL_FEATURES,
    flatten_record,
)
from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord

LabelExtractor = Callable[[list[UnifiedFeatureRecord]], float]


@dataclass(frozen=True)
class SequenceBuilderConfig:
    """Configuration for fixed-length temporal windows."""

    sequence_length: int = 30
    stride: int = 1
    tick_ms: int = 1000
    feature_names: Optional[list[str]] = None
    require_consecutive_ticks: bool = True

    def __post_init__(self) -> None:
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert self.stride > 0, "stride must be positive"
        assert self.tick_ms > 0, "tick_ms must be positive"


@dataclass(frozen=True)
class TemporalSequenceMetadata:
    """Traceability metadata for one temporal model window."""

    subject_id: str
    session_id: str
    stream_id: str
    sequence_start_ms: int
    sequence_end_ms: int
    record_start_ms: list[int]
    record_end_ms: list[int]
    speaker_ids: list[str]
    source_indices: list[int]


@dataclass
class TemporalSequence:
    """One fixed-length sequence of fused feature ticks."""

    features: np.ndarray
    metadata: TemporalSequenceMetadata
    label: Optional[float] = None
    calm_profanity_label: Optional[float] = None


def default_temporal_feature_names() -> list[str]:
    """Return stable default features for the Phase 2 temporal model."""
    names = (
        ACOUSTIC_FEATURES
        + LEXICAL_FEATURES
        + INTERACTION_FEATURES
        + [
            "acoustic_missing",
            "lexical_missing",
            "missing:acoustic",
            "missing:lexical",
            "missing:egemaps",
            "missing:yamnet_event_scores",
            "missing:profanity_probability",
            "missing:threat_probability",
            "missing:asr_confidence",
        ]
    )
    return list(dict.fromkeys(names))


def build_temporal_sequences(
    records: Iterable[UnifiedFeatureRecord],
    config: Optional[SequenceBuilderConfig] = None,
    label_extractor: Optional[LabelExtractor] = None,
    calm_profanity_extractor: Optional[LabelExtractor] = None,
) -> list[TemporalSequence]:
    """
    Convert fused tick records into fixed-length temporal windows.

    Records are grouped by subject and session, sorted by tick_start_ms, and
    optionally split at timestamp gaps so windows never cross discontinuities.
    """
    cfg = config or SequenceBuilderConfig()
    feature_names = cfg.feature_names or default_temporal_feature_names()
    indexed_records = list(enumerate(records))
    groups: dict[tuple[str, str], list[tuple[int, UnifiedFeatureRecord]]] = {}
    for source_index, record in indexed_records:
        subject_id = record.subject_id or record.speaker_id or "unknown_subject"
        groups.setdefault((subject_id, record.session_id), []).append(
            (source_index, record)
        )

    sequences: list[TemporalSequence] = []
    for (subject_id, session_id), group_records in sorted(groups.items()):
        ordered = sorted(
            group_records,
            key=lambda item: (
                item[1].tick_start_ms,
                item[1].tick_end_ms,
                item[0],
            ),
        )
        for segment in _split_consecutive_segments(ordered, cfg):
            sequences.extend(
                _windows_from_segment(
                    segment=segment,
                    subject_id=subject_id,
                    session_id=session_id,
                    feature_names=feature_names,
                    config=cfg,
                    label_extractor=label_extractor,
                    calm_profanity_extractor=calm_profanity_extractor,
                )
            )

    return sequences


class TemporalSequenceDataset(Dataset):
    """Torch dataset wrapper around fixed-length temporal sequences."""

    def __init__(self, sequences: Iterable[TemporalSequence]) -> None:
        self.sequences = list(sequences)
        if not self.sequences:
            raise ValueError("TemporalSequenceDataset requires at least one sequence")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence = self.sequences[index]
        features = torch.as_tensor(sequence.features, dtype=torch.float32)
        label_value = 0.0 if sequence.label is None else sequence.label
        return {
            "features": features,
            "label": torch.tensor(label_value, dtype=torch.float32),
            "metadata": sequence.metadata,
        }


def _split_consecutive_segments(
    ordered: list[tuple[int, UnifiedFeatureRecord]],
    config: SequenceBuilderConfig,
) -> list[list[tuple[int, UnifiedFeatureRecord]]]:
    if not ordered:
        return []
    if not config.require_consecutive_ticks:
        return [ordered]

    segments: list[list[tuple[int, UnifiedFeatureRecord]]] = [[ordered[0]]]
    for item in ordered[1:]:
        previous = segments[-1][-1][1]
        current = item[1]
        if current.tick_start_ms - previous.tick_start_ms == config.tick_ms:
            segments[-1].append(item)
        else:
            segments.append([item])
    return segments


def _windows_from_segment(
    segment: list[tuple[int, UnifiedFeatureRecord]],
    subject_id: str,
    session_id: str,
    feature_names: list[str],
    config: SequenceBuilderConfig,
    label_extractor: Optional[LabelExtractor],
    calm_profanity_extractor: Optional[LabelExtractor],
) -> list[TemporalSequence]:
    sequences: list[TemporalSequence] = []
    if len(segment) < config.sequence_length:
        return sequences

    for start in range(0, len(segment) - config.sequence_length + 1, config.stride):
        window = segment[start : start + config.sequence_length]
        source_indices = [source_index for source_index, _ in window]
        records = [record for _, record in window]
        features = np.vstack(
            [_record_to_feature_vector(record, feature_names) for record in records]
        )
        metadata = TemporalSequenceMetadata(
            subject_id=subject_id,
            session_id=session_id,
            stream_id=records[-1].stream_id,
            sequence_start_ms=records[0].tick_start_ms,
            sequence_end_ms=records[-1].tick_end_ms,
            record_start_ms=[record.tick_start_ms for record in records],
            record_end_ms=[record.tick_end_ms for record in records],
            speaker_ids=[record.speaker_id or "" for record in records],
            source_indices=source_indices,
        )
        label = label_extractor(records) if label_extractor else None
        calm_label = (
            calm_profanity_extractor(records) if calm_profanity_extractor else None
        )
        sequences.append(
            TemporalSequence(
                features=features,
                metadata=metadata,
                label=label,
                calm_profanity_label=calm_label,
            )
        )

    return sequences


def _record_to_feature_vector(
    record: UnifiedFeatureRecord, feature_names: list[str]
) -> np.ndarray:
    row = flatten_record(record)
    values = [float(row.get(name, np.nan)) for name in feature_names]
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

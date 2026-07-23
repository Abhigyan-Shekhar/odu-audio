"""
Timestamp alignment for acoustic and lexical feature fusion.

Acoustic windows and lexical utterance windows are aligned onto a causal
one-second grid. Lexical evidence is assigned only to ticks it overlaps; it is
not forward-filled beyond the source utterance.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, cast

import numpy as np

from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord
from src.audio_pipeline.schemas.lexical_record import LexicalFeatureRecord
from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord


@dataclass(frozen=True)
class FusionAlignmentConfig:
    """Configuration for feature-level one-second fusion."""

    tick_ms: int = 1000
    acoustic_arousal_event_keys: tuple[str, ...] = (
        "Shout",
        "Yell",
        "Screaming",
        "Angry music",
    )
    scream_shout_event_keys: tuple[str, ...] = ("Screaming", "Shout", "Yell")
    min_overlap_ms: int = 1
    config_hash: str = ""
    source_versions: dict[str, str] = field(default_factory=dict)


def align_to_fusion_ticks(
    acoustic_records: Iterable[AcousticFeatureRecord],
    lexical_records: Iterable[LexicalFeatureRecord],
    config: Optional[FusionAlignmentConfig] = None,
) -> list[UnifiedFeatureRecord]:
    """
    Align acoustic and lexical records onto a common one-second grid.

    Args:
        acoustic_records: Overlapping acoustic feature windows.
        lexical_records: Lexical feature windows or utterances.
        config: Optional fusion alignment settings.

    Returns:
        Ordered unified records, one per fusion tick.
    """
    cfg = config or FusionAlignmentConfig()
    acoustic = sorted(acoustic_records, key=lambda item: item.window_start_ms)
    lexical = sorted(lexical_records, key=lambda item: item.window_start_ms)

    if not acoustic and not lexical:
        return []

    starts = [r.window_start_ms for r in acoustic] + [
        r.window_start_ms for r in lexical
    ]
    ends = [r.window_end_ms for r in acoustic] + [r.window_end_ms for r in lexical]
    grid_start = (min(starts) // cfg.tick_ms) * cfg.tick_ms
    grid_end = ((max(ends) + cfg.tick_ms - 1) // cfg.tick_ms) * cfg.tick_ms

    records: list[UnifiedFeatureRecord] = []
    for tick_start in range(grid_start, grid_end, cfg.tick_ms):
        tick_end = tick_start + cfg.tick_ms
        acoustic_hits = [
            item
            for item in acoustic
            if _overlap_ms(
                tick_start, tick_end, item.window_start_ms, item.window_end_ms
            )
            >= cfg.min_overlap_ms
        ]
        lexical_hits = [
            item
            for item in lexical
            if _overlap_ms(
                tick_start, tick_end, item.window_start_ms, item.window_end_ms
            )
            >= cfg.min_overlap_ms
        ]

        session_id, stream_id = _choose_ids(acoustic_hits, lexical_hits)
        acoustic_values = _aggregate_acoustic(tick_start, tick_end, acoustic_hits, cfg)
        lexical_values = _aggregate_lexical(tick_start, tick_end, lexical_hits)
        missing_mask = _missing_mask(acoustic_values, lexical_values)

        source_versions = dict(cfg.source_versions)
        source_hashes: dict[str, str] = {}
        for acoustic_record in acoustic_hits:
            source_versions.update(acoustic_record.extractor_versions)
            source_hashes.update(acoustic_record.model_hashes)
        for lexical_record in lexical_hits:
            source_versions.update(lexical_record.extractor_versions)
            source_hashes.update(lexical_record.model_hashes)

        unified = UnifiedFeatureRecord(
            session_id=session_id,
            stream_id=stream_id,
            speaker_id=acoustic_values.get("speaker_id"),
            tick_start_ms=tick_start,
            tick_end_ms=tick_end,
            patient_probability=acoustic_values.get("patient_probability"),
            speaker_confidence=acoustic_values.get("patient_probability"),
            attribution_status=acoustic_values.get("attribution_status", "UNKNOWN"),
            overlap_probability=acoustic_values.get("overlap_probability"),
            egemaps=acoustic_values.get("egemaps"),
            yamnet_event_scores=acoustic_values.get("yamnet_event_scores", {}),
            emotion_embedding=acoustic_values.get("emotion_embedding"),
            profanity_probability=lexical_values.get("profanity_probability"),
            profanity_intensity=lexical_values.get("profanity_intensity"),
            toxicity_probability=lexical_values.get("toxicity_probability"),
            threat_probability=lexical_values.get("threat_probability"),
            directed_insult_probability=lexical_values.get(
                "directed_insult_probability"
            ),
            imperative_probability=lexical_values.get("imperative_probability"),
            repetition_probability=lexical_values.get("repetition_probability"),
            repeated_request_probability=lexical_values.get(
                "repeated_request_probability"
            ),
            distress_phrase_probability=lexical_values.get(
                "distress_phrase_probability"
            ),
            mutox_speech_score=lexical_values.get("mutox_speech_score"),
            mutox_text_score=lexical_values.get("mutox_text_score"),
            asr_confidence=lexical_values.get("asr_confidence"),
            language=lexical_values.get("language", "unknown"),
            code_mixed=lexical_values.get("code_mixed"),
            snr_db=acoustic_values.get("snr_db"),
            clipping_ratio=acoustic_values.get("clipping_ratio"),
            dropout_ratio=acoustic_values.get("dropout_ratio"),
            quality_status=acoustic_values.get("quality_status", "UNKNOWN"),
            acoustic_arousal_probability=acoustic_values.get(
                "acoustic_arousal_probability"
            ),
            pitch_energy_arousal_probability=acoustic_values.get(
                "pitch_energy_arousal_probability"
            ),
            scream_shout_probability=acoustic_values.get("scream_shout_probability"),
            feature_missing_mask=missing_mask,
            acoustic_missing=missing_mask["acoustic"],
            lexical_missing=missing_mask["lexical"],
            source_versions=source_versions,
            source_hashes=source_hashes,
            config_hash=cfg.config_hash or acoustic_values.get("config_hash", ""),
        ).with_interactions()
        records.append(unified)

    return records


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _choose_ids(
    acoustic_hits: list[AcousticFeatureRecord],
    lexical_hits: list[LexicalFeatureRecord],
) -> tuple[str, str]:
    if acoustic_hits:
        return acoustic_hits[-1].session_id, acoustic_hits[-1].stream_id
    if lexical_hits:
        return lexical_hits[-1].session_id, lexical_hits[-1].stream_id
    return "unknown", "unknown"


def _aggregate_acoustic(
    tick_start: int,
    tick_end: int,
    hits: list[AcousticFeatureRecord],
    config: FusionAlignmentConfig,
) -> dict:
    if not hits:
        return {}

    weights = np.array(
        [
            _overlap_ms(tick_start, tick_end, r.window_start_ms, r.window_end_ms)
            for r in hits
        ],
        dtype=float,
    )
    weights = weights / weights.sum()
    last = max(hits, key=lambda item: item.window_end_ms)

    yamnet_scores = _weighted_dict_mean(
        [record.yamnet_event_scores for record in hits], weights
    )
    arousal = _max_event_score(yamnet_scores, config.acoustic_arousal_event_keys)
    scream_shout = _max_event_score(yamnet_scores, config.scream_shout_event_keys)

    return {
        "speaker_id": last.speaker_id,
        "patient_probability": _weighted_optional_mean(
            [r.patient_probability for r in hits], weights
        ),
        "attribution_status": last.attribution_status,
        "overlap_probability": _weighted_optional_mean(
            [r.overlap_probability for r in hits], weights
        ),
        "egemaps": _weighted_vector_mean([r.egemaps for r in hits], weights),
        "yamnet_event_scores": yamnet_scores,
        "emotion_embedding": _weighted_vector_mean(
            [r.emotion_embedding for r in hits], weights, expected_size=None
        ),
        "snr_db": _weighted_optional_mean([r.snr_db for r in hits], weights),
        "clipping_ratio": max(r.clipping_ratio for r in hits),
        "dropout_ratio": max(r.dropout_ratio for r in hits),
        "quality_status": _worst_quality([r.quality_status for r in hits]),
        "acoustic_arousal_probability": arousal,
        "pitch_energy_arousal_probability": arousal,
        "scream_shout_probability": scream_shout,
        "config_hash": last.config_hash,
    }


def _aggregate_lexical(
    tick_start: int,
    tick_end: int,
    hits: list[LexicalFeatureRecord],
) -> dict:
    if not hits:
        return {}

    weights = np.array(
        [
            _overlap_ms(tick_start, tick_end, r.window_start_ms, r.window_end_ms)
            for r in hits
        ],
        dtype=float,
    )
    weights = weights / weights.sum()

    code_mixed_weight = sum(
        weight for record, weight in zip(hits, weights) if record.code_mixed
    )

    max_fields = [
        "profanity_probability",
        "profanity_intensity",
        "toxicity_probability",
        "threat_probability",
        "directed_insult_probability",
        "imperative_probability",
        "repetition_probability",
        "repeated_request_probability",
        "distress_phrase_probability",
    ]
    values = {
        field: max(getattr(record, field) for record in hits) for field in max_fields
    }
    values["asr_confidence"] = _weighted_optional_mean(
        [record.asr_confidence for record in hits], weights
    )
    values["mutox_speech_score"] = _weighted_optional_mean(
        [record.mutox_speech_score for record in hits], weights
    )
    values["mutox_text_score"] = _weighted_optional_mean(
        [record.mutox_text_score for record in hits], weights
    )
    values["language"] = max(hits, key=lambda record: record.window_end_ms).language
    values["code_mixed"] = code_mixed_weight >= 0.5
    return values


def _weighted_optional_mean(
    values: list[Optional[float]], weights: np.ndarray
) -> Optional[float]:
    usable = [
        (value, weight) for value, weight in zip(values, weights) if value is not None
    ]
    if not usable:
        return None
    denominator = sum(weight for _, weight in usable)
    if denominator == 0.0:
        return None
    return float(sum(value * weight for value, weight in usable) / denominator)


def _weighted_vector_mean(
    vectors: list[Optional[list[float]]],
    weights: np.ndarray,
    expected_size: Optional[int] = 88,
) -> Optional[list[float]]:
    usable = [
        (np.asarray(vector, dtype=float), weight)
        for vector, weight in zip(vectors, weights)
        if vector is not None
    ]
    if not usable:
        return None
    if expected_size is not None and any(
        len(vector) != expected_size for vector, _ in usable
    ):
        return None
    denominator = sum(weight for _, weight in usable)
    return cast(
        list[float],
        (sum(vector * weight for vector, weight in usable) / denominator).tolist(),
    )


def _weighted_dict_mean(
    dicts: list[dict[str, float]], weights: np.ndarray
) -> dict[str, float]:
    keys = sorted({key for item in dicts for key in item.keys()})
    output: dict[str, float] = {}
    for key in keys:
        usable = [
            (item[key], weight)
            for item, weight in zip(dicts, weights)
            if key in item and item[key] is not None
        ]
        if usable:
            denominator = sum(weight for _, weight in usable)
            output[key] = float(
                sum(value * weight for value, weight in usable) / denominator
            )
    return output


def _max_event_score(
    scores: dict[str, float], event_keys: tuple[str, ...]
) -> Optional[float]:
    matches = [
        value
        for name, value in scores.items()
        if any(key.lower() in name.lower() for key in event_keys)
    ]
    if not matches:
        return None
    return float(max(matches))


def _worst_quality(statuses: list[str]) -> str:
    order = {
        "UNKNOWN": 0,
        "OK": 1,
        "INSUFFICIENT_AUDIO": 2,
        "LOW_SNR": 3,
        "DROPOUT": 4,
        "CLIPPED": 5,
        "REJECTED": 6,
    }
    return max(statuses, key=lambda status: order.get(status, 0))


def _missing_mask(acoustic_values: dict, lexical_values: dict) -> dict[str, bool]:
    return {
        "acoustic": not bool(acoustic_values),
        "lexical": not bool(lexical_values),
        "egemaps": acoustic_values.get("egemaps") is None,
        "yamnet_event_scores": not bool(acoustic_values.get("yamnet_event_scores")),
        "emotion_embedding": acoustic_values.get("emotion_embedding") is None,
        "patient_probability": acoustic_values.get("patient_probability") is None,
        "overlap_probability": acoustic_values.get("overlap_probability") is None,
        "snr_db": acoustic_values.get("snr_db") is None,
        "profanity_probability": lexical_values.get("profanity_probability") is None,
        "toxicity_probability": lexical_values.get("toxicity_probability") is None,
        "threat_probability": lexical_values.get("threat_probability") is None,
        "asr_confidence": lexical_values.get("asr_confidence") is None,
        "language": not bool(lexical_values.get("language")),
        "code_mixed": lexical_values.get("code_mixed") is None,
    }

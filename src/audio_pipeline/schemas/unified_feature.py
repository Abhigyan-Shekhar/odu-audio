"""
Unified acoustic and lexical feature contract for one-second fusion ticks.

The unified record is intentionally transcript-free. Text can be stored in a
separate, access-controlled artifact, but model inputs and alert payloads should
only carry derived lexical probabilities.
"""

from dataclasses import dataclass, field
from typing import Optional


def _assert_probability(name: str, value: Optional[float]) -> None:
    if value is not None:
        assert 0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}"


@dataclass
class UnifiedFeatureRecord:
    """Fused feature record for a single causal one-second tick."""

    session_id: str
    stream_id: str
    speaker_id: Optional[str]
    tick_start_ms: int
    tick_end_ms: int
    subject_id: Optional[str] = None

    patient_probability: Optional[float] = None
    speaker_confidence: Optional[float] = None
    attribution_status: str = "UNKNOWN"
    overlap_probability: Optional[float] = None

    egemaps: Optional[list[float]] = None
    yamnet_event_scores: dict[str, float] = field(default_factory=dict)
    emotion_embedding: Optional[list[float]] = None
    affect_scores: dict[str, float] = field(default_factory=dict)

    profanity_probability: Optional[float] = None
    profanity_intensity: Optional[float] = None
    toxicity_probability: Optional[float] = None
    threat_probability: Optional[float] = None
    directed_insult_probability: Optional[float] = None
    imperative_probability: Optional[float] = None
    repetition_probability: Optional[float] = None
    repeated_request_probability: Optional[float] = None
    distress_phrase_probability: Optional[float] = None
    mutox_speech_score: Optional[float] = None
    mutox_text_score: Optional[float] = None
    asr_confidence: Optional[float] = None
    language: str = "unknown"
    code_mixed: Optional[bool] = None

    snr_db: Optional[float] = None
    clipping_ratio: Optional[float] = None
    dropout_ratio: Optional[float] = None
    quality_status: str = "UNKNOWN"

    acoustic_arousal_probability: Optional[float] = None
    pitch_energy_arousal_probability: Optional[float] = None
    scream_shout_probability: Optional[float] = None
    acoustic_x_profanity: Optional[float] = None
    acoustic_arousal_x_profanity: Optional[float] = None
    pitch_energy_x_threat: Optional[float] = None
    repetition_x_arousal: Optional[float] = None
    asr_conf_x_profanity: Optional[float] = None
    arousal_x_threat: Optional[float] = None
    arousal_x_distress: Optional[float] = None
    scream_x_distress: Optional[float] = None
    toxicity_x_asr_confidence: Optional[float] = None
    patient_probability_x_agitation_evidence: Optional[float] = None

    feature_missing_mask: dict[str, bool] = field(default_factory=dict)
    acoustic_missing: bool = True
    lexical_missing: bool = True
    source_versions: dict[str, str] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        assert self.session_id, "session_id must not be empty"
        assert self.stream_id, "stream_id must not be empty"
        assert (
            self.tick_start_ms <= self.tick_end_ms
        ), "tick_start_ms must be <= tick_end_ms"

        valid_attribution = {
            "PATIENT",
            "NON_PATIENT",
            "UNKNOWN",
            "OVERLAP",
            "LOW_CONFIDENCE",
        }
        assert self.attribution_status in valid_attribution, (
            f"attribution_status must be one of {valid_attribution}, "
            f"got {self.attribution_status}"
        )

        valid_quality = {
            "OK",
            "CLIPPED",
            "LOW_SNR",
            "DROPOUT",
            "REJECTED",
            "INSUFFICIENT_AUDIO",
            "UNKNOWN",
        }
        assert (
            self.quality_status in valid_quality
        ), f"quality_status must be one of {valid_quality}, got {self.quality_status}"

        if self.egemaps is not None:
            assert (
                len(self.egemaps) == 88
            ), f"eGeMAPSv02 must have 88 functionals, got {len(self.egemaps)}"

        probability_fields = [
            "patient_probability",
            "speaker_confidence",
            "overlap_probability",
            "profanity_probability",
            "profanity_intensity",
            "toxicity_probability",
            "threat_probability",
            "directed_insult_probability",
            "imperative_probability",
            "repetition_probability",
            "repeated_request_probability",
            "distress_phrase_probability",
            "mutox_speech_score",
            "mutox_text_score",
            "asr_confidence",
            "clipping_ratio",
            "dropout_ratio",
            "acoustic_arousal_probability",
            "pitch_energy_arousal_probability",
            "scream_shout_probability",
            "acoustic_x_profanity",
            "acoustic_arousal_x_profanity",
            "pitch_energy_x_threat",
            "repetition_x_arousal",
            "asr_conf_x_profanity",
            "arousal_x_threat",
            "arousal_x_distress",
            "scream_x_distress",
            "toxicity_x_asr_confidence",
            "patient_probability_x_agitation_evidence",
        ]
        for field_name in probability_fields:
            _assert_probability(field_name, getattr(self, field_name))

        if self.feature_missing_mask:
            self.acoustic_missing = bool(
                self.feature_missing_mask.get("acoustic", self.acoustic_missing)
            )
            self.lexical_missing = bool(
                self.feature_missing_mask.get("lexical", self.lexical_missing)
            )
        else:
            self.feature_missing_mask = {
                "acoustic": self.acoustic_missing,
                "lexical": self.lexical_missing,
            }

    @property
    def t0(self) -> int:
        """Alias for tick_start_ms used by downstream experiment manifests."""
        return self.tick_start_ms

    @property
    def t1(self) -> int:
        """Alias for tick_end_ms used by downstream experiment manifests."""
        return self.tick_end_ms

    def to_dict(self) -> dict:
        """Convert to a serialization-friendly dictionary."""
        data = self.__dict__.copy()
        data["t0"] = self.t0
        data["t1"] = self.t1
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedFeatureRecord":
        """Create a unified record from a dictionary."""
        payload = data.copy()
        payload.pop("t0", None)
        payload.pop("t1", None)
        return cls(**payload)

    def with_interactions(self) -> "UnifiedFeatureRecord":
        """
        Fill derived interaction features in-place and return this record.

        Missing inputs keep their corresponding interactions missing instead of
        being silently treated as zero.
        """
        arousal = self.acoustic_arousal_probability
        pitch_energy = self.pitch_energy_arousal_probability
        profanity = self.profanity_probability
        threat = self.threat_probability
        distress = self.distress_phrase_probability
        toxicity = self.toxicity_probability
        asr_confidence = self.asr_confidence
        scream = self.scream_shout_probability

        self.acoustic_x_profanity = (
            arousal * profanity
            if arousal is not None and profanity is not None
            else None
        )
        self.acoustic_arousal_x_profanity = self.acoustic_x_profanity
        self.pitch_energy_x_threat = (
            pitch_energy * threat
            if pitch_energy is not None and threat is not None
            else None
        )
        self.repetition_x_arousal = (
            self.repetition_probability * arousal
            if self.repetition_probability is not None and arousal is not None
            else None
        )
        self.asr_conf_x_profanity = (
            asr_confidence * profanity
            if asr_confidence is not None and profanity is not None
            else None
        )
        self.arousal_x_threat = (
            arousal * threat if arousal is not None and threat is not None else None
        )
        self.arousal_x_distress = (
            arousal * distress if arousal is not None and distress is not None else None
        )
        self.scream_x_distress = (
            scream * distress if scream is not None and distress is not None else None
        )
        self.toxicity_x_asr_confidence = (
            toxicity * asr_confidence
            if toxicity is not None and asr_confidence is not None
            else None
        )

        evidence_values = [
            value
            for value in [
                arousal,
                scream,
                profanity,
                threat,
                self.directed_insult_probability,
                distress,
                self.repetition_probability,
            ]
            if value is not None
        ]
        if self.patient_probability is not None and evidence_values:
            self.patient_probability_x_agitation_evidence = (
                self.patient_probability * max(evidence_values)
            )
        else:
            self.patient_probability_x_agitation_evidence = None

        return self

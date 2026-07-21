"""
Lexical feature schemas for ASR, profanity, threat, repetition, and toxicity.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordTimestamp:
    """Single ASR word with timing and confidence."""

    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0

    def __post_init__(self):
        assert self.word, "word must not be empty"
        assert self.start_ms <= self.end_ms, "start_ms must be <= end_ms"
        assert 0.0 <= self.confidence <= 1.0, (
            f"confidence must be in [0, 1], got {self.confidence}"
        )


@dataclass
class TranscriptSegment:
    """ASR output for a VAD speech segment or lexical chunk."""

    session_id: str
    stream_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str = "unknown"
    language_probability: float = 0.0
    avg_confidence: float = 0.0
    words: list[WordTimestamp] = field(default_factory=list)
    asr_model: str = "unknown"
    incomplete: bool = False
    code_mixed: bool = False

    def __post_init__(self):
        assert self.start_ms <= self.end_ms, "start_ms must be <= end_ms"
        assert 0.0 <= self.language_probability <= 1.0, (
            "language_probability must be in [0, 1]"
        )
        assert 0.0 <= self.avg_confidence <= 1.0, "avg_confidence must be in [0, 1]"


@dataclass
class LexicalFeatureRecord:
    """
    Lexical features for one speech window.

    This is the Person 2 deliverable:
    speech window -> transcript + word timestamps -> profanity/threat/repetition/toxicity.
    """

    session_id: str
    stream_id: str
    window_start_ms: int
    window_end_ms: int
    transcript: str
    normalized_transcript: str
    language: str = "unknown"
    language_probability: float = 0.0
    code_mixed: bool = False
    asr_confidence: float = 0.0
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    profanity_count: int = 0
    profanity_probability: float = 0.0
    profanity_intensity: float = 0.0
    profanity_matches: list[dict] = field(default_factory=list)
    incomplete_profanity_probability: float = 0.0
    threat_probability: float = 0.0
    directed_insult_probability: float = 0.0
    imperative_probability: float = 0.0
    repetition_probability: float = 0.0
    repeated_phrases: list[str] = field(default_factory=list)
    repeated_request_probability: float = 0.0
    distress_phrase_probability: float = 0.0
    mutox_speech_score: Optional[float] = None
    mutox_text_score: Optional[float] = None
    toxicity_probability: float = 0.0
    confidence_weighted_toxicity: float = 0.0
    flags: list[str] = field(default_factory=list)
    extractor_versions: dict[str, str] = field(default_factory=dict)
    model_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert self.window_start_ms <= self.window_end_ms, (
            "window_start_ms must be <= window_end_ms"
        )
        assert self.profanity_count >= 0, "profanity_count must be non-negative"

        probability_fields = [
            "language_probability",
            "asr_confidence",
            "profanity_probability",
            "profanity_intensity",
            "incomplete_profanity_probability",
            "threat_probability",
            "directed_insult_probability",
            "imperative_probability",
            "repetition_probability",
            "repeated_request_probability",
            "distress_phrase_probability",
            "toxicity_probability",
            "confidence_weighted_toxicity",
        ]
        for field_name in probability_fields:
            value = getattr(self, field_name)
            assert 0.0 <= value <= 1.0, f"{field_name} must be in [0, 1]"

        if self.mutox_speech_score is not None:
            assert 0.0 <= self.mutox_speech_score <= 1.0, (
                "mutox_speech_score must be in [0, 1]"
            )
        if self.mutox_text_score is not None:
            assert 0.0 <= self.mutox_text_score <= 1.0, (
                "mutox_text_score must be in [0, 1]"
            )

    def to_dict(self) -> dict:
        """Convert to a serialization-friendly dictionary."""
        data = self.__dict__.copy()
        data["word_timestamps"] = [word.__dict__.copy() for word in self.word_timestamps]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LexicalFeatureRecord":
        """Create from a dictionary."""
        payload = data.copy()
        payload["word_timestamps"] = [
            word if isinstance(word, WordTimestamp) else WordTimestamp(**word)
            for word in payload.get("word_timestamps", [])
        ]
        return cls(**payload)

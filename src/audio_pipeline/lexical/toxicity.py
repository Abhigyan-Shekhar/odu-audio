"""MuTox integration boundary with deterministic fallback scoring."""

from src.audio_pipeline.lexical.detectors import (
    directed_insult_probability,
    distress_phrase_probability,
    threat_probability,
)
from src.audio_pipeline.lexical.lexicon import ProfanityLexicon


class MuToxScorer:
    """
    Optional MuTox speech/text toxicity wrapper.

    The real MuTox model is intentionally loaded lazily by deployment code. This
    class provides a stable interface plus a conservative fallback for tests.
    """

    def __init__(self, text_model=None, speech_model=None):
        self.text_model = text_model
        self.speech_model = speech_model
        self.lexicon = ProfanityLexicon.default()

    def score_text(self, text: str) -> float:
        """Score text toxicity using MuTox-compatible interface or fallback."""
        if self.text_model is not None:
            return float(self.text_model.score(text))

        matches = self.lexicon.match(text)
        profanity = max((m["severity"] * m["confidence"] for m in matches), default=0.0)
        directed = directed_insult_probability(text, matches)
        threat = threat_probability(text)
        distress = distress_phrase_probability(text)
        return min(1.0, max(profanity, directed, threat, distress * 0.4))

    def score_speech(self, audio, sample_rate: int) -> float | None:
        """Score speech toxicity when a speech model is supplied."""
        if self.speech_model is None:
            return None
        return float(self.speech_model.score(audio, sample_rate=sample_rate))

    def get_version(self) -> str:
        """Return implementation version label."""
        if self.text_model is None and self.speech_model is None:
            return "heuristic-fallback"
        return "mutox-compatible"

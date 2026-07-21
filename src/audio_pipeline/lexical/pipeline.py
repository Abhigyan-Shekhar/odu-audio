"""Lexical feature extraction pipeline."""

from collections import deque

from src.audio_pipeline.lexical.detectors import (
    directed_insult_probability,
    distress_phrase_probability,
    imperative_probability,
    repetition_probability,
    threat_probability,
)
from src.audio_pipeline.lexical.language import identify_language
from src.audio_pipeline.lexical.lexicon import ProfanityLexicon
from src.audio_pipeline.lexical.normalization import looks_code_mixed, normalize_text, tokenize
from src.audio_pipeline.lexical.toxicity import MuToxScorer
from src.audio_pipeline.schemas.lexical_record import (
    LexicalFeatureRecord,
    TranscriptSegment,
)


class LexicalFeatureExtractor:
    """Converts transcript windows into Person 2 lexical feature records."""

    def __init__(
        self,
        lexicon: ProfanityLexicon | None = None,
        toxicity_scorer: MuToxScorer | None = None,
        repetition_context_windows: int = 6,
        low_asr_confidence_threshold: float = 0.45,
    ):
        self.lexicon = lexicon or ProfanityLexicon.default()
        self.toxicity_scorer = toxicity_scorer or MuToxScorer()
        self.low_asr_confidence_threshold = low_asr_confidence_threshold
        self._history: deque[str] = deque(maxlen=repetition_context_windows)

    def extract(self, transcript: TranscriptSegment) -> LexicalFeatureRecord:
        """Extract lexical features from one ASR transcript segment."""
        normalized = normalize_text(transcript.text)
        tokens = tokenize(normalized)
        language, language_probability, code_mixed = identify_language(transcript.text)
        code_mixed = code_mixed or transcript.code_mixed or looks_code_mixed(tokens)

        matches = self.lexicon.match(normalized)
        confidence_weight = max(0.0, min(1.0, transcript.avg_confidence))
        weighted_matches = [
            {
                **match,
                "confidence_weighted_score": match["severity"]
                * match["confidence"]
                * confidence_weight,
            }
            for match in matches
        ]

        profanity_probability = max(
            (match["confidence"] * confidence_weight for match in matches),
            default=0.0,
        )
        profanity_intensity = max(
            (
                match["severity"] * match["confidence"] * confidence_weight
                for match in matches
            ),
            default=0.0,
        )
        incomplete_probability = max(
            (
                0.75 * match["confidence"] * confidence_weight
                for match in matches
                if match["reason"] == "incomplete_prefix"
            ),
            default=0.0,
        )

        recent_texts = [*self._history, normalized]
        repeat_probability, repeats, repeated_request_probability = repetition_probability(
            recent_texts
        )
        self._history.append(normalized)

        threat = threat_probability(normalized)
        directed_insult = directed_insult_probability(normalized, matches)
        imperative = imperative_probability(normalized)
        distress = distress_phrase_probability(normalized)
        mutox_text = self.toxicity_scorer.score_text(normalized)
        toxicity = min(1.0, max(mutox_text, profanity_intensity, threat, directed_insult))
        weighted_toxicity = toxicity * confidence_weight

        flags: list[str] = []
        if transcript.avg_confidence < self.low_asr_confidence_threshold:
            flags.append("LOW_ASR_CONFIDENCE")
        if any(match["reason"] == "incomplete_prefix" for match in matches):
            flags.append("INCOMPLETE_PROFANITY_CANDIDATE")
        if code_mixed:
            flags.append("CODE_MIXED")
        if transcript.incomplete:
            flags.append("INCOMPLETE_UTTERANCE")

        return LexicalFeatureRecord(
            session_id=transcript.session_id,
            stream_id=transcript.stream_id,
            window_start_ms=transcript.start_ms,
            window_end_ms=transcript.end_ms,
            transcript=transcript.text,
            normalized_transcript=normalized,
            language=transcript.language if transcript.language != "unknown" else language,
            language_probability=max(transcript.language_probability, language_probability),
            code_mixed=code_mixed,
            asr_confidence=transcript.avg_confidence,
            word_timestamps=transcript.words,
            profanity_count=len(matches),
            profanity_probability=profanity_probability,
            profanity_intensity=profanity_intensity,
            profanity_matches=weighted_matches,
            incomplete_profanity_probability=incomplete_probability,
            threat_probability=threat,
            directed_insult_probability=directed_insult,
            imperative_probability=imperative,
            repetition_probability=repeat_probability,
            repeated_phrases=repeats,
            repeated_request_probability=repeated_request_probability,
            distress_phrase_probability=distress,
            mutox_text_score=mutox_text,
            toxicity_probability=toxicity,
            confidence_weighted_toxicity=weighted_toxicity,
            flags=flags,
            extractor_versions={
                "lexical": "native-rule-baseline",
                "toxicity": self.toxicity_scorer.get_version(),
            },
        )

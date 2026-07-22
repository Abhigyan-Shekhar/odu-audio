"""Unit tests for Person 2 lexical branch."""

from src.audio_pipeline.lexical.evaluation import (
    ASRProfanityCase,
    asr_profanity_miss_rates,
)
from src.audio_pipeline.lexical.hard_negatives import default_calm_profanity_prompts
from src.audio_pipeline.lexical.lexicon import ProfanityLexicon
from src.audio_pipeline.lexical.pipeline import LexicalFeatureExtractor
from src.audio_pipeline.schemas.lexical_record import TranscriptSegment, WordTimestamp


def _transcript(text: str, confidence: float = 0.9) -> TranscriptSegment:
    tokens = text.split()
    words = [
        WordTimestamp(
            word=token,
            start_ms=index * 250,
            end_ms=index * 250 + 200,
            confidence=confidence,
        )
        for index, token in enumerate(tokens)
    ]
    return TranscriptSegment(
        session_id="session_test",
        stream_id="stream_test",
        start_ms=0,
        end_ms=max(1000, len(tokens) * 250),
        text=text,
        language="unknown",
        avg_confidence=confidence,
        words=words,
    )


def test_lexicon_matches_asr_error_and_incomplete_profanity():
    lexicon = ProfanityLexicon.default()

    asr_error_matches = lexicon.match("what the fork are you doing")
    incomplete_matches = lexicon.match("fu get out")

    assert any(match["canonical"] == "fuck" for match in asr_error_matches)
    assert any(match["reason"] == "incomplete_prefix" for match in incomplete_matches)


def test_pipeline_extracts_profanity_threat_and_directed_insult():
    extractor = LexicalFeatureExtractor()
    record = extractor.extract(_transcript("you bastard I will hit you", 0.8))

    assert record.profanity_count >= 1
    assert record.profanity_probability > 0.0
    assert record.directed_insult_probability >= 0.7
    assert record.threat_probability >= 0.8
    assert record.confidence_weighted_toxicity <= record.toxicity_probability


def test_pipeline_detects_code_mixed_hinglish():
    extractor = LexicalFeatureExtractor()
    record = extractor.extract(_transcript("tum chup raho please stop", 0.85))

    assert record.language in {"hi-Latn", "hi-en"}
    assert record.code_mixed
    assert "CODE_MIXED" in record.flags
    assert record.imperative_probability > 0.0


def test_pipeline_detects_repeated_request_across_context():
    extractor = LexicalFeatureExtractor(repetition_context_windows=4)

    extractor.extract(_transcript("help me please", 0.9))
    extractor.extract(_transcript("help me please", 0.9))
    record = extractor.extract(_transcript("help me please", 0.9))

    assert record.repetition_probability > 0.0
    assert record.repeated_request_probability > 0.0
    assert any("help" in phrase for phrase in record.repeated_phrases)


def test_low_asr_confidence_downweights_profanity():
    extractor = LexicalFeatureExtractor()

    high = extractor.extract(_transcript("fuck", 0.95))
    low = extractor.extract(_transcript("fuck", 0.25))

    assert high.profanity_probability > low.profanity_probability
    assert "LOW_ASR_CONFIDENCE" in low.flags


def test_calm_profanity_prompts_are_content_hard_negatives():
    prompts = default_calm_profanity_prompts()

    assert prompts
    assert all(prompt.delivery == "calm" for prompt in prompts)
    assert all(prompt.expected_arousal_label == "low_arousal" for prompt in prompts)


def test_asr_profanity_miss_rates_by_condition():
    metrics = asr_profanity_miss_rates(
        [
            ASRProfanityCase("1", True, False, "incomplete", 0.4, "en"),
            ASRProfanityCase("2", True, True, "incomplete", 0.7, "en"),
            ASRProfanityCase("3", True, False, "shouted", 0.3, "hi-Latn"),
            ASRProfanityCase("4", False, False, "shouted", 0.9, "hi-Latn"),
        ]
    )

    assert metrics["incomplete"]["miss_rate"] == 0.5
    assert metrics["shouted"]["miss_rate"] == 1.0

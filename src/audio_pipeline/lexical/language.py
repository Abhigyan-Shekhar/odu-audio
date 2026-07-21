"""Lightweight language identification helpers."""

import re

from src.audio_pipeline.lexical.normalization import tokenize


_HINGLISH_HINTS = {
    "hai",
    "nahi",
    "mat",
    "karo",
    "mujhe",
    "tum",
    "tera",
    "teri",
    "chup",
    "bachao",
}


def identify_language(text: str) -> tuple[str, float, bool]:
    """
    Return (language, probability, code_mixed).

    Faster-Whisper language IDs should be preferred when available; this fallback
    keeps offline tests deterministic.
    """
    tokens = tokenize(text)
    if not tokens:
        return "unknown", 0.0, False

    has_devanagari = bool(re.search(r"[\u0900-\u097f]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    hinglish_hits = sum(1 for token in tokens if token in _HINGLISH_HINTS)

    if has_devanagari and has_latin:
        return "hi-en", 0.80, True
    if has_devanagari:
        return "hi", 0.85, False
    if hinglish_hits:
        probability = min(0.85, 0.55 + 0.10 * hinglish_hits)
        return "hi-Latn", probability, hinglish_hits < len(tokens)
    return "en", 0.70, False

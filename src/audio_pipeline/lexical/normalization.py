"""Text normalization utilities for English, Hindi, Hinglish, and ASR variants."""

import re
import unicodedata


_LEET_MAP = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)


def normalize_text(text: str) -> str:
    """
    Normalize ASR text without removing Indic scripts.

    The output is intended for matching, not display.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(_LEET_MAP)
    text = re.sub(r"[_\-*/|]+", " ", text)
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s\u0900-\u097f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into word-like units."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def token_ngrams(tokens: list[str], max_n: int = 4) -> list[tuple[int, int, str]]:
    """Return token n-grams as (start_index, end_index, text)."""
    ngrams: list[tuple[int, int, str]] = []
    for start in range(len(tokens)):
        for n in range(1, max_n + 1):
            end = start + n
            if end <= len(tokens):
                ngrams.append((start, end, " ".join(tokens[start:end])))
    return ngrams


def levenshtein_distance(left: str, right: str, max_distance: int = 3) -> int:
    """Bounded Levenshtein distance."""
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, rch in enumerate(right, 1):
            cost = 0 if lch == rch else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def looks_code_mixed(tokens: list[str]) -> bool:
    """Detect simple Latin + Devanagari code mixing."""
    has_latin = any(re.search(r"[a-z]", token) for token in tokens)
    has_devanagari = any(re.search(r"[\u0900-\u097f]", token) for token in tokens)
    return has_latin and has_devanagari

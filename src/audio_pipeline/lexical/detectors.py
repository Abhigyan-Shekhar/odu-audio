"""Rule-based lexical detectors used before learned classifiers are available."""

from collections import Counter

from src.audio_pipeline.lexical.normalization import normalize_text, tokenize

_SECOND_PERSON = {"you", "your", "tum", "tu", "tera", "teri", "aap", "तू", "तुम"}
_THREAT_VERBS = {
    "hit",
    "kill",
    "hurt",
    "beat",
    "slap",
    "throw",
    "marunga",
    "maarunga",
    "marungi",
    "maarungi",
    "nikal",
    "get out",
    "leave",
}
_IMPERATIVES = {
    "stop",
    "go",
    "leave",
    "shut",
    "come",
    "help",
    "listen",
    "ruk",
    "ja",
    "nikal",
    "chup",
}
_DISTRESS_PHRASES = {
    "help",
    "help me",
    "leave me",
    "stop",
    "no",
    "please stop",
    "mujhe bachao",
    "bachao",
    "mat karo",
    "nahi",
}
_REQUEST_WORDS = {"help", "water", "doctor", "come", "call", "please", "bachao"}


def threat_probability(text: str) -> float:
    """Estimate threat probability from simple multilingual lexical patterns."""
    normalized = normalize_text(text)
    tokens = set(tokenize(normalized))
    phrase_hit = any(phrase in normalized for phrase in _THREAT_VERBS if " " in phrase)
    verb_hit = any(verb in tokens for verb in _THREAT_VERBS if " " not in verb)
    second_person = bool(tokens.intersection(_SECOND_PERSON))
    if phrase_hit or (verb_hit and second_person):
        return 0.85
    if verb_hit:
        return 0.45
    return 0.0


def directed_insult_probability(text: str, profanity_matches: list[dict]) -> float:
    """Estimate whether abuse is directed at a person rather than general swearing."""
    tokens = set(tokenize(text))
    has_second_person = bool(tokens.intersection(_SECOND_PERSON))
    has_directed_entry = any(match.get("directed") for match in profanity_matches)
    if has_second_person and profanity_matches:
        return 0.90
    if has_directed_entry:
        return 0.70
    return 0.0


def imperative_probability(text: str) -> float:
    """Estimate forceful command language."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    imperative_hits = sum(1 for token in tokens if token in _IMPERATIVES)
    return min(1.0, imperative_hits / 2.0)


def distress_phrase_probability(text: str) -> float:
    """Estimate help/distress phrase presence."""
    normalized = normalize_text(text)
    if any(phrase in normalized for phrase in _DISTRESS_PHRASES):
        return 0.85
    return 0.0


def repeated_phrases(texts: list[str], min_count: int = 2) -> list[str]:
    """Find repeated unigrams, bigrams, or short utterances across recent windows."""
    candidates: list[str] = []
    for text in texts:
        tokens = tokenize(text)
        if 1 <= len(tokens) <= 6:
            candidates.append(" ".join(tokens))
        for i in range(max(0, len(tokens) - 1)):
            candidates.append(" ".join(tokens[i : i + 2]))
    counts = Counter(candidate for candidate in candidates if candidate)
    return [phrase for phrase, count in counts.items() if count >= min_count]


def repetition_probability(texts: list[str]) -> tuple[float, list[str], float]:
    """Estimate repeated phrase and repeated request probabilities."""
    repeats = repeated_phrases(texts)
    if not repeats:
        return 0.0, [], 0.0
    request_repeats = [
        phrase for phrase in repeats if set(phrase.split()).intersection(_REQUEST_WORDS)
    ]
    phrase_probability = min(1.0, 0.35 + 0.15 * len(repeats))
    request_probability = min(1.0, 0.45 + 0.20 * len(request_repeats))
    if not request_repeats:
        request_probability = 0.0
    return phrase_probability, repeats, request_probability

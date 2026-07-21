"""Pronunciation-aware profanity lexicon and ASR-error matching."""

from dataclasses import dataclass, field

from src.audio_pipeline.lexical.normalization import (
    levenshtein_distance,
    normalize_text,
    token_ngrams,
    tokenize,
)


@dataclass(frozen=True)
class LexiconEntry:
    """Profanity or abusive expression and its known variants."""

    canonical: str
    language: str
    severity: float
    variants: tuple[str, ...] = ()
    asr_confusions: tuple[str, ...] = ()
    incomplete_prefixes: tuple[str, ...] = ()
    directed: bool = False

    def normalized_terms(self) -> set[str]:
        terms = {self.canonical, *self.variants, *self.asr_confusions}
        return {normalize_text(term) for term in terms if normalize_text(term)}


@dataclass
class ProfanityLexicon:
    """
    Lexicon for English, Hindi/Hinglish, and local-language extensions.

    Matching uses exact variants, ASR confusions, incomplete prefixes, and bounded
    edit distance so shouted or partially spoken profanity can still be measured.
    """

    entries: list[LexiconEntry] = field(default_factory=list)
    edit_distance_threshold: int = 1

    @classmethod
    def default(cls) -> "ProfanityLexicon":
        return cls(
            entries=[
                LexiconEntry(
                    canonical="fuck",
                    language="en",
                    severity=0.85,
                    variants=("fucking", "fucker", "fuk", "phuck"),
                    asr_confusions=("fork", "fog", "fak", "fac"),
                    incomplete_prefixes=("fu", "fuc"),
                ),
                LexiconEntry(
                    canonical="shit",
                    language="en",
                    severity=0.55,
                    variants=("shitty", "shite"),
                    asr_confusions=("sheet", "ship", "sit"),
                    incomplete_prefixes=("shi",),
                ),
                LexiconEntry(
                    canonical="bastard",
                    language="en",
                    severity=0.65,
                    variants=("bastered",),
                    asr_confusions=("mustard", "pastor"),
                    incomplete_prefixes=("bast",),
                    directed=True,
                ),
                LexiconEntry(
                    canonical="chutiya",
                    language="hi-Latn",
                    severity=0.75,
                    variants=("chutia", "chootiya", "chutiye"),
                    asr_confusions=("chu diya", "chuti ya", "choo tea"),
                    incomplete_prefixes=("chu", "chut"),
                    directed=True,
                ),
                LexiconEntry(
                    canonical="madarchod",
                    language="hi-Latn",
                    severity=0.95,
                    variants=("maderchod", "madar chod", "mc"),
                    asr_confusions=("mother chod", "madam chod", "matar chod"),
                    incomplete_prefixes=("madar", "mader"),
                    directed=True,
                ),
                LexiconEntry(
                    canonical="behenchod",
                    language="hi-Latn",
                    severity=0.95,
                    variants=("bhenchod", "behen chod", "bc"),
                    asr_confusions=("bahan chod", "behind chod"),
                    incomplete_prefixes=("behen", "bhen"),
                    directed=True,
                ),
                LexiconEntry(
                    canonical="bhosdike",
                    language="hi-Latn",
                    severity=0.90,
                    variants=("bhosdi ke", "bhosadike", "bosdike"),
                    asr_confusions=("boss decay", "bose d k"),
                    incomplete_prefixes=("bhos", "bhosdi"),
                    directed=True,
                ),
                LexiconEntry(
                    canonical="haraami",
                    language="hi-Latn",
                    severity=0.65,
                    variants=("harami", "haraamkhor"),
                    asr_confusions=("harry me", "haram core"),
                    incomplete_prefixes=("haraam",),
                    directed=True,
                ),
                LexiconEntry(
                    canonical="saala",
                    language="hi-Latn",
                    severity=0.45,
                    variants=("sala", "saale"),
                    asr_confusions=("salah", "sale"),
                    incomplete_prefixes=("saa",),
                ),
            ]
        )

    def match(self, text: str) -> list[dict]:
        """Return profanity matches with confidence and match reason."""
        tokens = tokenize(text)
        matches: list[dict] = []
        seen: set[tuple[str, int, int]] = set()
        ngrams = token_ngrams(tokens)

        for start, end, phrase in ngrams:
            for entry in self.entries:
                terms = entry.normalized_terms()
                reason = None
                confidence = 0.0

                if phrase in terms:
                    reason = "exact_or_asr_variant"
                    confidence = 1.0
                elif any(
                    phrase.startswith(normalize_text(prefix))
                    and len(normalize_text(prefix)) >= 2
                    for prefix in entry.incomplete_prefixes
                ):
                    reason = "incomplete_prefix"
                    confidence = 0.65
                elif end - start == 1:
                    for term in terms:
                        distance = levenshtein_distance(
                            phrase, term, max_distance=self.edit_distance_threshold
                        )
                        if distance <= self.edit_distance_threshold:
                            reason = "edit_distance"
                            confidence = 0.75
                            break

                key = (entry.canonical, start, end)
                if reason is not None and key not in seen:
                    seen.add(key)
                    matches.append(
                        {
                            "canonical": entry.canonical,
                            "language": entry.language,
                            "matched_text": phrase,
                            "token_start": start,
                            "token_end": end,
                            "severity": entry.severity,
                            "confidence": confidence,
                            "reason": reason,
                            "directed": entry.directed,
                        }
                    )

        return matches

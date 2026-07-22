"""Templates for calm-profanity hard-negative dataset construction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CalmProfanityPrompt:
    """Counterfactual prompt for separating lexical content from vocal arousal."""

    prompt_id: str
    text: str
    language: str
    delivery: str = "calm"
    expected_content_label: str = "profanity"
    expected_arousal_label: str = "low_arousal"
    notes: str = ""


def default_calm_profanity_prompts() -> list[CalmProfanityPrompt]:
    """
    Minimal starter prompts for volunteer data collection.

    These are not clinical labels; they are counterfactual hard negatives to test
    that profanity alone does not become an agitation label.
    """
    return [
        CalmProfanityPrompt(
            prompt_id="calm_prof_en_001",
            text="I said the word damn calmly in a sentence.",
            language="en",
            notes="Mild profanity in neutral carrier phrase.",
        ),
        CalmProfanityPrompt(
            prompt_id="calm_prof_en_002",
            text="That was shit, but I am not upset.",
            language="en",
            notes="Profanity with explicit calm context.",
        ),
        CalmProfanityPrompt(
            prompt_id="calm_prof_hi_latn_001",
            text="Usne saala shabd shaant awaaz mein bola.",
            language="hi-Latn",
            notes="Romanized Hindi/Hinglish profanity in calm delivery.",
        ),
        CalmProfanityPrompt(
            prompt_id="calm_prof_hi_latn_002",
            text="Main chutiya shabd sirf example ke liye bol raha hoon.",
            language="hi-Latn",
            notes="Directed-profane term used as quoted content, calm delivery.",
        ),
    ]

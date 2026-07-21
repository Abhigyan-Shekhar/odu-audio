# Person 2: ASR, Profanity, and Language Modeling — Summary

## Responsibility

What was said.

## Core Deliverable

`speech window`
→ `transcript + word timestamps`
→ `profanity/threat/repetition/toxicity features`

The lexical branch is deliberately separate from agitation inference. Profanity is a content feature, not an agitation label.

## What Was Added

### Schemas

- `src/audio_pipeline/schemas/lexical_record.py`
  - `WordTimestamp`
  - `TranscriptSegment`
  - `LexicalFeatureRecord`

### Lexical Branch

- `src/audio_pipeline/lexical/asr.py`
  - Lazy `FasterWhisperASR` wrapper.
  - Requires `faster-whisper` only when ASR inference is actually run.
  - Emits transcript text, language, word timestamps, and average ASR confidence.

- `src/audio_pipeline/lexical/language.py`
  - Deterministic fallback language ID for English, Devanagari Hindi, and Romanized Hindi/Hinglish.
  - Flags simple code mixing.

- `src/audio_pipeline/lexical/lexicon.py`
  - English and Hindi/Hinglish starter profanity lexicon.
  - Includes spelling variants, incomplete prefixes, and common ASR confusions.
  - Uses exact matching plus bounded edit distance.

- `src/audio_pipeline/lexical/detectors.py`
  - Threat patterns.
  - Directed insult detection.
  - Imperative/request language.
  - Distress phrase detection.
  - Repeated phrase and repeated request detection over recent windows.

- `src/audio_pipeline/lexical/toxicity.py`
  - MuTox-compatible scoring boundary.
  - Deterministic fallback scorer for local tests and early pipeline work.

- `src/audio_pipeline/lexical/pipeline.py`
  - `LexicalFeatureExtractor` that produces the full Person 2 feature record.
  - Applies ASR-confidence weighting so low-confidence profanity is not over-trusted.
  - Flags low-confidence ASR, incomplete utterances, incomplete profanity candidates, and code mixing.

- `src/audio_pipeline/lexical/evaluation.py`
  - `asr_profanity_miss_rates()` to specifically measure whether ASR misses profanity under conditions such as:
    - incomplete profanity
    - shouted profanity
    - code-mixed profanity
  - Lightweight binary feature metrics for lexical model evaluation.

- `src/audio_pipeline/lexical/hard_negatives.py`
  - Starter calm-profanity prompts for hard-negative collection.
  - These are counterfactual prompts to verify that profanity alone does not become agitation.

### Package Import Safety

- `src/audio_pipeline/__init__.py` now lazily loads heavy model-backed components.
- This allows lexical modules and schemas to import without installing optional acoustic dependencies such as `opensmile`.

## Tests Added

- `tests/contracts/test_lexical_record_contract.py`
- `tests/unit/test_lexical_branch.py`

Coverage includes:

- lexical schema constraints and serialization;
- ASR-error profanity variants;
- incomplete profanity candidates;
- threat and directed insult detection;
- Hindi/Hinglish code-mixing flags;
- repeated request detection;
- ASR-confidence weighting;
- calm-profanity hard-negative prompts;
- ASR profanity miss-rate reporting by condition.

## Verification

Completed:

- `python -m compileall -q src/audio_pipeline/lexical src/audio_pipeline/schemas/lexical_record.py src/audio_pipeline/__init__.py tests/contracts/test_lexical_record_contract.py tests/unit/test_lexical_branch.py`
- Plain Python lexical assertion script passed.

Blocked by environment:

- `pytest` under `/opt/anaconda3/bin/python` segfaulted while importing pytest debugger machinery, before running project tests.
- Python 3.12 interpreters available locally did not have `pytest` and/or the repository's optional audio dependencies installed.

## Remaining Work

- Install and validate `faster-whisper` with real 16 kHz speech windows.
- Connect MuTox speech/text models when model weights and dependencies are available.
- Expand the Hindi/Hinglish/local-language profanity lexicon with deployment-specific review.
- Build ADIMA and DeToxy preprocessing jobs.
- Add an annotated ASR robustness benchmark for incomplete, shouted, noisy, and code-mixed profanity.
- Train learned classifiers for threat, directed insult, and toxicity after branch-level confusion matrices exist.

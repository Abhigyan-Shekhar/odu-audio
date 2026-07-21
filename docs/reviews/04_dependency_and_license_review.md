# Dependency & License Review

**Date:** 2026-07-21
**Reviewer:** [Pending Sign-off]
**Status:** Under Review

## 1. Core Dependencies & Licenses

| Dependency | Purpose | License | Notes |
|------------|---------|---------|-------|
| `numpy` / `scipy` | Signal processing | BSD-3-Clause | Standard |
| `sounddevice` | Audio capture | MIT | Standard |
| `onnxruntime` | YAMNet inference | MIT | Standard |
| `silero-vad` | Voice Activity Detection | MIT | Sourced from GitHub repo |
| `pyannote.audio` | Diarization & Speaker Embeddings | MIT | Requires HuggingFace token |
| `emotion2vec` | Acoustic affect embeddings | MIT | Sourced from GitHub repo |
| **`opensmile`** | eGeMAPS extraction | **audEERING Research License** | ⚠️ **Commercial use requires a paid license.** |

## 2. GPL Contamination
- An audit of the dependency tree reveals **no GPL or AGPL contamination**. All integrated code is under permissive licenses (MIT, BSD, Apache 2.0), with the notable exception of `opensmile`.

## 3. The openSMILE Exception
- The `opensmile` python package relies on a C-library developed by audEERING. While free for academic and research use, deployment of this pipeline in a for-profit commercial environment **mandates purchasing a commercial license from audEERING**.

**Approval:** [ ] Approved

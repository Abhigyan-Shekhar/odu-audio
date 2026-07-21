# Architecture Review

**Date:** 2026-07-21
**Reviewer:** [Pending Sign-off]
**Status:** Under Review

## 1. System Overview
The odu-audio pipeline is a front-end acoustic processing backbone designed for clinical environments. It streams audio, monitors quality, detects events, isolates patient speech, and extracts acoustic features (eGeMAPS) and affect embeddings (emotion2vec).

## 2. Schema Contracts
- `AudioChunk`: Carries waveform data with precise `timestamp`, `sequence_number`, and `sample_rate`.
- `AcousticFeatureRecord`: The primary output contract. Includes segment temporal bounds, speaker ID, quality status, eGeMAPS array, emotion2vec array, and YAMNet distress probabilities.
- **Approval:** [ ] Approved

## 3. Latency Targets & Backpressure
- **Capture-to-buffer:** <20ms
- **VAD Inference:** <5ms/chunk (Silero CPU)
- **Stable Endpointing:** 250-500ms hangover
- **Provisional Attribution:** <500ms
- **YAMNet Event Detection:** 1.0 - 1.5s (0.96s window)
- **Backpressure Mechanism:** `BoundedQueue` is implemented. If processing falls behind, the queue drops the oldest frames to prevent OOM crashes.
- **Approval:** [ ] Approved

## 4. Graceful Degradation
The `DegradationManager` monitors the 95th percentile latency:
- **Level 1 (>200ms):** Reduces emotion2vec frequency.
- **Level 2 (>500ms):** Disables emotion2vec.
- **Level 3 (>1000ms):** Disables diarization refinement.
- **Level 4 (>2000ms):** Disables eGeMAPS. Only VAD, quality, and YAMNet remain active.
- **Level 5 (>5000ms):** Quality/status events only.
- **Validation:** Confirmed via load tests (`test_degradation_load.py`).
- **Approval:** [ ] Approved

## 5. Error Handling & Isolation
- Expensive feature extractors (openSMILE) are wrapped in `ThreadPoolExecutor` with a strict timeout (e.g., 30s) to prevent a frozen C-library from halting the stream.
- The pipeline yields a `DropReason` instead of silently swallowing data if components fail or timeout.
- **Approval:** [ ] Approved

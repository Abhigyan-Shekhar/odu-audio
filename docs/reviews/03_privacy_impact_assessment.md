# Privacy Impact Assessment

**Date:** 2026-07-21
**Reviewer:** [Pending Sign-off]
**Status:** Under Review

## 1. Data Flows
1. **Ingestion:** Raw audio enters the system via OS-level drivers (Microphone) or File IO.
2. **Processing:** Raw audio resides in volatile memory (RAM) within `RingBuffer` and `BoundedQueue` constructs.
3. **Extraction:** Features (eGeMAPS, emotion2vec, YAMNet probabilities) are extracted from the memory buffers.
4. **Egress:** The extracted features are serialized to `acoustic_features.parquet`.

## 2. Biometric Data Handling (Speaker Embeddings)
- To perform Speaker Attribution, the system uses Pyannote (ECAPA-TDNN) to compute a high-dimensional mathematical vector (embedding) of the patient's voice (`SessionEnrollment`).
- **Mitigation:** These embeddings are stored strictly **in-memory** and are bound to the lifecycle of the Python process. They are **never** serialized to disk, logged, or transmitted across the network, effectively preventing the creation of a persistent biometric database of patients.

## 3. PII Identification and Control
- The raw audio stream is highly sensitive and may contain Protected Health Information (PHI) spoken by the patient or clinician.
- The pipeline's output (`acoustic_features.parquet`) contains abstract acoustic arrays (eGeMAPS, embeddings) and speaker metadata (`PATIENT`, `CLINICIAN`). It does *not* contain the raw audio or transcripts.
- However, acoustic features can theoretically be inverted or analyzed to reveal patient identity or affective state.
- **Requirement:** The Parquet outputs must be treated as PHI and subjected to the same encryption-at-rest and access controls as standard medical records.

## 4. Retention and Deletion
- As this pipeline only processes streaming data and outputs Parquet, retention and deletion of the resulting Parquet files must be managed by the downstream storage system.
- Internal buffers clear automatically as chunks slide out of the temporal window.

**Approval:** [ ] Approved

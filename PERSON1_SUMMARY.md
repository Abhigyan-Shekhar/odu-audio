# Person 1: Audio Pipeline & Acoustic Modeling — Summary

## Responsibility
Everything before lexical interpretation.

## Core Deliverable
`audio.wav` → `acoustic_features.parquet`

Speaker-specific speech windows with:
- Acoustic features (eGeMAPS, YAMNet events, emotion2vec embeddings)
- Speaker attribution (patient probability, status, method)
- Quality metrics (SNR, clipping, dropout)
- Reproducibility metadata (versions, hashes)

---

## What Changed from Original Specification

### ✅ Major Architectural Improvements

1. **Separated three systems** (was: single linear pipeline)
   - Real-time streaming inference
   - Offline dataset extraction
   - Model training & evaluation
   
2. **Branching pipeline** (was: linear VAD → diarization → features)
   - YAMNet now runs on **full audio stream** (NOT VAD-filtered)
   - Quality monitoring runs in parallel
   - Prevents losing distress events outside speech boundaries

3. **Component-specific latency targets** (was: universal <200ms)
   - Each component has algorithmic + compute latency
   - YAMNet: ~1.0-1.5s (cannot be sub-200ms)
   - emotion2vec: 1-3s (window-dependent)
   - VAD: <50ms preliminary, 250-500ms stable

4. **Graceful degradation** (was: vague "degradation under load")
   - 6 deterministic degradation levels (0-5)
   - Never silently drop windows → emit DropReason
   - Component activation logic at each level

5. **Patient attribution protocol** (was: undefined "patient-speaker filtering")
   - 5 explicit enrollment methods
   - States: PATIENT | NON_PATIENT | UNKNOWN | OVERLAP | LOW_CONFIDENCE
   - Biometric data handling (GDPR compliance)

6. **Moved augmentation to training** (was: in inference pipeline)
   - Augmentation NEVER runs during inference
   - Separate entry points prevent accidents
   - Training-only: noise, codec, SNR variation

7. **Replaced universal metrics with condition-specific** (was: VAD >95%, DER <10%)
   - VAD: metrics by SNR, vocal type (cry, whisper)
   - Diarization: patient-specific metrics (precision, recall, false attribution)
   - Feature reproducibility: bitwise + numerical tolerances

8. **Scoped privacy claims** (was: "compliant with medical regulations")
   - Technical controls described accurately
   - Compliance depends on deployment context
   - Biometric data explicitly flagged

9. **Framed emotion analysis carefully** (was: "emotion2vec embeddings")
   - "Acoustic affect embeddings" (not emotion classification)
   - Documented confounds (culture, illness, medication)
   - Explicit prohibited uses

---

## Key Files Created

### Documentation
- `README.md` — Complete project overview with architecture
- `ARCHITECTURE.md` — Detailed architectural decisions and rationale
- `TASKS.md` — 16-week implementation plan (10 phases)
- `PERSON1_SUMMARY.md` — This file

### Code Structure
- `src/audio_pipeline/` — Core package
  - `runtime/degradation.py` — Graceful degradation manager
  - `schemas/feature_record.py` — AcousticFeatureRecord schema
  - `schemas/speaker_attribution.py` — SpeakerAttribution schema
- Full directory structure created (see README)

### Configuration
- `configs/streaming_default.yaml` — Streaming pipeline config
- `requirements.txt` — Dependencies with installation notes

### Model Cards
- `model_cards/TEMPLATE.yaml` — Model card template
- `model_cards/emotion2vec.yaml` — emotion2vec detailed card

---

## Critical Design Decisions Explained

### 1. Why YAMNet Before VAD?

**Problem**: Speech-focused VAD may reject or fragment distress events (screams, cries, gasps).

**Solution**: Run YAMNet on full audio stream, then attribute events to patient using temporal overlap with diarization.

**Impact**: Ensures all distress events detected, not just those within speech boundaries.

---

### 2. Why Component-Specific Latency?

**Problem**: YAMNet requires 0.96s window + 0.48s hop. Universal <200ms is physically impossible.

**Solution**: Define latency at multiple levels:
- Capture: <20ms
- VAD preliminary: <50ms
- VAD stable: 250-500ms
- YAMNet: ~1.0-1.5s
- emotion2vec: 1-3s

**Impact**: Realistic expectations; system designers know what's available when.

---

### 3. Why Separate Streaming/Offline/Training?

**Problem**: Mixing inference and training leads to:
- Accidental augmentation during inference
- Latency-incompatible components in real-time path
- Confusion about data provenance

**Solution**: Three explicit entry points:
- `StreamProcessor.stream()` — Real-time
- `DatasetBuilder.extract()` — Batch features
- `LightGBMTrainer.fit()` — Training

**Impact**: Prevents silent errors; clear boundaries.

---

### 4. Why Explicit Attribution States?

**Problem**: "Target speaker identification" hides ambiguity. System might silently assign ambiguous speech to patient.

**Solution**: First-class states:
- PATIENT
- NON_PATIENT
- UNKNOWN
- OVERLAP
- LOW_CONFIDENCE

Plus: `patient_probability` (0.0-1.0) and `attribution_method`.

**Impact**: Downstream systems can handle uncertainty explicitly.

---

### 5. Why Never Silently Drop Windows?

**Problem**: Silent drops corrupt time series, lose critical events, hide system issues.

**Solution**: Always emit a record:
- `AcousticFeatureRecord` with features, OR
- `DropReason` explaining why dropped

**Impact**: Complete audit trail; debugging possible; no silent data loss.

---

### 6. Why "Affect Embedding" Not "Emotion"?

**Problem**: "Emotion" implies internal state detection. These models estimate **vocal expression patterns**.

**Solution**: Call it:
- `acoustic_affect_embedding`
- `vocal_arousal_score`
- `vocal_valence_score`

Document confounds: culture, illness, pain, medication, age.

**Impact**: Accurate expectations; prevents misuse.

---

## Patient Attribution: How It Works

### The Problem
**How does the system know who the patient is?**

This is non-trivial and has privacy implications.

### Supported Methods

| Method | How It Works | Privacy Impact |
|--------|--------------|----------------|
| Session enrollment | Patient speaks 5s at start; system extracts embedding | Biometric data (GDPR special-category) |
| Manual labeling | Clinician explicitly labels patient turn | Human-in-loop, gold standard |
| Channel assignment | Patient assigned to mic channel | No biometric, but requires setup |
| Embedding matching | Match to pre-existing template | Biometric linkage across sessions |
| Post-hoc correction | Human correction after diarization | Gold standard for datasets |

### SpeakerAttribution Output

```python
SpeakerAttribution(
    speaker_id="speaker_02",
    patient_probability=0.81,
    status="PATIENT",
    attribution_method="session_enrollment",
    overlap=False,
    provisional=True,
    embedding_distance=0.23,
    confidence_factors={"snr_db": 15.3}
)
```

### Privacy: Biometric Data Under GDPR

Voice embeddings are **biometric data** (UK ICO guidance). Requires:
- Legal basis (consent or legitimate interest)
- Data protection impact assessment (DPIA)
- Enhanced safeguards
- Purpose limitation
- Right to erasure

---

## Performance Metrics: What Actually Matters

### VAD

Not: "Accuracy >95%"

Yes: Condition-specific metrics

| Metric | Target | Condition |
|--------|--------|-----------|
| Speech miss rate | <5% | SNR ≥10 dB |
| False alarm rate | <5% | SNR ≥10 dB |
| Segment F1 | >0.90 | SNR ≥10 dB |
| Cry/scream recall | >0.80 | Non-speech vocalizations |
| Whisper detection | >0.70 | Low amplitude |

### Diarization

Not: "DER <10%" (universal)

Yes: Dataset-specific with conditions

```
DER < 10% on ODU-Clinical-Dataset-V1
  - 250 ms collar
  - Overlap included
  - Patient count: 2-3 speakers
  - SNR ≥ 10 dB
```

### Patient-Specific (Most Important)

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Patient precision | >0.90 | Minimize false patient attribution |
| Patient recall | >0.85 | Don't miss patient speech |
| False patient rate | <0.10 | Critical for privacy |
| Unknown rate | <0.15 | System confidence |

---

## Graceful Degradation: How It Works

Under load (latency P95 exceeds thresholds), system degrades **deterministically**:

| Level | Latency Trigger | Active Components | What's Disabled |
|-------|----------------|-------------------|-----------------|
| 0 | — | All | Normal operation |
| 1 | 200 ms | All (emotion2vec 2x hop) | Slower emotion updates |
| 2 | 500 ms | VAD, diarization, eGeMAPS, YAMNet | emotion2vec |
| 3 | 1000 ms | VAD, provisional attr., eGeMAPS, YAMNet | Diarization refinement |
| 4 | 2000 ms | VAD, quality, YAMNet | eGeMAPS, diarization |
| 5 | 5000 ms | Quality monitoring only | All features |

**Critical**: Never silent. If dropped, emit:

```python
DropReason(
    window_start_ms=12000,
    window_end_ms=14000,
    reason="DROPPED_BACKPRESSURE",
    degradation_level=3,
    timestamp_ms=14050,
    additional_info={"latency_p95_ms": 1250.0}
)
```

---

## Testing Strategy

### Performance Tests (Most Critical)

| Test | Target | File |
|------|--------|------|
| Capture-to-buffer latency | <20 ms | `test_latency.py` |
| Resampling compute | <5 ms/chunk | `test_latency.py` |
| VAD inference | <5 ms/chunk | `test_latency.py` |
| VAD preliminary | <50 ms | `test_latency.py` |
| Real-time capability | ≥1.0x speed | `test_realtime.py` |
| Sustained load | Hours | `test_realtime.py` |
| Degradation levels | All 6 levels | `test_degradation.py` |

### Accuracy Tests

| Test | Target | File |
|------|--------|------|
| VAD metrics | By SNR | `test_vad.py` |
| Diarization DER | <10% on dataset | `test_diarization.py` |
| Patient precision | >0.90 | `test_attribution.py` |
| Patient recall | >0.85 | `test_attribution.py` |
| False patient rate | <0.10 | `test_attribution.py` |
| Feature reproducibility | ±1e-6 | `test_features.py` |

### Edge Cases

- Short segments (<1s)
- Speaker overlap
- Cross-speaker windows
- Clipping and low SNR
- Dropout and silence
- Ring buffer boundaries
- Enrollment failures
- Model timeouts

---

## Privacy & Compliance: What's Required

### Technical Controls (Implemented)

- ✅ Pseudonymous IDs
- ✅ In-memory audio processing
- ✅ Encrypted temporary files
- ✅ Audit logging
- ✅ Data retention policies
- ✅ Access controls

### Organizational Requirements (Not Just Code)

- Legal basis for processing (GDPR Art. 6)
- Data protection impact assessment (DPIA)
- Consent or appropriate legal basis for biometric data
- Staff training on limitations
- Incident response procedures
- Vendor contracts (if applicable)
- Regular audits

### What You Can Claim

> "This system is designed to support privacy-preserving deployments with pseudonymous identifiers, in-memory processing, and audit logging. Regulatory compliance (HIPAA, GDPR) depends on the complete deployment environment, organizational controls, applicable jurisdiction, and validated operating procedures."

### What You Cannot Claim

> ~~"HIPAA compliant"~~  
> ~~"GDPR compliant"~~  
> ~~"Determines patient emotional state"~~  
> ~~"Suitable for autonomous diagnosis"~~

---

## Implementation Timeline

**Total: 16 weeks (4 months)**

| Phase | Duration | Focus |
|-------|----------|-------|
| 1 | Week 1-2 | Core infrastructure (capture, preprocessing, schemas) |
| 2 | Week 2-3 | Segmentation & quality (VAD, quality monitoring) |
| 3 | Week 3-4 | Event detection (YAMNet, full-stream) |
| 4 | Week 4-6 | Diarization & attribution (enrollment, provisional/batch) |
| 5 | Week 6-8 | Speech features (eGeMAPS, emotion2vec) |
| 6 | Week 8-10 | Training pipeline (augmentation, LightGBM) |
| 7 | Week 10-12 | Runtime system (degradation, streaming, offline) |
| 8 | Week 12-14 | Testing & validation (latency, accuracy, edge cases) |
| 9 | Week 14-15 | Privacy & compliance (DPIA, model cards, audit) |
| 10 | Week 15-16 | Documentation & deployment (docs, CLI, Docker) |

See `TASKS.md` for detailed task breakdown.

---

## Dependencies & External Accounts

### Required

- Python 3.9+
- HuggingFace account + token (for Pyannote)
- Accept Pyannote user agreements
- GPU (recommended for diarization & emotion2vec)

### Optional

- Pyannote commercial streaming license (for <300ms diarization)
- Clinical test dataset
- Target deployment hardware for benchmarking

---

## Success Criteria

1. ✅ Component-level latency targets met on target hardware
2. ✅ Processes audio at ≥1.0x real-time speed
3. ✅ VAD and diarization metrics meet targets on defined dataset
4. ✅ Graceful degradation works under load
5. ✅ No PHI in outputs; biometric data handled per GDPR
6. ✅ Features are numerically reproducible across runs
7. ✅ Complete model cards, data cards, deployment guide
8. ✅ >80% code coverage; all edge cases handled

---

## Next Steps

1. **Review** this specification with stakeholders
2. **Validate** latency targets on target hardware
3. **Acquire** HuggingFace account and Pyannote access
4. **Obtain** clinical test dataset (de-identified)
5. **Begin** Phase 1 implementation (core infrastructure)
6. **Set up** development environment (see `requirements.txt`)

---

## Questions to Resolve Before Implementation

1. **Dataset**: Which clinical dataset for evaluation? Need DER baselines.
2. **Hardware**: CPU-only or GPU available? Affects latency benchmarks.
3. **Pyannote streaming**: Commercial license available? Affects provisional diarization latency.
4. **Legal**: GDPR legal basis established? Biometric data requires explicit handling.
5. **Enrollment**: Which method preferred? (session_start, manual, channel, embedding?)
6. **Deployment**: On-premise or cloud? Affects privacy architecture.

---

## Contact & Ownership

**Person 1 Responsibilities**:
- Audio capture through acoustic feature extraction
- VAD, diarization, speaker attribution
- Quality monitoring and event detection
- Acoustic features (eGeMAPS, YAMNet, emotion2vec)
- Privacy-preserving pipeline
- Data manifest format

**Interfaces with**:
- Person 2 (lexical interpretation): receives `acoustic_features.parquet`
- Person 3 (clinical context): may need speaker attribution logic
- Data team: provides clinical test dataset
- Legal/compliance: validates privacy controls

---

## References

- **Silero VAD**: https://github.com/snakers4/silero-vad
- **Pyannote**: https://github.com/pyannote/pyannote-audio
- **YAMNet**: https://blog.tensorflow.org/2021/03/
- **emotion2vec**: https://github.com/ddlBoJack/emotion2vec
- **openSMILE**: https://audeering.github.io/opensmile/
- **eGeMAPS**: Eyben et al., 2016
- **HIPAA**: https://www.hhs.gov/hipaa
- **GDPR (biometric)**: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/

---

## Change Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2024-01-15 | Initial specification | Based on Person 1 task description |
| 2024-01-15 | Separated streaming/offline/training | Prevent augmentation in inference |
| 2024-01-15 | Moved YAMNet before VAD | Detect distress outside speech |
| 2024-01-15 | Component-specific latency | Realistic expectations |
| 2024-01-15 | Explicit attribution protocol | Handle uncertainty |
| 2024-01-15 | Graceful degradation (6 levels) | Deterministic behavior under load |
| 2024-01-15 | Scoped privacy/emotion claims | Accurate, defensible statements |

---

**This architecture proposal addresses all major architectural concerns from the feedback and is ready for technical review and iterative implementation. End-to-end processing, model integration, automated testing, performance benchmarking, security and privacy validation, and clinical-domain evaluation remain outstanding before production deployment.**

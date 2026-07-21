# Audio Pipeline Project Overview

## Person 1: Complete Specification

### 📦 What's Been Delivered

A production-oriented architecture proposal and implementation scaffold:

✅ **Separated architecture** — Streaming, Offline, Training (no accidental augmentation)  
✅ **Realistic latency targets** — Component-specific, not universal <200ms  
✅ **Branching pipeline** — YAMNet on full stream, not VAD-filtered  
✅ **Graceful degradation** — 6 deterministic levels with explicit drop reasons  
✅ **Patient attribution** — 5 methods with uncertainty handling  
✅ **Core schemas** — AcousticFeatureRecord, SpeakerAttribution (implemented, untested)  
✅ **Privacy by design** — Pseudonymous IDs, biometric data handling  
✅ **Scoped claims** — No false promises about compliance or emotion detection  
✅ **Phased backlog** — Implementation plan with dependencies

**Status**: Ready for technical review and iterative implementation  
**Not yet implemented**: Audio processing, model integration  
**Not yet validated**: Performance, security, clinical accuracy  

---

## 📂 Project Structure

```
odu-audio/
├── README.md                    ⭐ Main documentation (30+ pages)
├── ARCHITECTURE.md              ⭐ Design decisions & rationale
├── PERSON1_SUMMARY.md           ⭐ Executive summary
├── QUICKSTART.md                ⭐ Day-1 implementation guide
├── TASKS.md                     📋 16-week detailed plan
├── requirements.txt             📦 Dependencies
│
├── configs/
│   └── streaming_default.yaml   ⚙️  Production-ready config
│
├── model_cards/
│   ├── TEMPLATE.yaml            📄 Model card template
│   └── emotion2vec.yaml         📄 Complete emotion2vec card
│
└── src/audio_pipeline/
    ├── __init__.py              🐍 Package entry point
    ├── runtime/                 🔴 Real-time streaming
    │   └── degradation.py       ✅ Graceful degradation (complete)
    ├── offline/                 💾 Batch processing
    ├── capture/                 🎤 Microphone & ring buffer
    ├── segmentation/            🔊 VAD & endpointing
    ├── speakers/                👥 Diarization & attribution
    ├── features/                📊 eGeMAPS, YAMNet, emotion2vec
    ├── quality/                 ✓  Clipping, SNR, dropout
    ├── schemas/                 📋 Data contracts
    │   ├── feature_record.py    ✅ AcousticFeatureRecord (complete)
    │   └── speaker_attribution.py ✅ SpeakerAttribution (complete)
    ├── storage/                 💾 Parquet output
    └── privacy/                 🔒 Anonymization & audit
```

---

## 🎯 Key Deliverables

### Input
```
audio.wav (raw audio, any sample rate)
```

### Output
```
acoustic_features.parquet
├── Session identifiers (pseudonymous)
├── Temporal boundaries (ms + samples)
├── Speaker attribution (probability, status, method)
├── VAD metrics (probability, voiced ratio, overlap)
├── Acoustic features
│   ├── eGeMAPS (88 functionals)
│   ├── YAMNet event scores (dict)
│   └── emotion2vec embedding (list)
├── Quality metrics (SNR, clipping, dropout)
└── Reproducibility (versions, hashes)
```

---

## 🏗️ Three Separate Systems

### 1. Real-Time Streaming (`runtime/`)
```python
from audio_pipeline.runtime import StreamProcessor

processor = StreamProcessor(config_path="configs/streaming_default.yaml")
processor.start()

for feature_record in processor.stream():
    if feature_record.is_patient_speech():
        print(f"Patient: {feature_record.window_start_ms}ms")
```

**Use for**: Live clinical sessions, real-time monitoring

### 2. Offline Dataset (`offline/`)
```python
from audio_pipeline.offline import DatasetBuilder

builder = DatasetBuilder(config_path="configs/offline_default.yaml")
features_df = builder.extract("audio.wav")
features_df.to_parquet("dataset_features.parquet")
```

**Use for**: Dataset generation, research analysis

### 3. Training (`training/`)
```python
from training import LightGBMTrainer

trainer = LightGBMTrainer(config_path="configs/training_default.yaml")
trainer.load_data("dataset_features.parquet")
trainer.augment()  # Noise, codec, SNR (NEVER in inference)
trainer.train()
trainer.save("models/lightgbm_v1.pkl")
```

**Use for**: Model training, evaluation, calibration

---

## ⚡ Latency Targets (Component-Specific)

| Component | Target | Why |
|-----------|--------|-----|
| Capture-to-buffer | <20 ms | Hardware/OS constraint |
| Resampling | <5 ms/chunk | Must be faster than real-time |
| VAD inference | <5 ms/chunk | Silero is <1ms CPU |
| VAD preliminary | <50 ms | Initial speech decision |
| VAD stable | 250-500 ms | Reliable endpoint |
| Provisional attribution | <500 ms | Initial speaker ID |
| YAMNet | ~1.0-1.5 s | **Algorithmic: 0.96s window** |
| emotion2vec | 1-3 s | **Window-dependent** |
| Final diarization | Post-segment | Offline refinement |

**Critical**: YAMNet and emotion2vec **cannot** meet <200ms due to their window requirements.

---

## 🌊 Pipeline Flow (Branching, Not Linear)

```
                  Audio Input
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
   Quality Monitor          Event Detection
   (full stream)            (YAMNet, full stream)
         ↓                           ↓
   Clipping, SNR            Distress Events
         ↓
    VAD Segments
         ↓
   ┌─────┴─────┐
   ↓           ↓
Diarization  Provisional
(batch)      Attribution
   ↓           ↓
   └─────┬─────┘
         ↓
  Patient Speech
         ↓
  Acoustic Features
  (eGeMAPS, emotion2vec)
```

**Why YAMNet before VAD?**  
→ Screams/cries may fall outside speech VAD boundaries  
→ Need to detect ALL distress events, not just speech

---

## 📊 Performance Metrics (Not Universal Numbers)

### VAD (By Condition)
| Metric | Target | Condition |
|--------|--------|-----------|
| Speech miss rate | <5% | SNR ≥10 dB |
| False alarm rate | <5% | SNR ≥10 dB |
| Cry/scream recall | >80% | Non-speech |

### Diarization (Dataset-Specific)
```
DER < 10% on [Your Clinical Dataset]
  250ms collar, overlap included, 2-3 speakers, SNR ≥10 dB
```

### Patient Attribution (Most Important)
| Metric | Target |
|--------|--------|
| Patient precision | >90% |
| Patient recall | >85% |
| False patient rate | <10% |

---

## 🛡️ Graceful Degradation (6 Levels)

Under load (latency P95 exceeds threshold):

| Level | Trigger | Active | Disabled |
|-------|---------|--------|----------|
| 0 | — | All | — |
| 1 | 200ms | All (emotion2vec 2x hop) | — |
| 2 | 500ms | All except emotion2vec | emotion2vec |
| 3 | 1s | Provisional attribution | Diarization refinement |
| 4 | 2s | VAD, quality, YAMNet | Features |
| 5 | 5s | Quality only | All features |

**Critical**: Never silently drop. Always emit `DropReason`.

---

## 👤 Patient Attribution (5 Methods)

1. **Session enrollment** — Patient speaks 5s at start
2. **Manual labeling** — Clinician labels patient turn
3. **Channel assignment** — Patient on specific mic channel
4. **Embedding matching** — Match to pre-existing template
5. **Post-hoc correction** — Human correction after diarization

### Attribution States
- `PATIENT` — Attributed with confidence
- `NON_PATIENT` — Non-patient speaker
- `UNKNOWN` — Cannot determine
- `OVERLAP` — Multiple speakers
- `LOW_CONFIDENCE` — Below threshold

**Privacy**: Voice embeddings = biometric data (GDPR special-category)

---

## 🔒 Privacy & Compliance

### Technical Controls ✅
- Pseudonymous IDs (no PHI)
- In-memory processing
- Encrypted temp files
- Audit logging

### What You Can Claim ✅
> "Designed to support privacy-preserving deployments. Regulatory compliance depends on deployment environment, organizational controls, and validated procedures."

### What You Cannot Claim ❌
- ~~"HIPAA compliant"~~ (requires organizational controls)
- ~~"Determines emotion"~~ (acoustic affect, not internal state)
- ~~"Suitable for autonomous diagnosis"~~ (requires human oversight)

---

## 📚 Read in This Order

1. **PERSON1_SUMMARY.md** (10 min) — What changed & why
2. **QUICKSTART.md** (15 min) — Day 1 setup & Week 1 implementation
3. **ARCHITECTURE.md** (30 min) — Design decisions in depth
4. **README.md** (45 min) — Complete documentation
5. **TASKS.md** (20 min) — 16-week plan
6. **configs/streaming_default.yaml** (5 min) — Configuration
7. **model_cards/emotion2vec.yaml** (10 min) — Model card example

**Total reading time**: ~2.5 hours for complete understanding

---

## ⏱️ Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1 | Week 1-2 | Core infrastructure |
| 2 | Week 2-3 | VAD & quality |
| 3 | Week 3-4 | YAMNet events |
| 4 | Week 4-6 | Diarization & attribution |
| 5 | Week 6-8 | eGeMAPS & emotion2vec |
| 6 | Week 8-10 | Training pipeline |
| 7 | Week 10-12 | Runtime system |
| 8 | Week 12-14 | Testing & validation |
| 9 | Week 14-15 | Privacy & compliance |
| 10 | Week 15-16 | Documentation & deployment |

**Total: 16 weeks (4 months)**

---

## 🚀 Next Steps

### Immediate (Week 0)
- [ ] Review this specification with stakeholders
- [ ] Acquire HuggingFace account + Pyannote access
- [ ] Set up development environment
- [ ] Validate latency targets on target hardware

### Week 1
- [ ] Implement audio capture & ring buffer
- [ ] Implement 16 kHz resampling
- [ ] Benchmark capture-to-buffer latency (<20ms?)

### Week 2
- [ ] Integrate Silero VAD
- [ ] Implement preliminary + stable endpoints
- [ ] Benchmark VAD latency (<5ms inference?)

### Week 3-4
- [ ] Integrate YAMNet on full stream
- [ ] Test distress event detection
- [ ] Measure YAMNet availability (~1-1.5s?)

---

## ❓ Questions Before Implementation

1. **Dataset**: Which clinical dataset for evaluation?
2. **Hardware**: CPU-only or GPU available?
3. **Pyannote**: Commercial streaming license?
4. **Legal**: GDPR legal basis established?
5. **Enrollment**: Preferred method (session/manual/channel)?
6. **Deployment**: On-premise or cloud?

---

## ✅ Success Criteria

- [ ] Component latencies meet targets on hardware
- [ ] Real-time processing ≥1.0x speed
- [ ] VAD & diarization metrics on dataset
- [ ] Graceful degradation tested under load
- [ ] No PHI in outputs
- [ ] Features numerically reproducible
- [ ] Complete model cards
- [ ] >80% code coverage

---

## 🔗 Key References

- **Silero VAD**: https://github.com/snakers4/silero-vad
- **Pyannote**: https://github.com/pyannote/pyannote-audio
- **YAMNet**: https://blog.tensorflow.org/2021/03/
- **emotion2vec**: https://github.com/ddlBoJack/emotion2vec
- **openSMILE**: https://audeering.github.io/opensmile/
- **GDPR biometric**: https://ico.org.uk/

---

## 📧 Ownership

**Person 1**: Audio capture → acoustic features  
**Interfaces with**:
- Person 2 (lexical): receives `acoustic_features.parquet`
- Person 3 (clinical): may use speaker attribution
- Data team: provides test dataset
- Legal: validates privacy controls

---

## 🎉 Status

**Current**: Production-oriented architecture proposal and scaffold  
**Implemented**: Core schemas, degradation logic (untested)  
**Not Yet Complete**: Audio processing, model integration, testing, benchmarks, clinical evaluation  
**Next**: Technical review → Address known gaps → Phase 1 implementation

---

**This architecture proposal addresses all major architectural concerns and is ready for technical review and iterative implementation. End-to-end processing, model integration, automated testing, performance benchmarking, security and privacy validation, and clinical-domain evaluation remain outstanding before production deployment.**

🚀 Ready for technical review and implementation!

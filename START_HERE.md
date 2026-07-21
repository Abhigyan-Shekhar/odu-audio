# 👋 Start Here

## Welcome to the Audio Pipeline Project

This is Person 1's complete specification for the audio processing pipeline (everything before lexical interpretation).

---

## 🎯 Quick Links

### Start here (do this first):
1. **[NEXT_ACTIONS.md](NEXT_ACTIONS.md)** (5 min) — Verification commands and immediate next steps

### For verification and status:
2. **[FINAL_STATUS.md](FINAL_STATUS.md)** (10 min) — Delivery status, verification checklist
3. **[FILE_INVENTORY.md](FILE_INVENTORY.md)** (5 min) — Complete file manifest

### For understanding the project:
4. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** (5 min) — Visual summary
5. **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** (10 min) — What's implemented, what's not
6. **[PERSON1_SUMMARY.md](PERSON1_SUMMARY.md)** (15 min) — What changed & why

### Ready to implement? Start here:
6. **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** (45 min) — Vertical-slice approach, testable milestones
7. **[CONTRACTS.md](CONTRACTS.md)** (30 min) — Data contracts (freeze before implementation)
8. **[QUICKSTART.md](QUICKSTART.md)** (reference) — Code examples

### Want deep understanding? Read these:
9. **[ARCHITECTURE.md](ARCHITECTURE.md)** (45 min) — Design decisions, rationale
10. **[README.md](README.md)** (60 min) — Complete documentation
11. **[TASKS.md](TASKS.md)** (reference) — Original phased backlog

### Configuration & examples:
12. **[configs/streaming_default.yaml](configs/streaming_default.yaml)** — Production config
13. **[model_cards/emotion2vec.yaml](model_cards/emotion2vec.yaml)** — Model card example

---

## 📂 What's in This Repository?

```
✅ Production-oriented architecture proposal
✅ Core schemas (implemented, untested)
✅ Degradation manager (initial implementation, untested)
✅ Phased implementation backlog
✅ Model card templates
✅ Privacy & compliance considerations
✅ Performance targets (to be validated)
✅ Implementation guide and code examples
```

**Status**: Ready for technical review and iterative implementation  
**Not yet implemented**: Audio processing, model integration  
**Not yet validated**: Functional correctness, performance, security, clinical accuracy

---

## ⏱️ How Long Will This Take?

| Task | Time |
|------|------|
| Understand the project | 2-3 hours |
| Set up environment | 1-2 hours |
| Implement Phase 1 (core) | 2 weeks |
| Complete implementation | 16 weeks |

---

## 🚦 Status

**Specification**: ✅ Documented  
**Schemas**: ✅ Implemented (untested)  
**Implementation**: ⏸️ Not yet started  
**Validation**: ⏸️ No tests run  
**Next Step**: Technical review → Address known gaps → Begin Phase 1

---

## 🎓 Learning Path

### If you have 10 minutes:
→ Read **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**

### If you have 30 minutes:
→ Read **[PERSON1_SUMMARY.md](PERSON1_SUMMARY.md)**  
→ Skim **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)**

### If you have 2 hours:
→ Read everything above  
→ Read **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** in detail  
→ Read **[CONTRACTS.md](CONTRACTS.md)**  
→ Review **[configs/streaming_default.yaml](configs/streaming_default.yaml)**

### If you have a full day:
→ Read all documentation  
→ Review **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** and **[CONTRACTS.md](CONTRACTS.md)**  
→ Set up environment (see **[QUICKSTART.md](QUICKSTART.md)**)  
→ Test schemas in Python REPL  
→ Plan Week 1 work (Milestone 0: Engineering Baseline)

---

## 🔑 Key Concepts

This project implements **three separate systems**:

1. **Real-time streaming** (`runtime/`) — Live audio with latency constraints
2. **Offline batch** (`offline/`) — Dataset feature extraction
3. **Training** (`training/`) — Augmentation & LightGBM training

**Critical**: These are architecturally separate to prevent:
- Accidental augmentation during inference
- Latency-incompatible components in real-time path
- Confusion between provisional and final results

---

## 🎯 Core Deliverable

```
Input:  audio.wav (raw audio, any sample rate)
Output: acoustic_features.parquet (speaker-specific speech windows with features)
```

### What's in acoustic_features.parquet?
- Session identifiers (pseudonymous, no PHI)
- Temporal boundaries (milliseconds + samples)
- Speaker attribution (patient probability, status, method)
- VAD metrics (probability, voiced ratio, overlap)
- Acoustic features:
  - eGeMAPS (88 functionals)
  - YAMNet event scores (scream, cry, yell, gasp, groan)
  - emotion2vec embeddings (acoustic affect, not emotion)
- Quality metrics (SNR, clipping, dropout)
- Reproducibility metadata (versions, hashes)

---

## 💡 Major Improvements from Original

The specification was significantly enhanced based on expert feedback:

✅ **Separated streaming/offline/training** (was: single linear pipeline)  
✅ **Branching architecture** (YAMNet on full stream, not VAD-filtered)  
✅ **Component-specific latency** (not universal <200ms)  
✅ **Graceful degradation** (6 levels, never silent drops)  
✅ **Patient attribution protocol** (5 methods, uncertainty handling)  
✅ **Scoped privacy claims** (no false compliance promises)  
✅ **Realistic metrics** (condition-specific, not universal targets)  
✅ **Careful emotion framing** (affect embeddings, not emotion detection)

See **[PERSON1_SUMMARY.md](PERSON1_SUMMARY.md)** for detailed explanations.

---

## 🛠️ Implementation Order

**Vertical-slice approach** (see **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)**):

1. **Week 1**: Engineering baseline (tests, CI, contract freeze) ← Start here
2. **Week 2-3**: WAV-to-Parquet backbone (first working system!)
3. **Week 4-5**: Streaming infrastructure  
4. **Week 6**: VAD integration
5. **Week 7**: openSMILE (first real feature extractor)
6. **Week 8**: YAMNet (parallel to VAD)
7. **Week 9-10**: Speaker handling with uncertainty
8. **Week 11**: Degradation validation under load
9. **Week 12**: Evaluation pipeline
10. **Week 13-14**: Formal reviews

**Key principle**: Build one end-to-end path first, test thoroughly, then add components incrementally.

---

## ❓ FAQs

### Q: Can YAMNet meet <200ms latency?
**A**: No. YAMNet requires a 0.96s window + 0.48s hop, so availability is ~1.0-1.5s. This is an algorithmic constraint, not a performance issue.

### Q: Why is YAMNet before VAD?
**A**: Distress events (screams, cries, gasps) may fall outside speech VAD boundaries. Running YAMNet on the full stream ensures we detect all events.

### Q: Why separate streaming/offline/training?
**A**: Prevents accidental augmentation during inference, ensures appropriate latency expectations, and maintains clear architectural boundaries.

### Q: How do we know who the patient is?
**A**: Five methods supported: session enrollment, manual labeling, channel assignment, embedding matching, post-hoc correction. See **[ARCHITECTURE.md](ARCHITECTURE.md)** § Patient Attribution.

### Q: Is this HIPAA/GDPR compliant?
**A**: The system provides technical controls (pseudonymous IDs, encryption, audit logging) but compliance depends on organizational controls, deployment context, and validated procedures. Voice embeddings are biometric data under GDPR.

### Q: What if latency targets aren't met?
**A**: System degrades gracefully across 6 levels (see **[ARCHITECTURE.md](ARCHITECTURE.md)** § Graceful Degradation). Always emits records explaining why windows were dropped.

---

## 🚀 Quick Start (Day 1)

### 1. Set up environment
```bash
# Create environment
conda create -n audio-pipeline python=3.10
conda activate audio-pipeline

# Install core deps
pip install numpy scipy pandas pyarrow soundfile librosa torch torchaudio

# Install Silero VAD
pip install git+https://github.com/snakers4/silero-vad.git
```

### 2. Get HuggingFace token
- Create account: https://huggingface.co
- Accept agreements: https://huggingface.co/pyannote/speaker-diarization
- Set token: `export HF_TOKEN="your_token"`

### 3. Test schemas
```python
from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord

record = AcousticFeatureRecord(
    session_id="session_001",
    stream_id="stream_001",
    window_start_ms=0,
    window_end_ms=2000,
    source_start_sample=0,
    source_end_sample=32000,
    attribution_status="PATIENT",
    egemaps=[0.0] * 88,
)

print(record.is_patient_speech())  # Should work
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed examples.

---

## 📞 Contact & Ownership

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

## ✅ Before You Start

- [ ] Read **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** (validation status, known gaps)
- [ ] Read **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**
- [ ] Read **[PERSON1_SUMMARY.md](PERSON1_SUMMARY.md)**
- [ ] Skim **[QUICKSTART.md](QUICKSTART.md)**
- [ ] Engineering review with stakeholders
- [ ] Define stream message contract (timestamps, sequence numbers)
- [ ] Specify deployment target (hardware, environment)
- [ ] Document model artifact management
- [ ] Complete privacy threat model
- [ ] Define clinical evaluation plan
- [ ] Get HuggingFace account
- [ ] Set up development environment
- [ ] Obtain clinical test dataset (if available)
- [ ] Confirm hardware (CPU vs GPU)
- [ ] Establish GDPR legal basis (if applicable)

---

## 🎉 Ready?

→ **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** to understand what's implemented and what's not

→ **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** for vertical-slice milestones

→ **[CONTRACTS.md](CONTRACTS.md)** for data schemas (review and freeze first!)

→ **[QUICKSTART.md](QUICKSTART.md)** for environment setup and code examples

→ **[ARCHITECTURE.md](ARCHITECTURE.md)** for deep design understanding

---

**This architecture proposal is ready for technical review and iterative implementation. First milestone (Milestone 0): Engineering baseline with tests, CI, and contracts proposed as v0.1-provisional. Second milestone (Milestone 1): Working WAV-to-Parquet converter upon completion of defined acceptance criteria.** 🚀

# Audio Pipeline Architecture

## Executive Summary

This document describes the architecture for Person 1's audio processing pipeline, which handles everything before lexical interpretation. The system is designed around three **architecturally separate** concerns:

1. **Real-time streaming inference** — processes live audio with component-specific latency targets
2. **Offline dataset feature extraction** — batch processing for dataset generation
3. **Model training and evaluation** — augmentation, feature selection, and LightGBM training

## Core Architectural Principles

### 1. Separation of Concerns

The pipeline explicitly separates:
- **Streaming** (`runtime/`) vs **Batch** (`offline/`) processing
- **Inference** vs **Training** (`training/`)
- **Feature extraction** vs **Prediction**

This prevents:
- Accidental augmentation during inference
- Latency-incompatible components in real-time path
- Confusion between provisional and final results

### 2. Branching Pipeline Architecture

The pipeline branches **early** rather than flowing linearly:

```
                    Audio Input
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
       Quality Monitor      Event Detection
       (full stream)        (full stream, YAMNet)
              ↓                     ↓
         ┌────┴────┐         Distress Events
         ↓         ↓
    Clipping    SNR Est.
              
              ↓
         VAD Segments
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Diarization      Provisional
(batch)          Attribution
    ↓                   ↓
Patient Speech Windows
    ↓
Acoustic Features
(eGeMAPS, emotion2vec)
```

**Critical design decision**: YAMNet operates on the **full audio stream**, not VAD-filtered speech. This ensures distress events (screams, cries, gasps) are detected even if they fall outside speech boundaries.

### 3. Component-Specific Latency Targets

Instead of a universal "<200ms" requirement, each component has **algorithmic** and **compute** latency targets:

| Component | Algorithmic Latency | Compute Target | Notes |
|-----------|---------------------|----------------|-------|
| Capture-to-buffer | N/A | <20 ms | Hardware/OS dependent |
| Resampling | Instantaneous | <5 ms/chunk | Must be faster than real-time |
| VAD inference | ~30ms window | <5 ms | Silero is <1ms on CPU |
| VAD preliminary | <50 ms | — | Initial speech/silence decision |
| VAD stable endpoint | 250-500 ms | — | Reliable segment boundary |
| Quality monitoring | Instantaneous | <10 ms | Clipping, dropout, SNR |
| Provisional attribution | <500 ms | — | Initial speaker ID |
| YAMNet | 0.96s window | ~50-100 ms | **Cannot be sub-200ms** |
| emotion2vec | 1-3s window | ~50-200 ms | **Window-dependent** |
| Final diarization | Post-segment | Offline | Refinement after segment complete |

### 4. Graceful Degradation

Under load, the system degrades **deterministically** across 6 levels:

- **Level 0**: All extractors active (normal)
- **Level 1**: Reduce emotion2vec frequency (2x hop)
- **Level 2**: Disable emotion2vec
- **Level 3**: Disable diarization refinement (use provisional)
- **Level 4**: VAD + quality + critical events only
- **Level 5**: Quality/status monitoring only

**Critical**: Never silently drop windows. Always emit a `DropReason` record.

### 5. Privacy by Design

- **Pseudonymous IDs**: No PHI in feature tables
- **In-memory processing**: Raw audio not written to disk
- **Biometric data handling**: Speaker and emotion embeddings treated as GDPR special-category data
- **Audit logging**: All operations tracked
- **Data manifest**: Provenance tracking

## Data Flow

### Real-Time Streaming

```python
# Entry point
from audio_pipeline.runtime import StreamProcessor

processor = StreamProcessor(config_path="configs/streaming_default.yaml")
processor.start()

for feature_record in processor.stream():
    # feature_record: AcousticFeatureRecord
    # - session_id, stream_id (pseudonymous)
    # - window boundaries (ms and samples)
    # - speaker attribution (probability, status, method)
    # - VAD metrics (probability, voiced ratio, overlap)
    # - acoustic features (eGeMAPS, YAMNet events, emotion2vec)
    # - quality metrics (SNR, clipping, dropout)
    # - reproducibility (versions, hashes)
    pass
```

### Offline Dataset Extraction

```python
# Entry point
from audio_pipeline.offline import DatasetBuilder

builder = DatasetBuilder(config_path="configs/offline_default.yaml")
features_df = builder.extract("audio.wav")
features_df.to_parquet("dataset_features.parquet")
```

### Training

```python
# Entry point
from training import LightGBMTrainer

trainer = LightGBMTrainer(config_path="configs/training_default.yaml")
trainer.load_data("dataset_features.parquet")
trainer.augment()  # Room noise, codec, SNR variation
trainer.train()
trainer.calibrate()
trainer.evaluate()
trainer.save("models/lightgbm_baseline_v1.pkl")
```

## Patient-Speaker Attribution

### The Challenge

**How does the system know who the patient is?**

This is a major technical and privacy problem requiring explicit protocol.

### Supported Methods

1. **Session enrollment** (`AttributionMethod.SESSION_ENROLLMENT`)
   - Patient speaks for 5 seconds at session start
   - System extracts speaker embedding as reference
   - Subsequent speech matched against reference
   
2. **Manual labeling** (`AttributionMethod.MANUAL_LABEL`)
   - Clinician explicitly labels patient's initial utterance
   - Human-in-the-loop for ground truth
   
3. **Channel assignment** (`AttributionMethod.CHANNEL_ASSIGNMENT`)
   - Patient assigned to specific microphone channel
   - Multi-channel recording setup
   
4. **Embedding matching** (`AttributionMethod.EMBEDDING_MATCH`)
   - Match against pre-existing speaker template
   - Requires patient voice template from previous session
   
5. **Post-hoc correction** (`AttributionMethod.POST_HOC_CORRECTION`)
   - Human correction after initial diarization
   - Gold standard for dataset creation

### Attribution States

The system maintains **first-class states** for attribution:

- `PATIENT` — Attributed to patient with confidence ≥ threshold
- `NON_PATIENT` — Attributed to non-patient speaker
- `UNKNOWN` — Cannot determine speaker
- `OVERLAP` — Multiple speakers detected
- `LOW_CONFIDENCE` — Attribution confidence below threshold

**Never** silently assign ambiguous speech to patient.

### SpeakerAttribution Schema

```python
@dataclass
class SpeakerAttribution:
    speaker_id: str                    # Pseudonymous
    patient_probability: float         # 0.0-1.0
    status: AttributionStatus          # Enum
    attribution_method: AttributionMethod
    overlap: bool
    provisional: bool                  # Streaming vs batch
    embedding_distance: Optional[float]
    confidence_factors: dict[str, float]
```

### Privacy: Biometric Data

Voice embeddings constitute **biometric data under GDPR** (UK ICO guidance). This requires:
- Legal basis for processing
- Data protection impact assessment (DPIA)
- Enhanced technical and organizational measures
- Clear purpose limitation
- Explicit consent or appropriate legal basis

## Windowing Specifications

Each feature extractor has **explicit windowing contracts**:

### eGeMAPSv02 (openSMILE)

```yaml
egemaps:
  window_seconds: 2.0
  hop_seconds: 0.5
  minimum_voiced_ratio: 0.40
  feature_set: eGeMAPSv02
  feature_level: Functionals  # 88 features
```

**Edge cases**:
- Speech shorter than window → pad or skip (configurable)
- Window crosses speakers → mark as OVERLAP
- Insufficient voiced content → emit None
- Clipping detected → quality_status = CLIPPED

### YAMNet

```yaml
yamnet:
  native_window_seconds: 0.96
  native_hop_seconds: 0.48
  aggregation: max  # or mean
```

**Edge cases**:
- Model returns no valid embedding → emit None
- Multiple events detected → return dict with all probabilities
- Event overlaps with speaker boundary → attribute probabilistically

### emotion2vec

```yaml
emotion2vec:
  window_seconds: 3.0
  hop_seconds: 1.0
  pooling: mean
```

**Edge cases**:
- Window crosses ring buffer boundary → handle seamlessly or mark boundary
- Model timeout → emit DropReason, degrade to Level 2
- Insufficient audio → emit None

## Performance Metrics

### VAD Performance

**Not** a single "VAD accuracy >95%". Instead:

| Metric | Target | Condition |
|--------|--------|-----------|
| Speech miss rate | <5% | SNR ≥10 dB, normal speech |
| False alarm rate | <5% | SNR ≥10 dB |
| Segment F1 | >0.90 | SNR ≥10 dB |
| Onset latency | <50 ms | Preliminary decision |
| Offset latency | 250-500 ms | Stable endpoint |
| Cry/scream recall | >0.80 | Non-speech vocalizations |
| Whisper detection | >0.70 | Low-amplitude speech |

### Diarization Performance

**Not** "DER <10%" universally. Instead:

```
DER < 10% on [Specific Clinical Dataset]
  - 250 ms collar
  - Overlap included
  - Known patient count: 2-3 speakers
  - SNR ≥ 10 dB
```

**Patient-specific metrics** (more important than generic DER):

| Metric | Target |
|--------|--------|
| Patient speech precision | >0.90 |
| Patient speech recall | >0.85 |
| False patient attribution rate | <0.10 |
| Unknown/abstention rate | <0.15 |
| Overlap attribution performance | >0.70 F1 |

### Feature Reproducibility

- **Bitwise reproducible**: Same runtime, hardware, config
- **Numerically reproducible**: Within tolerance across platforms
  - eGeMAPSv02: ±1e-6 relative error
  - emotion2vec: ±1e-5 relative error
  - YAMNet: ±1e-5 relative error

**Record for reproducibility**:
- Package versions (pinned)
- Model file SHA256 hashes
- Configuration hash
- Resampler implementation
- CPU/GPU type and driver
- Precision mode (float32/float16)
- Random seeds
- Git commit hash

## Output Schemas

### AcousticFeatureRecord

Primary output for `acoustic_features.parquet`:

```python
@dataclass
class AcousticFeatureRecord:
    # Session (pseudonymous)
    session_id: str
    stream_id: str
    
    # Temporal
    window_start_ms: int
    window_end_ms: int
    source_start_sample: int
    source_end_sample: int
    
    # Speaker
    speaker_id: Optional[str]
    patient_probability: Optional[float]
    attribution_status: str
    attribution_method: Optional[str]
    
    # VAD
    vad_probability_mean: float
    voiced_ratio: float
    overlap_probability: float
    
    # Features
    egemaps: Optional[list[float]]  # 88 functionals
    yamnet_event_scores: dict[str, float]
    emotion_embedding: Optional[list[float]]
    
    # Quality
    snr_db: Optional[float]
    clipping_ratio: float
    dropout_ratio: float
    quality_status: str
    
    # Reproducibility
    extractor_versions: dict[str, str]
    config_hash: str
    model_hashes: dict[str, str]
```

### DropReason

For windows that were dropped or degraded:

```python
@dataclass
class DropReason:
    window_start_ms: int
    window_end_ms: int
    reason: str  # DROPPED_BACKPRESSURE | MODEL_TIMEOUT | INSUFFICIENT_AUDIO | ...
    degradation_level: int  # 0-5
    timestamp_ms: int
    additional_info: dict
```

## Emotion Analysis: Framing

**Avoid** calling the output "emotion". Use:
- `acoustic_affect_embedding`
- `vocal_arousal_score`
- `vocal_valence_score`
- `distress_event_probability`

**Important**: These models estimate patterns in **vocal expression**, not internal emotional state.

Affected by:
- Language and culture
- Physical illness and pain
- Medication effects
- Neurological differences
- Age and vocal characteristics
- Recording conditions

### Prohibited Uses

- Autonomous diagnosis
- Suicide risk decisions without human review
- Treatment denial
- Coercive monitoring
- Staff or patient disciplinary actions
- Immigration or legal proceedings
- Employment decisions

## Augmentation: Training Only

**Augmentation NEVER runs during inference.**

Augmentation is used ONLY in:
- Training dataset generation
- Robustness testing
- Controlled evaluation
- Simulation notebooks

Augmentation types:
- Room noise injection
- Codec simulation (telephony artifacts)
- SNR variation
- Clipping simulation

**Safety**: Separate entry points (`StreamProcessor` vs `DatasetBuilder`) make accidental inference-time augmentation structurally difficult.

## Testing Strategy

### Unit Tests
- Individual components (VAD, diarization, features, quality)
- Schema validation
- Edge cases (short segments, overlap, clipping)

### Integration Tests
- Streaming pipeline end-to-end
- Offline pipeline end-to-end
- Training pipeline end-to-end

### Performance Tests
- Latency benchmarks (component-level)
- Real-time capability (throughput, sustained load)
- Graceful degradation (all 6 levels)
- Memory profiling
- CPU/GPU utilization

### End-to-End Tests
- Real clinical audio (de-identified)
- Various acoustic conditions
- Multiple speakers and overlap
- Distress events
- Enrollment scenarios

## Compliance & Privacy

### Technical Controls

- Pseudonymous identifiers
- In-memory audio processing
- Encrypted temporary files
- Access audit logging
- Data retention policies

### Organizational Requirements

**Important**: Compliance depends on:
- Complete deployment environment
- Organizational controls
- Applicable jurisdiction
- Validated operating procedures

**Do not claim** "HIPAA compliant" or "GDPR compliant" based on implementation alone.

### Safer Wording

> "Designed to support privacy-preserving deployments. Regulatory compliance depends on the complete deployment environment, organizational controls, applicable jurisdiction, and validated operating procedures."

## Recommended Implementation Order

1. **Core infrastructure** (capture, preprocessing, schemas)
2. **Segmentation & quality** (VAD, quality monitoring)
3. **Event detection** (YAMNet, full-stream)
4. **Diarization & attribution** (enrollment, provisional/batch)
5. **Speech features** (eGeMAPS, emotion2vec)
6. **Training pipeline** (augmentation, LightGBM)
7. **Runtime system** (degradation, streaming, offline)
8. **Testing & validation**
9. **Privacy & compliance**
10. **Documentation & deployment**

See `TASKS.md` for detailed 16-week timeline.

## Dependencies

### Core
- Python 3.9+
- numpy, scipy, pandas, pyarrow
- sounddevice, soundfile, librosa

### Models
- torch, torchaudio
- Silero VAD (GitHub)
- Pyannote Audio (requires HuggingFace token)
- TensorFlow + TensorFlow Hub (YAMNet)
- emotion2vec (GitHub)
- openSMILE (eGeMAPS)
- LightGBM

### External Accounts
- HuggingFace account (for Pyannote)
- Pyannote commercial license (optional, for streaming diarization <300ms)

## Success Criteria

1. ✅ Component-level latency targets met
2. ✅ Real-time processing at ≥1.0x speed
3. ✅ VAD and diarization metrics on defined dataset
4. ✅ Graceful degradation works under load
5. ✅ No PHI in outputs; biometric data handled per GDPR
6. ✅ Numerical reproducibility across runs
7. ✅ Complete model cards and documentation
8. ✅ >80% code coverage

## References

- Silero VAD: https://github.com/snakers4/silero-vad
- Pyannote: https://github.com/pyannote/pyannote-audio
- YAMNet: https://blog.tensorflow.org/2021/03/
- emotion2vec: https://github.com/ddlBoJack/emotion2vec
- eGeMAPS: Eyben et al., 2016
- HIPAA: https://www.hhs.gov/hipaa
- GDPR (biometric data): https://ico.org.uk/for-organisations/

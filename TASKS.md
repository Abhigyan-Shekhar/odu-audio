# Person 1: Audio Pipeline & Acoustic Modeling Tasks

## Deliverables

### Primary Outputs
- `audio.wav` → `acoustic_features.parquet`
- Speaker-specific speech windows with acoustic features
- Privacy-preserving raw-audio pipeline
- Data manifest format

### Secondary Outputs
- `dropped_records.parquet` (windows dropped due to quality/attribution issues)
- Performance benchmarks (latency, throughput, real-time capability)
- Model cards for each component
- Privacy impact assessment documentation

---

## Phase 1: Core Infrastructure (Week 1-2)

### 1.1 Audio Capture & Ring Buffer
- [ ] Implement ring buffer with configurable size
- [ ] Microphone capture with sounddevice
- [ ] Handle device selection and channel configuration
- [ ] <20 ms capture-to-buffer latency
- [ ] Unit tests for buffer overflow, underflow
- [ ] Benchmark actual latency on target hardware

### 1.2 16 kHz Conversion & Preprocessing
- [ ] Implement resampling with anti-aliasing (librosa/resampy)
- [ ] DC offset removal
- [ ] Normalization
- [ ] <5 ms compute time per 30ms chunk
- [ ] Verify bitwise reproducibility
- [ ] Handle arbitrary input sample rates

### 1.3 Schemas & Data Contracts
- [ ] Finalize `AcousticFeatureRecord` dataclass
- [ ] Finalize `SpeakerAttribution` dataclass
- [ ] Define windowing contracts for each extractor
- [ ] Parquet schema validation
- [ ] Handle edge cases (short segments, overlap, cross-speaker windows)

---

## Phase 2: Segmentation & Quality (Week 2-3)

### 2.1 Silero VAD Integration
- [ ] Install and load Silero VAD v4
- [ ] Implement streaming VAD with ring buffer
- [ ] Preliminary decision: <50 ms
- [ ] Stable endpoint: 250-500 ms (configurable)
- [ ] Measure speech miss rate, false alarm rate, segment F1
- [ ] Test performance by SNR (10 dB, 20 dB, clean)
- [ ] Test on non-speech vocalizations (cry, scream, whisper)

### 2.2 Quality Monitoring (Parallel, Full-Stream)
- [ ] Clipping detection (ratio of samples at ±1.0)
- [ ] Dropout detection (zero-sample runs)
- [ ] SNR estimation (speech/noise ratio)
- [ ] Device/channel diagnostics
- [ ] <10 ms compute time target
- [ ] Quality status classification logic
- [ ] Unit tests for edge cases

---

## Phase 3: Event Detection (Week 3-4)

### 3.1 YAMNet Acoustic Event Detection
- [ ] Install YAMNet from TensorFlow Hub
- [ ] Implement full-stream processing (NOT VAD-filtered)
- [ ] Handle 0.96s window, 0.48s hop
- [ ] Extract scream, yell, cry, gasp, groan probabilities
- [ ] Temporal alignment with diarization for attribution
- [ ] ~1.0-1.5s availability latency
- [ ] Test on events outside speech VAD boundaries
- [ ] False positive analysis on non-distress speech

---

## Phase 4: Speaker Diarization & Attribution (Week 4-6)

### 4.1 Pyannote Diarization
- [ ] Set up HuggingFace token and model access
- [ ] Implement batch diarization mode
- [ ] Implement provisional streaming mode (if using commercial model)
- [ ] Speaker embedding extraction
- [ ] Measure DER on defined dataset (250ms collar, overlap included)
- [ ] Measure patient-specific metrics (precision, recall, false attribution)

### 4.2 Patient Enrollment
- [ ] Implement session-start enrollment
- [ ] Implement manual labeling interface
- [ ] Implement channel assignment method
- [ ] Implement embedding template matching
- [ ] Enrollment validation and confidence scoring
- [ ] Fallback handling (enrollment failure → manual)

### 4.3 Speaker Attribution Logic
- [ ] Implement `SpeakerAttribution` output
- [ ] Handle PATIENT | NON_PATIENT | UNKNOWN | OVERLAP | LOW_CONFIDENCE states
- [ ] Probabilistic attribution with confidence
- [ ] Temporal overlap resolution
- [ ] Attribution of YAMNet events to patient
- [ ] Privacy: biometric data handling (GDPR compliance)

---

## Phase 5: Speech Acoustic Features (Week 6-8)

### 5.1 openSMILE eGeMAPSv02
- [ ] Install openSMILE 2.5+
- [ ] Configure eGeMAPSv02 functionals extraction
- [ ] Windowing: 2.0s window, 0.5s hop, 0.40 min voiced ratio
- [ ] Validate 88-dimensional output
- [ ] Handle short segments, insufficient voiced content
- [ ] Handle cross-speaker windows
- [ ] Numerical reproducibility testing (±1e-6 tolerance)

### 5.2 emotion2vec Embeddings
- [ ] Install emotion2vec from GitHub
- [ ] Configure 3.0s window, 1.0s hop, mean pooling
- [ ] Extract affect embeddings
- [ ] Handle window edge cases
- [ ] Model card: acoustic affect (NOT emotion classification)
- [ ] Document cultural, linguistic, medical confounds
- [ ] Prohibited uses documentation

---

## Phase 6: Training Pipeline (Week 8-10)

### 6.1 Room Noise Augmentation
- [ ] Collect or generate room noise samples
- [ ] Implement noise injection (realistic SNR ranges)
- [ ] Codec simulation (telephony artifacts)
- [ ] SNR variation augmentation
- [ ] Clipping simulation
- [ ] **Ensure never runs during inference**
- [ ] Unit tests to prevent inference-time augmentation

### 6.2 LightGBM Acoustic Baseline
- [ ] Dataset splitting (train/val/test)
- [ ] Feature selection and engineering
- [ ] LightGBM training pipeline
- [ ] Hyperparameter tuning
- [ ] Probability calibration
- [ ] Evaluation metrics (AUC, calibration curves)
- [ ] Feature importance analysis
- [ ] Model card generation
- [ ] Output: `acoustic_predictions.parquet` with model version, confidence, abstention reason

---

## Phase 7: Runtime System (Week 10-12)

### 7.1 Graceful Degradation
- [ ] Implement `DegradationManager`
- [ ] Define 6 degradation levels (0-5)
- [ ] Latency monitoring (P95)
- [ ] Component activation/deactivation logic
- [ ] Deterministic degradation triggers
- [ ] Recovery logic with hysteresis
- [ ] Never silently drop windows: emit `DropReason` records
- [ ] Integration tests for each degradation level

### 7.2 Streaming Pipeline Integration
- [ ] Implement `StreamProcessor`
- [ ] Task scheduling and parallel execution
- [ ] Backpressure handling
- [ ] Quality → VAD → Diarization → Features pipeline
- [ ] Event detection (YAMNet) in parallel to VAD
- [ ] Real-time buffer management
- [ ] Timestamp alignment across components
- [ ] End-to-end integration tests

### 7.3 Offline Dataset Pipeline
- [ ] Implement `DatasetBuilder`
- [ ] Batch processing mode
- [ ] Parallel feature extraction
- [ ] Progress tracking and resumability
- [ ] Output: `dataset_features.parquet`

---

## Phase 8: Testing & Validation (Week 12-14)

### 8.1 Latency Tests
- [ ] Audio capture-to-buffer: <20 ms
- [ ] Resampling compute: <5 ms per chunk
- [ ] VAD inference: <5 ms per chunk
- [ ] Quality monitoring: <10 ms
- [ ] Preliminary speech activity: <50 ms
- [ ] Stable speech endpoint: 250-500 ms
- [ ] Provisional speaker attribution: <500 ms
- [ ] YAMNet availability: ~1.0-1.5s
- [ ] emotion2vec availability: 1-3s
- [ ] Benchmark on target hardware (CPU vs GPU)

### 8.2 Real-Time Processing Tests
- [ ] Process audio at ≥1.0x real-time speed
- [ ] Multi-stream concurrent processing
- [ ] Memory usage profiling
- [ ] CPU/GPU utilization
- [ ] Sustained load testing (hours)

### 8.3 Accuracy & Performance Metrics
- [ ] VAD: speech miss rate, false alarm rate, segment F1, onset/offset latency
- [ ] VAD by condition: SNR, cry/scream, whisper
- [ ] Diarization: DER on defined dataset
- [ ] Patient-specific: precision, recall, false attribution, unknown rate
- [ ] Feature reproducibility: bitwise and numerical
- [ ] YAMNet: precision/recall on distress events
- [ ] emotion2vec: embedding stability

### 8.4 Edge Cases & Robustness
- [ ] Short speech segments (<1s)
- [ ] Speaker overlap
- [ ] Cross-speaker windows
- [ ] Clipping and low SNR
- [ ] Dropout and silence
- [ ] Ring buffer boundary effects
- [ ] Enrollment failures
- [ ] Model failures and timeouts

---

## Phase 9: Privacy & Compliance (Week 14-15)

### 9.1 Privacy-Preserving Pipeline
- [ ] Pseudonymous ID generation
- [ ] Identity mapping system (separate from features)
- [ ] In-memory audio processing (no raw audio to disk)
- [ ] Encrypted temporary files
- [ ] Secure deletion of buffers
- [ ] Access audit logging

### 9.2 Data Manifest Format
- [ ] Define manifest schema
- [ ] Track all operations on audio
- [ ] Record extractor versions, model hashes, config hashes
- [ ] Provenance tracking
- [ ] Retention policy implementation

### 9.3 Compliance Documentation
- [ ] Privacy impact assessment
- [ ] GDPR biometric data handling (speaker embeddings)
- [ ] HIPAA safeguards documentation
- [ ] Data retention and deletion procedures
- [ ] Incident response plan
- [ ] Model cards for all components
- [ ] Prohibited uses documentation

---

## Phase 10: Documentation & Deployment (Week 15-16)

### 10.1 Documentation
- [ ] API documentation (docstrings)
- [ ] Configuration guide
- [ ] Deployment guide
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Model cards (silero, pyannote, yamnet, emotion2vec, lightgbm)
- [ ] Data card (dataset specification)

### 10.2 Example Notebooks
- [ ] 01_audio_exploration.ipynb
- [ ] 02_vad_tuning.ipynb
- [ ] 03_diarization_analysis.ipynb
- [ ] 04_feature_analysis.ipynb
- [ ] 05_model_training.ipynb

### 10.3 CLI & Deployment
- [ ] CLI for streaming mode
- [ ] CLI for offline batch mode
- [ ] Docker containerization
- [ ] Environment validation script
- [ ] Hardware requirements documentation
- [ ] Production deployment checklist

---

## Testing Checklist

### Unit Tests
- [ ] `test_capture.py`
- [ ] `test_vad.py`
- [ ] `test_diarization.py`
- [ ] `test_attribution.py`
- [ ] `test_features.py` (egemaps, yamnet, emotion2vec)
- [ ] `test_quality.py`
- [ ] `test_schemas.py`

### Integration Tests
- [ ] `test_streaming_pipeline.py`
- [ ] `test_offline_pipeline.py`
- [ ] `test_training_pipeline.py`

### Performance Tests
- [ ] `test_latency.py` (component-level and end-to-end)
- [ ] `test_realtime.py` (throughput, sustained load)
- [ ] `test_degradation.py` (graceful degradation levels)

### End-to-End Tests
- [ ] Real clinical audio samples (de-identified)
- [ ] Various acoustic conditions (clean, noisy, clipped)
- [ ] Multiple speakers and overlap
- [ ] Distress events (scream, cry)
- [ ] Enrollment scenarios

---

## Dependencies & External Accounts

### Required
- [ ] Python 3.9+ environment
- [ ] HuggingFace account & token (for pyannote)
- [ ] Accept pyannote user agreements
- [ ] GPU access (recommended for diarization & emotion2vec)

### Optional
- [ ] Pyannote commercial streaming license (for <300ms diarization)
- [ ] Clinical test dataset access
- [ ] Target deployment hardware for benchmarking

---

## Success Criteria

1. **Latency**: Component-level latency targets met on target hardware
2. **Real-time**: Processes audio at ≥1.0x real-time speed
3. **Accuracy**: VAD and diarization metrics meet targets on defined dataset
4. **Robustness**: Graceful degradation works under load
5. **Privacy**: No PHI in output; biometric data handled per GDPR
6. **Reproducibility**: Features are numerically reproducible across runs
7. **Documentation**: Complete model cards, data cards, and deployment guide
8. **Testing**: >80% code coverage, all edge cases handled

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Pyannote streaming model unavailable | High | Use batch diarization with provisional mode |
| GPU unavailable in deployment | Medium | Optimize CPU inference; degrade to minimal features |
| Latency targets not met | High | Implement degradation; profile and optimize bottlenecks |
| emotion2vec installation issues | Low | Make emotion2vec optional; document manual installation |
| Clinical test data unavailable | Medium | Use public datasets for baseline; document limitations |
| GDPR compliance complexity | High | Early legal review; conservative data handling |

---

## Timeline Summary

- **Week 1-2**: Core infrastructure (capture, preprocessing, schemas)
- **Week 2-3**: Segmentation & quality (VAD, quality monitoring)
- **Week 3-4**: Event detection (YAMNet)
- **Week 4-6**: Diarization & attribution
- **Week 6-8**: Speech features (eGeMAPS, emotion2vec)
- **Week 8-10**: Training pipeline (augmentation, LightGBM)
- **Week 10-12**: Runtime system (degradation, streaming, offline)
- **Week 12-14**: Testing & validation
- **Week 14-15**: Privacy & compliance
- **Week 15-16**: Documentation & deployment

**Total: 16 weeks** (4 months)

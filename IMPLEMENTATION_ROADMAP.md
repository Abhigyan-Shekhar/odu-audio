# Implementation Roadmap

## Vertical-Slice Approach: From Architecture to Working Pipeline

This roadmap replaces the linear 16-week plan with a vertical-slice strategy that delivers testable milestones incrementally.

**Philosophy**: Build one end-to-end path first, validate it thoroughly, then add components one at a time with isolated testing.

---

## Milestone 0: Engineering Baseline (Week 1)

**Goal**: Establish engineering infrastructure and test existing scaffold.

### 1. Freeze the Contracts

Define and review all data schemas:

#### Required Schemas
- [x] `AcousticFeatureRecord` (exists, needs review)
- [x] `SpeakerAttribution` (exists, needs review)
- [ ] `AudioFrame` (create)
- [ ] `SpeechSegment` (create)
- [ ] `ErrorEvent` (create)
- [ ] `DegradationEvent` (create)

#### Contract Rules to Document
- [ ] Timestamp format (monotonic, UTC, session-relative?)
- [ ] Sample-index origin (0-indexed, signed/unsigned)
- [ ] Session and stream ID format (UUID? pseudonymous?)
- [ ] Discontinuity marking protocol
- [ ] Sequence number handling
- [ ] Missing-value semantics (None vs 0.0 vs NaN)

**Exit Criterion**: Every module can be implemented against stable input/output schemas documented in `CONTRACTS.md`. Contracts approved as v0.1-provisional; stabilize at v1.0 after Milestone 1 validates them in practice.

---

### 2. Add Engineering Infrastructure

Set up modern Python tooling:

#### Project Configuration
```bash
# Create pyproject.toml with:
# - Build system (setuptools or poetry)
# - Dependency management with version locking
# - Tool configurations (black, ruff, mypy, pytest)

# Create:
pyproject.toml
poetry.lock (or requirements-lock.txt)
.pre-commit-config.yaml
```

#### Code Quality
- [ ] **Formatting**: black (automated)
- [ ] **Linting**: ruff (replace flake8/pylint)
- [ ] **Type checking**: mypy with strict mode
- [ ] **Import sorting**: isort

#### Testing Infrastructure
- [ ] pytest with coverage
- [ ] pytest-cov for coverage reporting
- [ ] pytest-timeout for hanging tests
- [ ] pytest-benchmark for performance tests
- [ ] Fixtures for test data (sample audio with various properties)
- [ ] Synthetic audio generation utilities
- [ ] Licensed test audio inventory (properly attributed)

#### CI Pipeline
```yaml
# .github/workflows/ci.yml or .gitlab-ci.yml
# - Run linting (black, ruff)
# - Run type checking (mypy)
# - Run tests with coverage
# - Generate coverage report (≥80% line coverage)
# - Test critical contract behaviors (see exit criterion)
# - Fail if any critical test fails
```

#### Structured Logging
```python
# Use structlog or Python logging with JSON formatter
# Define log levels for:
# - Component lifecycle events
# - Processing metrics
# - Degradation events
# - Error conditions
```

#### Configuration Validation
```python
# Use pydantic to validate YAML configs
# Fail fast on startup if config is invalid
```

#### Additional Infrastructure
- [ ] Dependency and model-license inventory
- [ ] Supported Python versions (document and test)
- [ ] Supported operating systems (document and test)
- [ ] Architecture decision records (ADRs)
- [ ] Configuration schema validation (pydantic models)
- [ ] Logging redaction rules (prevent PHI leakage)
- [ ] Model artifact and hash conventions
- [ ] Pre-commit hooks (.pre-commit-config.yaml)
- [ ] CI matrix definition (Python versions, OS variants)

**Exit Criterion**: 
- CI passes
- **Critical behavioral tests implemented and passing**:
  - Schema validation and serialization
  - Timestamp and sample-index continuity
  - Degradation transitions and recovery
  - Invalid configuration rejection
  - Parquet round-trip compatibility
- Overall line coverage ≥80%
- **Evidence bundle produced**:
  - `artifacts/test-report.xml`
  - `artifacts/coverage.xml`
  - `artifacts/dependency-lockfile`
  - `artifacts/configuration-snapshot.yaml`
  - `artifacts/checksums.txt`
  - `artifacts/milestone-0-acceptance.md`

**Milestone Owner**: __________  
**Reviewer**: __________  
**Acceptance Authority**: Technical Lead  
**Accepted**: ☐ Yes ☐ No — Date: __________

---

## Milestone 1: Offline WAV-to-Parquet Backbone (Week 2-3)

**Goal**: Build the simplest possible end-to-end path without any ML models.

**Target**: First executable end-to-end WAV-to-Parquet vertical slice.

### What to Implement

```
WAV file
    ↓
[Decode audio]
    ↓
[Convert to mono, 16 kHz]
    ↓
[Fixed 2-second windows with 0.5s hop]
    ↓
[Basic quality metrics: clipping, dropout, RMS]
    ↓
[Create AcousticFeatureRecord with placeholders]
    ↓
Parquet file
```

### Components

#### 1. Audio I/O (`src/audio_pipeline/io/`)
```python
# audio_reader.py
class AudioReader:
    """Read and decode audio files."""
    def read(self, path: str) -> AudioData
    # Returns: samples, sample_rate, channels, duration
    
# audio_data.py
@dataclass
class AudioData:
    samples: np.ndarray  # shape: (n_samples,) or (n_channels, n_samples)
    sample_rate: int
    n_channels: int
    duration_seconds: float
```

#### 2. Preprocessing (`src/audio_pipeline/preprocessing/`)
```python
# resampler.py
class Resampler:
    """High-quality resampling with anti-aliasing."""
    def resample(self, audio: np.ndarray, 
                 source_sr: int, target_sr: int) -> np.ndarray
    
# channel_mixer.py
def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel to mono."""
```

#### 3. Windowing (`src/audio_pipeline/windowing/`)
```python
# fixed_windower.py
class FixedWindower:
    """Create fixed-size windows with hop."""
    def __init__(self, window_seconds: float, hop_seconds: float)
    def window(self, audio: AudioData) -> Iterator[AudioWindow]

@dataclass
class AudioWindow:
    samples: np.ndarray
    start_sample: int
    end_sample: int
    start_time_ms: int
    end_time_ms: int
    sequence_number: int
```

#### 4. Basic Quality Metrics (`src/audio_pipeline/quality/`)
```python
# clipping.py
def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Return ratio of samples at ±threshold."""

# dropout.py  
def detect_dropout(audio: np.ndarray, threshold: float = 1e-6) -> float:
    """Return ratio of near-zero samples."""

# rms.py
def compute_rms(audio: np.ndarray) -> float:
    """Return RMS level in dB."""
```

#### 5. Offline Pipeline (`src/audio_pipeline/offline/`)
```python
# wav_to_parquet.py
class WavToParquetConverter:
    """Convert WAV files to Parquet with quality metrics."""
    
    def process(self, wav_path: str, output_path: str) -> None:
        # 1. Read audio
        # 2. Resample to 16 kHz
        # 3. Convert to mono
        # 4. Create windows
        # 5. Compute quality metrics per window
        # 6. Create AcousticFeatureRecord (placeholders for models)
        # 7. Write to Parquet
```

### Tests Required

**Comprehensive acceptance-test specification** containing the following planned cases:

```python
# tests/integration/test_wav_to_parquet.py
# Status: NOT YET IMPLEMENTED

def test_wav_to_parquet_basic():
    """Test basic conversion."""
    
def test_sample_rate_8khz():
    """8 kHz input."""
    
def test_sample_rate_16khz():
    """16 kHz input (native)."""
    
def test_sample_rate_44_1khz():
    """44.1 kHz input (CD quality)."""
    
def test_sample_rate_48khz():
    """48 kHz input (professional)."""
    
def test_mono_input():
    """Mono audio."""
    
def test_stereo_input():
    """Stereo audio converted to mono."""
    
def test_pcm_int16():
    """PCM 16-bit integer format."""
    
def test_pcm_float32():
    """PCM 32-bit float format."""
    
def test_unsupported_codec():
    """Reject unsupported codec gracefully."""
    
def test_empty_audio():
    """Empty audio file."""
    
def test_truncated_file():
    """Truncated/incomplete WAV file."""
    
def test_corrupt_wav():
    """Malformed WAV file."""
    
def test_very_short_audio():
    """Audio shorter than one window."""
    
def test_very_long_audio():
    """Multi-hour audio file."""
    
def test_nan_prevention():
    """Ensure no NaN in output."""
    
def test_infinity_prevention():
    """Ensure no Inf in output."""
    
def test_clipped_audio():
    """Audio with clipping detected."""
    
def test_silent_audio():
    """Silent audio (near-zero samples)."""
    
def test_timestamp_continuity():
    """Verify monotonic timestamps across windows."""
    
def test_sample_index_continuity():
    """Verify monotonic sample indices."""
    
def test_sequence_number_continuity():
    """Verify sequence numbers increment by 1."""
    
def test_atomic_output_write():
    """Output written atomically or with tmp + rename."""
    
def test_existing_output_behavior():
    """Behavior when output file exists (overwrite? error?)."""
    
def test_parquet_schema_versioning():
    """Output includes schema_version field."""
    
def test_parquet_roundtrip():
    """Write and read back Parquet successfully."""
    
def test_deterministic_output_locked_env():
    """Same input produces bitwise identical output on locked environment."""
    
def test_deterministic_output_tolerance():
    """Same input produces numerically equivalent output (within tolerance) on different environments."""
    
def test_cancellation_cleanup():
    """Interrupted processing cleans up temporary files."""
    
def test_failure_cleanup():
    """Failed processing cleans up temporary files."""
```

**Exit Criterion**: 
- All acceptance tests **implemented and passing**
- Bitwise deterministic on the locked reference environment
- Numerically equivalent within documented tolerances (e.g., ±1e-6 for RMS, ±1 sample for timestamps) on supported environments (Python 3.10+, Linux/macOS/Windows)
- CLI tool successfully converts WAV files to valid, versioned Parquet
- **Evidence bundle produced**:
  - `artifacts/test-report.xml`
  - `artifacts/benchmark.json`
  - `artifacts/sample-output.parquet`
  - `artifacts/golden-fixtures/` (reference outputs)
  - `artifacts/schema-v1.0.json`
  - `artifacts/milestone-1-acceptance.md`

**Milestone Owner**: __________  
**Reviewers**: Data owner, ML owner, Technical lead  
**Acceptance Authority**: Technical Lead  
**Accepted**: ☐ Yes ☐ No — Date: __________

### CLI

```python
# src/audio_pipeline/cli.py

@click.command()
@click.argument('input_wav', type=click.Path(exists=True))
@click.argument('output_parquet', type=click.Path())
@click.option('--config', type=click.Path())
def wav_to_parquet(input_wav, output_parquet, config):
    """Convert WAV file to Parquet with quality metrics."""
    converter = WavToParquetConverter.from_config(config)
    converter.process(input_wav, output_parquet)
```

### Example Output

```python
# After running:
# audio-pipeline wav-to-parquet input.wav output.parquet

# output.parquet contains:
session_id: "session_001"
stream_id: "stream_001"
window_start_ms: 0, 500, 1000, ...
window_end_ms: 2000, 2500, 3000, ...
source_start_sample: 0, 8000, 16000, ...
source_end_sample: 32000, 40000, 48000, ...

# Placeholders (None):
speaker_id: None
patient_probability: None
egemaps: None
yamnet_event_scores: {}
emotion_embedding: None

# Populated:
clipping_ratio: 0.002, 0.001, 0.0, ...
dropout_ratio: 0.0, 0.0, 0.0, ...
rms_db: -18.3, -17.9, -22.1, ...
quality_status: "OK", "OK", "OK", ...

# Reproducibility:
extractor_versions: {"resampler": "librosa-0.10.0", ...}
config_hash: "sha256:abc123..."
```

**Exit Criterion**: A deterministic WAV-to-Parquet integration test passes for various audio conditions.

---

## Milestone 2: Streaming Infrastructure (Week 4-5)

**Goal**: Add real-time streaming without models.

### What to Implement

#### 1. Microphone Abstraction (`src/audio_pipeline/capture/`)
```python
# audio_source.py
class AudioSource(ABC):
    """Abstract audio source."""
    @abstractmethod
    def read_chunk(self) -> AudioChunk
    
# microphone_source.py
class MicrophoneSource(AudioSource):
    """Real microphone input."""
    
# file_source.py
class FileSource(AudioSource):
    """File input (for testing streaming with files)."""
```

#### 2. Ring Buffer (already planned)
```python
# ring_buffer.py
class RingBuffer:
    """Thread-safe ring buffer with monotonic sequence numbers."""
    def write(self, chunk: np.ndarray) -> int  # Returns sequence number
    def read(self, n_samples: int) -> Optional[np.ndarray]
    def mark_discontinuity(self, reason: str) -> None
```

#### 3. Chunk Sequencing
```python
# audio_chunk.py
@dataclass
class AudioChunk:
    samples: np.ndarray
    sequence_number: int
    capture_timestamp_ns: int  # From time.time_ns()
    sample_rate: int
    discontinuity: bool
    discontinuity_reason: Optional[str]
```

#### 4. Streaming Resampler
```python
# streaming_resampler.py
class StreamingResampler:
    """Stateful resampler for streaming audio."""
    def __init__(self, source_sr: int, target_sr: int)
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray
    def reset(self) -> None
```

#### 5. Queue Management
```python
# bounded_queue.py
class BoundedQueue:
    """Queue with backpressure monitoring."""
    def put(self, item: T, timeout: float) -> bool
    def get(self, timeout: float) -> Optional[T]
    def qsize(self) -> int
    def is_full(self) -> bool
```

#### 6. Stream Processor
```python
# stream_processor.py
class StreamProcessor:
    """Coordinate streaming components."""
    
    def start(self) -> None:
        # Start capture thread
        # Start processing thread
        # Start output thread
        
    def stop(self) -> None:
        # Graceful shutdown
        # Flush buffers
        # Close files
        
    def stream(self) -> Iterator[AcousticFeatureRecord]:
        # Yield records as they're produced
```

### Tests Required

```python
# tests/integration/test_streaming.py

def test_synthetic_audio_stream():
    """Stream synthetic audio (sine wave)."""
    
def test_sequence_numbers():
    """Verify monotonic sequence numbers."""
    
def test_timestamp_continuity():
    """Verify timestamp continuity across chunks."""
    
def test_discontinuity_marking():
    """Verify discontinuity events are marked."""
    
def test_backpressure():
    """Slow consumer causes backpressure, not crash."""
    
def test_bounded_memory():
    """Long-running stream has bounded memory."""
    
def test_graceful_shutdown():
    """Shutdown doesn't lose data or deadlock."""
    
def test_file_as_stream():
    """Process file through streaming path."""
```

**Exit Criterion**: A long-running stream (>1 hour) preserves ordering, timestamps, and bounded memory usage with synthetic audio. Deterministic within defined tolerances.

---

## Milestone 3: VAD Integration (Week 6)

**Goal**: Add Silero VAD with replaceable interface.

### What to Implement

#### 1. VAD Interface (`src/audio_pipeline/segmentation/`)
```python
# vad_interface.py
class VADInterface(ABC):
    """Abstract VAD for testing and swapping."""
    @abstractmethod
    def process_chunk(self, audio: np.ndarray) -> float:
        """Return speech probability 0.0-1.0."""
        
# silero_vad.py
class SileroVAD(VADInterface):
    """Silero VAD implementation."""
    
# dummy_vad.py (for testing)
class DummyVAD(VADInterface):
    """Always returns fixed probability."""
```

#### 2. Endpointing
```python
# endpointer.py
class Endpointer:
    """Convert VAD probabilities to speech segments."""
    def __init__(self, 
                 threshold: float,
                 min_speech_duration_ms: int,
                 min_silence_duration_ms: int,
                 speech_pad_ms: int)
    
    def process(self, prob: float, timestamp_ms: int) -> Optional[Segment]
    
@dataclass
class Segment:
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int
    provisional: bool  # True until silence confirmed
```

### Metrics to Measure

Create benchmarking infrastructure:

```python
# benchmarks/vad_benchmark.py

def benchmark_vad_inference_time():
    """Measure inference time per chunk."""
    # Target: <5 ms on CPU
    
def benchmark_vad_onset_delay():
    """Measure speech onset detection delay."""
    # Target: <50 ms
    
def benchmark_vad_offset_delay():
    """Measure speech offset detection delay."""
    # Target: 250-500 ms
    
def measure_vad_false_alarms(dataset):
    """Measure false alarm rate by SNR."""
    
def measure_vad_miss_rate(dataset):
    """Measure miss rate by SNR and vocal type."""
```

### Tests Required

```python
# tests/unit/test_vad.py

def test_vad_interface():
    """Test VAD interface contract."""
    
def test_silero_loads():
    """Silero model loads successfully."""
    
def test_vad_probabilities():
    """VAD returns probabilities in [0, 1]."""
    
# tests/integration/test_vad_offline.py

def test_vad_on_clean_speech():
def test_vad_on_silence():
def test_vad_on_noisy_speech():
def test_vad_on_cry():
def test_vad_on_scream():

# tests/integration/test_vad_streaming.py

def test_vad_in_streaming_pipeline():
    """VAD works in streaming mode."""
```

**Exit Criterion**: VAD works in both offline and streaming paths with documented thresholds and measured latencies.

---

## Milestone 4: First Feature Extractor — openSMILE eGeMAPS (Week 7)

**Goal**: Add one real feature extractor with full error handling.

### Implementation Pattern

Every feature extractor should follow this pattern:

```python
# features/feature_extractor.py
class FeatureExtractor(ABC):
    """Abstract feature extractor."""
    
    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract features. Returns None on failure."""
    
    @abstractmethod
    def get_version(self) -> str:
        """Return version string."""
    
    @abstractmethod
    def get_model_hash(self) -> Optional[str]:
        """Return model file hash if applicable."""

# features/opensmile_extractor.py
class OpenSmileExtractor(FeatureExtractor):
    def __init__(self, 
                 feature_set: str = "eGeMAPSv02",
                 timeout_seconds: float = 5.0):
        self.feature_set = feature_set
        self.timeout = timeout_seconds
    
    def extract(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        try:
            with timeout(self.timeout):
                features = opensmile.process(audio, sr)
                if features is None or len(features) != 88:
                    logger.warning("openSMILE returned invalid output")
                    return None
                return features
        except TimeoutError:
            logger.error("openSMILE timed out")
            return None
        except Exception as e:
            logger.error(f"openSMILE failed: {e}")
            return None
```

### Configuration

```yaml
# configs/extractors.yaml
egemaps:
  enabled: true
  timeout_seconds: 5.0
  window_seconds: 2.0
  hop_seconds: 0.5
  min_voiced_ratio: 0.40
  on_failure: "emit_none"  # emit_none | skip_window | fail_pipeline
```

### Tests Required

```python
# tests/unit/test_opensmile.py

def test_opensmile_basic():
    """Extract features from clean speech."""
    
def test_opensmile_short_audio():
    """Handle audio shorter than window."""
    
def test_opensmile_silent_audio():
    """Handle silent audio."""
    
def test_opensmile_timeout():
    """Handle timeout gracefully."""
    
# tests/performance/test_opensmile_perf.py

def benchmark_opensmile_cpu():
    """Measure CPU time."""
    
def benchmark_opensmile_memory():
    """Measure memory usage."""
```

**Exit Criterion**: openSMILE extraction works in pipeline, handles failures gracefully, and has timeout protection.

---

## Milestone 5: YAMNet on Full Stream (Week 8)

**Goal**: Add YAMNet event detection on full audio stream (NOT VAD-filtered).

### Critical Architecture Point

YAMNet runs **in parallel** to VAD, not after it.

```python
# runtime/parallel_processor.py
class ParallelProcessor:
    """Run multiple processors on same audio."""
    
    def process(self, audio: AudioData) -> dict[str, Any]:
        results = {}
        
        # These run concurrently or sequentially:
        with ThreadPoolExecutor() as executor:
            quality_future = executor.submit(quality_monitor.process, audio)
            yamnet_future = executor.submit(yamnet_detector.process, audio)
            vad_future = executor.submit(vad.process, audio)
            
            results['quality'] = quality_future.result()
            results['yamnet'] = yamnet_future.result()
            results['vad'] = vad_future.result()
        
        return results
```

### YAMNet Implementation

```python
# features/yamnet_detector.py
class YAMNetDetector(FeatureExtractor):
    def __init__(self, 
                 target_events: list[str],
                 min_probability: float = 0.1,
                 timeout_seconds: float = 10.0):
        self.target_events = target_events
        self.min_probability = min_probability
        self.timeout = timeout_seconds
        
    def extract(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Return {event_name: max_probability}."""
        try:
            with timeout(self.timeout):
                scores = yamnet.predict(audio)
                return self._filter_target_events(scores)
        except TimeoutError:
            logger.error("YAMNet timed out")
            return {}
        except Exception as e:
            logger.error(f"YAMNet failed: {e}")
            return {}
```

### Tests Required

```python
# tests/integration/test_yamnet.py

def test_yamnet_on_scream():
    """Detect scream event."""
    
def test_yamnet_on_normal_speech():
    """Normal speech has low distress scores."""
    
def test_yamnet_parallel_to_vad():
    """YAMNet runs even when VAD says no speech."""
```

**Exit Criterion**: YAMNet detects distress events on full stream, including events outside VAD speech boundaries.

---

## Milestone 6: Speaker Handling (Week 9-10)

**Goal**: Add diarization and patient attribution with explicit uncertainty.

### Implementation Order

1. **Diarization** (batch mode first)
2. **Overlap detection**
3. **Enrollment** (session-start method first)
4. **Patient attribution**
5. **Confidence and UNKNOWN handling**

### Key Implementation

```python
# speakers/attribution.py
class SpeakerAttributor:
    """Attribute speech segments to patient."""
    
    def attribute(self, 
                  segment: SpeechSegment,
                  diarization: Diarization) -> SpeakerAttribution:
        
        # Check for overlap
        if self._has_overlap(segment, diarization):
            return SpeakerAttribution.overlap_detected(
                speaker_ids=self._get_overlapping_speakers(segment)
            )
        
        # Get speaker for this segment
        speaker_id = self._get_speaker(segment, diarization)
        
        if speaker_id is None:
            return SpeakerAttribution.unknown(reason="no_speaker_detected")
        
        # Compare to patient enrollment
        confidence = self._match_patient(speaker_id)
        
        if confidence < self.confidence_threshold:
            return SpeakerAttribution(
                speaker_id=speaker_id,
                patient_probability=confidence,
                status=AttributionStatus.LOW_CONFIDENCE,
                attribution_method=self.method,
                overlap=False
            )
        
        # High confidence attribution
        status = (AttributionStatus.PATIENT if confidence >= 0.5 
                  else AttributionStatus.NON_PATIENT)
        
        return SpeakerAttribution(
            speaker_id=speaker_id,
            patient_probability=confidence,
            status=status,
            attribution_method=self.method,
            overlap=False
        )
```

### Tests Required

```python
# tests/unit/test_attribution.py

def test_attribution_high_confidence():
def test_attribution_low_confidence():
def test_attribution_overlap():
def test_attribution_unknown():
def test_attribution_non_patient():

# Critical: test uncertainty handling
def test_ambiguous_segment_not_forced_to_patient():
    """Ambiguous speech produces UNKNOWN, not forced PATIENT."""
```

**Exit Criterion**: Ambiguous and overlapping speech produces explicit UNKNOWN/OVERLAP states, not forced patient attribution.

---

## Milestone 7: Degradation Under Load (Week 11)

**Goal**: Validate graceful degradation under realistic stress.

### Load Tests

```python
# tests/load/test_degradation.py

def test_degradation_level_1():
    """Trigger level 1 (reduce emotion2vec frequency)."""
    # Inject artificial latency
    # Verify degradation manager triggers level 1
    # Verify emotion2vec hop multiplier = 2
    
def test_degradation_level_2():
    """Trigger level 2 (disable emotion2vec)."""
    
def test_degradation_level_3():
    """Trigger level 3 (disable diarization refinement)."""
    
def test_degradation_recovery():
    """System recovers when latency drops."""
    
def test_degradation_deterministic():
    """Same latency profile produces same degradation path."""
    
def test_degradation_observable():
    """Degradation events are logged and emitted."""
    
def test_no_silent_drops():
    """Verify every window produces a record or DropReason."""

# Stress tests
def test_slow_model_inference():
def test_queue_growth():
def test_cpu_saturation():
def test_model_timeout():
def test_missing_gpu():
def test_corrupt_segments():
def test_multiple_concurrent_streams():
```

**Exit Criterion**: Degradation is deterministic, observable, reversible, and never silently drops data.

---

## Milestone 8: Evaluation Pipeline (Week 12)

**Goal**: Prepare for model training with proper evaluation setup.

### Before Training LightGBM

Define evaluation protocol:

```python
# evaluation/dataset_splitter.py
class DatasetSplitter:
    """Split dataset at patient level."""
    
    def split(self, 
              dataset: pd.DataFrame,
              strategy: str = "patient_stratified") -> dict:
        """
        Split ensuring:
        - No patient appears in multiple splits
        - Balanced site/device distribution
        - Balanced SNR distribution
        - Balanced vocal condition distribution
        """
        return {
            'train': ...,
            'val': ...,
            'test': ...
        }

# evaluation/metrics.py
def compute_patient_attribution_metrics(y_true, y_pred, y_prob):
    """
    Return:
    - Patient precision
    - Patient recall
    - False patient attribution rate
    - Calibration curve
    - Abstention rate
    """

def compute_calibration_metrics(y_prob, y_true):
    """Expected calibration error, reliability diagram."""
```

### Augmentation (Separate Package)

```python
# augmentation/ (separate from inference code)
class NoiseAugmenter:
    """Add room noise (TRAINING ONLY)."""
    
class CodecAugmenter:
    """Simulate codec artifacts (TRAINING ONLY)."""
    
# These should NEVER be imported by runtime/ or offline/
```

**Exit Criterion**: Evaluation pipeline prevents leakage, stratifies correctly, and reports patient-specific metrics.

---

## Milestone 9: Formal Reviews (Week 13-14)

**Goal**: Complete reviews before any production claim.

### Required Reviews

#### 1. Architecture Review
- [ ] Schema contracts reviewed and approved
- [ ] Latency targets validated on hardware
- [ ] Degradation behavior validated
- [ ] Error handling reviewed

#### 2. Security Threat Model
- [ ] Attack surface identified
- [ ] Threat scenarios documented
- [ ] Mitigations implemented
- [ ] Residual risks accepted

#### 3. Privacy Impact Assessment
- [ ] Data flows mapped
- [ ] PII identified and controlled
- [ ] Biometric data handling validated
- [ ] Retention and deletion tested
- [ ] Access controls validated

#### 4. Dependency and License Review
- [ ] All dependencies listed with versions
- [ ] All licenses documented
- [ ] openSMILE commercial license verified
- [ ] Model licenses verified
- [ ] No GPL contamination

#### 5. Clinical Intended-Use Review
- [ ] Intended uses documented
- [ ] Non-intended uses documented
- [ ] Prohibited uses documented
- [ ] Model limitations disclosed
- [ ] Performance characteristics documented

#### 6. Operational Readiness Review
- [ ] Monitoring strategy defined
- [ ] Alerting thresholds set
- [ ] Incident response plan documented
- [ ] Rollback procedure tested
- [ ] Deployment checklist complete

**Exit Criterion**: All reviews complete and sign-offs obtained.

---

## Immediate Next Steps

### Week 1: Engineering Baseline

**Day 1-2**:
1. Create `pyproject.toml` with dependencies
2. Set up black, ruff, mypy
3. Set up pytest
4. Write tests for existing schemas
5. Write tests for degradation.py

**Day 3-4**:
6. Create `CONTRACTS.md` documenting all schemas
7. Review and freeze `AcousticFeatureRecord`
8. Review and freeze `SpeakerAttribution`
9. Create `AudioFrame` schema
10. Create `ErrorEvent` and `DegradationEvent` schemas

**Day 5**:
11. Set up CI pipeline
12. Run full test suite
13. Generate coverage report
14. Fix any issues until CI passes

**Exit Criterion**: CI passes with ≥80% coverage on existing code.

---

### Week 2-3: First Milestone — WAV-to-Parquet

**Deliverable**: 
```bash
audio-pipeline wav-to-parquet input.wav output.parquet

# Produces deterministic Parquet file with:
# - Timestamps
# - Quality metrics  
# - Provenance metadata
# - Placeholder fields for future models
```

**This is the foundation for everything else.**

---

## Success Metrics by Milestone

| Milestone | Success Metric |
|-----------|---------------|
| 0. Baseline | CI passes, critical behaviors covered, ≥80% line coverage |
| 1. WAV→Parquet | All acceptance tests pass, deterministic output |
| 2. Streaming | 1-hour stream, bounded memory, preserved ordering |
| 3. VAD | Documented thresholds, measured latencies on hardware |
| 4. eGeMAPS | Handles failures, timeout protection, deterministic |
| 5. YAMNet | Detects events outside VAD boundaries |
| 6. Speaker | UNKNOWN states for ambiguous speech |
| 7. Degradation | Deterministic, observable, reversible under load |
| 8. Evaluation | No leakage, patient-level splitting |
| 9. Reviews | All formal reviews complete with sign-offs |

---

## What NOT to Do

❌ Don't integrate all models before testing one  
❌ Don't skip the WAV-to-Parquet milestone  
❌ Don't train LightGBM before evaluation pipeline exists  
❌ Don't claim "production-ready" before formal reviews  
❌ Don't write code without tests  
❌ Don't add features without integration tests  
❌ Don't optimize before measuring  
❌ Don't silently drop data under any circumstance  

---

## Timeline Estimate

| Milestone | Duration | Cumulative |
|-----------|----------|------------|
| 0. Engineering baseline | 1 week | 1 week |
| 1. WAV-to-Parquet | 2 weeks | 3 weeks |
| 2. Streaming infrastructure | 2 weeks | 5 weeks |
| 3. VAD integration | 1 week | 6 weeks |
| 4. openSMILE | 1 week | 7 weeks |
| 5. YAMNet | 1 week | 8 weeks |
| 6. Speaker handling | 2 weeks | 10 weeks |
| 7. Degradation validation | 1 week | 11 weeks |
| 8. Evaluation pipeline | 1 week | 12 weeks |
| 9. Formal reviews | 2 weeks | 14 weeks |

**Total: ~14 weeks** (compared to original 16-week estimate)

**Note**: This is a target timeline assuming single full-time engineer, GPU availability, and no major blockers. Actual duration depends on team size, infrastructure, dataset availability, and regulatory requirements.

**Advantage**: Working, tested vertical slice upon Milestone 1 completion instead of month 4.

---

## Key Principle

**Build the simplest thing that could possibly work, test it thoroughly, then add complexity incrementally.**

Each milestone is independently valuable and testable. If you need to stop at any point, you have a working system up to that milestone.

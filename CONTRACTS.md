# Data Contracts

## Purpose

This document defines the stable data contracts that all modules must implement against. **These contracts are frozen after engineering review.**

Changes to contracts require:
1. Engineering review
2. Backward compatibility analysis
3. Migration plan for existing data
4. Version increment

---

## Timestamp and Sequencing Rules

### Timestamp Format
- **Format**: Milliseconds since session start (int64)
- **Origin**: Session start time (session_start_timestamp_ns)
- **Monotonic**: Timestamps MUST be monotonic within a session
- **Session-relative**: Reset to 0 at session start

### Sample Index Rules
- **Format**: 0-indexed, unsigned int64
- **Origin**: First sample in session = 0
- **Monotonic**: Sample indices MUST be monotonic within a session
- **Discontinuity marking**: Gaps MUST be marked with discontinuity event

### Sequence Numbers
- **Format**: Unsigned int64
- **Origin**: First chunk/window in session = 0
- **Monotonic**: MUST increment by 1 for each chunk/window
- **No gaps**: Missing sequence numbers indicate dropped data

### Session and Stream IDs
- **Session ID**: UUID v4 (pseudonymous, no PHI)
- **Stream ID**: UUID v4 within session (for multi-stream sessions)
- **Format**: String representation of UUID
- **Immutable**: Cannot change during session

---

## Core Schemas

### 1. AudioFrame

Raw audio data container.

```python
@dataclass(frozen=True)
class AudioFrame:
    """
    Raw audio frame from capture or file.
    
    Immutable container for audio samples with metadata.
    """
    
    # Audio data
    samples: np.ndarray
    """Audio samples. Shape: (n_samples,) for mono, (n_channels, n_samples) for multi-channel."""
    
    sample_rate: int
    """Sample rate in Hz."""
    
    n_channels: int
    """Number of channels."""
    
    # Sequencing
    sequence_number: int
    """Monotonic sequence number (0-indexed)."""
    
    start_sample_index: int
    """Global sample index of first sample in this frame."""
    
    # Timing
    capture_timestamp_ns: int
    """Capture time in nanoseconds (time.time_ns())."""
    
    session_relative_timestamp_ms: int
    """Milliseconds since session start."""
    
    # Discontinuity marking
    discontinuity: bool
    """True if this frame follows a gap."""
    
    discontinuity_reason: Optional[str]
    """Reason for discontinuity: BUFFER_OVERFLOW | DEVICE_ERROR | FILE_BOUNDARY | None"""
    
    # Session context
    session_id: str
    """Session UUID."""
    
    stream_id: str
    """Stream UUID within session."""
    
    def __post_init__(self):
        assert self.samples.ndim in (1, 2), "samples must be 1D or 2D"
        assert self.sample_rate > 0, "sample_rate must be positive"
        assert self.n_channels > 0, "n_channels must be positive"
        assert self.sequence_number >= 0, "sequence_number must be non-negative"
        assert self.start_sample_index >= 0, "start_sample_index must be non-negative"
        
        if self.samples.ndim == 2:
            assert self.samples.shape[0] == self.n_channels
    
    @property
    def n_samples(self) -> int:
        """Number of samples in this frame."""
        return self.samples.shape[-1]
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.sample_rate
    
    @property
    def end_sample_index(self) -> int:
        """Global sample index of last sample (exclusive)."""
        return self.start_sample_index + self.n_samples
```

---

### 2. SpeechSegment

Speech segment identified by VAD.

```python
@dataclass
class SpeechSegment:
    """
    Speech segment identified by VAD.
    
    Represents a contiguous region of speech.
    """
    
    # Temporal boundaries
    start_ms: int
    """Start time in milliseconds from session start."""
    
    end_ms: int
    """End time in milliseconds from session start (exclusive)."""
    
    start_sample: int
    """Start sample index (inclusive)."""
    
    end_sample: int
    """End sample index (exclusive)."""
    
    # VAD metadata
    vad_probability_mean: float
    """Mean VAD probability across segment (0.0-1.0)."""
    
    vad_probability_min: float
    """Minimum VAD probability (0.0-1.0)."""
    
    provisional: bool
    """True if segment boundary not yet confirmed by silence."""
    
    # Session context
    session_id: str
    stream_id: str
    
    def __post_init__(self):
        assert self.start_ms <= self.end_ms
        assert self.start_sample <= self.end_sample
        assert 0.0 <= self.vad_probability_mean <= 1.0
        assert 0.0 <= self.vad_probability_min <= 1.0
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
    
    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample
```

---

### 3. SpeakerAttribution

**Already defined** in `src/audio_pipeline/schemas/speaker_attribution.py`.

**Contract rules**:
- `patient_probability` MUST be in [0.0, 1.0]
- `status` MUST be one of: PATIENT, NON_PATIENT, UNKNOWN, OVERLAP, LOW_CONFIDENCE
- `speaker_id` MUST be pseudonymous (no PHI)
- UNKNOWN and OVERLAP MUST be emitted when attribution is ambiguous
- Never force ambiguous speech to PATIENT status

---

### 4. AcousticFeatureRecord

**Already defined** in `src/audio_pipeline/schemas/feature_record.py`.

**Contract rules**:
- All timestamps and sample indices MUST be monotonic within session
- `session_id` and `stream_id` MUST be pseudonymous UUIDs
- `egemaps` MUST be 88-dimensional if present, None if extraction failed
- `yamnet_event_scores` MUST be dict, empty dict if extraction failed
- `emotion_embedding` MUST be consistent dimension if present, None if failed
- `quality_status` MUST be one of: OK, CLIPPED, LOW_SNR, DROPOUT, REJECTED, INSUFFICIENT_AUDIO, UNKNOWN
- `attribution_status` MUST match SpeakerAttribution.status values
- Never omit a window silently—emit record with appropriate status

---

### 5. ErrorEvent

Errors and exceptions during processing.

```python
@dataclass
class ErrorEvent:
    """
    Error event during processing.
    
    Used for logging, monitoring, and debugging.
    """
    
    # When
    timestamp_ns: int
    """When error occurred (time.time_ns())."""
    
    session_relative_ms: int
    """Milliseconds from session start."""
    
    # Where
    component: str
    """Component where error occurred (e.g., 'vad', 'diarization', 'egemaps')."""
    
    # What
    error_type: str
    """Error type (e.g., 'TimeoutError', 'ModelError', 'ValueError')."""
    
    error_message: str
    """Human-readable error message."""
    
    severity: str
    """ERROR | WARNING | CRITICAL"""
    
    # Context
    session_id: str
    stream_id: str
    
    window_start_ms: Optional[int] = None
    """Window start if error is window-specific."""
    
    window_end_ms: Optional[int] = None
    """Window end if error is window-specific."""
    
    stack_trace: Optional[str] = None
    """Stack trace (optional, for debugging)."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context."""
    
    def __post_init__(self):
        assert self.severity in {"ERROR", "WARNING", "CRITICAL"}
```

---

### 6. DegradationEvent

Graceful degradation level changes.

```python
@dataclass
class DegradationEvent:
    """
    Degradation level change event.
    
    Emitted when system changes degradation level.
    """
    
    # When
    timestamp_ns: int
    """When degradation changed (time.time_ns())."""
    
    session_relative_ms: int
    """Milliseconds from session start."""
    
    # What changed
    old_level: int
    """Previous degradation level (0-5)."""
    
    new_level: int
    """New degradation level (0-5)."""
    
    trigger_reason: str
    """Why degradation changed."""
    
    # Metrics
    latency_p95_ms: float
    """95th percentile latency that triggered change."""
    
    trigger_threshold_ms: float
    """Threshold that was crossed."""
    
    # Active components after change
    active_components: dict[str, bool]
    """Component activation state after change."""
    
    # Context
    session_id: str
    stream_id: str
    
    def __post_init__(self):
        assert 0 <= self.old_level <= 5
        assert 0 <= self.new_level <= 5
```

---

### 7. DropReason

**Already defined** in `src/audio_pipeline/schemas/feature_record.py`.

**Contract rules**:
- `reason` MUST be one of: DROPPED_BACKPRESSURE, MODEL_TIMEOUT, INSUFFICIENT_AUDIO, LOW_QUALITY, SPEAKER_UNKNOWN, DEGRADATION
- `degradation_level` MUST be 0-5
- Every dropped window MUST emit a DropReason record
- Never silently drop windows

---

## Missing Value Semantics

### When to use None
- Feature extraction failed completely
- Model timed out
- Model unavailable (degradation)
- Insufficient input data

### When to use 0.0
- Never use 0.0 to represent missing data
- Only use 0.0 for actual zero values

### When to use empty dict/list
- `yamnet_event_scores`: Empty dict if no events detected OR extraction failed
- Feature arrays: None if extraction failed, never empty list

### When to use NaN
- Avoid NaN in Parquet (compatibility issues)
- Use None instead

---

## Validation Rules

### At Schema Level
- Type checking with Python type hints
- Range validation in `__post_init__`
- Immutability for data containers (use `frozen=True`)

### At Module Boundaries
- Every module MUST validate inputs
- Every module MUST handle invalid inputs gracefully
- Every module MUST log validation failures

### Example Validation

```python
def validate_audio_frame(frame: AudioFrame) -> None:
    """Validate AudioFrame contract."""
    assert isinstance(frame, AudioFrame)
    assert frame.samples.dtype in (np.float32, np.float64, np.int16)
    assert frame.sample_rate in (8000, 16000, 22050, 44100, 48000)
    assert frame.n_channels in (1, 2)
    assert frame.sequence_number >= 0
    # etc.
```

---

## Versioning

### Schema Versions
- Every schema MUST have a version field
- Version format: `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes (requires migration)
- MINOR: Backward-compatible additions
- PATCH: Bug fixes, clarifications

### Parquet Schema Management

```python
# Include in every Parquet file metadata
schema_version: str = "1.0.0"
extractor_version: str = "audio-pipeline-0.1.0"
```

**Schema Migration Requirements**:
- `schema_version` field REQUIRED in metadata
- Compatibility policy documented per MAJOR version
- Nullable-field rules: new fields MUST be nullable or have defaults
- Column addition: allowed in MINOR versions
- Column removal: MAJOR version only, with deprecation period
- Migration tooling REQUIRED for MAJOR changes
- Reader behavior for unknown versions: reject or warn
- Feature-vector dimension/version metadata included
- Golden fixture files for each schema version

**Example Migration**:
```python
# v1.0.0 → v1.1.0 (add optional field)
# Reader must handle missing field gracefully

# v1.x.x → v2.0.0 (breaking change)
# Migration tool: convert_v1_to_v2.py
# Compatibility: readers MUST reject or convert
```

### Handling Version Mismatches
- Readers MUST check schema version
- Readers MUST reject incompatible MAJOR versions
- Readers SHOULD warn on MINOR version mismatches
- Readers MUST ignore PATCH differences

---

## Backward Compatibility

### Adding Fields
- ✅ OK: Add optional fields with defaults
- ❌ NOT OK: Add required fields to existing schemas

### Removing Fields
- ❌ NOT OK: Remove fields (MAJOR version bump required)
- ✅ OK: Deprecate fields, keep for compatibility

### Changing Types
- ❌ NOT OK: Change field types (MAJOR version bump required)
- ✅ OK: Widen types (int32 → int64 with MINOR bump)

---

## Testing Contract Compliance

Every schema MUST have tests:

```python
# tests/contracts/test_audio_frame_contract.py

def test_audio_frame_basic():
    """AudioFrame with valid data."""
    
def test_audio_frame_mono():
    """Mono audio."""
    
def test_audio_frame_stereo():
    """Stereo audio."""
    
def test_audio_frame_invalid_sample_rate():
    """Reject invalid sample rate."""
    with pytest.raises(AssertionError):
        AudioFrame(...)
    
def test_audio_frame_invalid_shape():
    """Reject invalid sample shape."""
    
def test_audio_frame_negative_sequence():
    """Reject negative sequence number."""
    
def test_audio_frame_discontinuity():
    """Handle discontinuity correctly."""
```

---

## Documentation Requirements

Every schema MUST document:
1. **Purpose**: What the schema represents
2. **Constraints**: Valid ranges, formats, invariants
3. **Relationships**: How it relates to other schemas
4. **Examples**: Code examples of creation and use
5. **Version history**: Changes across versions

---

## Review and Approval

**Status**: DRAFT (awaiting cross-functional engineering review)

**Versioning Strategy**:
- **v0.1-provisional**: Initial approval for prototype implementation
- **v1.0**: Stabilized after Milestone 1 validates contracts in practice
- **Later versions**: Require backward-compatible evolution or explicit migrations

**Required approvals for v0.1-provisional**:

| Role | Approval Scope | Reviewer |
|------|----------------|----------|
| Runtime owner | Streaming and concurrency contracts | __________ |
| Data owner | Parquet schema and retention | __________ |
| ML owner | Model input/output contracts | __________ |
| Security | Threat controls and secrets | __________ |
| Privacy | Identifiers, embeddings, retention | __________ |
| Clinical/domain owner | Intended use and interpretation | __________ |
| Technical lead | Final integration decision | __________ |

**Approval threshold for later changes**:
- **Patch (1.0.x)**: Technical lead only
- **Minor (1.x.0)**: Affected domain owners + technical lead
- **Major (x.0.0)**: Full re-approval

**Approval Date (v0.1)**: __________

**Contract Version**: 0.1-provisional

**Stabilization Date (v1.0)**: __________ (after Milestone 1 prototype validation)

---

## Change Log

| Version | Date | Change | Approved By |
|---------|------|--------|-------------|
| 1.0.0-draft | 2024-07-20 | Initial draft | Pending |

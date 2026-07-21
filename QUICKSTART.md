# Quick Start Guide

## For Person 1: Getting Started

This guide helps you get started with implementing the audio pipeline.

## Overview

You're building **three separate systems**:

1. **Real-time streaming** — processes live audio with low latency
2. **Offline batch** — extracts features from recorded audio files
3. **Training** — augmentation and LightGBM model training

**Critical**: These are architecturally separate. Don't mix them.

---

## Day 1: Setup Environment

### 1. Clone/Create Repository

```bash
cd ~/Desktop/odu-audio
git init
git add .
git commit -m "Initial audio pipeline structure"
```

### 2. Create Python Environment

```bash
# Using conda
conda create -n audio-pipeline python=3.10
conda activate audio-pipeline

# Or using venv
python3.10 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Core Dependencies

```bash
pip install numpy scipy pandas pyarrow soundfile librosa resampy pyyaml
```

### 4. Install PyTorch (for VAD & diarization)

```bash
# CPU only
pip install torch torchaudio

# With GPU (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# macOS Apple Silicon (M1/M2)
pip install torch torchaudio
```

### 5. Install Silero VAD

```bash
pip install git+https://github.com/snakers4/silero-vad.git
```

### 6. Set Up HuggingFace for Pyannote

```bash
# Create account at https://huggingface.co
# Accept user agreements at:
# https://huggingface.co/pyannote/speaker-diarization
# https://huggingface.co/pyannote/segmentation

pip install pyannote.audio huggingface-hub

# Set your token
export HF_TOKEN="your_token_here"
# Or save to ~/.bashrc or ~/.zshrc
```

---

## Day 2-3: Implement Core Schemas

**Start with schemas** — they define your data contracts.

### Priority Order

1. `src/audio_pipeline/schemas/feature_record.py` ✅ (already created)
2. `src/audio_pipeline/schemas/speaker_attribution.py` ✅ (already created)
3. `src/audio_pipeline/schemas/audio_frame.py` — Create this
4. `src/audio_pipeline/schemas/segment.py` — Create this

### Test Your Schemas

```python
# Test in Python REPL
from audio_pipeline.schemas.feature_record import AcousticFeatureRecord

record = AcousticFeatureRecord(
    session_id="session_001",
    stream_id="stream_001",
    window_start_ms=0,
    window_end_ms=2000,
    source_start_sample=0,
    source_end_sample=32000,
    attribution_status="PATIENT",
    egemaps=[0.0] * 88,  # 88 functionals
)

print(record.is_patient_speech())  # Should work
print(record.to_dict())  # Should serialize
```

---

## Week 1: Audio Capture & Ring Buffer

### Goal
Capture audio from microphone into a ring buffer with <20ms latency.

### Implementation

```python
# src/audio_pipeline/capture/microphone.py

import sounddevice as sd
import numpy as np
from collections import deque

class RingBuffer:
    def __init__(self, max_duration_sec=10.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        self.buffer = deque(maxlen=self.max_samples)
    
    def write(self, audio_chunk):
        self.buffer.extend(audio_chunk)
    
    def read(self, n_samples):
        if len(self.buffer) < n_samples:
            return None
        result = np.array(list(self.buffer)[-n_samples:])
        return result

class MicrophoneCapture:
    def __init__(self, sample_rate=16000, chunk_size=480):  # 30ms chunks
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.ring_buffer = RingBuffer(sample_rate=sample_rate)
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        self.ring_buffer.write(indata[:, 0])  # mono
    
    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size,
        )
        self.stream.start()
    
    def stop(self):
        self.stream.stop()
        self.stream.close()
```

### Test It

```python
# Test script
capture = MicrophoneCapture()
capture.start()

import time
time.sleep(2)  # Record for 2 seconds

audio = capture.ring_buffer.read(16000)  # Get 1 second
print(f"Captured {len(audio)} samples")

capture.stop()
```

### Measure Latency

```python
import time

def measure_capture_latency():
    capture = MicrophoneCapture(chunk_size=480)  # 30ms
    capture.start()
    
    start = time.time()
    time.sleep(0.1)  # Wait for buffer to fill
    audio = capture.ring_buffer.read(480)
    latency_ms = (time.time() - start) * 1000
    
    capture.stop()
    return latency_ms

latency = measure_capture_latency()
print(f"Capture latency: {latency:.1f} ms")
# Target: <20 ms
```

---

## Week 2: VAD Integration

### Goal
Integrate Silero VAD with <50ms preliminary decision.

### Quick Implementation

```python
# src/audio_pipeline/segmentation/vad.py

import torch
from typing import Optional

class SileroVAD:
    def __init__(self, threshold=0.5, sample_rate=16000):
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.get_speech_timestamps = utils[0]
    
    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Process audio chunk and return VAD probability.
        
        Args:
            audio_chunk: Audio samples (numpy array)
        
        Returns:
            VAD probability (0.0-1.0)
        """
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        return speech_prob
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if chunk contains speech"""
        prob = self.process_chunk(audio_chunk)
        return prob >= self.threshold
```

### Test VAD

```python
# Load test audio
import soundfile as sf

audio, sr = sf.read("test_audio.wav")

# Initialize VAD
vad = SileroVAD(threshold=0.5, sample_rate=sr)

# Process in chunks
chunk_size = 512  # Silero works well with 512 samples at 16kHz
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    if len(chunk) < chunk_size:
        break
    
    prob = vad.process_chunk(chunk)
    is_speech = prob >= 0.5
    print(f"Chunk {i//chunk_size}: prob={prob:.3f}, speech={is_speech}")
```

### Measure VAD Latency

```python
import time

def measure_vad_latency():
    vad = SileroVAD()
    audio_chunk = np.random.randn(512).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        vad.process_chunk(audio_chunk)
    
    # Measure
    start = time.time()
    for _ in range(100):
        vad.process_chunk(audio_chunk)
    latency_ms = (time.time() - start) * 1000 / 100
    
    return latency_ms

latency = measure_vad_latency()
print(f"VAD latency: {latency:.2f} ms")
# Target: <5 ms
# Silero is typically <1ms on CPU
```

---

## Week 3-4: YAMNet Event Detection

### Goal
Detect scream/yell/cry events on full audio stream (NOT VAD-filtered).

### Quick Implementation

```python
# src/audio_pipeline/features/acoustic_events.py

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class YAMNetDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Events we care about
        self.target_events = {
            'Scream': None,
            'Crying, sobbing': None,
            'Yell': None,
            'Gasp': None,
            'Groan': None,
            'Whimper': None,
        }
        
        # Load class names
        self.class_names = self.model.class_names().numpy()
        self._map_event_indices()
    
    def _map_event_indices(self):
        """Map event names to class indices"""
        for i, name in enumerate(self.class_names):
            name_str = name.decode('utf-8')
            if name_str in self.target_events:
                self.target_events[name_str] = i
    
    def detect(self, audio: np.ndarray) -> dict[str, float]:
        """
        Detect events in audio.
        
        Args:
            audio: Audio samples (float32, mono, 16kHz)
        
        Returns:
            Dict of {event_name: max_probability}
        """
        # YAMNet expects audio as float32 in [-1, 1]
        audio = audio.astype(np.float32)
        
        # Run inference
        scores, embeddings, spectrogram = self.model(audio)
        scores = scores.numpy()
        
        # Extract target event scores
        results = {}
        for event_name, class_idx in self.target_events.items():
            if class_idx is not None:
                # Take max probability across all windows
                max_prob = np.max(scores[:, class_idx])
                results[event_name] = float(max_prob)
            else:
                results[event_name] = 0.0
        
        return results
```

### Test YAMNet

```python
# Test on audio file
import soundfile as sf

audio, sr = sf.read("scream_audio.wav")

detector = YAMNetDetector(sample_rate=sr)
events = detector.detect(audio)

for event, prob in events.items():
    if prob > 0.1:  # Filter low probabilities
        print(f"{event}: {prob:.3f}")
```

---

## Week 4-6: Speaker Diarization

### Goal
Implement patient enrollment and speaker attribution.

### Quick Start with Pyannote

```python
# src/audio_pipeline/speakers/diarization.py

from pyannote.audio import Pipeline
import torch

class SpeakerDiarizer:
    def __init__(self, hf_token: str):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
    
    def diarize(self, audio_file: str):
        """
        Run speaker diarization on audio file.
        
        Returns:
            pyannote.core.Annotation object
        """
        diarization = self.pipeline(audio_file)
        return diarization
```

### Test Diarization

```python
import os

# Get token from environment
hf_token = os.getenv("HF_TOKEN")

diarizer = SpeakerDiarizer(hf_token=hf_token)
diarization = diarizer.diarize("test_audio.wav")

# Print results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {speaker}")
```

---

## Week 10+: Put It All Together

### Streaming Pipeline Entry Point

```python
# Example usage
from audio_pipeline.runtime import StreamProcessor

processor = StreamProcessor(config_path="configs/streaming_default.yaml")
processor.start()

for feature_record in processor.stream():
    # Process each window
    if feature_record.is_patient_speech(threshold=0.7):
        print(f"Patient speech: {feature_record.window_start_ms}ms")
        
        # Check for distress events
        for event, prob in feature_record.yamnet_event_scores.items():
            if prob > 0.5:
                print(f"  ⚠️  {event}: {prob:.2f}")
        
        # Access features
        if feature_record.egemaps is not None:
            print(f"  eGeMAPS: {len(feature_record.egemaps)} features")
```

---

## Common Issues & Solutions

### Issue: PyTorch not finding CUDA

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with correct CUDA version
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Pyannote authentication fails

```bash
# Make sure you accepted user agreements at:
# https://huggingface.co/pyannote/speaker-diarization

# Set token
export HF_TOKEN="your_token"

# Or in Python
import os
os.environ["HF_TOKEN"] = "your_token"
```

### Issue: Audio capture fails

```bash
# List available devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Specify device explicitly
sd.InputStream(device=0, ...)  # Use device 0
```

---

## Development Workflow

### 1. Write Component

Example: `src/audio_pipeline/quality/clipping.py`

### 2. Write Unit Test

Example: `tests/unit/test_clipping.py`

```python
import pytest
import numpy as np
from audio_pipeline.quality.clipping import detect_clipping

def test_no_clipping():
    audio = np.random.randn(16000) * 0.5  # No clipping
    ratio = detect_clipping(audio, threshold=0.99)
    assert ratio < 0.01

def test_heavy_clipping():
    audio = np.ones(16000)  # All clipped
    ratio = detect_clipping(audio, threshold=0.99)
    assert ratio > 0.95
```

### 3. Run Test

```bash
pytest tests/unit/test_clipping.py -v
```

### 4. Benchmark (for critical components)

```python
import time
import numpy as np

def benchmark_clipping_detection():
    audio = np.random.randn(16000)
    
    # Warmup
    for _ in range(10):
        detect_clipping(audio)
    
    # Measure
    start = time.time()
    for _ in range(1000):
        detect_clipping(audio)
    
    latency_ms = (time.time() - start) * 1000 / 1000
    print(f"Clipping detection: {latency_ms:.3f} ms")

benchmark_clipping_detection()
```

---

## Recommended Reading Order

1. **PERSON1_SUMMARY.md** — High-level overview
2. **ARCHITECTURE.md** — Design decisions and rationale
3. **README.md** — Complete project documentation
4. **TASKS.md** — Detailed 16-week plan
5. **This file (QUICKSTART.md)** — Implementation guide
6. **configs/streaming_default.yaml** — Configuration reference
7. **model_cards/emotion2vec.yaml** — Example model card

---

## Getting Help

### Documentation
- Silero VAD: https://github.com/snakers4/silero-vad
- Pyannote: https://github.com/pyannote/pyannote-audio
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- openSMILE: https://audeering.github.io/opensmile/

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_vad.py -v

# Run with coverage
pytest tests/ --cov=audio_pipeline --cov-report=html
```

---

## Next Steps

1. ✅ Review this quick start
2. ✅ Set up environment (Python, PyTorch, HuggingFace)
3. ✅ Test schemas (`feature_record.py`, `speaker_attribution.py`)
4. ⏸ Implement audio capture (Week 1)
5. ⏸ Implement VAD (Week 2)
6. ⏸ Implement YAMNet (Week 3-4)
7. ⏸ Implement diarization (Week 4-6)

**Good luck!** 🚀

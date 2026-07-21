"""
Streaming integration tests for VAD, ensuring segments propagate and metrics are computed.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.audio_pipeline.capture.file_source import FileSource
from src.audio_pipeline.runtime.stream_processor import StreamProcessor
from src.audio_pipeline.segmentation.dummy_vad import DummyVAD


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def generate_speech_wav(path, sample_rate=16000, duration=4.0):
    """Generate a WAV file containing interleaved speech and silence."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.zeros(n_samples, dtype=np.float32)

    # 1.0s speech, 1.0s silence, 1.0s speech, 1.0s silence
    signal[0:16000] = 0.5 * np.sin(2 * np.pi * 440 * t[0:16000])
    signal[32000:48000] = 0.5 * np.sin(2 * np.pi * 440 * t[32000:48000])

    sf.write(path, signal, sample_rate, subtype="PCM_16")
    return signal


def test_vad_in_streaming_pipeline(temp_dir):
    wav_path = os.path.join(temp_dir, "speech.wav")
    generate_speech_wav(wav_path, sample_rate=16000, duration=4.0)

    # Run StreamProcessor using FileSource and a mock VAD to simulate speech detection
    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    pattern = [0.9] * 10 + [0.1] * 10 + [0.9] * 10 + [0.1] * 10
    dummy_vad = DummyVAD(pattern=pattern)
    processor = StreamProcessor(
        source, config={"quality": {"snr_min_db": 0.0}}, vad=dummy_vad
    )

    processor.start()
    records = list(processor.stream())
    processor.stop()

    # Verify that speech segments are captured
    assert len(processor.speech_segments) > 0

    # The segments should have correct temporal ordering and labels
    for seg in processor.speech_segments:
        assert seg.start_ms < seg.end_ms
        assert seg.start_sample < seg.end_sample
        assert seg.provisional is False

    # AcousticFeatureRecord should contain VAD metrics
    assert len(records) > 0
    assert any(rec.vad_probability_mean > 0.1 for rec in records)
    assert any(rec.voiced_ratio > 0.0 for rec in records)


def test_streaming_with_dummy_vad(temp_dir):
    wav_path = os.path.join(temp_dir, "speech.wav")
    generate_speech_wav(wav_path, sample_rate=16000, duration=2.0)

    # Configure DummyVAD with a pattern (8 chunks of speech, 12 chunks of silence)
    pattern = [0.9] * 8 + [0.1] * 12
    dummy_vad = DummyVAD(pattern=pattern)

    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    processor = StreamProcessor(
        source, config={"quality": {"snr_min_db": 0.0}}, vad=dummy_vad
    )

    processor.start()
    records = list(processor.stream())
    processor.stop()

    assert len(records) > 0
    assert len(processor.speech_segments) > 0

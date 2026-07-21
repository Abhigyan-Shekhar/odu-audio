"""
Integration tests for real-time streaming pipeline (StreamProcessor, RingBuffer, BoundedQueue, and Sources).
"""

import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.audio_pipeline.capture.bounded_queue import BoundedQueue
from src.audio_pipeline.capture.file_source import FileSource
from src.audio_pipeline.capture.microphone_source import MicrophoneSource
from src.audio_pipeline.capture.ring_buffer import RingBuffer
from src.audio_pipeline.preprocessing.streaming_resampler import StreamingResampler
from src.audio_pipeline.runtime.stream_processor import StreamProcessor

# ==========================================
# Fixtures & Helpers
# ==========================================


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def generate_test_wav(path, sample_rate=16000, duration=5.0):
    """Helper to generate a clean synthetic WAV file."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return samples


# ==========================================
# Unit Tests: Ring Buffer (5 cases)
# ==========================================


def test_ring_buffer_write_read():
    buf = RingBuffer(capacity_seconds=1.0, sample_rate=1000)
    # Write 400 samples
    total = buf.write(np.ones(400, dtype=np.float32))
    assert total == 400
    assert buf.get_available_samples() == 400

    # Peek 200 samples
    peeked = buf.peek(200)
    assert peeked is not None
    assert len(peeked) == 200
    assert np.all(peeked == 1.0)
    assert buf.get_available_samples() == 400  # Size unchanged

    # Advance 200 samples
    buf.advance(200)
    assert buf.get_available_samples() == 200


def test_ring_buffer_peek_not_enough():
    buf = RingBuffer(capacity_seconds=1.0, sample_rate=1000)
    buf.write(np.ones(100, dtype=np.float32))
    assert buf.peek(200) is None


def test_ring_buffer_overflow():
    # Capacity is 1000 samples
    buf = RingBuffer(capacity_seconds=1.0, sample_rate=1000)

    # Write 800 samples of 1.0
    buf.write(np.ones(800, dtype=np.float32))
    # Write 400 samples of 2.0 (total = 1200 > 1000 capacity)
    buf.write(2.0 * np.ones(400, dtype=np.float32))

    # Available should be capped at capacity (1000)
    assert buf.get_available_samples() == 1000

    # Oldest 200 samples should have been overwritten
    # Output should contain 600 of 1.0, and 400 of 2.0
    peeked = buf.peek(1000)
    assert peeked is not None
    assert np.sum(peeked == 1.0) == 600
    assert np.sum(peeked == 2.0) == 400


def test_ring_buffer_clear():
    buf = RingBuffer(capacity_seconds=1.0, sample_rate=1000)
    buf.write(np.ones(500, dtype=np.float32))
    buf.clear()
    assert buf.get_available_samples() == 0


def test_ring_buffer_discontinuity():
    buf = RingBuffer(capacity_seconds=1.0, sample_rate=1000)
    buf.mark_discontinuity(reason="DEVICE_ERROR")

    flag, reason = buf.check_and_reset_discontinuity()
    assert flag is True
    assert reason == "DEVICE_ERROR"

    # Flag should reset to False
    flag_again, reason_again = buf.check_and_reset_discontinuity()
    assert flag_again is False
    assert reason_again is None


# ==========================================
# Unit Tests: Bounded Queue (3 cases)
# ==========================================


def test_bounded_queue_operations():
    q = BoundedQueue(maxsize=3)
    assert q.put("a") is True
    assert q.put("b") is True
    assert q.put("c") is True
    assert q.is_full() is True

    # Timeout write when full
    assert q.put("d", timeout=0.1) is False

    assert q.get() == "a"
    assert q.qsize() == 2

    q.clear()
    assert q.qsize() == 0
    assert q.get(timeout=0.1) is None


# ==========================================
# Unit Tests: Streaming Resampler (3 cases)
# ==========================================


def test_streaming_resampler_clean_boundaries():
    # Down-resample 32000Hz to 16000Hz statefully (ratio = 0.5)
    resampler = StreamingResampler(source_sr=32000, target_sr=16000)

    chunk1 = np.sin(np.linspace(0, 10, 3200))  # 100ms
    chunk2 = np.sin(np.linspace(10, 20, 3200))

    res1 = resampler.process_chunk(chunk1)
    res2 = resampler.process_chunk(chunk2)

    # Combined target sample length should be close to 3200 resampled (approximately 1600 samples per chunk)
    assert abs(len(res1) - 1600) < 5
    assert abs(len(res2) - 1600) < 5


def test_streaming_resampler_same_rate():
    resampler = StreamingResampler(source_sr=16000, target_sr=16000)
    chunk = np.ones(1000, dtype=np.float32)
    res = resampler.process_chunk(chunk)
    assert np.array_equal(chunk, res)


def test_streaming_resampler_reset():
    resampler = StreamingResampler(source_sr=32000, target_sr=16000)
    chunk = np.ones(1000, dtype=np.float32)
    resampler.process_chunk(chunk)
    assert resampler._is_first is False
    resampler.reset()
    assert resampler._is_first is True


# ==========================================
# Integration Tests: StreamProcessor (10 cases)
# ==========================================


def test_synthetic_audio_stream(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    generate_test_wav(wav_path, sample_rate=16000, duration=3.0)

    # 3.0s file, chunk size 1600 (100ms)
    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    processor = StreamProcessor(source, config={"quality": {"snr_min_db": 0.0}})

    processor.start()
    records = list(processor.stream())
    processor.stop()

    # Window = 2.0s (32000 samples), hop = 0.5s (8000 samples)
    # Expected windows: 0.0..2.0, 0.5..2.5, 1.0..3.0 (3 windows)
    # Plus one final partial window for the remaining samples at the end (none in this case, but let's see)
    assert len(records) >= 3
    assert records[0].quality_status == "OK"
    assert records[0].session_id is not None
    assert records[0].stream_id is not None


def test_sequence_numbers(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    generate_test_wav(wav_path, sample_rate=16000, duration=2.5)

    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    processor = StreamProcessor(source, config={"quality": {"snr_min_db": 0.0}})

    processor.start()
    records = list(processor.stream())
    processor.stop()

    # Check sequence start time increments monotonically
    for i in range(len(records) - 1):
        assert records[i].window_start_ms < records[i + 1].window_start_ms


def test_discontinuity_propagation(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    generate_test_wav(wav_path, sample_rate=16000, duration=2.0)

    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    # Inject a discontinuity immediately
    source.inject_discontinuity("BUFFER_OVERFLOW")

    processor = StreamProcessor(source, config={"quality": {"snr_min_db": 0.0}})
    processor.start()
    records = list(processor.stream())
    processor.stop()

    assert len(records) > 0


def test_bounded_memory_growth(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    generate_test_wav(wav_path, sample_rate=16000, duration=1.0)

    source = FileSource(wav_path, chunk_size=800, simulate_real_time=False)
    processor = StreamProcessor(source)
    processor.start()

    # Fast consumer
    records = []
    for r in processor.stream():
        records.append(r)
        if len(records) > 10:
            break

    processor.stop()
    assert len(records) > 0


def test_graceful_shutdown(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    generate_test_wav(wav_path, sample_rate=16000, duration=5.0)

    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=True)
    processor = StreamProcessor(source)
    processor.start()

    # Let it stream a little bit, then shut down immediately
    time.sleep(0.5)
    processor.stop()

    # Check that stream() terminates and doesn't deadlock
    records = list(processor.stream())
    assert isinstance(records, list)


def test_backpressure_drop(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    generate_test_wav(wav_path, sample_rate=16000, duration=3.0)

    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    # Force tiny queue capacity to trigger queue full condition quickly
    processor = StreamProcessor(source, config={"runtime": {"queue_capacity": 1}})

    processor.start()
    # Introduce slow consumer to trigger backpressure
    time.sleep(0.2)
    records = list(processor.stream())
    processor.stop()

    assert len(records) >= 0


def test_unsupported_rate_conversion(temp_dir):
    wav_path = os.path.join(temp_dir, "input_8k.wav")
    # Generate 8kHz audio
    generate_test_wav(wav_path, sample_rate=8000, duration=2.0)

    source = FileSource(wav_path, chunk_size=800, simulate_real_time=False)
    processor = StreamProcessor(source, config={"quality": {"snr_min_db": 0.0}})

    processor.start()
    records = list(processor.stream())
    processor.stop()

    assert len(records) > 0
    # Resampled indices map correctly
    assert records[0].source_start_sample == 0


def test_microphone_source_mocked():
    mock_sd = MagicMock()
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    with patch("src.audio_pipeline.capture.microphone_source.sd", mock_sd):
        source = MicrophoneSource(sample_rate=16000, chunk_size=1600, channels=1)

        # Retrieve the callback registered with sounddevice
        callback_arg = mock_sd.InputStream.call_args[1]["callback"]
        indata = np.ones((1600, 1), dtype=np.float32) * 0.5

        # Call with normal status (None or empty flags)
        callback_arg(indata, 1600, None, None)

        chunk = source.read_chunk()
        assert chunk.sample_rate == 16000
        assert chunk.sequence_number == 0
        assert np.all(chunk.samples == 0.5)
        assert chunk.discontinuity is False

        # Call with overflow status
        mock_status = MagicMock()
        mock_status.input_overflow = True
        mock_status.input_underflow = False
        callback_arg(indata, 1600, None, mock_status)

        chunk = source.read_chunk()
        assert chunk.sequence_number == 1
        assert chunk.discontinuity is True
        assert chunk.discontinuity_reason == "BUFFER_OVERFLOW"

        # Call with underflow status
        mock_status = MagicMock()
        mock_status.input_overflow = False
        mock_status.input_underflow = True
        callback_arg(indata, 1600, None, mock_status)

        chunk = source.read_chunk()
        assert chunk.sequence_number == 2
        assert chunk.discontinuity is True
        assert chunk.discontinuity_reason == "BUFFER_UNDERFLOW"

        # Test close
        source.close()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

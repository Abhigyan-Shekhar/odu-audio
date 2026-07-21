"""
Unit tests for VAD components, including interface checks, Silero, Dummy VAD, and Endpointer.
"""

import numpy as np

from src.audio_pipeline.segmentation.dummy_vad import DummyVAD
from src.audio_pipeline.segmentation.endpointer import Endpointer
from src.audio_pipeline.segmentation.silero_vad import SileroVAD
from src.audio_pipeline.segmentation.vad_interface import VADInterface


def test_vad_interface():
    """Verify that VADInterface contract is satisfied by implementations."""
    assert issubclass(SileroVAD, VADInterface)
    assert issubclass(DummyVAD, VADInterface)


def test_silero_loads():
    """Verify Silero VAD loads successfully and retrieves version."""
    vad = SileroVAD()
    assert vad.get_version() == "silero_vad_v4"
    assert vad._sample_rate == 16000


def test_vad_probabilities():
    """Verify VAD processes chunks and returns correct probabilities."""
    vad = SileroVAD()
    # 512 samples at 16kHz
    chunk_silent = np.zeros(512, dtype=np.float32)
    prob_silent = vad.process_chunk(chunk_silent)
    assert 0.0 <= prob_silent <= 1.0

    # Sine wave chunk (should have higher speech probability)
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    chunk_speech = 0.5 * np.sin(2 * np.pi * 440 * t)
    prob_speech = vad.process_chunk(chunk_speech)
    assert 0.0 <= prob_speech <= 1.0


def test_endpointer_state_transitions():
    """Verify endpointer state machine correctly handles speech and silence transitions."""
    # Min speech duration = 100ms, min silence duration = 200ms, pad = 50ms
    # Using 100ms chunks (1600 samples)
    endpointer = Endpointer(
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=200,
        speech_pad_ms=50,
        sample_rate=16000,
    )

    # 1. Start silence
    seg = endpointer.process(
        prob=0.1,
        start_ms=0,
        end_ms=100,
        start_sample=0,
        end_sample=1600,
        session_id="sess_1",
        stream_id="str_1",
    )
    assert seg is None

    # 2. Speech onset
    seg = endpointer.process(
        prob=0.8,
        start_ms=100,
        end_ms=200,
        start_sample=1600,
        end_sample=3200,
        session_id="sess_1",
        stream_id="str_1",
    )
    assert seg is not None
    assert seg.provisional is True
    # Verify padded start index (1600 samples - 50ms pad * 16 samples/ms = 800)
    assert seg.start_sample == 800
    assert seg.start_ms == 50

    # 3. Continued speech
    seg = endpointer.process(
        prob=0.9,
        start_ms=200,
        end_ms=300,
        start_sample=3200,
        end_sample=4800,
        session_id="sess_1",
        stream_id="str_1",
    )
    assert seg is not None
    assert seg.provisional is True
    assert seg.end_sample == 4800

    # 4. First silence chunk (under min_silence_duration_ms threshold)
    seg = endpointer.process(
        prob=0.2,
        start_ms=300,
        end_ms=400,
        start_sample=4800,
        end_sample=6400,
        session_id="sess_1",
        stream_id="str_1",
    )
    assert seg is not None
    assert seg.provisional is True

    # 5. Second silence chunk (triggers silence threshold: total silence = 200ms)
    seg = endpointer.process(
        prob=0.1,
        start_ms=400,
        end_ms=500,
        start_sample=6400,
        end_sample=8000,
        session_id="sess_1",
        stream_id="str_1",
    )
    # Speech ended at 300ms (last speech chunk end). Padded end: 300ms + 50ms = 350ms
    assert seg is not None
    assert seg.provisional is False
    assert seg.end_ms == 350
    assert seg.end_sample == 3200 + 1600 + int(
        50 * 16.0
    )  # last speech end (4800) + pad (800) = 5600
    assert seg.duration_ms == 300  # 350ms - 50ms start = 300ms

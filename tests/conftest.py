"""
Pytest configuration and global fixtures.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_session_id():
    return "session-f47ac10b-58cc-4372-a567-0e02b2c3d479"


@pytest.fixture
def sample_stream_id():
    return "stream-a47bc10b-58cc-4372-a567-0e02b2c3d480"


@pytest.fixture
def mono_audio_samples():
    # 1 second of mono silent audio at 16000 Hz
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def stereo_audio_samples():
    # 1 second of stereo silent audio at 16000 Hz
    return np.zeros((2, 16000), dtype=np.float32)


@pytest.fixture
def sample_audio_frame(mono_audio_samples, sample_session_id, sample_stream_id):
    from src.audio_pipeline.schemas.audio_frame import AudioFrame

    return AudioFrame(
        samples=mono_audio_samples,
        sample_rate=16000,
        n_channels=1,
        sequence_number=0,
        start_sample_index=0,
        capture_timestamp_ns=1600000000000000000,
        session_relative_timestamp_ms=0,
        discontinuity=False,
        discontinuity_reason=None,
        session_id=sample_session_id,
        stream_id=sample_stream_id,
    )


@pytest.fixture
def sample_speech_segment(sample_session_id, sample_stream_id):
    from src.audio_pipeline.schemas.segment import SpeechSegment

    return SpeechSegment(
        start_ms=100,
        end_ms=1100,
        start_sample=1600,
        end_sample=17600,
        vad_probability_mean=0.95,
        vad_probability_min=0.85,
        provisional=False,
        session_id=sample_session_id,
        stream_id=sample_stream_id,
    )

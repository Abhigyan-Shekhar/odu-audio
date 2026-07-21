"""
Unit tests verifying AudioFrame contract properties and constraints.
"""

import pytest
import numpy as np
from src.audio_pipeline.schemas.audio_frame import AudioFrame
from dataclasses import FrozenInstanceError


def test_audio_frame_instantiation(sample_audio_frame, sample_session_id, sample_stream_id):
    assert sample_audio_frame.sample_rate == 16000
    assert sample_audio_frame.n_channels == 1
    assert sample_audio_frame.sequence_number == 0
    assert sample_audio_frame.start_sample_index == 0
    assert sample_audio_frame.session_id == sample_session_id
    assert sample_audio_frame.stream_id == sample_stream_id
    assert not sample_audio_frame.discontinuity
    assert sample_audio_frame.discontinuity_reason is None


def test_audio_frame_immutability(sample_audio_frame):
    with pytest.raises(FrozenInstanceError):
        sample_audio_frame.sequence_number = 1


def test_audio_frame_dimensions(sample_session_id, sample_stream_id):
    # 3D array should raise an assertion error
    samples_3d = np.zeros((2, 2, 100), dtype=np.float32)
    with pytest.raises(AssertionError, match="samples must be 1D or 2D"):
        AudioFrame(
            samples=samples_3d,
            sample_rate=16000,
            n_channels=2,
            sequence_number=0,
            start_sample_index=0,
            capture_timestamp_ns=0,
            session_relative_timestamp_ms=0,
            discontinuity=False,
            discontinuity_reason=None,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )


def test_audio_frame_channel_mismatch(sample_session_id, sample_stream_id):
    # 2D array with 2 channels, but n_channels=1
    samples_2d = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(AssertionError):
        AudioFrame(
            samples=samples_2d,
            sample_rate=16000,
            n_channels=1,
            sequence_number=0,
            start_sample_index=0,
            capture_timestamp_ns=0,
            session_relative_timestamp_ms=0,
            discontinuity=False,
            discontinuity_reason=None,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )


def test_audio_frame_invalid_values(mono_audio_samples, sample_session_id, sample_stream_id):
    # Non-positive sample_rate
    with pytest.raises(AssertionError, match="sample_rate must be positive"):
        AudioFrame(
            samples=mono_audio_samples,
            sample_rate=0,
            n_channels=1,
            sequence_number=0,
            start_sample_index=0,
            capture_timestamp_ns=0,
            session_relative_timestamp_ms=0,
            discontinuity=False,
            discontinuity_reason=None,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )

    # Negative sequence_number
    with pytest.raises(AssertionError, match="sequence_number must be non-negative"):
        AudioFrame(
            samples=mono_audio_samples,
            sample_rate=16000,
            n_channels=1,
            sequence_number=-1,
            start_sample_index=0,
            capture_timestamp_ns=0,
            session_relative_timestamp_ms=0,
            discontinuity=False,
            discontinuity_reason=None,
            session_id=sample_session_id,
            stream_id=sample_stream_id,
        )


def test_audio_frame_computed_properties(sample_audio_frame):
    # 1 second of audio at 16000 Hz = 16000 samples
    assert sample_audio_frame.n_samples == 16000
    assert sample_audio_frame.duration_seconds == 1.0
    assert sample_audio_frame.end_sample_index == 16000

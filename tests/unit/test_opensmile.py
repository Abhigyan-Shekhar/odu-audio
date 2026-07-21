"""
Unit tests for the openSMILE eGeMAPS feature extractor.
"""

import time
from unittest.mock import MagicMock

import numpy as np

from src.audio_pipeline.features.opensmile_extractor import OpenSmileExtractor


def test_opensmile_basic():
    """Verify feature extraction of 88 functionals on clean speech-like signal."""
    extractor = OpenSmileExtractor()
    assert "opensmile" in extractor.get_version()
    assert extractor.get_model_hash() is not None

    # 2 seconds of 16kHz sine wave
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    features = extractor.extract(signal, 16000)

    assert features is not None
    assert len(features) == 88
    assert all(isinstance(f, (float, np.float32, np.float64)) for f in features)


def test_opensmile_short_audio():
    """Verify handling of audio signals that are empty or invalid."""
    extractor = OpenSmileExtractor()
    # Empty audio
    features_empty = extractor.extract(np.array([], dtype=np.float32), 16000)
    assert features_empty is None

    # Invalid dimension
    features_2d = extractor.extract(np.zeros((2, 1000), dtype=np.float32), 16000)
    assert features_2d is None


def test_opensmile_silent_audio():
    """Verify openSMILE behaves correctly on pure silence."""
    extractor = OpenSmileExtractor()
    signal = np.zeros(32000, dtype=np.float32)
    features = extractor.extract(signal, 16000)
    assert features is not None
    assert len(features) == 88


def test_opensmile_timeout():
    """Verify openSMILE handles timeouts gracefully via ThreadPoolExecutor."""
    extractor = OpenSmileExtractor(timeout_seconds=0.05)

    def slow_process_signal(*args, **kwargs):
        time.sleep(0.2)
        return MagicMock()

    # Mock process_signal to simulate long run
    extractor._smile.process_signal = slow_process_signal
    features = extractor.extract(np.zeros(32000, dtype=np.float32), 16000)
    assert features is None

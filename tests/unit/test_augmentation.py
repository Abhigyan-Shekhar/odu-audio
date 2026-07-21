"""
Unit tests for augmentation module.
"""

import numpy as np

from src.audio_pipeline.augmentation.codec_augmenter import CodecAugmenter
from src.audio_pipeline.augmentation.noise_augmenter import NoiseAugmenter


def test_noise_augmenter():
    """Test that NoiseAugmenter modifies the waveform appropriately."""
    augmenter = NoiseAugmenter(min_snr_db=5.0, max_snr_db=15.0, random_state=42)

    # Generate a simple sine wave
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    augmented = augmenter(waveform, sample_rate)

    assert augmented.shape == waveform.shape
    assert augmented.dtype == np.float32
    # The augmented signal should not be exactly identical to the original
    assert not np.allclose(waveform, augmented)
    # The augmented signal should be within [-1, 1] range due to clipping
    assert np.max(augmented) <= 1.0
    assert np.min(augmented) >= -1.0


def test_noise_augmenter_empty():
    """Test empty waveform handling."""
    augmenter = NoiseAugmenter()
    waveform = np.array([], dtype=np.float32)
    augmented = augmenter(waveform, 16000)
    assert augmented.size == 0


def test_noise_augmenter_silence():
    """Test silence waveform handling (no noise should be added if signal power is 0)."""
    augmenter = NoiseAugmenter()
    waveform = np.zeros(100, dtype=np.float32)
    augmented = augmenter(waveform, 16000)
    assert np.allclose(waveform, augmented)


def test_codec_augmenter():
    """Test that CodecAugmenter downsamples and quantizes properly."""
    augmenter = CodecAugmenter(target_sample_rate=8000, random_state=42)

    # Generate a simple high-frequency sine wave (should be attenuated/changed)
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate)
    waveform = 0.5 * np.sin(2 * np.pi * 6000 * t).astype(np.float32)

    augmented = augmenter(waveform, sample_rate)

    assert augmented.shape == waveform.shape
    assert augmented.dtype == np.float32
    assert not np.allclose(waveform, augmented)
    assert np.max(augmented) <= 1.0
    assert np.min(augmented) >= -1.0


def test_codec_augmenter_low_sr():
    """Test when sample rate is already lower or equal to target."""
    augmenter = CodecAugmenter(target_sample_rate=16000)
    waveform = np.random.randn(100).astype(np.float32)
    augmented = augmenter(waveform, 8000)

    # Should just return the same waveform if sr <= target
    assert np.allclose(waveform, augmented)


def test_codec_augmenter_empty():
    """Test empty waveform handling."""
    augmenter = CodecAugmenter()
    waveform = np.array([], dtype=np.float32)
    augmented = augmenter(waveform, 16000)
    assert augmented.size == 0

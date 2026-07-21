"""
Simulate codec artifacts for training data.
"""

import numpy as np
import scipy.signal as signal


class CodecAugmenter:
    """Simulate codec artifacts (TRAINING ONLY)."""

    def __init__(self, target_sample_rate: int = 8000, random_state: int = 42):
        """
        Initialize CodecAugmenter.

        Args:
            target_sample_rate: The sample rate to simulate downsampling to (e.g., 8kHz for telephony).
            random_state: Seed for reproducibility.
        """
        self.target_sample_rate = target_sample_rate
        self.rng = np.random.default_rng(random_state)

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply downsampling and simple quantization to simulate codec compression.

        Args:
            waveform: 1D numpy array of audio samples.
            sample_rate: Original sample rate.

        Returns:
            Augmented waveform.
        """
        if waveform.size == 0 or sample_rate <= self.target_sample_rate:
            return waveform

        # 1. Simulate downsampling and upsampling (bandlimiting)
        # We use scipy.signal.resample_poly to avoid introducing severe aliasing,
        # but capture the loss of high frequencies typical of low-bitrate codecs.

        # Downsample
        num_downsampled = int(len(waveform) * self.target_sample_rate / sample_rate)
        downsampled = signal.resample(waveform, num_downsampled)

        # Upsample back to original length
        bandlimited = signal.resample(downsampled, len(waveform))

        # 2. Simulate 8-bit mu-law quantization (simplified)
        # Mu-law compresses dynamic range, we simulate the quantization noise.
        mu = 255.0
        # Compress
        compressed = (
            np.sign(bandlimited) * np.log1p(mu * np.abs(bandlimited)) / np.log1p(mu)
        )
        # Quantize to 8-bit (256 levels)
        quantized = np.round(compressed * 128) / 128
        # Expand
        expanded = np.sign(quantized) * (1 / mu) * ((1 + mu) ** np.abs(quantized) - 1)

        result: np.ndarray = np.clip(expanded, -1.0, 1.0).astype(np.float32)
        return result

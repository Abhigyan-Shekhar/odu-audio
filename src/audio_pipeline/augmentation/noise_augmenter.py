"""
Simulate background noise for training data.
"""

import numpy as np


class NoiseAugmenter:
    """Add room noise (TRAINING ONLY)."""

    def __init__(
        self, min_snr_db: float = 5.0, max_snr_db: float = 20.0, random_state: int = 42
    ):
        """
        Initialize NoiseAugmenter.

        Args:
            min_snr_db: Minimum Signal-to-Noise Ratio in dB.
            max_snr_db: Maximum Signal-to-Noise Ratio in dB.
            random_state: Seed for reproducibility.
        """
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.rng = np.random.default_rng(random_state)

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply additive white noise to the waveform based on a random SNR.

        Args:
            waveform: 1D numpy array of audio samples.
            sample_rate: Sample rate (ignored for white noise, kept for interface consistency).

        Returns:
            Augmented waveform.
        """
        if waveform.size == 0:
            return waveform

        # Calculate signal power
        signal_power = np.mean(waveform**2)
        if signal_power == 0:
            return waveform

        # Choose a random SNR
        snr_db = self.rng.uniform(self.min_snr_db, self.max_snr_db)

        # Calculate required noise power
        # SNR_dB = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate noise
        noise = self.rng.normal(0, np.sqrt(noise_power), size=waveform.shape)

        # Add noise
        augmented = waveform + noise

        # Ensure we don't exceed [-1.0, 1.0] bounds if original was normalized
        # We use soft clipping or hard clipping depending on preference. Here, simple hard clipping:
        return np.clip(augmented, -1.0, 1.0).astype(np.float32)

"""
Audio reader component for decoding audio files.
"""

import os

import numpy as np
import soundfile as sf

from src.audio_pipeline.io.audio_data import AudioData


class AudioReader:
    """
    Read and decode audio files.
    """

    def read(self, path: str) -> AudioData:
        """
        Read and decode an audio file using soundfile.

        Args:
            path: Path to the audio file.

        Returns:
            An AudioData instance containing the samples and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is corrupt or unsupported.
            RuntimeError: If soundfile fails to decode the file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            # Read samples as float32 in [-1.0, 1.0] range
            samples, sample_rate = sf.read(path, dtype="float32")
        except sf.LibsndfileError as e:
            # Distinguish malformed/corrupt files vs generic read errors
            raise ValueError(
                f"Failed to read corrupt/malformed or unsupported audio file: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error while decoding audio file: {e}")

        # Check for NaN or Inf
        if np.isnan(samples).any() or np.isinf(samples).any():
            raise ValueError("Audio data contains NaN or Inf values")

        # Ensure we have some samples
        if len(samples) == 0:
            raise ValueError("Audio file is empty (contains 0 samples)")

        # Transpose multi-channel samples from soundfile shape (n_samples, n_channels)
        # to our target format (n_channels, n_samples)
        if samples.ndim == 2:
            n_samples, n_channels = samples.shape
            samples = samples.T
        else:
            n_channels = 1
            n_samples = len(samples)

        duration_seconds = n_samples / sample_rate

        return AudioData(
            samples=samples,
            sample_rate=sample_rate,
            n_channels=n_channels,
            duration_seconds=duration_seconds,
        )

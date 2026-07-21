"""
Audio resampler component using librosa.
"""

from typing import cast

import librosa
import numpy as np


class Resampler:
    """
    High-quality resampling with anti-aliasing.
    """

    def resample(self, audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate using librosa.

        Args:
            audio: Input audio array. Shape (n_samples,) or (n_channels, n_samples).
            source_sr: Source sample rate in Hz.
            target_sr: Target sample rate in Hz.

        Returns:
            Resampled numpy array.
        """
        if source_sr == target_sr:
            return cast(np.ndarray, audio.copy())

        # librosa.resample accepts both 1D and 2D arrays (shape (n_channels, n_samples))
        return cast(
            np.ndarray,
            librosa.resample(
                y=audio,
                orig_sr=source_sr,
                target_sr=target_sr,
                res_type="kaiser_best",
            ),
        )

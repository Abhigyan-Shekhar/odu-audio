"""
Stateful streaming resampler to avoid boundary filter click artifacts.
"""

from typing import cast

import librosa
import numpy as np


class StreamingResampler:
    """
    Stateful resampler for streaming audio chunk-by-chunk.
    Retains overlap history to prevent filter boundary artifacts.
    """

    def __init__(self, source_sr: int, target_sr: int, history_len: int = 512):
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.history_len = history_len

        self._ratio = target_sr / source_sr
        self._history = np.zeros(self.history_len, dtype=np.float32)
        self._is_first = True

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a single 1D chunk of samples, returning the resampled version.

        Args:
            chunk: 1D numpy array of mono samples.

        Returns:
            Resampled 1D numpy array of mono samples.
        """
        if chunk.ndim != 1:
            raise ValueError("StreamingResampler expects 1D mono audio arrays.")

        if self.source_sr == self.target_sr:
            return cast(np.ndarray, chunk.copy())

        # If it's the first chunk, we can't prepend history yet.
        # We initialize history with the start of the chunk or keep it as zeros.
        if self._is_first:
            self._history = np.zeros(self.history_len, dtype=np.float32)
            self._is_first = False

        # Prepend history
        concat = np.concatenate([self._history, chunk])

        # Resample the combined array
        resampled_concat = cast(
            np.ndarray,
            librosa.resample(
                y=concat,
                orig_sr=self.source_sr,
                target_sr=self.target_sr,
                res_type="kaiser_best",
            ),
        )

        # Slice off the part corresponding to the prepended history
        # M_resampled is the exact number of resampled samples for the history length
        M_resampled = int(self.history_len * self._ratio)
        resampled_chunk = resampled_concat[M_resampled:]

        # Update history with the end of the current chunk
        if len(chunk) >= self.history_len:
            self._history = chunk[-self.history_len :].copy()
        else:
            # Shift history left and copy current chunk to the end
            self._history = np.concatenate([self._history[len(chunk) :], chunk])

        return resampled_chunk

    def reset(self) -> None:
        """Reset the resampler state."""
        self._history = np.zeros(self.history_len, dtype=np.float32)
        self._is_first = True

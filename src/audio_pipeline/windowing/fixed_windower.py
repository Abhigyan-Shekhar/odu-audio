"""
Fixed windower component for slicing audio data into windows.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.audio_pipeline.io.audio_data import AudioData


@dataclass
class AudioWindow:
    """
    Represent a slice of audio data.
    """

    samples: np.ndarray
    """Audio samples in this window."""

    start_sample: int
    """Start sample index in the source audio (0-indexed)."""

    end_sample: int
    """End sample index in the source audio (exclusive)."""

    start_time_ms: int
    """Start time in milliseconds from session start."""

    end_time_ms: int
    """End time in milliseconds from session start."""

    sequence_number: int
    """Monotonic sequence number (0-indexed)."""


class FixedWindower:
    """
    Slices mono audio data into fixed-size windows with a configurable hop size.
    """

    def __init__(self, window_seconds: float, hop_seconds: float):
        """
        Initialize the windower.

        Args:
            window_seconds: Width of each window in seconds.
            hop_seconds: Step size between windows in seconds.
        """
        assert window_seconds > 0, "window_seconds must be positive"
        assert hop_seconds > 0, "hop_seconds must be positive"
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds

    def window(self, audio: AudioData) -> Iterator[AudioWindow]:
        """
        Generate fixed-size windows from the decoded AudioData.
        Expects mono (1D) samples in the AudioData object.

        Args:
            audio: The input AudioData containing a 1D samples array.

        Returns:
            An iterator yielding AudioWindow instances.
        """
        # Ensure audio samples are mono (1D)
        samples = audio.samples
        if samples.ndim != 1:
            raise ValueError(
                f"FixedWindower expects 1D mono audio samples, got shape {samples.shape}"
            )

        sample_rate = audio.sample_rate
        total_samples = len(samples)

        window_samples = int(self.window_seconds * sample_rate)
        hop_samples = int(self.hop_seconds * sample_rate)

        if total_samples == 0:
            return

        sequence_number = 0
        start_sample = 0

        while start_sample < total_samples:
            end_sample = start_sample + window_samples

            # If this is a partial window at the end, keep it but clamp to total_samples
            is_last = False
            if end_sample >= total_samples:
                end_sample = total_samples
                is_last = True

            window_data = samples[start_sample:end_sample]

            # Calculate timestamps in milliseconds
            start_time_ms = int((start_sample / sample_rate) * 1000)
            end_time_ms = int((end_sample / sample_rate) * 1000)

            # Yield if the window actually has samples
            if len(window_data) > 0:
                yield AudioWindow(
                    samples=window_data,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    sequence_number=sequence_number,
                )
                sequence_number += 1

            # If we reached the end or it's a zero-hop (asserted positive anyway)
            if (
                is_last
                or start_sample + hop_samples >= total_samples
                or hop_samples == 0
            ):
                break

            start_sample += hop_samples

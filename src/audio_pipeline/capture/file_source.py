"""
File-based audio source playing back WAV files chunk-by-chunk in simulated real-time.
"""

import time
from typing import Optional

import soundfile as sf

from src.audio_pipeline.capture.audio_chunk import AudioChunk
from src.audio_pipeline.capture.audio_source import AudioSource


class FileSource(AudioSource):
    """
    Audio source that reads from a WAV file and simulates real-time capture streaming.
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1600,  # 100 ms chunks at 16kHz
        simulate_real_time: bool = True,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.simulate_real_time = simulate_real_time

        try:
            self._snd_file = sf.SoundFile(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open audio file {file_path}: {e}")

        self._sample_rate = int(self._snd_file.samplerate)
        self.channels = self._snd_file.channels

        self._sequence_number = 0
        self._next_discontinuity = False
        self._next_discontinuity_reason: Optional[str] = None
        self._last_read_time: Optional[float] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def inject_discontinuity(self, reason: str = "BUFFER_OVERFLOW") -> None:
        """Inject a simulated discontinuity in the next chunk read."""
        self._next_discontinuity = True
        self._next_discontinuity_reason = reason

    def read_chunk(self) -> AudioChunk:
        """
        Read the next chunk of samples from the file.
        Simulates real-time speed if simulate_real_time is True.
        """
        if self._snd_file is None:
            raise RuntimeError("FileSource is closed.")

        # Read frames from soundfile
        # soundfile returns shape (chunk_size, channels) for multi-channel
        samples = self._snd_file.read(self.chunk_size, dtype="float32")

        if len(samples) == 0:
            raise EOFError("End of audio file reached.")

        # Transpose to channel-first shape (n_channels, n_samples)
        if self.channels > 1:
            samples = samples.T
        else:
            # Mono 1D array
            if samples.ndim > 1:
                samples = samples.squeeze()

        # Simulate real-time pacing
        n_samples = samples.shape[-1]
        chunk_duration = n_samples / self.sample_rate

        if self.simulate_real_time:
            now = time.perf_counter()
            if self._last_read_time is not None:
                elapsed = now - self._last_read_time
                sleep_time = chunk_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._last_read_time = time.perf_counter()

        disc = self._next_discontinuity
        reason = self._next_discontinuity_reason

        # Reset discontinuity flags
        self._next_discontinuity = False
        self._next_discontinuity_reason = None

        chunk = AudioChunk(
            samples=samples,
            sequence_number=self._sequence_number,
            capture_timestamp_ns=time.time_ns(),
            sample_rate=self.sample_rate,
            discontinuity=disc,
            discontinuity_reason=reason,
        )
        self._sequence_number += 1
        return chunk

    def close(self) -> None:
        """Close the file handle."""
        if hasattr(self, "_snd_file") and self._snd_file is not None:
            try:
                self._snd_file.close()
            except Exception:
                pass
            self._snd_file = None

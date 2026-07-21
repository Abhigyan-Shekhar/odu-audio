"""
Real-time microphone input source using sounddevice.
"""

import queue
import time
from typing import Any, Optional

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from src.audio_pipeline.capture.audio_chunk import AudioChunk
from src.audio_pipeline.capture.audio_source import AudioSource


class MicrophoneSource(AudioSource):
    """
    Audio source that captures raw input from the system microphone.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1600,  # 100 ms chunks at 16kHz
        channels: int = 1,
        device_index: Optional[int] = None,
    ):
        self._sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index

        self._queue: queue.Queue = queue.Queue()
        self._sequence_number = 0
        self._discontinuity_flag = False
        self._discontinuity_reason: Optional[str] = None

        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Please ensure sounddevice is installed "
                "and PortAudio is available on your system."
            )

        # Start input stream using sounddevice
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_index,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            raise RuntimeError(f"Failed to open microphone input stream: {e}")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Background callback from sounddevice thread."""
        # status check for buffer overflow/underflow
        discontinuity = False
        reason = None
        if status:
            discontinuity = True
            if status.input_overflow:
                reason = "BUFFER_OVERFLOW"
            elif status.input_underflow:
                reason = "BUFFER_UNDERFLOW"
            else:
                reason = "DEVICE_ERROR"

        # Transpose to channel-first shape (n_channels, n_samples)
        # indata has shape (frames, channels)
        samples = indata.T.copy()
        if self.channels == 1:
            samples = samples[0]  # mono 1D array

        self._queue.put((samples, discontinuity, reason))

    def read_chunk(self) -> AudioChunk:
        """
        Block until next chunk is available from the callback queue.
        """
        if not self._stream.active:
            raise RuntimeError("Microphone source stream is not active/closed.")

        try:
            # Block for a reasonable amount of time (e.g. 5.0s max timeout to prevent infinite hangs)
            samples, disc, reason = self._queue.get(timeout=5.0)
        except queue.Empty:
            raise TimeoutError("Microphone input callback timeout.")

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
        """Stop and close the sounddevice stream."""
        if hasattr(self, "_stream") and self._stream is not None:
            try:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

"""
Thread-safe circular ring buffer for sliding window audio operations.
"""

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RingBuffer:
    """
    Thread-safe circular buffer for 1D audio samples.
    Supports writing new samples and reading/peeking overlapping sliding windows.
    """

    def __init__(self, capacity_seconds: float = 60.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.capacity = int(capacity_seconds * sample_rate)

        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write_idx = 0
        self._read_idx = 0
        self._size = 0

        self._lock = threading.Lock()
        self._total_written = 0
        self._discontinuity_flag = False
        self._discontinuity_reason: Optional[str] = None

    def write(self, chunk: np.ndarray) -> int:
        """
        Write a 1D numpy array of samples into the ring buffer.
        If the buffer overflows, the oldest samples are overwritten.

        Returns:
            The total number of samples written to this buffer since initialization.
        """
        if chunk.ndim != 1:
            raise ValueError("RingBuffer only accepts 1D mono audio arrays.")

        n_samples = len(chunk)
        if n_samples == 0:
            return self._total_written

        with self._lock:
            if n_samples > self.capacity:
                # Chunk is larger than buffer capacity; keep only the last capacity samples
                chunk = chunk[-self.capacity :]
                n_samples = self.capacity

            # Check for overflow
            if self._size + n_samples > self.capacity:
                overflow_samples = (self._size + n_samples) - self.capacity
                logger.warning(
                    f"RingBuffer overflow: Overwriting {overflow_samples} oldest samples."
                )
                # Discard the oldest samples
                self._read_idx = (self._read_idx + overflow_samples) % self.capacity
                self._size -= overflow_samples

            # Write samples
            first_write_len = min(n_samples, self.capacity - self._write_idx)
            self._buffer[self._write_idx : self._write_idx + first_write_len] = chunk[
                :first_write_len
            ]

            if first_write_len < n_samples:
                # Wrap around and write remainder
                second_write_len = n_samples - first_write_len
                self._buffer[0:second_write_len] = chunk[first_write_len:]
                self._write_idx = second_write_len
            else:
                self._write_idx = (self._write_idx + first_write_len) % self.capacity

            self._size += n_samples
            self._total_written += n_samples
            return self._total_written

    def peek(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Read the oldest n_samples from the buffer without removing them.

        Returns:
            A numpy array of shape (n_samples,) or None if buffer does not contain enough samples.
        """
        with self._lock:
            if self._size < n_samples:
                return None

            first_read_len = min(n_samples, self.capacity - self._read_idx)
            out = np.empty(n_samples, dtype=np.float32)
            out[:first_read_len] = self._buffer[
                self._read_idx : self._read_idx + first_read_len
            ]

            if first_read_len < n_samples:
                second_read_len = n_samples - first_read_len
                out[first_read_len:] = self._buffer[0:second_read_len]

            return out

    def advance(self, n_samples: int) -> None:
        """
        Consume (remove) the oldest n_samples from the buffer.
        """
        with self._lock:
            if n_samples > self._size:
                n_samples = self._size
            self._read_idx = (self._read_idx + n_samples) % self.capacity
            self._size -= n_samples

    def clear(self) -> None:
        """
        Clear all samples from the buffer.
        """
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0
            self._size = 0
            self._discontinuity_flag = False
            self._discontinuity_reason = None

    def get_available_samples(self) -> int:
        """Return the number of unconsumed samples currently in the buffer."""
        with self._lock:
            return self._size

    def mark_discontinuity(self, reason: str = "BUFFER_OVERFLOW") -> None:
        """Flag a discontinuity. The processing logic will check this flag."""
        with self._lock:
            self._discontinuity_flag = True
            self._discontinuity_reason = reason

    def check_and_reset_discontinuity(self) -> tuple[bool, Optional[str]]:
        """Check the discontinuity status and reset the flag atomically."""
        with self._lock:
            flag = self._discontinuity_flag
            reason = self._discontinuity_reason
            self._discontinuity_flag = False
            self._discontinuity_reason = None
            return flag, reason

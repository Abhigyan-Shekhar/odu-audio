"""
Thread-safe bounded queue with backpressure support.
"""

import queue
from typing import Generic, Optional, TypeVar, cast

T = TypeVar("T")


class BoundedQueue(Generic[T]):
    """
    Thread-safe queue wrapper with strict bounds and timeout operations
    for real-time pipeline backpressure control.
    """

    def __init__(self, maxsize: int = 100):
        self._maxsize = maxsize
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """
        Put an item in the queue.
        Blocks for up to timeout seconds if the queue is full.

        Returns:
            True if item was queued successfully, False if timed out (backpressure).
        """
        try:
            self._queue.put(item, block=True, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Get an item from the queue.
        Blocks for up to timeout seconds if the queue is empty.

        Returns:
            The item if fetched successfully, or None if timed out.
        """
        try:
            return cast(Optional[T], self._queue.get(block=True, timeout=timeout))
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return self._queue.qsize()

    def is_full(self) -> bool:
        """Return True if the queue is full."""
        return self._queue.full()

    def clear(self) -> None:
        """Discard all items in the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

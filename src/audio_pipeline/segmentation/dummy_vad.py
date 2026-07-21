"""
Mock/Dummy VAD implementation for testing and isolated verification.
"""

from typing import List, Optional

import numpy as np

from src.audio_pipeline.segmentation.vad_interface import VADInterface


class DummyVAD(VADInterface):
    """
    Mock VAD returning configured speech probabilities.
    """

    def __init__(
        self, default_prob: float = 0.0, pattern: Optional[List[float]] = None
    ):
        self.default_prob = default_prob
        self.pattern = pattern
        self._idx = 0

    def process_chunk(self, audio: np.ndarray) -> float:
        """Return next probability in the pattern, or the default probability."""
        if self.pattern is not None and len(self.pattern) > 0:
            prob = self.pattern[self._idx % len(self.pattern)]
            self._idx += 1
            return prob
        return self.default_prob

    def get_version(self) -> str:
        return "dummy_vad_v1"

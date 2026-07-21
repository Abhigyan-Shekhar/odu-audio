"""
Abstract interface for Voice Activity Detection (VAD).
"""

from abc import ABC, abstractmethod

import numpy as np


class VADInterface(ABC):
    """
    Abstract interface for interchangeable VAD models.
    """

    @abstractmethod
    def process_chunk(self, audio: np.ndarray) -> float:
        """
        Process a single 1D numpy audio array (16 kHz mono) and return a speech probability.

        Args:
            audio: 1D numpy float32 array.

        Returns:
            Speech probability float in range [0.0, 1.0].
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """
        Return the model name/version.
        """
        pass

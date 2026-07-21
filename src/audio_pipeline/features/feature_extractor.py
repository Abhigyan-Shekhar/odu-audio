"""
Abstract base class for feature extractors.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class FeatureExtractor(ABC):
    """
    Abstract feature extractor base class.
    """

    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> Optional[Any]:
        """
        Extract features from a 1D audio signal.

        Args:
            audio: 1D float32 numpy array.
            sr: sample rate of the audio.

        Returns:
            Features array/dict or None if extraction failed.
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """
        Return the extractor/model version identifier.
        """
        pass

    @abstractmethod
    def get_model_hash(self) -> Optional[str]:
        """
        Return the file/config hash of the model if applicable.
        """
        pass

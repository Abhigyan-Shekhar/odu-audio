"""
openSMILE eGeMAPS feature extractor with timeout protection.
"""

import concurrent.futures
import hashlib
import logging
from typing import Optional

import numpy as np
import opensmile

from src.audio_pipeline.features.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class OpenSmileExtractor(FeatureExtractor):
    """
    Extracts 88 eGeMAPSv02 functionals from audio signals using openSMILE.
    """

    def __init__(
        self,
        feature_set: str = "eGeMAPSv02",
        feature_level: str = "Functionals",
        timeout_seconds: float = 5.0,
    ):
        self.feature_set_name = feature_set
        self.feature_level_name = feature_level
        self.timeout_seconds = timeout_seconds

        # Map to openSMILE enum values
        try:
            fs = getattr(opensmile.FeatureSet, feature_set)
            fl = getattr(opensmile.FeatureLevel, feature_level)
            self._smile = opensmile.Smile(feature_set=fs, feature_level=fl)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize openSMILE with {feature_set}/{feature_level}: {e}"
            )

        # Compute hash of configuration
        cfg_str = f"{feature_set}_{feature_level}"
        self._config_hash = hashlib.sha256(cfg_str.encode()).hexdigest()[:8]

    def extract(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Extract features with timeout protection.
        """
        if audio.ndim != 1:
            logger.error("openSMILE expects 1D mono audio arrays.")
            return None

        if len(audio) == 0:
            logger.warning("Empty audio passed to openSMILE extractor.")
            return None

        # Execute extraction in a separate thread to support timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._smile.process_signal, audio, sr)
            try:
                df = future.result(timeout=self.timeout_seconds)
                if df is None or df.empty:
                    logger.warning("openSMILE returned empty features.")
                    return None
                features = np.asarray(df.to_numpy().flatten(), dtype=np.float32)
                if len(features) != 88:
                    logger.warning(
                        f"openSMILE output length mismatch: expected 88, got {len(features)}"
                    )
                    return None
                return features
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"openSMILE extraction timed out after {self.timeout_seconds}s"
                )
                return None
            except Exception as e:
                logger.error(f"openSMILE extraction failed: {e}")
                return None

    def get_version(self) -> str:
        return f"opensmile_{opensmile.__version__}"

    def get_model_hash(self) -> Optional[str]:
        return self._config_hash

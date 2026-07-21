"""
YAMNet distress event detector using ONNX Runtime.
"""

import concurrent.futures
import csv
import hashlib
import logging
import os
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from src.audio_pipeline.features.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class YAMNetDetector(FeatureExtractor):
    """
    Detects AudioSet events (e.g. distress sounds) in raw audio using YAMNet in ONNX format.
    """

    def __init__(
        self,
        target_events: Optional[List[str]] = None,
        min_probability: float = 0.1,
        timeout_seconds: float = 5.0,
    ):
        self.target_events = target_events or [
            "Scream",
            "Crying, sobbing",
            "Yell",
            "Gasp",
            "Groan",
            "Whimper",
            "Wheeze",
        ]
        self.min_probability = min_probability
        self.timeout_seconds = timeout_seconds

        # Configure cache directories and paths
        cache_dir = os.path.expanduser("~/.cache/yamnet/")
        os.makedirs(cache_dir, exist_ok=True)
        self.model_path = os.path.join(cache_dir, "yamnet.onnx")
        self.class_map_path = os.path.join(cache_dir, "yamnet_class_map.csv")

        # Download model and class map if they are missing
        self._ensure_resources()

        # Load class map
        self._classes: Dict[int, str] = {}
        with open(self.class_map_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    self._classes[int(row[0])] = row[2]

        # Map target events to their indices (handling common aliases)
        aliases = {
            "Scream": "Screaming",
        }
        self._event_indices: Dict[str, int] = {}
        for idx, name in self._classes.items():
            for target in self.target_events:
                mapped_target = aliases.get(target, target)
                if name == mapped_target:
                    self._event_indices[target] = idx

        # Initialize ONNX inference session on CPU
        try:
            self._session = ort.InferenceSession(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YAMNet ONNX model: {e}")

        # Compute hash of model file for reproducibility
        with open(self.model_path, "rb") as f:
            self._model_hash = hashlib.sha256(f.read()).hexdigest()[:8]

    def _ensure_resources(self) -> None:
        """Download model and class map files from public repositories if not present."""
        if not os.path.exists(self.model_path):
            logger.info("Downloading YAMNet ONNX model...")
            url = "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx"
            urllib.request.urlretrieve(url, self.model_path)

        if not os.path.exists(self.class_map_path):
            logger.info("Downloading YAMNet class map...")
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            urllib.request.urlretrieve(url, self.class_map_path)

    def extract(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
        """
        Extract target event scores from audio signal.
        """
        if audio.ndim != 1:
            logger.error("YAMNet expects 1D mono audio arrays.")
            return None

        if len(audio) == 0:
            return {event: 0.0 for event in self.target_events}

        # Normalize sample rate if needed (YAMNet expects 16 kHz)
        if sr != 16000:
            logger.warning(
                f"YAMNet expects 16 kHz sample rate, got {sr} Hz. Results may be degraded."
            )

        # Ensure float32 format
        waveform = audio.astype(np.float32)

        # Run inference with timeout protection
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._session.run, None, {"waveform": waveform})
            try:
                outputs = future.result(timeout=self.timeout_seconds)
                scores = outputs[0]  # shape [N, 521]

                if scores.ndim != 2 or scores.shape[1] != 521:
                    logger.error(f"Unexpected YAMNet output shape: {scores.shape}")
                    return None

                results = {}
                for name in self.target_events:
                    if name in self._event_indices:
                        idx = self._event_indices[name]
                        # Compute max score across all frames in this window
                        max_val = float(np.max(scores[:, idx]))
                        results[name] = (
                            max_val if max_val >= self.min_probability else 0.0
                        )
                    else:
                        results[name] = 0.0
                return results

            except concurrent.futures.TimeoutError:
                logger.error(
                    f"YAMNet extraction timed out after {self.timeout_seconds}s"
                )
                return None
            except Exception as e:
                logger.error(f"YAMNet extraction failed: {e}")
                return None

    def get_version(self) -> str:
        return "yamnet_onnx_v1"

    def get_model_hash(self) -> Optional[str]:
        return self._model_hash

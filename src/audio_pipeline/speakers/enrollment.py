"""
Session enrollment module for patient speaker identification.

Enrolls a reference speaker embedding at session start and computes
cosine similarity scores for subsequent attribution decisions.
All embeddings are kept in-memory only — never persisted to disk.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SessionEnrollment:
    """
    Manages patient enrollment within a single recording session.

    Computes a speaker embedding from a short enrollment utterance and
    provides similarity scoring for subsequent speaker turns.

    Privacy note: embeddings are never written to disk.
    """

    def __init__(self, confidence_threshold: float = 0.70):
        """
        Args:
            confidence_threshold: Cosine similarity above which a speaker
                                  turn is considered a confirmed match.
        """
        self.confidence_threshold = confidence_threshold
        self._reference_embedding: Optional[np.ndarray] = None
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load the speaker embedding model."""
        if self._embedding_model is None:
            from pyannote.audio import Inference, Model

            # SpeechBrain ECAPA-TDNN via pyannote pretrained embeddings.
            # This model is free to download without a HF token.
            model = Model.from_pretrained("pyannote/embedding")
            self._embedding_model = Inference(model, window="whole")
        return self._embedding_model

    def enroll(self, audio: np.ndarray, sr: int) -> bool:
        """
        Compute and store a reference embedding from an enrollment utterance.

        Args:
            audio: 1D float32 mono audio of the enrollment utterance.
            sr:    Sample rate (expected 16000 Hz).

        Returns:
            True on successful enrollment, False on failure.
        """
        if len(audio) < sr * 1:
            logger.warning(
                "Enrollment audio is shorter than 1 second; quality may be poor."
            )

        try:
            import torch

            inference = self._get_embedding_model()
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            input_dict = {"waveform": waveform, "sample_rate": sr}
            embedding = inference(input_dict)
            self._reference_embedding = np.array(embedding).flatten()
            logger.info("Patient enrollment completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            return False

    def enroll_from_embedding(self, embedding: np.ndarray) -> None:
        """
        Directly set the reference embedding (for testing / pre-computed embeddings).

        Args:
            embedding: Pre-computed speaker embedding vector.
        """
        self._reference_embedding = np.array(embedding, dtype=np.float64).flatten()

    def compare(self, speaker_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between a speaker turn embedding and the
        enrolled patient reference.

        Args:
            speaker_embedding: Speaker embedding from diarized turn.

        Returns:
            Cosine similarity in [0, 1]. Returns 0.0 if not enrolled.
        """
        if self._reference_embedding is None:
            logger.warning("No patient enrollment found; returning 0.0 similarity.")
            return 0.0

        sim = _cosine_similarity(
            self._reference_embedding,
            np.array(speaker_embedding, dtype=np.float64).flatten(),
        )
        # Clamp to [0, 1] since cosine similarity can be negative for dissimilar vectors
        return max(0.0, float(sim))

    def is_enrolled(self) -> bool:
        """Return True if a reference embedding has been recorded."""
        return self._reference_embedding is not None

    def clear(self) -> None:
        """
        Wipe the in-memory enrollment embedding (call at session end).
        """
        self._reference_embedding = None
        logger.info("Session enrollment cleared.")

    def get_embedding_for_speaker(
        self, audio: np.ndarray, sr: int
    ) -> Optional[np.ndarray]:
        """
        Compute a speaker embedding for an arbitrary audio segment.
        Used internally by the attributor to compare speaker turns.

        Args:
            audio: 1D float32 mono audio.
            sr:    Sample rate.

        Returns:
            Embedding vector or None on failure.
        """
        try:
            import torch

            inference = self._get_embedding_model()
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            input_dict = {"waveform": waveform, "sample_rate": sr}
            embedding = inference(input_dict)
            return np.array(embedding, dtype=np.float64).flatten()
        except Exception as e:
            logger.error(f"Speaker embedding extraction failed: {e}")
            return None

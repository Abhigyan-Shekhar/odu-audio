"""
Silero VAD model integration using PyTorch Hub.
"""

import numpy as np
import torch

from src.audio_pipeline.segmentation.vad_interface import VADInterface


class SileroVAD(VADInterface):
    """
    Silero VAD implementation executing on CPU.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._sample_rate = 16000

        # Load Silero VAD model using PyTorch Hub (defaulting to CPU)
        try:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            # Put model in evaluation mode
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {e}")

    def process_chunk(self, audio: np.ndarray) -> float:
        """
        Run inference on the 1D audio chunk.
        """
        if audio.ndim != 1:
            raise ValueError("SileroVAD expects 1D mono audio arrays.")

        if len(audio) == 0:
            return 0.0

        # Convert numpy array to torch tensor
        tensor = torch.from_numpy(audio).float()

        # Run inference (always on CPU for edge compatibility)
        with torch.no_grad():
            prob = self._model(tensor, self._sample_rate).item()

        return float(prob)

    def get_version(self) -> str:
        return "silero_vad_v4"

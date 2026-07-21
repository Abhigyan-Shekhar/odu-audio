"""
Offline integration tests for VAD, including clean speech, silence, and noisy signals.
"""

from unittest.mock import MagicMock

import numpy as np
import torch

from src.audio_pipeline.segmentation.silero_vad import SileroVAD


def test_vad_on_clean_speech():
    vad = SileroVAD()
    # Mock the internal PyTorch model to return 0.9 for speech
    vad._model = MagicMock(return_value=torch.tensor(0.9))

    signal = np.sin(np.linspace(0, 10, 512)).astype(np.float32)
    prob = vad.process_chunk(signal)
    assert prob > 0.5


def test_vad_on_silence():
    vad = SileroVAD()
    # Mock the internal PyTorch model to return 0.05 for silence
    vad._model = MagicMock(return_value=torch.tensor(0.05))

    signal = np.zeros(512, dtype=np.float32)
    prob = vad.process_chunk(signal)
    assert prob < 0.3


def test_vad_on_noisy_speech():
    vad = SileroVAD()
    # Mock the internal PyTorch model to return 0.8 for noisy speech
    vad._model = MagicMock(return_value=torch.tensor(0.8))

    signal = np.random.normal(0, 0.05, 512).astype(np.float32)
    prob = vad.process_chunk(signal)
    assert prob > 0.4

"""
Integration tests for YAMNet audio event detection.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from src.audio_pipeline.capture.file_source import FileSource
from src.audio_pipeline.features.yamnet_detector import YAMNetDetector
from src.audio_pipeline.runtime.stream_processor import StreamProcessor
from src.audio_pipeline.segmentation.dummy_vad import DummyVAD


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def test_yamnet_on_scream():
    """Verify YAMNetDetector correctly maps Audioset indices to class display names."""
    detector = YAMNetDetector()
    assert detector.get_version() == "yamnet_onnx_v1"
    assert detector.get_model_hash() is not None

    # Retrieve scream index
    scream_idx = detector._event_indices["Scream"]
    crying_idx = detector._event_indices["Crying, sobbing"]

    # Mock output scores: 1 frame, 521 classes
    mock_scores = np.zeros((1, 521), dtype=np.float32)
    mock_scores[0, scream_idx] = 0.85
    mock_scores[0, crying_idx] = 0.40

    detector._session.run = MagicMock(return_value=[mock_scores, None, None])

    res = detector.extract(np.zeros(16000, dtype=np.float32), 16000)
    assert res is not None
    assert res["Scream"] == pytest.approx(0.85)
    assert res["Crying, sobbing"] == pytest.approx(0.40)
    assert res["Yell"] == 0.0  # below default min_probability


def test_yamnet_on_normal_speech():
    """Verify YAMNetDetector returns low/zero scores when no events exceed threshold."""
    detector = YAMNetDetector(min_probability=0.5)

    # All classes have low probability
    mock_scores = np.ones((1, 521), dtype=np.float32) * 0.05
    detector._session.run = MagicMock(return_value=[mock_scores, None, None])

    res = detector.extract(np.zeros(16000, dtype=np.float32), 16000)
    assert res is not None
    assert all(val == 0.0 for val in res.values())


def test_yamnet_parallel_to_vad(temp_dir):
    """Verify YAMNet event extraction runs even when VAD says no speech."""
    wav_path = os.path.join(temp_dir, "input.wav")
    # 4.0s of zero audio
    sf.write(wav_path, np.zeros(64000, dtype=np.float32), 16000)

    # Mock VAD to always return silence (prob = 0.05)
    dummy_vad = DummyVAD(default_prob=0.05)

    # Mock YAMNet detector to return high scream score (e.g. 0.9)
    mock_yamnet = MagicMock()
    mock_yamnet.extract.return_value = {
        "Scream": 0.9,
        "Crying, sobbing": 0.0,
        "Yell": 0.0,
        "Gasp": 0.0,
        "Groan": 0.0,
        "Whimper": 0.0,
        "Wheeze": 0.0,
    }
    mock_yamnet.get_version.return_value = "mock_yamnet"
    mock_yamnet.get_model_hash.return_value = "hash123"

    # Configure eGeMAPS with min_voiced_ratio = 0.5 (which will fail since voiced_ratio is 0.0)
    config = {
        "features": {
            "egemaps": {
                "enable": True,
                "minimum_voiced_ratio": 0.5,
                "on_failure": "emit_none",
            },
            "yamnet": {
                "enable": True,
            },
        }
    }

    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    processor = StreamProcessor(
        source, config=config, vad=dummy_vad, yamnet_detector=mock_yamnet
    )

    processor.start()
    records = list(processor.stream())
    processor.stop()

    assert len(records) > 0
    for record in records:
        # eGeMAPS must be skipped (None) due to voiced ratio failure
        assert record.egemaps is None
        # YAMNet scores must still be populated in parallel
        assert record.yamnet_event_scores["Scream"] == 0.9
        assert record.extractor_versions["yamnet"] == "mock_yamnet"

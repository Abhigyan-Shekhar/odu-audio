"""
Integration tests for WavToParquetConverter with injected model mocks.

All tests use DummyVAD and DummyDiarizer — no model downloads, no HF token
required. These tests verify that the orchestration logic (VAD aggregation,
YAMNet per-window, diarization mapping, three output files) is correct.

Real-model smoke tests are gated behind pytest marks and require network
access + HF_TOKEN. Run them separately with:
    pytest -m "real_models" tests/integration/test_wav_to_parquet_with_models.py
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.audio_pipeline.offline.wav_to_parquet import WavToParquetConverter
from src.audio_pipeline.segmentation.dummy_vad import DummyVAD
from src.audio_pipeline.speakers.dummy_diarizer import DummyDiarizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def make_wav(path: str, duration: float = 3.0, sr: int = 16000) -> str:
    """Write a mono 440 Hz sine wave WAV to *path*."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, samples, sr)
    return path


def run_with_dummies(
    temp_dir: str,
    duration: float = 3.0,
    vad_prob: float = 0.8,
    simulate_overlap: bool = False,
    simulate_no_speech: bool = False,
    yamnet: object = None,
    config: dict = None,
) -> tuple:
    """
    Run the converter with injected DummyVAD and DummyDiarizer.

    Returns (output_dir, wav_path).
    """
    wav_path = make_wav(os.path.join(temp_dir, "input.wav"), duration=duration)
    output_dir = os.path.join(temp_dir, "output")

    vad = DummyVAD(default_prob=vad_prob)
    diarizer = DummyDiarizer(
        simulate_overlap=simulate_overlap,
        simulate_no_speech=simulate_no_speech,
    )

    converter = WavToParquetConverter(
        config=config or {"quality": {"snr_min_db": 0.0}},
        vad=vad,
        yamnet=yamnet,
        diarizer=diarizer,
    )
    converter.process(wav_path, output_dir)
    return output_dir, wav_path


# ---------------------------------------------------------------------------
# Test: four output artefacts are produced
# ---------------------------------------------------------------------------


def test_three_output_files_exist(temp_dir):
    """All four output artefacts must be written on every run."""
    output_dir, _ = run_with_dummies(temp_dir)

    assert os.path.exists(os.path.join(output_dir, "acoustic_features.parquet"))
    assert os.path.exists(os.path.join(output_dir, "speech_segments.parquet"))
    assert os.path.exists(os.path.join(output_dir, "speaker_turns.parquet"))
    assert os.path.exists(os.path.join(output_dir, "extraction_metadata.json"))


# ---------------------------------------------------------------------------
# Test: acoustic_features.parquet
# ---------------------------------------------------------------------------


def test_vad_fields_populated(temp_dir):
    """DummyVAD returning 0.8 must produce non-zero vad_probability_mean and voiced_ratio."""
    output_dir, _ = run_with_dummies(temp_dir, vad_prob=0.8)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    assert (
        df["vad_probability_mean"].iloc[0] > 0.0
    ), "Expected non-zero vad_probability_mean"
    assert df["voiced_ratio"].iloc[0] > 0.0, "Expected non-zero voiced_ratio"


def test_vad_silent_returns_zero(temp_dir):
    """DummyVAD returning 0.0 must produce zero vad_probability_mean and voiced_ratio."""
    output_dir, _ = run_with_dummies(temp_dir, vad_prob=0.0)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    assert df["vad_probability_mean"].iloc[0] == pytest.approx(0.0)
    assert df["voiced_ratio"].iloc[0] == pytest.approx(0.0)


def test_yamnet_fields_populated(temp_dir):
    """A mock YAMNet returning fixed scores must appear in yamnet_event_scores."""
    # Build a mock FeatureExtractor that always returns a fixed dict
    mock_yamnet = MagicMock()
    mock_yamnet.extract.return_value = {
        "Scream": 0.9,
        "Crying, sobbing": 0.3,
        "Yell": 0.1,
        "Gasp": 0.0,
        "Groan": 0.0,
        "Whimper": 0.0,
        "Wheeze": 0.0,
    }
    mock_yamnet.get_version.return_value = "mock_yamnet_v1"
    mock_yamnet.get_model_hash.return_value = "deadbeef"

    output_dir, _ = run_with_dummies(temp_dir, yamnet=mock_yamnet)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    scores_raw = df["yamnet_event_scores"].iloc[0]
    # yamnet_event_scores is stored as a JSON string in Parquet
    scores = json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw
    assert isinstance(scores, dict), "yamnet_event_scores should be a dict"
    assert scores.get("Scream") == pytest.approx(0.9)
    assert scores.get("Crying, sobbing") == pytest.approx(0.3)


def test_diarization_speaker_mapped(temp_dir):
    """DummyDiarizer assigns SPEAKER_00 to every window; attribution stays UNKNOWN."""
    output_dir, _ = run_with_dummies(temp_dir)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    assert (
        df["speaker_id"] == "SPEAKER_00"
    ).all(), "Expected SPEAKER_00 for all windows"
    # Patient attribution must never be set by diarization alone
    assert (
        df["attribution_status"] == "UNKNOWN"
    ).all(), (
        "attribution_status must remain UNKNOWN — diarizer doesn't identify the patient"
    )
    assert (
        df["patient_probability"].isna().all()
    ), "patient_probability must be None/NaN"


def test_no_speaker_window(temp_dir):
    """DummyDiarizer with no speech returns None speaker_id and UNKNOWN status."""
    output_dir, _ = run_with_dummies(temp_dir, simulate_no_speech=True)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    assert (
        df["speaker_id"].isna().all()
    ), "Expected None speaker_id when diarizer finds no speech"
    assert (df["attribution_status"] == "UNKNOWN").all()


def test_overlap_detected(temp_dir):
    """DummyDiarizer with simulated overlap must produce OVERLAP status."""
    output_dir, _ = run_with_dummies(temp_dir, simulate_overlap=True)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    # At least some windows should have OVERLAP status
    assert (
        df["attribution_status"] == "OVERLAP"
    ).any(), "Expected at least one OVERLAP window when diarizer simulates simultaneous speakers"
    assert (df["overlap_probability"] > 0.0).any()


def test_window_count_correct(temp_dir):
    """3s audio with 2s window / 0.5s hop should produce exactly 3 windows."""
    output_dir, _ = run_with_dummies(temp_dir, duration=3.0)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert len(df) == 3


def test_timestamps_monotonic(temp_dir):
    """window_start_ms must be strictly increasing across rows."""
    output_dir, _ = run_with_dummies(temp_dir, duration=4.0)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    starts = df["window_start_ms"].tolist()
    assert starts == sorted(starts), "window_start_ms must be monotonically increasing"
    assert len(set(starts)) == len(starts), "window_start_ms values must be unique"


def test_no_patient_attribution_without_enrollment(temp_dir):
    """Pyannote labels (SPEAKER_00 etc.) must never produce PATIENT status."""
    output_dir, _ = run_with_dummies(temp_dir)
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert (
        "PATIENT" not in df["attribution_status"].values
    ), "PATIENT status must not appear from diarization alone — requires enrollment"


# ---------------------------------------------------------------------------
# Test: speech_segments.parquet
# ---------------------------------------------------------------------------


def test_speech_segments_schema(temp_dir):
    """speech_segments.parquet must have all required columns."""
    output_dir, _ = run_with_dummies(temp_dir, vad_prob=0.8)
    df = pd.read_parquet(os.path.join(output_dir, "speech_segments.parquet"))

    required_cols = {
        "session_id",
        "stream_id",
        "start_ms",
        "end_ms",
        "start_sample",
        "end_sample",
        "duration_ms",
        "vad_probability_mean",
        "vad_probability_min",
        "provisional",
    }
    assert required_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {required_cols - set(df.columns)}"


def test_speech_segments_populated_when_vad_active(temp_dir):
    """When DummyVAD returns high probability, at least one speech segment must appear."""
    output_dir, _ = run_with_dummies(temp_dir, duration=5.0, vad_prob=0.9)
    df = pd.read_parquet(os.path.join(output_dir, "speech_segments.parquet"))
    assert (
        len(df) >= 1
    ), "Expected at least one speech segment when VAD probability is high"


def test_speech_segments_empty_when_vad_silent(temp_dir):
    """When DummyVAD returns 0.0 probability, speech_segments must be empty."""
    output_dir, _ = run_with_dummies(temp_dir, duration=3.0, vad_prob=0.0)
    df = pd.read_parquet(os.path.join(output_dir, "speech_segments.parquet"))
    assert len(df) == 0, "Expected empty speech_segments when VAD is silent"


def test_speech_segments_boundaries_valid(temp_dir):
    """start_ms must be <= end_ms for every speech segment."""
    output_dir, _ = run_with_dummies(temp_dir, duration=5.0, vad_prob=0.9)
    df = pd.read_parquet(os.path.join(output_dir, "speech_segments.parquet"))
    if not df.empty:
        assert (df["start_ms"] <= df["end_ms"]).all()
        assert (df["start_sample"] <= df["end_sample"]).all()


# ---------------------------------------------------------------------------
# Test: speaker_turns.parquet
# ---------------------------------------------------------------------------


def test_speaker_turns_schema(temp_dir):
    """speaker_turns.parquet must have all required columns."""
    output_dir, _ = run_with_dummies(temp_dir)
    df = pd.read_parquet(os.path.join(output_dir, "speaker_turns.parquet"))

    required_cols = {
        "start_ms",
        "end_ms",
        "duration_ms",
        "speaker_id",
        "overlap",
        "confidence",
        "diarizer_version",
    }
    assert required_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {required_cols - set(df.columns)}"


def test_speaker_turns_populated(temp_dir):
    """DummyDiarizer produces at least one turn per run."""
    output_dir, _ = run_with_dummies(temp_dir)
    df = pd.read_parquet(os.path.join(output_dir, "speaker_turns.parquet"))
    assert len(df) >= 1


def test_speaker_turns_speaker_id_matches(temp_dir):
    """Turns from DummyDiarizer must use its configured speaker_id."""
    output_dir, _ = run_with_dummies(temp_dir)
    df = pd.read_parquet(os.path.join(output_dir, "speaker_turns.parquet"))
    assert (df["speaker_id"] == "SPEAKER_00").all()


def test_speaker_turns_empty_when_no_diarizer(temp_dir):
    """Without a diarizer, speaker_turns.parquet must be an empty DataFrame."""
    wav_path = make_wav(os.path.join(temp_dir, "input.wav"))
    output_dir = os.path.join(temp_dir, "output")

    converter = WavToParquetConverter(
        config={"quality": {"snr_min_db": 0.0}},
        vad=None,
        yamnet=None,
        diarizer=None,
    )
    converter.process(wav_path, output_dir)

    df = pd.read_parquet(os.path.join(output_dir, "speaker_turns.parquet"))
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Test: extraction_metadata.json
# ---------------------------------------------------------------------------


def test_extraction_metadata_schema(temp_dir):
    """extraction_metadata.json must contain required provenance fields."""
    output_dir, _ = run_with_dummies(temp_dir)
    with open(os.path.join(output_dir, "extraction_metadata.json")) as f:
        meta = json.load(f)

    required_keys = {
        "session_id",
        "stream_id",
        "input_file",
        "input_file_sha256",
        "normalized_sample_rate",
        "duration_seconds",
        "config_hash",
        "models",
        "processing_status",
    }
    assert required_keys.issubset(
        set(meta.keys())
    ), f"Missing keys: {required_keys - set(meta.keys())}"


def test_extraction_metadata_models_versions(temp_dir):
    """models dict must record the version strings from injected models."""
    output_dir, _ = run_with_dummies(temp_dir)
    with open(os.path.join(output_dir, "extraction_metadata.json")) as f:
        meta = json.load(f)

    assert meta["models"]["vad"] == "dummy_vad_v1"
    assert meta["models"]["diarizer"] == "dummy_diarizer_v1"
    assert meta["models"]["yamnet"] is None  # not injected in run_with_dummies


def test_extraction_metadata_no_model_when_disabled(temp_dir):
    """When no models are injected, all model version fields must be None."""
    wav_path = make_wav(os.path.join(temp_dir, "input.wav"))
    output_dir = os.path.join(temp_dir, "output")

    WavToParquetConverter(config={"quality": {"snr_min_db": 0.0}}).process(
        wav_path, output_dir
    )

    with open(os.path.join(output_dir, "extraction_metadata.json")) as f:
        meta = json.load(f)

    assert meta["models"]["vad"] is None
    assert meta["models"]["yamnet"] is None
    assert meta["models"]["diarizer"] is None


def test_extraction_metadata_processing_status(temp_dir):
    """processing_status must be 'complete' on successful run."""
    output_dir, _ = run_with_dummies(temp_dir)
    with open(os.path.join(output_dir, "extraction_metadata.json")) as f:
        meta = json.load(f)
    assert meta["processing_status"] == "complete"


# ---------------------------------------------------------------------------
# Test: backwards compatibility (no models injected = old behaviour)
# ---------------------------------------------------------------------------


def test_no_models_placeholders_intact(temp_dir):
    """Without any models injected, all model-derived fields must be at placeholder values."""
    wav_path = make_wav(os.path.join(temp_dir, "input.wav"))
    output_dir = os.path.join(temp_dir, "output")

    WavToParquetConverter(config={"quality": {"snr_min_db": 0.0}}).process(
        wav_path, output_dir
    )

    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert (df["vad_probability_mean"] == 0.0).all()
    assert (df["voiced_ratio"] == 0.0).all()
    assert df["speaker_id"].isna().all()
    assert (df["attribution_status"] == "UNKNOWN").all()

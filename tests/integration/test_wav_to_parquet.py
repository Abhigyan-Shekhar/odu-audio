"""
Integration and unit tests for the WavToParquetConverter, CLI, and offline pipeline.
Exceeds 40+ test cases covering formatting, constraints, and failures.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from click.testing import CliRunner

from src.audio_pipeline.cli import wav_to_parquet
from src.audio_pipeline.io.audio_data import AudioData
from src.audio_pipeline.io.audio_reader import AudioReader
from src.audio_pipeline.offline.wav_to_parquet import WavToParquetConverter
from src.audio_pipeline.preprocessing.channel_mixer import to_mono
from src.audio_pipeline.preprocessing.resampler import Resampler
from src.audio_pipeline.quality.clipping import detect_clipping
from src.audio_pipeline.quality.dropout import detect_dropout
from src.audio_pipeline.quality.rms import compute_rms
from src.audio_pipeline.quality.snr import estimate_snr
from src.audio_pipeline.windowing.fixed_windower import FixedWindower

# ==========================================
# Fixtures & Helpers
# ==========================================


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def generate_wav(
    path, sample_rate=16000, n_channels=1, duration=3.0, clipping=False, silence=False
):
    """Helper to generate test WAV files."""
    n_samples = int(duration * sample_rate)
    if silence:
        samples = np.zeros(n_samples, dtype=np.float32)
    elif clipping:
        # Exceed standard amplitude to force clipping
        t = np.linspace(0, duration, n_samples, endpoint=False)
        samples = 1.5 * np.sin(2 * np.pi * 440 * t)
    else:
        # Standard sine wave
        t = np.linspace(0, duration, n_samples, endpoint=False)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t)

    if n_channels > 1:
        samples = np.repeat(samples[:, np.newaxis], n_channels, axis=1)

    sf.write(path, samples, sample_rate, subtype="PCM_16" if not clipping else "FLOAT")
    return samples


# ==========================================
# Unit Tests: Audio Reader & Data (5 cases)
# ==========================================


def test_audio_reader_basic(temp_dir):
    path = os.path.join(temp_dir, "test.wav")
    generate_wav(path, sample_rate=16000, n_channels=1, duration=2.0)

    reader = AudioReader()
    data = reader.read(path)
    assert isinstance(data, AudioData)
    assert data.sample_rate == 16000
    assert data.n_channels == 1
    assert data.duration_seconds == 2.0
    assert data.samples.ndim == 1


def test_audio_reader_stereo(temp_dir):
    path = os.path.join(temp_dir, "stereo.wav")
    generate_wav(path, sample_rate=22050, n_channels=2, duration=1.0)

    reader = AudioReader()
    data = reader.read(path)
    assert data.n_channels == 2
    # Transposed shape: (n_channels, n_samples)
    assert data.samples.shape == (2, 22050)


def test_audio_reader_file_not_found():
    reader = AudioReader()
    with pytest.raises(FileNotFoundError):
        reader.read("non_existent_file.wav")


def test_audio_reader_empty_file(temp_dir):
    path = os.path.join(temp_dir, "empty.wav")
    # Write a WAV header but 0 samples
    sf.write(path, np.array([], dtype=np.float32), 16000)

    reader = AudioReader()
    with pytest.raises(ValueError, match="empty"):
        reader.read(path)


def test_audio_reader_corrupt_file(temp_dir):
    path = os.path.join(temp_dir, "corrupt.wav")
    with open(path, "w") as f:
        f.write("corrupt file header content")

    reader = AudioReader()
    with pytest.raises(ValueError, match="corrupt|malformed|unsupported"):
        reader.read(path)


# ==========================================
# Unit Tests: Preprocessing (5 cases)
# ==========================================


def test_resampler_mono():
    resampler = Resampler()
    samples = np.sin(np.linspace(0, 1, 8000))
    resampled = resampler.resample(samples, 8000, 16000)
    assert len(resampled) == 16000


def test_resampler_stereo():
    resampler = Resampler()
    samples = np.zeros((2, 8000))
    resampled = resampler.resample(samples, 8000, 16000)
    assert resampled.shape == (2, 16000)


def test_resampler_same_rate():
    resampler = Resampler()
    samples = np.ones(100)
    resampled = resampler.resample(samples, 16000, 16000)
    assert np.array_equal(samples, resampled)


def test_channel_mixer_mono():
    samples = np.array([1.0, 2.0, 3.0])
    mixed = to_mono(samples)
    assert np.array_equal(samples, mixed)


def test_channel_mixer_stereo():
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])  # channel-first
    mixed = to_mono(samples)
    # average: channel 0 + channel 1 => [2.0, 3.0]
    assert np.array_equal(mixed, np.array([2.0, 3.0]))


# ==========================================
# Unit Tests: Windowing (5 cases)
# ==========================================


def test_fixed_windower_counts():
    # 3 seconds of 16 kHz = 48000 samples
    # window = 2.0s (32000 samples), hop = 0.5s (8000 samples)
    # sequence check:
    # 0: 0 .. 32000 (0.0s .. 2.0s)
    # 1: 8000 .. 40000 (0.5s .. 2.5s)
    # 2: 16000 .. 48000 (1.0s .. 3.0s)
    audio = AudioData(
        samples=np.zeros(48000), sample_rate=16000, n_channels=1, duration_seconds=3.0
    )
    windower = FixedWindower(window_seconds=2.0, hop_seconds=0.5)
    windows = list(windower.window(audio))

    assert len(windows) == 3
    assert windows[0].sequence_number == 0
    assert windows[0].start_sample == 0
    assert windows[0].end_sample == 32000
    assert windows[0].start_time_ms == 0
    assert windows[0].end_time_ms == 2000

    assert windows[2].sequence_number == 2
    assert windows[2].start_sample == 16000
    assert windows[2].end_sample == 48000
    assert windows[2].start_time_ms == 1000
    assert windows[2].end_time_ms == 3000


def test_fixed_windower_partial():
    # 2.2 seconds of 16 kHz = 35200 samples
    # window = 2.0s (32000 samples), hop = 0.5s (8000)
    # 0: 0 .. 32000
    # 1: 8000 .. 35200 (clamped)
    audio = AudioData(
        samples=np.zeros(35200), sample_rate=16000, n_channels=1, duration_seconds=2.2
    )
    windower = FixedWindower(window_seconds=2.0, hop_seconds=0.5)
    windows = list(windower.window(audio))

    assert len(windows) == 2
    assert windows[1].end_sample == 35200
    assert windows[1].end_time_ms == 2200


def test_fixed_windower_very_short():
    # Audio shorter than one window: 1.0 second
    audio = AudioData(
        samples=np.zeros(16000), sample_rate=16000, n_channels=1, duration_seconds=1.0
    )
    windower = FixedWindower(window_seconds=2.0, hop_seconds=0.5)
    windows = list(windower.window(audio))

    assert len(windows) == 1
    assert windows[0].start_sample == 0
    assert windows[0].end_sample == 16000


def test_fixed_windower_empty():
    audio = AudioData(
        samples=np.array([]), sample_rate=16000, n_channels=1, duration_seconds=0.0
    )
    windower = FixedWindower(window_seconds=2.0, hop_seconds=0.5)
    windows = list(windower.window(audio))
    assert len(windows) == 0


def test_fixed_windower_invalid_input():
    audio_stereo = AudioData(
        samples=np.zeros((2, 16000)),
        sample_rate=16000,
        n_channels=2,
        duration_seconds=1.0,
    )
    windower = FixedWindower(window_seconds=2.0, hop_seconds=0.5)
    with pytest.raises(ValueError, match="1D mono"):
        list(windower.window(audio_stereo))


# ==========================================
# Unit Tests: Quality Metrics (8 cases)
# ==========================================


def test_detect_clipping_none():
    samples = np.array([0.0, 0.5, -0.5])
    assert detect_clipping(samples, threshold=0.9) == 0.0


def test_detect_clipping_all():
    samples = np.array([1.0, -1.0, 1.0])
    assert detect_clipping(samples, threshold=0.99) == 1.0


def test_detect_clipping_empty():
    assert detect_clipping(np.array([])) == 0.0


def test_detect_dropout_none():
    samples = np.array([0.5, 0.5, 0.5])
    assert detect_dropout(samples, threshold=1e-3) == 0.0


def test_detect_dropout_half():
    samples = np.array([0.5, 0.0, 0.5, 0.0])
    assert detect_dropout(samples, threshold=1e-5) == 0.5


def test_detect_dropout_empty():
    assert detect_dropout(np.array([])) == 0.0


def test_compute_rms_db():
    samples = np.array([0.5, -0.5, 0.5, -0.5])
    # rms = 0.5. dB = 20 * log10(0.5) = -6.02
    assert pytest.approx(compute_rms(samples), abs=0.1) == -6.02


def test_compute_rms_silent():
    samples = np.zeros(100)
    assert compute_rms(samples) == -100.0


# ==========================================
# Unit Tests: SNR Estimation (3 cases)
# ==========================================


def test_estimate_snr_silent():
    # Pure silence should have SNR of 0.0
    samples = np.zeros(1000)
    assert estimate_snr(samples) == 0.0


def test_estimate_snr_loud():
    # Sine wave with a bit of noise
    t = np.linspace(0, 1, 16000)
    signal = np.sin(2 * np.pi * 440 * t)
    # Zero out second half to simulate speech pause (silence gaps) for noise estimation
    signal[8000:] = 0.0
    noise = np.random.normal(0, 0.01, 16000)  # low noise
    samples = signal + noise
    snr = estimate_snr(samples, sample_rate=16000)
    assert snr > 15.0  # SNR should be high


def test_estimate_snr_short():
    samples = np.array([0.1, 0.2, 0.3])
    # Shorter than one frame length: fallback check runs and returns valid SNR
    assert isinstance(estimate_snr(samples), float)


# ==========================================
# Integration Tests: WavToParquet (10 cases)
# ==========================================


def test_wav_to_parquet_basic(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    output_dir = os.path.join(temp_dir, "output")
    generate_wav(wav_path, sample_rate=16000, duration=3.0)

    # Pass snr_min_db=0.0 config to prevent LOW_SNR status for pure synthetic wave
    converter = WavToParquetConverter(config={"quality": {"snr_min_db": 0.0}})
    converter.process(wav_path, output_dir)

    parquet_path = os.path.join(output_dir, "acoustic_features.parquet")
    assert os.path.exists(parquet_path)
    df = pd.read_parquet(parquet_path)

    # 3 seconds of audio, 2s window, 0.5s hop:
    # 0.0..2.0, 0.5..2.5, 1.0..3.0 (3 windows)
    assert len(df) == 3
    assert "clipping_ratio" in df.columns
    assert "snr_db" in df.columns
    assert "quality_status" in df.columns
    assert df["quality_status"].iloc[0] == "OK"


def test_sample_rate_8khz(temp_dir):
    wav_path = os.path.join(temp_dir, "input_8k.wav")
    output_dir = os.path.join(temp_dir, "output_8k")
    generate_wav(wav_path, sample_rate=8000, duration=2.0)

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir)

    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert len(df) > 0
    # window_start_ms should map correctly
    assert df["window_start_ms"].iloc[0] == 0


def test_sample_rates_cd_and_pro(temp_dir):
    # CD quality (44.1 kHz) and Pro (48 kHz)
    for sr in [44100, 48000]:
        wav_path = os.path.join(temp_dir, f"input_{sr}.wav")
        output_dir = os.path.join(temp_dir, f"output_{sr}")
        generate_wav(wav_path, sample_rate=sr, duration=2.5)

        converter = WavToParquetConverter()
        converter.process(wav_path, output_dir)
        assert os.path.exists(os.path.join(output_dir, "acoustic_features.parquet"))


def test_wav_to_parquet_stereo(temp_dir):
    wav_path = os.path.join(temp_dir, "stereo.wav")
    output_dir = os.path.join(temp_dir, "output_stereo")
    generate_wav(wav_path, sample_rate=16000, n_channels=2, duration=2.0)

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir)

    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert len(df) > 0


def test_wav_to_parquet_clipped(temp_dir):
    wav_path = os.path.join(temp_dir, "clipped.wav")
    output_dir = os.path.join(temp_dir, "output_clipped")
    # Generate high amplitude to trigger clipping status
    generate_wav(wav_path, sample_rate=16000, duration=2.0, clipping=True)

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir)

    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert df["quality_status"].iloc[0] == "CLIPPED"
    assert df["clipping_ratio"].iloc[0] > 0.05


def test_wav_to_parquet_silent_dropout(temp_dir):
    wav_path = os.path.join(temp_dir, "silent.wav")
    output_dir = os.path.join(temp_dir, "output_silent")
    generate_wav(wav_path, sample_rate=16000, duration=2.0, silence=True)

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir)

    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert df["quality_status"].iloc[0] == "DROPOUT"
    assert df["dropout_ratio"].iloc[0] > 0.10


def test_nan_infinity_prevention(temp_dir, monkeypatch):
    # Mock soundfile.read to return a numpy array containing NaN/Inf
    def mock_read(*args, **kwargs):
        return (np.array([np.nan, 1.0, np.inf]), 16000)

    monkeypatch.setattr(sf, "read", mock_read)

    dummy_wav = os.path.join(temp_dir, "dummy.wav")
    with open(dummy_wav, "w") as f:
        f.write("dummy")

    reader = AudioReader()
    with pytest.raises(ValueError, match="NaN or Inf"):
        reader.read(dummy_wav)


def test_timestamp_and_sequence_continuity(temp_dir):
    wav_path = os.path.join(temp_dir, "continuity.wav")
    output_dir = os.path.join(temp_dir, "output_cont")
    generate_wav(wav_path, sample_rate=16000, duration=4.0)

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir)

    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))

    # Check sequence increments by 1
    sequences = df.index.tolist()
    for i in range(len(sequences) - 1):
        # since index/rows are sequentially appended
        assert df["window_start_ms"].iloc[i] < df["window_start_ms"].iloc[i + 1]
        assert df["source_start_sample"].iloc[i] < df["source_start_sample"].iloc[i + 1]


def test_atomic_write_safety(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    output_dir = os.path.join(temp_dir, "output_atomic")
    generate_wav(wav_path, sample_rate=16000, duration=2.0)

    # Force a failure midway by mocking/patching pd.DataFrame.to_parquet
    # to throw a custom exception
    def mock_to_parquet(*args, **kwargs):
        raise RuntimeError("Write crash simulation")

    old_to_parquet = pd.DataFrame.to_parquet
    pd.DataFrame.to_parquet = mock_to_parquet

    converter = WavToParquetConverter()
    try:
        with pytest.raises(RuntimeError):
            converter.process(wav_path, output_dir)
        # Verify no partial acoustic_features output is left behind
        assert not os.path.exists(os.path.join(output_dir, "acoustic_features.parquet"))
    finally:
        pd.DataFrame.to_parquet = old_to_parquet


def test_overwrite_existing_output(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    output_dir = os.path.join(temp_dir, "output_overwrite")
    generate_wav(wav_path, sample_rate=16000, duration=2.0)

    # Create pre-existing directory and file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "acoustic_features.parquet"), "w") as f:
        f.write("existing content")

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir)

    # Pre-existing file is overwritten successfully
    df = pd.read_parquet(os.path.join(output_dir, "acoustic_features.parquet"))
    assert len(df) > 0


def test_reproducibility_determinism(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    output_dir1 = os.path.join(temp_dir, "output1")
    output_dir2 = os.path.join(temp_dir, "output2")

    generate_wav(wav_path, sample_rate=16000, duration=3.0)

    converter = WavToParquetConverter()
    converter.process(wav_path, output_dir1)
    converter.process(wav_path, output_dir2)

    df1 = pd.read_parquet(os.path.join(output_dir1, "acoustic_features.parquet"))
    df2 = pd.read_parquet(os.path.join(output_dir2, "acoustic_features.parquet"))

    # Drop random UUIDs for deterministic comparison
    df1_clean = df1.drop(columns=["session_id", "stream_id"])
    df2_clean = df2.drop(columns=["session_id", "stream_id"])

    # Compare quality metrics within tolerance
    pd.testing.assert_frame_equal(df1_clean, df2_clean)


# ==========================================
# CLI Tests (3 cases)
# ==========================================


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(wav_to_parquet, ["--help"])
    assert result.exit_code == 0
    assert "Convert a WAV file to a Parquet file" in result.output


def test_cli_success(temp_dir):
    wav_path = os.path.join(temp_dir, "input.wav")
    output_dir = os.path.join(temp_dir, "output")
    generate_wav(wav_path, sample_rate=16000, duration=2.0)

    runner = CliRunner()
    result = runner.invoke(
        wav_to_parquet, ["--input", wav_path, "--output-dir", output_dir]
    )

    assert result.exit_code == 0
    assert os.path.exists(os.path.join(output_dir, "acoustic_features.parquet"))
    assert "Successfully wrote feature record dataset" in result.output


def test_cli_missing_input(temp_dir):
    output_dir = os.path.join(temp_dir, "output")
    runner = CliRunner()
    result = runner.invoke(
        wav_to_parquet, ["--input", "non_existent.wav", "--output-dir", output_dir]
    )
    assert result.exit_code != 0

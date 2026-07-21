"""
WavToParquetConverter component for offline feature extraction and Parquet serialization.
"""

import hashlib
import json
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.audio_pipeline.io.audio_reader import AudioReader
from src.audio_pipeline.preprocessing.channel_mixer import to_mono
from src.audio_pipeline.preprocessing.resampler import Resampler
from src.audio_pipeline.quality.clipping import detect_clipping
from src.audio_pipeline.quality.dropout import detect_dropout
from src.audio_pipeline.quality.snr import estimate_snr
from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord
from src.audio_pipeline.windowing.fixed_windower import FixedWindower


class WavToParquetConverter:
    """
    Offline converter that processes WAV files into AcousticFeatureRecord Parquet datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with config.

        Args:
            config: A dictionary representing the pipeline configuration.
        """
        self.config = config or {}

        # Extractor parameters from config
        features_cfg = self.config.get("features", {})
        egemaps_cfg = features_cfg.get("egemaps", {})
        self.window_seconds = egemaps_cfg.get("window_seconds", 2.0)
        self.hop_seconds = egemaps_cfg.get("hop_seconds", 0.5)

        # YAMNet event classes for placeholder struct schema
        yamnet_cfg = features_cfg.get("yamnet", {})
        self.event_classes = yamnet_cfg.get(
            "event_classes",
            ["Scream", "Crying, sobbing", "Yell", "Gasp", "Groan", "Whimper", "Wheeze"],
        )

        # Quality parameters
        quality_cfg = self.config.get("quality", {})
        self.clipping_threshold = quality_cfg.get("clipping_threshold", 0.99)
        self.clipping_max_ratio = quality_cfg.get("clipping_max_ratio", 0.05)
        self.dropout_zero_threshold = quality_cfg.get("dropout_zero_threshold", 1e-6)
        self.dropout_max_ratio = quality_cfg.get("dropout_max_ratio", 0.10)
        self.snr_min_db = quality_cfg.get("snr_min_db", 10.0)

        # Audio handlers
        self.reader = AudioReader()
        self.resampler = Resampler()

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "WavToParquetConverter":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            An initialized WavToParquetConverter.
        """
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        return cls(config)

    def _compute_config_hash(self) -> str:
        """Compute SHA256 of configuration dictionary for reproducibility."""
        # Convert dictionary to stable JSON representation
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    def process(self, wav_path: str, output_path: str) -> None:
        """
        Process WAV file end-to-end and write output to Parquet file atomically.

        Args:
            wav_path: Path to input WAV file.
            output_path: Target path for output Parquet file.
        """
        # 1. Read raw audio file
        audio_data = self.reader.read(wav_path)
        original_sr = audio_data.sample_rate

        # 2. Resample to target 16 kHz
        target_sr = 16000
        resampled_samples = self.resampler.resample(
            audio_data.samples,
            source_sr=original_sr,
            target_sr=target_sr,
        )

        # 3. Downmix multi-channel to mono
        mono_samples = to_mono(resampled_samples)

        # Create temp AudioData wrapper for the windower
        mono_audio_data = audio_data.__class__(
            samples=mono_samples,
            sample_rate=target_sr,
            n_channels=1,
            duration_seconds=len(mono_samples) / target_sr,
        )

        # 4. Generate windows
        windower = FixedWindower(
            window_seconds=self.window_seconds,
            hop_seconds=self.hop_seconds,
        )
        windows = list(windower.window(mono_audio_data))

        records: List[AcousticFeatureRecord] = []
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"

        config_hash = self._compute_config_hash()

        # 5. Extract features & quality metrics per window
        for window in windows:
            clipping_ratio = detect_clipping(
                window.samples, threshold=self.clipping_threshold
            )
            dropout_ratio = detect_dropout(
                window.samples, threshold=self.dropout_zero_threshold
            )
            snr_db = estimate_snr(window.samples, sample_rate=target_sr)

            # Determine quality status
            if clipping_ratio >= self.clipping_max_ratio:
                quality_status = "CLIPPED"
            elif dropout_ratio >= self.dropout_max_ratio:
                quality_status = "DROPOUT"
            elif snr_db < self.snr_min_db:
                quality_status = "LOW_SNR"
            else:
                quality_status = "OK"

            # Map resampled window samples back to source WAV sample indices
            ratio = original_sr / target_sr
            source_start_sample = int(window.start_sample * ratio)
            source_end_sample = int(window.end_sample * ratio)

            # Instantiating the dataclass triggers validator checks
            record = AcousticFeatureRecord(
                session_id=session_id,
                stream_id=stream_id,
                window_start_ms=window.start_time_ms,
                window_end_ms=window.end_time_ms,
                source_start_sample=source_start_sample,
                source_end_sample=source_end_sample,
                attribution_status="UNKNOWN",
                patient_probability=None,
                attribution_method=None,
                vad_probability_mean=0.0,
                voiced_ratio=0.0,
                overlap_probability=0.0,
                egemaps=None,
                yamnet_event_scores={cls: 0.0 for cls in self.event_classes},
                emotion_embedding=None,
                snr_db=snr_db,
                clipping_ratio=clipping_ratio,
                dropout_ratio=dropout_ratio,
                quality_status=quality_status,
                extractor_versions={
                    "resampler": "librosa",
                    "audio_reader": "soundfile",
                    "quality_metrics": "native",
                },
                config_hash=config_hash,
                model_hashes={"none": ""},
            )
            records.append(record)

        # Convert records to Pandas DataFrame
        df_records = pd.DataFrame([r.to_dict() for r in records])

        # Ensure output folder exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write atomically using a temporary file in the same directory
        temp_fd, temp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
        os.close(temp_fd)

        try:
            df_records.to_parquet(
                temp_path,
                index=False,
                compression="snappy",
            )
            # Rename temp file to destination atomically
            os.replace(temp_path, output_path)
        except Exception as e:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

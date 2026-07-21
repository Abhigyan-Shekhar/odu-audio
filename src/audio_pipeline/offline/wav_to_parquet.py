"""
WavToParquetConverter — offline feature extraction and Parquet serialization.

Produces four output artefacts inside ``output_dir``:

* ``acoustic_features.parquet``  — one row per fixed feature window
* ``speech_segments.parquet``    — one row per VAD/endpointer speech boundary
* ``speaker_turns.parquet``      — one row per Pyannote diarizer turn
* ``extraction_metadata.json``   — provenance: models, versions, config hash

Models are injected via the constructor so that tests can use deterministic
dummies (DummyVAD, DummyDiarizer) without downloading any weights.
"""

import hashlib
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.audio_pipeline.features.feature_extractor import FeatureExtractor
from src.audio_pipeline.io.audio_reader import AudioReader
from src.audio_pipeline.preprocessing.channel_mixer import to_mono
from src.audio_pipeline.preprocessing.resampler import Resampler
from src.audio_pipeline.quality.clipping import detect_clipping
from src.audio_pipeline.quality.dropout import detect_dropout
from src.audio_pipeline.quality.snr import estimate_snr
from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord
from src.audio_pipeline.schemas.segment import SpeechSegment
from src.audio_pipeline.segmentation.endpointer import Endpointer
from src.audio_pipeline.segmentation.offline_vad import (
    VadFrame,
    aggregate_vad_for_window,
    run_vad_frames,
)
from src.audio_pipeline.segmentation.vad_interface import VADInterface
from src.audio_pipeline.speakers.diarizer_interface import (
    DiarizationResult,
    DiarizedTurn,
    DiarizeriInterface,
)
from src.audio_pipeline.windowing.fixed_windower import FixedWindower

logger = logging.getLogger(__name__)

TARGET_SR = 16000


class WavToParquetConverter:
    """
    Offline converter that processes WAV files into acoustic feature datasets.

    Models (VAD, YAMNet, diarizer) are optional — if not injected, the
    corresponding fields in ``acoustic_features.parquet`` remain at their
    placeholder values (0.0 / None / empty dict), preserving full backwards
    compatibility with existing tests.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vad: Optional[VADInterface] = None,
        yamnet: Optional[FeatureExtractor] = None,
        diarizer: Optional[DiarizeriInterface] = None,
    ):
        """
        Args:
            config:   Pipeline configuration dictionary (loaded from YAML).
            vad:      VADInterface implementation (SileroVAD or DummyVAD).
                      When None, VAD fields are set to placeholder 0.0 values.
            yamnet:   FeatureExtractor for audio event detection (YAMNetDetector).
                      When None, yamnet_event_scores is an empty placeholder dict.
            diarizer: DiarizeriInterface implementation (PyannoteDiarizer or DummyDiarizer).
                      When None, speaker attribution fields remain UNKNOWN/None.
        """
        self.config = config or {}

        # Model instances (all optional — None means skip that model)
        self._vad = vad
        self._yamnet = yamnet
        self._diarizer = diarizer

        # Feature window parameters
        features_cfg = self.config.get("features", {})
        egemaps_cfg = features_cfg.get("egemaps", {})
        self.window_seconds = egemaps_cfg.get("window_seconds", 2.0)
        self.hop_seconds = egemaps_cfg.get("hop_seconds", 0.5)

        # YAMNet placeholder event classes (used when yamnet is None)
        yamnet_cfg = features_cfg.get("yamnet", {})
        self.event_classes: List[str] = yamnet_cfg.get(
            "event_classes",
            ["Scream", "Crying, sobbing", "Yell", "Gasp", "Groan", "Whimper", "Wheeze"],
        )

        # Quality thresholds
        quality_cfg = self.config.get("quality", {})
        self.clipping_threshold = quality_cfg.get("clipping_threshold", 0.99)
        self.clipping_max_ratio = quality_cfg.get("clipping_max_ratio", 0.05)
        self.dropout_zero_threshold = quality_cfg.get("dropout_zero_threshold", 1e-6)
        self.dropout_max_ratio = quality_cfg.get("dropout_max_ratio", 0.10)
        self.snr_min_db = quality_cfg.get("snr_min_db", 10.0)

        # VAD / endpointer parameters
        vad_cfg = self.config.get("vad", {})
        self.vad_threshold = vad_cfg.get("threshold", 0.5)
        self.overlap_threshold = quality_cfg.get("overlap_threshold", 0.3)

        # Audio I/O
        self.reader = AudioReader()
        self.resampler = Resampler()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "WavToParquetConverter":
        """
        Load configuration from a YAML file and return an un-armed converter.

        Models must still be injected separately (this factory does not
        instantiate heavyweight models to avoid side-effects on import).
        """
        config: Dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        return cls(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, wav_path: str, output_dir: str) -> None:
        """
        Process a WAV file and write four output artefacts to ``output_dir``.

        Execution order:
          1. Read and normalise audio (mono 16 kHz).
          2. Run diarization once on the full waveform.
          3. Run VAD once on the full waveform (512-sample frames).
          4. For each fixed 2-second window:
               a. Quality metrics.
               b. Aggregate VAD frames → mean_prob, voiced_ratio.
               c. Run YAMNet on window samples.
               d. Map diarizer turns → dominant speaker, overlap_probability.
               e. Build AcousticFeatureRecord.
          5. Feed VAD frames through Endpointer → SpeechSegment list.
          6. Write acoustic_features.parquet (atomic).
          7. Write speech_segments.parquet (atomic).
          8. Write speaker_turns.parquet (atomic).
          9. Write extraction_metadata.json (atomic).

        Args:
            wav_path:   Path to input WAV file.
            output_dir: Directory where output artefacts are written.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # 1. Read and normalise
        # ----------------------------------------------------------------
        audio_data = self.reader.read(wav_path)
        original_sr = audio_data.sample_rate

        resampled = self.resampler.resample(
            audio_data.samples, source_sr=original_sr, target_sr=TARGET_SR
        )
        mono: np.ndarray = to_mono(resampled)

        from src.audio_pipeline.io.audio_data import AudioData

        mono_audio = AudioData(
            samples=mono,
            sample_rate=TARGET_SR,
            n_channels=1,
            duration_seconds=len(mono) / TARGET_SR,
        )

        # ----------------------------------------------------------------
        # 2. Diarization — once on the full waveform
        # ----------------------------------------------------------------
        diarization_result: Optional[DiarizationResult] = None
        if self._diarizer is not None:
            logger.info("Running diarization on full waveform...")
            try:
                diarization_result = self._diarizer.diarize(mono, TARGET_SR)
                logger.info(
                    f"Diarization complete: {diarization_result.num_speakers} speakers, "
                    f"{len(diarization_result.turns)} turns."
                )
            except Exception as e:
                logger.error(
                    f"Diarization failed: {e}. Continuing without speaker turns."
                )
                diarization_result = None

        # ----------------------------------------------------------------
        # 3. VAD — once on the full waveform (512-sample frames)
        # ----------------------------------------------------------------
        vad_frames: List[VadFrame] = []
        if self._vad is not None:
            logger.info("Running VAD on full waveform...")
            try:
                vad_frames = run_vad_frames(mono, self._vad)
                logger.info(f"VAD complete: {len(vad_frames)} frames processed.")
            except Exception as e:
                logger.error(
                    f"VAD failed: {e}. Continuing with placeholder probabilities."
                )
                vad_frames = []

        # ----------------------------------------------------------------
        # 4. Window loop
        # ----------------------------------------------------------------
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"
        config_hash = self._compute_config_hash()

        windower = FixedWindower(
            window_seconds=self.window_seconds,
            hop_seconds=self.hop_seconds,
        )
        windows = list(windower.window(mono_audio))

        feature_records: List[AcousticFeatureRecord] = []
        ratio = original_sr / TARGET_SR

        for window in windows:
            # --- Quality metrics ---
            clipping_ratio = detect_clipping(
                window.samples, threshold=self.clipping_threshold
            )
            dropout_ratio = detect_dropout(
                window.samples, threshold=self.dropout_zero_threshold
            )
            snr_db = estimate_snr(window.samples, sample_rate=TARGET_SR)

            if clipping_ratio >= self.clipping_max_ratio:
                quality_status = "CLIPPED"
            elif dropout_ratio >= self.dropout_max_ratio:
                quality_status = "DROPOUT"
            elif snr_db is not None and snr_db < self.snr_min_db:
                quality_status = "LOW_SNR"
            else:
                quality_status = "OK"

            # --- VAD aggregation ---
            vad_prob_mean = 0.0
            voiced_ratio = 0.0
            if vad_frames:
                vad_prob_mean, voiced_ratio = aggregate_vad_for_window(
                    vad_frames,
                    window.start_time_ms,
                    window.end_time_ms,
                    threshold=self.vad_threshold,
                )

            # --- YAMNet ---
            yamnet_scores: Dict[str, float] = {cls: 0.0 for cls in self.event_classes}
            if self._yamnet is not None:
                try:
                    result = self._yamnet.extract(window.samples, TARGET_SR)
                    if result is not None:
                        yamnet_scores = result
                except Exception as e:
                    logger.error(
                        f"YAMNet failed on window {window.start_time_ms}ms: {e}"
                    )

            # --- Diarization window mapping ---
            speaker_id, attribution_status, overlap_prob = (
                self._map_diarization_to_window(
                    diarization_result, window.start_time_ms, window.end_time_ms
                )
            )

            # --- Source sample mapping (back to original SR) ---
            source_start = int(window.start_sample * ratio)
            source_end = int(window.end_sample * ratio)

            record = AcousticFeatureRecord(
                session_id=session_id,
                stream_id=stream_id,
                window_start_ms=window.start_time_ms,
                window_end_ms=window.end_time_ms,
                source_start_sample=source_start,
                source_end_sample=source_end,
                speaker_id=speaker_id,
                patient_probability=None,  # Pyannote doesn't identify the patient
                attribution_status=attribution_status,
                attribution_method=None,
                vad_probability_mean=vad_prob_mean,
                voiced_ratio=voiced_ratio,
                overlap_probability=overlap_prob,
                egemaps=None,  # openSMILE not wired in this phase
                yamnet_event_scores=yamnet_scores,
                emotion_embedding=None,  # emotion2vec not wired in this phase
                snr_db=snr_db,
                clipping_ratio=clipping_ratio,
                dropout_ratio=dropout_ratio,
                quality_status=quality_status,
                extractor_versions=self._build_extractor_versions(),
                config_hash=config_hash,
                model_hashes=self._build_model_hashes(),
            )
            feature_records.append(record)

        # ----------------------------------------------------------------
        # 5. Endpointer — produce SpeechSegment list from VAD frames
        # ----------------------------------------------------------------
        speech_segments: List[SpeechSegment] = []
        if vad_frames:
            endpointer = Endpointer(
                threshold=self.vad_threshold,
                sample_rate=TARGET_SR,
            )
            for frame in vad_frames:
                seg = endpointer.process(
                    prob=frame.probability,
                    start_ms=frame.start_ms,
                    end_ms=frame.end_ms,
                    start_sample=frame.start_sample,
                    end_sample=frame.end_sample,
                    session_id=session_id,
                    stream_id=stream_id,
                )
                if seg is not None and not seg.provisional:
                    speech_segments.append(seg)
            # Flush any open segment at end of stream
            final = endpointer.flush(session_id, stream_id)
            if final is not None:
                speech_segments.append(final)

        # ----------------------------------------------------------------
        # 6–9. Write output artefacts
        # ----------------------------------------------------------------
        self._write_acoustic_features(feature_records, output_dir)
        self._write_speech_segments(speech_segments, output_dir)
        self._write_speaker_turns(diarization_result, output_dir)
        self._write_metadata(
            wav_path=wav_path,
            mono_audio=mono_audio,
            session_id=session_id,
            stream_id=stream_id,
            config_hash=config_hash,
            output_dir=output_dir,
        )

        logger.info(
            f"Processing complete. "
            f"{len(feature_records)} windows, "
            f"{len(speech_segments)} speech segments, "
            f"{len(diarization_result.turns) if diarization_result else 0} speaker turns."
        )

    # ------------------------------------------------------------------
    # Diarization-to-window mapping
    # ------------------------------------------------------------------

    def _map_diarization_to_window(
        self,
        result: Optional[DiarizationResult],
        window_start_ms: int,
        window_end_ms: int,
    ) -> Tuple[Optional[str], str, float]:
        """
        Map Pyannote turns to a single feature window.

        Policy:
          - No turns overlap the window → (None, UNKNOWN, 0.0)
          - One or more turns overlap   → dominant speaker (most ms), status UNKNOWN
          - Simultaneous speech exceeds threshold → status OVERLAP

        NOTE: attribution_status is always UNKNOWN (never PATIENT) because
        Pyannote returns anonymous labels like SPEAKER_00. Patient identity
        requires a separate enrollment step.

        Returns:
            (speaker_id, attribution_status, overlap_probability)
        """
        if result is None or not result.turns:
            return None, "UNKNOWN", 0.0

        overlapping: List[DiarizedTurn] = result.turns_for_window(
            window_start_ms, window_end_ms
        )

        if not overlapping:
            return None, "UNKNOWN", 0.0

        window_dur_ms = window_end_ms - window_start_ms

        # Duration each speaker occupies within this window
        durations: Dict[str, int] = {}
        for t in overlapping:
            clipped_start = max(t.start_ms, window_start_ms)
            clipped_end = min(t.end_ms, window_end_ms)
            duration = max(0, clipped_end - clipped_start)
            durations[t.speaker_id] = durations.get(t.speaker_id, 0) + duration

        dominant = max(durations, key=lambda s: durations[s])

        # Overlap probability: total clipped speech beyond one window duration
        total_speech_ms = sum(durations.values())
        if window_dur_ms > 0 and total_speech_ms > window_dur_ms:
            overlap_prob = min(1.0, (total_speech_ms - window_dur_ms) / window_dur_ms)
        else:
            overlap_prob = 0.0

        status = "OVERLAP" if overlap_prob > self.overlap_threshold else "UNKNOWN"

        return dominant, status, float(overlap_prob)

    # ------------------------------------------------------------------
    # Artefact writers (all atomic: write to temp then rename)
    # ------------------------------------------------------------------

    def _atomic_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to Parquet atomically using a temp-file rename."""
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False, compression="snappy")
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _atomic_json(self, data: dict, path: str) -> None:
        """Write a dict to JSON atomically using a temp-file rename."""
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _write_acoustic_features(
        self, records: List[AcousticFeatureRecord], output_dir: str
    ) -> None:
        df = pd.DataFrame([r.to_dict() for r in records])
        self._atomic_parquet(df, os.path.join(output_dir, "acoustic_features.parquet"))

    def _write_speech_segments(
        self, segments: List[SpeechSegment], output_dir: str
    ) -> None:
        rows = [
            {
                "session_id": s.session_id,
                "stream_id": s.stream_id,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "start_sample": s.start_sample,
                "end_sample": s.end_sample,
                "duration_ms": s.duration_ms,
                "vad_probability_mean": s.vad_probability_mean,
                "vad_probability_min": s.vad_probability_min,
                "provisional": s.provisional,
            }
            for s in segments
        ]
        df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
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
                ]
            )
        )
        self._atomic_parquet(df, os.path.join(output_dir, "speech_segments.parquet"))

    def _write_speaker_turns(
        self, result: Optional[DiarizationResult], output_dir: str
    ) -> None:
        rows = []
        if result is not None:
            for t in result.turns:
                rows.append(
                    {
                        "start_ms": t.start_ms,
                        "end_ms": t.end_ms,
                        "duration_ms": t.duration_ms,
                        "speaker_id": t.speaker_id,
                        "overlap": t.overlap,
                        "confidence": t.confidence,
                        "diarizer_version": result.diarizer_version,
                    }
                )
        df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "start_ms",
                    "end_ms",
                    "duration_ms",
                    "speaker_id",
                    "overlap",
                    "confidence",
                    "diarizer_version",
                ]
            )
        )
        self._atomic_parquet(df, os.path.join(output_dir, "speaker_turns.parquet"))

    def _write_metadata(
        self,
        wav_path: str,
        mono_audio: Any,
        session_id: str,
        stream_id: str,
        config_hash: str,
        output_dir: str,
    ) -> None:
        # Hash the input file for provenance
        try:
            with open(wav_path, "rb") as f:
                input_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            input_hash = "unavailable"

        metadata = {
            "session_id": session_id,
            "stream_id": stream_id,
            "input_file": os.path.basename(wav_path),
            "input_file_sha256": input_hash,
            "original_sample_rate": mono_audio.sample_rate,
            "normalized_sample_rate": TARGET_SR,
            "duration_seconds": mono_audio.duration_seconds,
            "config_hash": config_hash,
            "models": {
                "vad": self._vad.get_version() if self._vad is not None else None,
                "yamnet": (
                    self._yamnet.get_version() if self._yamnet is not None else None
                ),
                "diarizer": (
                    self._diarizer.get_version() if self._diarizer is not None else None
                ),
            },
            "model_hashes": self._build_model_hashes(),
            "processing_status": "complete",
            "warnings": [],
        }
        self._atomic_json(
            metadata, os.path.join(output_dir, "extraction_metadata.json")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_config_hash(self) -> str:
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    def _build_extractor_versions(self) -> Dict[str, str]:
        versions: Dict[str, str] = {
            "resampler": "librosa",
            "audio_reader": "soundfile",
            "quality_metrics": "native",
        }
        if self._vad is not None:
            versions["vad"] = self._vad.get_version()
        if self._yamnet is not None:
            versions["yamnet"] = self._yamnet.get_version()
        if self._diarizer is not None:
            versions["diarizer"] = self._diarizer.get_version()
        return versions

    def _build_model_hashes(self) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        if self._yamnet is not None:
            h = self._yamnet.get_model_hash()
            if h is not None:
                hashes["yamnet"] = h
        return hashes

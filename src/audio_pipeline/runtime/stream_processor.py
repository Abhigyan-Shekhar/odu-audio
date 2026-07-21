"""
Real-time streaming audio processor coordinator.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Iterator, Optional

import numpy as np

from src.audio_pipeline.capture.audio_chunk import AudioChunk
from src.audio_pipeline.capture.audio_source import AudioSource
from src.audio_pipeline.capture.bounded_queue import BoundedQueue
from src.audio_pipeline.capture.ring_buffer import RingBuffer
from src.audio_pipeline.features.feature_extractor import FeatureExtractor
from src.audio_pipeline.preprocessing.channel_mixer import to_mono
from src.audio_pipeline.preprocessing.streaming_resampler import StreamingResampler
from src.audio_pipeline.quality.clipping import detect_clipping
from src.audio_pipeline.quality.dropout import detect_dropout
from src.audio_pipeline.quality.snr import estimate_snr
from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord
from src.audio_pipeline.schemas.segment import SpeechSegment
from src.audio_pipeline.segmentation.endpointer import Endpointer
from src.audio_pipeline.segmentation.vad_interface import VADInterface

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Coordinates real-time streaming audio capture, preprocessing,
    VAD segmentation, and feature/quality extraction in background threads.
    """

    def __init__(
        self,
        source: AudioSource,
        config: Optional[Dict[str, Any]] = None,
        vad: Optional[VADInterface] = None,
        egemaps_extractor: Optional[FeatureExtractor] = None,
        yamnet_detector: Optional[FeatureExtractor] = None,
    ):
        self.source = source
        self.config = config or {}

        # Features configuration
        features_cfg = self.config.get("features", {})
        egemaps_cfg = features_cfg.get("egemaps", {})
        self.window_seconds = egemaps_cfg.get("window_seconds", 2.0)
        self.hop_seconds = egemaps_cfg.get("hop_seconds", 0.5)

        yamnet_cfg = features_cfg.get("yamnet", {})
        self.event_classes = yamnet_cfg.get(
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

        # VAD & Segmentation Configuration
        seg_cfg = self.config.get("segmentation", {})
        self.vad_model = seg_cfg.get("model", "silero_vad_v4")
        self.vad_threshold = seg_cfg.get("threshold", 0.5)
        self.min_speech_duration_ms = seg_cfg.get("min_speech_duration_ms", 250)
        self.min_silence_duration_ms = seg_cfg.get("min_silence_duration_ms", 500)
        self.speech_pad_ms = seg_cfg.get("speech_pad_ms", 100)

        # eGeMAPS Configuration
        self.egemaps_enabled = egemaps_cfg.get(
            "enable", egemaps_cfg.get("enabled", True)
        )
        self.egemaps_timeout = egemaps_cfg.get("timeout_seconds", 5.0)
        self.min_voiced_ratio = egemaps_cfg.get(
            "minimum_voiced_ratio", egemaps_cfg.get("min_voiced_ratio", 0.40)
        )
        self.egemaps_on_failure = egemaps_cfg.get(
            "on_failure", "emit_none"
        )  # emit_none | skip_window | fail_pipeline
        self.egemaps_feature_set = egemaps_cfg.get("feature_set", "eGeMAPSv02")
        self.egemaps_feature_level = egemaps_cfg.get("feature_level", "Functionals")

        self._egemaps_extractor: Optional[FeatureExtractor] = None
        if egemaps_extractor is not None:
            self._egemaps_extractor = egemaps_extractor
        else:
            if self.egemaps_enabled:
                from src.audio_pipeline.features.opensmile_extractor import (
                    OpenSmileExtractor,
                )

                self._egemaps_extractor = OpenSmileExtractor(
                    feature_set=self.egemaps_feature_set,
                    feature_level=self.egemaps_feature_level,
                    timeout_seconds=self.egemaps_timeout,
                )
            else:
                self._egemaps_extractor = None

        # YAMNet Configuration
        self.yamnet_enabled = yamnet_cfg.get("enable", yamnet_cfg.get("enabled", True))
        self.yamnet_timeout = yamnet_cfg.get("timeout_seconds", 5.0)
        self.yamnet_min_probability = yamnet_cfg.get("min_probability", 0.1)

        self._yamnet_extractor: Optional[FeatureExtractor] = None
        if yamnet_detector is not None:
            self._yamnet_extractor = yamnet_detector
        else:
            if self.yamnet_enabled:
                from src.audio_pipeline.features.yamnet_detector import YAMNetDetector

                self._yamnet_extractor = YAMNetDetector(
                    target_events=self.event_classes,
                    min_probability=self.yamnet_min_probability,
                    timeout_seconds=self.yamnet_timeout,
                )
            else:
                self._yamnet_extractor = None

        # Runtime configuration
        runtime_cfg = self.config.get("runtime", {})
        self.queue_capacity = runtime_cfg.get("queue_capacity", 100)

        # Internal buffers and queues
        self._chunk_queue: BoundedQueue[AudioChunk] = BoundedQueue(
            maxsize=self.queue_capacity
        )
        self._output_queue: BoundedQueue[AcousticFeatureRecord] = BoundedQueue(
            maxsize=1000
        )
        self._ring_buffer: Optional[RingBuffer] = None
        self._resampler: Optional[StreamingResampler] = None

        # VAD state tracking
        self.speech_segments: list[SpeechSegment] = []
        self._vad_history: list[tuple[int, float]] = []
        self._vad_sample_buffer = np.array([], dtype=np.float32)
        self._vad_sample_counter = 0

        if vad is not None:
            self._vad = vad
        else:
            if self.vad_model == "silero_vad_v4":
                from src.audio_pipeline.segmentation.silero_vad import SileroVAD

                self._vad = SileroVAD(threshold=self.vad_threshold)
            else:
                from src.audio_pipeline.segmentation.dummy_vad import DummyVAD

                self._vad = DummyVAD(default_prob=0.0)

        self._endpointer = Endpointer(
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sample_rate=16000,
        )

        # Threading state
        self._running = False
        self._capture_done = False
        self._process_done = False
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # Session metadata
        self._session_id = (
            f"session_{hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]}"
        )
        self._stream_id = (
            f"stream_{hashlib.md5(str(time.time_ns() + 1).encode()).hexdigest()[:8]}"
        )

    def start(self) -> None:
        """Start the capture and processing threads."""
        if self._running:
            return

        self._running = True
        self._capture_done = False
        self._process_done = False
        self._stop_event.clear()

        # Reset VAD buffers
        self.speech_segments = []
        self._vad_history = []
        self._vad_sample_buffer = np.array([], dtype=np.float32)
        self._vad_sample_counter = 0

        # Instantiate streaming resampler and ring buffer (using target 16kHz)
        self._resampler = StreamingResampler(
            source_sr=self.source.sample_rate, target_sr=16000
        )
        self._ring_buffer = RingBuffer(capacity_seconds=60.0, sample_rate=16000)

        # Start background threads
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="audio-capture-thread", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, name="audio-process-thread", daemon=True
        )

        self._capture_thread.start()
        self._process_thread.start()
        logger.info("StreamProcessor started successfully.")

    def stop(self) -> None:
        """Stop threads and release resources."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Join capture thread
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        # Join processing thread
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        self.source.close()
        logger.info("StreamProcessor stopped and audio source closed.")

    def _capture_loop(self) -> None:
        """Reads audio chunks from the AudioSource and writes to bounded chunk queue."""
        try:
            while self._running and not self._stop_event.is_set():
                try:
                    chunk = self.source.read_chunk()
                    # Put chunk onto queue. 0.1s block timeout before dropping under backpressure
                    success = self._chunk_queue.put(chunk, timeout=0.1)
                    if not success:
                        logger.warning(
                            "Ingestion backpressure: queue full, chunk dropped."
                        )
                        # If queue is full, mark a discontinuity in the ring buffer
                        if self._ring_buffer is not None:
                            self._ring_buffer.mark_discontinuity(
                                reason="BUFFER_OVERFLOW"
                            )
                except EOFError:
                    # Normal termination (e.g. file source reached EOF)
                    logger.info("Ingestion complete (EOF reached).")
                    break
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}")
                    if self._ring_buffer is not None:
                        self._ring_buffer.mark_discontinuity(reason="DEVICE_ERROR")
                    break
        finally:
            self._capture_done = True

    def _process_loop(self) -> None:
        """Processes chunks from the queue, handles resampling, windowing, and metrics."""
        try:
            assert self._ring_buffer is not None
            assert self._resampler is not None

            window_samples = int(self.window_seconds * 16000)
            hop_samples = int(self.hop_seconds * 16000)
            config_hash = hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()
            ).hexdigest()[:8]

            cumulative_resampled_samples = 0
            sequence_number = 0
            original_sr = self.source.sample_rate

            while not self._capture_done or self._chunk_queue.qsize() > 0:
                chunk = self._chunk_queue.get(timeout=0.1)
                if chunk is None:
                    if self._capture_done:
                        # Ingestion has stopped and queue is empty, terminate processing loop
                        break
                    continue

                # 1. Downmix to mono
                mono_samples = to_mono(chunk.samples)

                # 2. Resample statefully to 16 kHz
                resampled = self._resampler.process_chunk(mono_samples)

                # 3. Propagate chunk-level discontinuities to ring buffer
                if chunk.discontinuity:
                    self._ring_buffer.mark_discontinuity(
                        reason=chunk.discontinuity_reason or "BUFFER_OVERFLOW"
                    )

                # 4. Write to RingBuffer
                self._ring_buffer.write(resampled)

                # 5. Run VAD chunking and endpointing on resampled audio
                self._vad_sample_buffer = np.concatenate(
                    [self._vad_sample_buffer, resampled]
                )
                while len(self._vad_sample_buffer) >= 512:
                    vad_chunk = self._vad_sample_buffer[:512]
                    self._vad_sample_buffer = self._vad_sample_buffer[512:]

                    prob = self._vad.process_chunk(vad_chunk)

                    start_sample = self._vad_sample_counter
                    end_sample = self._vad_sample_counter + 512
                    self._vad_sample_counter += 512

                    start_ms = int((start_sample / 16000.0) * 1000)
                    end_ms = int((end_sample / 16000.0) * 1000)

                    # Record history for windowed average VAD probability
                    midpoint_ms = (start_ms + end_ms) // 2
                    self._vad_history.append((midpoint_ms, prob))

                    # Feed to endpointer
                    seg = self._endpointer.process(
                        prob=prob,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        session_id=self._session_id,
                        stream_id=self._stream_id,
                    )
                    if seg is not None and not seg.provisional:
                        self.speech_segments.append(seg)

                # 6. Extract features from overlapping windows
                ratio = original_sr / 16000.0

                while self._ring_buffer.get_available_samples() >= window_samples:
                    window_data = self._ring_buffer.peek(window_samples)
                    if window_data is None:
                        break

                    # Extract quality metrics
                    clipping_ratio = detect_clipping(
                        window_data, threshold=self.clipping_threshold
                    )
                    dropout_ratio = detect_dropout(
                        window_data, threshold=self.dropout_zero_threshold
                    )
                    snr_db = estimate_snr(window_data, sample_rate=16000)

                    # Determine quality status
                    if clipping_ratio >= self.clipping_max_ratio:
                        quality_status = "CLIPPED"
                    elif dropout_ratio >= self.dropout_max_ratio:
                        quality_status = "DROPOUT"
                    elif snr_db < self.snr_min_db:
                        quality_status = "LOW_SNR"
                    else:
                        quality_status = "OK"

                    # Map resampled windows back to source WAV sample indices
                    source_start_sample = int(cumulative_resampled_samples * ratio)
                    source_end_sample = int(
                        (cumulative_resampled_samples + window_samples) * ratio
                    )

                    # Calculate windowed VAD metrics
                    w_start_ms = int((cumulative_resampled_samples / 16000.0) * 1000)
                    w_end_ms = int(
                        ((cumulative_resampled_samples + window_samples) / 16000.0)
                        * 1000
                    )

                    overlapping = [
                        p for t, p in self._vad_history if w_start_ms <= t < w_end_ms
                    ]
                    if overlapping:
                        vad_probability_mean = float(np.mean(overlapping))
                        voiced_ratio = float(
                            np.sum(np.array(overlapping) >= self.vad_threshold)
                            / len(overlapping)
                        )
                    else:
                        vad_probability_mean = 0.0
                        voiced_ratio = 0.0

                    # Prune history to keep memory bounded
                    self._vad_history = [
                        (t, p) for t, p in self._vad_history if t >= w_start_ms - 5000
                    ]

                    # Extract eGeMAPS features if enabled
                    egemaps_features = None
                    skip_this_window = False
                    if self.egemaps_enabled and self._egemaps_extractor is not None:
                        if voiced_ratio < self.min_voiced_ratio:
                            if self.egemaps_on_failure == "skip_window":
                                skip_this_window = True
                            elif self.egemaps_on_failure == "fail_pipeline":
                                raise RuntimeError(
                                    f"Voiced ratio {voiced_ratio:.2f} is below threshold {self.min_voiced_ratio:.2f}"
                                )
                            else:
                                egemaps_features = None
                        else:
                            extracted = self._egemaps_extractor.extract(
                                window_data, sr=16000
                            )
                            if extracted is None:
                                if self.egemaps_on_failure == "skip_window":
                                    skip_this_window = True
                                elif self.egemaps_on_failure == "fail_pipeline":
                                    raise RuntimeError(
                                        "eGeMAPS feature extraction failed"
                                    )
                                else:
                                    egemaps_features = None
                            else:
                                egemaps_features = extracted.tolist()

                    if skip_this_window:
                        self._ring_buffer.advance(hop_samples)
                        cumulative_resampled_samples += hop_samples
                        sequence_number += 1
                        continue

                    # Extract YAMNet distress event scores (runs in parallel to VAD on the full stream)
                    yamnet_scores = {cls: 0.0 for cls in self.event_classes}
                    if self.yamnet_enabled and self._yamnet_extractor is not None:
                        extracted_events = self._yamnet_extractor.extract(
                            window_data, sr=16000
                        )
                        if extracted_events is not None and isinstance(
                            extracted_events, dict
                        ):
                            yamnet_scores = extracted_events

                    # Instantiating the dataclass triggers validator checks
                    record = AcousticFeatureRecord(
                        session_id=self._session_id,
                        stream_id=self._stream_id,
                        window_start_ms=w_start_ms,
                        window_end_ms=w_end_ms,
                        source_start_sample=source_start_sample,
                        source_end_sample=source_end_sample,
                        attribution_status="UNKNOWN",
                        patient_probability=None,
                        attribution_method=None,
                        vad_probability_mean=vad_probability_mean,
                        voiced_ratio=voiced_ratio,
                        overlap_probability=0.0,
                        egemaps=egemaps_features,
                        yamnet_event_scores=yamnet_scores,
                        emotion_embedding=None,
                        snr_db=snr_db,
                        clipping_ratio=clipping_ratio,
                        dropout_ratio=dropout_ratio,
                        quality_status=quality_status,
                        extractor_versions={
                            "resampler": "librosa",
                            "audio_reader": "soundfile",
                            "quality_metrics": "native",
                            "egemaps": (
                                self._egemaps_extractor.get_version()
                                if self._egemaps_extractor
                                else "none"
                            ),
                            "yamnet": (
                                self._yamnet_extractor.get_version()
                                if self._yamnet_extractor
                                else "none"
                            ),
                        },
                        config_hash=config_hash,
                        model_hashes={
                            "egemaps": (
                                self._egemaps_extractor.get_model_hash() or ""
                                if self._egemaps_extractor
                                else ""
                            ),
                            "yamnet": (
                                self._yamnet_extractor.get_model_hash() or ""
                                if self._yamnet_extractor
                                else ""
                            ),
                        },
                    )

                    # Put to output queue (0.1s block timeout to control output backpressure)
                    self._output_queue.put(record, timeout=0.1)

                    # Advance sliding window by hop length
                    self._ring_buffer.advance(hop_samples)
                    cumulative_resampled_samples += hop_samples
                    sequence_number += 1

            # 7. Process any remaining trailing VAD samples and flush endpointer
            if len(self._vad_sample_buffer) > 0:
                pad_size = 512 - len(self._vad_sample_buffer)
                final_chunk = np.concatenate(
                    [self._vad_sample_buffer, np.zeros(pad_size, dtype=np.float32)]
                )
                prob = self._vad.process_chunk(final_chunk)

                start_sample = self._vad_sample_counter
                end_sample = self._vad_sample_counter + len(self._vad_sample_buffer)
                self._vad_sample_counter += len(self._vad_sample_buffer)

                start_ms = int((start_sample / 16000.0) * 1000)
                end_ms = int((end_sample / 16000.0) * 1000)

                midpoint_ms = (start_ms + end_ms) // 2
                self._vad_history.append((midpoint_ms, prob))

                seg = self._endpointer.process(
                    prob=prob,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    session_id=self._session_id,
                    stream_id=self._stream_id,
                )
                if seg is not None and not seg.provisional:
                    self.speech_segments.append(seg)
                self._vad_sample_buffer = np.array([], dtype=np.float32)

            # Flush the endpointer
            final_seg = self._endpointer.flush(self._session_id, self._stream_id)
            if final_seg is not None and not final_seg.provisional:
                self.speech_segments.append(final_seg)

            # 8. Handle final partial window on shutdown/EOF
            available = self._ring_buffer.get_available_samples()
            if available > 0:
                window_data = self._ring_buffer.peek(available)
                if window_data is not None and len(window_data) > 0:
                    clipping_ratio = detect_clipping(
                        window_data, threshold=self.clipping_threshold
                    )
                    dropout_ratio = detect_dropout(
                        window_data, threshold=self.dropout_zero_threshold
                    )
                    snr_db = estimate_snr(window_data, sample_rate=16000)

                    if clipping_ratio >= self.clipping_max_ratio:
                        quality_status = "CLIPPED"
                    elif dropout_ratio >= self.dropout_max_ratio:
                        quality_status = "DROPOUT"
                    elif snr_db < self.snr_min_db:
                        quality_status = "LOW_SNR"
                    else:
                        quality_status = "OK"

                    source_start_sample = int(cumulative_resampled_samples * ratio)
                    source_end_sample = int(
                        (cumulative_resampled_samples + available) * ratio
                    )

                    w_start_ms = int((cumulative_resampled_samples / 16000.0) * 1000)
                    w_end_ms = int(
                        ((cumulative_resampled_samples + available) / 16000.0) * 1000
                    )

                    overlapping = [
                        p for t, p in self._vad_history if w_start_ms <= t < w_end_ms
                    ]
                    if overlapping:
                        vad_probability_mean = float(np.mean(overlapping))
                        voiced_ratio = float(
                            np.sum(np.array(overlapping) >= self.vad_threshold)
                            / len(overlapping)
                        )
                    else:
                        vad_probability_mean = 0.0
                        voiced_ratio = 0.0

                    egemaps_features = None
                    skip_this_window = False
                    if self.egemaps_enabled and self._egemaps_extractor is not None:
                        if voiced_ratio < self.min_voiced_ratio:
                            if self.egemaps_on_failure == "skip_window":
                                skip_this_window = True
                            elif self.egemaps_on_failure == "fail_pipeline":
                                raise RuntimeError(
                                    f"Voiced ratio {voiced_ratio:.2f} is below threshold {self.min_voiced_ratio:.2f}"
                                )
                            else:
                                egemaps_features = None
                        else:
                            extracted = self._egemaps_extractor.extract(
                                window_data, sr=16000
                            )
                            if extracted is None:
                                if self.egemaps_on_failure == "skip_window":
                                    skip_this_window = True
                                elif self.egemaps_on_failure == "fail_pipeline":
                                    raise RuntimeError(
                                        "eGeMAPS feature extraction failed"
                                    )
                                else:
                                    egemaps_features = None
                            else:
                                egemaps_features = extracted.tolist()

                    yamnet_scores = {cls: 0.0 for cls in self.event_classes}
                    if self.yamnet_enabled and self._yamnet_extractor is not None:
                        extracted_events = self._yamnet_extractor.extract(
                            window_data, sr=16000
                        )
                        if extracted_events is not None and isinstance(
                            extracted_events, dict
                        ):
                            yamnet_scores = extracted_events

                    if not skip_this_window:
                        record = AcousticFeatureRecord(
                            session_id=self._session_id,
                            stream_id=self._stream_id,
                            window_start_ms=w_start_ms,
                            window_end_ms=w_end_ms,
                            source_start_sample=source_start_sample,
                            source_end_sample=source_end_sample,
                            attribution_status="UNKNOWN",
                            patient_probability=None,
                            attribution_method=None,
                            vad_probability_mean=vad_probability_mean,
                            voiced_ratio=voiced_ratio,
                            overlap_probability=0.0,
                            egemaps=egemaps_features,
                            yamnet_event_scores=yamnet_scores,
                            emotion_embedding=None,
                            snr_db=snr_db,
                            clipping_ratio=clipping_ratio,
                            dropout_ratio=dropout_ratio,
                            quality_status=quality_status,
                            extractor_versions={
                                "resampler": "librosa",
                                "audio_reader": "soundfile",
                                "quality_metrics": "native",
                                "egemaps": (
                                    self._egemaps_extractor.get_version()
                                    if self._egemaps_extractor
                                    else "none"
                                ),
                                "yamnet": (
                                    self._yamnet_extractor.get_version()
                                    if self._yamnet_extractor
                                    else "none"
                                ),
                            },
                            config_hash=config_hash,
                            model_hashes={
                                "egemaps": (
                                    self._egemaps_extractor.get_model_hash() or ""
                                    if self._egemaps_extractor
                                    else ""
                                ),
                                "yamnet": (
                                    self._yamnet_extractor.get_model_hash() or ""
                                    if self._yamnet_extractor
                                    else ""
                                ),
                            },
                        )
                        self._output_queue.put(record, timeout=0.1)
        finally:
            self._process_done = True
            self._running = False

    def stream(self) -> Iterator[AcousticFeatureRecord]:
        """
        Yield extracted feature records as they are produced in real-time.
        """
        while not self._process_done or self._output_queue.qsize() > 0:
            record = self._output_queue.get(timeout=0.1)
            if record is not None:
                yield record
            else:
                if self._process_done:
                    # Thread stopped and no more outputs left
                    break

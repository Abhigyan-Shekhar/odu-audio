"""Faster-Whisper ASR integration boundary."""

from typing import Optional

import numpy as np

from src.audio_pipeline.lexical.language import identify_language
from src.audio_pipeline.schemas.lexical_record import TranscriptSegment, WordTimestamp
from src.audio_pipeline.schemas.segment import SpeechSegment


class FasterWhisperASR:
    """Lazy Faster-Whisper wrapper with injectable model for tests."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        model=None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = model

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is not installed. Install it to run ASR inference."
                ) from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment: SpeechSegment,
        language: Optional[str] = None,
    ) -> TranscriptSegment:
        """
        Transcribe one speech segment.

        Faster-Whisper expects 16 kHz mono audio; callers should resample upstream.
        """
        if sample_rate != 16000:
            raise ValueError("FasterWhisperASR expects 16 kHz mono audio")

        model = self._load_model()
        segments, info = model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            vad_filter=False,
        )

        text_parts: list[str] = []
        words: list[WordTimestamp] = []
        confidences: list[float] = []

        for asr_segment in segments:
            text_parts.append(asr_segment.text.strip())
            for word in getattr(asr_segment, "words", []) or []:
                probability = float(getattr(word, "probability", 0.0) or 0.0)
                confidences.append(probability)
                words.append(
                    WordTimestamp(
                        word=str(word.word).strip(),
                        start_ms=segment.start_ms + int(float(word.start) * 1000),
                        end_ms=segment.start_ms + int(float(word.end) * 1000),
                        confidence=probability,
                    )
                )

        text = " ".join(part for part in text_parts if part).strip()
        fallback_language, fallback_prob, code_mixed = identify_language(text)
        detected_language = getattr(info, "language", None) or fallback_language
        language_probability = float(
            getattr(info, "language_probability", fallback_prob) or fallback_prob
        )
        avg_confidence = (
            float(sum(confidences) / len(confidences)) if confidences else 0.0
        )

        return TranscriptSegment(
            session_id=segment.session_id,
            stream_id=segment.stream_id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            text=text,
            language=detected_language,
            language_probability=language_probability,
            avg_confidence=avg_confidence,
            words=words,
            asr_model=f"faster-whisper:{self.model_size}",
            incomplete=not text.endswith((".", "?", "!", "।")),
            code_mixed=code_mixed,
        )

    def get_version(self) -> str:
        return f"faster-whisper:{self.model_size}"

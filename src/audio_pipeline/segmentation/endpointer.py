"""
VAD-driven speech endpointer state machine.
"""

from typing import Optional

import numpy as np

from src.audio_pipeline.schemas.segment import SpeechSegment


class Endpointer:
    """
    State machine that converts VAD probabilities into finalized SpeechSegments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        self._active_segment: Optional[dict] = None

    def process(
        self,
        prob: float,
        start_ms: int,
        end_ms: int,
        start_sample: int,
        end_sample: int,
        session_id: str,
        stream_id: str,
    ) -> Optional[SpeechSegment]:
        """
        Process a single VAD probability and return a SpeechSegment if updated or finalized.
        """
        chunk_duration_ms = end_ms - start_ms

        if prob >= self.threshold:
            if self._active_segment is None:
                # Start of a new speech segment
                padded_start_ms = max(0, start_ms - self.speech_pad_ms)
                padded_start_sample = max(
                    0,
                    start_sample
                    - int(self.speech_pad_ms * (self.sample_rate / 1000.0)),
                )
                self._active_segment = {
                    "start_ms": start_ms,
                    "start_sample": start_sample,
                    "last_speech_end_ms": end_ms,
                    "last_speech_end_sample": end_sample,
                    "padded_start_ms": padded_start_ms,
                    "padded_start_sample": padded_start_sample,
                    "probabilities": [prob],
                    "silence_duration_ms": 0,
                }
            else:
                # Extending active speech segment
                self._active_segment["last_speech_end_ms"] = end_ms
                self._active_segment["last_speech_end_sample"] = end_sample
                self._active_segment["probabilities"].append(prob)
                self._active_segment["silence_duration_ms"] = 0

            # Yield provisional segment
            return self._create_segment_object(
                self._active_segment, session_id, stream_id, provisional=True
            )

        else:
            # Silence chunk
            if self._active_segment is not None:
                self._active_segment["silence_duration_ms"] += chunk_duration_ms

                # Check if silence threshold is exceeded to finalize segment
                if (
                    self._active_segment["silence_duration_ms"]
                    >= self.min_silence_duration_ms
                ):
                    final_seg = self._finalize_active_segment(session_id, stream_id)
                    return final_seg
                else:
                    # Still provisional but currently in a silence pad
                    return self._create_segment_object(
                        self._active_segment, session_id, stream_id, provisional=True
                    )

        return None

    def flush(self, session_id: str, stream_id: str) -> Optional[SpeechSegment]:
        """
        Force finalization of any active provisional segment at the end of the stream.
        """
        if self._active_segment is not None:
            return self._finalize_active_segment(session_id, stream_id)
        return None

    def _finalize_active_segment(
        self, session_id: str, stream_id: str
    ) -> Optional[SpeechSegment]:
        """Finalize the active speech segment, checking duration and adding padding."""
        seg = self._active_segment
        self._active_segment = None

        if seg is None:
            return None

        # Check speech duration without silence/tail padding
        speech_duration_ms = seg["last_speech_end_ms"] - seg["start_ms"]
        if speech_duration_ms < self.min_speech_duration_ms:
            return None

        # Yield finalized non-provisional segment
        return self._create_segment_object(
            seg, session_id, stream_id, provisional=False
        )

    def _create_segment_object(
        self, seg: dict, session_id: str, stream_id: str, provisional: bool
    ) -> SpeechSegment:
        """Helper to package the segment dictionary into a SpeechSegment dataclass."""
        probs = seg["probabilities"]
        mean_prob = float(np.mean(probs))
        min_prob = float(np.min(probs))

        # Add tail padding for finalization
        if not provisional:
            # Finalized segment: add speech_pad_ms at the end
            padded_end_ms = seg["last_speech_end_ms"] + self.speech_pad_ms
            padded_end_sample = seg["last_speech_end_sample"] + int(
                self.speech_pad_ms * (self.sample_rate / 1000.0)
            )
        else:
            # Provisional segment: end times match the last speech chunk
            padded_end_ms = seg["last_speech_end_ms"]
            padded_end_sample = seg["last_speech_end_sample"]

        return SpeechSegment(
            start_ms=seg["padded_start_ms"],
            end_ms=padded_end_ms,
            start_sample=seg["padded_start_sample"],
            end_sample=padded_end_sample,
            vad_probability_mean=mean_prob,
            vad_probability_min=min_prob,
            provisional=provisional,
            session_id=session_id,
            stream_id=stream_id,
        )

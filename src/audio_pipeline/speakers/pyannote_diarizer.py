"""
Pyannote.audio speaker diarization backend.

Wraps the pyannote speaker diarization pipeline, accepting audio as an
in-memory waveform tensor to avoid FFmpeg / torchcodec dependency.
"""

import concurrent.futures
import logging
from typing import Optional

import numpy as np
import torch

from src.audio_pipeline.speakers.diarizer_interface import (
    DiarizationResult,
    DiarizedTurn,
    DiarizeriInterface,
)

logger = logging.getLogger(__name__)

_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer(DiarizeriInterface):
    """
    Speaker diarizer using pyannote.audio 3.x pipeline.

    The pipeline accepts pre-loaded waveform tensors directly, so no
    FFmpeg / torchcodec is required at inference time.

    NOTE: Loading this model requires:
    1. A Hugging Face access token (hf_token).
    2. Accepting pyannote model terms at:
       https://huggingface.co/pyannote/speaker-diarization-3.1

    For offline / CI use, use DummyDiarizer instead.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = _PYANNOTE_MODEL,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        timeout_seconds: float = 30.0,
        device: str = "cpu",
    ):
        """
        Args:
            hf_token:        HuggingFace access token. May also be set via
                             the HF_TOKEN environment variable.
            model_name:      HuggingFace model ID for the diarization pipeline.
            min_speakers:    Lower bound on speaker count hint (optional).
            max_speakers:    Upper bound on speaker count hint (optional).
            timeout_seconds: Per-segment inference timeout.
            device:          Torch device string, e.g. 'cpu', 'cuda', 'mps'.
        """
        import os

        from pyannote.audio import Pipeline

        token = hf_token or os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "A HuggingFace token is required to load the pyannote "
                "speaker-diarization pipeline. Pass hf_token= or set the "
                "HF_TOKEN environment variable."
            )

        logger.info(f"Loading pyannote pipeline: {model_name}")
        pipeline = Pipeline.from_pretrained(model_name, token=token)
        if pipeline is None:
            raise RuntimeError(
                f"Failed to load pyannote pipeline '{model_name}'. "
                "Ensure the model name is correct and the HF token is valid."
            )
        self._pipeline: Pipeline = pipeline
        self._pipeline.to(torch.device(device))
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.timeout_seconds = timeout_seconds
        self._model_name = model_name

    def diarize(self, audio: np.ndarray, sr: int) -> DiarizationResult:
        """
        Run pyannote diarization on a mono float32 waveform.
        """
        if audio.ndim != 1:
            logger.error("PyannoteDiarizer expects 1D mono audio.")
            return DiarizationResult(diarizer_version=self.get_version())

        # Convert to [channel, time] tensor as required by pyannote
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        input_dict = {"waveform": waveform, "sample_rate": sr}

        kwargs: dict = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._pipeline.__call__, input_dict, **kwargs  # type: ignore[operator]
            )
            try:
                annotation = future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"PyannoteDiarizer timed out after {self.timeout_seconds}s"
                )
                return DiarizationResult(diarizer_version=self.get_version())
            except Exception as e:
                logger.error(f"PyannoteDiarizer inference failed: {e}")
                return DiarizationResult(diarizer_version=self.get_version())

        return self._annotation_to_result(annotation)

    def _annotation_to_result(self, annotation) -> DiarizationResult:
        """Convert pyannote Annotation to DiarizationResult."""

        turns = []
        speaker_ids: set[str] = set()

        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            turns.append(
                DiarizedTurn(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_id=speaker,
                )
            )
            speaker_ids.add(speaker)

        # Detect overlapping turns
        overlapping: list[str] = []
        for i, t1 in enumerate(turns):
            for t2 in turns[i + 1 :]:
                if t1.start_ms < t2.end_ms and t1.end_ms > t2.start_ms:
                    t1.overlap = True
                    t2.overlap = True
                    if t1.speaker_id not in overlapping:
                        overlapping.append(t1.speaker_id)
                    if t2.speaker_id not in overlapping:
                        overlapping.append(t2.speaker_id)

        return DiarizationResult(
            turns=turns,
            num_speakers=len(speaker_ids),
            has_overlap=len(overlapping) > 0,
            overlapping_speaker_ids=overlapping,
            diarizer_version=self.get_version(),
        )

    def get_version(self) -> str:
        return f"pyannote_{self._model_name.replace('/', '_')}"

"""
Offline VAD helper — runs Silero (or any VADInterface) frame-by-frame over a
complete waveform and returns per-frame probabilities that can be aggregated
into any window size downstream.

This avoids duplicating the streaming logic and allows unit tests to inject
DummyVAD without downloading the Silero model.
"""

from typing import List, NamedTuple, Tuple

import numpy as np

from src.audio_pipeline.segmentation.vad_interface import VADInterface

# Silero expects exactly 512 samples at 16 kHz for v4
VAD_CHUNK_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


class VadFrame(NamedTuple):
    """A single VAD inference result for one 512-sample chunk."""

    start_sample: int
    """First sample index of this chunk (inclusive)."""

    end_sample: int
    """Last sample index of this chunk (exclusive, may be ≤ len(audio) for padding)."""

    start_ms: int
    """Start time in milliseconds from the beginning of the audio."""

    end_ms: int
    """End time in milliseconds from the beginning of the audio."""

    probability: float
    """Speech probability returned by the VAD model (0.0–1.0)."""


def run_vad_frames(audio: np.ndarray, vad: VADInterface) -> List[VadFrame]:
    """
    Run the VAD model on every 512-sample chunk of *audio*.

    The final incomplete chunk is zero-padded to exactly 512 samples before
    inference, but its ``end_sample`` is recorded as ``len(audio)`` (not
    padded) so that callers can use it for accurate time mapping.

    Args:
        audio: 1-D float32 numpy array at 16 kHz (mono).
        vad:   Any VADInterface implementation (real Silero or DummyVAD).

    Returns:
        A list of :class:`VadFrame` objects, one per 512-sample chunk.
    """
    if audio.ndim != 1:
        raise ValueError("run_vad_frames expects a 1D mono audio array.")

    frames: List[VadFrame] = []
    n = len(audio)

    if n == 0:
        return frames

    for start in range(0, n, VAD_CHUNK_SAMPLES):
        end = min(start + VAD_CHUNK_SAMPLES, n)
        chunk = audio[start:end]

        # Zero-pad the final incomplete chunk
        if len(chunk) < VAD_CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, VAD_CHUNK_SAMPLES - len(chunk)))

        prob = vad.process_chunk(chunk)

        start_ms = int(start / VAD_SAMPLE_RATE * 1000)
        # Use the *actual* (non-padded) end for time accuracy
        end_ms = int(end / VAD_SAMPLE_RATE * 1000)

        frames.append(VadFrame(start, end, start_ms, end_ms, float(prob)))

    return frames


def aggregate_vad_for_window(
    frames: List[VadFrame],
    window_start_ms: int,
    window_end_ms: int,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Aggregate VAD frame probabilities that overlap a feature window.

    A frame is considered overlapping if its time interval intersects
    ``[window_start_ms, window_end_ms)``.

    Args:
        frames:          All frames produced by :func:`run_vad_frames`.
        window_start_ms: Feature window start in milliseconds.
        window_end_ms:   Feature window end in milliseconds.
        threshold:       Probability threshold above which a frame is voiced.

    Returns:
        A tuple ``(mean_probability, voiced_ratio)``:
        - ``mean_probability``: Mean speech probability across overlapping frames.
        - ``voiced_ratio``: Fraction of overlapping frames with probability ≥ threshold.
        Both values are 0.0 if no frames overlap the window.
    """
    overlapping = [
        f for f in frames if f.start_ms < window_end_ms and f.end_ms > window_start_ms
    ]

    if not overlapping:
        return 0.0, 0.0

    probs = [f.probability for f in overlapping]
    mean_prob = float(np.mean(probs))
    voiced_ratio = sum(1 for p in probs if p >= threshold) / len(probs)

    return mean_prob, float(voiced_ratio)

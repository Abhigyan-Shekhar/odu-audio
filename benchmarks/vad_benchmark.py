"""
Voice Activity Detection (VAD) benchmark suite.
"""

import time

import numpy as np

from src.audio_pipeline.segmentation.endpointer import Endpointer
from src.audio_pipeline.segmentation.silero_vad import SileroVAD


def benchmark_vad_inference_time():
    """Measure average CPU inference time per 512-sample (32ms) chunk."""
    print("Running inference benchmark...")
    vad = SileroVAD()
    chunk = np.random.normal(0, 0.1, 512).astype(np.float32)

    # Warm up
    for _ in range(10):
        _ = vad.process_chunk(chunk)

    # Measure
    n_iterations = 200
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = vad.process_chunk(chunk)
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_iterations) * 1000
    print(f"Average CPU Inference time: {avg_ms:.4f} ms per chunk (target: <5 ms)")
    return avg_ms


def benchmark_vad_onset_and_offset_delay():
    """Measure detection delay for onset (start) and offset (end) of speech."""
    print("Running delay benchmark...")
    # Endpointer: threshold = 0.5, silence = 300ms, padding = 100ms
    endpointer = Endpointer(
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        sample_rate=16000,
    )

    # Create a stream of chunks:
    # 0.0 .. 1.0s: silence (prob = 0.1)
    # 1.0 .. 2.0s: speech (prob = 0.9)
    # 2.0 .. 3.0s: silence (prob = 0.1)
    # Chunk size: 32ms (512 samples)
    chunk_dur_ms = 32
    speech_start_ms = 1000
    speech_end_ms = 2000

    detected_start_ms = None
    detected_end_ms = None

    for t_ms in range(0, 3000, chunk_dur_ms):
        is_speech = speech_start_ms <= t_ms < speech_end_ms
        prob = 0.9 if is_speech else 0.1

        seg = endpointer.process(
            prob=prob,
            start_ms=t_ms,
            end_ms=t_ms + chunk_dur_ms,
            start_sample=int(t_ms * 16.0),
            end_sample=int((t_ms + chunk_dur_ms) * 16.0),
            session_id="bench",
            stream_id="bench",
        )

        if seg is not None:
            # Check for onset
            if seg.provisional and detected_start_ms is None:
                detected_start_ms = (
                    seg.start_ms + endpointer.speech_pad_ms
                )  # Remove start padding for true onset time

            # Check for offset
            if not seg.provisional:
                # Decision finalized at current chunk end time
                detected_end_ms = t_ms + chunk_dur_ms

    onset_delay = detected_start_ms - speech_start_ms if detected_start_ms else 9999
    offset_delay = detected_end_ms - speech_end_ms if detected_end_ms else 9999

    print(f"Speech onset delay: {onset_delay} ms (target: <50 ms)")
    print(f"Speech offset delay: {offset_delay} ms (target: 250-500 ms)")

    return onset_delay, offset_delay


def main():
    print("=" * 50)
    print("Voice Activity Detection Benchmark")
    print("=" * 50)

    inf_time = benchmark_vad_inference_time()
    onset_delay, offset_delay = benchmark_vad_onset_and_offset_delay()

    print("=" * 50)
    # Validate targets
    assert inf_time < 5.0, f"Inference too slow: {inf_time:.2f} ms"
    assert onset_delay <= 50, f"Onset delay too high: {onset_delay} ms"
    assert 200 <= offset_delay <= 600, f"Offset delay out of bounds: {offset_delay} ms"
    print("All VAD benchmark targets satisfied successfully!")


if __name__ == "__main__":
    main()

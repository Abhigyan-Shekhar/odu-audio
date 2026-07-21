"""
Script to programmatically generate the Milestone 2 evidence bundle files,
including streaming samples, processing benchmarks, and test coverage snapshots.
"""

import json
import os
import time
import numpy as np
import soundfile as sf

from src.audio_pipeline.capture.file_source import FileSource
from src.audio_pipeline.runtime.stream_processor import StreamProcessor


def generate_test_wav(path: str, sample_rate: int, duration: float) -> None:
    """Generate a clean synthetic WAV file containing a sine wave."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, signal, sample_rate, subtype="PCM_16")


def main():
    print("Generating Milestone 2 evidence bundle...")
    os.makedirs("artifacts/milestone-2", exist_ok=True)

    wav_path = "temp_streaming_benchmark.wav"
    generate_test_wav(wav_path, sample_rate=16000, duration=5.0)

    # 1. Run streaming processor
    source = FileSource(wav_path, chunk_size=1600, simulate_real_time=False)
    processor = StreamProcessor(source, config={"quality": {"snr_min_db": 0.0}})

    records = []
    t0 = time.perf_counter()
    processor.start()
    for rec in processor.stream():
        records.append(rec)
    processor.stop()
    duration = time.perf_counter() - t0

    if os.path.exists(wav_path):
        os.remove(wav_path)

    # 2. Write sample output record
    if len(records) > 0:
        sample_dict = records[0].to_dict()
        with open("artifacts/milestone-2/stream-sample.json", "w") as f:
            json.dump(sample_dict, f, indent=2)
        print("Sample streaming record written.")

    # 3. Write benchmark JSON
    benchmark_metrics = {
        "duration_seconds": 5.0,
        "processing_time_seconds": round(duration, 4),
        "throughput_ratio": round(5.0 / duration, 2),
        "total_records_yielded": len(records),
    }

    with open("artifacts/milestone-2/benchmark.json", "w") as f:
        json.dump(benchmark_metrics, f, indent=2)
    print("Benchmark complete:", benchmark_metrics)


if __name__ == "__main__":
    main()

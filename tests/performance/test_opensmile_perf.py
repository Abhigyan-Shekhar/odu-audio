"""
Performance and benchmarking tests for openSMILE eGeMAPS extraction.
"""

import resource
import time

import numpy as np

from src.audio_pipeline.features.opensmile_extractor import OpenSmileExtractor


def test_benchmark_opensmile_cpu():
    """Benchmark CPU processing time for 2.0s window extraction."""
    extractor = OpenSmileExtractor()
    signal = np.random.normal(0, 0.1, 32000).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = extractor.extract(signal, 16000)

    n_iterations = 20
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = extractor.extract(signal, 16000)
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_iterations) * 1000
    print(f"Average eGeMAPS CPU extraction time: {avg_ms:.2f} ms")
    # A 2.0s window has a hop of 0.5s (500 ms). Target is well within 500 ms.
    assert avg_ms < 200.0, f"Extraction too slow: {avg_ms:.2f} ms"


def test_benchmark_opensmile_memory():
    """Benchmark memory consumption during sequential extractions."""
    extractor = OpenSmileExtractor()
    signal = np.random.normal(0, 0.1, 32000).astype(np.float32)

    # Get initial RSS memory
    initial_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Run extraction 50 times
    for _ in range(50):
        _ = extractor.extract(signal, 16000)

    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_diff = final_rss - initial_rss
    print(f"Initial RSS: {initial_rss}, Final RSS: {final_rss}, Diff: {rss_diff}")

    # Verify no massive memory leaks
    # In macOS, ru_maxrss is in bytes, on Linux it is in KB.
    # We assert the diff is less than 50MB (50 * 1024 * 1024 bytes) to be safe.
    assert rss_diff < 50 * 1024 * 1024, f"Suspicious memory leak: {rss_diff} bytes"

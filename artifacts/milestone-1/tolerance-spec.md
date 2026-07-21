# Milestone 1 Tolerance Specification

This document defines the numeric tolerances, validation thresholds, and mathematical constraints implemented in the WAV-to-Parquet conversion pipeline.

## 1. Resampling
- **Target sample rate**: 16,000 Hz.
- **Algorithm**: Kaiser window interpolation (`kaiser_best`).
- **Antialiasing**: Automatically enabled.
- **Passband/Stopband ripple**: $\le 0.01\%$.

## 2. Windowing
- **Default window width**: 2.0 seconds (32,000 samples at 16 kHz).
- **Default hop size**: 0.5 seconds (8,000 samples at 16 kHz).
- **Overlap duration**: 1.5 seconds.
- **Overlap ratio**: 75%.
- **Padding**: None. Gaps are truncated. Partial windows at the end of audio are clamped to `total_samples` and yielded, then sliding stops.

## 3. Quality Metric Thresholds
- **Clipping**:
  - Absolute amplitude threshold: `0.99`.
  - Max clipping ratio allowed before marking `CLIPPED`: `0.05` (5% of samples).
- **Dropout**:
  - Near-zero threshold: `1e-6`.
  - Max dropout ratio allowed before marking `DROPOUT`: `0.10` (10% of samples).
- **Signal-to-Noise Ratio (SNR)**:
  - Minimum SNR allowed before marking `LOW_SNR`: `10.0` dB.
  - Calculated using frame-based RMS energy ratios:
    - Frame size: 50 ms.
    - Signal power estimated from top 30% loudest frames.
    - Noise power estimated from bottom 10% quietest frames (clamped to a minimum noise floor of `1e-5` / -100 dB).

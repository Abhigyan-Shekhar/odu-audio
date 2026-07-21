"""
SNR (Signal-to-Noise Ratio) estimation component.
"""

import numpy as np


def estimate_snr(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimate Signal-to-Noise Ratio (SNR) in dB using a simple frame-based energy ratio.

    Divides the audio into 50 ms frames. Estimates signal level from the top 30%
    loudest frames and background noise level from the bottom 10% quietest frames.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Sample rate in Hz (default: 16000).

    Returns:
        The estimated SNR in dB. Returns 0.0 if estimation cannot be completed or results in invalid values.
    """
    if len(audio) == 0:
        return 0.0

    # Frame duration: 50 ms
    frame_len = int(0.05 * sample_rate)

    # If the audio segment is shorter than one 50 ms frame
    if frame_len <= 0 or len(audio) < frame_len:
        rms_total = np.sqrt(np.mean(audio**2))
        if rms_total < 1e-5:
            return 0.0
        peak = np.max(np.abs(audio))
        if peak == 0:
            return 0.0
        snr = float(20 * np.log10(peak / max(rms_total, 1e-5)))
        return 0.0 if np.isnan(snr) or np.isinf(snr) else snr

    # Compute RMS for each non-overlapping frame
    n_frames = len(audio) // frame_len
    frame_rms = []
    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        frame_rms.append(np.sqrt(np.mean(frame**2)))

    rms_array = np.array(frame_rms)
    sorted_rms = np.sort(rms_array)

    # Signal level: average of loudest 30% of frames
    top_n = max(1, int(0.3 * len(sorted_rms)))
    signal = np.mean(sorted_rms[-top_n:])

    # Noise level: average of quietest 10% of frames
    bottom_n = max(1, int(0.1 * len(sorted_rms)))
    noise = np.mean(sorted_rms[:bottom_n])

    # Clamp noise to a minimum noise floor (-100 dB) to prevent division by zero or infinite SNR
    noise = max(noise, 1e-5)

    if signal <= noise:
        return 0.0

    snr = float(20 * np.log10(signal / noise))

    if np.isnan(snr) or np.isinf(snr):
        return 0.0

    return snr

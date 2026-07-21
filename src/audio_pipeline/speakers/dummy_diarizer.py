"""
Dummy diarizer for testing — returns a single speaker for the entire segment.
"""

import numpy as np

from src.audio_pipeline.speakers.diarizer_interface import (
    DiarizationResult,
    DiarizedTurn,
    DiarizeriInterface,
)


class DummyDiarizer(DiarizeriInterface):
    """
    Mock diarizer that always assigns the whole segment to one speaker.

    Used in automated tests to avoid loading heavyweight pyannote models.
    Optionally simulates overlap for testing OVERLAP handling.
    """

    def __init__(
        self,
        speaker_id: str = "SPEAKER_00",
        simulate_overlap: bool = False,
        simulate_no_speech: bool = False,
        second_speaker_id: str = "SPEAKER_01",
    ):
        """
        Args:
            speaker_id:          Primary speaker label to return.
            simulate_overlap:    If True, returns two overlapping turns.
            simulate_no_speech:  If True, returns an empty DiarizationResult.
            second_speaker_id:   Label of second speaker when simulating overlap.
        """
        self.speaker_id = speaker_id
        self.simulate_overlap = simulate_overlap
        self.simulate_no_speech = simulate_no_speech
        self.second_speaker_id = second_speaker_id

    def diarize(self, audio: np.ndarray, sr: int) -> DiarizationResult:
        duration_ms = int(len(audio) / sr * 1000)

        if self.simulate_no_speech or duration_ms == 0:
            return DiarizationResult(
                turns=[],
                num_speakers=0,
                has_overlap=False,
                overlapping_speaker_ids=[],
                diarizer_version=self.get_version(),
            )

        if self.simulate_overlap:
            # Two speakers both occupying the full duration (worst-case overlap)
            overlap_ms = duration_ms // 2
            turns = [
                DiarizedTurn(
                    start_ms=0,
                    end_ms=duration_ms,
                    speaker_id=self.speaker_id,
                    overlap=True,
                ),
                DiarizedTurn(
                    start_ms=overlap_ms,
                    end_ms=duration_ms,
                    speaker_id=self.second_speaker_id,
                    overlap=True,
                ),
            ]
            return DiarizationResult(
                turns=turns,
                num_speakers=2,
                has_overlap=True,
                overlapping_speaker_ids=[self.speaker_id, self.second_speaker_id],
                diarizer_version=self.get_version(),
            )

        # Default: single speaker for the entire segment
        return DiarizationResult(
            turns=[
                DiarizedTurn(
                    start_ms=0,
                    end_ms=duration_ms,
                    speaker_id=self.speaker_id,
                )
            ],
            num_speakers=1,
            has_overlap=False,
            overlapping_speaker_ids=[],
            diarizer_version=self.get_version(),
        )

    def get_version(self) -> str:
        return "dummy_diarizer_v1"

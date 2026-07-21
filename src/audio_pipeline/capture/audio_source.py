"""
Abstract audio source definition.
"""

from abc import ABC, abstractmethod

from src.audio_pipeline.capture.audio_chunk import AudioChunk


class AudioSource(ABC):
    """
    Abstract base class for all streaming audio sources.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """
        Return the sample rate of this audio source.
        """
        pass

    @abstractmethod
    def read_chunk(self) -> AudioChunk:
        """
        Read the next chunk of audio.
        This call blocks until a chunk is available.

        Returns:
            An AudioChunk containing samples and metadata.

        Raises:
            EOFError: If the source has ended (e.g. end of file).
            RuntimeError: If capture fails.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the audio source and release resources.
        """
        pass

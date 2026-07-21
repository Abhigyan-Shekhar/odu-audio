"""
Audio Pipeline and Acoustic Modeling

Core package for real-time streaming and offline feature extraction.
"""

__version__ = "0.1.0"

from .pipeline import AudioPipeline
from .runtime.stream_processor import StreamProcessor
from .offline.dataset_builder import DatasetBuilder

__all__ = ["AudioPipeline", "StreamProcessor", "DatasetBuilder"]

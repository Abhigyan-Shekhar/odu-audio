"""
Graceful degradation manager for streaming pipeline.

Implements deterministic degradation levels under system load.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class DegradationLevel(IntEnum):
    """Graceful degradation levels"""
    LEVEL_0 = 0  # All extractors active (normal operation)
    LEVEL_1 = 1  # Reduce emotion2vec update frequency
    LEVEL_2 = 2  # Disable emotion2vec
    LEVEL_3 = 3  # Disable diarization refinement, use provisional attribution
    LEVEL_4 = 4  # VAD + quality + critical events only
    LEVEL_5 = 5  # Quality/status events only (no features)


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation"""
    
    enable: bool = True
    """Enable graceful degradation"""
    
    backpressure_threshold_ms: float = 500.0
    """Maximum acceptable processing latency before triggering degradation"""
    
    level_1_trigger_ms: float = 200.0
    """Latency threshold for level 1"""
    
    level_2_trigger_ms: float = 500.0
    """Latency threshold for level 2"""
    
    level_3_trigger_ms: float = 1000.0
    """Latency threshold for level 3"""
    
    level_4_trigger_ms: float = 2000.0
    """Latency threshold for level 4"""
    
    level_5_trigger_ms: float = 5000.0
    """Latency threshold for level 5"""
    
    recovery_delay_seconds: float = 5.0
    """Time to wait at lower latency before recovering to previous level"""
    
    recovery_threshold_ratio: float = 0.5
    """Latency must be below threshold * ratio to recover"""


class DegradationManager:
    """
    Manages graceful degradation of the streaming pipeline.
    
    Monitors processing latency and adjusts active components to maintain
    real-time performance under load.
    """
    
    def __init__(self, config: DegradationConfig):
        self.config = config
        self.current_level = DegradationLevel.LEVEL_0
        self.last_level_change_time = time.time()
        self.recent_latencies: list[float] = []
        self.max_latency_history = 20
        
    def record_latency(self, latency_ms: float):
        """
        Record processing latency for a window.
        
        Args:
            latency_ms: Processing latency in milliseconds
        """
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > self.max_latency_history:
            self.recent_latencies.pop(0)
    
    def get_current_latency_p95(self) -> float:
        """Get 95th percentile of recent latencies"""
        if not self.recent_latencies:
            return 0.0
        sorted_latencies = sorted(self.recent_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]
    
    def should_degrade(self) -> bool:
        """Check if we should move to a higher degradation level"""
        if not self.config.enable:
            return False
        
        if self.current_level >= DegradationLevel.LEVEL_5:
            return False  # Already at max degradation
        
        latency_p95 = self.get_current_latency_p95()
        next_level = self.current_level + 1
        threshold = self._get_threshold_for_level(next_level)
        
        return latency_p95 > threshold
    
    def should_recover(self) -> bool:
        """Check if we should move to a lower degradation level"""
        if not self.config.enable:
            return False
        
        if self.current_level <= DegradationLevel.LEVEL_0:
            return False  # Already at normal operation
        
        # Must wait recovery delay before attempting recovery
        time_since_change = time.time() - self.last_level_change_time
        if time_since_change < self.config.recovery_delay_seconds:
            return False
        
        latency_p95 = self.get_current_latency_p95()
        current_threshold = self._get_threshold_for_level(self.current_level)
        recovery_threshold = current_threshold * self.config.recovery_threshold_ratio
        
        return latency_p95 < recovery_threshold
    
    def degrade(self) -> Optional[DegradationLevel]:
        """
        Move to next degradation level.
        
        Returns:
            New degradation level if changed, None otherwise
        """
        if not self.should_degrade():
            return None
        
        old_level = self.current_level
        self.current_level = DegradationLevel(min(self.current_level + 1, DegradationLevel.LEVEL_5))
        self.last_level_change_time = time.time()
        
        logger.warning(
            f"Degrading from level {old_level} to {self.current_level} "
            f"(latency p95: {self.get_current_latency_p95():.1f} ms)"
        )
        
        return self.current_level
    
    def recover(self) -> Optional[DegradationLevel]:
        """
        Recover to previous degradation level.
        
        Returns:
            New degradation level if changed, None otherwise
        """
        if not self.should_recover():
            return None
        
        old_level = self.current_level
        self.current_level = DegradationLevel(max(self.current_level - 1, DegradationLevel.LEVEL_0))
        self.last_level_change_time = time.time()
        
        logger.info(
            f"Recovering from level {old_level} to {self.current_level} "
            f"(latency p95: {self.get_current_latency_p95():.1f} ms)"
        )
        
        return self.current_level
    
    def update(self) -> Optional[DegradationLevel]:
        """
        Update degradation level based on recent latencies.
        
        Returns:
            New degradation level if changed, None otherwise
        """
        if self.should_degrade():
            return self.degrade()
        elif self.should_recover():
            return self.recover()
        return None
    
    def _get_threshold_for_level(self, level: DegradationLevel) -> float:
        """Get latency threshold for a given degradation level"""
        thresholds = {
            DegradationLevel.LEVEL_1: self.config.level_1_trigger_ms,
            DegradationLevel.LEVEL_2: self.config.level_2_trigger_ms,
            DegradationLevel.LEVEL_3: self.config.level_3_trigger_ms,
            DegradationLevel.LEVEL_4: self.config.level_4_trigger_ms,
            DegradationLevel.LEVEL_5: self.config.level_5_trigger_ms,
        }
        return thresholds.get(level, float('inf'))
    
    def is_component_active(self, component: str) -> bool:
        """
        Check if a component should be active at current degradation level.
        
        Args:
            component: Component name (e.g., 'emotion2vec', 'diarization', 'yamnet')
        
        Returns:
            True if component should be active
        """
        level = self.current_level
        
        # Level 0: all active
        if level == DegradationLevel.LEVEL_0:
            return True
        
        # Level 1: reduce emotion2vec frequency (handled by component)
        if level == DegradationLevel.LEVEL_1:
            return True
        
        # Level 2: disable emotion2vec
        if level == DegradationLevel.LEVEL_2:
            return component != "emotion2vec"
        
        # Level 3: disable diarization refinement
        if level == DegradationLevel.LEVEL_3:
            return component not in {"emotion2vec", "diarization_refinement"}
        
        # Level 4: VAD + quality + critical events only
        if level == DegradationLevel.LEVEL_4:
            return component in {"vad", "quality", "yamnet", "clipping"}
        
        # Level 5: quality/status only
        if level == DegradationLevel.LEVEL_5:
            return component in {"quality", "clipping"}
        
        return False
    
    def get_emotion2vec_hop_multiplier(self) -> int:
        """
        Get hop multiplier for emotion2vec at current degradation level.
        
        At level 1, we reduce update frequency by returning a multiplier > 1.
        
        Returns:
            Hop multiplier (1 = normal, 2 = half frequency, etc.)
        """
        if self.current_level == DegradationLevel.LEVEL_1:
            return 2  # Update every 2 seconds instead of 1
        return 1
    
    def get_status_summary(self) -> dict:
        """Get current degradation status summary"""
        return {
            "level": self.current_level.value,
            "level_name": self.current_level.name,
            "latency_p95_ms": self.get_current_latency_p95(),
            "time_since_change_s": time.time() - self.last_level_change_time,
            "active_components": {
                "vad": self.is_component_active("vad"),
                "diarization": self.is_component_active("diarization"),
                "diarization_refinement": self.is_component_active("diarization_refinement"),
                "egemaps": self.is_component_active("egemaps"),
                "yamnet": self.is_component_active("yamnet"),
                "emotion2vec": self.is_component_active("emotion2vec"),
                "quality": self.is_component_active("quality"),
            },
            "emotion2vec_hop_multiplier": self.get_emotion2vec_hop_multiplier(),
        }

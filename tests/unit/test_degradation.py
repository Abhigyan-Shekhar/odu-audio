"""
Unit tests for the Graceful Degradation Manager.
"""

import time

import pytest

from src.audio_pipeline.runtime.degradation import (
    DegradationConfig,
    DegradationLevel,
    DegradationManager,
)


@pytest.fixture
def base_config():
    return DegradationConfig(
        enable=True,
        backpressure_threshold_ms=500.0,
        level_1_trigger_ms=200.0,
        level_2_trigger_ms=500.0,
        level_3_trigger_ms=1000.0,
        level_4_trigger_ms=2000.0,
        level_5_trigger_ms=5000.0,
        recovery_delay_seconds=1.0,  # short delay for testing
        recovery_threshold_ratio=0.5,
    )


def test_degradation_manager_p95_latency(base_config):
    mgr = DegradationManager(base_config)

    # Empty case
    assert mgr.get_current_latency_p95() == 0.0

    # 20 samples: 1..20
    for latency in range(1, 21):
        mgr.record_latency(float(latency))

    # p95 should pick index 19 in sorted array (0.95 * 20 = 19, which is the 20th item = 20.0)
    assert mgr.get_current_latency_p95() == 20.0


def test_degradation_transitions(base_config):
    mgr = DegradationManager(base_config)
    assert mgr.current_level == DegradationLevel.LEVEL_0

    # Trigger Level 1 (> 200 ms)
    for _ in range(20):
        mgr.record_latency(250.0)
    assert mgr.should_degrade()
    new_level = mgr.degrade()
    assert new_level == DegradationLevel.LEVEL_1
    assert mgr.current_level == DegradationLevel.LEVEL_1

    # Trigger Level 2 (> 500 ms)
    for _ in range(20):
        mgr.record_latency(600.0)
    assert mgr.should_degrade()
    new_level = mgr.degrade()
    assert new_level == DegradationLevel.LEVEL_2

    # Max degradation LEVEL_5
    mgr.current_level = DegradationLevel.LEVEL_4
    for _ in range(20):
        mgr.record_latency(6000.0)
    assert mgr.should_degrade()
    new_level = mgr.degrade()
    assert new_level == DegradationLevel.LEVEL_5
    assert not mgr.should_degrade()  # Already at max


def test_recovery_transitions(base_config):
    mgr = DegradationManager(base_config)
    mgr.current_level = DegradationLevel.LEVEL_2
    mgr.last_level_change_time = (
        time.time() - 2.0
    )  # mock time passing (past recovery delay)

    # Threshold for LEVEL_2 is 500 ms. Recovery ratio is 0.5.
    # So recovery threshold is 500 * 0.5 = 250 ms.
    # Latency 150 ms is below 250 ms.
    for _ in range(20):
        mgr.record_latency(150.0)

    assert mgr.should_recover()
    new_level = mgr.recover()
    assert new_level == DegradationLevel.LEVEL_1


def test_recovery_delay_blocking(base_config):
    mgr = DegradationManager(base_config)
    mgr.current_level = DegradationLevel.LEVEL_2
    mgr.last_level_change_time = time.time()  # just changed now

    for _ in range(20):
        mgr.record_latency(50.0)

    # Should not recover because not enough time has passed since last level change
    assert not mgr.should_recover()


def test_component_activation_by_level(base_config):
    mgr = DegradationManager(base_config)

    # Level 0: everything active
    mgr.current_level = DegradationLevel.LEVEL_0
    assert mgr.is_component_active("emotion2vec")
    assert mgr.is_component_active("vad")
    assert mgr.is_component_active("quality")

    # Level 1: everything active (frequency scaling handled inside components)
    mgr.current_level = DegradationLevel.LEVEL_1
    assert mgr.is_component_active("emotion2vec")
    assert mgr.is_component_active("vad")

    # Level 2: disable emotion2vec
    mgr.current_level = DegradationLevel.LEVEL_2
    assert not mgr.is_component_active("emotion2vec")
    assert mgr.is_component_active("vad")

    # Level 3: disable diarization refinement
    mgr.current_level = DegradationLevel.LEVEL_3
    assert not mgr.is_component_active("emotion2vec")
    assert not mgr.is_component_active("diarization_refinement")
    assert mgr.is_component_active("vad")
    assert mgr.is_component_active("diarization")  # provisional diarization active

    # Level 4: VAD + quality + critical events (YAMNet) only
    mgr.current_level = DegradationLevel.LEVEL_4
    assert not mgr.is_component_active("egemaps")
    assert mgr.is_component_active("vad")
    assert mgr.is_component_active("yamnet")
    assert mgr.is_component_active("quality")

    # Level 5: Quality/status events only
    mgr.current_level = DegradationLevel.LEVEL_5
    assert not mgr.is_component_active("vad")
    assert not mgr.is_component_active("yamnet")
    assert mgr.is_component_active("quality")


def test_emotion2vec_hop_multiplier(base_config):
    mgr = DegradationManager(base_config)

    mgr.current_level = DegradationLevel.LEVEL_0
    assert mgr.get_emotion2vec_hop_multiplier() == 1

    mgr.current_level = DegradationLevel.LEVEL_1
    assert mgr.get_emotion2vec_hop_multiplier() == 2

    mgr.current_level = DegradationLevel.LEVEL_2
    assert (
        mgr.get_emotion2vec_hop_multiplier() == 1
    )  # component itself is inactive anyway


def test_degradation_status_summary(base_config):
    mgr = DegradationManager(base_config)
    summary = mgr.get_status_summary()
    assert summary["level"] == 0
    assert summary["level_name"] == "LEVEL_0"
    assert "active_components" in summary

"""
Unit tests for configuration loading and validation.
"""

import pytest
import yaml
import os
from src.audio_pipeline.runtime.degradation import DegradationConfig


def test_streaming_default_yaml_exists():
    config_path = "configs/streaming_default.yaml"
    assert os.path.exists(config_path), f"Config file not found at {config_path}"


def test_load_and_validate_default_yaml():
    config_path = "configs/streaming_default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Check top-level keys
    expected_sections = [
        "capture",
        "preprocessing",
        "vad",
        "diarization",
        "enrollment",
        "quality",
        "features",
        "storage",
        "degradation",
        "privacy",
        "logging",
    ]
    for section in expected_sections:
        assert section in config, f"Missing section '{section}' in default configuration"
        
    # Validate capture properties
    assert config["capture"]["sample_rate"] == 16000
    assert config["capture"]["channels"] == 1
    
    # Validate degradation config mapping
    deg_cfg = config["degradation"]
    degradation_obj = DegradationConfig(
        enable=deg_cfg["enable"],
        backpressure_threshold_ms=deg_cfg["backpressure_threshold_ms"],
        level_1_trigger_ms=deg_cfg["level_1_trigger_ms"],
        level_2_trigger_ms=deg_cfg["level_2_trigger_ms"],
        level_3_trigger_ms=deg_cfg["level_3_trigger_ms"],
        level_4_trigger_ms=deg_cfg["level_4_trigger_ms"],
        level_5_trigger_ms=deg_cfg["level_5_trigger_ms"],
        recovery_delay_seconds=deg_cfg["recovery_delay_seconds"],
        recovery_threshold_ratio=deg_cfg["recovery_threshold_ratio"],
    )
    
    assert degradation_obj.enable
    assert degradation_obj.level_1_trigger_ms == 200.0
    assert degradation_obj.level_5_trigger_ms == 5000.0

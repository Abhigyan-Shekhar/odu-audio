#!/usr/bin/env python3
"""Train Phase 1 static fusion baselines from a fused feature table."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.audio_pipeline.models.lightgbm_fusion import (
    LightGBMFusionConfig,
    records_to_frame,
    train_variant_models,
)
from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--input", required=True, help="Fused CSV/Parquet/JSONL table")
    parser.add_argument("--output-dir", default="artifacts/fusion")
    args = parser.parse_args()

    config_data = yaml.safe_load(Path(args.config).read_text()) or {}
    training_config = LightGBMFusionConfig(**config_data.get("training", {}))
    frame = load_frame(args.input)
    results = train_variant_models(frame, training_config, args.output_dir)

    summary = {}
    for name, result in results.items():
        if isinstance(result, dict):
            summary[name] = result
        else:
            summary[name] = {
                "config_hash": result.config_hash,
                "model_hash": result.model_hash,
                "feature_count": len(result.feature_names),
                **result.validation_metrics,
            }

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def load_frame(path: str) -> pd.DataFrame:
    """Load a fused table or JSONL UnifiedFeatureRecord payloads."""
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    if source.suffix in {".jsonl", ".ndjson"}:
        records = [
            UnifiedFeatureRecord.from_dict(json.loads(line))
            for line in source.read_text().splitlines()
            if line.strip()
        ]
        return records_to_frame(records)
    raise ValueError(f"Unsupported input format: {source.suffix}")


if __name__ == "__main__":
    main()

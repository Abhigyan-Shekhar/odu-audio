"""
LightGBM baseline for static fusion over UnifiedFeatureRecord rows.

This module keeps training and explanation utilities independent from the
streaming runtime so Phase 1 experiments are reproducible and easy to test.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.audio_pipeline.schemas.unified_feature import UnifiedFeatureRecord

INTERACTION_FEATURES = [
    "acoustic_x_profanity",
    "acoustic_arousal_x_profanity",
    "pitch_energy_x_threat",
    "repetition_x_arousal",
    "asr_conf_x_profanity",
    "arousal_x_threat",
    "arousal_x_distress",
    "scream_x_distress",
    "toxicity_x_asr_confidence",
    "patient_probability_x_agitation_evidence",
]

ACOUSTIC_FEATURES = [
    "patient_probability",
    "overlap_probability",
    "snr_db",
    "clipping_ratio",
    "dropout_ratio",
    "acoustic_arousal_probability",
    "pitch_energy_arousal_probability",
    "scream_shout_probability",
]

LEXICAL_FEATURES = [
    "profanity_probability",
    "profanity_intensity",
    "toxicity_probability",
    "threat_probability",
    "directed_insult_probability",
    "imperative_probability",
    "repetition_probability",
    "repeated_request_probability",
    "distress_phrase_probability",
    "mutox_speech_score",
    "mutox_text_score",
    "asr_confidence",
    "code_mixed",
]


@dataclass(frozen=True)
class LightGBMFusionConfig:
    """Training configuration for the static fusion baseline."""

    label_column: str = "verbal_agitation_label"
    group_column: str = "patient_id"
    validation_size: float = 0.2
    test_size: float = 0.0
    random_state: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    early_stopping_rounds: int = 50
    class_weight: Optional[str] = "balanced"
    calm_profanity_column: Optional[str] = "calm_profanity_label"
    event_hours_column: Optional[str] = "duration_hours"
    variants: dict[str, list[str]] = field(default_factory=dict)

    def variant_map(self) -> dict[str, list[str]]:
        """Return configured or default Phase 1 comparison variants."""
        if self.variants:
            return self.variants
        return {
            "acoustic_only": ACOUSTIC_FEATURES + ["egemaps_", "missing:acoustic"],
            "lexical_only": LEXICAL_FEATURES + ["missing:lexical"],
            "yamnet_event_only": ["yamnet_", "acoustic_arousal_probability"],
            "acoustic_lexical": ACOUSTIC_FEATURES + LEXICAL_FEATURES,
            "acoustic_lexical_interactions": (
                ACOUSTIC_FEATURES + LEXICAL_FEATURES + INTERACTION_FEATURES
            ),
            "full": ["*"],
        }


@dataclass
class TrainedFusionModel:
    """Serializable LightGBM fusion artifact."""

    model: Any
    feature_names: list[str]
    config_hash: str
    model_hash: str
    validation_metrics: dict[str, float]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict agitation probabilities for a feature frame."""
        X = frame.reindex(columns=self.feature_names, fill_value=np.nan)
        return np.asarray(self.model.predict_proba(X))[:, 1]

    def save(self, path: str | Path) -> None:
        """Write the trained artifact with model/config hashes."""
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedFusionModel":
        """Load a trained static fusion artifact."""
        import joblib

        return cast(TrainedFusionModel, joblib.load(path))


def records_to_frame(records: Iterable[UnifiedFeatureRecord]) -> pd.DataFrame:
    """Flatten unified records into a tabular model input frame."""
    return pd.DataFrame([flatten_record(record) for record in records])


def flatten_record(record: UnifiedFeatureRecord) -> dict[str, float | int | str]:
    """Flatten nested unified record fields into scalar model columns."""
    row: dict[str, float | int | str] = {
        "session_id": record.session_id,
        "stream_id": record.stream_id,
        "subject_id": record.subject_id or "",
        "speaker_id": record.speaker_id or "",
        "tick_start_ms": record.tick_start_ms,
        "tick_end_ms": record.tick_end_ms,
        "t0": record.t0,
        "t1": record.t1,
        "attribution_status": record.attribution_status,
        "quality_status": record.quality_status,
        "language": record.language,
        "acoustic_missing": int(record.acoustic_missing),
        "lexical_missing": int(record.lexical_missing),
    }

    scalar_names = ACOUSTIC_FEATURES + LEXICAL_FEATURES + INTERACTION_FEATURES
    for name in scalar_names:
        value = getattr(record, name)
        if isinstance(value, bool):
            row[name] = int(value)
        elif value is not None:
            row[name] = float(value)
        else:
            row[name] = np.nan

    for index, value in enumerate(record.egemaps or []):
        row[f"egemaps_{index:03d}"] = float(value)
    for name, value in record.yamnet_event_scores.items():
        row[f"yamnet_{_clean_column(name)}"] = float(value)
    for index, value in enumerate(record.emotion_embedding or []):
        row[f"emotion_{index:03d}"] = float(value)
    for name, missing in record.feature_missing_mask.items():
        row[f"missing:{_clean_column(name)}"] = int(missing)

    return row


def train_variant_models(
    dataset: pd.DataFrame,
    config: LightGBMFusionConfig,
    output_dir: Optional[str | Path] = None,
) -> dict[str, TrainedFusionModel | dict[str, float]]:
    """
    Train Phase 1 comparison models plus a rule baseline.

    Splitting is group-disjoint using the configured patient or speaker column;
    random window-level splits are deliberately not supported.
    """
    _validate_dataset(dataset, config)
    train_frame, val_frame = group_disjoint_train_val_split(dataset, config)

    results: dict[str, TrainedFusionModel | dict[str, float]] = {}
    results["rule_baseline"] = evaluate_rule_baseline(val_frame, config.label_column)

    for variant_name, selectors in config.variant_map().items():
        feature_names = select_feature_columns(dataset, selectors, config)
        if not feature_names:
            continue
        artifact = train_lightgbm_fusion(
            train_frame=train_frame,
            val_frame=val_frame,
            feature_names=feature_names,
            config=config,
        )
        results[variant_name] = artifact
        if output_dir is not None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            artifact.save(path / f"{variant_name}.joblib")

    return results


def train_lightgbm_fusion(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_names: list[str],
    config: LightGBMFusionConfig,
) -> TrainedFusionModel:
    """Train one LightGBM model with early stopping and balanced weights."""
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LightGBM is required for training. Install project dependencies "
            "from pyproject.toml before running scripts/train_fusion.py."
        ) from exc

    X_train = train_frame.reindex(columns=feature_names, fill_value=np.nan)
    y_train = train_frame[config.label_column].astype(int)
    X_val = val_frame.reindex(columns=feature_names, fill_value=np.nan)
    y_val = val_frame[config.label_column].astype(int)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    probabilities = np.asarray(model.predict_proba(X_val))[:, 1]
    metrics = classification_metrics(y_val.to_numpy(), probabilities)
    metrics.update(false_positive_metrics(val_frame, probabilities, config))
    config_hash = stable_hash(config.__dict__)
    model_hash = stable_hash(model.booster_.model_to_string())
    return TrainedFusionModel(
        model=model,
        feature_names=feature_names,
        config_hash=config_hash,
        model_hash=model_hash,
        validation_metrics=metrics,
    )


def group_disjoint_train_val_split(
    dataset: pd.DataFrame,
    config: LightGBMFusionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a leakage-safe group-disjoint train/validation split."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.validation_size,
        random_state=config.random_state,
    )
    train_idx, val_idx = next(
        splitter.split(dataset, groups=dataset[config.group_column])
    )
    return dataset.iloc[train_idx].copy(), dataset.iloc[val_idx].copy()


def group_disjoint_train_val_test_split(
    dataset: pd.DataFrame,
    config: LightGBMFusionConfig,
) -> dict[str, pd.DataFrame]:
    """
    Create leakage-safe train/validation/test splits when a test set is requested.

    A zero test_size preserves the Phase 1 training path while still exposing an
    explicit group-disjoint test split utility for experiment scripts.
    """
    if config.test_size <= 0.0:
        train, val = group_disjoint_train_val_split(dataset, config)
        return {"train": train, "val": val, "test": dataset.iloc[[]].copy()}

    holdout_size = config.validation_size + config.test_size
    if holdout_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.0")

    first_config = LightGBMFusionConfig(
        **{**config.__dict__, "validation_size": holdout_size}
    )
    train, holdout = group_disjoint_train_val_split(dataset, first_config)
    relative_test_size = config.test_size / holdout_size
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=config.random_state,
    )
    val_idx, test_idx = next(
        splitter.split(holdout, groups=holdout[config.group_column])
    )
    return {
        "train": train,
        "val": holdout.iloc[val_idx].copy(),
        "test": holdout.iloc[test_idx].copy(),
    }


def select_feature_columns(
    dataset: pd.DataFrame,
    selectors: list[str],
    config: LightGBMFusionConfig,
) -> list[str]:
    """Resolve explicit feature names and prefix selectors into numeric columns."""
    excluded = {
        config.label_column,
        config.group_column,
        "session_id",
        "stream_id",
        "subject_id",
        "speaker_id",
        "attribution_status",
        "quality_status",
        "language",
    }
    columns: list[str] = []
    for selector in selectors:
        if selector == "*":
            candidates = [
                column
                for column in dataset.columns
                if column not in excluded
                and pd.api.types.is_numeric_dtype(dataset[column])
            ]
        elif selector.endswith("_") or selector.endswith(":"):
            candidates = [
                column
                for column in dataset.columns
                if column.startswith(selector)
                and pd.api.types.is_numeric_dtype(dataset[column])
            ]
        elif selector in dataset.columns and pd.api.types.is_numeric_dtype(
            dataset[selector]
        ):
            candidates = [selector]
        else:
            candidates = []
        for column in candidates:
            if column not in columns and column not in excluded:
                columns.append(column)
    return columns


def evaluate_rule_baseline(
    frame: pd.DataFrame,
    label_column: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Explainable rule baseline: arousal plus hostile or distress language."""
    arousal = frame.get(
        "acoustic_arousal_probability", pd.Series(0.0, index=frame.index)
    )
    scream = frame.get("scream_shout_probability", pd.Series(0.0, index=frame.index))
    threat = frame.get("threat_probability", pd.Series(0.0, index=frame.index))
    insult = frame.get("directed_insult_probability", pd.Series(0.0, index=frame.index))
    distress = frame.get(
        "distress_phrase_probability", pd.Series(0.0, index=frame.index)
    )
    profanity = frame.get("profanity_probability", pd.Series(0.0, index=frame.index))
    patient = frame.get("patient_probability", pd.Series(1.0, index=frame.index))
    score = patient * np.maximum.reduce(
        [
            arousal * np.maximum(threat, insult),
            arousal * distress,
            scream * np.maximum(distress, threat),
            arousal * profanity * 0.7,
        ]
    )
    metrics = classification_metrics(frame[label_column].to_numpy(dtype=int), score)
    metrics["threshold_false_positive_rate"] = float(
        np.mean((score >= threshold) & (frame[label_column].to_numpy(dtype=int) == 0))
    )
    return metrics


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """AUPRC-first validation metrics for imbalanced event detection."""
    metrics = {"auprc": float(average_precision_score(y_true, probabilities))}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["auroc"] = float("nan")
    metrics.update(threshold_metrics(y_true, probabilities, threshold=0.5))
    return metrics


def threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Precision and recall at a selected validation threshold."""
    predictions = probabilities >= threshold
    positives = y_true == 1
    true_positives = np.sum(predictions & positives)
    false_positives = np.sum(predictions & ~positives)
    false_negatives = np.sum(~predictions & positives)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )
    return {
        f"precision_at_{threshold:g}": float(precision),
        f"recall_at_{threshold:g}": float(recall),
    }


def false_positive_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    config: LightGBMFusionConfig,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Report false alarms per hour and calm-profanity false-positive rate."""
    predictions = probabilities >= threshold
    negatives = frame[config.label_column].to_numpy(dtype=int) == 0
    false_positives = predictions & negatives
    total_hours = (
        float(frame[config.event_hours_column].sum())
        if config.event_hours_column and config.event_hours_column in frame
        else len(frame) / 3600.0
    )
    metrics = {
        "false_alarms_per_hour": float(false_positives.sum() / max(total_hours, 1e-9))
    }
    calm_column = config.calm_profanity_column
    if calm_column and calm_column in frame:
        calm_mask = frame[calm_column].to_numpy(dtype=bool)
        denominator = max(int(calm_mask.sum()), 1)
        metrics["calm_profanity_false_positive_rate"] = float(
            (false_positives & calm_mask).sum() / denominator
        )
    return metrics


def tree_shap_contributions(
    artifact: TrainedFusionModel,
    frame: pd.DataFrame,
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Return per-row TreeSHAP-style contributions from LightGBM.

    LightGBM exposes contribution values directly, so this does not require the
    optional shap package.
    """
    X = frame.reindex(columns=artifact.feature_names, fill_value=np.nan)
    contributions = artifact.model.predict(X, pred_contrib=True)
    rows: list[list[tuple[str, float]]] = []
    for row in contributions[:, :-1]:
        ranked = sorted(
            zip(artifact.feature_names, row),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        rows.append([(name, float(value)) for name, value in ranked[:top_k]])
    return rows


def _validate_dataset(dataset: pd.DataFrame, config: LightGBMFusionConfig) -> None:
    required = [config.label_column, config.group_column]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if dataset[config.group_column].isna().any():
        raise ValueError(f"{config.group_column} must not contain missing values")


def _clean_column(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def stable_hash(value: object) -> str:
    """Return a stable SHA256 hash for configs, models, and feature lists."""
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

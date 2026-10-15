"""Shared feature, splitting, training, and evaluation utilities.

All results from the bundled dataset are synthetic-data results only.
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit

SENSOR_FEATURES = [
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "rainfall_intensity_mm_h", "rainfall_accum_24h_mm",
    "humidity_rh_pct", "temperature_c", "vibration_g",
]
RISK_TARGET = "landslide_risk_label"
FAULT_TARGET = "sensor_fault_label"
METADATA_COLUMNS = [
    "row_id", "timestamp", "node_id", "environmental_scenario",
    "landslide_risk_label", "sensor_fault_label", "faulty_sensor",
    "fault_episode_id", "dataset_source",
]
SEED = 42


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = set(SENSOR_FEATURES + ["timestamp", "node_id", RISK_TARGET,
                                      FAULT_TARGET, "fault_episode_id"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    """Add one missingness indicator per sensor feature, preserving raw values."""
    X = X[SENSOR_FEATURES].copy()
    for col in SENSOR_FEATURES:
        X[f"missing_{col}"] = X[col].isna().astype("int8")
    return X


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def make_group_ids(df: pd.DataFrame, block_hours: int = 12) -> pd.Series:
    """Group normal rows by 12-hour time block and fault rows by fault episode.

    All normal nodes within the same time block share a group. Every row from a
    nonzero fault episode shares one group, even if the episode crosses a time
    block boundary. This avoids the transitive merging of many adjacent blocks
    that can occur when long episodes overlap block boundaries.
    """
    ts = pd.to_datetime(df["timestamp"])
    block = ((ts - ts.min()).dt.total_seconds() // (block_hours * 3600)).astype(int)
    episode = df["fault_episode_id"].fillna(0).astype(int)
    keys = [f"episode_{e}" if e != 0 else f"normal_block_{b}"
            for e, b in zip(episode.to_numpy(), block.to_numpy())]
    key_to_group = {key: i for i, key in enumerate(sorted(set(keys)))}
    return pd.Series([key_to_group[key] for key in keys], index=df.index, dtype=int)


def split_by_groups(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    seed: int = SEED):
    """Create approximately 60/20/20 train/validation/test splits by group."""
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(splitter1.split(X, y, groups))
    X_tv, y_tv, g_tv = X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_rel, val_rel = next(splitter2.split(X_tv, y_tv, g_tv))
    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]
    return train_idx, val_idx, test_idx


def build_model(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def save_evaluation(model, X_test: pd.DataFrame, y_test: pd.Series,
                    output_dir: str | Path, model_name: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = model.predict(X_test)
    labels = sorted(set(y_test.astype(str)) | set(pd.Series(pred).astype(str)))
    metrics = {
        "model": model_name,
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "note": "Synthetic dataset evaluation only; not real-world validation.",
    }
    (out / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_test, pred, labels=labels, zero_division=0)
    (out / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=labels, xticks_rotation=35, ax=ax, colorbar=False)
    ax.set_title(f"{model_name}: synthetic test split")
    fig.tight_layout()
    fig.savefig(out / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


def train_one(df: pd.DataFrame, task: str, models_dir: str | Path,
              results_dir: str | Path) -> dict:
    if task == "risk":
        # Labels were assigned from the true environment. Train the first baseline
        # only on rows without injected sensor faults to avoid contradictory inputs.
        subset = df[df[FAULT_TARGET].eq("Normal")].copy()
        subset = subset.dropna(subset=SENSOR_FEATURES)
        target = RISK_TARGET
        X = subset[SENSOR_FEATURES].copy()
        model_name = "landslide_risk_model"
    elif task == "fault":
        subset = df.copy()
        target = FAULT_TARGET
        X = add_missing_flags(subset[SENSOR_FEATURES])
        model_name = "sensor_fault_model"
    else:
        raise ValueError("task must be 'risk' or 'fault'")

    y = subset[target].astype(str)
    groups = make_group_ids(subset)
    train_idx, val_idx, test_idx = split_by_groups(X, y, groups)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Validation split is reserved for inspecting class coverage; the final model
    # is trained on train+validation after the split design is checked.
    model = build_model()
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_summary = {
        "validation_rows": int(len(y_val)),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_classes": sorted(y_val.unique().tolist()),
        "train_classes": sorted(y_train.unique().tolist()),
        "test_classes": sorted(y_test.unique().tolist()),
        "split_groups": {
            "train": sorted(groups.iloc[train_idx].unique().tolist()),
            "validation": sorted(groups.iloc[val_idx].unique().tolist()),
            "test": sorted(groups.iloc[test_idx].unique().tolist()),
        },
        "input_features": list(X.columns),
        "target": target,
        "training_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "training_rows_by_class": y_train.value_counts().to_dict(),
        "test_rows_by_class": y_test.value_counts().to_dict(),
        "note": "Synthetic dataset only; labels are illustrative, not verified real-world outcomes.",
    }

    # Refit the saved production model on train + validation only. The test set
    # remains held out for the reported final evaluation.
    fit_idx = np.concatenate([train_idx, val_idx])
    final_model = build_model()
    final_model.fit(X.iloc[fit_idx], y.iloc[fit_idx])

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, models_path / f"{model_name}.joblib")
    (models_path / f"{model_name}_features.json").write_text(
        json.dumps({"features": list(X.columns), "target": target}, indent=2), encoding="utf-8")

    metrics = save_evaluation(final_model, X_test, y_test, results_dir, model_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / f"{model_name}_split_summary.json").write_text(
        json.dumps(val_summary, indent=2), encoding="utf-8")
    return {"metrics": metrics, "summary": val_summary}

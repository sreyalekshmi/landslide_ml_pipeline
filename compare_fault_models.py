
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.pipeline_utils import (
    load_dataset,
    SENSOR_FEATURES,
    FAULT_TARGET,
    add_missing_flags,
    make_group_ids,
    split_by_groups,
)

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "synthetic_landslide_sensor_dataset.csv"
OUT = ROOT / "results"
SEED = 42


def build_pipeline(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("classifier", classifier),
    ])


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
    }

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("\nConfusion matrix:")
    labels = sorted(y_test.unique())
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    ).to_string())

    return metrics


def main():
    df = load_dataset(CSV)
    X = add_missing_flags(df[SENSOR_FEATURES])
    y = df[FAULT_TARGET].astype(str)
    groups = make_group_ids(df)

    train_idx, _, test_idx = split_by_groups(X, y, groups)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = []
    for name, classifier in models.items():
        pipeline = build_pipeline(classifier)
        metrics = evaluate(
            name, pipeline, X_train, y_train, X_test, y_test
        )
        results.append(metrics)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fault_model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nComparison saved to: {OUT / 'fault_model_comparison.json'}")
    print("\nThese results use synthetic data only.")


if __name__ == "__main__":
    main()

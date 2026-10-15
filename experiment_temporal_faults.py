
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.pipeline_utils import (
    load_dataset,
    SENSOR_FEATURES,
    FAULT_TARGET,
    make_group_ids,
    split_by_groups,
    add_missing_flags,
    SEED,
)

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "synthetic_landslide_sensor_dataset.csv"


def make_features(df):
    """Create causal features; never use future readings."""
    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
    ordered = ordered.sort_values(
        ["node_id", "timestamp"]
    ).reset_index(drop=True)

    feature_frames = []
    for col in SENSOR_FEATURES:
        values = ordered[col]
        previous = values.groupby(ordered["node_id"]).shift(1)

        # Reset temporal comparisons at each fault-episode boundary.
        # Normal rows use their own time block as the history segment.
        block = (
            (ordered["timestamp"] - ordered["timestamp"].min())
            .dt.total_seconds() // (12 * 3600)
        ).astype(int)

        episode = ordered["fault_episode_id"].fillna(0).astype(int)
        segment = [
            f"episode_{e}" if e != 0 else f"normal_block_{b}"
            for e, b in zip(episode, block)
        ]
        segment = pd.Series(segment, index=ordered.index)

        previous_segment = segment.groupby(ordered["node_id"]).shift(1)
        same_segment = segment.eq(previous_segment)

        repeat = (
            values.notna()
            & previous.notna()
            & values.eq(previous)
            & same_segment
        ).astype("int8")

        # Past-only count of consecutive equal readings.
        # At each row, this count includes the current reading and prior
        # matching readings in the same segment, but no future observations.
        run_length = pd.Series(0, index=ordered.index, dtype="int64")
        for _, idx in ordered.groupby("node_id", sort=False).groups.items():
            current_run = 0
            previous_value = None
            previous_seg = None

            for i in idx:
                value = values.loc[i]
                seg = segment.loc[i]

                if pd.isna(value):
                    current_run = 0
                    previous_value = None
                elif seg != previous_seg or previous_value is None:
                    current_run = 1
                elif value == previous_value:
                    current_run += 1
                else:
                    current_run = 1

                run_length.loc[i] = current_run
                previous_value = value
                previous_seg = seg

        feature_frames.append(pd.DataFrame({
            f"repeat_{col}": repeat,
            f"run_length_{col}": run_length,
        }, index=ordered.index))

    temporal = pd.concat(feature_frames, axis=1)
    base = add_missing_flags(ordered[SENSOR_FEATURES]).reset_index(drop=True)
    X = pd.concat([base, temporal.reset_index(drop=True)], axis=1)

    return ordered, X


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def report(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Macro F1:", round(
        f1_score(y_true, y_pred, average="macro", zero_division=0), 4
    ))
    print(classification_report(
        y_true, y_pred, zero_division=0
    ))


def main():
    df = load_dataset(CSV)
    ordered, X_temporal = make_features(df)

    # Reorder targets and groups to match the chronological feature rows.
    y = ordered[FAULT_TARGET].astype(str).reset_index(drop=True)
    groups = make_group_ids(ordered).reset_index(drop=True)

    X_base = add_missing_flags(
        ordered[SENSOR_FEATURES]
    ).reset_index(drop=True)

    train_idx, val_idx, test_idx = split_by_groups(
        X_temporal, y, groups
    )

    print("Rows:", len(ordered))
    print(
        "Split sizes:",
        len(train_idx), len(val_idx), len(test_idx)
    )
    print("Training and validation are used for development.")
    print("Test set remains held out until final evaluation.")

    # Validation comparison on the same split.
    baseline = build_model()
    temporal = build_model()

    baseline.fit(X_base.iloc[train_idx], y.iloc[train_idx])
    temporal.fit(X_temporal.iloc[train_idx], y.iloc[train_idx])

    report(
        "BASELINE VALIDATION",
        y.iloc[val_idx],
        baseline.predict(X_base.iloc[val_idx]),
    )
    report(
        "TEMPORAL VALIDATION",
        y.iloc[val_idx],
        temporal.predict(X_temporal.iloc[val_idx]),
    )

    # Refit both models on the same train + validation rows.
    fit_idx = sorted(list(train_idx) + list(val_idx))
    baseline.fit(X_base.iloc[fit_idx], y.iloc[fit_idx])
    temporal.fit(X_temporal.iloc[fit_idx], y.iloc[fit_idx])

    report(
        "BASELINE TEST",
        y.iloc[test_idx],
        baseline.predict(X_base.iloc[test_idx]),
    )
    report(
        "TEMPORAL TEST",
        y.iloc[test_idx],
        temporal.predict(X_temporal.iloc[test_idx]),
    )

    print("\nNo existing model files were modified.")
    print(
        "Research limitation: the split is group-aware, but temporal "
        "history and episode boundaries must be verified against the "
        "actual data-generation and deployment process."
    )


if __name__ == "__main__":
    main()


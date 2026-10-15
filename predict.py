"""Predict from a CSV containing sensor features. Does not read serial hardware yet."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from src.pipeline_utils import SENSOR_FEATURES, add_missing_flags

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Predict synthetic-model outputs from a sensor CSV")
    parser.add_argument("--task", choices=["risk", "fault"], required=True)
    parser.add_argument("--input", required=True, help="CSV with the eight named sensor feature columns")
    args = parser.parse_args()
    model_name = "landslide_risk_model" if args.task == "risk" else "sensor_fault_model"
    model = joblib.load(ROOT / "models" / f"{model_name}.joblib")
    df = pd.read_csv(args.input)
    missing_cols = sorted(set(SENSOR_FEATURES) - set(df.columns))
    if missing_cols:
        raise SystemExit(f"Input CSV is missing required sensor columns: {missing_cols}")
    X = df[SENSOR_FEATURES].copy()
    # Flag missing readings before model imputation. Risk model is a baseline
    # trained on normal, complete readings, so reject missing values for risk.
    if args.task == "risk":
        if X.isna().any().any():
            raise SystemExit("Risk prediction input contains missing sensor values. Validate/recover readings first.")
    else:
        X = add_missing_flags(X)
    pred = model.predict(X)
    out = df.copy()
    out["prediction"] = pred
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        for i, label in enumerate(model.classes_):
            out[f"probability_{label}"] = probs[:, i]
    output_path = Path(args.input).with_name(Path(args.input).stem + f"_{args.task}_predictions.csv")
    out.to_csv(output_path, index=False)
    print(out[["prediction"]].head(10).to_string(index=False))
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__": main()

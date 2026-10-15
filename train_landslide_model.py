from pathlib import Path
from src.pipeline_utils import load_dataset, train_one

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "synthetic_landslide_sensor_dataset.csv"

def main():
    df = load_dataset(CSV)
    print(f"Loaded {len(df):,} rows from {CSV.name}")
    print("Risk label counts (all rows):")
    print(df["landslide_risk_label"].value_counts().to_string())
    print("\nTraining risk model on rows labelled sensor_fault_label == 'Normal'.")
    result = train_one(df, "risk", ROOT / "models", ROOT / "results")
    print("\nTest metrics:")
    for k, v in result["metrics"].items(): print(f"{k}: {v}")
    print(f"\nSaved model under: {ROOT / 'models'}")

if __name__ == "__main__": main()

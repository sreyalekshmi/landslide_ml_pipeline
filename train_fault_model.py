from pathlib import Path
from src.pipeline_utils import load_dataset, train_one

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "synthetic_landslide_sensor_dataset.csv"

def main():
    df = load_dataset(CSV)
    print(f"Loaded {len(df):,} rows from {CSV.name}")
    print("Fault label counts:")
    print(df["sensor_fault_label"].value_counts().to_string())
    print("\nFault coverage by sensor and fault type:")
    print(df[df["sensor_fault_label"] != "Normal"].pivot_table(
        index="sensor_fault_label", columns="faulty_sensor", values="row_id",
        aggfunc="count", fill_value=0).to_string())
    result = train_one(df, "fault", ROOT / "models", ROOT / "results")
    print("\nTest metrics:")
    for k, v in result["metrics"].items(): print(f"{k}: {v}")
    print(f"\nSaved model under: {ROOT / 'models'}")

if __name__ == "__main__": main()

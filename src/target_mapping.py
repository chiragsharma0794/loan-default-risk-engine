import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan.csv"
OUTPUT_PATH = BASE_DIR / "data" / "interim" / "loan_data_target_mapped_sample.csv"

TARGET_MAP = {
    "Fully Paid": 0,
    "Current": 0,
    "Charged Off": 1,
    "Default": 1
}

def main():
    print(f"Reading from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, nrows=100000, low_memory=False)

    df = df.loc[:, ~df.columns.duplicated()]

    df = df[df["loan_status"].isin(TARGET_MAP.keys())].copy()
    df["target_default"] = df["loan_status"].map(TARGET_MAP)

    print("Target distribution:")
    print(df["target_default"].value_counts(dropna=False))

    print("\nTarget percentage:")
    print((df["target_default"].value_counts(normalize=True) * 100).round(2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved target-mapped sample to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
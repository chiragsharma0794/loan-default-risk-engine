from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "table" / "column_audit.csv"

def main():
    print(f"Reading from: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH, nrows=50000, low_memory=False)

    df = df.loc[:, ~df.columns.duplicated()]

    summary = pd.DataFrame({
        "column_name": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "non_null_count": [df[col].notna().sum() for col in df.columns],
        "null_count": [df[col].isna().sum() for col in df.columns],
        "null_pct": [round(df[col].isna().mean() * 100, 2) for col in df.columns],
        "n_unique": [df[col].nunique(dropna=True) for col in df.columns],
        "sample_values": [
            ", ".join(map(str, df[col].dropna().astype(str).head(3).tolist()))
            for col in df.columns
        ]
    })

    summary["is_fully_null"] = summary["null_pct"] == 100.0

    summary = summary.sort_values(by=["null_pct", "n_unique"], ascending=[False, True])
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved column audit to: {OUTPUT_PATH.resolve()}")

    print("\nTop 20 fully-null columns:")
    print(summary[summary["null_pct"] == 100].head(20))

    print("\nTop 40 non-empty columns:")
    print(summary[summary["null_pct"] < 100].head(40))

if __name__ == "__main__":
    main()

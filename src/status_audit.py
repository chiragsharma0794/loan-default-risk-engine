import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan.csv"

def main():
    status_counts = {}

    for chunk in pd.read_csv(DATA_PATH, usecols=["loan_status"], chunksize=200000, low_memory=False):
        counts = chunk["loan_status"].value_counts(dropna=False)
        for status, count in counts.items():
            status_counts[status] = status_counts.get(status, 0) + count

    result = pd.Series(status_counts).sort_values(ascending=False)

    print("Full loan_status distribution:")
    print(result)

    print("\nPercentage distribution:")
    print((result / result.sum() * 100).round(2))

if __name__ == "__main__":
    main()
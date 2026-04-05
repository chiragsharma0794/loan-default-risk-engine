from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan.csv"
SAMPLE_PATH = BASE_DIR / "outputs" / "sample" / "loan_sample_10000.csv"

def main():
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=100000, low_memory=True):
        parts.append(chunk.sample(n=min(500, len(chunk)), random_state=42))

    sample = pd.concat(parts, ignore_index=True)

    if len(sample) > 10000:
        sample = sample.sample(n=10000, random_state=42)

    sample.to_csv(SAMPLE_PATH, index=False)
    print("Saved sample to:", SAMPLE_PATH)
    print("Sample shape:", sample.shape)

if __name__ == "__main__":
    main()
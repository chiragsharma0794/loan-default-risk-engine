"""Frozen chronological cohorts, replacing physical-chunk class balancing."""
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "loan.csv"
VALIDATION_PATH = PROJECT_ROOT / "outputs" / "validation" / "data_validation.json"
SEED = 42
TARGET_MAP = {"Fully Paid": 0, "Charged Off": 1}
POLICY = {
    "version": 1,
    "term": "36 months",
    "target_map": TARGET_MAP,
    "train": ["2007-01-01", "2013-01-01"],
    "validation": ["2013-01-01", "2014-01-01"],
    "holdout": ["2014-01-01", "2015-01-01"],
    "boundary_rule": "inclusive start, exclusive end",
    "sampling": "all eligible rows; no class balancing",
    "seed": SEED,
    "primary_metric": "roc_auc",
    "secondary_metrics": ["average_precision", "log_loss", "brier_score"],
}


def source_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_splits(frame):
    """Assign eligibility from metadata only; no feature or score decisions."""
    dates = pd.to_datetime(frame["issue_d"], format="%b-%Y", errors="raise")
    if dates.isna().any():
        raise ValueError("Missing issue_d prevents chronological assignment")
    eligible = (
        frame["term"].str.strip().eq(POLICY["term"])
        & frame["loan_status"].isin(TARGET_MAP)
    )
    split = pd.Series("excluded", index=frame.index, dtype="string")
    for name in ("train", "validation", "holdout"):
        start, end = POLICY[name]
        split.loc[eligible & dates.ge(start) & dates.lt(end)] = name
    return split


def read_contract(path=VALIDATION_PATH):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report["policy"] != POLICY or not report["integrity_passed"]:
        raise ValueError("Split contract changed or integrity checks failed")
    return report


def load_development_data(path=RAW_DATA_PATH, report_path=VALIDATION_PATH, chunksize=50000):
    """Return only train/validation; final holdout never enters model fitting.

    CSV chunks may physically contain holdout records, but these are immediately
    filtered before preprocessing. The full-file checksum is integrity-only.
    """
    report = read_contract(report_path)
    if source_sha256(path) != report["source"]["sha256"]:
        raise ValueError("Raw source changed: do not silently redefine frozen splits")
    parts = {"train": [], "validation": []}
    usecols = list(dict.fromkeys(report["preprocessing"]["raw_features"] + ["issue_d", "loan_status", "term"]))
    for chunk in pd.read_csv(path, usecols=usecols, dtype="string", chunksize=chunksize):
        splits = assign_splits(chunk)
        for name in parts:
            selected = chunk.loc[splits.eq(name)].copy()
            if not selected.empty:
                selected["target"] = selected["loan_status"].map(TARGET_MAP).astype("int8")
                selected.index.name = "source_row"
                parts[name].append(selected)
    result = {name: pd.concat(chunks) for name, chunks in parts.items()}
    for name, frame in result.items():
        if len(frame) != report["splits"][name]["rows"]:
            raise ValueError(f"{name} row count differs from frozen contract")
    if not result["train"].index.intersection(result["validation"].index).empty:
        raise ValueError("Development splits overlap")
    return result["train"], result["validation"], report


if __name__ == "__main__":
    raise SystemExit("Run src/check_leakage.py to validate/freeze cohorts. No balanced sample is generated.")

"""Stage 2 integrity audit. Does not train, score, or tune any model."""
import json
from collections import Counter

import numpy as np
import pandas as pd

try:
    from .stratified_sampling import RAW_DATA_PATH, VALIDATION_PATH, POLICY, TARGET_MAP, assign_splits, source_sha256
    from .feature_screening import candidate_columns, fit_preprocessor, transform_features
except ImportError:
    from stratified_sampling import RAW_DATA_PATH, VALIDATION_PATH, POLICY, TARGET_MAP, assign_splits, source_sha256
    from feature_screening import candidate_columns, fit_preprocessor, transform_features


def check_leakage(path=RAW_DATA_PATH, output_path=VALIDATION_PATH, chunksize=50000):
    fingerprint = source_sha256(path)
    original_stat = path.stat()
    full_hashes, eligible_hashes, split_codes = [], [], []
    parts = {"train": [], "validation": []}
    counts, nonmissing_ids = Counter(), Counter()
    bounds = {}
    total = 0
    for chunk in pd.read_csv(path, dtype="string", chunksize=chunksize):
        columns = chunk.columns.tolist()
        candidates = candidate_columns(columns)
        full_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        split = assign_splits(chunk)
        counts.update(split.value_counts().to_dict())
        for col in ("id", "member_id"):
            nonmissing_ids[col] += int(chunk[col].notna().sum())
        selected = split.ne("excluded")
        eligible_hashes.append(pd.util.hash_pandas_object(chunk.loc[selected, candidates], index=False).to_numpy())
        split_codes.append(split[selected].map({"train": 0, "validation": 1, "holdout": 2}).to_numpy(dtype="int8"))
        for name in ("train", "validation", "holdout"):
            selected = split.eq(name)
            if selected.any():
                dates = pd.to_datetime(chunk.loc[selected, "issue_d"], format="%b-%Y")
                low, high = dates.min().strftime("%Y-%m"), dates.max().strftime("%Y-%m")
                old = bounds.get(name, [low, high])
                bounds[name] = [min(low, old[0]), max(high, old[1])]
                if name in parts:
                    parts[name].append(chunk.loc[selected, candidates + ["loan_status"]].copy())
        total += len(chunk)
        if total % 500000 == 0:
            print(f"Integrity-scanned {total:,} rows", flush=True)
    final_stat = path.stat()
    if (original_stat.st_size, original_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source changed during audit")
    full = pd.Series(np.concatenate(full_hashes))
    signatures = pd.DataFrame({"hash": np.concatenate(eligible_hashes), "split": np.concatenate(split_codes)})
    raw_duplicates = int(full.duplicated().sum())
    candidate_duplicates = int(signatures["hash"].duplicated().sum())
    cross_split = int((signatures.groupby("hash")["split"].nunique() > 1).sum())
    if any(counts[name] == 0 for name in ("train", "validation", "holdout")):
        raise ValueError("An evaluation cohort is empty")
    train, validation = (pd.concat(parts[name]) for name in ("train", "validation"))
    if not train.index.intersection(validation.index).empty:
        raise ValueError("Source row overlap")
    x_train, schema = fit_preprocessor(train)
    x_validation = transform_features(validation, schema)
    if not x_train.dtypes.equals(x_validation.dtypes):
        raise ValueError("Development preprocessing dtype mismatch")
    hashes_train = pd.util.hash_pandas_object(x_train, index=False)
    hashes_validation = pd.util.hash_pandas_object(x_validation, index=False)
    model_overlap = len(set(hashes_train).intersection(hashes_validation))
    splits = {}
    for name in ("train", "validation", "holdout"):
        splits[name] = {"rows": counts[name], "issue_month_range": bounds[name]}
        if name in parts:
            frame = train if name == "train" else validation
            y = frame.loan_status.map(TARGET_MAP)
            if y.nunique() != 2:
                raise ValueError(f"{name} needs both outcome classes")
            splits[name].update({"positive_rows": int(y.sum()), "positive_rate": float(y.mean())})
    report = {
        "stage": 2,
        "source": {"path": "data/loan.csv", "sha256": fingerprint, "bytes": final_stat.st_size,
                   "rows": total, "columns": columns},
        "policy": POLICY,
        "splits": splits,
        "excluded_rows": counts["excluded"],
        "checks": {
            "duplicate_full_row_fingerprints": raw_duplicates,
            "duplicate_eligible_application_fingerprints": candidate_duplicates,
            "cross_split_application_fingerprint_groups": cross_split,
            "train_validation_model_feature_fingerprint_overlap": model_overlap,
            "nonmissing_identifiers": dict(nonmissing_ids),
            "customer_overlap": "not verifiable: IDs absent; fingerprints do not establish customer identity",
            "development_dtypes_match": True,
            "holdout_use": "eligibility, row counts, date bounds and duplicate integrity only; no profiling, fitting or scoring",
        },
        "preprocessing": schema,
        "development_checks": {
            "train_numeric_infinity_count": int(np.isinf(x_train.select_dtypes(include="number").to_numpy()).sum()),
            "validation_numeric_infinity_count": int(np.isinf(x_validation.select_dtypes(include="number").to_numpy()).sum()),
            "validation_unseen_categories": {
                col: int((validation[col].notna() & x_validation[col].isna()).sum())
                for col in schema["categories"]
            },
            "train_emp_length_num_missing": int(x_train["emp_length_num"].isna().sum()),
            "train_emp_length_num_values": sorted(x_train["emp_length_num"].dropna().unique().tolist()),
        },
        "runtime": {"pandas": pd.__version__, "numpy": np.__version__},
        "integrity_passed": raw_duplicates == 0 and cross_split == 0 and model_overlap == 0,
    }
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        for key in ("source", "policy", "preprocessing"):
            if previous[key] != report[key]:
                raise ValueError(f"Frozen {key} changed; do not overwrite the existing contract")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"splits": splits, "checks": report["checks"],
                      "features": len(schema["features"]), "integrity_passed": report["integrity_passed"]}, indent=2))
    if not report["integrity_passed"]:
        raise ValueError("Integrity findings must be resolved before modeling")
    return report


if __name__ == "__main__":
    check_leakage()

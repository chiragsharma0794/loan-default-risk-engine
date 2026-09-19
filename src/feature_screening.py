"""Prediction-time exclusions and training-fitted preprocessing."""
import numpy as np
import pandas as pd

try:
    from .feature_engineering import engineer_features
except ImportError:
    from feature_engineering import engineer_features

EXCLUDED_COLUMNS = {
    "id", "member_id", "url", "emp_title", "title", "desc",
    "target", "target_default", "loan_status", "issue_d", "source_row", "split",
    "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
    "last_fico_range_high", "last_fico_range_low", "pymnt_plan",
    "deferral_term", "payment_plan_start_date", "orig_projected_additional_accrued_interest",
    "disbursement_method",
}
CATEGORICAL_COLUMNS = {
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "zip_code", "addr_state",
    "earliest_cr_line", "initial_list_status", "application_type",
    "verification_status_joint", "sec_app_earliest_cr_line",
}


def candidate_columns(columns):
    return [c for c in columns if c not in EXCLUDED_COLUMNS and "hardship" not in c]


def normalize_features(df, columns):
    out = df.loc[:, columns].copy()
    for col in columns:
        if col in CATEGORICAL_COLUMNS:
            out[col] = out[col].astype("string").str.strip().replace("", pd.NA)
        else:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    for col in ("annual_inc", "loan_amnt", "total_bal_ex_mort", "open_acc", "total_acc"):
        if col in out:
            out.loc[out[col] < 0, col] = np.nan
    return out


def screen_features(df, feature_columns=None):
    """Fit missingness filtering on train, or apply its explicit column list."""
    columns = candidate_columns(df.columns) if feature_columns is None else feature_columns
    if set(columns) - set(candidate_columns(columns)):
        raise ValueError("Metadata or post-origination feature in requested schema")
    out = normalize_features(df, columns)
    if feature_columns is None:
        out = out.loc[:, out.isna().mean().le(0.8)]
    return out


def fit_preprocessor(train):
    screened = screen_features(train)
    engineered = engineer_features(screened.copy())
    schema = {
        "raw_features": screened.columns.tolist(),
        "features": engineered.columns.tolist(),
        "dropped_missing": [c for c in candidate_columns(train.columns) if c not in screened],
        "categories": {
            c: sorted(engineered[c].dropna().astype(str).unique().tolist())
            for c in engineered if c in CATEGORICAL_COLUMNS
        },
        "numeric_missing": "NaN, handled natively by LightGBM; no imputation",
        "unseen_category": "missing using training category vocabulary",
    }
    return transform_features(train, schema), schema


def transform_features(df, schema):
    out = engineer_features(screen_features(df, schema["raw_features"]))
    out = out.loc[:, schema["features"]]
    for col, categories in schema["categories"].items():
        known = out[col].where(out[col].isin(categories))
        out[col] = pd.Categorical(known, categories=categories)
    numeric = out.select_dtypes(include="number").columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


if __name__ == "__main__":
    raise SystemExit("Preprocessing is fitted on training only by check_leakage.py; no pooled CSV screening.")

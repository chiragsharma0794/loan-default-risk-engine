import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "loan.csv"

TARGET_MAP = {
    "Fully Paid": 0,
    "Current": 0,
    "Charged Off": 1,
    "Default": 1
}

def clean_percent_column(series):
    if series.dtype == "object":
        series = series.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(series, errors="coerce")

def main():
    print("Reading data...")
    df = pd.read_csv(DATA_PATH, nrows=100000, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()]

    df = df[df["loan_status"].isin(TARGET_MAP.keys())].copy()
    df["target_default"] = df["loan_status"].map(TARGET_MAP)

    print("\nTarget distribution:")
    print(df["target_default"].value_counts())

    features = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "revol_util",
        "total_acc",
        "open_acc",
        "inq_last_6mths"
    ]

    df = df[features + ["target_default"]].copy()

    df["int_rate"] = clean_percent_column(df["int_rate"])
    df["revol_util"] = clean_percent_column(df["revol_util"])

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMissing values before modelling:")
    print(df.isna().sum())

    X = df[features]
    y = df["target_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    results = X_test.copy()
    results["actual"] = y_test.values
    results["risk_score"] = y_prob

    output_path = BASE_DIR / "outputs" / "basic_model_output.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nSaved predictions to: {output_path}")

if __name__ == "__main__":
    main()
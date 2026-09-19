# Loan Default Risk Engine

A reproducible study of loan-default ranking with LightGBM, chronological evaluation, SHAP explanations and Power BI reporting data. The selected model achieves **0.65950 holdout ROC-AUC**, compared with **0.64283** for the baseline. This is a historical modeling project, not a production credit-decision system.

## Problem and data

The task is to estimate default risk using candidate application/credit features and examine where modeled risk differs across loan segments. Historical availability of every bureau field has not been established, so prediction-at-origination suitability remains an assumption to validate.

The local Lending Club-style loan dataset contains **2,260,668 rows and 145 columns**, occupying 1,189,395,649 bytes. The source is `data/loan.csv`; `data/LCDataDictionary.xlsx` supplies field descriptions. Download provenance and redistribution permission are not established by the repository; the dataset is not bundled for publication.

The modeled population contains only resolved **36-month** loans: Fully Paid = 0, Charged Off = 1. Current/unresolved loans and other statuses are excluded. Classes retain their natural prevalence; no class balancing is used.

| Split | Issue period | Loans | Role |
|---|---|---:|---|
| Training | 2007–2012 | 72,566 | Fit preprocessing and models |
| Validation | 2013 | 100,422 | Early stopping and parameter selection |
| Final holdout | 2014 | 162,570 | One final baseline/candidate comparison |

The holdout contains 22,315 defaults (13.72639%). It has already been used for evaluation and is unavailable for further tuning. Row positions (`source_row`) identify records only within the exact source file; both original identifier fields are missing, preventing borrower-level overlap checks.

## Pipeline and leakage controls

1. `check_leakage.py` audits the raw file in chunks and freezes the split/preprocessing contract in `outputs/validation/data_validation.json`.
2. `stratified_sampling.py` loads all eligible development rows using the time split, despite its historical filename. It does not perform balanced sampling.
3. `feature_screening.py` excludes known target, repayment, recovery, settlement, hardship, recent servicing/FICO and metadata fields. Missingness filtering (>80% missing) and categorical vocabularies are fitted on training only. Unknown categories become missing; LightGBM handles numeric missing values without imputation.
4. `feature_engineering.py` adds log income/loan amount, loan-to-income and balance-to-income ratios, open-account ratio, and numeric employment length. Income ratios use income + 1 only for positive income; invalid/negative financial inputs and infinities become missing. The final schema contains 72 inputs.
5. `lgbm_model.py` freezes a baseline; `tune_lgbm.py` searches on validation; `final_evaluation.py` performs the fixed final comparison.
6. `shap_explainability.py` explains the saved final model; `prepare_powerbi_data.py` joins narrow business dimensions to saved predictions and exports reporting tables.

The integrity audit found no duplicate full-row fingerprints or cross-split candidate-feature fingerprints and no training/validation transformed-feature overlap. These checks do not establish borrower independence or prove that every field was available at origination. Full raw data is not loaded into one dataframe; eligible development matrices are retained in memory.

## LightGBM and tuning

Baseline: learning rate 0.05, maximum depth 7, up to 500 trees, validation early stopping after 50 rounds; selected 31 trees. Class weighting is disabled. Model, search, bootstrap and explanation seeds are 42.

Tuning used **24 seeded random trials** on the full training/validation cohorts, a 1,500-tree cap and validation early stopping. The search covered learning rate, leaf/depth limits, minimum child samples, feature/row sampling, L1/L2 regularization and minimum split gain. Trial 12 won validation AUC: **0.65963371**, versus baseline **0.64264643**. See [search details](docs/tuning_results.md) and [exact parameters](outputs/tuning/best_params.json).

The final tuned configuration uses 57 trees, depth 3, 8 leaves, learning rate 0.03719515, minimum child samples 50, feature fraction 0.85, row fraction 1.0, L1 0.00118283 and L2 0.06745782. It was refitted on the original 72,566 training rows, not training plus validation, with the tree count frozen before holdout scoring. Full parameters and environment metadata are saved in [final metrics](outputs/final_model/metrics.json).

## Results

Both models were evaluated on the same 162,570-row holdout:

| Metric | Baseline | Final tuned | Better direction |
|---|---:|---:|---|
| ROC-AUC (primary) | 0.64283263 | **0.65949575** | Higher |
| Average precision | 0.20590956 | **0.21580617** | Higher |
| Log loss | 0.38518541 | **0.38281766** | Lower |
| Brier score | 0.11506916 | **0.11452292** | Lower |

AUC improved by **0.01666311**. The paired, class-stratified 500-bootstrap 95% interval for the difference is **[0.01522395, 0.01807960]**, assuming independent loan rows. This supported selecting the tuned model under the recorded selection rule. Average precision is not trapezoidal PR-AUC. See [final evaluation](docs/final_evaluation.md).

At the unchanged **0.5 threshold**, both models predict zero positives: TN 140,255, FP 0, FN 22,315, TP 0. Recall is 0%; precision is undefined. Final mean probability is 13.13030%, versus a 13.72639% observed default rate; calibration remains imperfect. Ranking improvement does not establish a useful operational approval threshold.

The former README's 0.6885 AUC and production-readiness claims are not supported by the frozen evaluation. Historical sampling and outputs are not directly comparable to this chronological, naturally imbalanced cohort.

## Explainability

TreeSHAP explains the saved 57-tree model on a uniform random sample of **10,000 holdout rows** (seed 42), plus four separate local examples. Contributions are in default log-odds and reconstruct saved scores; they are not causal effects.

The leading global drivers are interest rate, annual income, sub-grade, ZIP code and accounts opened in the past 24 months. Higher interest rates, recent account openings and loan-to-income ratios generally raise modeled scores; higher income generally lowers them. ZIP/date proxies and correlated features need care. Debt-to-income is not a leading driver in this final model.

![Final model SHAP importance](outputs/explainability/shap_importance.png)

[Explanation report](docs/explainability.md) includes sampling, effect directions, dependence plots and two true-negative/two false-negative local cases. No true-positive or false-positive cases exist at threshold 0.5.

## Power BI reporting

The local export contains seven CSVs (19.26 MB total) providing scored loans, band summaries/dimension, baseline/final metrics, calibration, gain importance and SHAP importance. [Data dictionary and relationships](docs/powerbi_data.md) define their grain and interpretation. Reporting bands are [0,10%), [10%,15%), [15%,20%) and [20%,100%]; they are descriptive, not decision rules.

Six aggregate/reference CSVs are included in Git; `scored_loans.csv` stays local and is regenerated by the reporting script. A fresh clone needs that local export to use every dashboard page.

**Dashboard status: specification only.** Direct Power BI editing was unavailable. The [four-page dashboard design](docs/powerbi_dashboard_design.md) specifies Executive Risk Overview, Portfolio Risk Analysis, Model Performance and Explainability, with exact fields, layouts, relationships, 28 DAX measures, slicers and acceptance checks. No PBIX/PBIP or validated screenshots exist. DAX execution, visual rendering and interaction checks remain pending in Power BI. Portfolio slicing must not imply that frozen model metrics or global SHAP values were recomputed for a segment.

## Repository map

Environment internals, caches and individual plot filenames are omitted below. Row-level data stays local.

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── loan.csv                         # local only
│   └── LCDataDictionary.xlsx
├── docs/
│   ├── execution_plan.md
│   ├── PROJECT_STATE.md
│   ├── project_audit.md
│   ├── data_validation.md
│   ├── baseline_results.md
│   ├── tuning_results.md
│   ├── final_evaluation.md
│   ├── explainability.md
│   ├── powerbi_data.md
│   └── powerbi_dashboard_design.md
├── src/
│   ├── check_leakage.py
│   ├── stratified_sampling.py
│   ├── feature_screening.py
│   ├── feature_engineering.py
│   ├── lgbm_model.py
│   ├── tune_lgbm.py
│   ├── final_evaluation.py
│   ├── shap_explainability.py
│   ├── prepare_powerbi_data.py
│   ├── basic_model.py                   # legacy exploration
│   ├── column_audit.py                  # legacy exploration
│   ├── loan_sample_10000.py              # legacy exploration
│   ├── status_audit.py                   # legacy exploration
│   └── target_mapping.py                # legacy exploration
├── tests/
│   ├── test_stage2_validation.py
│   ├── test_final_evaluation.py
│   ├── test_shap_examples.py
│   └── test_powerbi_data.py
└── outputs/
    ├── validation/data_validation.json
    ├── baseline/                       # model, metrics, gain, local predictions
    ├── tuning/                         # trials, parameters, model, local predictions
    ├── final_model/                    # model, metrics, calibration, local predictions
    ├── explainability/                 # 5 CSVs, 10 PNGs, SHAP arrays, metadata
    ├── powerbi/                        # 7 CSVs and provenance manifest
    ├── model_comparison.csv
    ├── sample/loan_sample_10000.csv     # legacy local sample
    ├── table/column_audit.csv           # legacy partial audit
    ├── basic_model_output.csv          # legacy local predictions
    ├── lgbm_feature_importance.png      # legacy plot
    └── shap_summary.png                # legacy plot
```

Legacy scripts refer to `data/raw/loan.csv`, and `basic_model.py`/`target_mapping.py` use a different target including Current loans. They are retained for provenance, not run by the validated pipeline. Shared preprocessing is reused; repeated metric calculations across frozen stages are left intact to preserve code hashes. Paths in active source are repository-relative; local `venv` metadata is machine-specific and must not be copied to another machine.

## Setup and reproduction

Run from the repository root. Frozen modeling runs record **Python 3.12.13 on Windows**; the currently available interpreter reports **3.12.14**. Stage 9 tests pass on the latter and all installed package versions match `requirements.txt`. Exact clean-environment retraining has not been repeated. Prefer the recorded Python patch version when reproducing historical model artifacts; platform/runtime differences can affect byte-for-byte equality.

For a new environment, use a Python 3.12 installation (the Windows launcher must be installed for this example):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -B -m unittest discover -s tests
```

For the existing local environment, run only the last two checks. These tests use small synthetic examples; they do not retrain the model or require the full dataset. On other systems substitute the environment's Python executable. A clean-environment rebuild remains untested.

To regenerate artifacts, use a **separate fresh workspace** containing the same source code/dependencies and the original raw CSV at `data/loan.csv`, without existing generated stage directories or `outputs/model_comparison.csv`. Keep the saved results in this workspace. Obtain the matching source through an authorized source; the repository does not provide a confirmed download URL. Expected raw SHA-256:

```text
23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c
```

Then run in this order, in that fresh workspace only:

```powershell
.\venv\Scripts\python.exe -B src/check_leakage.py
.\venv\Scripts\python.exe -B src/lgbm_model.py
.\venv\Scripts\python.exe -B src/tune_lgbm.py
.\venv\Scripts\python.exe -B src/final_evaluation.py
.\venv\Scripts\python.exe -B src/shap_explainability.py
.\venv\Scripts\python.exe -B src/prepare_powerbi_data.py
```

Sampling/screening/engineering are imported helpers, not separate CSV-producing steps. Stage 3–7 commands refuse existing outputs; downstream stages expect the standard upstream paths, so changing one output directory alone is not a full rerun strategy. The integrity audit scans the full raw file in chunks; subsequent steps hash it and select needed rows/columns. Reproducing these numbers requires the same complete source CSV. Save LightGBM models through its native save API; manual newline conversion previously broke serialized tree offsets on Windows.

A fresh rerun reproduces a historical evaluation, not a new untouched holdout. For new modeling decisions, reserve new evaluation data. Build the Power BI report separately from its specification and complete the listed engine/rendering checks before presenting it as implemented.

## Limitations and next improvements

- Resolved historical 36-month loans are a selected population, not all applications or today's portfolio; unresolved loans introduce a maturity/selection concern.
- Missing borrower IDs prevent borrower-disjoint checks and clustered uncertainty estimates. Bureau timing and actual prediction-time availability remain unverified.
- The model's discrimination is modest; its default threshold detects no defaults. Calibration and a business-cost-based threshold need separate development data and a new held-out evaluation, not optimization on this consumed holdout.
- ZIP/date proxies and correlated inputs warrant subgroup, stability and fairness investigation. SHAP establishes neither causation nor compliance.
- Dataset provenance/licensing, clean-environment reproducibility and actual Power BI implementation remain open. Production deployment has not been verified.

Future work should establish data provenance and timing, validate on a newer independent cohort, assess calibration and decision costs on development data, and implement/test the specified report. These are proposed improvements, not completed experiments.

## Publication notes

The results table and SHAP graphic above describe the evaluated model. The older plots are retained for history. Power BI remains a design, with implementation and screenshots pending.

Source, tests, pinned dependencies, reports, small models, aggregate metrics and explanation plots are included. Raw data, row-level predictions, SHAP arrays and environments stay local. The original history contains `outputs/sample/loan_sample_10000.csv` and `outputs/basic_model_output.csv`; these are removed from the updated branch tip, but remain accessible in older commits. The data dictionary contains field descriptions rather than loan records. Dataset acquisition and redistribution rights remain unverified.

All 13 tests and dependency checks passed, and saved artifact hashes matched. The change record is in [PROJECT_STATE.md](docs/PROJECT_STATE.md).

# Loan Default Risk Prediction — Original Work Plan

## Project Goal

This plan records the nine-stage sequence used to revise `loan-default-risk-prediction`. It describes the starting point and planned work; results are recorded in the individual reports and README.

At the start, the project contained:

* LightGBM classification model
* SHAP explainability
* ROC-AUC of approximately 0.68
* Around 2.2 million rows of local data
* Existing scripts for sampling, feature screening, feature engineering, leakage checking, model training, and SHAP

The work covered:

1. Understand and validate the existing project
2. Establish a reliable baseline
3. Tune LightGBM efficiently
4. Compare baseline vs tuned model
5. Update SHAP for the final model
6. Prepare Power BI datasets
7. Build a professional Power BI dashboard
8. Improve documentation and portfolio quality

---

# Engineering constraints

The stages separate data validation, baseline fitting, tuning and holdout evaluation. Model decisions use repository evidence; the holdout stays outside model fitting and parameter search until the final comparison. Large-data inspection uses samples, schemas or chunked reads.

Working code and shared preprocessing are reused where practical. Refactoring is limited to demonstrated problems; changed code is checked before its results are used. Each report records decisions, evidence and unresolved issues, with a short project-state record for navigation.

---

# Stage 1 — Project Audit

## Objective

Understand the current project before making changes.

## Inspect

* `README.md`
* `.gitignore`
* `data/`
* `outputs/`
* `src/stratified_sampling.py`
* `src/feature_screening.py`
* `src/feature_engineering.py`
* `src/lgbm_model.py`
* `src/check_leakage.py`
* `src/shap_explainability.py`

## Check

* Available datasets and schemas
* Target column
* Target distribution
* Sampling logic
* Feature screening
* Feature engineering
* Train/validation/test split
* Categorical and missing-value handling
* LightGBM parameters
* Evaluation metrics
* Leakage checks
* SHAP implementation
* Existing outputs
* How the current ~0.68 ROC-AUC is produced

Inspect large data through samples, metadata or chunked reads.

## Record

Document:

* Current pipeline
* Important findings
* Risks/issues
* Potential improvements
* Anything that must be fixed before modeling

Record findings in `docs/project_audit.md`.

The initial audit is read-only; fixes and tuning follow separately.

---

# Stage 2 — Validate Data Split and Leakage

## Objective

Make sure model evaluation is trustworthy before tuning.

## Check

* Target leakage
* Future-information leakage
* Duplicate records
* Same customer/loan appearing across splits
* Features unavailable at prediction time
* Sampling bias
* Class imbalance
* Whether the current split should be random, stratified, or time-based
* Preprocessing consistency
* Missing-value handling
* Categorical handling

Review existing sampling and leakage-check scripts before changing them.

## Decide

Define the evaluation strategy that will be used for all later experiments:

* Training set
* Validation strategy
* Final holdout/test set
* Random seed
* Primary metric
* Useful secondary metrics

ROC-AUC remains the primary metric unless there is a strong reason otherwise.

Useful secondary metrics may include:

* PR-AUC
* Log loss
* KS statistic
* Precision
* Recall
* F1
* Calibration

Only use metrics that add real value.

## Record

* Leakage findings
* Final split strategy
* Final evaluation methodology
* Any required fixes

Tuning follows the split and leakage review.

---

# Stage 3 — Reproduce and Freeze Baseline

## Objective

Create a reproducible baseline before tuning.

## Tasks

Run the existing LightGBM model using the validated methodology from Stage 2.

Record:

* Training rows
* Validation rows
* Test rows
* Default rate
* Number of features
* Model parameters
* Random seed
* ROC-AUC
* Approved secondary metrics
* Best iteration if applicable
* Feature importance

Compare the reproduced result with the existing ~0.68 ROC-AUC.

## Outputs

Prefer a simple structure such as:

```text
outputs/baseline/
├── metrics.json
├── feature_importance.csv
├── predictions.csv
└── model.txt
```

Save the artifacts needed for later comparison.

## Record

* Reproduced baseline metrics
* Exact baseline parameters
* Split information
* Any difference from the historical result

The baseline configuration stays fixed during reproduction.

---

# Stage 4 — LightGBM Hyperparameter Tuning

## Objective

Improve LightGBM using an efficient and defensible tuning process.

## Rules

* Never tune using the final holdout/test set.
* Keep the feature set unchanged unless an actual bug is found.
* Avoid large brute-force grids.
* Use early stopping.
* Keep runtime practical.

Use Optuna or another efficient method if appropriate.

Consider parameters such as:

* `learning_rate`
* `num_leaves`
* `max_depth`
* `min_child_samples`
* `feature_fraction` / `colsample_bytree`
* `bagging_fraction` / `subsample`
* `bagging_freq`
* `lambda_l1`
* `lambda_l2`
* `min_split_gain`

Only tune imbalance-related parameters if justified by the data.

If tuning on all 2.2 million rows is unnecessarily expensive, use a representative training sample for parameter search and then validate the selected parameters on the larger training data.

## Suggested Files

```text
src/tune_lgbm.py
```

```text
outputs/tuning/
├── tuning_trials.csv
└── best_params.json
```

Keep search code and results together.

## Record

* Tuning method
* Search space
* Number of trials
* Best parameters
* Best validation result
* Baseline validation result
* Runtime considerations

Holdout evaluation follows parameter selection.

---

# Stage 5 — Final Model Evaluation

## Objective

Determine whether tuning actually improved the model.

## Tasks

Train the tuned candidate correctly and compare it with the frozen baseline on the same untouched holdout/test dataset.

Compare:

* ROC-AUC
* Approved secondary metrics
* Precision/recall where useful
* Confusion matrix at clearly documented thresholds
* Calibration if relevant

Model selection also needs evidence from the reserved holdout.

Use holdout performance to determine whether the tuning improvement generalizes.

## Outputs

Prefer:

```text
outputs/final_model/
```

and a simple model comparison file such as:

```text
outputs/model_comparison.csv
```

## Record

Report:

* Baseline holdout results
* Tuned-model holdout results
* Absolute difference
* Whether improvement generalized
* Final model selected
* Evidence supporting that selection

---

# Stage 6 — Final SHAP Explainability

## Objective

Generate explainability for the final selected model.

## Tasks

Reuse the existing `shap_explainability.py` where possible.

Generate:

* Global feature importance
* SHAP summary/beeswarm plot
* Direction of important feature effects
* A few useful dependence plots
* A few representative local explanations

Possible local examples:

* Correct high-risk prediction
* Correct low-risk prediction
* False positive
* False negative

Use a representative sample instead of millions of rows if full SHAP computation is unnecessarily expensive.

Also check whether SHAP exposes suspiciously dominant or potentially leaky features.

## Outputs

Prefer:

```text
outputs/explainability/
```

Create a simple machine-readable SHAP importance table containing fields such as:

```text
feature
mean_abs_shap
importance_rank
```

## Record

* Main model drivers
* Direction of effects
* Interesting local examples
* Any suspicious behavior
* SHAP sampling method

---

# Stage 7 — Prepare Power BI Data

## Objective

Create clean reporting datasets for Power BI.

Power BI uses the prepared reporting tables.

## Suggested Reporting Tables

### Loan-level scored data

Where available:

```text
loan_id
actual_default
predicted_probability
predicted_class
risk_band
```

Add only useful business dimensions that actually exist.

### Risk-band summary

```text
risk_band
loan_count
default_count
default_rate
avg_predicted_probability
```

### Model performance

Include baseline and final model metrics.

### Feature importance

```text
feature
importance
importance_rank
```

### SHAP importance

```text
feature
mean_abs_shap
importance_rank
```

Create additional segment-level tables only when supported by actual dataset columns.

Reporting dimensions come from the source data.

## Suggested Script

```text
src/prepare_powerbi_data.py
```

## Output

```text
outputs/powerbi/
```

## Tasks

* Define useful risk bands
* Document risk-band boundaries
* Decide whether Power BI needs full scored data, aggregated data, or both
* Keep files reasonably sized
* Validate exported totals against Python results
* Provide a simple data dictionary

## Record

* Reporting tables
* Purpose of each table
* Relationships between tables
* Risk-band methodology
* Recommended Power BI data model

---

# Stage 8 — Power BI Dashboard

## Objective

Build a professional dashboard without unnecessary complexity.

## Suggested Pages

### 1. Executive Risk Overview

Possible KPIs:

* Total loans
* Actual defaults
* Default rate
* Average predicted probability
* High-risk loans
* Final ROC-AUC

Useful visuals:

* Risk-band distribution
* Default rate by risk band
* Actual vs predicted risk
* Risk trend if time data exists

### 2. Portfolio Risk Analysis

Analyze risk across meaningful dimensions that actually exist in the dataset.

Examples only if available:

* Loan amount
* Income band
* Loan purpose
* Employment category
* Region
* Credit variables
* Time period

### 3. Model Performance

Show:

* Baseline vs final model
* ROC-AUC
* PR-AUC if used
* Confusion matrix
* Threshold metrics
* Calibration information if available

### 4. Explainability

Show:

* Top SHAP drivers
* Feature importance
* Direction of impact
* Simple business interpretation

## Design Rules

* Keep the layout clean
* Avoid visual clutter
* Use consistent formatting
* Use meaningful slicers
* Prefer DAX measures over unnecessary calculated columns
* Predictive relationships are presented as associations

If direct Power BI editing is available, build the dashboard.

If it is not available, create a precise dashboard specification including:

* Page layouts
* Required visuals
* Fields
* Relationships
* DAX measures
* Filters and slicers

Create:

```text
docs/powerbi_dashboard_design.md
```

if useful.

---

# Stage 9 — Documentation and Portfolio Cleanup

## Objective

Make the finished project easy to understand and reproduce.

The last stage covers documentation and cleanup, with modeling results fixed.

## Review

* Repository structure
* Hard-coded paths
* Duplicate code
* Random seeds
* Output locations
* Naming
* Unused files
* `.gitignore`
* Dependencies
* README
* Reproducibility

Refactor only when there is a clear benefit.

## README

Update `README.md` to include:

1. Project overview
2. Business problem
3. Dataset and scale
4. Pipeline
5. Leakage prevention
6. Feature engineering
7. LightGBM modeling
8. Hyperparameter tuning
9. Baseline vs final results
10. SHAP explainability
11. Power BI dashboard
12. Repository structure
13. How to run the project
14. Key results
15. Limitations
16. Future improvements

Use the recorded results. Performance claims stay within the evaluated cohort and metrics.

Create or update `requirements.txt` if needed.

## Completion record

Provide:

* Final repository tree
* Files added
* Files modified
* Final model metrics
* Reproduction instructions
* Known limitations
* Recommended GitHub presentation

---

# Experiment records

Each stage report records the methods, findings, checks and remaining issues. The README and `docs/PROJECT_STATE.md` describe the completed project; earlier reports retain the state of knowledge at the time of each experiment.

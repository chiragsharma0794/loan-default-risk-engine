# Initial project audit

Audit date: 2026-09-19, before the modeling fixes. This report records the original pipeline and its available evidence; later changes are covered in the validation and experiment reports.

## Evidence and available data

The audit covered the README, ignore rules, six pipeline scripts, supporting scripts, two saved plots and data dictionary. A chunked scan of `loan_status` and `issue_d` used 100,000 rows at a time. Schema checks used the first 1,000 raw rows; feature checks used the existing 10,000-row sample.

| Available file | Verified contents |
| --- | --- |
| `data/loan.csv` | 1,189,395,649 bytes; 2,260,668 rows; 145 columns |
| `data/LCDataDictionary.xlsx` | LoanStats, browseNotes, RejectStats sheets; field descriptions, not observations |
| `outputs/sample/loan_sample_10000.csv` | 10,000 rows, same 145 ordered columns as raw header; no target |
| `outputs/table/column_audit.csv` | 145 rows, 8 columns; same raw field names; each field's null/non-null counts total 50,000, not the full dataset |
| `outputs/basic_model_output.csv` | 19,849 rows, 10 columns; eight numeric inputs plus `actual`, `risk_score` |
| `outputs/lgbm_feature_importance.png` | Saved top-20 gain plot |
| `outputs/shap_summary.png` | Saved top-20 SHAP beeswarm |

No `data/raw`, `data/interim`, or `data/processed` directories exist. The stratified, filtered, and engineered datasets referenced by the pipeline are absent. No saved LightGBM model, predictions, metrics file, training log, split manifest, or dependency manifest was found. `docs/PROJECT_STATE.md` was empty. This workspace has no local `.git` entry; Git status could not be obtained.

The raw schema includes identifiers (`id`, `member_id`), loan terms (`loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`), borrower/credit fields (`annual_inc`, `dti`, `emp_length`, `earliest_cr_line`), dates (`issue_d`), `loan_status`, repayment/balance fields, hardship fields, and settlement fields. `target` is generated, not present in raw data. The sample has 109 numeric and 36 text columns under the inspected runtime. Dtypes are inferred, not a frozen schema. The existing column audit provides the complete field inventory, but its missingness statistics cover only the first 50,000 rows.

## Target distribution

Full raw-file counts from the two-column chunk scan:

| `loan_status` | Rows | Current mapping |
| --- | ---: | --- |
| Fully Paid | 1,041,952 | 0 |
| Current | 919,695 | 0 |
| Charged Off | 261,655 | 1 |
| Default | 31 | 1 |
| Late (31-120 days) | 21,897 | Excluded |
| In Grace Period | 8,952 | Excluded |
| Late (16-30 days) | 3,737 | Excluded |
| Does not meet the credit policy. Status:Fully Paid | 1,988 | Excluded |
| Does not meet the credit policy. Status:Charged Off | 761 | Excluded |

Mapping retains 2,223,333 rows: 261,686 positives and 1,961,647 negatives, a **11.76999% mapped default rate**; 37,335 rows are excluded. This is a status-based label rate, not an eventual-default estimate. The dictionary defines `loan_status` as current status. `Current` loans have unresolved outcomes.

The existing 10,000-row sample contains 4,560 Fully Paid, 4,118 Current, 1,132 Charged Off, and 190 excluded-status rows. It is not the balanced modeling sample. Mapped sample rows total 9,810. Both ID fields are entirely missing in those rows; customer/loan overlap cannot simply be assumed checkable from them.

## Current pipeline, as implemented

1. **Sampling:** `stratified_sampling.py` expects `data/raw/loan.csv`, maps the four statuses to `target`, and processes 100,000-row physical chunks. It retains every positive and at most an equal number of negatives per chunk, skipping chunks with no positives. If the combined result exceeds 100,000 rows, it samples 50,000 per class and shuffles, using seed 42. This is the intended size, not a verified saved artifact. It retains sampled chunks in memory before final reduction. Edge cases with insufficient rows per class or no positive chunks are not guarded.
2. **Screening:** removes six identifier/free-text fields, the explicit repayment/settlement/date blacklist (including `issue_d` and last FICO fields), all names containing `hardship`, and columns with strictly more than 80% missingness. `loan_status` remains until modeling. This missingness selection occurs before splitting. As an illustrative in-memory execution on the mapped existing sample, `screen_features` reduced 9,810 × 146 to 9,810 × 88: 86 potential inputs plus status and target. This is not the historical training schema.
3. **Engineering:** adds `log_annual_inc`, `log_loan_amnt`, `loan_to_inc_ratio`, `total_bal_to_inc_ratio`, `open_acc_ratio`, and `emp_length_num` where source fields exist. Ratios divide by the denominator plus 1. Original fields remain. No finite-value/domain checks or general imputation are implemented. The employment regex is `r'(d+)'`, matching literal d characters rather than digits: it matched zero employment values in the 10,000-row sample; the subsequent fill would set all these engineered values to zero. The function also reads its configured input CSV header to report added columns, so it is not fully independent of that missing file.
4. **LightGBM:** drops target and status; converts object/category columns to pandas categories before splitting. Numeric missing values are passed through without imputation; text dates such as `earliest_cr_line` are treated as categories rather than durations. Uses stratified 80/20 random split with seed 42. Explicit classifier arguments: `n_estimators=500`, `learning_rate=0.05`, `max_depth=7`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`; all other parameters depend on library defaults. Fits with the 20% subset as `eval_set` and 50-round early stopping, with no explicit `eval_metric`. It prints ROC-AUC on that same subset and saves only the gain PNG. Imported classification-report functionality is not called. No separate validation set or untouched final holdout is established.
5. **Leakage helper:** trains a 50-tree LightGBM model on the entire engineered dataset and prints the top ten gain features. It is a diagnostic heuristic, not a temporal, duplicate, split-overlap, or prediction-time availability check. Running it would consume all rows, including any implicitly intended holdout.
6. **SHAP:** independently trains `LGBMClassifier(n_estimators=77, random_state=42, n_jobs=-1)` on all engineered rows, then explains 10,000 sampled training rows with seed 42 using TreeExplainer. Handles a list-valued binary output by selecting class 1 and saves only a beeswarm. It does not load the evaluated model and does not specify the evaluated model's learning rate, maximum depth, or class weight. The comment that 77 was the best iteration is not supported by a saved training log.

The supporting sample script selects up to 500 rows from each physical chunk, then reduces to 10,000, so the final shorter chunk can be overrepresented. The older basic model instead reads the first 100,000 rows, applies the same status mapping, and fits median imputation, standard scaling, and balanced logistic regression on eight numeric features. These are separate workflows, not the current LightGBM baseline.

## What the existing results establish

- README reports LightGBM ROC-AUC **0.6885** and an earlier **0.9975**. Neither is independently recoverable from saved model/prediction/log artifacts. The current code's route to the reported score is balanced sampling → screening → engineering → random 80/20 split → early stopping and ROC-AUC on the same 20%. No baseline reproduction was attempted in Stage 1.
- Stored basic-model predictions contain 19,845 negatives and only 4 positives. A tie-aware rank calculation over these existing scores gives ROC-AUC **0.3673721340**; scores range from approximately 0.0000056453 to 1.0. This is an audit of an existing output, not new holdout evaluation, and does not verify the LightGBM claim. Its schema matches `basic_model.py`; exact run provenance is absent.
- The gain plot ranks `sub_grade`, `zip_code`, and `earliest_cr_line` above `int_rate`. The SHAP plot's top five are `sub_grade`, `zip_code`, `acc_open_past_24mths`, `earliest_cr_line`, and `int_rate`. Higher interest rate and loan-to-income values visually tend toward positive SHAP values, but the plot cannot prove causal effects or absence of leakage. README's description omits dominant categorical drivers and overstates production readiness.
- Raw `issue_d` spans June 2007–December 2018, with no missing dates in the scanned column. Physical chunks are not uniform date strata: chunk 5 spans September 2016–March 2018; chunk 14 spans January 2015–March 2017; chunk 22 spans June 2007–December 2017. Chunk 1 has only 19 mapped positives out of 100,000 raw rows, versus 18,398 in chunk 13. Matching class counts within a chunk does not establish matched dates, loan maturity, or absence of temporal bias.

## Risks and prerequisites before further modeling

1. **Execution is blocked as laid out.** Scripts point to `data/raw/loan.csv`, while data is at `data/loan.csv`; intermediates are missing. Screening also assumes the processed directory already exists. The configured input path and output-directory creation need correction.
2. **Define outcomes and evaluation first.** Review the treatment of unresolved Current loans, excluded statuses, prediction time, and outcome horizon. Establish train/validation/final holdout boundaries before preprocessing and before any modeling or leakage-helper fit. Preserve identifiers/dates separately for validation rather than discarding evidence early. The split and outcome policy were unresolved at this point.
3. **Sampling and preprocessing can bias evaluation.** Artificial 50/50 sampling changes prevalence; reported probabilities cannot be assumed calibrated to the portfolio. Chunk balancing is not evidence of temporal control. Missingness-based selection and categorical vocabularies are currently derived before splitting; define consistent training-fitted preprocessing. `class_weight='balanced'` adds no class reweighting when the training classes are already equal.
4. **Feature availability remains unverified.** A blacklist is incomplete evidence of origination availability. For example, `pymnt_plan` survives illustrative screening and is described as whether a payment plan has been put in place. Review it and other retained fields against timing semantics. Non-hardship-named related fields such as `deferral_term` and `payment_plan_start_date` rely on missingness filtering rather than explicit removal. Dominant zip/date features warrant review, not an unsupported conclusion that they leak.
5. **Fix the demonstrated employment parsing bug before the baseline.** Also check numerical validity of logs/ratios and define meaningful missing-value handling. Actual engineered values cannot be verified because that dataset is absent.
6. **Reproducibility is incomplete.** `py` reports no installed Python. The bundled runtime provides pandas 3.0.1 and NumPy/openpyxl, but lacks LightGBM, scikit-learn, and SHAP. Its sample CSV inference uses pandas string dtype, while model scripts select only object/category for conversion; verify compatibility in the chosen environment. Record dependency versions and model defaults before reproducing results.
7. **Artifacts need provenance in later stages.** Persist split information, metrics, parameters, best iteration, feature schema, model, and appropriate predictions when reproducing the baseline. SHAP must eventually explain that selected model, not its separate full-data fit. Those artifacts were absent at audit time.
8. **Ignore rules miss the actual raw location.** `.gitignore` excludes `data/raw/`, `data/interim/`, and `data/processed/`, but not `data/loan.csv`, the dictionary, or row-level files under outputs. Review before publishing; no claim is made about remote tracking status.

## Audit conclusion

The audit identified path, target, preprocessing and evaluation issues to resolve before a reliable baseline or tuning. Duplicate checks, borrower overlap, feature availability and evaluation design were still open. This inspection left source, data and outputs unchanged.

# Data split, leakage checks and evaluation policy

Established 2026-09-19, before model fitting. The split checks and preprocessing schema are recorded in `outputs/validation/data_validation.json`.

## Population and split decision

Use a **retrospective, loan-level benchmark of resolved 36-month loans**. Label Fully Paid = 0 and Charged Off = 1; exclude Current, Default, grace/late, and credit-policy exception statuses. Default is excluded rather than assumed to be a final charged-off outcome. No class balancing or downsampling. Seed 42 for subsequent stochastic model operations.

| Partition | Origination period (`issue_d`) | Rows | Positives | Positive rate |
| --- | --- | ---: | ---: | ---: |
| Train | June 2007–December 2012 | 72,566 | 9,130 | 12.58165% |
| Validation | January–December 2013 | 100,422 | 12,378 | 12.32598% |
| Final holdout | January–December 2014 | 162,570 | Not profiled | Not profiled |

The code uses half-open year boundaries: train `[2007-01-01, 2013-01-01)`, validation `[2013-01-01, 2014-01-01)`, holdout `[2014-01-01, 2015-01-01)`. It excludes 1,925,110 raw rows outside this population. This narrower scope is intentional and must be visible in future results; it is not an all-portfolio model.

Why these cohorts: the metadata-only preflight found all 100,422 36-month loans issued in 2013 and all 162,570 issued in 2014 had Fully Paid/Charged Off outcomes. In contrast, 2015's 36-month cohort still has 320 other-status rows and 2016 has 107,949. A 2014 36-month loan's scheduled term ends by December 2017, before the December 2018 originations present in this extract. This is evidence of observation opportunity, **not an asserted extraction date or a guarantee of a fixed outcome horizon**. Older training cohorts exclude credit-policy exception statuses. No unresolved outcome is relabeled as a success.

A random split would mix vintages with different observation opportunities and credit-variable coverage. The chronological split measures transfer to later origination cohorts. Dates determine membership before preprocessing and are then excluded from predictors. The old class-balanced chunk sample is no longer generated or consumed.

## Integrity and leakage findings

- Scanned all 2,260,668 raw records in bounded 50,000-row chunks, reading all columns only for duplicate integrity. There were **zero repeated full-row fingerprints**. No deduplication was necessary. Fingerprints use pandas row hashing with fixed string parsing; absence of repeats rules out identical parsed rows, but does not establish entity identity.
- Among the 335,558 eligible rows, there were **zero repeated application-feature fingerprints**, including across partitions. These fingerprints omit labels, issue date, identifiers, repayment/settlement/hardship fields, and the additional exclusions below. There were also zero overlapping final 72-feature fingerprints between training and validation. No holdout model-feature profiling was performed.
- `id` and `member_id` have **zero nonmissing values in the entire source**. Raw CSV record positions identify rows reproducibly but are not loan/customer identifiers and never enter the model. Repeat customers or slightly differing records for the same loan cannot be ruled out. Claims about unseen-customer generalization are unsupported.
- The previous `check_leakage.py` fitted on every row and ranked gain; it did not validate leakage. It now performs integrity and preprocessing checks without importing or training LightGBM.
- The source SHA-256 is `23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c`. The contract records this fingerprint, source schema, boundaries, counts, and training-fitted category vocabularies. The modeling loader rejects a changed source or policy; the auditor refuses to silently replace a changed source, policy, or preprocessing contract.

### Prediction-time availability

Prediction time is **after loan terms and underwriting grade/rate are assigned, before repayment performance is observed**. Retaining `int_rate`, `grade`, `sub_grade`, and `installment` is appropriate to that stated decision point; these must not be advertised as inputs to an earlier, pre-underwriting decision.

The local LoanStats dictionary explicitly describes payment plans, hardship deferrals, interest accrued for hardship, payments received to date, latest credit pulls, and settlement outcomes. Screening now explicitly excludes `pymnt_plan`, `deferral_term`, `payment_plan_start_date`, and `orig_projected_additional_accrued_interest`, in addition to all prior repayment/settlement/latest-FICO exclusions and all names containing `hardship`. Their removal no longer depends on accidentally exceeding a missingness threshold.

`funded_amnt` and `disbursement_method` are also conservatively excluded: the dictionary describes committed funding and the method by which the borrower receives funds, without proving these are available at the stated scoring boundary. This is a conservative availability decision, not evidence that either field caused a measured score increase. IDs, free text, target/status, and issue date are excluded from predictors.

Remaining predictors are loan terms, application characteristics, and credit-history/bureau variables. `earliest_cr_line` is a borrower's earliest credit-line date, not a future payment date; it remains categorical to avoid an unnecessary feature redesign. The dictionary is field documentation, not timestamped provenance. Bureau fields described as current balances/delinquencies are assumed to be the application credit snapshot; the local files do not independently prove every value's historical availability. This assumption and missing customer identity remain material limitations. Dominant zip/date features in old plots alone are not evidence of leakage.

## Preprocessing decisions and fixes

- Corrected the active raw-data path to `data/loan.csv`. No missing intermediate CSVs are required: the development loader selects eligible train/validation records from raw chunks and never returns holdout rows.
- Fit the existing strictly-greater-than-80%-missing removal rule on training only. This retains 66 source predictors; the six existing engineered features yield **72 model inputs**. The complete retained/dropped lists are saved in the contract. Validation and holdout reuse this feature list.
- Explicitly parse categorical columns as strings and numeric columns as numbers; unexpected nonnumeric tokens raise an error. This avoids relying on pandas object-dtype inference, which differs in pandas 3. Training alone defines category vocabularies. Validation/holdout must reuse their names, order, dtypes, and vocabularies; unseen categories become missing.
- Actual validation checks found 328 unseen zip-code values and 525 unseen earliest-credit-line values (counts of affected rows); they map to missing under the frozen vocabulary. All other retained categorical fields had zero unseen values. Training and validation dtypes match.
- Leave numeric missing values as NaN for LightGBM; no median or other statistics are fitted on pooled data. Nonfinite numeric values and negative values in the five log/ratio source columns become missing. Income ratios are missing when income is zero or invalid. Existing `+1` denominator formulas otherwise remain unchanged.
- Fixed employment digit extraction and removed the filesystem-dependent header read inside feature engineering. `< 1 year` maps to 0, `10+ years` to 10, and missing/unparseable employment remains missing rather than conflated with zero experience. Actual training values cover 0–10 with 2,538 missing. Both development matrices contain zero numeric infinities.
- The old pooled preprocessing command-line entry points now stop with instructions to use the frozen training-only contract. The LightGBM entry point uses only training and validation, uses unweighted natural class prevalence (`class_weight=None`), and monitors explicit validation AUC for 50-round early stopping. The existing 500-estimator limit, learning rate 0.05, depth 7, seed 42, and parallelism remain unchanged. These are evaluation corrections, not a tuning search.

## Evaluation methodology

1. **Stage 3 baseline:** use the fixed training rows and preprocessing schema; use 2013 validation exclusively for early stopping. Report training/validation counts and prevalence, exact dependency versions and full model parameters, best iteration, feature schema, runtime, and validation metrics. Holdout reporting is limited to its already-frozen row count. Scoring is reserved for the final comparison.
2. **Primary metric:** ROC-AUC. **Secondary metrics:** average precision (explicitly this PR summary, not trapezoidal PR-AUC), log loss, and Brier score. These add imbalance-sensitive ranking and probability-quality checks. They use the cohort's natural prevalence and unweighted observations. No accuracy/KS/F1 duplication is required for model selection.
3. **Stage 4:** train candidates on the same training rows with the same preprocessing and seed; choose parameters/iteration counts from validation only. Validation performance after selection is optimistic and is not a final generalization estimate. Dates, cohort eligibility, features and class weights stay fixed during the search.
4. **Stage 5:** freeze both candidate configurations and iteration counts before opening holdout for modeling. Train both on the same original training cohort; do not silently refit on train+validation. Score both once on the same 2014 holdout. Report the approved metrics, their absolute differences, and a reliability table/plot. If a confusion matrix/precision/recall is useful, use a predeclared 0.5 descriptive threshold; no threshold optimization on holdout and no claim that 0.5 is a business-optimal decision. Any cost-based threshold requires explicit costs and validation-only selection beforehand. Holdout findings must not drive a new tuning cycle against this same test set.
5. **Later SHAP:** explain the saved selected model; the old script's independent full-data training is not part of this workflow and must not be run as-is. Updating SHAP remains Stage 6.

### Limits on interpretation

The extract does not contain a reliable historical label-availability timestamp or documented observation cutoff. Eventual outcomes of training loans may have become known after the next cohort originated. Therefore this is a **retrospective chronological cohort comparison, not a simulated historical deployment or a fixed-horizon probability-of-default model**. A true point-in-time backtest requires additional provenance/outcome timing and potentially a maturity gap. Results also do not establish current-portfolio calibration, performance on 60-month/active loans, or independence of repeat borrowers.

The historical README score of 0.6885 is not directly comparable: cohort, target treatment, sampling prevalence, preprocessing, class weighting, and split semantics have changed. No historical-score reproduction is claimed.

## Checks and implementation

- Ran the rewritten integrity auditor on the actual raw file; all implemented integrity gates passed and saved the contract.
- Ran five regression tests covering date boundaries, excluded outcomes, missing dates, holdout isolation, changed-source rejection, train-only feature/category fitting, post-outcome exclusions, employment parsing, and invalid numerical inputs.
- The LightGBM input loader returned 72,566 × 72 training and 100,422 × 72 validation, with matching dtypes, disjoint source rows and matching labels. The five modified source files also passed syntax checks.
- Runtime for validation: pandas 3.0.1, NumPy 2.3.5 in the bundled Python. LightGBM, scikit-learn, and SHAP remain unavailable there. Stage 3 must establish and record a compatible modeling environment before fitting; the modified estimator call has not been executed. Training compatibility was still untested.
- Modified: `src/stratified_sampling.py`, `src/check_leakage.py`, `src/feature_screening.py`, `src/feature_engineering.py`, `src/lgbm_model.py`, and `docs/PROJECT_STATE.md`. Created this document, `outputs/validation/data_validation.json`, and `tests/test_stage2_validation.py`. Raw data and all pre-existing outputs are unchanged.

The split and preprocessing policy were ready for a baseline once modeling dependencies were installed. Holdout access at this point was limited to eligibility and integrity checks; fitting, selection and scoring used no holdout data.

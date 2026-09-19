# Power BI reporting data

Import the seven UTF-8 CSVs in `outputs/powerbi/`, not the raw loan file. Both the full scored cohort and compact summaries are provided: loan rows support business slicing; summaries provide reconciled reference totals. These are frozen historical holdout results for resolved 36-month loans issued in 2014, not a current portfolio or deployment evaluation. Dashboard implementation is covered by the separate design specification.

## Tables and recommended model

| Table | Grain / key | Purpose / relationships |
|---|---|---|
| scored_loans | One loan record; unique source_row | Main reporting fact, final tuned model only. Business dimensions remain on this table. |
| risk_bands | One band; unique risk_band_id | Dimension; single-direction 1:* to scored_loans on risk_band_id. Sort risk_band by risk_band_id. |
| risk_band_summary | One band; unique risk_band_id | Static cohort reconciliation/reference table; keep disconnected and preferably hidden from report authors. Do not join it to loan rows or expect it to change with business slicers. |
| model_performance | One model; unique model | Baseline versus tuned holdout comparison; single-direction 1:* to calibration on model. Keep separate from the loan fact. |
| calibration | One model and probability bin; composite key model, bin_lower | Fixed-bin calibration from Stage 5. Empty bins retained. |
| feature_importance | One input feature; unique feature | Final model's total training split gain, all 72 inputs. Standalone global table. |
| shap_importance | One input feature; unique feature | Stage 6 global importance from the frozen 10,000-row random holdout sample. Standalone global table. |

Disable automatic relationship detection. No relationships connect feature-level tables or cohort-level metrics to individual loans. Global importance and model metrics do not respond to loan segment filters; label them accordingly. No many-to-many relationships are needed. Use issue_month as a Date with monthly granularity; day-level origination dates are unavailable. Full loan rows allow purpose, state, grade, sub-grade, home ownership and month segmentation without additional aggregate tables. Filtered default rates must use default counts divided by loan counts, not averages of group percentages. See [dashboard design](powerbi_dashboard_design.md) for layouts and DAX measures.

## Risk bands

Reporting boundaries are fixed at 10%, 15%, and 20%, with no outcome optimization or quantile fitting. They describe modeled probabilities, not validated approval rules or guaranteed loss rates. Lower bounds are inclusive, upper bounds exclusive, except that 100% belongs to the final band.

| ID | Label | Probability interval |
|---|---|---|
| 1 | Below 10% | [0, 0.10) |
| 2 | 10% to below 15% | [0.10, 0.15) |
| 3 | 15% to below 20% | [0.15, 0.20) |
| 4 | 20% and above | [0.20, 1.00] |

The frozen predicted_class threshold remains probability >= 0.5. Even band 4 loans can have predicted_class=0. All observed holdout scores are below 0.5; both models have zero predicted positives, zero recall and undefined precision. Calibration is imperfect. Do not interpret band names as operational credit decisions.

## Data dictionary

Power BI types: Whole number for IDs, counts, ranks and binary flags; Decimal number for probabilities, rates, amounts and importance; Text for categories; Date for issue_month. Probabilities and rates use fractions [0,1], except interest_rate_pct, which retains source percentage points (e.g. 12 means 12%). Do not multiply stored probabilities by 100 before percentage formatting. Amounts retain source monetary units; these are not loss estimates.

| Table / fields | Meaning / null handling |
|---|---|
| scored_loans.source_row | Zero-based CSV data-record position, excluding header; joins only to the exact raw file hash in manifest. Not a loan ID or borrower ID: original identifiers are unavailable. Non-null and unique. |
| scored_loans.actual_default | 1=Charged Off, 0=Fully Paid. Other statuses excluded upstream. |
| scored_loans.predicted_probability, predicted_class | Saved final tuned default probability; binary threshold decision at 0.5. Non-null. |
| scored_loans.risk_band_id, risk_band | Band key 1–4 and descriptive label; non-null. |
| scored_loans.issue_month | Original issue_d parsed to ISO first day of month; first day is a month key, not a known issue day. |
| scored_loans.term_months | Contract term; 36 throughout this cohort. |
| scored_loans.loan_amount, annual_income, interest_rate_pct | Original loan_amnt, annual_inc, int_rate. Numeric missing values remain blank; zero remains zero. |
| scored_loans.grade, sub_grade, purpose, home_ownership, state | Original categories (state from addr_state); whitespace trimmed, missing categories labeled Unknown. |
| risk_bands.lower_probability, upper_probability, upper_inclusive | Fractional band boundaries; upper_inclusive is 1 only for the last band. All lower bounds inclusive. |
| risk_band_summary.loan_count, default_count | Loan records and actual defaults per band, including zero for empty bands. |
| risk_band_summary.default_rate, avg_predicted_probability | Observed default fraction and mean modeled probability per band; blank for empty bands. Band key and label match risk_bands. |
| model_performance.model, is_final, split | baseline or tuned; is_final=1 for tuned; split=holdout_2014. |
| model_performance.loan_count, default_count | Full cohort totals repeated for each model; never sum across models. |
| model_performance.roc_auc, average_precision, log_loss, brier_score | Frozen Stage 5 discrimination/probability metrics. AUC/AP higher is better; loss/Brier lower is better. |
| model_performance.threshold, threshold_rule, precision, recall | Frozen 0.5 rule and classification metrics; precision blank when denominator is zero, not zero performance. |
| model_performance.predicted_positive_rows, tn, fp, fn, tp | Predicted positives and confusion-matrix counts. |
| model_performance.mean_probability, observed_default_rate, mean_prediction_minus_default_rate | Cohort mean score, actual fraction and signed difference; difference may be negative. |
| calibration.model, bin_lower, bin_upper | Model identifier and fixed 0.1-wide probability bins; lower inclusive, upper exclusive except 1.0. |
| calibration.rows, defaults, mean_probability, observed_default_rate | Bin counts, defaults, mean score and observed fraction; means/rates blank in empty bins, counts zero. |
| feature_importance.feature, importance, importance_rank | Input feature name, total LightGBM split gain, rank 1=highest; alphabetical tie break. Not a signed effect or causal measure. |
| shap_importance.feature, mean_abs_shap, importance_rank, importance_share | Frozen Stage 6 feature, mean absolute log-odds contribution, saved rank, fraction of total absolute importance. Shares sum to one; noncausal. |

`manifest.json` is provenance/verification metadata, not a reporting table. It records input/output hashes, row counts, sizes, missing-value counts, sample size and verification status. Numeric blanks above should import as null; never replace undefined precision with zero. Avoid inference of borrower identity, current exposure, losses, protected attributes or feature availability from these exports.

## Reproduction and verification

Export totals: 162,570 loan rows, 22,315 defaults (13.72639%); seven CSVs totaling 19,258,145 bytes (19.26 MB). Observed default rates across ascending reporting bands are 6.09%, 12.55%, 19.90%, and 25.90%; these descriptive holdout results were not used to optimize boundaries. All 13 repository tests passed. Frozen model, prediction, calibration and SHAP inputs remained unchanged.

Run `venv\Scripts\python.exe -B src/prepare_powerbi_data.py` from the repository root for a new export. The script refuses to overwrite an existing output directory. It verifies the raw SHA-256 before joining, reads only 11 raw columns in 100,000-row chunks, and checks one-to-one coverage, actual outcomes, issue year and term against frozen predictions. It loads the final model solely for split-gain importance; it does not fit, predict, tune or recompute SHAP.

Every CSV is reread and compared with its Python table. Exported risk-band totals and means are rebuilt from exported loan rows. Counts/defaults and mean probability match Stage 5. Frozen input hashes are checked again after export. Boundary/invalid-value and empty-band tests protect reporting semantics. Historical cohort selection, unknown borrower overlap, unverified bureau timing and geographic/date proxies remain limitations carried forward from prior stages.

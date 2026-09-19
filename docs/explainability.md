# SHAP analysis

Analysis date: 2026-09-19. SHAP uses the saved **57-tree final model** and its 72-feature preprocessing schema, without refitting.

## Sampling and method

- Simple random sample of **10,000** of the 162,570 final holdout loans, without replacement, seed 42. The sample covers every month of 2014 and contains 1,391 defaults (**13.91%**, versus 13.72639% in the full cohort). It represents this historical resolved-36-month population, not all raw loans.
- Added four deliberately selected local examples, giving **10,004 explained rows**. These four extra rows do not enter global importance or effect plots. Their selection is described below.
- Used `shap.TreeExplainer` with `feature_perturbation='tree_path_dependent'` and `model_output='raw'`. The reference uses the saved trees' training path counts; no new background dataset or refitting was introduced.
- Contributions are in **default log-odds**, not probability percentage points. Positive values raise the model's default score relative to its reference. The reference is -1.99645209 log-odds. Applying the logistic function to reference plus all contributions reconstructs the prediction.
- Raw data was read in chunks, retaining only the selected rows and required columns. Sampling IDs, full SHAP arrays and feature names are saved for reproducibility; source-row positions are not customer/loan identifiers.

## Main drivers

Global importance is mean absolute SHAP on the random sample. Shares below divide each feature's mean absolute contribution by the sum over all features; they are not shares of defaults or causal effects.

| Rank | Feature | Mean absolute SHAP | Importance share |
| --- | --- | ---: | ---: |
| 1 | `int_rate` | 0.220221 | 35.24% |
| 2 | `annual_inc` | 0.091705 | 14.68% |
| 3 | `sub_grade` | 0.074487 | 11.92% |
| 4 | `zip_code` | 0.064130 | 10.26% |
| 5 | `acc_open_past_24mths` | 0.040744 | 6.52% |
| 6 | `earliest_cr_line` | 0.038602 | 6.18% |
| 7 | `log_annual_inc` | 0.026780 | 4.29% |
| 8 | `purpose` | 0.022386 | 3.58% |
| 9 | `loan_to_inc_ratio` | 0.021638 | 3.46% |
| 10 | `inq_last_6mths` | 0.014230 | 2.28% |

All 72 features appear in `shap_importance.csv`; 21 have nonzero sample attribution. `dti` ranks 17 with only approximately 0.048% of total mean absolute SHAP. The older README's driver narrative should not be treated as the explanation of this final model. Training gain importance and sample SHAP importance measure different things and need not rank features identically.

## Direction of effects

The dependence plots and numeric summaries support the following **model associations**, not interventions or causal claims:

| Feature | Lower group | Higher group | Higher-minus-lower mean SHAP |
| --- | --- | --- | ---: |
| Interest rate | <= 8.67 | >= 15.59 | +0.70177 |
| Annual income | <= 40,000 | >= 95,000 | -0.25594 |
| Accounts opened in past 24 months | <= 2 | >= 6 | +0.10417 |
| Loan-to-income ratio | <= 0.10000 | >= 0.29091 | +0.06492 |

Groups use sample 20th/80th percentile cutoffs, including ties; differences are log-odds contributions. Higher interest rates, more recent account openings, and higher loan-to-income ratios generally raise modeled risk; higher income generally lowers it. The curves can plateau or be nonmonotonic locally. Income's dependence plot uses a nonnegative symlog axis to retain its long tail without removing observations.

The categorical table shows negative mean contributions for the displayed A/B sub-grades and positive ones for the displayed C sub-grades. The sub-grade plot shows the 12 most frequent categories, sorted by mean SHAP rather than label. Categorical features are grey in the beeswarm; numerical color gradients must not be interpreted as an ordering of ZIP codes or date strings. Correlated pairs such as rate/sub-grade and income/log-income share attribution.

## Local explanations

Selected the loan nearest the full cohort's score P10 and P90 **within each actual outcome**, breaking ties by source-row position. This produces typical lower/higher-score comparisons rather than cherry-picked extremes. Higher/lower describe relative score only; the classification threshold remains 0.5.

| Example | Source row | Actual default | Predicted probability | Classification at 0.5 |
| --- | ---: | ---: | ---: | --- |
| Lower-score non-default | 2,091,289 | 0 | 6.93768% | True negative |
| Higher-score non-default | 1,959,698 | 0 | 19.40505% | True negative |
| Lower-score default | 1,919,938 | 1 | 6.93776% | False negative |
| Higher-score default | 1,944,706 | 1 | 19.40504% | False negative |

- The lower-score non-default's rate of 7.62, annual income of 73,900 and A3 sub-grade reduce its score; its higher loan-to-income ratio pushes the other way.
- The higher-score non-default's rate of 16.29, income of 22,000 and D2 sub-grade push risk upward, although this loan ultimately repaid.
- The lower-score default has strong downward contributions from its rate of 7.69, A4 sub-grade and credit-line category Aug-1993, despite upward contributions from seven recent account openings and low available bankcard credit. A favorable modeled profile does not guarantee repayment.
- The higher-score default has upward contributions from its rate of 18.24, D5 sub-grade and 11 recent account openings. Its score is still below 0.5, so it remains a false negative.

There are **no true-positive or false-positive examples at 0.5 anywhere in the saved final predictions**, because that threshold flags no loans. The threshold remains 0.5. Waterfalls explain why the model assigned a probability, not why the realized loan outcome occurred.

## Suspicious behavior and limits

ZIP code and earliest credit-line category remain substantial geographic/date proxies. ZIP code is the largest absolute contributor in **760 of the 10,000** sampled rows, with contributions ranging approximately -0.406 to +0.604 log-odds. Earliest credit-line category is largest in 134 sampled rows. These effects warrant caution about geography, credit vintage, category sparsity and transport to another population; the categorical effect table exposes their category counts and mean contributions.

No known excluded target, repayment, settlement or hardship field appears in the model schema. The exclusion list was checked against the schema, but this cannot rule out all leakage. SHAP cannot verify historical field availability, establish causation, or resolve missing customer identity. The Stage 5 threshold/calibration limitations and retrospective cohort scope remain unchanged. The feature set and model were kept fixed.

## Outputs and checks

`outputs/explainability/` contains:

- `shap_importance.csv` with `feature`, `mean_abs_shap`, `importance_rank` and share; plus the global bar chart and beeswarm.
- Three numeric dependence plots (interest rate, income, recent account openings), one sub-grade category plot, and `numeric_effects.csv` / `categorical_effects.csv`.
- Four local waterfall plots, `local_examples.csv` and the complete `local_contributions.csv`.
- Compressed `shap_values.npz` and `explanation_metadata.json` containing sampling, model/source hashes, units, versions and verification results.

All ten plots were visually inspected; the income axis and category-order caption were corrected using saved attributions, without recomputing SHAP or changing the sample. Numerical checks found maximum additive log-odds error **4.0e-15**, probability reconstruction error **4.2e-16**, and exact agreement with LightGBM's native contributions and saved prediction probabilities. Sample selection, global means/ranks, local labels/sums and file hashes were independently checked. **All 11 regression tests passed.** Final model/evaluation artifacts and the preprocessing contract remain unchanged.

The implementation is in `src/shap_explainability.py`, with local-case coverage in `tests/test_shap_examples.py` and dependencies in `requirements.txt`. SHAP 0.52.0, Matplotlib 3.11.2 and supporting packages are pinned; existing modeling dependency versions were preserved. Initial extraction, SHAP and plotting took approximately 30.47 seconds, excluding installation/import startup and subsequent visual/export checks. The existing historical SHAP PNG was not overwritten.

The entry point is `venv\Scripts\python.exe -B src/shap_explainability.py`; it refuses to overwrite an existing explanation folder. Reporting exports and the dashboard design are documented separately.

# Power BI dashboard specification

Design date: 2026-09-19. This was the Stage 8 specification fallback. The report has since been implemented in [powerbi/LoanRisk.pbip](../powerbi/LoanRisk.pbip). See the [implementation notes](../powerbi/README.md) for Desktop checks, setup, and differences from the proposed layout below.

## Data and import contract

Use the seven Stage 7 CSVs in `outputs/powerbi/`; all seven hashes match `manifest.json`. Keep the table names equal to the filenames without `.csv`. Import mode; UTF-8; comma delimiter; first row as headers; decimal parsing with English (United States) locale. Parameterize the reporting folder so the future report is portable. The 1.19 GB raw loan CSV and model scoring are outside the report refresh.

Follow [powerbi_data.md](powerbi_data.md) for all column types and definitions. Explicitly set `model_performance.precision` to Decimal number even though both rows are blank. Preserve nulls. Set `issue_month` to Date, format `MMM yyyy`, use the column itself (not the automatic date hierarchy), and disable automatic date/time tables for this report. Set numeric IDs and ranks to Do not summarize. Format `interest_rate_pct` as `0.00`, not a percentage: source value 12 already means 12%. The supplied tables contain no income bins, employment fields, borrower IDs or loss estimates.

Active, single-direction relationships only:

| One side | Many side | Filter direction |
|---|---|---|
| `risk_bands[risk_band_id]` | `scored_loans[risk_band_id]` | risk_bands to scored_loans |
| `model_performance[model]` | `calibration[model]` | model_performance to calibration |

Keep `feature_importance` and `shap_importance` disconnected. Load `risk_band_summary` as a hidden, disconnected QA reference, never as the source for slicer-responsive visuals. Turn off automatic relationship creation. Hide `scored_loans[source_row]`, duplicate fact band label/key, and boundary/helper fields from the report field list after setup. Use `risk_bands[risk_band]` in every band visual/slicer; sort it by `risk_bands[risk_band_id]`.

No relationships join portfolio slices to frozen performance or global explanations. Both must remain explicitly labeled as full-cohort/sample results. No calculated columns, custom visuals, additional reporting datasets, or new modeling runs are required.

## Layout and visual language

Four pages, each 1280 × 900 pixels, 24 px outside margin, 16 px gutters. Coordinates below are x/y/width/height in pixels and reserve space for titles inside each panel. Use Segoe UI, 26 px page title, 16 px visual titles, 12–14 px axes/body, 28 px KPI values. Background `#F4F6FA`, white panels, primary text `#172B4D`. Actual default rate `#B54708`, predicted probability/final model `#007F7A`, baseline `#667085`. Use labels and line markers as well as color; no 3D charts or decorative gauges. Rate axes start at zero; do not truncate bar axes. Rates show two percentage decimals; AUC/AP/loss/Brier show four decimals; counts use separators and no display-unit abbreviation.

Shared header (24/16/1232/68): page title plus subtitle “2014 resolved 36-month loans · Historical holdout”. Page navigator at 24/848/1232/32. Footer note above navigator: “Predictive associations; not causal or an approval policy.” On pages 3–4 additionally state “Portfolio filters do not apply.” Configure descriptive alt text, logical tab order, readable legends and keyboard-accessible navigation during implementation.

## Page 1 — Executive Risk Overview

Slicer strip at y=96, height=52: issue_month (24/96/240/52; dropdown), purpose (280/96/280/52), grade (576/96/200/52), risk_band (792/96/300/52). Reset button at 1108/96/148/52. Each is multi-select with All as initial state; no default exclusions.

Six cards at y=168, height=88, width=192, x=24,232,440,648,856,1064, in order: `[Total Loans]`, `[Actual Defaults]`, `[Default Rate]`, `[Average Predicted Probability]`, `[Loans at 20% or Above]`, `[Final ROC-AUC]`. Visible titles match those names; final AUC subtitle “Full holdout · fixed”. The 20% card subtitle is “Reporting band, not 0.5 classification”; share is a tooltip `[Share at 20% or Above]`.

| Position | Visual / title | Exact fields and settings |
|---|---|---|
| 24/276/600/240 | Clustered columns / Loan distribution by modeled probability | X: `risk_bands[risk_band]`; Y: `[Total Loans]`; band order 1–4. Tooltips: `[Actual Defaults]`, `[Default Rate]`, `[Average Predicted Probability]`. Show all bands. |
| 640/276/616/240 | Clustered columns / Observed versus predicted risk by band | X: same band dimension; Y: `[Default Rate]`, `[Average Predicted Probability]`; tooltips `[Total Loans]`, `[Actual Defaults]`. Shared percentage axis. |
| 24/536/1232/252 | Line chart / Risk by origination month | X: `scored_loans[issue_month]`, ascending continuous Date; Y: `[Default Rate]`, `[Average Predicted Probability]`; tooltips `[Total Loans]`, `[Actual Defaults]`. This is outcome by issue month, not date of default or monitoring drift. |

At 24/800/1232/32 display “Classification threshold 0.5 flags no loans in this cohort. Reporting bands describe scores only.”

## Page 2 — Portfolio Risk Analysis

Same four synced slicers and reset layout as page 1. Add unsynced page-local dropdowns for `scored_loans[state]` (24/168/296/52) and `scored_loans[home_ownership]` (336/168/296/52), both initially All. To their right show `[Total Loans]` (648/168/296/52) and `[Default Rate]` (960/168/296/52) as compact cards.

| Position | Visual / title | Exact fields and settings |
|---|---|---|
| 24/240/600/282 | Clustered horizontal bars / Risk by loan purpose | Y: `scored_loans[purpose]`; X: `[Default Rate]`, `[Average Predicted Probability]`; sort `[Total Loans]` descending. Retain all 13 observed purposes with scroll; no hidden Top N. Tooltips counts and `[Average Loan Amount]`, `[Average Annual Income]`. |
| 640/240/616/282 | Clustered columns / Risk by credit grade | X: `scored_loans[grade]`, alphabetical A–G; Y: same two rates; tooltips counts and mean loan amount. Show small-group counts, especially grade G (179 loans before filters). |
| 24/542/1232/246 | Table / State-level portfolio | `scored_loans[state]`, `[Total Loans]`, `[Actual Defaults]`, `[Default Rate]`, `[Average Predicted Probability]`, `[Average Loan Amount]`, `[Average Annual Income]`. All 49 observed states retained, scroll enabled; sort count descending. Totals use measures in total context. Amount columns use source monetary units with separators, no inferred exposure or loss labels. |

At 24/800/1232/32: “Small segments can have unstable rates; compare counts alongside percentages. Income and loan size shown are averages.” Use state text/table, not geocoding or a map. Regions and income bands are not defined in the reporting data.

## Page 3 — Model Performance

No portfolio slicers, synced filters, model selector, or threshold slider. Every visual is full holdout. At 24/96/1232/60: “Tuned model improved ranking; threshold 0.5 still detects zero defaults. Average precision is reported, not trapezoidal PR-AUC.”

| Position | Visual / title | Exact fields and settings |
|---|---|---|
| 24/176/600/222 | Table / Baseline versus final | Row field `model_performance[model]`; measures `[Model ROC-AUC]`, `[Model Average Precision]`, `[Model Log Loss]`, `[Model Brier Score]`. Ascending model sort; no total row. Subtitle: “tuned = selected final; AUC/AP higher is better, loss/Brier lower is better”. |
| 640/176/296/90 | Card / Final ROC-AUC | `[Final ROC-AUC]`. |
| 952/176/304/90 | Card / ROC-AUC improvement | `[ROC-AUC Gain]`, signed decimal `+0.0000;-0.0000;0.0000`, not percent relative gain. |
| 640/286/616/112 | Text panel / Threshold 0.5 | Three compact cards `[Final Threshold]`, `[Final Recall]`, `[Final Precision Label]`. Caption “No predicted positives; precision undefined.” |
| 24/418/392/370 | Four-card 2 × 2 confusion matrix / Final model at 0.5 | Static row headers actual 0 / actual 1, column headers predicted 0 / predicted 1. Top-left `[Final TN]`, top-right `[Final FP]`, bottom-left `[Final FN]`, bottom-right `[Final TP]`. Use equal cells and raw counts; no accuracy headline. |
| 432/418/404/370 | Line chart / Baseline calibration | Visual filter `model_performance[model] = baseline`, `calibration[rows] > 0`; X `calibration[bin_lower]` sorted numeric; Y `[Calibration Mean Score]`, `[Calibration Default Rate]`; tooltip `[Calibration Loans]`, bin_upper. |
| 852/418/404/370 | Line chart / Final calibration | Same, model=tuned. Match baseline axes: X 0–0.5; Y 0–1. Subtitle “Bin lower boundary; nonempty bins only”. Show point markers and counts in tooltip. |

Calibration lines compare mean predictions and outcomes within identical fixed 0.1-wide bins; X is the bin lower edge, not mean score. Empty bins stay absent, never zero-rate points. At 24/800/1232/32: “Final 30–<40% bin contains only 15 loans; calibration estimates there are unstable.” ROC/PR curve tables have not been prepared, so this page uses numeric discrimination metrics.

## Page 4 — Explainability

No portfolio slicers or feature cross-filtering between tables. At 24/96/1232/60: “Final model · SHAP: random 10,000-row holdout sample, seed 42 · Contributions in default log-odds · Noncausal”.

| Position | Visual / title | Exact fields and settings |
|---|---|---|
| 24/176/600/348 | Horizontal bars / Top 10 SHAP drivers | Y `shap_importance[feature]`; X `[Mean Absolute SHAP]`; visual filter `importance_rank <= 10`; sort measure descending. Tooltip `[SHAP Importance Share]`. |
| 640/176/616/348 | Horizontal bars / Top 10 training-gain drivers | Y `feature_importance[feature]`; X `[Training Gain]`; visual filter `importance_rank <= 10`; sort measure descending. Gain is not a probability or signed impact. |
| 24/544/760/244 | Static text / Direction of modeled associations | Use the four statements below, with their lower/higher groups and signed SHAP differences. This panel is fixed explanatory text, not a live segment calculation. |
| 800/544/456/244 | Static text / Interpretation and limits | Interest rate is the largest SHAP driver. ZIP and earliest credit-line date are substantial proxies. Rate/sub-grade and income/log-income share attribution. Gain and SHAP rankings answer different questions. |

Direction statements from `outputs/explainability/numeric_effects.csv` (20th/80th percentile sample groups including ties):

- Interest rate: <=8.67 versus >=15.59; higher group mean SHAP is +0.70177 log-odds higher.
- Annual income: <=40,000 versus >=95,000; higher group mean SHAP is -0.25594 lower.
- Accounts opened in past 24 months: <=2 versus >=6; higher group mean SHAP is +0.10417 higher.
- Loan-to-income ratio: <=0.10000 versus >=0.29091; higher group mean SHAP is +0.06492 higher.

Caption at 24/800/1232/32: “Associations can be nonlinear; changing a feature does not establish a causal change in default risk.” Magnitude bars alone cannot establish direction. Existing Stage 6 report/plots remain the source of detailed local explanations; no new SHAP computation is needed.

## Filters, interactions and reset behavior

Sync only month, purpose, grade and risk_band slicers between pages 1 and 2. Do not sync them to pages 3–4, even as hidden slicers. State and home-ownership filters apply only to page 2. No report-level population filters are necessary because the export already fixes the cohort. Do not add filters on actual_default or predicted_class that would silently alter the comparison population.

On pages 1–2, slicers filter all portfolio visuals; set interactions to None for the fixed AUC card. For predictable selection behavior, disable chart/table-to-chart/table filtering and highlighting on all four pages: slicing is controlled exclusively by explicit slicers. On page 3 disable comparison-table clicks filtering calibration. All four bands remain visible with zero counts, while rates remain blank when the current selection has no loans.

Create a data-only reset bookmark for each portfolio page using Selected visuals (slicers only): All in all four shared slicers, plus All in the two local slicers on page 2. Assign its reset button. Reset on either page resets the shared slicers; resetting page 1 does not clear the page 2 local state/home filters. Keep their visible selection text so the scope is clear. Navigation must not reset filters. A no-data selection should show 0 loan/default counts and blank rates, not a misleading 0% rate.

## DAX measures

Create each named definition below as a separate measure, not a calculated column or one pasted script. Store portfolio measures on scored_loans, model measures on model_performance, calibration measures on calibration, and importance measures on their corresponding tables. Display folders are optional. Avoid implicit sums of model metrics. Formulas intentionally preserve portfolio filter intersections and undefined values.

```dax
Total Loans = COALESCE(COUNTROWS('scored_loans'), 0)

Actual Defaults = COALESCE(SUM('scored_loans'[actual_default]), 0)

Default Rate = DIVIDE([Actual Defaults], [Total Loans])

Average Predicted Probability = AVERAGE('scored_loans'[predicted_probability])

Loans at 20% or Above =
    CALCULATE([Total Loans], KEEPFILTERS('risk_bands'[risk_band_id] = 4))

Share at 20% or Above = DIVIDE([Loans at 20% or Above], [Total Loans])

Average Loan Amount = AVERAGE('scored_loans'[loan_amount])

Average Annual Income = AVERAGE('scored_loans'[annual_income])

Model ROC-AUC = SELECTEDVALUE('model_performance'[roc_auc])

Model Average Precision = SELECTEDVALUE('model_performance'[average_precision])

Model Log Loss = SELECTEDVALUE('model_performance'[log_loss])

Model Brier Score = SELECTEDVALUE('model_performance'[brier_score])

Final ROC-AUC =
    CALCULATE([Model ROC-AUC], REMOVEFILTERS('model_performance'),
              'model_performance'[is_final] = 1)

Baseline ROC-AUC =
    CALCULATE([Model ROC-AUC], REMOVEFILTERS('model_performance'),
              'model_performance'[model] = "baseline")

ROC-AUC Gain = [Final ROC-AUC] - [Baseline ROC-AUC]

Final Threshold =
    CALCULATE(SELECTEDVALUE('model_performance'[threshold]),
              REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)

Final Recall =
    CALCULATE(SELECTEDVALUE('model_performance'[recall]),
              REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)

Final Precision Label =
    VAR PrecisionValue =
        CALCULATE(SELECTEDVALUE('model_performance'[precision]),
                  REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)
    RETURN IF(ISBLANK(PrecisionValue), "Undefined", FORMAT(PrecisionValue, "0.00%"))

Final TN =
    CALCULATE(SELECTEDVALUE('model_performance'[tn]),
              REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)

Final FP =
    CALCULATE(SELECTEDVALUE('model_performance'[fp]),
              REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)

Final FN =
    CALCULATE(SELECTEDVALUE('model_performance'[fn]),
              REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)

Final TP =
    CALCULATE(SELECTEDVALUE('model_performance'[tp]),
              REMOVEFILTERS('model_performance'), 'model_performance'[is_final] = 1)

Calibration Loans = SUM('calibration'[rows])

Calibration Mean Score =
    DIVIDE(SUMX('calibration', 'calibration'[rows] * 'calibration'[mean_probability]),
           [Calibration Loans])

Calibration Default Rate = DIVIDE(SUM('calibration'[defaults]), [Calibration Loans])

Mean Absolute SHAP = SELECTEDVALUE('shap_importance'[mean_abs_shap])

SHAP Importance Share = SELECTEDVALUE('shap_importance'[importance_share])

Training Gain = SELECTEDVALUE('feature_importance'[importance])
```

Formats: counts `#,0`; probabilities/default rates/recall/SHAP share `0.00%`; AUC/AP/log loss/Brier `0.0000`; threshold `0.0`; SHAP magnitude `0.0000`; gain `#,0.0`; monetary averages `#,0`. Undefined precision uses the text label above. SHAP shares are shares of total mean absolute attribution, not shares of defaults.

DAX semantics checked against Microsoft documentation: [KEEPFILTERS](https://learn.microsoft.com/en-us/dax/keepfilters-function-dax) intersects the existing band selection; [REMOVEFILTERS](https://learn.microsoft.com/en-us/dax/removefilters-function-dax) fixes model-selection context; [SELECTEDVALUE](https://learn.microsoft.com/en-us/dax/selectedvalue-function-dax) avoids aggregating multiple metric values; [DIVIDE](https://learn.microsoft.com/en-us/dax/divide-function-dax) leaves a zero-denominator result blank. DAX execution still requires Power BI; these references and Python checks do not constitute engine validation.

## Reference results and acceptance checks

Reference values from the reporting CSVs:

| Check | Expected result |
|---|---|
| Unfiltered overview | 162,570 loans; 22,315 defaults; default rate 13.72639478%; mean score 13.13029927% |
| Score >=20% reporting band | 12,615 loans; 7.75973427% of cohort; these are not positive classifications at 0.5 |
| Final AUC and gain | 0.65949575; gain +0.01666311 over baseline 0.64283263 |
| Average precision baseline / final | 0.20590956 / 0.21580617 |
| Log loss baseline / final | 0.38518541 / 0.38281766 |
| Brier baseline / final | 0.11506916 / 0.11452292 |
| Final confusion cells TN / FP / FN / TP | 140,255 / 0 / 22,315 / 0; recall 0%; precision Undefined |
| Grade A only | 35,333 loans; 1,902 defaults; rate 5.38306965%; mean score 7.49911399%; 3 loans >=20% |
| January only | 11,495 loans; 1,433 defaults; rate 12.46628969%; mean score 13.20893831%; 803 loans >=20% |
| Band 1 only | 47,587 loans; 2,898 defaults; rate 6.08989850%; mean score 7.74406985%; >=20% count 0 |
| Band counts 1 / 2 / 3 / 4 | 47,587 / 57,463 / 44,905 / 12,615 |
| Default counts 1 / 2 / 3 / 4 | 2,898 / 7,214 / 8,936 / 3,267 |
| SHAP top five in order | int_rate, annual_inc, sub_grade, zip_code, acc_open_past_24mths |
| Gain top five in order | int_rate, zip_code, earliest_cr_line, annual_inc, sub_grade |

All seven CSV hashes and row counts match the frozen Stage 7 manifest. Table/column references, relationships, direction statements and the reference arithmetic were checked against repository outputs. Monthly totals sum to the full cohort; no raw-data reread, retraining, model change or threshold change occurred.

Before accepting the Power BI implementation: import with explicit types; create relationships/measures; compile all DAX; compare every reference above; confirm multi-select intersections and empty-slice blanks; ensure pages 3–4 and fixed AUC do not respond to portfolio slicers; check reset/navigation behavior; exclude empty calibration bins; review page rendering at 100% and keyboard order. Confirm no clipped titles, misleading totals or summation of AUC. Save a PBIX/PBIP only after these checks.

Desktop implementation and core acceptance checks are complete; remaining portability/accessibility checks are listed in the [implementation notes](../powerbi/README.md).

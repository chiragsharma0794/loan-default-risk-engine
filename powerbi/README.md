# Loan risk report

Open `LoanRisk.pbip` in Power BI Desktop. The four pages cover the historical 2014 holdout: Overview, Portfolio Risk, Model Performance, and Risk Drivers. Each page uses a 1280 x 720 canvas and the same native page navigator.

## Open on another computer

1. Place all seven reporting CSVs in `outputs/powerbi/`. The loan-level `scored_loans.csv` is intentionally excluded from Git; obtain the existing frozen export or follow the [reporting-data instructions](../docs/powerbi_data.md).
2. Set the `SourceFolder` Power Query parameter to that folder's absolute path, using **Transform data > Edit parameters**. Its checked-in default points to the original local project. Alternatively, edit that expression in `LoanRisk.SemanticModel/definition/expressions.tmdl` before opening Desktop.
3. Refresh. A fresh clone has definitions but no imported data cache.

The project uses PBIP, enhanced PBIR report definitions, and a TMDL semantic model (Desktop converted the earlier `model.bim` to TMDL on the latest save; all 28 measures and both relationships were verified unchanged). Desktop may show a format-preview notice. Tested with Desktop 2.157.1354.0; older releases may not support the saved report schema.

## Model and interactions

Seven imports: `scored_loans` (162,570 rows), `risk_bands` (4), `risk_band_summary` (4), `model_performance` (2), `calibration` (20), `feature_importance` (72), and `shap_importance` (72).

Two single-direction relationships:

- `risk_bands[risk_band_id]` (1) to `scored_loans[risk_band_id]` (*).
- `model_performance[model]` (1) to `calibration[model]` (*).

The hidden risk-band summary stays disconnected. Global model metrics, SHAP, and gain have no loan-level relationship. The model contains the same [28 documented DAX measures](../docs/powerbi_dashboard_design.md#dax-measures); no calculated tables were added.

Four dropdowns on Portfolio Risk filter month, purpose, grade and risk band. Overview always shows the full cohort. Performance and SHAP remain global. Chart selections do not cross-filter other visuals. Clear selections with each slicer's eraser; use the bottom navigator (Ctrl+click in Desktop edit mode) or page tabs to switch pages.

The redesign keeps all 28 measure definitions, both relationships, original columns and imports unchanged. Three calculated text columns only provide shorter band labels, purpose labels without underscores and readable SHAP feature names. No source categories, feature definitions or numbers changed. The original four probability bands are retained.

## Checks performed

All four pages opened and rendered in Desktop; the project was saved and reopened. All 28 measures executed in the local DAX engine, and the live model had exactly two relationships. Table counts and risk-band sort metadata matched the CSVs.

| DAX check | Result |
|---|---:|
| Loans / defaults | 162,570 / 22,315 |
| Default rate / mean score | 13.72639478% / 13.13029927% |
| Loans scoring at least 20% | 12,615 |
| Baseline / tuned ROC-AUC | 0.64283263 / 0.65949575 |
| Grade A loans / defaults / score >=20% | 35,333 / 1,902 / 3 |
| January loans / defaults / score >=20% | 11,495 / 1,433 / 803 |
| Lowest band loans / defaults / score >=20% | 47,587 / 2,898 / 0 |
| Empty slice | Counts 0; rate and mean score blank |

The saved report opens on Overview with every slicer at All (162,570 loans); baseline ROC-AUC is displayed as 0.6428 (stored 0.64283263), matching `outputs/model_comparison.csv`. Baseline/final AP, log loss, Brier score, calibration bins, and all four confusion cells matched the frozen outputs. The redesigned pages are checked at fit-to-page size. Grade selection, clearing filters, fixed global metrics, navigation and band ordering are reviewed in Desktop. Screenshots in `docs/images/` are cropped Desktop captures (interface chrome removed), not mockups. All seven reporting hashes and six referenced frozen model/SHAP input hashes remained unchanged.

## Limits

The 0.5 threshold predicts no positives: recall is zero and precision is undefined. Threshold optimization remains future work. Calibration curves use populated fixed-width bins; sparse high-score bins are unstable. SHAP describes model behavior, not causality.

Desktop uses the viewer's regional number grouping (the captures show `1,62,570`). All main charts fit without scrolling. State/home-ownership slicers, the dense state table, training-gain chart and confusion-cell cards were removed from the canvas. Loan counts and secondary figures are available in chart tooltips; financial cards report existing average principal and income, not invented exposure or loss estimates. Exhaustive keyboard/screen-reader testing, a clean-machine refresh, and Power BI Service publishing have not been tested.

Local `.pbi` caches and PBIX/ABF data files are ignored. Keep them out of commits: they can contain the full loan-level export.

The pre-redesign report and previous screenshots are preserved locally in `.local-backups/LoanRisk-before-redesign-20260919-204009.zip` (excluded from Git).

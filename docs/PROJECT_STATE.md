# Project state

- Stages 1–9: completed (2026-09-19); Stage 8 used the specification fallback. Overview/reproduction/tree: [README](../README.md). Dashboard: [design](powerbi_dashboard_design.md); [data dictionary](powerbi_data.md).
- Population: resolved 36-month loans; Fully Paid=0, Charged Off=1; natural prevalence. Time split: train 2007–2012 (72,566), validation 2013 (100,422), holdout 2014 (162,570; 22,315 defaults). Seed 42.
- Final: tuned LightGBM, 57 trees, 72 inputs; `outputs/final_model/model.txt`. Primary ROC-AUC: baseline 0.64283263, final 0.65949575. Final AP 0.21580617; log loss 0.38281766; Brier 0.11452292.
- Holdout consumed: no further tuning/threshold optimization. Threshold 0.5 unchanged; zero predicted positives/recall, precision undefined; imperfect calibration.
- Leakage controls: known post-outcome/servicing fields excluded; borrower overlap and bureau timing unresolved. Historical cohort; ZIP/date proxies warrant caution. SHAP is noncausal, based on 10,000 random holdout rows plus four local examples.
- Reporting: seven CSVs plus manifest in `outputs/powerbi/`, 19.26 MB. Source-row keys are not loan IDs. Bands [0,0.10), [0.10,0.15), [0.15,0.20), [0.20,1] are descriptive only.
- Dashboard: `powerbi/LoanRisk.pbip` implemented; four pages, seven imports, two relationships, all 28 DAX measures executed. Desktop rendering, reference totals and core slicer/sync/reset behavior checked. Real screenshots in `docs/images/`. See `powerbi/README.md` for setup and remaining checks.
- Stage 9 changes: README rewritten; `.gitignore` covers actual raw/row-level data; state filename normalized to `.md`. Source, dependencies and generated artifacts retained; publication preparation follows the original GitHub history.
- Verification: 13 tests pass; all dependency pins match; pip check passes; frozen modeling code and stage artifact/reporting hashes match; README links checked. Python now 3.12.14 versus recorded 3.12.13; fresh-environment rebuild not tested.
- Git publication: local main based on original origin/main history. Legacy sample/basic-model row CSVs removed from the updated tree, retained locally and in old commits. Raw data, predictions, environments and secrets excluded. Data provenance/redistribution rights remain unverified.
- Documentation style cleanup: removed agent handoffs and repetitive action logs; methods, results and limitations retained. Source/tests/outputs unchanged.
- Next: execution plan and Desktop dashboard complete. Clean-machine refresh, accessibility review, Service publishing and new-data model validation remain open. Frozen ML/reporting hashes unchanged.

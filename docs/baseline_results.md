# Baseline experiment

Completed 2026-09-19. One baseline fit, no tuning. Used the unchanged Stage 2 model configuration and frozen data/preprocessing contract. Final holdout remains unscored.

## Results

| Item | Result |
| --- | ---: |
| Training rows (2007–2012) | 72,566 |
| Training positives / rate | 9,130 / 12.58165% |
| Validation rows (2013) | 100,422 |
| Validation positives / rate | 12,378 / 12.32598% |
| Reserved holdout rows (2014) | 162,570 |
| Model input features | 72 |
| Validation ROC-AUC | **0.6426464329932009** |
| Validation average precision | 0.18717428225413663 |
| Validation log loss | 0.3603006160067117 |
| Validation Brier score | 0.10531420551436682 |
| Best iteration / saved trees | **31** |
| Iterations evaluated before stopping | 81 |
| Data loading and preprocessing | 26.31 seconds |
| Model fitting | 1.38 seconds |
| Total through model/artifact checks | 28.37 seconds |

Holdout default rate and metrics were not computed. Counts and eligibility remain those frozen in Stage 2. Training reported 67 usable features from the 72 input columns; the saved schema and importance table preserve all 72. Runtime excludes initial dependency installation and later independent export checks.

## Configuration and reproducibility

- Population: resolved 36-month loans, Fully Paid=0 / Charged Off=1; no resampling or class weighting. Training, validation, and holdout boundaries are unchanged.
- Classifier: GBDT; estimator cap 500; learning rate 0.05; maximum depth 7; 31 leaves; minimum child samples 20; minimum child weight 0.001; minimum split gain 0; L1/L2 0; column fraction 1; row fraction 1; row-sampling frequency 0; binning sample cap 200,000; `class_weight=None`; seed 42; `n_jobs=-1` (resolved to 20 threads).
- Objective resolves to binary classification. Monitor validation AUC only, with 50-round patience. The single baseline fit used training rows only.
- The complete sklearn parameter dictionary, resolved booster parameters, all 81 validation AUC observations, source checksum, contract checksum, code checksums, environment versions, and timings are recorded in `outputs/baseline/metrics.json`. `model.txt` also contains LightGBM's serialized parameter configuration and categorical metadata.
- Python 3.12.13, LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.0.1, NumPy 2.3.5, SciPy 1.18.1. `requirements.txt` pins these and the supporting runtime dependencies. The local project `venv/` inherits the bundled runtime's packages; it is ignored by existing ignore rules. A fresh environment can install the pinned requirements independently.
- The baseline entry point refuses to overwrite a nonempty output directory. Later experiments must use separate output locations and preserve this baseline.

On a fresh checkout with Python 3.12 and the same raw file plus Stage 2 contract:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B src/lgbm_model.py
```

The last command requires an empty `outputs/baseline/`. To deliberately verify a future reproduction without overwriting the frozen baseline, call `train_lgbm_model` with a separate output directory. Seed and versions are recorded, but bitwise equality of a new fit across different hardware/threading is not claimed.

## Saved artifacts and verification

| Artifact | Contents |
| --- | --- |
| `outputs/baseline/metrics.json` | Metrics, exact parameters, cohort information, history, environment and provenance |
| `outputs/baseline/model.txt` | Frozen 31-tree LightGBM model |
| `outputs/baseline/predictions.csv` | 100,422 validation predictions, labels, split name and source-row position |
| `outputs/baseline/feature_importance.csv` | All 72 feature names, gain ranks, gain values and split counts at the best iteration |

Source-row positions are zero-based raw CSV record positions, **not customer or loan identifiers**. Prediction rows were independently checked against raw date metadata: every one belongs to the 2013 validation cohort.

The five Stage 2 regression tests passed in the modeling environment. Reloading `model.txt` reproduced every validation probability with maximum absolute difference **0.0**. Independent checks recomputed all four metrics from the saved CSV (agreement within 1e-14), verified artifact/code hashes, labels/counts, the importance schema, absence of holdout metrics, and the overwrite guard.

## Historical comparison and issues

The README's unverified historical ROC-AUC is 0.6885. This run's validation ROC-AUC is numerically **0.0458535670 lower** (4.585 percentage points). This is **not a like-for-like deterioration estimate**: the target treatment, cohort/term coverage, sampling prevalence, preprocessing, class weighting, and split differ. This baseline applies to the Stage 2 benchmark; the historical run remains unreproduced.

Top gain drivers are `zip_code`, `earliest_cr_line`, `int_rate`, `annual_inc`, and `sub_grade`. The dominance of the two categorical fields warrants attention when interpreting later results, but gain does not prove leakage or causation. Feature selection was not changed in response to this result.

LightGBM warned about high-cardinality categorical bins and deprecated `eval_set` usage. Both were nonfatal; training, serialization and prediction verification passed. Parameters/API usage were left unchanged during baseline fitting; versions are pinned.

Limits from Stage 2 remain: this is a retrospective resolved-loan cohort benchmark, not a point-in-time deployment backtest; missing customer IDs prevent borrower-overlap verification; bureau snapshot timing is assumed. Validation was used for early stopping, so final generalization remains unknown until Stage 5's holdout evaluation.

The baseline uses the Stage 2 preprocessing unchanged. Its implementation is in `src/lgbm_model.py`, dependencies in `requirements.txt`, and four saved artifacts in `outputs/baseline/`. Historical outputs were retained.

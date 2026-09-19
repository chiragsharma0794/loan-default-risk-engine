# Hyperparameter search

Run date: 2026-09-19. The search completed all 24 trials using the baseline split and preprocessing. Holdout evaluation followed separately.

## Method and scope

Used a bounded random search, drawing all configurations before scoring candidates with NumPy's default random generator, seed 42. Each trial trained on all 72,566 training rows and used the same 100,422 validation rows. No additional sampling of the cohort or dependencies were needed. The same 72 input features, training-fitted category vocabularies, class weighting (`None`), and model seed 42 were retained.

Selection was strictly maximum validation ROC-AUC, with an earlier trial winning an exact tie. Secondary metrics did not drive selection. All trials used explicit validation AUC, 50-round early stopping, and a 1,500-tree ceiling to accommodate lower learning rates. None reached that ceiling. The search used one chronological validation cohort rather than cross-validation.

| Parameter | Search space |
| --- | --- |
| Learning rate | Log-uniform 0.015–0.12 |
| Maximum depth | 3, 5, 7, 9, or unlimited (-1) |
| Leaves | Draw 7, 15, 31, or 63; cap at `2**max_depth` for finite depth (so effective values can include 8 or 32) |
| Minimum child samples | 20, 50, 100, 200, 400 |
| Column sampling fraction | 0.7, 0.85, 1.0 |
| Row sampling fraction | 0.7, 0.85, 1.0 |
| Row sampling frequency | 1 when fraction < 1; otherwise 0 |
| L1 regularization | Log-uniform 0.0001–10 |
| L2 regularization | Log-uniform 0.001–30 |
| Minimum split gain | 0, 0.01, 0.05, 0.1 |

Other parameters come from the frozen baseline. LightGBM may consider fewer columns splittable under different minimum-child constraints; the supplied feature schema itself was not changed.

## Results

| Validation metric | Baseline | Best candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.64264643 | **0.65963371** | **+0.01698728** |
| Average precision | 0.18717428 | 0.19656789 | +0.00939360 |
| Log loss | 0.36030062 | 0.35860873 | -0.00169189 |
| Brier score | 0.10531421 | 0.10483189 | -0.00048232 |

Trial **12** was selected at **57 trees**. Exact selected search parameters:

```json
{
  "learning_rate": 0.03719515393564381,
  "max_depth": 3,
  "num_leaves": 8,
  "min_child_samples": 50,
  "colsample_bytree": 0.85,
  "subsample": 1.0,
  "subsample_freq": 0,
  "reg_alpha": 0.0011828328138181756,
  "reg_lambda": 0.06745781561587151,
  "min_split_gain": 0.0
}
```

The complete classifier and resolved booster parameters are in `outputs/tuning/best_params.json`. Its `fixed_iteration_parameters_for_stage5` sets `n_estimators=57`; the search configuration's 1,500 is only the ceiling, not the number of trees to refit blindly. The saved model already contains the selected 57 trees trained solely on the original training cohort.

Trials took **34.65 seconds** total. Loading and verification of the baseline took **25.36 seconds**; search and initial exports took **60.40 seconds** overall. These timings exclude later independent verification and the serialization repair described below. Full-cohort tuning was practical, so no smaller tuning sample was used.

## Checks and saved files

- Baseline code, artifacts, preprocessing and package versions matched the saved records before the search. Baseline validation predictions also matched. The files and preprocessing contract were unchanged afterwards.
- Verified deterministic, distinct parameter draws, valid leaf/depth and bagging combinations, and all five existing split/preprocessing regression tests.
- All 24 trial rows matched the drawn configurations. Checks also covered the maximum-AUC winner, saved metrics, validation row IDs/labels and output hashes. The overwrite guard prevents silently extending or replacing the frozen search.
- Disk verification exposed a Windows newline-conversion issue in manual model-text saving. Fixed the new script to use native LightGBM serialization and re-saved the **same** selected model without retraining. Reloading the actual disk file then reproduced every validation probability with maximum absolute difference **0.0**. The report records both the original search-code hash and corrected code hash; neither parameters nor trial results changed.
- Nonfatal LightGBM categorical-bin, deprecated `eval_set`, and no-positive-gain warnings occurred. All trials completed with finite probabilities. No changes were made in response to their validation scores beyond the predefined search.

Created `src/tune_lgbm.py` and these frozen outputs:

- `outputs/tuning/tuning_trials.csv`: parameters, four validation metrics, iterations and timing for every trial.
- `outputs/tuning/best_params.json`: exact selected configuration, search definition, metric comparison, provenance, runtime and Stage 5 parameters.
- `outputs/tuning/candidate_model.txt`: selected 57-tree model.
- `outputs/tuning/validation_predictions.csv`: candidate predictions for the same validation rows as baseline.

The search left baseline artifacts, preprocessing, dependencies and SHAP unchanged. To reproduce on a fresh output location, use the pinned project environment and `run_tuning(output_dir=...)`; the default command is `venv\Scripts\python.exe -B src/tune_lgbm.py` and refuses an existing nonempty tuning directory.

## Interpretation

The AUC gain is **1.6987 percentage points on validation**, not proven generalization. Trial 7 was close (0.65949960), and selecting from 24 trials increases validation optimism. This is a candidate for final comparison, not a production model or a confirmed winner on holdout. Missing borrower IDs, assumed bureau snapshot timing, and the retrospective resolved-36-month-loan population remain limitations.

The candidate was ready for comparison with the baseline using the same preprocessing and original training cohort. The 162,570-row 2014 holdout was still unscored, with no default rate or metrics computed. The evaluation policy excludes train+validation refitting and retuning against holdout.

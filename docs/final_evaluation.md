# Holdout evaluation

Completed 2026-09-19. **Selected the tuned 57-tree model.** The improvement observed during tuning also appeared on the reserved 2014 cohort. Parameters and the 0.5 threshold were fixed before this evaluation; calibration and SHAP were not fitted here.

## Evaluation controls

Verified baseline/tuning artifact hashes, source-code hashes, the frozen preprocessing contract, and package versions before evaluation. Refit the candidate with its already-selected 57-tree parameters on the original 72,566 training rows only, without an evaluation set or early stopping. It reproduced all saved candidate validation probabilities with maximum absolute difference **0.0**, before holdout access.

The 31-tree baseline remained frozen. Both models used the same 72-feature schema and training category vocabularies. The holdout contained **162,570** resolved 36-month loans issued in 2014, including **22,315 defaults (13.72639%)**. Source-row IDs were unique and disjoint from development rows. Both models scored this same holdout once; later export checks used those saved scores, not new model selection or rescoring.

Before opening holdout, the script persisted the selection rule: choose the tuned candidate only if its holdout ROC-AUC is higher and the paired 95% bootstrap interval for the AUC difference has a lower bound above zero; otherwise retain the baseline. The bootstrap uses 500 class-stratified, paired loan-row resamples, seed 42. It assumes independent rows; missing customer IDs prevent borrower-cluster uncertainty estimates.

## Holdout results

| Metric | Frozen baseline | Tuned candidate | Tuned minus baseline |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.64283263 | **0.65949575** | **+0.01666311** |
| Average precision | 0.20590956 | 0.21580617 | +0.00989661 |
| Log loss | 0.38518541 | 0.38281766 | -0.00236774 |
| Brier score | 0.11506916 | 0.11452292 | -0.00054624 |

The paired 95% interval for the AUC gain is **[+0.01522395, +0.01807960]**, or approximately **+1.52 to +1.81 percentage points**. The point gain is +1.66631 percentage points, close to the validation gain of +1.69873 points. All three secondary metrics improved. The tuned model therefore met the selection rule on holdout as well as improving validation AUC.

This supports generalization to the specified historical cohort, not to modern portfolios, active/60-month loans, or an actual historical deployment. The historical README score of 0.6885 remains unrelated to this controlled comparison.

## Threshold and calibration issues

At the predeclared descriptive threshold **probability >= 0.5**, neither model predicts a default. Their identical confusion matrices are:

| Actual outcome | Predicted non-default | Predicted default |
| --- | ---: | ---: |
| Non-default | 140,255 | 0 |
| Default | 22,315 | 0 |

Both have **0% recall**. Precision is **undefined**, recorded as null because there are no positive predictions. Maximum probabilities are 0.43336 for baseline and 0.32618 for tuned. AUC improvement therefore does not imply useful classification at 0.5. This threshold was not optimized, and it is not a recommended business decision threshold. A future operational threshold would need an explicit cost/capacity policy and fresh validation; this consumed holdout must not be used to optimize it.

The tuned model's mean probability is **13.13030%**, versus **13.72639%** observed defaults, an aggregate underprediction of **0.59610 percentage points**. Baseline underprediction is 0.67135 points. Better log loss/Brier score does not establish perfect calibration. The tuned model's nonempty reliability bins show:

| Predicted probability bin | Loans | Mean predicted | Observed default rate |
| --- | ---: | ---: | ---: |
| [0.0, 0.1) | 47,587 | 7.74407% | 6.08990% |
| [0.1, 0.2) | 102,368 | 14.55147% | 15.77641% |
| [0.2, 0.3) | 12,600 | 21.90513% | 25.88095% |
| [0.3, 0.4) | 15 | 31.07139% | 40.00000% |

The last bin has too few observations for a stable calibration conclusion. `calibration.csv` includes all ten fixed equal-width bins for both models, including empty bins. No calibration correction was learned from holdout.

## Final model and artifacts

The selected model uses depth 3, 8 leaves, learning rate 0.03719515393564381, minimum child samples 50, column sampling 0.85, row sampling 1.0 (frequency 0), L1 0.0011828328138181756, L2 0.06745781561587151, minimum split gain 0, no class weighting, seed 42, and exactly **57 trees**. Full parameters and provenance are in the final metrics file. It was trained on the original training cohort, not train+validation.

Created:

- `src/final_evaluation.py`: fixed-configuration refit, guarded holdout evaluation, paired uncertainty and exports.
- `tests/test_final_evaluation.py`: numerical checks for weighted AUC/ties, paired resampling, threshold behavior and calibration bins.
- `outputs/final_model/model.txt`: selected model, saved with native LightGBM serialization.
- `outputs/final_model/metrics.json`: comparison, selection rule, parameters, interval, confusion matrices, provenance and hashes.
- `outputs/final_model/holdout_predictions.csv`: both models' probabilities and actual outcomes on identical source rows. Row positions are not borrower/loan identifiers.
- `outputs/final_model/calibration.csv`: reliability tables for both models.
- `outputs/model_comparison.csv`: four approved metrics and signed absolute differences.
- This report and the project-state record.

All 10 regression/numerical tests passed. The final model reload reproduced its validation probabilities; exported holdout metrics, confusion matrices, calibration totals, row dates/terms/labels, and hashes were independently checked. The final model has 57 trees and 72 input columns. Existing baseline/tuning files and preprocessing were unchanged. The overwrite guard blocks accidental repeated holdout evaluation.

Evaluation took **64.10 seconds** through internal checks, including **0.37 seconds** for the fixed candidate refit; these timings exclude later independent export checks. The pinned environment and dependencies were unchanged.

The 2014 holdout has now been used for evaluation and cannot support further tuning. Subsequent SHAP analysis uses `outputs/final_model/model.txt`. Missing borrower identity, assumed bureau snapshot timing and retrospective cohort limitations remain.

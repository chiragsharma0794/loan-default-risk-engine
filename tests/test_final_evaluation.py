"""Numerical checks for holdout reporting; never opens the raw data."""
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.final_evaluation import (
    weighted_auc_from_bins, paired_auc_interval, evaluate_predictions, calibration_table,
)


class FinalEvaluationTests(unittest.TestCase):
    def test_weighted_auc_matches_sklearn_with_ties_and_zero_weights(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        scores = np.array([0.1, 0.5, 0.5, 0.5, 0.8, 0.9])
        weights = np.array([2, 1, 3, 0, 2, 1])
        values, groups = np.unique(scores, return_inverse=True)
        positive = np.bincount(groups[y == 1], weights=weights[y == 1], minlength=len(values))
        negative = np.bincount(groups[y == 0], weights=weights[y == 0], minlength=len(values))
        self.assertAlmostEqual(weighted_auc_from_bins(positive, negative),
                               roc_auc_score(y, scores, sample_weight=weights), places=14)

    def test_paired_bootstrap_identical_and_opposite_perfect_models(self):
        y = np.array([0, 0, 1, 1])
        good = np.array([0.1, 0.2, 0.8, 0.9])
        same = paired_auc_interval(y, good, good, replicates=30)
        self.assertEqual((same['lower'], same['upper']), (0.0, 0.0))
        improved = paired_auc_interval(y, 1 - good, good, replicates=30)
        self.assertEqual((improved['lower'], improved['upper']), (1.0, 1.0))
        self.assertEqual(improved, paired_auc_interval(y, 1 - good, good, replicates=30))

    def test_confusion_matrix_includes_threshold_equality(self):
        result = evaluate_predictions([0, 1, 1, 0], [0.1, 0.4, 0.5, 0.8])
        self.assertEqual(result['confusion_matrix'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})
        self.assertEqual(result['precision'], 0.5)
        self.assertEqual(result['recall'], 0.5)

    def test_no_positive_predictions_has_undefined_precision(self):
        result = evaluate_predictions([0, 1], [0.1, 0.2])
        self.assertIsNone(result['precision'])
        self.assertEqual(result['recall'], 0.0)
        self.assertEqual(result['predicted_positive_rows'], 0)

    def test_calibration_boundaries_and_empty_bins(self):
        rows = calibration_table([0, 1, 1], [0.0, 0.1, 1.0], 'test')
        self.assertEqual(sum(row['rows'] for row in rows), 3)
        self.assertEqual(sum(row['defaults'] for row in rows), 2)
        self.assertEqual([rows[i]['rows'] for i in (0, 1, 9)], [1, 1, 1])
        self.assertIsNone(rows[2]['mean_probability'])
        self.assertEqual(rows[9]['observed_default_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()

"""Reporting bands must preserve boundary semantics and totals, including empty bands."""
import unittest

import numpy as np
import pandas as pd

from src.prepare_powerbi_data import assign_risk_bands, summarize_bands


class PowerBIDataTests(unittest.TestCase):
    def test_boundaries_and_classification_are_distinct(self):
        probabilities = [0, .099999, .1, .149999, .15, .199999, .2, .499999, .5, 1]
        np.testing.assert_array_equal(assign_risk_bands(probabilities), [1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
        for invalid in [np.nan, np.inf, -.01, 1.01]:
            with self.assertRaises(ValueError):
                assign_risk_bands([invalid])

    def test_empty_bands_and_weighted_totals(self):
        frame = pd.DataFrame({'risk_band_id': [1, 1, 4], 'actual_default': [0, 1, 1],
                              'predicted_probability': [.02, .08, .25]})
        summary = summarize_bands(frame)
        self.assertEqual(summary.loan_count.tolist(), [2, 0, 0, 1])
        self.assertEqual(summary.default_count.tolist(), [1, 0, 0, 1])
        self.assertTrue(summary.loc[summary.loan_count.eq(0), 'default_rate'].isna().all())
        self.assertAlmostEqual((summary.default_rate * summary.loan_count).sum() / 3, 2 / 3)
        self.assertAlmostEqual((summary.avg_predicted_probability * summary.loan_count).sum() / 3,
                               frame.predicted_probability.mean())


if __name__ == '__main__':
    unittest.main()

"""Local examples must respect the unchanged classification threshold."""
import unittest
import pandas as pd
from src.shap_explainability import choose_local_examples


class ShapExampleTests(unittest.TestCase):
    def test_score_rank_does_not_turn_false_negatives_into_true_positives(self):
        predictions = pd.DataFrame({
            'source_row': [15, 12, 13, 14, 18, 11],
            'actual_default': [0, 0, 1, 1, 1, 0],
            'tuned_probability': [0.05, 0.05, 0.08, 0.20, 0.30, 0.25],
        })
        result = choose_local_examples(predictions)
        self.assertEqual([row['source_row'] for row in result], [12, 11, 13, 18])
        self.assertEqual([row['classification'] for row in result], ['TN', 'TN', 'FN', 'FN'])
        self.assertTrue(all(row['predicted_class_at_0_5'] == 0 for row in result))
        self.assertEqual(result, choose_local_examples(predictions.sample(frac=1, random_state=9)))


if __name__ == '__main__':
    unittest.main()

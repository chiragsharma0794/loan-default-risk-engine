"""Regression checks for split isolation and preprocessing; no model training."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.stratified_sampling import POLICY, assign_splits, load_development_data, source_sha256
from src.feature_screening import fit_preprocessor, transform_features, EXCLUDED_COLUMNS


class SplitTests(unittest.TestCase):
    def test_boundaries_and_unresolved_outcomes(self):
        data = pd.DataFrame({
            "issue_d": ["Dec-2012", "Jan-2013", "Dec-2013", "Jan-2014", "Dec-2014", "Jan-2015",
                        "Jan-2012", "Jan-2012", "Jan-2012"],
            "term": [" 36 months"] * 8 + ["60 months"],
            "loan_status": ["Fully Paid"] * 6 + ["Current", "Default", "Charged Off"],
        })
        self.assertEqual(assign_splits(data).tolist(),
                         ["train", "validation", "validation", "holdout", "holdout", "excluded",
                          "excluded", "excluded", "excluded"])

    def test_missing_dates_fail(self):
        data = pd.DataFrame({"issue_d": [None], "term": ["36 months"], "loan_status": ["Fully Paid"]})
        with self.assertRaises(ValueError):
            assign_splits(data)

    def test_loader_excludes_holdout_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "loan.csv"
            report_path = Path(directory) / "contract.json"
            data = pd.DataFrame({
                "issue_d": ["Dec-2012", "Jan-2013", "Jan-2014"],
                "term": ["36 months"] * 3,
                "loan_status": ["Fully Paid", "Charged Off", "Charged Off"],
                "loan_amnt": [1000, 2000, 999999],
            })
            data.to_csv(raw, index=False)
            report_path.write_text(json.dumps({
                "policy": POLICY, "integrity_passed": True,
                "source": {"sha256": source_sha256(raw)},
                "preprocessing": {"raw_features": ["loan_amnt", "term"]},
                "splits": {"train": {"rows": 1}, "validation": {"rows": 1}},
            }), encoding="utf-8")
            train, validation, _ = load_development_data(raw, report_path, chunksize=1)
            self.assertEqual(train.index.tolist(), [0])
            self.assertEqual(validation.index.tolist(), [1])
            self.assertNotIn("999999", pd.concat([train, validation]).loan_amnt.tolist())
            with raw.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Raw source changed"):
                load_development_data(raw, report_path)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "loan_amnt": [1000] * 6,
            "annual_inc": [100, 200, 0, None, -1, np.inf],
            "emp_length": ["10+ years", "< 1 year", "1 year", None, "n/a", "3 years"],
            "grade": pd.Series(["A", "A", "B", "B", "A", "B"], dtype="string"),
            "dti": [1, None, None, None, None, None],
            "loan_status": ["Fully Paid"] * 6,
            "target": [0] * 6,
            "pymnt_plan": ["n"] * 6,
            "deferral_term": [3] * 6,
            "payment_plan_start_date": ["Jan-2019"] * 6,
            "orig_projected_additional_accrued_interest": [100] * 6,
            "hardship_flag": ["N"] * 6,
            "funded_amnt": [1000] * 6,
            "disbursement_method": ["CASH"] * 6,
        })

    def test_training_only_schema_and_unseen_categories(self):
        x_train, schema = fit_preprocessor(self.train)
        validation = self.train.copy()
        validation["dti"] = 20
        validation["grade"] = "C"
        x_validation = transform_features(validation, schema)
        self.assertNotIn("dti", x_validation)
        self.assertFalse(set(x_train).intersection(EXCLUDED_COLUMNS))
        self.assertNotIn("hardship_flag", x_train)
        self.assertTrue(x_validation.grade.isna().all())
        self.assertEqual(x_train.grade.cat.categories.tolist(), ["A", "B"])
        self.assertTrue(x_train.dtypes.equals(x_validation.dtypes))

    def test_employment_and_invalid_financial_inputs(self):
        features, _ = fit_preprocessor(self.train)
        self.assertEqual(features.emp_length_num.iloc[:3].tolist(), [10.0, 0.0, 1.0])
        self.assertTrue(features.emp_length_num.iloc[3:5].isna().all())
        self.assertEqual(features.emp_length_num.iloc[5], 3.0)
        self.assertTrue(features.loan_to_inc_ratio.iloc[2:].isna().all())
        self.assertFalse(np.isinf(features.select_dtypes(include="number").to_numpy()).any())


if __name__ == "__main__":
    unittest.main()

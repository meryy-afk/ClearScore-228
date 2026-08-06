"""Regression checks for the transaction-based ClearScore MVP.

Run with: python -m unittest test_pipeline.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis_service import analyze_csv
from benford import analyze_benford
from data_loader import DataValidationError, load_csv


class ClearScorePipelineTests(unittest.TestCase):
    def test_good_business_has_low_risk(self) -> None:
        report = analyze_csv("good_business.csv")
        self.assertGreaterEqual(report["clear_score"], 70)
        self.assertEqual(report["risk_level"], "Низкий")
        self.assertGreater(report["metrics"]["cash_buffer_days"], 13)

    def test_risky_business_has_high_risk(self) -> None:
        report = analyze_csv("risky_business.csv")
        self.assertLess(report["clear_score"], 40)
        self.assertEqual(report["risk_level"], "Высокий")

    def test_missing_cash_balance_is_carried_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cash_gap.csv"
            path.write_text(
                "date,revenue,expenses,cash_balance\n"
                "2026-07-01,100,50,1000\n"
                "2026-07-03,100,50,\n",
                encoding="utf-8",
            )
            data = load_csv(path)
        self.assertEqual(data.loc[1, "cash_balance"], 1000.0)
        self.assertEqual(data.loc[2, "cash_balance"], 1000.0)

    def test_invalid_csv_is_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            load_csv("invalid_data.csv")

    def test_daily_totals_do_not_trigger_benford_fraud_flag(self) -> None:
        result = analyze_benford([100000, 120000, 110000] * 10)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertFalse(result["suspicious_data_risk"])


if __name__ == "__main__":
    unittest.main()

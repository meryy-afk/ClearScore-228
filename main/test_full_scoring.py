"""Run the full Benford + XGBoost + SHAP ClearScore chain on one CSV file.

The CSV supplies the transaction amounts. The SBA features below represent the
loan-application profile of that same demo business and are required because
the current XGBoost model was trained on SBA loan data, not daily sales.
"""

from __future__ import annotations

import json

from data_loader import load_csv
from sba_experiment import calculate_sba_ml_score


INPUT_FILE = "good_business.csv"
DEMO_SBA_FEATURES = {
    "Term": 84,
    "NoEmp": 8,
    "NewExist": 1,
    "CreateJob": 2,
    "RetainedJob": 8,
    "FranchiseCode": 0,
    "UrbanRural": 1,
    "RevLineCr": 0,
    "LowDoc": 0,
    "NAICS": 722511,
    "ApprovalFY": 2012,
    "DisbursementGross": 5000000,
    "GrAppv": 5000000,
    "SBA_Appv": 4250000,
}


def main() -> None:
    transactions = load_csv(INPUT_FILE)
    # The SBA experiment needs individual operations; daily totals must not be
    # tested with Benford, so this script is retained only as a code example.
    transactions = transactions.rename(columns={"revenue": "amount"})
    report = calculate_sba_ml_score(transactions, DEMO_SBA_FEATURES)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Optional SBA/XGBoost experiment kept separate from the transaction MVP."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from benford import analyze_benford
from explainability import explain_business


def calculate_sba_ml_score(transactions: pd.DataFrame, model_features: Mapping[str, float]) -> dict:
    """Combine the separate SBA default model with a raw-transaction check.

    This is experimental and is intentionally not called by ``main.py``.
    """
    if "amount" not in transactions.columns:
        raise ValueError("Экспериментальная SBA-оценка требует отдельный столбец amount.")
    explanation = explain_business(model_features)
    if explanation["default_probability"] is None:
        raise ValueError("SBA-модель не смогла сформировать вероятность дефолта.")
    benford = analyze_benford(transactions["amount"])
    penalty = 20 if benford.get("suspicious_data_risk", False) else 0
    score = float(np.clip((1 - explanation["default_probability"]) * 100 - penalty, 0, 100))
    return {
        "clear_score": round(score, 2),
        "default_probability": explanation["default_probability"],
        "fraud_penalty": penalty,
        "benford_check": benford,
        "shap_explanations": explanation["top_factors"],
        "experimental": True,
    }

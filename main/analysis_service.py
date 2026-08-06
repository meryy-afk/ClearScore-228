"""One stable JSON contract for the ClearScore transaction MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from benford import analyze_benford
from data_loader import load_csv
from metrics import calculate_metrics
from scoring import calculate_score


def analyze_business(
    daily_data: pd.DataFrame, transaction_amounts: Iterable[float] | None = None
) -> dict[str, Any]:
    """Return one complete financial-health report without persisting data.

    Benford is run only when separate individual operation amounts are passed.
    Daily aggregates are deliberately not used for this statistical test.
    """
    metrics = calculate_metrics(daily_data)
    score = calculate_score(metrics)
    data_quality = _data_quality(metrics)
    benford = _benford_status(transaction_amounts)
    return {
        "clear_score": score["clear_score"],
        "risk_level": score["risk_level"],
        "metrics": metrics,
        "risk_flags": [penalty["code"] for penalty in score["penalties"]],
        "reasons": score["penalties"],
        "recommendations": score["recommendations"],
        "data_quality": data_quality,
        "benford_check": benford,
    }


def analyze_csv(file_path: str | Path) -> dict[str, Any]:
    """Load a daily CSV and produce the standard in-memory report."""
    return analyze_business(load_csv(file_path))


def _data_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    sufficient_history = metrics["days_analyzed"] >= 30
    return {
        "history_status": "sufficient" if sufficient_history else "limited",
        "days_analyzed": metrics["days_analyzed"],
        "message": None if sufficient_history else "Для устойчивой оценки рекомендуется не менее 30 дней истории.",
    }


def _benford_status(transaction_amounts: Iterable[float] | None) -> dict[str, Any]:
    if transaction_amounts is None:
        return {
            "status": "not_run",
            "suspicious_data_risk": False,
            "message": "Проверка Бенфорда доступна только для набора отдельных операций, а не дневных итогов.",
        }
    return analyze_benford(transaction_amounts)

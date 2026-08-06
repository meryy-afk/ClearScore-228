"""Explainable financial metrics for ClearScore's daily-data MVP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import LOW_CASH_DAYS_RESERVE


REQUIRED_COLUMNS = ("date", "revenue", "expenses", "cash_balance")


class MetricsValidationError(ValueError):
    """Raised when a DataFrame is unsuitable for financial calculations."""


def calculate_metrics(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate cash stability, trend and variability metrics in memory."""
    _validate_data(data)

    revenue = data["revenue"].astype(float)
    expenses = data["expenses"].astype(float)
    cash_balance = data["cash_balance"].astype(float)
    daily_net_flow = revenue - expenses
    average_daily_revenue = float(revenue.mean())
    average_daily_expenses = float(expenses.mean())
    average_daily_net_flow = float(daily_net_flow.mean())
    latest_cash_balance = float(cash_balance.iloc[-1])

    # Comparable in meaning to the JPMorgan cash-buffer benchmark: how many
    # days of typical operating outflows the latest balance can cover if
    # inflows stop. It intentionally differs from runway, which uses net loss.
    cash_buffer_days = (
        None
        if average_daily_expenses <= 0
        else float(max(latest_cash_balance, 0) / average_daily_expenses)
    )

    runway_days, runway_is_unlimited = _calculate_runway(
        latest_cash_balance, average_daily_net_flow
    )
    conservative_runway_days, conservative_runway_is_unlimited = _calculate_conservative_runway(
        latest_cash_balance, daily_net_flow
    )

    revenue_std = float(np.std(revenue.to_numpy(), ddof=0))
    volatility = 0.0 if average_daily_revenue == 0 else (revenue_std / average_daily_revenue) * 100
    cash_peak = cash_balance.cummax()
    drawdown = ((cash_peak - cash_balance) / cash_peak.where(cash_peak > 0, np.nan)).fillna(0)
    low_cash_threshold = average_daily_expenses * LOW_CASH_DAYS_RESERVE

    return {
        "period_start": data["date"].iloc[0].strftime("%Y-%m-%d"),
        "period_end": data["date"].iloc[-1].strftime("%Y-%m-%d"),
        "days_analyzed": int(len(data)),
        "total_revenue": float(revenue.sum()),
        "total_expenses": float(expenses.sum()),
        "net_cash_flow": float(daily_net_flow.sum()),
        "average_daily_revenue": average_daily_revenue,
        "average_daily_expenses": average_daily_expenses,
        "average_daily_net_flow": average_daily_net_flow,
        "latest_cash_balance": latest_cash_balance,
        "cash_buffer_days": cash_buffer_days,
        "expense_to_revenue_pct": 0.0 if revenue.sum() == 0 else float(expenses.sum() / revenue.sum() * 100),
        "runway_days": runway_days,
        "runway_is_unlimited": runway_is_unlimited,
        "conservative_runway_days": conservative_runway_days,
        "conservative_runway_is_unlimited": conservative_runway_is_unlimited,
        "revenue_volatility_pct": float(volatility),
        "revenue_trend_14d_pct": _revenue_trend(revenue),
        "unprofitable_days": int((daily_net_flow < 0).sum()),
        "unprofitable_days_pct": float((daily_net_flow < 0).mean()),
        "max_cash_drawdown_pct": float(drawdown.max()),
        "low_cash_days": int((cash_balance < low_cash_threshold).sum()),
        "low_cash_days_pct": float((cash_balance < low_cash_threshold).mean()),
        "low_cash_threshold": float(low_cash_threshold),
    }


def _calculate_runway(balance: float, average_net_flow: float) -> tuple[float | None, bool]:
    if balance <= 0:
        return 0.0, False
    if average_net_flow >= 0:
        return None, True
    return float(balance / abs(average_net_flow)), False


def _calculate_conservative_runway(
    balance: float, daily_net_flow: pd.Series
) -> tuple[float | None, bool]:
    """Use the 75th-percentile daily loss as a cautious runway scenario."""
    if balance <= 0:
        return 0.0, False
    losses = (-daily_net_flow[daily_net_flow < 0]).to_numpy()
    if len(losses) == 0:
        return None, True
    cautious_daily_loss = float(np.percentile(losses, 75))
    return float(balance / cautious_daily_loss), False


def _revenue_trend(revenue: pd.Series) -> float | None:
    """Compare the most recent 14 days with the previous 14 days."""
    if len(revenue) < 28:
        return None
    previous = float(revenue.iloc[-28:-14].mean())
    recent = float(revenue.iloc[-14:].mean())
    if previous == 0:
        return None
    return float((recent / previous - 1) * 100)


def _validate_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise MetricsValidationError("Для расчёта метрик нужен pandas.DataFrame.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise MetricsValidationError("Для расчёта не хватает столбцов: " + ", ".join(missing) + ".")
    if data.empty:
        raise MetricsValidationError("Невозможно посчитать метрики: данных нет.")
    if data[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise MetricsValidationError("В финансовых данных есть пустые числовые значения.")

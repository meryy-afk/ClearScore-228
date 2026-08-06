"""The primary, transaction-based ClearScore rules engine."""

from __future__ import annotations

from typing import Any

from config import (
    CASH_DRAWDOWN_RULES,
    LOW_CASH_DAYS_PENALTY,
    MAX_SCORE,
    NEGATIVE_NET_FLOW_PENALTY,
    NON_POSITIVE_BALANCE_PENALTY,
    REVENUE_TREND_RULES,
    RISK_LEVELS,
    RUNWAY_RULES,
    UNPROFITABLE_DAYS_RULES,
    VOLATILITY_RULES,
    ZERO_NET_FLOW_PENALTY,
)


class ScoringValidationError(ValueError):
    """Raised when metrics do not meet the scoring engine's contract."""


REQUIRED_METRICS = {
    "runway_days", "runway_is_unlimited", "net_cash_flow", "latest_cash_balance",
    "revenue_volatility_pct", "revenue_trend_14d_pct", "unprofitable_days_pct",
    "max_cash_drawdown_pct", "low_cash_days_pct",
}


def calculate_score(metrics: dict[str, Any]) -> dict[str, Any]:
    """Calculate an explainable 0--100 financial-stability score.

    This is the only score used by the MVP. It relies solely on the uploaded
    daily financial history; the separate SBA/XGBoost experiment is not mixed
    into this result.
    """
    _validate_metrics(metrics)
    penalties: list[dict[str, Any]] = []

    if not metrics["runway_is_unlimited"]:
        _apply_less_than(penalties, "runway", float(metrics["runway_days"]), RUNWAY_RULES,
                         "Запас прочности меньше {threshold} дней.")

    net_flow = float(metrics["net_cash_flow"])
    if net_flow < 0:
        _add(penalties, "negative_net_flow", NEGATIVE_NET_FLOW_PENALTY,
             "Расходы за период превысили выручку: денежный поток отрицательный.")
    elif net_flow == 0:
        _add(penalties, "zero_net_flow", ZERO_NET_FLOW_PENALTY,
             "Выручка полностью равна расходам: финансовый резерв не создаётся.")

    if float(metrics["latest_cash_balance"]) <= 0:
        _add(penalties, "non_positive_balance", NON_POSITIVE_BALANCE_PENALTY,
             "На последнюю дату отсутствует положительный остаток денежных средств.")

    _apply_greater_or_equal(penalties, "revenue_volatility", float(metrics["revenue_volatility_pct"]),
                            VOLATILITY_RULES, "Высокая волатильность выручки: {threshold}% и выше.")
    trend = metrics["revenue_trend_14d_pct"]
    if trend is not None:
        _apply_less_than(penalties, "revenue_decline", float(trend), REVENUE_TREND_RULES,
                         "Выручка за последние 14 дней снизилась более чем на {absolute_threshold}%.", absolute=True)
    _apply_greater_or_equal(penalties, "unprofitable_days", float(metrics["unprofitable_days_pct"]),
                            UNPROFITABLE_DAYS_RULES, "Доля убыточных дней: {percent_threshold}% и выше.", percent=True)
    _apply_greater_or_equal(penalties, "cash_drawdown", float(metrics["max_cash_drawdown_pct"]),
                            CASH_DRAWDOWN_RULES, "Максимальная просадка остатка: {percent_threshold}% и выше.", percent=True)
    if float(metrics["low_cash_days_pct"]) >= 0.30:
        _add(penalties, "low_cash_days", LOW_CASH_DAYS_PENALTY,
             "Более 30% дней остаток был ниже недельного резерва расходов.")

    total_penalty = sum(item["points"] for item in penalties)
    clear_score = max(0, min(MAX_SCORE, MAX_SCORE - total_penalty))
    return {
        "clear_score": clear_score,
        "risk_level": _risk_level(clear_score),
        "total_penalty": total_penalty,
        "penalties": penalties,
        "recommendations": _recommendations(penalties),
    }


def _apply_less_than(penalties: list[dict[str, Any]], code: str, value: float, rules: tuple,
                      template: str, absolute: bool = False) -> None:
    for threshold, points in rules:
        if value < threshold:
            display = abs(threshold) if absolute else threshold
            _add(penalties, code, points, template.format(threshold=display, absolute_threshold=display))
            return


def _apply_greater_or_equal(penalties: list[dict[str, Any]], code: str, value: float, rules: tuple,
                            template: str, percent: bool = False) -> None:
    for threshold, points in rules:
        if value >= threshold:
            display = threshold * 100 if percent else threshold
            _add(penalties, code, points, template.format(threshold=display, percent_threshold=display))
            return


def _add(penalties: list[dict[str, Any]], code: str, points: int, reason: str) -> None:
    penalties.append({"code": code, "points": points, "reason": reason})


def _risk_level(score: int) -> str:
    for threshold, label in RISK_LEVELS:
        if score >= threshold:
            return label
    return "Высокий"


def _recommendations(penalties: list[dict[str, Any]]) -> list[str]:
    codes = {item["code"] for item in penalties}
    recommendations = []
    if {"runway", "non_positive_balance", "low_cash_days"} & codes:
        recommendations.append("Увеличьте денежный резерв: отложите необязательные расходы или ускорьте поступления.")
    if {"negative_net_flow", "zero_net_flow", "unprofitable_days"} & codes:
        recommendations.append("Пересмотрите регулярные расходы и стремитесь к устойчивому положительному денежному потоку.")
    if {"revenue_volatility", "revenue_decline"} & codes:
        recommendations.append("Планируйте расходы по консервативному сценарию и работайте над стабильностью продаж.")
    if "cash_drawdown" in codes:
        recommendations.append("Контролируйте просадку остатка: задайте минимальный уровень денег на счёте.")
    return recommendations or ["Показатели стабильны. Сохраняйте резерв и регулярно обновляйте данные."]


def _validate_metrics(metrics: dict[str, Any]) -> None:
    if not isinstance(metrics, dict):
        raise ScoringValidationError("Для скоринга нужен словарь метрик из metrics.py.")
    missing = sorted(REQUIRED_METRICS - set(metrics))
    if missing:
        raise ScoringValidationError("Для скоринга не хватает метрик: " + ", ".join(missing) + ".")

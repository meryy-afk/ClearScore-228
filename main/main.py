"""Console entry point for the ClearScore transaction MVP."""

from __future__ import annotations

import argparse
import json

from analysis_service import analyze_csv
from data_loader import DataValidationError
from metrics import MetricsValidationError
from scoring import ScoringValidationError


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearScore SME: оценка финансовой устойчивости бизнеса.")
    parser.add_argument("file_path", nargs="?", default="test_data.csv", help="CSV или XLSX с date, revenue, expenses и cash_balance.")
    parser.add_argument("--json", action="store_true", help="Вывести полный JSON-отчёт.")
    args = parser.parse_args()
    try:
        report = analyze_csv(args.file_path)
    except (DataValidationError, MetricsValidationError, ScoringValidationError, ValueError) as error:
        print(f"Ошибка обработки данных: {error}")
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    metrics = report["metrics"]
    print("\n" + "=" * 60)
    print("CLEARSCORE SME — ФИНАНСОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"Период: {metrics['period_start']} — {metrics['period_end']} ({metrics['days_analyzed']} дней)")
    print(f"Выручка: {_money(metrics['total_revenue'])}")
    print(f"Расходы: {_money(metrics['total_expenses'])}")
    print(f"Чистый поток: {_money(metrics['net_cash_flow'])}")
    print(f"Денежный буфер: {_days(metrics['cash_buffer_days'], metrics['cash_buffer_days'] is None)}")
    print(f"Runway: {_days(metrics['runway_days'], metrics['runway_is_unlimited'])}")
    print(f"ClearScore: {report['clear_score']} / 100 — риск {report['risk_level']}")
    print("\nПричины:")
    for reason in report["reasons"] or [{"reason": "Существенных факторов риска не обнаружено."}]:
        print(f"• {reason['reason']}")
    print("\nРекомендации:")
    for item in report["recommendations"]:
        print(f"• {item}")
    print("=" * 60)


def _money(value: float) -> str:
    return f"{value:,.0f} KZT".replace(",", " ")


def _days(value: float | None, unlimited: bool) -> str:
    return "не ограничен" if unlimited else f"{value:.1f} дней"


if __name__ == "__main__":
    raise SystemExit(main())

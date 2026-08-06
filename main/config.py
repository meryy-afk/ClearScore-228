"""Business rules for the transaction-based ClearScore MVP.

Changing thresholds here changes the score without changing the calculation
code. Values are deliberately transparent for a demo and must be calibrated
with real outcome data before any production use.
"""

MIN_HISTORY_DAYS = 30
LOW_CASH_DAYS_RESERVE = 7
BENFORD_MIN_SAMPLE_SIZE = 100
BENFORD_RISK_THRESHOLD = 0.7

MAX_SCORE = 100
RISK_LEVELS = ((70, "Низкий"), (40, "Средний"), (0, "Высокий"))

# Cash-buffer benchmarks are adapted from JPMorgan Chase Institute, "Cash is
# King: Flows, Balances, and Buffer Days" (2016): 25th percentile = 13 days,
# median = 27 days, 75th percentile = 62 days. The source covers US small
# businesses; these are transparent MVP benchmarks, not Kazakhstan default
# probabilities. https://www.jpmorganchase.com/institute/all-topics/business-growth-and-entrepreneurship/report-cash-flows-balances-and-buffer-days
CASH_BUFFER_BENCHMARKS_DAYS = {
    "critical": 7,       # Conservative operational alert, not a universal scientific cutoff.
    "lower_quartile": 13,
    "median": 27,
    "upper_quartile": 62,
}

# Runway answers a different question: how long current cash covers the
# average *net loss*. These policy penalties are open MVP choices, tested on
# deterministic scenarios; they require calibration on local outcomes later.
RUNWAY_RULES = ((7, 40), (14, 25), (27, 10))
VOLATILITY_RULES = ((80, 20), (50, 12), (30, 5))
REVENUE_TREND_RULES = ((-30, 15), (-15, 8))
UNPROFITABLE_DAYS_RULES = ((0.60, 12), (0.40, 6))
CASH_DRAWDOWN_RULES = ((0.50, 10), (0.25, 5))

NEGATIVE_NET_FLOW_PENALTY = 20
ZERO_NET_FLOW_PENALTY = 8
NON_POSITIVE_BALANCE_PENALTY = 20
LOW_CASH_DAYS_PENALTY = 10

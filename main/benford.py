"""Benford-law checks for financial amount series.

This module is a data-quality signal, not proof of fraud. A low score marks a
distribution as unusual enough to merit manual review.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from config import BENFORD_MIN_SAMPLE_SIZE, BENFORD_RISK_THRESHOLD


BENFORD_PROBABILITIES = np.log10(1 + 1 / np.arange(1, 10))
RISK_THRESHOLD = BENFORD_RISK_THRESHOLD


def analyze_benford(values: Iterable[Union[int, float, str]]) -> dict[str, object]:
    """Compare first-digit frequencies of amounts with Benford's law.

    ``benford_score`` is the Chi-Square test p-value, bounded between 0 and 1.
    Higher values mean the observed digits are more consistent with the
    expected Benford distribution. Zero and non-numeric values are ignored.

    Args:
        values: Financial amounts, for example a pandas Series of revenue or
            expense values.

    Returns:
        A JSON-ready result with the score, Chi-Square statistics, observed
        distribution and the ``suspicious_data_risk`` flag.
    """
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    first_digits = numeric_values.map(_first_digit)
    first_digits = first_digits.dropna().astype(int)

    if first_digits.empty:
        raise ValueError("Для проверки Бенфорда нужны ненулевые числовые суммы.")

    if len(first_digits) < BENFORD_MIN_SAMPLE_SIZE:
        return {
            "status": "insufficient_data",
            "benford_score": None,
            "suspicious_data_risk": False,
            "risk_threshold": RISK_THRESHOLD,
            "sample_size": int(len(first_digits)),
            "minimum_sample_size": BENFORD_MIN_SAMPLE_SIZE,
            "message": "Для надёжной проверки Бенфорда нужно не менее 100 отдельных операций.",
        }

    observed_counts = first_digits.value_counts().reindex(range(1, 10), fill_value=0)
    expected_counts = BENFORD_PROBABILITIES * len(first_digits)
    chi_square, p_value = chisquare(observed_counts.to_numpy(), f_exp=expected_counts)
    benford_score = float(np.clip(p_value, 0.0, 1.0))

    return {
        "status": "completed",
        "benford_score": benford_score,
        "suspicious_data_risk": benford_score < RISK_THRESHOLD,
        "risk_threshold": RISK_THRESHOLD,
        "sample_size": int(len(first_digits)),
        "chi_square": float(chi_square),
        "p_value": float(p_value),
        "observed_distribution": {
            str(digit): float(observed_counts.loc[digit] / len(first_digits))
            for digit in range(1, 10)
        },
        "expected_distribution": {
            str(digit): float(BENFORD_PROBABILITIES[digit - 1]) for digit in range(1, 10)
        },
    }


def _first_digit(value: float) -> float:
    """Extract the first significant digit of an amount; return NaN for zero."""
    absolute_value = abs(float(value))
    if absolute_value == 0:
        return np.nan
    return float(str(f"{absolute_value:.15g}").lstrip("0.")[0])

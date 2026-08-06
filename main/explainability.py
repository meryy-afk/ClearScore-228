"""Safe SHAP explanations for one ClearScore SBA model prediction.

SHAP/XGBoost versions occasionally disagree about model metadata. This module
therefore always falls back to XGBoost's built-in feature importances instead
of letting an explanation error stop the scoring pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd


MODEL_PATH = Path("model.pkl")


def explain_business(
    features: Union[Mapping[str, float], pd.DataFrame], model_path: Path = MODEL_PATH
) -> dict[str, Any]:
    """Return default probability and the three strongest explanatory factors.

    SHAP values are calculated from the underlying XGBoost Booster to avoid
    sklearn-wrapper compatibility problems. If SHAP is unavailable or raises
    any exception, the function returns a safe fallback based on the model's
    ``feature_importances_`` and never propagates the SHAP exception.
    """
    try:
        model, feature_names, model_name = _load_model(model_path)
        feature_frame = _to_feature_frame(features, feature_names)
        default_probability = float(model.predict_proba(feature_frame)[0, 1])
    except Exception as error:
        return _unavailable_result(error)

    try:
        top_factors = _shap_top_factors(model, feature_frame, feature_names)
        return {
            "model_name": model_name,
            "default_probability": default_probability,
            "top_factors": top_factors,
            "explanation_method": "shap",
            "explanation_warning": None,
        }
    except Exception as error:
        return {
            "model_name": model_name,
            "default_probability": default_probability,
            "top_factors": _fallback_top_factors(model, feature_frame, feature_names),
            "explanation_method": "feature_importance_fallback",
            "explanation_warning": (
                "SHAP-объяснение недоступно; использована встроенная важность "
                f"признаков модели. Причина: {type(error).__name__}."
            ),
        }


def _shap_top_factors(
    model: Any, feature_frame: pd.DataFrame, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Calculate local SHAP factors through the native XGBoost Booster."""
    import shap

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    try:
        explainer = shap.TreeExplainer(booster)
        raw_values = explainer.shap_values(feature_frame)
    except Exception:
        # Newer SHAP releases may support the wrapper more reliably than the
        # native booster for a particular model serialisation.
        explainer = shap.Explainer(model)
        explanation = explainer(feature_frame)
        raw_values = explanation.values

    shap_values = _one_row_values(raw_values)
    if len(shap_values) != len(feature_names):
        raise ValueError("Число SHAP-значений не совпадает с числом признаков.")

    factors = []
    for feature_name, value, shap_value in zip(feature_names, feature_frame.iloc[0], shap_values):
        increases_default_risk = shap_value > 0
        factors.append(
            {
                "feature": feature_name,
                "value": float(value),
                "shap_value": float(shap_value),
                "effect": "снижает рейтинг" if increases_default_risk else "повышает рейтинг",
                "impact_on_default_risk": "повышает" if increases_default_risk else "снижает",
            }
        )
    return sorted(factors, key=lambda factor: abs(factor["shap_value"]), reverse=True)[:3]


def _fallback_top_factors(
    model: Any, feature_frame: pd.DataFrame, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Return top global feature importances if local SHAP fails safely."""
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if len(importances) != len(feature_names):
        importances = np.zeros(len(feature_names), dtype=float)

    ranked_indices = np.argsort(importances)[::-1][:3]
    return [
        {
            "feature": feature_names[index],
            "value": float(feature_frame.iloc[0, index]),
            "importance": float(importances[index]),
            "effect": "направление не определено в fallback",
            "impact_on_default_risk": "требуется SHAP для определения направления",
        }
        for index in ranked_indices
    ]


def _load_model(model_path: Path) -> tuple[Any, list[str], str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    with model_path.open("rb") as file:
        saved_model = pickle.load(file)

    if isinstance(saved_model, dict) and "model" in saved_model:
        model = saved_model["model"]
        feature_names = list(saved_model.get("features", []))
        model_name = str(saved_model.get("model_name", type(model).__name__))
    else:
        model = saved_model
        feature_names = list(getattr(model, "feature_names_in_", []))
        model_name = type(model).__name__

    if not feature_names:
        raise ValueError("В model.pkl отсутствует список признаков.")
    return model, feature_names, model_name


def _to_feature_frame(
    features: Union[Mapping[str, float], pd.DataFrame], feature_names: list[str]
) -> pd.DataFrame:
    if isinstance(features, Mapping):
        frame = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame) and len(features) == 1:
        frame = features.copy()
    else:
        raise ValueError("Передайте словарь признаков или DataFrame с одной строкой.")

    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError("Не хватает признаков для модели: " + ", ".join(missing))
    return frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")


def _one_row_values(raw_values: Any) -> np.ndarray:
    """Normalise SHAP output across SHAP versions to a one-row 1D array."""
    if hasattr(raw_values, "values"):
        raw_values = raw_values.values
    if isinstance(raw_values, list):
        raw_values = raw_values[1] if len(raw_values) > 1 else raw_values[0]

    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]
    if values.ndim != 1:
        raise ValueError("Неподдерживаемый формат SHAP-значений.")
    return values


def _unavailable_result(error: Exception) -> dict[str, Any]:
    """Return a JSON-safe result even if model loading/prediction fails."""
    return {
        "model_name": None,
        "default_probability": None,
        "top_factors": [],
        "explanation_method": "unavailable",
        "explanation_warning": f"Объяснение недоступно: {type(error).__name__}.",
    }

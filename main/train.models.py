"""Train and compare ML models on ``clean_sba.csv``.

Run after data preparation:
    python train_models.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import FEATURE_COLUMNS


INPUT_FILE = Path("clean_sba.csv")
MODEL_FILE = Path("model.pkl")
RANDOM_STATE = 42


def evaluate_model(
    name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    """Fit one model and return standard binary-classification metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


def train_models(input_file: Path = INPUT_FILE, model_file: Path = MODEL_FILE) -> pd.DataFrame:
    """Split prepared data, train three models and save the best by ROC-AUC."""
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Не найден {input_file}. Сначала выполните: python prepare_data.py"
        )

    data = pd.read_csv(input_file)
    missing = sorted(set(FEATURE_COLUMNS + ["is_default"]) - set(data.columns))
    if missing:
        raise ValueError("В clean_sba.csv не хватает столбцов: " + ", ".join(missing))

    x = data[FEATURE_COLUMNS]
    y = data["is_default"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "Для XGBoost установите зависимость командой: pip install xgboost"
        ) from error

    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted_models = {}
    for name, model in models.items():
        print(f"Обучается: {name}...")
        results.append(evaluate_model(name, model, x_train, y_train, x_test, y_test))
        fitted_models[name] = model

    comparison = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    with model_file.open("wb") as file:
        pickle.dump(
            {"model_name": best_name, "model": best_model, "features": FEATURE_COLUMNS},
            file,
        )
    print(f"\nЛучшая модель по ROC-AUC: {best_name}")
    print(f"Сохранена в: {model_file}")
    return comparison


if __name__ == "__main__":
    train_models()

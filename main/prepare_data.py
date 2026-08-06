"""Prepare the SBA National dataset for the ClearScore ML experiment.

Run from the project directory:
    python prepare_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("SBAnational.csv")
OUTPUT_FILE = Path("clean_sba.csv")

# These are available at loan approval time. Outcome/post-default columns such
# as BalanceGross and ChgOffPrinGr are deliberately not training features.
FEATURE_COLUMNS = [
    "Term",
    "NoEmp",
    "NewExist",
    "CreateJob",
    "RetainedJob",
    "FranchiseCode",
    "UrbanRural",
    "RevLineCr",
    "LowDoc",
    "NAICS",
    "ApprovalFY",
    "DisbursementGross",
    "GrAppv",
    "SBA_Appv",
]
MONEY_COLUMNS = [
    "DisbursementGross",
    "BalanceGross",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv",
]


def clean_money(values: pd.Series) -> pd.Series:
    """Convert SBA's dollar-formatted amounts such as '$12,500.00' to floats."""
    cleaned = (
        values.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def encode_yes_no(values: pd.Series) -> pd.Series:
    """Encode SBA's Y/N fields; unknown and missing values become 0 for MVP."""
    normalized = values.astype("string").str.strip().str.upper()
    return normalized.map({"Y": 1, "N": 0}).fillna(0).astype("int8")


def create_target(statuses: pd.Series) -> pd.Series:
    """Create is_default: ChgOff/CHGOFF is 1, paid-in-full variants are 0."""
    normalized = (
        statuses.astype("string")
        .str.upper()
        .str.replace(r"[^A-Z]", "", regex=True)
    )
    target = pd.Series(pd.NA, index=statuses.index, dtype="Int8")
    target.loc[normalized.isin({"CHGOFF", "CHARGEDOFF"})] = 1
    # SBA commonly stores a successfully repaid loan as "P I F". "PID" is
    # also accepted to match the project specification.
    target.loc[normalized.isin({"PIF", "PID", "PAIDINFULL", "PAID"})] = 0
    return target


def prepare_data(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE) -> pd.DataFrame:
    """Read SBA data, clean fields, create target and save a model-ready CSV."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Не найден исходный файл: {input_file}")

    raw = pd.read_csv(input_file, low_memory=False)
    required = set(FEATURE_COLUMNS + MONEY_COLUMNS + ["MIS_Status"])
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("В SBA-файле не хватает столбцов: " + ", ".join(missing))

    for column in MONEY_COLUMNS:
        raw[column] = clean_money(raw[column])

    raw["is_default"] = create_target(raw["MIS_Status"])
    clean = raw.loc[raw["is_default"].notna(), FEATURE_COLUMNS + ["is_default"]].copy()
    clean["is_default"] = clean["is_default"].astype("int8")

    clean["RevLineCr"] = encode_yes_no(clean["RevLineCr"])
    clean["LowDoc"] = encode_yes_no(clean["LowDoc"])
    clean["UrbanRural"] = pd.to_numeric(clean["UrbanRural"], errors="coerce")
    clean["ApprovalFY"] = pd.to_numeric(
        clean["ApprovalFY"].astype("string").str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )

    for column in clean.columns:
        if column == "is_default":
            continue
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = clean[column].fillna(clean[column].median())

    # A column consisting entirely of missing values has no median.
    clean = clean.fillna(0)
    clean.to_csv(output_file, index=False)

    print(f"Готово: {output_file} сохранён.")
    print(f"Строк для обучения: {len(clean):,}".replace(",", " "))
    print(f"Доля дефолтов: {clean['is_default'].mean():.2%}")
    return clean


if __name__ == "__main__":
    prepare_data()

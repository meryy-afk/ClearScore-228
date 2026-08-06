"""Loading and normalising daily financial data for the ClearScore SME MVP.

The MVP accepts a CSV where each row represents one calendar day.  The loader
does not persist the source file or its contents: it only returns an in-memory
``pandas.DataFrame`` for the calculation modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = ("date", "revenue", "expenses", "cash_balance")


class DataValidationError(ValueError):
    """Raised when an uploaded CSV cannot be used in the scoring pipeline."""


CsvSource = Union[str, Path]


def load_csv(file_path: CsvSource) -> pd.DataFrame:
    """Load, validate and normalise a daily-finance CSV file.

    Required columns are ``date``, ``revenue``, ``expenses`` and
    ``cash_balance``. Dates are converted to ``datetime64`` and sorted.
    Empty revenue/expense cells and absent calendar days are filled with zero.
    ``cash_balance`` is carried forward from the last known balance, because a
    missing reporting day does not mean the business had no cash.

    Args:
        file_path: Path to a UTF-8 (or UTF-8 with BOM) CSV file.

    Returns:
        A DataFrame with exactly the required columns, one row per calendar
        day, sorted from oldest to newest.

    Raises:
        DataValidationError: If the file is missing, unreadable, malformed or
            contains invalid/duplicate daily records.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DataValidationError(f"CSV-файл не найден: {path}")
    if path.suffix.lower() != ".csv":
        raise DataValidationError("Для MVP поддерживаются только файлы .csv.")

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise DataValidationError(
            "Не удалось прочитать CSV в UTF-8. Сохраните файл в кодировке UTF-8."
        ) from error
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DataValidationError(f"Не удалось прочитать CSV-файл: {error}") from error

    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise DataValidationError(
            "В CSV отсутствуют обязательные столбцы: "
            + ", ".join(missing_columns)
            + "."
        )

    data = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    if data.empty:
        raise DataValidationError("CSV-файл не содержит строк с финансовыми данными.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    invalid_dates = data["date"].isna()
    if invalid_dates.any():
        rows = _row_numbers(invalid_dates)
        raise DataValidationError(
            f"Некорректные или пустые даты в строках CSV: {rows}."
        )
    data["date"] = data["date"].dt.normalize()

    if data["date"].duplicated().any():
        duplicate_dates = data.loc[data["date"].duplicated(keep=False), "date"]
        dates = ", ".join(duplicate_dates.dt.strftime("%Y-%m-%d").unique())
        raise DataValidationError(
            "В CSV должна быть только одна строка на дату. Повторяющиеся даты: "
            f"{dates}."
        )

    for column in REQUIRED_COLUMNS[1:]:
        data[column] = _to_number(data[column])
        if data[column].isna().any():
            # Missing values are zero in this MVP; non-numeric text is not.
            original = frame.loc[data[column].isna(), column]
            invalid_text = original.notna() & original.astype(str).str.strip().ne("")
            if invalid_text.any():
                rows = _row_numbers(invalid_text)
                raise DataValidationError(
                    f"Столбец '{column}' содержит нечисловые значения в строках CSV: {rows}."
                )
        if column in ("revenue", "expenses"):
            data[column] = data[column].fillna(0.0)

    for column in ("revenue", "expenses"):
        if (data[column] < 0).any():
            rows = _row_numbers(data[column] < 0)
            raise DataValidationError(
                f"Столбец '{column}' не может содержать отрицательные значения "
                f"(строки CSV: {rows})."
            )

    data = data.sort_values("date").set_index("date")
    full_period = pd.date_range(data.index.min(), data.index.max(), freq="D")
    data = data.reindex(full_period).rename_axis("date")
    data["revenue"] = data["revenue"].fillna(0.0)
    data["expenses"] = data["expenses"].fillna(0.0)
    data["cash_balance"] = data["cash_balance"].ffill().bfill().fillna(0.0)
    data = data.reset_index()

    return data.astype({"revenue": "float64", "expenses": "float64", "cash_balance": "float64"})


def _to_number(values: pd.Series) -> pd.Series:
    """Convert common CSV number formats (spaces and decimal commas) to float."""
    cleaned = (
        values.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _row_numbers(mask: pd.Series) -> str:
    """Return human-friendly CSV row numbers (including the header row)."""
    return ", ".join(str(index + 2) for index in mask[mask].index)

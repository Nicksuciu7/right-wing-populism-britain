# shared cleaning helpers for demographics and attitudes
# standardises ids, weights, recodes, and summary rows

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "stage",
    "year",
    "output_name",
    "source_var",
    "rows_total",
    "non_missing_count",
    "non_missing_rate",
]


def build_respondent_ids(values: pd.Series, year: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").round().astype("Int64").astype("string")
    return numeric.where(numeric != "<NA>").map(
        lambda value: f"{year}_{value}" if pd.notna(value) else pd.NA
    )


def clean_numeric(
    values: pd.Series,
    *,
    missing_codes: list[int] | tuple[int, ...] | set[int] = (),
    extra_missing_codes: list[int] | tuple[int, ...] | set[int] = (),
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
    missing = {
        float(code)
        for code in list(missing_codes) + list(extra_missing_codes)
        if code is not None
    }
    if missing:
        numeric = numeric.mask(numeric.isin(missing))
    if min_value is not None:
        numeric = numeric.where(numeric >= float(min_value))
    if max_value is not None:
        numeric = numeric.where(numeric <= float(max_value))
    return numeric


def clean_weight(values: pd.Series, missing_codes: list[int]) -> pd.Series:
    weight = clean_numeric(values, missing_codes=missing_codes)
    return weight.where(weight > 0)


def recode_categorical(
    values: pd.Series,
    *,
    mapping: dict[int, str],
    missing_codes: list[int] | tuple[int, ...] | set[int] = (),
    extra_missing_codes: list[int] | tuple[int, ...] | set[int] = (),
) -> pd.Series:
    numeric = clean_numeric(
        values,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing_codes,
    )
    return numeric.map({float(key): value for key, value in mapping.items()})


def rescale_attitude(values: pd.Series, *, spec: dict, missing_codes: list[int]) -> pd.Series:
    numeric = clean_numeric(
        values,
        missing_codes=missing_codes,
        extra_missing_codes=spec.get("exclude_codes", []),
        min_value=spec.get("min"),
        max_value=spec.get("max"),
    )
    min_value = float(spec["min"])
    max_value = float(spec["max"])
    span = max_value - min_value
    if span <= 0:
        return pd.Series(index=numeric.index, dtype=float)
    scaled = (numeric - min_value) / span
    if str(spec.get("direction", "normal")).strip().lower() == "reverse":
        scaled = 1.0 - scaled
    return scaled.astype(float)


def make_summary_row(
    *,
    stage: str,
    year: int,
    output_name: str,
    source_var: str,
    values: pd.Series,
) -> dict[str, object]:
    total = int(len(values))
    non_missing = int(values.notna().sum())
    rate = float(non_missing / total) if total else 0.0
    return {
        "stage": stage,
        "year": int(year),
        "output_name": output_name,
        "source_var": source_var,
        "rows_total": total,
        "non_missing_count": non_missing,
        "non_missing_rate": rate,
    }


def write_stage_summary(
    summary_path: str | Path,
    stage: str,
    rows: list[dict[str, object]],
) -> None:
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    current = pd.DataFrame(columns=SUMMARY_COLUMNS)
    if summary_file.exists():
        current = pd.read_csv(summary_file)
        if "stage" in current.columns:
            current = current[current["stage"] != stage].copy()

    stage_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    combined = stage_df if current.empty else pd.concat([current, stage_df], ignore_index=True)
    combined.to_csv(summary_file, index=False)

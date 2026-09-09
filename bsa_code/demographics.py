# harmonises demographic variables for each wave
# reads raw files and writes demographic parquets

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .harmonisation_utils import (
    build_respondent_ids,
    clean_numeric,
    make_summary_row,
    recode_categorical,
    write_stage_summary,
)
from .logging_utils import log_event, setup_logging


def harmonise_demographics(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    derived_dir = Path(config["paths"]["derived_dir"])
    raw_dir = derived_dir / "raw"
    output_dir = derived_dir / "demographics"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(derived_dir / "logs" / "demographics.log")

    summary_rows: list[dict[str, object]] = []
    categories = config["demographics"]["categories"]

    for wave in config["waves"]:
        year = int(wave["year"])
        frame = pd.read_parquet(raw_dir / f"wave_{year}.parquet")
        missing_codes = list(wave.get("missing_codes", []))
        demo_map = wave.get("demographics_map", {})

        result = pd.DataFrame(
            {
                "respondent_id": build_respondent_ids(frame[wave["id_var"]], year),
                "wave_year": int(year),
                "year": int(year),
            }
        )

        for target, spec in demo_map.items():
            # Optional, explicitly labelled post-submission corrections. The
            # submitted configuration continues to use its original recodes.
            spec = dict(spec) if isinstance(spec, dict) else {"source": spec}
            spec.update(config.get("demographic_overrides", {}).get(str(year), {}).get(target, {}))
            source = spec.get("source") if isinstance(spec, dict) else spec
            series = frame[source]

            if target == "age":
                values = _harmonise_age(series, year=year, source=source, missing_codes=spec.get("missing_codes", missing_codes))
            elif target == "gender":
                values = _harmonise_gender(series, year=year, source=source, missing_codes=missing_codes)
            elif target == "education":
                values = _harmonise_education(series, year=year, missing_codes=missing_codes, mapping_overrides=spec.get("mapping_overrides"))
            elif target == "income":
                values = recode_categorical(
                    series,
                    mapping={1: "q1", 2: "q2", 3: "q3", 4: "q4"},
                    missing_codes=missing_codes,
                    extra_missing_codes=[7, 8, 9, 97, 98, 99],
                )
            elif target == "class":
                values = _harmonise_class(series, year=year, source=source, missing_codes=missing_codes)
            elif target == "region":
                values = _harmonise_region(series, year=year, source=source, missing_codes=missing_codes)
            elif target == "religion":
                values = _harmonise_religion(series, source=source, missing_codes=missing_codes)
            elif target == "religiosity":
                religion_values = result["religion"] if "religion" in result.columns else None
                values = _harmonise_religiosity(
                    series,
                    missing_codes=missing_codes,
                    religion_values=religion_values,
                )
            elif target == "ethnicity":
                values = _harmonise_ethnicity(series, source=source, missing_codes=missing_codes)
            elif target == "tenure":
                values = _harmonise_tenure(series, year=year, source=source, missing_codes=missing_codes)
            elif target == "employment_status":
                values = _harmonise_employment(series, year=year, source=source, missing_codes=missing_codes)
            elif target == "marital_status":
                values = _harmonise_marital_status(series, source=source, missing_codes=missing_codes)
            else:
                continue

            if target in categories:
                values = pd.Categorical(values, categories=categories[target])
            result[target] = values
            summary_rows.append(
                make_summary_row(
                    stage="demographics",
                    year=year,
                    output_name=target,
                    source_var=str(source),
                    values=pd.Series(values),
                )
            )

        output_path = output_dir / f"demographics_{year}.parquet"
        result.to_parquet(output_path, index=False)
        log_event(
            logger,
            "demographics_wave_complete",
            wave_year=year,
            rows=int(len(result)),
            output=str(output_path),
        )

    write_stage_summary(config["harmonisation_qc"]["summary_path"], "demographics", summary_rows)
    log_event(logger, "demographics_complete", output=str(output_dir))


def _harmonise_gender(
    values: pd.Series,
    *,
    year: int,
    source: str,
    missing_codes: list[int],
) -> pd.Series:
    source_key = str(source).strip().lower()
    if source_key.startswith("dvsex"):
        mapping = {1: "female", 2: "male"}
    else:
        mapping = {1: "male", 2: "female"}
    return recode_categorical(values, mapping=mapping, missing_codes=missing_codes, extra_missing_codes=[8, 9])


def _harmonise_age(
    values: pd.Series,
    *,
    year: int,
    source: str,
    missing_codes: list[int],
) -> pd.Series:
    age = clean_numeric(
        values,
        missing_codes=missing_codes,
        min_value=17,
        max_value=120,
    )
    source_key = str(source).strip().lower()
    if source_key == "respage_archive":
        age = age.copy()
        lower_band_midpoint = 19.0 if int(year) == 2024 else 20.0
        age = age.mask(age.eq(22), lower_band_midpoint)
        age = age.mask(age.eq(80), 85.0)
    return age


def _harmonise_education(values: pd.Series, *, year: int, missing_codes: list[int], mapping_overrides: dict | None = None) -> pd.Series:
    """Recode qualifications; explicit overrides preserve the submitted profile."""
    if year >= 2023:
        mapping = {
            1: "high",
            2: "mid",
            3: "mid",
            4: "low",
            5: "low",
            6: "low",
        }
        extra_missing = [8, 9]
    else:
        mapping = {
            1: "high",
            2: "high",
            3: "high",
            4: "mid",
            5: "low",
            6: "low",
            8: "low",
        }
        extra_missing = [7, 9]
    if mapping_overrides:
        mapping.update({int(code): label for code, label in mapping_overrides.items()})
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )


def _harmonise_class(
    values: pd.Series,
    *,
    year: int,
    source: str,
    missing_codes: list[int],
) -> pd.Series:
    mapping = {
        1: "high",
        2: "mid",
        3: "mid",
        4: "mid",
        5: "low",
    }
    if year >= 2023:
        extra_missing = [8]
    else:
        mapping[8] = "other"
        extra_missing = []
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )


def _harmonise_region(values: pd.Series, *, year: int, source: str, missing_codes: list[int]) -> pd.Series:
    source_key = str(source).strip().lower()
    if source_key in {"gor2", "revised_gor2"}:
        mapping = {**{code: "England" for code in range(1, 11)}, 11: "Wales", 12: "Scotland"}
    elif source_key == "gor" and int(year) == 2024:
        mapping = {
            **{code: "England" for code in (1, 2, 3, 4, 5, 8, 9, 11, 12)},
            6: "Northern Ireland",
            7: "Scotland",
            10: "Wales",
        }
    else:
        mapping = {
            **{code: "England" for code in range(1, 10)},
            10: "Wales",
            11: "Scotland",
            12: "Northern Ireland",
            13: "Other",
            14: "Other",
        }
    return recode_categorical(values, mapping=mapping, missing_codes=missing_codes)


def _harmonise_religion(values: pd.Series, *, source: str, missing_codes: list[int]) -> pd.Series:
    source_key = str(source).strip().lower()
    if source_key == "religsum20":
        mapping = {
            1: "christian",
            2: "christian",
            3: "christian",
            4: "non_christian",
            5: "none",
            6: "non_christian",
        }
        extra_missing = [8, 9, 98, 99]
    else:
        mapping = {
            1: "christian",
            2: "christian",
            3: "christian",
            4: "non_christian",
            5: "none",
        }
        extra_missing = [6, 8, 9, 98, 99]
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )


def _harmonise_ethnicity(values: pd.Series, *, source: str, missing_codes: list[int]) -> pd.Series:
    source_key = str(source).strip().lower()
    if source_key == "raceori3":
        mapping = {
            1: "black",
            2: "black",
            3: "black",
            4: "asian",
            5: "asian",
            6: "asian",
            7: "asian",
            8: "asian",
            9: "white",
            10: "mixed",
            11: "other",
        }
        extra_missing = [98, 99]
    else:
        mapping = {1: "black", 2: "asian", 3: "white", 4: "mixed", 5: "other"}
        extra_missing = [8, 9]
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )


def _harmonise_tenure(
    values: pd.Series,
    *,
    year: int,
    source: str,
    missing_codes: list[int],
) -> pd.Series:
    source_key = str(source).strip().lower()
    if "tenhhl" in source_key or "archive" in source_key or source_key == "tenuree":
        mapping = {
            1: "own",
            2: "own",
            3: "own",
            4: "rent",
            5: "rent",
            6: "rent",
            7: "rent",
            8: "rent",
            9: "rent",
            10: "rent",
            11: "other",
            12: "other",
            13: "other",
        }
        extra_missing = [98, 99]
    else:
        mapping = {
            1: "own",
            2: "rent",
            3: "rent",
            4: "rent",
            5: "other",
        }
        extra_missing = [9, 98, 99]
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )


def _harmonise_employment(
    values: pd.Series,
    *,
    year: int,
    source: str,
    missing_codes: list[int],
) -> pd.Series:
    source_key = str(source).strip().lower()
    if source_key == "reconact":
        mapping = {
            1: "student",
            2: "student",
            3: "employed",
            4: "employed",
            5: "employed",
            6: "unemployed",
            7: "inactive",
            8: "retired",
            9: "inactive",
            10: "inactive",
        }
        extra_missing = [98, 99]
    elif source_key == "reconsum20":
        mapping = {
            1: "student",
            2: "employed",
            3: "employed",
            4: "employed",
            5: "unemployed",
            6: "retired",
            7: "inactive",
        }
        extra_missing = [8, 9, 98, 99]
    elif year >= 2022:
        mapping = {
            1: "student",
            2: "employed",
            3: "employed",
            4: "unemployed",
            5: "retired",
            6: "inactive",
            7: "inactive",
        }
        extra_missing = [8, 9, 98, 99]
    else:
        mapping = {
            1: "student",
            2: "employed",
            3: "unemployed",
            4: "retired",
            5: "inactive",
        }
        extra_missing = [8, 9, 98, 99]
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )


def _harmonise_religiosity(
    values: pd.Series,
    *,
    missing_codes: list[int],
    religion_values: pd.Series | None,
) -> pd.Series:
    output = recode_categorical(
        values,
        mapping={
            1: "weekly_or_more",
            2: "monthly",
            3: "monthly",
            4: "yearly_or_less",
            5: "yearly_or_less",
            6: "yearly_or_less",
            7: "never",
        },
        missing_codes=missing_codes,
        extra_missing_codes=[-1, 8, 98, 99],
    )
    if religion_values is not None:
        religion = pd.Series(religion_values, index=output.index).astype("string")
        no_religion_skipped = output.isna() & religion.eq("none")
        output = output.mask(no_religion_skipped, "never")
    return output


def _harmonise_marital_status(
    values: pd.Series,
    *,
    source: str,
    missing_codes: list[int],
) -> pd.Series:
    source_key = str(source).strip().lower()
    if source_key == "marstat":
        mapping = {1: "partnered", 2: "partnered", 3: "separated", 4: "widowed", 5: "single"}
        extra_missing = [8, 9]
    elif source_key in {"marstat6", "marstat6_archive"}:
        mapping = {
            1: "partnered",
            2: "partnered",
            3: "partnered",
            4: "separated",
            5: "separated",
            6: "widowed",
            7: "single",
        }
        extra_missing = [8, 9]
    else:
        mapping = {1: "partnered", 2: "separated", 3: "widowed", 4: "single", 5: "other"}
        extra_missing = [8, 9]
    return recode_categorical(
        values,
        mapping=mapping,
        missing_codes=missing_codes,
        extra_missing_codes=extra_missing,
    )

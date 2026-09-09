# config sanity check/ error check

import json
import warnings
from pathlib import Path

import pandas as pd
import pyreadstat
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from .attitude_map import load_attitude_map
from .config import load_config
from .output_layout import (
    build_diagnostics_dir,
    build_model_frame_meta_path,
    build_model_frame_parquet_path,
    measurement_figures_dir,
    measurement_tables_dir,
    modelling_figures_dir,
    modelling_tables_dir,
    trend_figures_dir,
    trend_tables_dir,
)


REQUIRED_TOP_LEVEL = ["project", "paths", "waves", "demographics", "scores", "thresholds"]
REQUIRED_WAVE_KEYS = ["year", "file", "file_type", "weight_var", "id_var", "missing_codes", "demographics_map"]


def _read_columns(wave: dict) -> list[str]:
    file_path = Path(wave["file"])
    file_type = wave["file_type"].lower()
    if file_type == "sav":
        meta = pyreadstat.read_sav(file_path, metadataonly=True)[1]
        return meta.column_names
    if file_type == "tab":
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip("\n")
        return header.split("\t")
    raise ValueError(f"Unsupported file_type: {file_type}")


def _read_key_vars(wave: dict, columns: list[str]) -> pd.DataFrame:
    file_path = Path(wave["file"])
    file_type = wave["file_type"].lower()
    if file_type == "sav":
        return pyreadstat.read_sav(
            file_path,
            usecols=columns,
            apply_value_formats=False,
            formats_as_category=False,
            user_missing=False,
        )[0]
    if file_type == "tab":
        return pd.read_csv(file_path, sep="\t", usecols=columns, low_memory=False)
    raise ValueError(f"Unsupported file_type: {file_type}")


def _validate_key_var_usability(
    wave: dict,
    id_var: str,
    weight_var: str,
    *,
    id_duplicate_policy: str = "warn",
    id_duplicate_max_rate: float = 0.0,
) -> dict:
    year = wave.get("year")
    df_key = _read_key_vars(wave, [id_var, weight_var])
    if df_key.empty:
        raise ValueError(f"Wave {year} appears to contain no rows for key-variable validation")

    if id_var not in df_key.columns:
        raise KeyError(f"Wave {year} key-variable read missing id_var: {id_var}")
    if weight_var not in df_key.columns:
        raise KeyError(f"Wave {year} key-variable read missing weight_var: {weight_var}")

    id_non_missing = int(df_key[id_var].notna().sum())
    if id_non_missing == 0:
        raise ValueError(f"Wave {year} id_var '{id_var}' exists but is all missing")
    id_series = df_key[id_var].dropna()
    dup_count = int(id_series.duplicated().sum())
    dup_rate = float(dup_count / len(id_series)) if len(id_series) else 0.0
    dup_status = "ok"
    if dup_count > 0 and not id_series.empty:
        if dup_rate > float(id_duplicate_max_rate):
            msg = (
                f"Wave {year} id_var '{id_var}' has duplicate values "
                f"(duplicate_count={dup_count}, duplicate_rate={dup_rate:.6f}, "
                f"max_allowed_rate={float(id_duplicate_max_rate):.6f})"
            )
            if id_duplicate_policy == "fail":
                raise ValueError(msg)
            if id_duplicate_policy == "warn":
                dup_status = "warn"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            elif id_duplicate_policy == "ignore":
                dup_status = "ignored"

    weight_numeric = pd.to_numeric(df_key[weight_var], errors="coerce")
    weight_non_missing = int(weight_numeric.notna().sum())
    if weight_non_missing == 0:
        raise ValueError(
            f"Wave {year} weight_var '{weight_var}' exists but has no parseable numeric values"
        )

    parseable_rate = float(weight_non_missing / len(df_key))
    if parseable_rate < 0.5:
        raise ValueError(
            f"Wave {year} weight_var '{weight_var}' parseable rate is too low ({parseable_rate:.3f})"
        )

    non_missing_weights = weight_numeric.dropna()
    if non_missing_weights.empty:
        raise ValueError(f"Wave {year} weight_var '{weight_var}' has no usable values")
    weight_positive_count = int((non_missing_weights > 0).sum())
    if weight_positive_count == 0:
        raise ValueError(f"Wave {year} weight_var '{weight_var}' has no positive values")

    return {
        "id_var": id_var,
        "id_rows_total": int(len(df_key)),
        "id_non_missing": id_non_missing,
        "id_duplicate_count": dup_count,
        "id_duplicate_rate": dup_rate,
        "id_duplicate_policy": id_duplicate_policy,
        "id_duplicate_max_rate": float(id_duplicate_max_rate),
        "id_duplicate_status": dup_status,
        "weight_var": weight_var,
        "weight_parseable_count": weight_non_missing,
        "weight_parseable_rate": parseable_rate,
        "weight_positive_count": weight_positive_count,
    }


def validate_config(config_path: str | Path) -> dict:
    config = load_config(config_path)
    validation_cfg = config.get("validation", {}) or {}
    if not isinstance(validation_cfg, dict):
        raise ValueError("validation must be an object/dict if provided")
    default_id_duplicate_policy = str(
        validation_cfg.get("id_duplicate_policy", "warn")
    ).lower()
    if default_id_duplicate_policy not in {"warn", "fail", "ignore"}:
        raise ValueError(
            "validation.id_duplicate_policy must be one of: warn, fail, ignore"
        )
    default_id_duplicate_max_rate = validation_cfg.get("id_duplicate_max_rate", 0.0)
    try:
        default_id_duplicate_max_rate = float(default_id_duplicate_max_rate)
    except (TypeError, ValueError):
        raise ValueError("validation.id_duplicate_max_rate must be numeric")
    if not (0.0 <= default_id_duplicate_max_rate <= 1.0):
        raise ValueError("validation.id_duplicate_max_rate must be between 0 and 1")

    validation_summary = {
        "config_path": str(config_path),
        "project_seed": config.get("project", {}).get("seed"),
        "id_duplicate_policy_default": default_id_duplicate_policy,
        "id_duplicate_max_rate_default": float(default_id_duplicate_max_rate),
        "waves": {},
    }

    missing_top = [key for key in REQUIRED_TOP_LEVEL if key not in config]
    if missing_top:
        raise KeyError(f"Missing top-level config keys: {missing_top}")

    project = config.get("project", {})
    if "seed" not in project:
        raise KeyError("project.seed missing in config")
    seed = project.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("project.seed must be an int")

    expected_years = project.get("expected_years")
    if expected_years is not None:
        if not isinstance(expected_years, list) or not all(
            isinstance(y, int) and not isinstance(y, bool) for y in expected_years
        ):
            raise ValueError("project.expected_years must be a list of ints")
        dup_expected = {y for y in expected_years if expected_years.count(y) > 1}
        if dup_expected:
            raise ValueError(f"Duplicate years in project.expected_years: {sorted(dup_expected)}")

    waves = config.get("waves", [])
    if not waves:
        raise ValueError("Config has no waves defined")

    years = [w.get("year") for w in waves]
    dup_years = {y for y in years if years.count(y) > 1}
    if dup_years:
        raise ValueError(f"Duplicate wave years in config: {sorted(dup_years)}")

    if expected_years:
        if sorted(expected_years) != sorted(years):
            raise ValueError(
                f"Wave years do not match expected_years: {sorted(years)} vs {sorted(expected_years)}"
            )

    paths = config.get("paths", {})
    for key in ["derived_dir", "outputs_dir"]:
        val = paths.get(key)
        if not isinstance(val, str) or not val.strip():
            raise KeyError(f"paths.{key} missing or not a string")
    attitude_map_path = paths.get("attitude_map")
    if not isinstance(attitude_map_path, str) or not attitude_map_path.strip():
        raise KeyError("paths.attitude_map missing or not a string in config")
    attitude_path = Path(attitude_map_path)
    if not attitude_path.exists():
        raise FileNotFoundError(f"Attitude map path not found: {attitude_path}")

    attitude_map = load_attitude_map(attitude_path)
    modelling = config.get("modelling", {}) or {}
    if not isinstance(modelling, dict):
        raise ValueError("modelling must be an object/dict if provided")
    ridge_alpha = modelling.get("ridge_alpha")
    if ridge_alpha is not None:
        try:
            ridge_alpha = float(ridge_alpha)
        except (TypeError, ValueError):
            raise ValueError("modelling.ridge_alpha must be numeric")
        if ridge_alpha <= 0:
            raise ValueError("modelling.ridge_alpha must be greater than 0")

    for wave in waves:
        for key in REQUIRED_WAVE_KEYS:
            if key not in wave:
                raise KeyError(f"Wave {wave.get('year')} missing key: {key}")

        file_type = wave.get("file_type")
        if not isinstance(file_type, str):
            raise ValueError(f"Wave {wave.get('year')} file_type must be a string")
        file_type = file_type.lower()
        if file_type not in {"sav", "tab"}:
            raise ValueError(f"Unsupported file_type for wave {wave.get('year')}: {file_type}")

        file_path = Path(wave["file"])
        if not file_path.exists():
            raise FileNotFoundError(f"Missing wave file: {file_path}")

        if file_type == "tab":
            dictionary = wave.get("dictionary")
            if dictionary:
                dict_path = Path(dictionary)
                if not dict_path.exists():
                    raise FileNotFoundError(f"Missing dictionary file: {dict_path}")

        columns = _read_columns(wave)

        weight_var = wave.get("weight_var")
        if not isinstance(weight_var, str) or not weight_var:
            raise KeyError(f"Wave {wave.get('year')} weight_var missing or not a string")
        if weight_var not in columns:
            raise KeyError(f"Weight var not found in wave {wave.get('year')}: {weight_var}")

        id_var = wave.get("id_var")
        if not isinstance(id_var, str) or not id_var:
            raise KeyError(f"Wave {wave.get('year')} id_var missing or not a string")
        if id_var not in columns:
            raise KeyError(f"ID var not found in wave {wave.get('year')}: {id_var}")



        id_duplicate_policy = str(
            wave.get("id_duplicate_policy", default_id_duplicate_policy)
        ).lower()
        if id_duplicate_policy not in {"warn", "fail", "ignore"}:
            raise ValueError(
                f"Wave {wave.get('year')} id_duplicate_policy must be one of: warn, fail, ignore"
            )
        id_duplicate_max_rate = wave.get(
            "id_duplicate_max_rate", default_id_duplicate_max_rate
        )
        try:
            id_duplicate_max_rate = float(id_duplicate_max_rate)
        except (TypeError, ValueError):
            raise ValueError(
                f"Wave {wave.get('year')} id_duplicate_max_rate must be numeric"
            )
        if not (0.0 <= id_duplicate_max_rate <= 1.0):
            raise ValueError(
                f"Wave {wave.get('year')} id_duplicate_max_rate must be between 0 and 1"
            )
        key_check = _validate_key_var_usability(
            wave,
            id_var=id_var,
            weight_var=weight_var,
            id_duplicate_policy=id_duplicate_policy,
            id_duplicate_max_rate=id_duplicate_max_rate,
        )
        validation_summary["waves"][str(wave.get("year"))] = key_check

        demo_map = wave.get("demographics_map", {})
        if not isinstance(demo_map, dict):
            raise ValueError(f"Wave {wave.get('year')} demographics_map must be a dict")
        for _, spec in demo_map.items():
            source = spec.get("source") if isinstance(spec, dict) else spec
            if source and source not in columns:
                raise KeyError(
                    f"Demographic source var not found in wave {wave.get('year')}: {source}"
                )


        constructs = attitude_map.get("constructs", {})
        for _, construct in constructs.items():
            for _, item in construct.get("items", {}).items():
                wave_spec = item.get("waves", {}).get(str(wave.get("year")))
                if not wave_spec:
                    continue
                var = wave_spec.get("var")
                if var and var not in columns:
                    raise KeyError(
                        f"Attitude var not found in wave {wave.get('year')}: {var}"
                    )

    return validation_summary


def _check_columns(name: str, df: pd.DataFrame, expected: list[str]) -> list[str]:
    errors = []
    missing = [col for col in expected if col not in df.columns]
    extra = [col for col in df.columns if col not in expected]
    if missing:
        errors.append(f"{name} missing columns: {missing}")
    if extra:
        errors.append(f"{name} extra columns: {extra}")
    return errors


def validate_schema_outputs(config_path: str | Path, strict: bool = True) -> list[str]:
    config = load_config(config_path)
    derived_dir = Path(config["paths"]["derived_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    errors: list[str] = []
    trend_cfg = config.get("trend_analysis", {}) or {}
    trends_enabled = bool(trend_cfg.get("enabled", True))
    measurement_cfg = config.get("measurement", {}) or {}
    measurement_blocks = measurement_cfg.get("blocks", {}) or {}
    measurement_constructs = [
        str(name)
        for name, blocks in measurement_blocks.items()
        if isinstance(blocks, dict) and blocks
    ]

    main_expected = [
        measurement_tables_dir(outputs_dir) / "measurement_items_final.csv",
        measurement_tables_dir(outputs_dir) / "measurement_pca_factor_summary.csv",
        measurement_figures_dir(outputs_dir) / "measurement_component_summary_table.png",
        measurement_figures_dir(outputs_dir) / "measurement_component_questions_table.png",
        modelling_tables_dir(outputs_dir) / "model_comparison.csv",
        modelling_tables_dir(outputs_dir) / "regression_metrics.csv",
        modelling_figures_dir(outputs_dir) / "predicted_vs_actual.png",
        modelling_figures_dir(outputs_dir) / "regression_metrics_table.png",
    ]
    main_expected.extend(
        measurement_figures_dir(outputs_dir) / f"pca_scree_{construct}.png"
        for construct in measurement_constructs
    )
    main_expected.extend(
        measurement_figures_dir(outputs_dir) / f"pca_correlation_matrix_{construct}.png"
        for construct in measurement_constructs
    )
    if trends_enabled:
        main_expected.extend(
            [
                trend_tables_dir(outputs_dir) / "rwp_prevalence_by_year.csv",
                trend_figures_dir(outputs_dir) / "rwp_prevalence_by_year.png",
                trend_tables_dir(outputs_dir) / "predictor_trends.csv",
                trend_figures_dir(outputs_dir) / "predictor_trends.png",
                trend_tables_dir(outputs_dir) / "predictor_trend_model_fit.csv",
                trend_figures_dir(outputs_dir) / "predictor_trend_model_fit_table.png",
                trend_tables_dir(outputs_dir) / "education_by_year_rwp.csv",
                trend_figures_dir(outputs_dir) / "education_by_year_rwp.png",
                trend_tables_dir(outputs_dir) / "employment_status_by_year_rwp.csv",
                trend_figures_dir(outputs_dir) / "employment_status_by_year_rwp.png",
            ]
        )
    for path in main_expected:
        if not path.exists():
            errors.append(f"main output missing: {path}")

    meta_path = build_model_frame_meta_path(derived_dir)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        predictors = meta.get("predictors", [])
        if not predictors:
            track_meta = meta.get("predictor_tracks", {})
            track_predictors = []
            for track in track_meta.values():
                track_predictors.extend(track.get("predictors", []))
            predictors = sorted(set(track_predictors))
    else:
        predictors = config["demographics"]["numeric"] + config["demographics"]["categorical"]
    numeric_predictors = [f for f in config["demographics"]["numeric"] if f in predictors]
    categorical_predictors = [f for f in predictors if f not in numeric_predictors]

    model_path = build_model_frame_parquet_path(derived_dir)
    if not model_path.exists():
        errors.append(f"model_frame missing: {model_path}")
    else:
        model_df = pd.read_parquet(model_path)
        expected = ["respondent_id", "wave_year", "weight", "RWP_top_20", "rwp_score"] + predictors
        errors.extend(_check_columns("model_frame", model_df, expected))

        if "respondent_id" in model_df.columns and not is_string_dtype(model_df["respondent_id"]):
            errors.append("model_frame respondent_id is not string dtype")
        if "wave_year" in model_df.columns and not is_numeric_dtype(model_df["wave_year"]):
            errors.append("model_frame wave_year is not numeric dtype")
        if "weight" in model_df.columns and not is_numeric_dtype(model_df["weight"]):
            errors.append("model_frame weight is not numeric dtype")
        for col in ["RWP_top_20"]:
            if col in model_df.columns and not (
                is_integer_dtype(model_df[col]) or is_bool_dtype(model_df[col])
            ):
                errors.append(f"model_frame {col} is not integer/bool dtype")
        for col in ["rwp_score"]:
            if col in model_df.columns and not is_numeric_dtype(model_df[col]):
                errors.append(f"model_frame {col} is not numeric dtype")

        for col in numeric_predictors:
            if col in model_df.columns and not is_numeric_dtype(model_df[col]):
                errors.append(f"model_frame {col} is not numeric dtype")
        for col in categorical_predictors:
            if col in model_df.columns and not (
                is_categorical_dtype(model_df[col]) or is_string_dtype(model_df[col])
            ):
                errors.append(f"model_frame {col} is not categorical/string dtype")

    scores_path = derived_dir / "scores" / "pooled_scored.parquet"
    if not scores_path.exists():
        errors.append(f"pooled_scored missing: {scores_path}")
    else:
        scores_df = pd.read_parquet(scores_path)
        subscale_defs_path = (
            build_diagnostics_dir(derived_dir) / "attitude_subscale_definitions.json"
        )
        subscale_names: list[str] = []
        if subscale_defs_path.exists():
            subscale_defs = json.loads(subscale_defs_path.read_text(encoding="utf-8"))
            for subscales in subscale_defs.values():
                subscale_names.extend(list(subscales.keys()))
        expected_scores = [
            "respondent_id",
            "wave_year",
            "weight",
            "rwp_score",
            "RWP_top_20",
        ] + sorted(set(subscale_names))
        errors.extend(_check_columns("pooled_scored", scores_df, expected_scores))
        for col in [
            "weight",
            "rwp_score",
        ] + subscale_names:
            if col in scores_df.columns and not is_numeric_dtype(scores_df[col]):
                errors.append(f"pooled_scored {col} is not numeric dtype")
        for col in ["RWP_top_20"]:
            if col in scores_df.columns and not (
                is_integer_dtype(scores_df[col]) or is_bool_dtype(scores_df[col])
            ):
                errors.append(f"pooled_scored {col} is not integer/bool dtype")
        if "respondent_id" in scores_df.columns and not is_string_dtype(scores_df["respondent_id"]):
            errors.append("pooled_scored respondent_id is not string dtype")
        if "wave_year" in scores_df.columns and not is_numeric_dtype(scores_df["wave_year"]):
            errors.append("pooled_scored wave_year is not numeric dtype")

    if strict and errors:
        raise ValueError("Schema validation failed: " + "; ".join(errors))
    return errors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate code config")
    parser.add_argument(
        "--config",
        default="config/code_config.json",
        help="Path to code config JSON",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate derived schema/dtypes if outputs exist",
    )
    args = parser.parse_args()
    validate_config(args.config)
    if args.validate_schema:
        validate_schema_outputs(args.config)
    print("Config validation passed.")


if __name__ == "__main__":
    main()

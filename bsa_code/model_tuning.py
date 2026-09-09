# runs ridge tuning
# reads the model frame and writes tuning tables

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .config import load_config
from .logging_utils import log_event, setup_logging
from .model_frame import _resolve_target_column
from .model_preprocessing import _filter_features, _resolve_categorical_levels
from .output_layout import build_model_frame_parquet_path
from .provenance import config_sha256, file_sha256
from .train_eval import (
    BINARY_TARGET_COLUMNS,
    MODEL_YEAR_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    WAVE_YEAR_COLUMN,
    _build_regression_specs,
    _build_regression_workflow,
    _evaluate_regression,
    _fit_workflow,
    _parse_settings,
    _primary_train_test_split,
    _resolve_predictors,
    _select_feature_frame,
    _split_holdout,
    _year_stratify_labels,
)

ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
CV_SPLITS = 5
TUNING_GRID_FILENAME = "ridge_tuning_grid.csv"
TUNING_DECISION_FILENAME = "ridge_tuning_decision.json"


def tune_model(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    settings = _parse_settings(config)
    logger = setup_logging(settings.derived_dir / "logs" / "tune_model.log")
    if settings.task != "regression":
        raise ValueError("tune_model requires modelling.task='regression'.")
    if not settings.holdout_years:
        raise ValueError("tune_model requires a configured holdout year.")

    model_frame_path = build_model_frame_parquet_path(settings.derived_dir)
    if not model_frame_path.exists():
        raise FileNotFoundError(f"Model frame not found: {model_frame_path}")

    df = pd.read_parquet(model_frame_path)
    modelling_cfg = config.get("modelling", {}) or {}
    target = _resolve_target_column(df, modelling_cfg, logger, stage="tune_model")
    if target in BINARY_TARGET_COLUMNS:
        raise ValueError(f"Binary target '{target}' is not allowed in tune_model.")

    train_frame, _ = _split_holdout(df, target, settings)
    if train_frame.empty:
        raise ValueError("No non-holdout rows remain after filtering the model frame.")

    predictors_requested, _ = _resolve_predictors(config, settings)
    if not predictors_requested:
        raise ValueError("No predictors are available for the configured modelling track.")

    numeric_features = [
        name for name in config["demographics"]["numeric"] if name in predictors_requested
    ]
    categorical_features = [
        name for name in predictors_requested if name not in numeric_features
    ]

    train_df, test_df, split_fallbacks = _primary_train_test_split(
        train_frame,
        settings,
        logger,
    )
    (
        numeric_features,
        categorical_features,
        missing_rates,
        dropped_missing,
    ) = _filter_features(
        train_df[predictors_requested],
        numeric_features,
        categorical_features,
        settings.max_missing_rate,
    )
    predictors_used = numeric_features + categorical_features
    if not predictors_used:
        raise ValueError("All predictors were dropped by the tuning missingness filter.")
    if MODEL_YEAR_COLUMN not in train_df.columns:
        raise KeyError("year must be present in the model frame for tuning.")
    categorical_levels = _resolve_categorical_levels(
        config.get("demographics", {}).get("categories", {}),
        categorical_features,
    )

    grid_df, cv_strategy = _run_alpha_grid(
        train_df=train_df,
        target=target,
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_levels=categorical_levels,
        project_seed=settings.project_seed,
    )
    selected_row = grid_df.loc[grid_df["selected"]].iloc[0]
    selected_alpha = float(selected_row["alpha"])
    incumbent_alpha = float(settings.ridge_alpha)

    incumbent_eval = _evaluate_alpha(
        alpha=incumbent_alpha,
        train_df=train_df,
        test_df=test_df,
        target=target,
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_levels=categorical_levels,
    )
    candidate_eval = _evaluate_alpha(
        alpha=selected_alpha,
        train_df=train_df,
        test_df=test_df,
        target=target,
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_levels=categorical_levels,
    )

    decision = _build_tuning_decision(
        incumbent_alpha=incumbent_alpha,
        incumbent_eval=incumbent_eval,
        selected_alpha=selected_alpha,
        selected_cv_row=selected_row,
        candidate_eval=candidate_eval,
        holdout_years=sorted(int(year) for year in settings.holdout_years),
        cv_strategy=cv_strategy,
        predictors_requested=predictors_requested,
        predictors_used=predictors_used,
        dropped_missing=dropped_missing,
        missing_rates=missing_rates,
        split_fallbacks=split_fallbacks,
    )
    decision["input_fingerprint"] = {
        "configuration": config_sha256(config),
        "model_frame": file_sha256(model_frame_path),
    }

    grid_path = settings.tables_dir / TUNING_GRID_FILENAME
    grid_df.to_csv(grid_path, index=False)
    decision_path = settings.model_dir / TUNING_DECISION_FILENAME
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    log_event(
        logger,
        "tune_model_complete",
        target_column=target,
        incumbent_alpha=incumbent_alpha,
        selected_alpha=selected_alpha,
        config_updated=bool(decision["selection"]["config_updated"]),
        holdout_years=";".join(str(year) for year in settings.holdout_years),
    )

def _run_alpha_grid(
    *,
    train_df: pd.DataFrame,
    target: str,
    predictors_used: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_levels: list[list[str]] | None,
    project_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(train_df) < CV_SPLITS:
        raise ValueError(
            f"Need at least {CV_SPLITS} training rows to run cross-validation tuning."
        )

    fold_rows: list[dict[str, object]] = []
    splitter, split_strategy = _build_cv_splitter(train_df, project_seed)
    train_reset = train_df.reset_index(drop=True)

    for alpha in ALPHA_GRID:
        for fold_number, (fit_idx, valid_idx) in enumerate(splitter, start=1):
            fit_df = train_reset.iloc[fit_idx].copy()
            valid_df = train_reset.iloc[valid_idx].copy()
            evaluation = _fit_and_evaluate_alpha(
                alpha=alpha,
                fit_df=fit_df,
                eval_df=valid_df,
                target=target,
                predictors_used=predictors_used,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                categorical_levels=categorical_levels,
            )
            weighted = evaluation["metrics_weighted"]
            fold_rows.append(
                {
                    "alpha": float(alpha),
                    "fold": int(fold_number),
                    "weighted_rmse": weighted.get("rmse"),
                    "weighted_mae": weighted.get("mae"),
                    "weighted_r2": weighted.get("r2"),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby("alpha", as_index=False)
        .agg(
            cv_weighted_rmse=("weighted_rmse", "mean"),
            cv_weighted_mae=("weighted_mae", "mean"),
            cv_weighted_r2=("weighted_r2", "mean"),
            cv_folds=("fold", "nunique"),
        )
        .sort_values(
            ["cv_weighted_rmse", "cv_weighted_mae", "alpha"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )
    summary_df["rank"] = np.arange(1, len(summary_df) + 1, dtype=int)
    selected_alpha = float(summary_df.iloc[0]["alpha"])
    summary_df["selected"] = summary_df["alpha"].astype(float).eq(selected_alpha)
    return summary_df, split_strategy


def _build_cv_splitter(
    train_df: pd.DataFrame,
    project_seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    labels = _year_stratify_labels(train_df)
    if labels is not None and int(labels.value_counts().min()) >= CV_SPLITS:
        splitter = StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=project_seed,
        )
        indices = list(splitter.split(train_df, labels))
        return indices, {"name": "StratifiedKFold", "stratified_on": WAVE_YEAR_COLUMN}

    splitter = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=project_seed)
    indices = list(splitter.split(train_df))
    return indices, {"name": "KFold", "stratified_on": None}


def _evaluate_alpha(
    *,
    alpha: float,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    predictors_used: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_levels: list[list[str]] | None,
) -> dict[str, object]:
    spec = _ridge_spec(
        alpha=alpha,
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_levels=categorical_levels,
    )
    model = _build_regression_workflow(spec)
    fit_used_weights = _fit_workflow(
        model,
        _select_feature_frame(train_df, spec.predictors),
        train_df[target].astype(float),
        train_df[SAMPLE_WEIGHT_COLUMN].astype(float),
    )
    test_eval = _evaluate_regression(
        model=model,
        frame=test_df,
        predictors_used=list(spec.predictors),
        target=target,
        split_name="test",
    )
    return {
        "alpha": float(alpha),
        "spec": spec,
        "fit_used_sample_weight": bool(fit_used_weights),
        "test": test_eval,
    }


def _fit_and_evaluate_alpha(
    *,
    alpha: float,
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    predictors_used: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_levels: list[list[str]] | None,
) -> dict[str, object]:
    spec = _ridge_spec(
        alpha=alpha,
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_levels=categorical_levels,
    )
    model = _build_regression_workflow(spec)
    _fit_workflow(
        model,
        _select_feature_frame(fit_df, spec.predictors),
        fit_df[target].astype(float),
        fit_df[SAMPLE_WEIGHT_COLUMN].astype(float),
    )
    return _evaluate_regression(
        model=model,
        frame=eval_df,
        predictors_used=list(spec.predictors),
        target=target,
        split_name="cv_validation",
    )


def _ridge_spec(
    *,
    alpha: float,
    predictors_used: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_levels: list[list[str]] | None,
):
    specs = _build_regression_specs(
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        ridge_alpha=float(alpha),
        categorical_levels=categorical_levels,
    )
    return next(spec for spec in specs if spec.is_main_model)


def _build_tuning_decision(
    *,
    incumbent_alpha: float,
    incumbent_eval: dict[str, object],
    selected_alpha: float,
    selected_cv_row: pd.Series,
    candidate_eval: dict[str, object],
    holdout_years: list[int],
    cv_strategy: dict[str, object],
    predictors_requested: list[str],
    predictors_used: list[str],
    dropped_missing: list[str],
    missing_rates: dict[str, float],
    split_fallbacks: dict[str, object],
) -> dict[str, object]:
    incumbent_test = incumbent_eval["test"]["metrics_weighted"]
    candidate_test = candidate_eval["test"]["metrics_weighted"]
    test_r2_gain = _metric_gain(
        candidate_test.get("r2"),
        incumbent_test.get("r2"),
    )
    config_updated = bool(selected_alpha != incumbent_alpha)
    if config_updated:
        reason = "cv_selected_alpha_differs_from_config"
    else:
        reason = "cv_selected_alpha_matches_config"

    return {
        "candidate_grid": [float(alpha) for alpha in ALPHA_GRID],
        "cv": {
            "splits": CV_SPLITS,
            "strategy": cv_strategy,
            "selection_metric": "mean_weighted_rmse",
            "tie_break_order": ["mean_weighted_mae", "higher_alpha"],
            "selected": {
                "alpha": float(selected_alpha),
                "cv_weighted_rmse": _series_value(selected_cv_row, "cv_weighted_rmse"),
                "cv_weighted_mae": _series_value(selected_cv_row, "cv_weighted_mae"),
                "cv_weighted_r2": _series_value(selected_cv_row, "cv_weighted_r2"),
                "rank": _series_int(selected_cv_row, "rank"),
            },
        },
        "incumbent": {
            "alpha": float(incumbent_alpha),
            "test_weighted": _metrics_payload(incumbent_test),
        },
        "candidate": {
            "alpha": float(selected_alpha),
            "test_weighted": _metrics_payload(candidate_test),
        },
        "comparison": {
            "test_weighted_r2_gain": test_r2_gain,
            "test_weighted_rmse_delta": _metric_delta(
                candidate_test.get("rmse"),
                incumbent_test.get("rmse"),
            ),
            "test_weighted_mae_delta": _metric_delta(
                candidate_test.get("mae"),
                incumbent_test.get("mae"),
            ),
        },
        "selection": {
            "config_updated": False,
            "selected_alpha_differs_from_config": config_updated,
            "handoff": "train_eval reads this decision after validating input fingerprints",
            "reason": reason,
            "holdout_years_reserved_for_final_evaluation": holdout_years,
        },
        "predictors": {
            "requested": predictors_requested,
            "used": predictors_used,
            "dropped_missingness": dropped_missing,
            "missing_rates": missing_rates,
        },
        "split_fallbacks": split_fallbacks,
    }


def _metrics_payload(metrics: dict[str, object]) -> dict[str, float | int | None]:
    return {
        "n": _value_or_none(metrics.get("n")),
        "rmse": _value_or_none(metrics.get("rmse")),
        "mae": _value_or_none(metrics.get("mae")),
        "r2": _value_or_none(metrics.get("r2")),
        "observed_mean": _value_or_none(metrics.get("observed_mean")),
        "predicted_mean": _value_or_none(metrics.get("predicted_mean")),
        "observed_std": _value_or_none(metrics.get("observed_std")),
        "predicted_std": _value_or_none(metrics.get("predicted_std")),
    }


def _series_value(row: pd.Series, key: str) -> float | None:
    return _value_or_none(row.get(key))


def _series_int(row: pd.Series, key: str) -> int | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return int(value)


def _metric_gain(candidate, incumbent) -> float | None:
    candidate_value = _value_or_none(candidate)
    incumbent_value = _value_or_none(incumbent)
    if candidate_value is None or incumbent_value is None:
        return None
    return float(candidate_value - incumbent_value)


def _metric_delta(candidate, incumbent) -> float | None:
    candidate_value = _value_or_none(candidate)
    incumbent_value = _value_or_none(incumbent)
    if candidate_value is None or incumbent_value is None:
        return None
    return float(candidate_value - incumbent_value)


def _value_or_none(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)

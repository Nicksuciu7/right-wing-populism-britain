# trains and evaluates the weighted ridge regression model
# reads the model frame and writes tables, and figures

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import load_config
from .provenance import read_tuned_alpha
from .logging_utils import log_event, setup_logging
from .model_frame import _resolve_target_column, get_predictors_for_track
from .model_preprocessing import (
    _build_preprocessor,
    _filter_features,
    _resolve_categorical_levels,
)
from .output_layout import (
    build_model_frame_meta_path,
    build_model_frame_parquet_path,
    modelling_figures_dir,
    modelling_tables_dir,
)
from .plotting import (
    PALETTE,
    apply_plot_theme,
    finalize_figure,
    format_profile_label,
    style_axes,
    with_profile_prefix,
)

BINARY_TARGET_COLUMNS = {
    "RWP_top_20",
}

MODEL_ORDER = ("mean_baseline", "year_only_baseline", "ridge_regression")
CLEAR_MEAN_BASELINE_GAIN_PCT = 5.0
RESPONDENT_ID_COLUMN = "respondent_id"
WAVE_YEAR_COLUMN = "wave_year"
MODEL_YEAR_COLUMN = "year"
SAMPLE_WEIGHT_COLUMN = "weight"
DEFAULT_RIDGE_ALPHA = 300.0
MANAGED_MODEL_OUTPUTS = (
    "model_comparison.csv",
    "regression_metrics.csv",
    "predicted_vs_actual.png",
    "regression_metrics_table.png",
)


@dataclass
class ModelSettings:
    derived_dir: Path
    outputs_dir: Path
    profile_label: str
    tables_dir: Path
    figures_dir: Path
    model_dir: Path
    holdout_years: set[int]
    task: str
    predictor_track: str
    train_size: float
    stratify_on_year: bool
    max_missing_rate: float
    project_seed: int
    ridge_alpha: float


@dataclass(frozen=True)
class RegressionSpec:
    model: str
    description: str
    is_baseline: bool
    is_main_model: bool
    model_family: str
    implementation: str
    model_params: dict[str, object]
    predictors: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categorical_levels: tuple[tuple[str, ...], ...] | None


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        arr = np.asarray(y, dtype=float)
        mask = np.isfinite(arr)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            mask = mask & np.isfinite(weights) & (weights > 0)
            self.constant_ = (
                float(np.average(arr[mask], weights=weights[mask])) if mask.any() else 0.0
            )
        else:
            self.constant_ = float(np.mean(arr[mask])) if mask.any() else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), getattr(self, "constant_", 0.0), dtype=float)


def train_and_evaluate(config_path: str = "config/code_config.json") -> None:
    apply_plot_theme()
    config = load_config(config_path)
    settings = _parse_settings(config)
    tuned_alpha = read_tuned_alpha(
        config, settings.model_dir / "ridge_tuning_decision.json",
        build_model_frame_parquet_path(settings.derived_dir),
    )
    if tuned_alpha is not None:
        settings = replace(settings, ridge_alpha=tuned_alpha)
    logger = setup_logging(settings.derived_dir / "logs" / "train_eval.log")

    if settings.task != "regression":
        raise ValueError(
            "The dissertation modelling code now expects modelling.task='regression'."
        )

    _clear_previous_outputs(settings)

    df = pd.read_parquet(build_model_frame_parquet_path(settings.derived_dir))
    modelling_cfg = config.get("modelling", {})
    target = _resolve_target_column(df, modelling_cfg, logger, stage="train_eval")
    if target in BINARY_TARGET_COLUMNS:
        raise ValueError(
            f"Binary target '{target}' is not allowed in the default regression path."
        )

    train_frame, holdout_frame = _split_holdout(df, target, settings)
    if train_frame.empty:
        log_event(
            logger,
            "train_eval_skipped",
            reason="no_rows_after_target_weight_filter",
            target_column=target,
        )
        return

    predictors_requested, dropped_structural = _resolve_predictors(config, settings)
    numeric_features = [
        name for name in config["demographics"]["numeric"] if name in predictors_requested
    ]
    categorical_features = [
        name for name in predictors_requested if name not in numeric_features
    ]
    if not predictors_requested:
        log_event(
            logger,
            "train_eval_skipped",
            reason="no_predictors_for_track",
            predictor_track=settings.predictor_track,
        )
        return

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
        log_event(
            logger,
            "train_eval_skipped",
            reason="all_predictors_dropped_by_missingness",
            predictor_track=settings.predictor_track,
        )
        return

    if MODEL_YEAR_COLUMN not in train_df.columns:
        raise KeyError("year must be present in the model frame for year_only_baseline.")

    specs = _build_regression_specs(
        predictors_used=predictors_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        ridge_alpha=settings.ridge_alpha,
        categorical_levels=_resolve_categorical_levels(
            config.get("demographics", {}).get("categories", {}),
            categorical_features,
        ),
    )
    comparison_df = _compare_models(
        specs=specs,
        train_df=train_df,
        test_df=test_df,
        target=target,
        logger=logger,
    )
    interpretation = _build_interpretation(comparison_df)

    main_spec = next(spec for spec in specs if spec.is_main_model)
    main_model = _build_regression_workflow(main_spec)
    main_fit_used_weights = _fit_workflow(
        main_model,
        _select_feature_frame(train_df, main_spec.predictors),
        train_df[target].astype(float),
        train_df[SAMPLE_WEIGHT_COLUMN].astype(float),
    )

    test_eval = _evaluate_regression(
        model=main_model,
        frame=test_df,
        predictors_used=list(main_spec.predictors),
        target=target,
        split_name="test",
    )
    holdout_eval = _evaluate_regression(
        model=main_model,
        frame=holdout_frame,
        predictors_used=list(main_spec.predictors),
        target=target,
        split_name="holdout",
    )

    _write_training_outputs(
        settings=settings,
        model=main_model,
        main_spec=main_spec,
        target=target,
        predictors_requested=predictors_requested,
        predictors_used=predictors_used,
        dropped_structural=dropped_structural,
        dropped_missing=dropped_missing,
        missing_rates=missing_rates,
        train_df=train_df,
        test_df=test_df,
        holdout_df=holdout_frame,
        comparison_df=comparison_df,
        interpretation=interpretation,
        split_fallbacks=split_fallbacks,
        main_fit_used_weights=main_fit_used_weights,
        test_eval=test_eval,
        holdout_eval=holdout_eval,
    )

    metrics_weighted = test_eval["metrics_weighted"]
    log_event(
        logger,
        "train_eval_complete",
        task=settings.task,
        target_column=target,
        predictor_track=settings.predictor_track,
        main_model=main_spec.model,
        weighted_rmse=metrics_weighted["rmse"],
        weighted_mae=metrics_weighted["mae"],
        weighted_r2=metrics_weighted["r2"],
        holdout_rows=int(len(holdout_frame)),
    )


def _parse_settings(config: dict) -> ModelSettings:
    derived_dir = Path(config["paths"]["derived_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    tables_dir = modelling_tables_dir(outputs_dir)
    figures_dir = modelling_figures_dir(outputs_dir)
    model_dir = derived_dir / "models"
    for path in (outputs_dir, tables_dir, figures_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    modelling_cfg = config.get("modelling", {}) or {}
    ridge_alpha = modelling_cfg.get("ridge_alpha", DEFAULT_RIDGE_ALPHA)
    try:
        ridge_alpha = float(ridge_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("modelling.ridge_alpha must be numeric") from exc
    if ridge_alpha <= 0:
        raise ValueError("modelling.ridge_alpha must be greater than 0")

    return ModelSettings(
        derived_dir=derived_dir,
        outputs_dir=outputs_dir,
        profile_label=format_profile_label(config.get("project", {}).get("profile")),
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        model_dir=model_dir,
        holdout_years={int(year) for year in config.get("holdout_years", [])},
        task=str(modelling_cfg.get("task", "regression")).strip().lower(),
        predictor_track=str(modelling_cfg.get("predictor_track", "main")).strip().lower(),
        train_size=float(config.get("split", {}).get("train_size", 0.7)),
        stratify_on_year=bool(config.get("split", {}).get("stratify_on_year", False)),
        max_missing_rate=float(modelling_cfg.get("max_missing_rate", 1.0)),
        project_seed=int(config.get("project", {}).get("seed", 0)),
        ridge_alpha=ridge_alpha,
    )


def _clear_previous_outputs(settings: ModelSettings) -> None:
    managed_paths = [
        settings.model_dir / "regression_model.joblib",
        settings.model_dir / "regression_workflow.joblib",
        settings.model_dir / "regression_metadata.json",
    ]
    for name in MANAGED_MODEL_OUTPUTS:
        target_dir = settings.figures_dir if name.endswith(".png") else settings.tables_dir
        managed_paths.append(target_dir / name)
    for path in managed_paths:
        if path.exists():
            path.unlink()


def _split_holdout(
    df: pd.DataFrame,
    target: str,
    settings: ModelSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.dropna(subset=[target, SAMPLE_WEIGHT_COLUMN]).copy()
    if not settings.holdout_years:
        return cleaned, pd.DataFrame(columns=cleaned.columns)
    holdout = cleaned[cleaned[WAVE_YEAR_COLUMN].isin(settings.holdout_years)].copy()
    train = cleaned[~cleaned[WAVE_YEAR_COLUMN].isin(settings.holdout_years)].copy()
    return train, holdout


def _resolve_predictors(
    config: dict,
    settings: ModelSettings,
) -> tuple[list[str], list[str]]:
    meta_path = build_model_frame_meta_path(settings.derived_dir)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    predictors, track_meta = get_predictors_for_track(meta, config, settings.predictor_track)
    dropped_structural = track_meta.get("dropped_structural", meta.get("dropped_structural", []))
    return predictors, dropped_structural


def _primary_train_test_split(
    train_frame: pd.DataFrame,
    settings: ModelSettings,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    fallback = {"primary_split_stratify_failed": False}
    stratify = _year_stratify_labels(train_frame) if settings.stratify_on_year else None
    try:
        train_df, test_df = train_test_split(
            train_frame,
            train_size=settings.train_size,
            stratify=stratify,
            random_state=settings.project_seed,
        )
    except Exception as exc:
        fallback["primary_split_stratify_failed"] = stratify is not None
        log_event(
            logger,
            "primary_split_fallback",
            error=f"{type(exc).__name__}: {exc}",
        )
        train_df, test_df = train_test_split(
            train_frame,
            train_size=settings.train_size,
            random_state=settings.project_seed,
        )
    return train_df.copy(), test_df.copy(), fallback


def _year_stratify_labels(frame: pd.DataFrame) -> pd.Series | None:
    if WAVE_YEAR_COLUMN not in frame.columns:
        return None
    labels = frame[WAVE_YEAR_COLUMN].astype(str)
    counts = labels.value_counts()
    if counts.empty or int(counts.min()) < 2:
        return None
    return labels


def _build_regression_specs(
    *,
    predictors_used: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    ridge_alpha: float,
    categorical_levels: list[list[str]] | None,
) -> list[RegressionSpec]:
    return [
        RegressionSpec(
            model="mean_baseline",
            description="Predict the weighted training-set mean for every observation.",
            is_baseline=True,
            is_main_model=False,
            model_family="weighted_mean_baseline",
            implementation="bsa_code.train_eval.MeanBaselineRegressor",
            model_params={},
            predictors=(),
            numeric_features=(),
            categorical_features=(),
            categorical_levels=None,
        ),
        RegressionSpec(
            model="year_only_baseline",
            description="Weighted linear regression using year only.",
            is_baseline=True,
            is_main_model=False,
            model_family="weighted_linear_regression",
            implementation="sklearn.linear_model.LinearRegression",
            model_params={},
            predictors=(MODEL_YEAR_COLUMN,),
            numeric_features=(MODEL_YEAR_COLUMN,),
            categorical_features=(),
            categorical_levels=None,
        ),
        RegressionSpec(
            model="ridge_regression",
            description=(
                "Weighted ridge regression using the main demographic predictor set "
                f"(alpha={ridge_alpha:g})."
            ),
            is_baseline=False,
            is_main_model=True,
            model_family="weighted_ridge_regression",
            implementation="sklearn.linear_model.Ridge",
            model_params={"alpha": float(ridge_alpha)},
            predictors=tuple(predictors_used),
            numeric_features=tuple(numeric_features),
            categorical_features=tuple(categorical_features),
            categorical_levels=(
                tuple(tuple(levels) for levels in categorical_levels)
                if categorical_levels is not None
                else None
            ),
        ),
    ]


def _compare_models(
    *,
    specs: list[RegressionSpec],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    logger,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in specs:
        row = {
            "model": spec.model,
            "description": spec.description,
            "is_baseline": bool(spec.is_baseline),
            "is_main_model": bool(spec.is_main_model),
            "predictors": "|".join(spec.predictors),
            "predictor_count": int(len(spec.predictors)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "status": "ok",
            "error": None,
        }
        try:
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
            row.update(
                {
                    "fit_used_sample_weight": bool(fit_used_weights),
                    **_comparison_metric_fields(test_eval),
                }
            )
        except Exception as exc:
            log_event(
                logger,
                "model_comparison_failure",
                model=spec.model,
                error=f"{type(exc).__name__}: {exc}",
            )
            row.update(
                {
                    "fit_used_sample_weight": False,
                    **_empty_metric_fields(),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df["model_order"] = comparison_df["model"].map(
        {name: idx for idx, name in enumerate(MODEL_ORDER)}
    )
    comparison_df = comparison_df.sort_values("model_order").drop(columns=["model_order"])
    comparison_df = _attach_baseline_comparisons(comparison_df)
    return comparison_df.reset_index(drop=True)


def _comparison_metric_fields(evaluation: dict[str, object]) -> dict[str, object]:
    metrics = evaluation["metrics"]
    metrics_weighted = evaluation["metrics_weighted"]
    return {
        "n": metrics["n"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "weighted_rmse": metrics_weighted["rmse"],
        "weighted_mae": metrics_weighted["mae"],
        "weighted_r2": metrics_weighted["r2"],
        "pearson": metrics["pearson"],
        "spearman": metrics["spearman"],
        "observed_mean": metrics["observed_mean"],
        "predicted_mean": metrics["predicted_mean"],
        "observed_std": metrics["observed_std"],
        "predicted_std": metrics["predicted_std"],
        "weighted_observed_mean": metrics_weighted["observed_mean"],
        "weighted_predicted_mean": metrics_weighted["predicted_mean"],
        "weighted_observed_std": metrics_weighted["observed_std"],
        "weighted_predicted_std": metrics_weighted["predicted_std"],
    }


def _empty_metric_fields() -> dict[str, object]:
    return {
        "n": None,
        "rmse": None,
        "mae": None,
        "r2": None,
        "weighted_rmse": None,
        "weighted_mae": None,
        "weighted_r2": None,
        "pearson": None,
        "spearman": None,
        "observed_mean": None,
        "predicted_mean": None,
        "observed_std": None,
        "predicted_std": None,
        "weighted_observed_mean": None,
        "weighted_predicted_mean": None,
        "weighted_observed_std": None,
        "weighted_predicted_std": None,
    }


def _attach_baseline_comparisons(comparison_df: pd.DataFrame) -> pd.DataFrame:
    out = comparison_df.copy()
    mean_row = out[out["model"] == "mean_baseline"]
    out["weighted_rmse_gain_vs_mean_baseline_pct"] = None
    out["weighted_mae_gain_vs_mean_baseline_pct"] = None
    out["beats_mean_baseline_clearly"] = None
    out["predicted_sd_ratio"] = out.apply(
        lambda row: _safe_ratio(row["predicted_std"], row["observed_std"]),
        axis=1,
    )

    if mean_row.empty:
        return out

    mean_rmse = mean_row.iloc[0]["weighted_rmse"]
    mean_mae = mean_row.iloc[0]["weighted_mae"]
    out["weighted_rmse_gain_vs_mean_baseline_pct"] = out["weighted_rmse"].apply(
        lambda value: _pct_gain(mean_rmse, value)
    )
    out["weighted_mae_gain_vs_mean_baseline_pct"] = out["weighted_mae"].apply(
        lambda value: _pct_gain(mean_mae, value)
    )
    out["beats_mean_baseline_clearly"] = out.apply(
        lambda row: _beats_mean_baseline_clearly(
            row["weighted_rmse_gain_vs_mean_baseline_pct"],
            row["weighted_mae_gain_vs_mean_baseline_pct"],
        ),
        axis=1,
    )
    return out


def _pct_gain(baseline_value, model_value) -> float | None:
    if baseline_value in (None, 0) or model_value is None:
        return None
    try:
        baseline_float = float(baseline_value)
        model_float = float(model_value)
    except (TypeError, ValueError):
        return None
    if baseline_float == 0.0:
        return None
    return float(100.0 * (baseline_float - model_float) / baseline_float)


def _safe_ratio(numerator, denominator) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0.0:
        return None
    return float(num / den)


def _beats_mean_baseline_clearly(rmse_gain_pct, mae_gain_pct) -> bool | None:
    if rmse_gain_pct is None or mae_gain_pct is None:
        return None
    return bool(
        rmse_gain_pct >= CLEAR_MEAN_BASELINE_GAIN_PCT
        and mae_gain_pct >= CLEAR_MEAN_BASELINE_GAIN_PCT
    )


def _build_regression_workflow(spec: RegressionSpec) -> Pipeline:
    if spec.model == "mean_baseline":
        return Pipeline(steps=[("model", MeanBaselineRegressor())])
    if spec.model == "ridge_regression":
        estimator = Ridge(alpha=float(spec.model_params.get("alpha", DEFAULT_RIDGE_ALPHA)))
    else:
        estimator = LinearRegression()

    preprocessor = _build_preprocessor(
        list(spec.numeric_features),
        list(spec.categorical_features),
        (
            [list(levels) for levels in spec.categorical_levels]
            if spec.categorical_levels is not None
            else None
        ),
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def _fit_workflow(
    workflow,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None,
) -> bool:
    estimator = workflow.named_steps["model"]
    supports_sample_weight = _supports_sample_weight(estimator)
    if supports_sample_weight and sample_weight is not None:
        workflow.fit(X, y, model__sample_weight=np.asarray(sample_weight, dtype=float))
        return True
    workflow.fit(X, y)
    return False


def _supports_sample_weight(estimator) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def _select_feature_frame(
    frame: pd.DataFrame,
    predictors: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    columns = list(predictors)
    if not columns:
        return frame.iloc[:, :0].copy()
    return frame.loc[:, columns].copy()


def _evaluate_regression(
    *,
    model,
    frame: pd.DataFrame,
    predictors_used: list[str],
    target: str,
    split_name: str,
) -> dict[str, object] | None:
    if frame.empty:
        return None

    observed = frame[target].astype(float)
    predicted = pd.Series(
        model.predict(_select_feature_frame(frame, predictors_used)),
        index=frame.index,
        dtype=float,
    )
    weights = frame[SAMPLE_WEIGHT_COLUMN].astype(float)
    residual = observed - predicted

    predictions = frame[[RESPONDENT_ID_COLUMN, WAVE_YEAR_COLUMN, SAMPLE_WEIGHT_COLUMN]].copy()
    predictions["split"] = split_name
    predictions["observed"] = observed
    predictions["predicted"] = predicted
    predictions["residual"] = residual

    metrics = _regression_metrics(observed, predicted)
    metrics_weighted = _regression_metrics(observed, predicted, sample_weight=weights)
    metrics["pearson"] = _safe_correlation(observed, predicted, method="pearson")
    metrics["spearman"] = _safe_correlation(observed, predicted, method="spearman")

    return {
        "split": split_name,
        "predictions": predictions,
        "metrics": metrics,
        "metrics_weighted": metrics_weighted,
    }


def _regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> dict[str, float | int | None]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if weights is not None:
        mask = mask & np.isfinite(weights) & (weights > 0)
        weights = weights[mask]
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]

    if y_true_arr.size == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "observed_mean": None,
            "predicted_mean": None,
            "observed_std": None,
            "predicted_std": None,
        }

    residual = y_true_arr - y_pred_arr
    abs_residual = np.abs(residual)
    sq_residual = residual**2
    if weights is None:
        observed_mean = float(np.mean(y_true_arr))
        predicted_mean = float(np.mean(y_pred_arr))
        observed_std = float(np.std(y_true_arr))
        predicted_std = float(np.std(y_pred_arr))
        rmse = float(np.sqrt(np.mean(sq_residual)))
        mae = float(np.mean(abs_residual))
        ss_res = float(np.sum(sq_residual))
        ss_tot = float(np.sum((y_true_arr - observed_mean) ** 2))
    else:
        observed_mean = float(np.average(y_true_arr, weights=weights))
        predicted_mean = float(np.average(y_pred_arr, weights=weights))
        observed_std = _weighted_std(y_true_arr, weights)
        predicted_std = _weighted_std(y_pred_arr, weights)
        rmse = float(np.sqrt(np.average(sq_residual, weights=weights)))
        mae = float(np.average(abs_residual, weights=weights))
        ss_res = float(np.sum(weights * sq_residual))
        ss_tot = float(np.sum(weights * (y_true_arr - observed_mean) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - (ss_res / ss_tot))
    return {
        "n": int(y_true_arr.size),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "observed_mean": observed_mean,
        "predicted_mean": predicted_mean,
        "observed_std": observed_std,
        "predicted_std": predicted_std,
    }


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0 or weights.size == 0:
        return None
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return None
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return float(np.sqrt(variance))


def _safe_correlation(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    method: str,
) -> float | None:
    observed = pd.Series(np.asarray(y_true, dtype=float))
    predicted = pd.Series(np.asarray(y_pred, dtype=float))
    if observed.nunique(dropna=True) < 2 or predicted.nunique(dropna=True) < 2:
        return None
    try:
        value = observed.corr(predicted, method=method)
    except Exception:
        return None
    return None if pd.isna(value) else float(value)


def _build_interpretation(comparison_df: pd.DataFrame) -> dict[str, object]:
    rows = _records_for_json(comparison_df)
    by_model = {row["model"]: row for row in rows}
    main_row = next((row for row in rows if row.get("is_main_model")), None)
    year_row = by_model.get("year_only_baseline")
    mean_row = by_model.get("mean_baseline")

    summary: list[str] = []
    if main_row is not None and mean_row is not None:
        rmse_gain = main_row.get("weighted_rmse_gain_vs_mean_baseline_pct")
        mae_gain = main_row.get("weighted_mae_gain_vs_mean_baseline_pct")
        if _beats_mean_baseline_clearly(rmse_gain, mae_gain):
            summary.append(f"{main_row['model']} beats the mean baseline clearly.")
        elif rmse_gain is not None and mae_gain is not None and rmse_gain > 0 and mae_gain > 0:
            summary.append(f"{main_row['model']} improves on the mean baseline, but only marginally.")
        else:
            summary.append(
                f"{main_row['model']} does not beat the mean baseline on the main weighted error metrics."
            )
    if year_row is not None and mean_row is not None:
        rmse_gain = year_row.get("weighted_rmse_gain_vs_mean_baseline_pct")
        mae_gain = year_row.get("weighted_mae_gain_vs_mean_baseline_pct")
        if rmse_gain is not None and mae_gain is not None and rmse_gain > 0 and mae_gain > 0:
            summary.append("year_only_baseline improves on the mean baseline.")
        else:
            summary.append("year_only_baseline does not improve on the mean baseline.")
    if main_row is not None:
        sd_ratio = main_row.get("predicted_sd_ratio")
        pearson = main_row.get("pearson")
        if sd_ratio is not None and pearson is not None:
            if sd_ratio < 0.4 or pearson < 0.2:
                summary.append(f"{main_row['model']} still collapses strongly toward the mean.")
            elif sd_ratio < 0.7 or pearson < 0.4:
                summary.append(
                    f"{main_row['model']} learns some variation, but predictions remain noticeably shrunken."
                )
            else:
                summary.append(f"{main_row['model']} appears to learn substantial variation.")

    return {
        "criteria": {
            "beats_mean_baseline_clearly": (
                "Weighted RMSE and weighted MAE are both at least "
                f"{CLEAR_MEAN_BASELINE_GAIN_PCT:.1f}% lower than the mean baseline."
            ),
            "variation_assessment": (
                "Uses the predicted/observed standard-deviation ratio and Pearson correlation."
            ),
        },
        "rows": rows,
        "summary": summary,
    }


def _records_for_json(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _write_training_outputs(
    *,
    settings: ModelSettings,
    model,
    main_spec: RegressionSpec,
    target: str,
    predictors_requested: list[str],
    predictors_used: list[str],
    dropped_structural: list[str],
    dropped_missing: list[str],
    missing_rates: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    interpretation: dict[str, object],
    split_fallbacks: dict[str, object],
    main_fit_used_weights: bool,
    test_eval: dict[str, object],
    holdout_eval: dict[str, object] | None,
) -> None:
    model_path = settings.model_dir / "regression_model.joblib"
    joblib.dump(model, model_path)

    comparison_path = settings.tables_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    main_row_df = comparison_df[comparison_df["is_main_model"]].copy()
    main_row_df.insert(1, "target_column", target)
    if holdout_eval is not None:
        holdout_metrics = holdout_eval.get("metrics", {})
        holdout_metrics_weighted = holdout_eval.get("metrics_weighted", {})
        main_row_df["holdout_years"] = [
            ";".join(
                str(int(year)) for year in sorted(holdout_df[WAVE_YEAR_COLUMN].dropna().unique())
            )
        ]
        main_row_df["holdout_n"] = holdout_metrics.get("n")
        main_row_df["holdout_rmse"] = holdout_metrics.get("rmse")
        main_row_df["holdout_mae"] = holdout_metrics.get("mae")
        main_row_df["holdout_r2"] = holdout_metrics.get("r2")
        main_row_df["holdout_weighted_rmse"] = holdout_metrics_weighted.get("rmse")
        main_row_df["holdout_weighted_mae"] = holdout_metrics_weighted.get("mae")
        main_row_df["holdout_weighted_r2"] = holdout_metrics_weighted.get("r2")
    regression_metrics_path = settings.tables_dir / "regression_metrics.csv"
    main_row_df.to_csv(regression_metrics_path, index=False)
    _write_regression_metrics_table(
        metrics_row=main_row_df.iloc[0],
        profile_label=settings.profile_label,
        out_path=settings.figures_dir / "regression_metrics_table.png",
    )

    _write_predicted_vs_actual_plot(
        predictions=test_eval["predictions"],
        target=target,
        profile_label=settings.profile_label,
        test_metrics=test_eval["metrics_weighted"],
        out_path=settings.figures_dir / "predicted_vs_actual.png",
    )
    holdout_payload = None
    if holdout_eval is not None:
        holdout_payload = {
            "model": main_spec.model,
            "target_column": target,
            "years": sorted(int(year) for year in holdout_df[WAVE_YEAR_COLUMN].dropna().unique()),
            "metrics": holdout_eval["metrics"],
            "metrics_weighted": holdout_eval["metrics_weighted"],
        }

    model_meta = {
        "task": settings.task,
        "target_column": target,
        "main_model": main_spec.model,
        "estimation": {
            "model_family": main_spec.model_family,
            "implementation": main_spec.implementation,
            "model_params": main_spec.model_params,
            "fit_uses_sample_weight": bool(main_fit_used_weights),
            "sample_weight_column": SAMPLE_WEIGHT_COLUMN,
            "model_year_column": MODEL_YEAR_COLUMN,
            "wave_key_column": WAVE_YEAR_COLUMN,
        },
        "main_model_predictors": list(main_spec.predictors),
        "predictor_track": settings.predictor_track,
        "predictors_requested": predictors_requested,
        "predictors_used": predictors_used,
        "dropped_predictors_structural": dropped_structural,
        "dropped_predictors_missingness": dropped_missing,
        "missing_rates": missing_rates,
        "split_summary": _split_summary(
            train_df=train_df,
            test_df=test_df,
            holdout_df=holdout_df,
        ),
        "split_fallbacks": split_fallbacks,
        "training": {
            "fit_used_sample_weight": bool(main_fit_used_weights),
        },
        "comparison_rows": _records_for_json(comparison_df),
        "interpretation": interpretation,
        "metrics": {
            "test_unweighted": test_eval["metrics"],
            "test_weighted": test_eval["metrics_weighted"],
            "holdout": holdout_payload,
        },
        "outputs": {
            "regression_model": str(model_path),
            "model_comparison": str(comparison_path),
            "regression_metrics_csv": str(regression_metrics_path),
            "predicted_vs_actual_plot": str(settings.figures_dir / "predicted_vs_actual.png"),
            "regression_metrics_table_plot": str(
                settings.figures_dir / "regression_metrics_table.png"
            ),
        },
    }
    (settings.model_dir / "regression_metadata.json").write_text(
        json.dumps(model_meta, indent=2),
        encoding="utf-8",
    )


def _write_predicted_vs_actual_plot(
    *,
    predictions: pd.DataFrame,
    target: str,
    profile_label: str,
    test_metrics: dict[str, object],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    hexbin = ax.hexbin(
        predictions["observed"],
        predictions["predicted"],
        gridsize=34,
        mincnt=1,
        cmap="Blues",
        linewidths=0.25,
        edgecolors="white",
    )
    lower = float(min(predictions["observed"].min(), predictions["predicted"].min()))
    upper = float(max(predictions["observed"].max(), predictions["predicted"].max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color=PALETTE["gray"])
    style_axes(
        ax,
        title=with_profile_prefix(profile_label, f"observed vs predicted {target}"),
        xlabel=f"Observed {target}",
        ylabel="Predicted score",
        grid_axis="both",
        title_loc="center",
    )
    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Count")
    annotation = [
        f"Weighted R²: {float(test_metrics['r2']):.3f}" if test_metrics.get("r2") is not None else None,
        (
            "Weighted RMSE: "
            f"{float(test_metrics['rmse']):.3f}"
            if test_metrics.get("rmse") is not None
            else None
        ),
    ]
    annotation = [line for line in annotation if line]
    if annotation:
        ax.text(
            0.03,
            0.97,
            "\n".join(annotation),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": PALETTE["edge"], "boxstyle": "round,pad=0.3"},
        )
    finalize_figure(fig, out_path)


def _write_regression_metrics_table(
    *,
    metrics_row: pd.Series,
    profile_label: str,
    out_path: Path,
) -> None:
    display_rows = [
        ("Model", _format_metric_value(metrics_row.get("model")), ""),
        ("Predictors", _format_metric_value(metrics_row.get("predictor_count")), ""),
        ("Test rows", _format_metric_value(metrics_row.get("n")), ""),
        ("Holdout year", "", _format_metric_value(metrics_row.get("holdout_years"))),
        ("Holdout rows", "", _format_metric_value(metrics_row.get("holdout_n"))),
        ("Weighted R²", _format_metric_value(metrics_row.get("weighted_r2")), _format_metric_value(metrics_row.get("holdout_weighted_r2"))),
        ("Weighted RMSE", _format_metric_value(metrics_row.get("weighted_rmse")), _format_metric_value(metrics_row.get("holdout_weighted_rmse"))),
        ("Weighted MAE", _format_metric_value(metrics_row.get("weighted_mae")), _format_metric_value(metrics_row.get("holdout_weighted_mae"))),
        ("Pearson", _format_metric_value(metrics_row.get("pearson")), ""),
        ("Spearman", _format_metric_value(metrics_row.get("spearman")), ""),
        ("Predicted SD ratio", _format_metric_value(metrics_row.get("predicted_sd_ratio")), ""),
        ("RMSE gain vs mean baseline", _format_pct_value(metrics_row.get("weighted_rmse_gain_vs_mean_baseline_pct")), ""),
        ("MAE gain vs mean baseline", _format_pct_value(metrics_row.get("weighted_mae_gain_vs_mean_baseline_pct")), ""),
    ]

    fig_height = max(5.8, 0.42 * len(display_rows))
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    ax.axis("off")
    ax.set_title(
        with_profile_prefix(profile_label, "prediction performance summary"),
        loc="left",
        pad=10,
    )

    table = ax.table(
        cellText=display_rows,
        colLabels=["Metric", "Test split", "Holdout"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.38, 0.31, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.18)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["edge"])
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor(PALETTE["bg_alt"])
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        else:
            cell.set_facecolor("#FAFCFF" if row_idx % 2 == 0 else "white")
            cell.set_text_props(color=PALETTE["ink"])

    finalize_figure(fig, out_path)


def _format_metric_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    text = str(value)
    if len(text) > 60:
        return textwrap.fill(text, width=60, break_long_words=False, break_on_hyphens=False)
    return text


def _format_pct_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1f}%"


def _split_summary(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict[str, object]:
    def _frame_summary(frame: pd.DataFrame | None) -> dict[str, object]:
        if frame is None or frame.empty:
            return {"rows": 0, "weight_sum": 0.0, "year_counts": {}}
        return {
            "rows": int(len(frame)),
            "weight_sum": float(frame[SAMPLE_WEIGHT_COLUMN].sum()),
            "year_counts": {
                str(int(year)): int(count)
                for year, count in frame[WAVE_YEAR_COLUMN].value_counts().items()
            },
        }

    return {
        "train": _frame_summary(train_df),
        "test": _frame_summary(test_df),
        "holdout": _frame_summary(holdout_df),
    }

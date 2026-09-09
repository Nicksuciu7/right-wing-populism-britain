# builds trend outputs
# reads scores and model frame

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import load_config
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
    trend_figures_dir,
    trend_tables_dir,
)
from .plotting import (
    PALETTE,
    apply_plot_theme,
    finalize_figure,
    format_profile_label,
    style_axes,
    with_profile_prefix,
)

apply_plot_theme()


_BINARY_OUTCOME_COLUMNS = {"RWP_top_20"}
_MANAGED_TREND_OUTPUTS = (
    ("table", "rwp_prevalence_by_year.csv"),
    ("figure", "rwp_prevalence_by_year.png"),
    ("table", "predictor_trends.csv"),
    ("figure", "predictor_trends.png"),
    ("table", "predictor_trend_model_fit.csv"),
    ("figure", "predictor_trend_model_fit_table.png"),
    ("table", "education_by_year_rwp.csv"),
    ("figure", "education_by_year_rwp.png"),
    ("table", "employment_status_by_year_rwp.csv"),
    ("figure", "employment_status_by_year_rwp.png"),
    ("table", "religion_by_year_rwp.csv"),
    ("figure", "religion_by_year_rwp.png"),
)


def _trend_outputs_enabled(config: dict) -> bool:
    trend_cfg = config.get("trend_analysis", {}) or {}
    return bool(trend_cfg.get("enabled", True))


def _clear_managed_trend_outputs(outputs_dir: Path) -> None:
    tables_dir = trend_tables_dir(outputs_dir)
    figures_dir = trend_figures_dir(outputs_dir)
    for kind, name in _MANAGED_TREND_OUTPUTS:
        base_dir = tables_dir if kind == "table" else figures_dir
        path = base_dir / name
        if path.exists():
            path.unlink()


def _resolve_prevalence_label_spec(
    df: pd.DataFrame,
    trend_cfg: dict,
    logger,
    *,
    stage: str,
) -> tuple[str, str]:
    requested_label = str(
        trend_cfg.get(
            "prevalence_label_column",
            trend_cfg.get("label_column", "RWP_top_20"),
        )
    )
    label_col = requested_label
    if label_col not in df.columns:
        fallback_label = "RWP_top_20" if "RWP_top_20" in df.columns else None
        if fallback_label is None:
            raise KeyError(
                f"Requested trend label column '{requested_label}' is not present and no RWP_top_20 fallback exists."
            )
        log_event(
            logger,
            f"{stage}_label_fallback",
            requested_label=requested_label,
            fallback_label=fallback_label,
        )
        label_col = fallback_label
    label_display_name = str(
        trend_cfg.get(
            "prevalence_label_display_name",
            trend_cfg.get("label_display_name", "RWP top 20%"),
        )
    )
    return label_col, label_display_name


def _resolve_trend_target_spec(
    df: pd.DataFrame,
    config: dict,
    trend_cfg: dict,
    logger,
    *,
    stage: str,
) -> tuple[str, str]:
    requested_target = trend_cfg.get("target_column")
    target_col = _resolve_target_column(
        df,
        config.get("modelling", {}),
        logger,
        stage=stage,
        override_target=str(requested_target) if requested_target is not None else None,
    )
    if target_col in _BINARY_OUTCOME_COLUMNS:
        raise ValueError(f"Trend target must be continuous, not binary: {target_col}")
    target_display_name = str(
        trend_cfg.get("target_display_name", trend_cfg.get("label_display_name", target_col))
    )
    return target_col, target_display_name


def _feature_to_predictor(feature: str, predictors: list[str]) -> str:
    if feature.startswith("num__"):
        return feature[len("num__") :]
    if feature.startswith("cat__"):
        remainder = feature[len("cat__") :]
        for pred in sorted(predictors, key=len, reverse=True):
            if remainder == pred or remainder.startswith(f"{pred}_"):
                return pred
        return remainder.split("_", 1)[0]
    return feature


def _format_predictor_label(predictor: str) -> str:
    return " ".join(str(predictor).replace("_", " ").split())


def _profile_title_prefix(config: dict) -> str:
    return format_profile_label(config.get("project", {}).get("profile"))


def _annotate_line_end_labels(ax, line_labels: list[dict[str, object]], *, x_offset: float) -> None:
    if not line_labels:
        return
    lower, upper = ax.get_ylim()
    y_range = max(upper - lower, 1e-6)
    min_gap = max(0.012, 0.045 * y_range)

    ordered = sorted(
        enumerate(line_labels),
        key=lambda pair: float(pair[1]["y"]),
    )
    adjusted: dict[int, float] = {}
    cursor = lower
    for idx, payload in ordered:
        y_val = max(float(payload["y"]), cursor)
        adjusted[idx] = y_val
        cursor = y_val + min_gap

    overflow = max(adjusted.values()) - upper
    if overflow > 0:
        for idx in adjusted:
            adjusted[idx] -= overflow
    underflow = lower - min(adjusted.values())
    if underflow > 0:
        for idx in adjusted:
            adjusted[idx] += underflow

    for idx, payload in enumerate(line_labels):
        x_val = float(payload["x"])
        y_val = float(payload["y"])
        label_y = adjusted[idx]
        color = payload["color"]
        ax.plot(
            [x_val, x_val + x_offset * 0.55],
            [y_val, label_y],
            color=color,
            linewidth=1.0,
            alpha=0.5,
        )
        ax.text(
            x_val + x_offset,
            label_y,
            str(payload["label"]),
            color=color,
            fontsize=8,
            ha="left",
            va="center",
        )


def _is_missing_encoded_feature(feature: str) -> bool:
    if feature.startswith("cat__") and feature.endswith("_missing"):
        return True
    return False


def rwp_prevalence(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    profile_label = _profile_title_prefix(config)
    derived_dir = Path(config["paths"]["derived_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    tables_dir = trend_tables_dir(outputs_dir)
    figures_dir = trend_figures_dir(outputs_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(derived_dir / "logs" / "trends.log")
    if not _trend_outputs_enabled(config):
        _clear_managed_trend_outputs(outputs_dir)
        log_event(
            logger,
            "prevalence_skipped",
            reason="trend_outputs_disabled",
            profile=profile_label,
        )
        return

    trend_cfg = config.get("trend_analysis", {})
    df = pd.read_parquet(derived_dir / "scores" / "pooled_scored.parquet")
    holdout_years = set(config.get("holdout_years", []))
    exclude_holdout_years = bool(trend_cfg.get("exclude_holdout_years", False))
    if exclude_holdout_years and holdout_years:
        df = df[~df["wave_year"].isin(holdout_years)].copy()

    label_col, label_name = _resolve_prevalence_label_spec(
        df,
        trend_cfg,
        logger,
        stage="rwp_prevalence",
    )

    rows = []
    for year in sorted(df["wave_year"].dropna().astype(int).unique().tolist()):
        df_year = df[df["wave_year"] == year].dropna(subset=[label_col, "weight"])
        prevalence = (
            float((df_year[label_col] * df_year["weight"]).sum() / df_year["weight"].sum())
            if not df_year.empty
            else None
        )
        rows.append({"year": int(year), "weighted_prevalence": prevalence, "n": int(len(df_year))})

    prev_df = pd.DataFrame(rows)
    plotted_years = sorted(prev_df["year"].dropna().astype(int).unique().tolist())
    prev_path = tables_dir / "rwp_prevalence_by_year.csv"
    prev_df.to_csv(prev_path, index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(
        prev_df["year"],
        prev_df["weighted_prevalence"],
        marker="o",
        color=PALETTE["blue"],
    )
    style_axes(
        ax,
        title=with_profile_prefix(profile_label, f"{label_name} prevalence by year"),
        xlabel="Year",
        ylabel="Weighted prevalence",
        grid_axis="y",
        integer_x=True,
        percent_y=True,
        title_loc="center",
    )
    if plotted_years:
        ax.set_xticks(plotted_years)
        ax.set_xlim(min(plotted_years) - 0.2, max(plotted_years) + 0.35)
    ax.set_ylim(0.0, 0.5)
    for year, prevalence in zip(prev_df["year"], prev_df["weighted_prevalence"]):
        if prevalence is None or pd.isna(prevalence):
            continue
        ax.text(
            float(year),
            float(prevalence) + 0.008,
            f"{float(prevalence):.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["blue"],
        )

    plot_path = figures_dir / "rwp_prevalence_by_year.png"
    finalize_figure(fig, plot_path)

    log_event(
        logger,
        "prevalence_complete",
        output=str(prev_path),
        plot=str(plot_path),
        holdout_excluded=sorted(holdout_years),
    )


def predictor_trends(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    profile_label = _profile_title_prefix(config)
    derived_dir = Path(config["paths"]["derived_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    tables_dir = trend_tables_dir(outputs_dir)
    figures_dir = trend_figures_dir(outputs_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(derived_dir / "logs" / "trends.log")
    if not _trend_outputs_enabled(config):
        _clear_managed_trend_outputs(outputs_dir)
        log_event(
            logger,
            "predictor_trends_skipped",
            reason="trend_outputs_disabled",
            profile=profile_label,
        )
        return

    trend_cfg = config.get("trend_analysis", {})
    predictor_track = str(trend_cfg.get("predictor_track", "trend"))

    df = pd.read_parquet(build_model_frame_parquet_path(derived_dir))
    holdout_years = set(config.get("holdout_years", []))
    exclude_holdout_years = bool(trend_cfg.get("exclude_holdout_years", False))
    if exclude_holdout_years and holdout_years:
        df = df[~df["wave_year"].isin(holdout_years)].copy()

    target_col, target_name = _resolve_trend_target_spec(
        df,
        config,
        trend_cfg,
        logger,
        stage="predictor_trends",
    )
    df = df.dropna(subset=[target_col, "weight"]).copy()

    meta_path = build_model_frame_meta_path(derived_dir)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    predictors, _ = get_predictors_for_track(meta, config, predictor_track)

    min_wave_coverage = int(trend_cfg.get("min_wave_coverage", 0))
    max_missing_rate = float(trend_cfg.get("max_missing_rate", 1.0))
    exclude_features = {str(v) for v in trend_cfg.get("exclude_features", [])}

    kept_predictors: list[str] = []
    years_sorted = sorted(df["wave_year"].dropna().astype(int).unique().tolist())
    for predictor in predictors:
        if predictor not in df.columns or predictor in exclude_features:
            continue
        waves_present = 0
        for year in years_sorted:
            df_year = df[df["wave_year"] == year]
            missing_rate = float(df_year[predictor].isna().mean()) if not df_year.empty else 1.0
            if missing_rate <= max_missing_rate:
                waves_present += 1
        if waves_present >= min_wave_coverage:
            kept_predictors.append(predictor)

    numeric_features = [f for f in config["demographics"]["numeric"] if f in kept_predictors]
    categorical_features = [f for f in kept_predictors if f not in numeric_features]
    if not kept_predictors:
        log_event(
            logger,
            "predictor_trends_skipped",
            reason="no_predictors_after_trend_filter",
            predictor_track=predictor_track,
        )
        return

    X = df[kept_predictors]
    nf, cf, _, _ = _filter_features(X, numeric_features, categorical_features, max_missing_rate)
    predictors_used = nf + cf
    if not predictors_used:
        log_event(
            logger,
            "predictor_trends_skipped",
            reason="all_predictors_dropped_by_missingness",
            predictor_track=predictor_track,
        )
        return

    preprocessor = _build_preprocessor(
        nf,
        cf,
        _resolve_categorical_levels(
            config.get("demographics", {}).get("categories", {}),
            cf,
        ),
    )
    preprocessor.fit(X[predictors_used])
    feature_names = preprocessor.get_feature_names_out()
    permutation_repeats = max(1, int(trend_cfg.get("permutation_repeats", 20)))
    permutation_seed = int(config.get("project", {}).get("seed", 0))

    rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for year in years_sorted:
        df_year = df[df["wave_year"] == year]
        if df_year.empty:
            continue
        X_year = preprocessor.transform(df_year[predictors_used])
        y_year = df_year[target_col].astype(float)
        w_year = df_year["weight"].astype(float)

        model = LinearRegression()
        model.fit(X_year, y_year, sample_weight=w_year)
        y_pred = np.asarray(model.predict(X_year), dtype=float).reshape(-1)

        feature_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coef": np.asarray(model.coef_, dtype=float).reshape(-1),
            }
        )
        feature_df["predictor"] = feature_df["feature"].apply(
            lambda value: _feature_to_predictor(value, predictors_used)
        )
        if bool(trend_cfg.get("overview_exclude_missing_levels", True)):
            filtered = feature_df[~feature_df["feature"].apply(_is_missing_encoded_feature)].copy()
            if not filtered.empty:
                feature_df = filtered
        feature_df["abs_coef"] = feature_df["coef"].abs()

        predictor_year_df = (
            feature_df.groupby("predictor", as_index=False)
            .agg(
                mean_abs_coef=("abs_coef", "mean"),
                max_abs_coef=("abs_coef", "max"),
                n_levels=("feature", "nunique"),
            )
            .sort_values("predictor")
        )
        predictor_year_df.insert(0, "year", int(year))
        predictor_year_df["predictor_track"] = predictor_track
        predictor_year_df["target_column"] = target_col

        metrics = _trend_regression_metrics(y_year.to_numpy(), y_pred)
        metrics_weighted = _trend_regression_metrics(
            y_year.to_numpy(),
            y_pred,
            sample_weight=w_year.to_numpy(),
        )
        permutation_df = _grouped_permutation_importance(
            raw_X=df_year[predictors_used],
            y=y_year,
            weights=w_year,
            preprocessor=preprocessor,
            model=model,
            predictors=predictors_used,
            baseline_metrics=metrics_weighted,
            repeats=permutation_repeats,
            seed=permutation_seed + int(year),
        )
        predictor_year_df = predictor_year_df.merge(
            permutation_df,
            on="predictor",
            how="left",
        )
        rows.extend(predictor_year_df.to_dict(orient="records"))
        fit_rows.append(
            {
                "year": int(year),
                "n": int(metrics["n"]),
                "weighted_n": int(metrics_weighted["n"]),
                "predictor_count": int(len(predictors_used)),
                "encoded_feature_count": int(len(feature_names)),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "weighted_rmse": metrics_weighted["rmse"],
                "weighted_mae": metrics_weighted["mae"],
                "weighted_r2": metrics_weighted["r2"],
                "pearson": _safe_correlation(y_year.to_numpy(), y_pred, method="pearson"),
                "spearman": _safe_correlation(y_year.to_numpy(), y_pred, method="spearman"),
            }
        )

    trend_df = pd.DataFrame(rows)
    if trend_df.empty:
        log_event(
            logger,
            "predictor_trends_skipped",
            reason="no_trend_rows_written",
            predictor_track=predictor_track,
        )
        return

    trend_path = tables_dir / "predictor_trends.csv"
    trend_df.to_csv(trend_path, index=False)
    fit_df = pd.DataFrame(fit_rows).sort_values("year").reset_index(drop=True)
    fit_path = tables_dir / "predictor_trend_model_fit.csv"
    fit_df.to_csv(fit_path, index=False)
    _write_trend_fit_table_figure(
        fit_df=fit_df,
        profile_label=profile_label,
        out_path=figures_dir / "predictor_trend_model_fit_table.png",
    )

    overview_metric = str(trend_cfg.get("overview_metric", "permutation_rmse_increase")).lower()
    metric_options = {
        "mean_abs": ("mean_abs_coef", "Mean absolute coefficient across encoded levels"),
        "max_abs": ("max_abs_coef", "Max absolute coefficient across encoded levels"),
        "permutation": (
            "permutation_rmse_increase_mean",
            "Grouped permutation RMSE increase",
        ),
        "permutation_rmse": (
            "permutation_rmse_increase_mean",
            "Grouped permutation RMSE increase",
        ),
        "permutation_rmse_increase": (
            "permutation_rmse_increase_mean",
            "Grouped permutation RMSE increase",
        ),
        "permutation_r2_drop": (
            "permutation_r2_drop_mean",
            "Drop in weighted R² (in-sample)",
        ),
    }
    metric_col, metric_label = metric_options.get(
        overview_metric,
        metric_options["permutation_rmse_increase"],
    )
    overview_top_n = max(1, int(trend_cfg.get("overview_top_n_predictors", 5)))
    overview_predictors = [str(v) for v in trend_cfg.get("overview_predictors", [])]
    overview_exclude_predictors = {
        str(v) for v in trend_cfg.get("overview_exclude_predictors", [])
    }
    overview_y_max = trend_cfg.get("overview_y_max")
    try:
        overview_y_max = float(overview_y_max) if overview_y_max is not None else None
    except (TypeError, ValueError):
        overview_y_max = None
    if overview_y_max is not None and overview_y_max <= 0:
        overview_y_max = None

    plot_df = trend_df.copy()
    if overview_exclude_predictors:
        plot_df = plot_df[~plot_df["predictor"].isin(overview_exclude_predictors)].copy()
    plot_df = plot_df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        log_event(
            logger,
            "predictor_trends_plot_skipped",
            reason="no_rows_for_overview_metric",
            overview_metric=overview_metric,
            metric_column=metric_col,
        )
        return

    rank_df = (
        plot_df.groupby("predictor", as_index=False)[metric_col]
        .mean()
        .sort_values(metric_col, ascending=False)
    )
    available_predictors = set(rank_df["predictor"].astype(str).tolist())
    ordered_predictors = [p for p in overview_predictors if p in available_predictors]
    if not ordered_predictors:
        ordered_predictors = rank_df.head(overview_top_n)["predictor"].tolist()
    _write_group_summary_outputs(
        df=df,
        config=config,
        group_col="education",
        target_col=target_col,
        target_name=target_name,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        profile_label=profile_label,
    )
    _write_group_summary_outputs(
        df=df,
        config=config,
        group_col="employment_status",
        target_col=target_col,
        target_name=target_name,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        profile_label=profile_label,
    )
    _write_group_summary_outputs(
        df=df,
        config=config,
        group_col="religion",
        target_col=target_col,
        target_name=target_name,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        profile_label=profile_label,
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    cmap = plt.get_cmap("tab10")
    metric_max = 0.0
    end_labels: list[dict[str, object]] = []
    for idx, predictor in enumerate(ordered_predictors):
        sub = plot_df[plot_df["predictor"] == predictor].sort_values("year")
        if sub.empty:
            continue
        color = cmap(idx % 10)
        ax.plot(
            sub["year"],
            sub[metric_col],
            marker="o",
            color=color,
            linewidth=2.2,
        )
        metric_max = max(metric_max, float(sub[metric_col].max()))
        last_row = sub.iloc[-1]
        end_labels.append(
            {
                "x": float(last_row["year"]),
                "y": float(last_row[metric_col]),
                "label": _format_predictor_label(predictor),
                "color": color,
            }
        )

    style_axes(
        ax,
        title=with_profile_prefix(profile_label, "Predictor strength over time"),
        xlabel="Year",
        ylabel=metric_label,
        grid_axis="both",
        integer_x=True,
        title_loc="center",
    )
    if years_sorted:
        ax.set_xticks(years_sorted)
        ax.set_xlim(min(years_sorted) - 0.2, max(years_sorted) + 0.6)
    if overview_y_max is not None:
        y_upper = overview_y_max
    elif metric_col == "permutation_rmse_increase_mean":
        y_upper = max(0.001, 1.18 * metric_max)
    else:
        y_upper = max(0.12, 1.12 * metric_max)
    ax.set_ylim(0, y_upper)
    _annotate_line_end_labels(ax, end_labels, x_offset=0.14)

    fig.tight_layout()
    plot_path = figures_dir / "predictor_trends.png"
    finalize_figure(fig, plot_path)

    log_event(
        logger,
        "predictor_trends_complete",
        output=str(trend_path),
        plot=str(plot_path),
        fit_table=str(fit_path),
        predictor_track=predictor_track,
        target_column=target_col,
        overview_metric=metric_col,
        permutation_repeats=permutation_repeats,
    )


def _grouped_permutation_importance(
    *,
    raw_X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    preprocessor,
    model,
    predictors: list[str],
    baseline_metrics: dict[str, float | int | None],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_rmse = baseline_metrics.get("rmse")
    baseline_r2 = baseline_metrics.get("r2")
    y_arr = y.to_numpy(dtype=float)
    w_arr = weights.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    for predictor in predictors:
        if predictor not in raw_X.columns:
            continue
        observed = raw_X[predictor].reset_index(drop=True)
        unique_count = int(observed.astype("string").nunique(dropna=False))
        rmse_increases: list[float] = []
        r2_drops: list[float] = []

        if unique_count > 1:
            values = observed.to_numpy(copy=True)
            for _ in range(repeats):
                permuted = values.copy()
                rng.shuffle(permuted)
                permuted_X = raw_X.reset_index(drop=True).copy()
                permuted_X[predictor] = permuted
                permuted_transformed = preprocessor.transform(permuted_X[predictors])
                permuted_pred = np.asarray(
                    model.predict(permuted_transformed),
                    dtype=float,
                ).reshape(-1)
                permuted_metrics = _trend_regression_metrics(
                    y_arr,
                    permuted_pred,
                    sample_weight=w_arr,
                )
                if baseline_rmse is not None and permuted_metrics.get("rmse") is not None:
                    rmse_increases.append(float(permuted_metrics["rmse"]) - float(baseline_rmse))
                if baseline_r2 is not None and permuted_metrics.get("r2") is not None:
                    r2_drops.append(float(baseline_r2) - float(permuted_metrics["r2"]))

        rows.append(
            {
                "predictor": predictor,
                "permutation_repeats": int(repeats),
                "permutation_unique_values": unique_count,
                "permutation_rmse_increase_mean": _mean_or_none(rmse_increases),
                "permutation_rmse_increase_std": _std_or_none(rmse_increases),
                "permutation_r2_drop_mean": _mean_or_none(r2_drops),
                "permutation_r2_drop_std": _std_or_none(r2_drops),
            }
        )

    return pd.DataFrame(rows)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _std_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _trend_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
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
        return {"n": 0, "rmse": None, "mae": None, "r2": None}

    residual = y_true_arr - y_pred_arr
    abs_residual = np.abs(residual)
    sq_residual = residual**2
    if weights is None:
        observed_mean = float(np.mean(y_true_arr))
        rmse = float(np.sqrt(np.mean(sq_residual)))
        mae = float(np.mean(abs_residual))
        ss_res = float(np.sum(sq_residual))
        ss_tot = float(np.sum((y_true_arr - observed_mean) ** 2))
    else:
        observed_mean = float(np.average(y_true_arr, weights=weights))
        rmse = float(np.sqrt(np.average(sq_residual, weights=weights)))
        mae = float(np.average(abs_residual, weights=weights))
        ss_res = float(np.sum(weights * sq_residual))
        ss_tot = float(np.sum(weights * (y_true_arr - observed_mean) ** 2))
    r2 = None if ss_tot <= 0 else float(1.0 - (ss_res / ss_tot))
    return {"n": int(y_true_arr.size), "rmse": rmse, "mae": mae, "r2": r2}


def _safe_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
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


def _write_trend_fit_table_figure(
    *,
    fit_df: pd.DataFrame,
    profile_label: str,
    out_path: Path,
) -> None:
    if fit_df.empty:
        return

    display_df = fit_df.loc[
        :,
        [
            "year",
            "n",
            "weighted_r2",
            "weighted_rmse",
            "weighted_mae",
        ],
    ].copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].apply(_format_trend_fit_value)

    fig, ax = plt.subplots(figsize=(7.8, max(2.8, 0.44 * (len(display_df) + 1))))
    ax.axis("off")
    ax.set_title(
        with_profile_prefix(profile_label, "Trend model fit by year"),
        loc="left",
        pad=6,
        fontsize=12,
    )

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=[
            "Year",
            "N",
            "Weighted R2",
            "Weighted RMSE",
            "Weighted MAE",
        ],
        bbox=[0.0, 0.02, 1.0, 0.86],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    col_widths = [0.12, 0.14, 0.22, 0.25, 0.27]
    for col_idx, width in enumerate(col_widths):
        for row_idx in range(len(display_df) + 1):
            table[(row_idx, col_idx)].set_width(width)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["edge"])
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor(PALETTE["bg_alt"])
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
            cell.set_height(0.16)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=PALETTE["ink"])
            cell.set_height(0.15)

    finalize_figure(fig, out_path)


def _format_trend_fit_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _write_group_summary_outputs(
    *,
    df: pd.DataFrame,
    config: dict,
    group_col: str,
    target_col: str,
    target_name: str,
    tables_dir: Path,
    figures_dir: Path,
    profile_label: str,
) -> None:
    if group_col not in df.columns:
        return

    level_order = [
        str(level)
        for level in config.get("demographics", {}).get("categories", {}).get(group_col, [])
        if str(level) not in {"missing", "other"}
    ]

    group_df = df.dropna(subset=[group_col, target_col, "weight"]).copy()
    if group_df.empty:
        return
    group_df[group_col] = group_df[group_col].astype(str)
    group_df = group_df[group_df[group_col] != "missing"].copy()
    if group_df.empty:
        return

    rows: list[dict[str, object]] = []
    years_sorted = sorted(group_df["wave_year"].dropna().astype(int).unique().tolist())
    for year in years_sorted:
        year_df = group_df[group_df["wave_year"] == year]
        for level in level_order:
            level_df = year_df[year_df[group_col] == level]
            if level_df.empty:
                continue
            weights = level_df["weight"].astype(float)
            rows.append(
                {
                    "year": int(year),
                    group_col: level,
                    "n": int(len(level_df)),
                    "weighted_mean_rwp_score": float(
                        np.average(level_df[target_col].astype(float), weights=weights)
                    ),
                }
            )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return
    summary_df[group_col] = pd.Categorical(
        summary_df[group_col],
        categories=level_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values(["year", group_col]).reset_index(drop=True)

    csv_path = tables_dir / f"{group_col}_by_year_rwp.csv"
    summary_df.to_csv(csv_path, index=False)
    _write_group_summary_trend_figure(
        summary_df=summary_df,
        group_col=group_col,
        target_name=target_name,
        profile_label=profile_label,
        out_path=figures_dir / f"{group_col}_by_year_rwp.png",
    )


def _write_group_summary_trend_figure(
    *,
    summary_df: pd.DataFrame,
    group_col: str,
    target_name: str,
    profile_label: str,
    out_path: Path,
) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df[group_col] = plot_df[group_col].astype(str)
    years_sorted = sorted(plot_df["year"].dropna().astype(int).unique().tolist())
    levels = [level for level in plot_df[group_col].astype(str).unique().tolist()]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    cmap = plt.get_cmap("tab10")
    end_labels: list[dict[str, object]] = []

    score_min = None
    score_max = None
    for idx, level in enumerate(levels):
        sub = plot_df[plot_df[group_col] == level].sort_values("year")
        if sub.empty:
            continue
        color = cmap(idx % 10)
        label = _format_predictor_label(level)
        ax.plot(
            sub["year"],
            sub["weighted_mean_rwp_score"],
            marker="o",
            linewidth=2.2,
            color=color,
        )
        end_row = sub.iloc[-1]
        end_labels.append(
            {
                "x": float(end_row["year"]),
                "y": float(end_row["weighted_mean_rwp_score"]),
                "label": label,
                "color": color,
            }
        )
        level_min = float(sub["weighted_mean_rwp_score"].min())
        level_max = float(sub["weighted_mean_rwp_score"].max())
        score_min = level_min if score_min is None else min(score_min, level_min)
        score_max = level_max if score_max is None else max(score_max, level_max)

    group_title = group_col.replace("_", " ").title()
    style_axes(
        ax,
        title=with_profile_prefix(
            profile_label, f"{group_title} differences in mean {target_name}"
        ),
        xlabel="Year",
        ylabel=f"Weighted mean {target_name}",
        grid_axis="both",
        integer_x=True,
        title_loc="center",
    )
    if years_sorted:
        ax.set_xticks(years_sorted)
        ax.set_xlim(min(years_sorted) - 0.2, max(years_sorted) + 0.45)
    if score_min is not None and score_max is not None:
        # Use the same complete score scale across all subgroup comparisons.
        ax.set_ylim(0.0, 1.0)
    if score_min is not None and score_max is not None and score_min < 0 < score_max:
        ax.axhline(0, color=PALETTE["gray"], linewidth=1.0, linestyle=":", alpha=0.8)
    _annotate_line_end_labels(ax, end_labels, x_offset=0.12)

    finalize_figure(fig, out_path)

# builds the model frame used for modelling and trends

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .logging_utils import log_event, setup_logging
from .output_layout import (
    build_model_frame_dir,
    build_model_frame_meta_path,
    build_model_frame_parquet_path,
)




def build_model_frame(config_path: str = "config/code_config.json") -> None:
    _build_model_frame_impl(config_path)


def _compute_wave_coverage(
    derived_dir: Path,
    waves: list[dict],
    predictors: list[str],
    structural_missing_rate: float,
    exclude_years: set[int] | None = None,
) -> dict:
    coverage = {pred: {"waves_present": 0, "missing_rates": {}} for pred in predictors}
    for wave in waves:
        year = wave["year"]
        if exclude_years and year in exclude_years:
            continue
        demo_path = derived_dir / "demographics" / f"demographics_{year}.parquet"
        if not demo_path.exists():
            continue
        demo = pd.read_parquet(demo_path)
        for pred in predictors:
            if pred not in demo.columns:
                continue
            missing_rate = float(demo[pred].isna().mean())
            coverage[pred]["missing_rates"][str(year)] = missing_rate
            if missing_rate < structural_missing_rate:
                coverage[pred]["waves_present"] += 1
    return coverage
def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
def _resolve_track_config(track: str, config: dict) -> dict:
    modelling_cfg = config.get("modelling", {})
    trend_cfg = config.get("trend_analysis", {})
    if track == "main":
        default_min = modelling_cfg.get("min_wave_coverage", 0)
        default_exclude = []
    else:
        default_min = trend_cfg.get("min_wave_coverage", modelling_cfg.get("min_wave_coverage", 0))
        default_exclude = trend_cfg.get("exclude_features", [])

    track_cfg = (config.get("predictor_tracks") or {}).get(track, {})
    min_wave_coverage = track_cfg.get("min_wave_coverage", default_min)
    if min_wave_coverage is None:
        min_wave_coverage = default_min
    include_features = track_cfg.get("include_features", [])
    if include_features is None:
        include_features = []
    exclude_features = track_cfg.get("exclude_features", default_exclude)
    if exclude_features is None:
        exclude_features = default_exclude

    return {
        "min_wave_coverage": min_wave_coverage,
        "include_features": include_features,
        "exclude_features": exclude_features,
    }


def _build_predictor_tracks(
    config: dict,
    base_predictors: list[str],
    coverage: dict,
    available_columns: list[str],
) -> tuple[list[str], dict]:
    track_meta: dict[str, dict] = {}
    all_predictors: list[str] = []
    for track in ("main", "trend"):
        track_cfg = _resolve_track_config(track, config)
        exclude_features = set(track_cfg.get("exclude_features") or [])
        if track == "trend":
            exclude_features |= set(config.get("trend_analysis", {}).get("exclude_features", []))
        dropped_structural = [
            pred
            for pred in base_predictors
            if coverage.get(pred, {}).get("waves_present", 0)
            < int(track_cfg["min_wave_coverage"])
        ]
        predictors = [pred for pred in base_predictors if pred not in dropped_structural]
        include_features = track_cfg.get("include_features") or []
        if not isinstance(include_features, list):
            include_features = [include_features]
        predictors.extend(include_features)
        predictors = _dedupe(predictors)
        if exclude_features:
            predictors = [pred for pred in predictors if pred not in exclude_features]

        missing_from_frame = [pred for pred in predictors if pred not in available_columns]
        predictors = [pred for pred in predictors if pred in available_columns]

        track_meta[track] = {
            "predictors": predictors,
            "min_wave_coverage": track_cfg["min_wave_coverage"],
            "include_features": include_features,
            "exclude_features": sorted(exclude_features),
            "dropped_structural": dropped_structural,
            "missing_from_frame": missing_from_frame,
        }
        all_predictors.extend(predictors)

    all_predictors = _dedupe(all_predictors)
    return all_predictors, track_meta


def get_predictors_for_track(
    meta: dict | None, config: dict, track: str
) -> tuple[list[str], dict]:
    if meta:
        track_meta = meta.get("predictor_tracks", {})
        if isinstance(track_meta, dict) and track in track_meta:
            return track_meta.get(track, {}).get("predictors", []), track_meta.get(track, {})
        predictors = meta.get("predictors", [])
        if predictors:
            return predictors, {}
    predictors = config["demographics"]["numeric"] + config["demographics"]["categorical"]
    return predictors, {}


def _resolve_target_column(
    df: pd.DataFrame,
    modelling_cfg: dict | None = None,
    logger=None,
    *,
    stage: str,
    override_target: str | None = None,
) -> str:
    cfg = modelling_cfg or {}
    requested_target = (
        override_target
        if override_target is not None
        else cfg.get("target_column", "rwp_score")
    )
    target = str(requested_target or "rwp_score")
    if target in df.columns:
        return target

    fallback_target = None
    for candidate in (
        "rwp_score",
    ):
        if candidate in df.columns:
            fallback_target = candidate
            break
    if fallback_target is None:
        raise KeyError(
            f"Requested target column '{target}' is not present and no fallback target exists."
        )
    if logger is not None:
        log_event(
            logger,
            f"{stage}_target_fallback",
            requested_target=target,
            fallback_target=fallback_target,
        )
    return fallback_target




def _build_model_frame_impl(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    derived_dir = Path(config["paths"]["derived_dir"])
    build_model_frame_dir(derived_dir).mkdir(parents=True, exist_ok=True)
    log_path = derived_dir / "logs" / "model_frame.log"
    logger = setup_logging(log_path)
    holdout_years = set(config.get("holdout_years", []))
    modelling_cfg = config.get("modelling", {})

    overlap_warn_rate = modelling_cfg.get("merge_key_overlap_warn_rate", 0.999)
    overlap_fail_rate = modelling_cfg.get("merge_key_overlap_fail_rate", 0.95)
    try:
        overlap_warn_rate = float(overlap_warn_rate)
    except (TypeError, ValueError):
        overlap_warn_rate = 0.999
    try:
        overlap_fail_rate = float(overlap_fail_rate)
    except (TypeError, ValueError):
        overlap_fail_rate = 0.95
    overlap_warn_rate = min(max(overlap_warn_rate, 0.0), 1.0)
    overlap_fail_rate = min(max(overlap_fail_rate, 0.0), 1.0)
    if overlap_warn_rate < overlap_fail_rate:
        overlap_warn_rate = overlap_fail_rate
    key_overlap_rows = []


    frames = []
    for wave in config["waves"]:
        year = wave["year"]
        demo_path = derived_dir / "demographics" / f"demographics_{year}.parquet"
        score_path = derived_dir / "scores" / f"wave_scores_{year}.parquet"
        demo = pd.read_parquet(demo_path)
        scores = pd.read_parquet(score_path)
        key_cols = ["respondent_id", "wave_year"]
        demo_dup_keys = int(demo.duplicated(key_cols).sum())
        scores_dup_keys = int(scores.duplicated(key_cols).sum())
        if demo_dup_keys or scores_dup_keys:
            log_event(
                logger,
                "model_frame_merge_key_violation",
                wave_year=int(year),
                demo_dup_keys=demo_dup_keys,
                scores_dup_keys=scores_dup_keys,
                demo_rows=int(len(demo)),
                scores_rows=int(len(scores)),
            )
            raise ValueError(
                "Duplicate merge keys before model-frame assembly "
                f"(year={year}, demo_dup_keys={demo_dup_keys}, scores_dup_keys={scores_dup_keys})"
            )
        demo_keys = demo[key_cols]
        score_keys = scores[key_cols]
        overlap_n = int(
            demo_keys.merge(
                score_keys,
                on=key_cols,
                how="inner",
                validate="one_to_one",
            ).shape[0]
        )
        demo_rows = int(len(demo))
        score_rows = int(len(scores))
        demo_overlap_rate = float(overlap_n / demo_rows) if demo_rows else 1.0
        score_overlap_rate = float(overlap_n / score_rows) if score_rows else 1.0
        overlap_rate_min = float(min(demo_overlap_rate, score_overlap_rate))
        key_overlap_rows.append(
            {
                "year": int(year),
                "demo_rows": demo_rows,
                "scores_rows": score_rows,
                "overlap_rows": overlap_n,
                "demo_overlap_rate": demo_overlap_rate,
                "scores_overlap_rate": score_overlap_rate,
                "overlap_rate_min": overlap_rate_min,
                "warn_threshold": float(overlap_warn_rate),
                "fail_threshold": float(overlap_fail_rate),
            }
        )
        if overlap_rate_min < overlap_warn_rate:
            log_event(
                logger,
                "model_frame_merge_key_overlap_warning",
                wave_year=int(year),
                overlap_rows=overlap_n,
                demo_rows=demo_rows,
                scores_rows=score_rows,
                demo_overlap_rate=demo_overlap_rate,
                scores_overlap_rate=score_overlap_rate,
                overlap_rate_min=overlap_rate_min,
                warn_threshold=float(overlap_warn_rate),
                fail_threshold=float(overlap_fail_rate),
            )
        if overlap_rate_min < overlap_fail_rate:
            log_event(
                logger,
                "model_frame_merge_key_overlap_failure",
                wave_year=int(year),
                overlap_rows=overlap_n,
                demo_rows=demo_rows,
                scores_rows=score_rows,
                demo_overlap_rate=demo_overlap_rate,
                scores_overlap_rate=score_overlap_rate,
                overlap_rate_min=overlap_rate_min,
                fail_threshold=float(overlap_fail_rate),
            )
            raise ValueError(
                "Low merge-key overlap before model-frame assembly "
                f"(year={year}, overlap_rate_min={overlap_rate_min:.6f}, "
                f"demo_overlap_rate={demo_overlap_rate:.6f}, "
                f"scores_overlap_rate={score_overlap_rate:.6f})"
            )
        merged = demo.merge(
            scores,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)



    row_loss_rows = []
    df_baseline = df
    row_loss_target_requested = str(
        modelling_cfg.get("target_column", "rwp_score")
    )
    row_loss_target = _resolve_target_column(
        df_baseline,
        modelling_cfg,
        logger,
        stage="model_frame_row_loss",
        override_target=row_loss_target_requested,
    )
    for year in sorted(df_baseline["wave_year"].unique()):
        df_year = df_baseline[df_baseline["wave_year"] == year]
        total_rows = int(len(df_year))
        missing_rwp_top_20 = (
            int(df_year["RWP_top_20"].isna().sum())
            if "RWP_top_20" in df_year.columns
            else None
        )
        missing_target = int(df_year[row_loss_target].isna().sum())
        missing_weight = int(df_year["weight"].isna().sum())
        missing_any = int(df_year[[row_loss_target, "weight"]].isna().any(axis=1).sum())
        row_loss_rows.append(
            {
                "year": int(year),
                "target_column": row_loss_target,
                "total_rows": total_rows,
                "missing_rwp_top_20": missing_rwp_top_20,
                "missing_target": missing_target,
                "missing_weight": missing_weight,
                "missing_target_or_weight": missing_any,
                "missing_rwp_top_20_or_weight": missing_any if row_loss_target == "RWP_top_20" else None,
                "rows_after_weight_filter": total_rows - missing_weight,
                "rows_after_filter": total_rows - missing_any,
            }
        )

    df = df.dropna(subset=["weight"]).copy()



    if "year" not in df.columns and "wave_year" in df.columns:
        df["year"] = df["wave_year"]
        log_event(
            logger,
            "model_frame_year_alias_created",
            source_column="wave_year",
            target_column="year",
        )


    base_predictors = config["demographics"]["numeric"] + config["demographics"]["categorical"]
    structural_missing_rate = modelling_cfg.get("structural_missing_rate", 1.0)
    coverage = _compute_wave_coverage(
        derived_dir,
        config.get("waves", []),
        base_predictors,
        structural_missing_rate,
        exclude_years=holdout_years,
    )

    all_predictors, predictor_tracks = _build_predictor_tracks(
        config, base_predictors, coverage, df.columns.tolist()
    )
    main_track = predictor_tracks.get("main", {})
    dropped_structural = main_track.get("dropped_structural", [])

    label_columns = [
        col
        for col in [
            "RWP_top_20",
            "rwp_score",
        ]
        if col in df.columns
    ]
    model_frame_columns = ["respondent_id", "wave_year", "weight"] + label_columns + all_predictors


    df = df.loc[:, list(dict.fromkeys(model_frame_columns))]


    miss_rows = []
    for year in sorted(df["wave_year"].unique()):
        df_year = df[df["wave_year"] == year]
        for pred in all_predictors:
            if pred not in df_year.columns:
                continue
            miss_rows.append(
                {
                    "year": int(year),
                    "predictor": pred,
                    "missing_rate": float(df_year[pred].isna().mean()),
                    "missing_count": int(df_year[pred].isna().sum()),
                }
            )

    categories = config["demographics"]["categories"]
    for col in config["demographics"]["categorical"]:
        if col in df.columns:
            cat_values = categories.get(col)
            if not cat_values:
                continue
            # Keep real NaNs here so train/trend missingness filters see the original coverage.
            # The modelling preprocessor adds the "missing" category after feature selection.
            df[col] = pd.Categorical(df[col], categories=cat_values)


    out_path = build_model_frame_parquet_path(derived_dir)
    df.to_parquet(out_path, index=False)

    meta = {
        "predictors": all_predictors,
        "base_predictors": base_predictors,
        "label_columns_available": label_columns,
        "default_target_column": row_loss_target,
        "model_year_column": "year",
        "wave_key_column": "wave_year",
        "predictor_tracks": predictor_tracks,
        "dropped_structural": dropped_structural,
        "min_wave_coverage": main_track.get("min_wave_coverage"),
        "coverage": coverage,
        "structural_missing_rate": structural_missing_rate,
        "merge_key_overlap": key_overlap_rows,
        "row_loss": row_loss_rows,
        "predictor_missingness_by_year": miss_rows,
    }
    meta_path = build_model_frame_meta_path(derived_dir)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_event(
        logger,
        "model_frame_complete",
        n_rows=len(df),
        output=str(out_path),
        meta=str(meta_path),
    )

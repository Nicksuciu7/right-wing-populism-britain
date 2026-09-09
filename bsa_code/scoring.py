# main scoring logic
# reads harmonised attitudes and writes scored files, diagnostics, and measurement outputs

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .attitude_map import load_attitude_map
from .config import load_config
from .logging_utils import log_event, setup_logging
from .output_layout import (
    build_diagnostics_dir,
    measurement_figures_dir,
    measurement_tables_dir,
)
from .plotting import (
    PALETTE,
    apply_plot_theme,
    finalize_figure,
    format_profile_label,
    style_axes,
    with_profile_prefix,
)
from .scoring_pca import (
    _align_component_signs,
    _assign_items_to_factors,
    _corr_from_frame,
    _detect_scree_elbow,
    _label_factors,
    _pca_from_corr,
    _rotate_loadings,
    _save_scree_plot,
)
from .scoring_reliability import _construct_reliability
from .scoring_scores import (
    _subscale_display_name,
    _weighted_mean_std,
)
from .scoring_thresholds import (
    _threshold_eligibility_masks,
    _thresholds_for_rule,
)


@dataclass
class ScoringSettings:
    derived_dir: Path
    outputs_dir: Path
    profile_label: str
    measurement_tables_dir: Path
    measurement_figures_dir: Path
    diagnostics_dir: Path
    labels_dir: Path
    scores_dir: Path
    model_dir: Path
    waves: list[dict]
    holdout_years: set[int]
    blocks: dict[str, dict[str, list[str]]]
    subscale_display_names: dict[str, str]
    components: dict[str, object]
    knee_max_components: int | None
    knee_fallback: int
    corr_min: float
    rotation: str
    loading_min: float
    loading_gap: float
    min_items_per_subscale: int
    subscale_min_items: dict[str, int]
    min_answered_share: dict[str, float]
    min_subscale_coverage: float | dict[str, float]
    threshold_rule: str
    threshold_weighted: bool
    standardize_enabled: bool
    standardize_use_weights: bool
    pca_reference_years: list[int]
    strict_shared_item_years: list[int]
    pca_structure_mode: str
    pca_structure_path: Path
    pca_min_components_with_items: int | dict[str, int]
    pca_component_guard_strict: bool
    min_prevalence_warn: float
    min_prevalence_fail: float
    strict_prevalence: bool
    project_seed: int


@dataclass
class MeasurementModel:
    subscale_defs: dict[str, dict[str, list[str]]]
    factor_loading_rows: list[dict[str, object]]
    factor_summary_rows: list[dict[str, object]]


def build_scores(config_path: str = "config/code_config.json") -> None:
    apply_plot_theme()
    config = load_config(config_path)
    settings = _parse_settings(config)
    logger = setup_logging(settings.derived_dir / "logs" / "scoring.log")

    attitude_map = _load_attitude_map_if_available(Path(config["paths"]["attitude_map"]))
    pooled_train, frames_by_year = _load_attitude_frames(settings)

    construct_items, block_report = _clean_item_blocks(
        pooled_train,
        settings,
        attitude_map,
    )

    measurement = _build_measurement_model(
        pooled_train=pooled_train,
        settings=settings,
        attitude_map=attitude_map,
        construct_items=construct_items,
        block_report=block_report,
        logger=logger,
    )

    pooled_scored, reliability = _score_waves(
        frames_by_year=frames_by_year,
        settings=settings,
        measurement=measurement,
    )

    label_outputs = _apply_labels(
        pooled_scored=pooled_scored,
        settings=settings,
        logger=logger,
    )

    _write_scoring_outputs(
        pooled_scored=pooled_scored,
        settings=settings,
        attitude_map=attitude_map,
        measurement=measurement,
        reliability=reliability,
        label_outputs=label_outputs,
    )

    log_event(
        logger,
        "scoring_complete",
        pooled=str(settings.scores_dir / "pooled_scored.parquet"),
        thresholds=str(settings.labels_dir / "thresholds.json"),
        prevalence=str(settings.labels_dir / "weighted_prevalence_by_year.csv"),
        respondent_scoring_method="observed_assigned_item_mean",
    )


def _parse_settings(config: dict) -> ScoringSettings:
    derived_dir = Path(config["paths"]["derived_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    tables_dir = measurement_tables_dir(outputs_dir)
    figures_dir = measurement_figures_dir(outputs_dir)
    diagnostics_dir = build_diagnostics_dir(derived_dir)
    labels_dir = derived_dir / "labels"
    scores_dir = derived_dir / "scores"
    model_dir = derived_dir / "models"
    for path in (
        outputs_dir,
        tables_dir,
        figures_dir,
        diagnostics_dir,
        labels_dir,
        scores_dir,
        model_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    measurement_cfg = config.get("measurement", {})
    scores_cfg = config.get("scores", {})
    threshold_cfg = config.get("thresholds", {})
    standardize_cfg = measurement_cfg.get("standardize_subscales", {})
    if isinstance(standardize_cfg, dict):
        standardize_enabled = bool(standardize_cfg.get("enabled", False))
        standardize_use_weights = bool(standardize_cfg.get("use_weights", True))
    else:
        standardize_enabled = bool(standardize_cfg)
        standardize_use_weights = True

    pca_structure_cfg = measurement_cfg.get("pca_structure", {})
    pca_structure_mode = "fit"
    pca_structure_path = model_dir / "pca_scoring_structure.json"
    if isinstance(pca_structure_cfg, dict):
        mode = str(pca_structure_cfg.get("mode", "fit")).strip().lower()
        if mode in {"fit", "reuse"}:
            pca_structure_mode = mode
        path_val = pca_structure_cfg.get("path")
        if isinstance(path_val, str) and path_val.strip():
            pca_structure_path = Path(path_val.strip())

    threshold_rule = str(
        measurement_cfg.get(
            "subscale_threshold_rule", threshold_cfg.get("rule", "top_quartile")
        )
    )

    waves = config.get("waves", [])
    subscale_display_names = {
        str(key): str(value)
        for key, value in measurement_cfg.get("subscale_display_names", {}).items()
        if value is not None
    }

    return ScoringSettings(
        derived_dir=derived_dir,
        outputs_dir=outputs_dir,
        profile_label=format_profile_label(config.get("project", {}).get("profile")),
        measurement_tables_dir=tables_dir,
        measurement_figures_dir=figures_dir,
        diagnostics_dir=diagnostics_dir,
        labels_dir=labels_dir,
        scores_dir=scores_dir,
        model_dir=model_dir,
        waves=waves,
        holdout_years={int(year) for year in config.get("holdout_years", [])},
        blocks=measurement_cfg.get("blocks", {}),
        subscale_display_names=subscale_display_names,
        components=measurement_cfg.get("components", {}),
        knee_max_components=measurement_cfg.get("components_knee_max_components", 10),
        knee_fallback=int(measurement_cfg.get("components_knee_fallback", 2)),
        corr_min=float(measurement_cfg.get("correlation_min_abs", 0.1)),
        rotation=str(measurement_cfg.get("rotation", "varimax")),
        loading_min=float(measurement_cfg.get("loading_min", 0.3)),
        loading_gap=float(measurement_cfg.get("loading_gap", 0.1)),
        min_items_per_subscale=int(measurement_cfg.get("min_items_per_subscale", 1)),
        subscale_min_items={
            str(key): int(value)
            for key, value in measurement_cfg.get("subscale_min_items", {}).items()
        },
        min_answered_share={
            str(construct): float(
                measurement_cfg.get("subscale_min_answered_share", {}).get(
                    str(construct),
                    scores_cfg.get("min_answered_share", 0.6),
                )
            )
            for construct in measurement_cfg.get("blocks", {})
        },
        min_subscale_coverage=measurement_cfg.get("min_subscale_coverage", 0.0),
        threshold_rule=threshold_rule,
        threshold_weighted=bool(
            measurement_cfg.get(
                "subscale_threshold_weighted", threshold_cfg.get("weighted", True)
            )
        ),
        standardize_enabled=standardize_enabled,
        standardize_use_weights=standardize_use_weights,
        pca_reference_years=sorted(
            {int(year) for year in measurement_cfg.get("pca_reference_years", [])}
        ),
        strict_shared_item_years=sorted(
            {int(year) for year in measurement_cfg.get("strict_shared_item_years", [])}
        ),
        pca_structure_mode=pca_structure_mode,
        pca_structure_path=pca_structure_path,
        pca_min_components_with_items=measurement_cfg.get(
            "pca_min_components_with_assigned_items", 1
        ),
        pca_component_guard_strict=bool(
            measurement_cfg.get("pca_component_guard_strict", False)
        ),
        min_prevalence_warn=float(scores_cfg.get("min_prevalence_warn", 0.0)),
        min_prevalence_fail=float(scores_cfg.get("min_prevalence_fail", 0.0)),
        strict_prevalence=bool(scores_cfg.get("strict_prevalence", False)),
        project_seed=int(config.get("project", {}).get("seed", 0)),
    )


def _load_attitude_map_if_available(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_attitude_map(path)


def _load_attitude_frames(
    settings: ScoringSettings,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    frames_all: list[pd.DataFrame] = []
    frames_train: list[pd.DataFrame] = []
    by_year: dict[int, pd.DataFrame] = {}
    for wave in settings.waves:
        year = int(wave["year"])
        frame = pd.read_parquet(settings.derived_dir / "attitudes" / f"attitudes_{year}.parquet")
        by_year[year] = frame
        frames_all.append(frame)
        if year not in settings.holdout_years:
            frames_train.append(frame)
    if not frames_train:
        frames_train = frames_all
    return pd.concat(frames_train, ignore_index=True), by_year


def _clean_item_blocks(
    pooled_train: pd.DataFrame,
    settings: ScoringSettings,
    attitude_map: dict,
) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
    construct_items: dict[str, list[str]] = {}
    block_report: dict[str, dict[str, object]] = {}

    for construct, blocks in settings.blocks.items():
        if not isinstance(blocks, dict) or not blocks:
            continue
        block_report[construct] = {}
        kept_all: set[str] = set()
        for block_name, raw_items in blocks.items():
            item_names = [str(item) for item in raw_items]
            item_cols = []
            for item in item_names:
                item_col = item if "__" in item else f"{construct}__{item}"
                if item_col in pooled_train.columns:
                    item_cols.append(item_col)

            dropped_low_corr: list[str] = []
            if len(item_cols) >= 2:
                corr = pooled_train[item_cols].corr().abs()
                mask = ~np.eye(len(item_cols), dtype=bool)
                max_corr = corr.where(mask).max(axis=1)
                dropped_low_corr = sorted(
                    col.split("__", 1)[1]
                    for col, value in max_corr.items()
                    if pd.notna(value) and float(value) < settings.corr_min
                )

            kept_cols = [
                col
                for col in item_cols
                if col.split("__", 1)[1] not in set(dropped_low_corr)
            ]
            kept_all.update(kept_cols)
            block_report[construct][block_name] = {
                "requested_items": [
                    str(item) if "__" in str(item) else f"{construct}__{item}"
                    for item in raw_items
                ],
                "kept_items": kept_cols,
                "dropped_low_correlation_items": dropped_low_corr,
            }

        filtered_items = sorted(kept_all)
        if settings.strict_shared_item_years:
            filtered_items = [
                item_col
                for item_col in filtered_items
                if _item_present_in_all_years(
                    item_col=item_col,
                    construct=construct,
                    years=settings.strict_shared_item_years,
                    attitude_map=attitude_map,
                )
            ]
        construct_items[construct] = filtered_items

    return construct_items, block_report


def _item_present_in_all_years(
    *,
    item_col: str,
    construct: str,
    years: list[int],
    attitude_map: dict,
) -> bool:
    if not years:
        return True
    source_construct = construct
    item_key = item_col
    if "__" in item_col:
        source_construct, item_key = item_col.split("__", 1)
    item = (
        attitude_map.get("constructs", {})
        .get(source_construct, {})
        .get("items", {})
        .get(item_key, {})
    )
    waves = item.get("waves", {})
    if not waves:
        return True
    available_years = {int(year) for year in waves.keys()}
    return all(int(year) in available_years for year in years)


def _build_measurement_model(
    *,
    pooled_train: pd.DataFrame,
    settings: ScoringSettings,
    attitude_map: dict,
    construct_items: dict[str, list[str]],
    block_report: dict[str, dict[str, object]],
    logger,
) -> MeasurementModel:
    pca_reference_pool = pooled_train
    reference_note = None
    if settings.pca_reference_years:
        ref_pool = pooled_train[pooled_train["wave_year"].isin(settings.pca_reference_years)].copy()
        if not ref_pool.empty:
            pca_reference_pool = ref_pool
        else:
            reference_note = "reference_years_not_found_fallback_to_scoring_pool"
            log_event(
                logger,
                "pca_reference_years_fallback",
                requested_years=settings.pca_reference_years,
            )
    reference_years_used = sorted(
        int(year) for year in pca_reference_pool["wave_year"].dropna().unique()
    )

    reused_payload = None
    if settings.pca_structure_mode == "reuse":
        if not settings.pca_structure_path.exists():
            raise FileNotFoundError(
                f"PCA structure reuse requested but file is missing: {settings.pca_structure_path}"
            )
        reused_payload = json.loads(settings.pca_structure_path.read_text(encoding="utf-8"))

    subscale_defs: dict[str, dict[str, list[str]]] = {}
    factor_loading_rows: list[dict[str, object]] = []
    factor_summary_rows: list[dict[str, object]] = []
    pca_structure_payload: dict[str, object] = {
        "schema_version": 2,
        "rotation": settings.rotation,
        "loading_min": settings.loading_min,
        "loading_gap": settings.loading_gap,
        "reference_years_used": reference_years_used,
        "reference_note": reference_note,
        "measurement_method": "pca_structure_with_observed_item_scoring",
        "pca_fitting_missing_data": {
            "item_missing_rule": "pairwise_complete_correlation_matrix",
            "respondent_scores_use_imputed_values": False,
        },
        "constructs": {},
    }
    for construct, items in construct_items.items():
        if not items:
            raise ValueError(f"No items available for construct after cleaning: {construct}")

        fitted = _fit_one_construct(
            construct=construct,
            items=items,
            pooled_reference=pca_reference_pool,
            settings=settings,
            reused_payload=reused_payload,
            block_report=block_report,
            logger=logger,
        )

        subscale_defs[construct] = fitted["subscale_map"]
        factor_loading_rows.extend(fitted["factor_loading_rows"])
        factor_summary_rows.extend(fitted["factor_summary_rows"])
        pca_structure_payload["constructs"][construct] = fitted["structure_payload"]

    settings.pca_structure_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pca_structure_path.write_text(
        json.dumps(pca_structure_payload, indent=2),
        encoding="utf-8",
    )

    return MeasurementModel(
        subscale_defs=subscale_defs,
        factor_loading_rows=factor_loading_rows,
        factor_summary_rows=factor_summary_rows,
    )

def _fit_one_construct(
    *,
    construct: str,
    items: list[str],
    pooled_reference: pd.DataFrame,
    settings: ScoringSettings,
    reused_payload: dict | None,
    block_report: dict[str, dict[str, object]],
    logger,
) -> dict[str, object]:
    payload_construct = {}
    if isinstance(reused_payload, dict):
        payload_construct = reused_payload.get("constructs", {}).get(construct, {})

    if settings.pca_structure_mode == "reuse" and payload_construct:
        return _reuse_construct_model(
            construct=construct,
            items=items,
            payload_construct=payload_construct,
            settings=settings,
        )

    X = pooled_reference[items]


    corr = _corr_from_frame(X, items)
    if corr is None:
        raise ValueError(f"Could not compute a valid PCA correlation matrix for {construct}")
    corr_diagnostics = _pca_correlation_diagnostics(X, corr, construct=construct, settings=settings)
    log_event(
        logger,
        "pca_correlation_diagnostics",
        construct=construct,
        item_count=corr_diagnostics["item_count"],
        pairwise_n_min=corr_diagnostics["pairwise_n_min"],
        pairwise_n_median=corr_diagnostics["pairwise_n_median"],
        pairwise_n_max=corr_diagnostics["pairwise_n_max"],
        nonfinite_correlation_count=corr_diagnostics["nonfinite_correlation_count"],
        min_eigenvalue=corr_diagnostics["min_eigenvalue"],
        negative_eigenvalue_count=corr_diagnostics["negative_eigenvalue_count"],
    )
    if int(corr_diagnostics["nonfinite_correlation_count"]) > 0:
        raise ValueError(
            f"PCA correlation matrix for {construct} contains non-finite values: "
            f"{corr_diagnostics['nonfinite_correlation_count']}"
        )
    if float(corr_diagnostics["min_eigenvalue"]) < -1e-8:
        raise ValueError(
            f"PCA correlation matrix for {construct} is not positive semi-definite "
            f"(min_eigenvalue={corr_diagnostics['min_eigenvalue']:.6g})."
        )
    _write_measurement_correlation_outputs(
        corr=corr,
        construct=construct,
        settings=settings,
    )
    eigvals_all, ratio_all, loadings_all = _pca_from_corr(corr, len(items))

    component_spec = settings.components.get(construct, 2)
    component_mode = "fixed"
    elbow_component = None
    if isinstance(component_spec, str) and component_spec.strip().lower() in {
        "knee",
        "elbow",
        "scree_knee",
        "auto_knee",
    }:
        component_mode = "knee"

    if component_mode == "knee":
        elbow_component = _detect_scree_elbow(
            ratio_all,
            max_components=settings.knee_max_components,
        )
        n_components = (
            int(elbow_component)
            if elbow_component is not None
            else int(settings.knee_fallback)
        )
    else:
        try:
            n_components = int(component_spec)
        except Exception:
            n_components = 2
    n_components = max(1, min(int(n_components), len(items)))
    selection_method = "scree_elbow" if component_mode == "knee" else "fixed_component_count"
    selection_fallback_used = bool(component_mode == "knee" and elbow_component is None)

    scree_path = settings.measurement_figures_dir / f"pca_scree_{construct}.png"
    _save_scree_plot(
        ratio_all,
        scree_path,
        construct,
        retained_components=n_components,
        max_components=settings.knee_max_components,
        show_elbow=(component_mode == "knee"),
    )

    rotated = _rotate_loadings(loadings_all[:, :n_components], settings.rotation)
    rotated, sign_alignment = _align_component_signs(items, rotated)
    assignments = _assign_items_to_factors(
        items,
        rotated,
        settings.loading_min,
        settings.loading_gap,
    )
    factor_labels = _label_factors(
        construct,
        items,
        assignments,
        {
            block_name: info.get("kept_items", [])
            for block_name, info in block_report.get(construct, {}).items()
        },
        n_components,
    )

    subscale_map: dict[str, list[str]] = {}
    for item in items:
        factor_idx = assignments[item].get("factor")
        if factor_idx is None:
            continue
        subscale = factor_labels.get(int(factor_idx))
        if subscale is None:
            continue
        subscale_map.setdefault(subscale, []).append(item)

    _write_measurement_component_correlation_outputs(
        corr=corr,
        subscale_map=subscale_map,
        settings=settings,
    )

    min_components_required = _min_components_required(settings, construct, n_components)
    assigned_components = {
        int(info["factor"])
        for info in assignments.values()
        if info.get("factor") is not None
    }
    if len(assigned_components) < min_components_required and settings.pca_component_guard_strict:
        raise ValueError(
            f"PCA component guard failed for {construct}: "
            f"assigned={len(assigned_components)} < required={min_components_required}"
        )

    factor_loading_rows: list[dict[str, object]] = []
    factor_summary_rows: list[dict[str, object]] = []
    for item_idx, item in enumerate(items):
        info = assignments[item]
        factor_idx = info.get("factor")
        assigned_subscale = factor_labels.get(int(factor_idx)) if factor_idx is not None else None
        row = {
            "construct": construct,
            "item": item,
            "assigned_component": int(factor_idx + 1) if factor_idx is not None else None,
            "assigned_subscale": assigned_subscale,
            "assigned_subscale_display": _subscale_display_name(
                assigned_subscale, settings.subscale_display_names
            ),
            "primary_loading": info.get("loading"),
            "primary_abs_loading": info.get("abs_loading"),
            "second_abs_loading": info.get("second_abs_loading"),
            "loading_gap": info.get("loading_gap"),
        }
        for comp_idx in range(rotated.shape[1]):
            row[f"component_{comp_idx + 1}"] = float(rotated[item_idx, comp_idx])
        factor_loading_rows.append(row)

    for comp_idx in range(n_components):
        comp_items = [
            item for item, info in assignments.items() if info.get("factor") == comp_idx
        ]
        factor_summary_rows.append(
            {
                "construct": construct,
                "component": int(comp_idx + 1),
                "component_label": factor_labels.get(
                    comp_idx, f"{construct}_factor{comp_idx + 1}"
                ),
                "component_label_display": _subscale_display_name(
                    factor_labels.get(comp_idx, f"{construct}_factor{comp_idx + 1}"),
                    settings.subscale_display_names,
                ),
                "assigned_items": int(len(comp_items)),
                "assigned_item_list": ";".join(comp_items),
                "selection_method": selection_method,
                "selection_requested": str(component_spec),
                "elbow_component": (
                    int(elbow_component) if elbow_component is not None else None
                ),
                "selection_fallback_used": bool(selection_fallback_used),
                "retained_components": int(n_components),
                "respondent_scoring_method": "observed_assigned_item_mean",
                "respondent_scores_use_imputed_values": False,
                "minimum_answered_share": float(settings.min_answered_share[construct]),
            }
        )

    return {
        "item_order": list(items),
        "rotated": rotated,
        "n_components": int(n_components),
        "sign_alignment": sign_alignment,
        "subscale_map": subscale_map,
        "assignments": {
            item: {**info, "subscale": factor_labels.get(info["factor"]) if info.get("factor") is not None else None}
            for item, info in assignments.items()
        },
        "factor_loading_rows": factor_loading_rows,
        "factor_summary_rows": factor_summary_rows,
        "structure_payload": {
            "items": list(items),
            "n_components": int(n_components),
            "rotation": settings.rotation,
            "explained_variance_ratio": [float(v) for v in ratio_all],
            "explained_variance": [float(v) for v in eigvals_all],
            "component_selection": {
                "method": selection_method,
                "requested": str(component_spec),
                "elbow_component": (
                    int(elbow_component) if elbow_component is not None else None
                ),
                "fallback_used": bool(selection_fallback_used),
                "fallback_components": int(settings.knee_fallback),
                "max_components_considered": settings.knee_max_components,
            },
            "rotated_loadings": np.asarray(rotated, dtype=float).tolist(),
            "factor_labels": {str(k): str(v) for k, v in factor_labels.items()},
            "assignments": {
                item: {
                    **info,
                    "subscale": factor_labels.get(info["factor"]) if info.get("factor") is not None else None,
                }
                for item, info in assignments.items()
            },
            "subscale_map": subscale_map,
            "respondent_scoring": {
                "method": "observed_assigned_item_mean",
                "uses_imputed_responses": False,
                "minimum_answered_share": float(settings.min_answered_share[construct]),
                "minimum_items_per_subscale_default": int(settings.min_items_per_subscale),
            },
            "pca_fitting_missing_data": {
                "item_missing_rule": "pairwise_complete_correlation_matrix",
                "respondent_scores_use_imputed_values": False,
            },
            "pca_correlation_diagnostics": corr_diagnostics,
            "item_correlation_matrix": {
                item: {
                    other_item: float(corr.loc[item, other_item])
                    for other_item in items
                }
                for item in items
            },
            "sign_alignment": sign_alignment,
        },
    }


def _reuse_construct_model(
    *,
    construct: str,
    items: list[str],
    payload_construct: dict,
    settings: ScoringSettings,
) -> dict[str, object]:
    stored_items = [str(item) for item in payload_construct.get("items", [])]
    if sorted(stored_items) != sorted(items):
        raise ValueError(
            f"PCA structure reuse item mismatch for {construct}: "
            f"scoring_items={items}; reused_items={stored_items}"
        )

    rotated = np.asarray(payload_construct.get("rotated_loadings", []), dtype=float)
    if rotated.ndim != 2 or rotated.shape[0] != len(stored_items):
        raise ValueError(f"Invalid reused loadings payload for {construct}")
    n_components = int(payload_construct.get("n_components", rotated.shape[1]))
    rotated, sign_alignment = _align_component_signs(stored_items, rotated)

    assignments_raw = payload_construct.get("assignments", {})
    factor_labels_raw = payload_construct.get("factor_labels", {})
    subscale_map = {
        str(name): [str(item) for item in cols]
        for name, cols in payload_construct.get("subscale_map", {}).items()
        if isinstance(cols, list)
    }
    factor_labels = {
        int(key): str(value)
        for key, value in factor_labels_raw.items()
        if str(key).isdigit()
    }
    assignments: dict[str, dict[str, object]] = {}
    for item in stored_items:
        info = assignments_raw.get(item, {})
        factor_idx = info.get("factor")
        if factor_idx is not None:
            factor_idx = int(factor_idx)
        assignments[item] = {
            "factor": factor_idx,
            "loading": info.get("loading"),
            "abs_loading": info.get("abs_loading"),
            "second_abs_loading": info.get("second_abs_loading"),
            "loading_gap": info.get("loading_gap"),
            "subscale": factor_labels.get(factor_idx) if factor_idx is not None else None,
        }

    factor_loading_rows: list[dict[str, object]] = []
    factor_summary_rows: list[dict[str, object]] = []
    for item_idx, item in enumerate(stored_items):
        info = assignments[item]
        row = {
            "construct": construct,
            "item": item,
            "assigned_component": int(info["factor"] + 1) if info.get("factor") is not None else None,
            "assigned_subscale": info.get("subscale"),
            "assigned_subscale_display": _subscale_display_name(
                info.get("subscale"), settings.subscale_display_names
            ),
            "primary_loading": info.get("loading"),
            "primary_abs_loading": info.get("abs_loading"),
            "second_abs_loading": info.get("second_abs_loading"),
            "loading_gap": info.get("loading_gap"),
        }
        for comp_idx in range(rotated.shape[1]):
            row[f"component_{comp_idx + 1}"] = float(rotated[item_idx, comp_idx])
        factor_loading_rows.append(row)

    for comp_idx in range(n_components):
        comp_items = [item for item, info in assignments.items() if info.get("factor") == comp_idx]
        factor_summary_rows.append(
            {
                "construct": construct,
                "component": int(comp_idx + 1),
                "component_label": factor_labels.get(
                    comp_idx, f"{construct}_factor{comp_idx + 1}"
                ),
                "component_label_display": _subscale_display_name(
                    factor_labels.get(comp_idx, f"{construct}_factor{comp_idx + 1}"),
                    settings.subscale_display_names,
                ),
                "assigned_items": int(len(comp_items)),
                "assigned_item_list": ";".join(comp_items),
            }
        )

    scree_path = settings.measurement_figures_dir / f"pca_scree_{construct}.png"
    explained_ratio = np.asarray(payload_construct.get("explained_variance_ratio", []), dtype=float)
    if explained_ratio.size:
        _save_scree_plot(
            explained_ratio,
            scree_path,
            construct,
            retained_components=n_components,
            max_components=settings.knee_max_components,
            show_elbow=True,
        )

    corr_payload = payload_construct.get("item_correlation_matrix")
    if isinstance(corr_payload, dict):
        corr = pd.DataFrame(corr_payload)
        corr = corr.loc[stored_items, stored_items]
        _write_measurement_correlation_outputs(
            corr=corr,
            construct=construct,
            settings=settings,
        )
        _write_measurement_component_correlation_outputs(
            corr=corr,
            subscale_map=subscale_map,
            settings=settings,
        )

    component_selection = payload_construct.get("component_selection", {})
    selection_method = str(component_selection.get("method", "reused_structure"))
    selection_requested = str(component_selection.get("requested", "reused_structure"))
    elbow_component = component_selection.get("elbow_component")
    try:
        elbow_component = int(elbow_component) if elbow_component is not None else None
    except Exception:
        elbow_component = None
    selection_fallback_used = bool(component_selection.get("fallback_used", False))

    return {
        "item_order": stored_items,
        "rotated": rotated,
        "n_components": n_components,
        "sign_alignment": sign_alignment,
        "subscale_map": subscale_map,
        "assignments": assignments,
        "factor_loading_rows": factor_loading_rows,
        "factor_summary_rows": [
            {
                **row,
                "selection_method": selection_method,
                "selection_requested": selection_requested,
                "elbow_component": elbow_component,
                "selection_fallback_used": bool(selection_fallback_used),
                "retained_components": int(n_components),
                "respondent_scoring_method": "observed_assigned_item_mean",
                "respondent_scores_use_imputed_values": False,
                "minimum_answered_share": float(settings.min_answered_share[construct]),
            }
            for row in factor_summary_rows
        ],
        "structure_payload": payload_construct,
    }


def _min_components_required(
    settings: ScoringSettings, construct: str, n_components: int
) -> int:
    value = settings.pca_min_components_with_items
    if isinstance(value, dict):
        value = value.get(construct, value.get("default", 1))
    try:
        return max(1, min(int(value), int(n_components)))
    except Exception:
        return 1


def _score_waves(
    *,
    frames_by_year: dict[int, pd.DataFrame],
    settings: ScoringSettings,
    measurement: MeasurementModel,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pooled_scores: list[pd.DataFrame] = []
    reliability = {"pooled": {}, "per_wave": {}}

    for year in sorted(frames_by_year):
        df = frames_by_year[year]
        df_out = df[["respondent_id", "wave_year", "weight"]].copy()
        reliability["per_wave"][str(year)] = {}

        for construct, subscales in measurement.subscale_defs.items():
            for subscale, items in subscales.items():
                items_present = [
                    col for col in items if col in df.columns and df[col].notna().any()
                ]
                available_for_reliability = items_present
                if not items_present or len(items_present) < _min_items_for_subscale(settings, subscale):
                    df_out[subscale] = np.nan
                else:
                    answered = df[items_present].notna().sum(axis=1)
                    min_required = max(
                        1,
                        math.ceil(settings.min_answered_share[construct] * len(items_present)),
                    )
                    scores = df[items_present].mean(axis=1, skipna=True)
                    df_out[subscale] = scores.where(answered >= min_required)

                reliability["per_wave"][str(year)][subscale] = _construct_reliability(
                    df, available_for_reliability
                )

        subscale_cols = [
            subscale
            for construct_scales in measurement.subscale_defs.values()
            for subscale in construct_scales
            if subscale in df_out.columns
        ]

        if settings.standardize_enabled:
            weights = df_out["weight"] if settings.standardize_use_weights else None
            for subscale in subscale_cols:
                mean, std = _weighted_mean_std(df_out[subscale], weights)
                if mean is None or std in (None, 0.0):
                    continue
                df_out[subscale] = (df_out[subscale] - mean) / std

        _build_score_columns(
            df_out=df_out,
            settings=settings,
            measurement=measurement,
        )
        pooled_scores.append(df_out)

    pooled_scored = pd.concat(pooled_scores, ignore_index=True)

    pooled_source = pd.concat(frames_by_year.values(), ignore_index=True)
    for construct, subscales in measurement.subscale_defs.items():
        for subscale, items in subscales.items():
            available = [col for col in items if col in pooled_source.columns]
            reliability["pooled"][subscale] = _construct_reliability(pooled_source, available)

    return pooled_scored, reliability


def _min_items_for_subscale(settings: ScoringSettings, subscale: str) -> int:
    return max(1, int(settings.subscale_min_items.get(subscale, settings.min_items_per_subscale)))


def _coverage_threshold(settings: ScoringSettings, construct: str) -> float:
    if isinstance(settings.min_subscale_coverage, dict):
        return float(settings.min_subscale_coverage.get(construct, 0.0))
    return float(settings.min_subscale_coverage or 0.0)


def _build_score_columns(
    *,
    df_out: pd.DataFrame,
    settings: ScoringSettings,
    measurement: MeasurementModel,
) -> None:
    coverage = {
        subscale: float(df_out[subscale].notna().mean())
        for construct_scales in measurement.subscale_defs.values()
        for subscale in construct_scales
        if subscale in df_out.columns
    }

    def _mean_or_nan(columns: list[str]) -> pd.Series:
        if not columns:
            return pd.Series(np.nan, index=df_out.index)
        return df_out[columns].mean(axis=1, skipna=True)

    construct_scores: dict[str, pd.Series] = {}
    for construct, subscales in measurement.subscale_defs.items():
        kept_subscales = [
            subscale
            for subscale in subscales
            if subscale in df_out.columns
            and coverage.get(subscale, 0.0) >= _coverage_threshold(settings, construct)
        ]
        construct_scores[construct] = _mean_or_nan(kept_subscales)

    if "rwp" not in construct_scores:
        raise KeyError("Expected a single active 'rwp' construct for score building.")
    df_out["rwp_score"] = construct_scores["rwp"]


def _apply_labels(
    *,
    pooled_scored: pd.DataFrame,
    settings: ScoringSettings,
    logger,
) -> dict[str, object]:
    threshold_source = pooled_scored
    if settings.holdout_years:
        threshold_source = pooled_scored[~pooled_scored["wave_year"].isin(settings.holdout_years)]
        if threshold_source.empty:
            threshold_source = pooled_scored

    threshold_cols = ["rwp_score"] if "rwp_score" in pooled_scored.columns else []
    if not threshold_cols:
        raise KeyError("rwp_score is missing from pooled_scored and cannot be labelled.")

    eligibility = _threshold_eligibility_masks(
        threshold_source,
        threshold_cols,
        weight_col="weight",
        weighted=settings.threshold_weighted,
    )
    main_thresholds = _thresholds_for_rule(
        settings.threshold_rule,
        threshold_source,
        threshold_cols,
        "weight",
        settings.threshold_weighted,
        seed=settings.project_seed,
        eligibility_masks=eligibility,
    )

    threshold_variants = {settings.threshold_rule: main_thresholds}
    idx_threshold = main_thresholds.get("rwp_score")
    pooled_scored["RWP_top_20"] = (
        pooled_scored["rwp_score"] >= idx_threshold
    ).where(pooled_scored["rwp_score"].notna()) if idx_threshold is not None else pd.Series(pd.NA, index=pooled_scored.index)
    pooled_scored["RWP_top_20"] = pooled_scored["RWP_top_20"].astype("Int64")

    prevalence_rows = _prevalence_rows(pooled_scored, label_col="RWP_top_20")
    prevalence_checks = _prevalence_check_rows(
        pooled_scored,
        label_col="RWP_top_20",
        warn_threshold=settings.min_prevalence_warn,
        fail_threshold=settings.min_prevalence_fail,
    )
    if settings.strict_prevalence:
        failing_years = [
            str(row["year"])
            for row in prevalence_checks
            if row["status"] in {"warn", "fail"}
        ]
        if failing_years:
            raise ValueError(
                "Label prevalence below threshold in year(s): " + ", ".join(failing_years)
            )
    if any(row["status"] in {"warn", "fail"} for row in prevalence_checks):
        log_event(
            logger,
            "label_prevalence_warning",
            rows=prevalence_checks,
        )

    thresholds_payload = {
        "primary_threshold_rule": settings.threshold_rule,
        "threshold_weighted": bool(settings.threshold_weighted),
        "threshold_years": sorted(
            int(year) for year in threshold_source["wave_year"].dropna().unique()
        ),
        "thresholds": threshold_variants,
        "score_column": "rwp_score",
        "label_column": "RWP_top_20",
    }

    return {
        "thresholds_payload": thresholds_payload,
        "prevalence_rows": prevalence_rows,
        "prevalence_checks": prevalence_checks,
    }


def _prevalence_rows(df: pd.DataFrame, *, label_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in sorted(df["wave_year"].dropna().astype(int).unique()):
        sub = df[df["wave_year"] == year].dropna(subset=[label_col, "weight"])
        if sub.empty:
            prevalence = None
            unweighted = None
        else:
            prevalence = float((sub[label_col] * sub["weight"]).sum() / sub["weight"].sum())
            unweighted = float(sub[label_col].mean())
        rows.append(
            {
                "year": int(year),
                "weighted_prevalence": prevalence,
                "unweighted_prevalence": unweighted,
            }
        )
    return rows


def _prevalence_check_rows(
    df: pd.DataFrame,
    *,
    label_col: str,
    warn_threshold: float,
    fail_threshold: float,
) -> list[dict[str, object]]:
    rows = _prevalence_rows(df, label_col=label_col)
    out = []
    for row in rows:
        weighted = row["weighted_prevalence"]
        status = "ok"
        if weighted is not None:
            if fail_threshold and weighted <= fail_threshold:
                status = "fail"
            elif warn_threshold and weighted <= warn_threshold:
                status = "warn"
        out.append(
            {
                "year": row["year"],
                "weighted_prevalence": weighted,
                "unweighted_prevalence": row["unweighted_prevalence"],
                "status": status,
            }
        )
    return out


def _write_scoring_outputs(
    *,
    pooled_scored: pd.DataFrame,
    settings: ScoringSettings,
    attitude_map: dict,
    measurement: MeasurementModel,
    reliability: dict[str, object],
    label_outputs: dict[str, object],
) -> None:
    pooled_scored.to_parquet(settings.scores_dir / "pooled_scored.parquet", index=False)
    for year in sorted(pooled_scored["wave_year"].dropna().astype(int).unique()):
        pooled_scored[pooled_scored["wave_year"] == year].to_parquet(
            settings.scores_dir / f"wave_scores_{int(year)}.parquet",
            index=False,
        )

    (settings.diagnostics_dir / "attitude_subscale_definitions.json").write_text(
        json.dumps(measurement.subscale_defs, indent=2),
        encoding="utf-8",
    )
    (settings.diagnostics_dir / "attitude_subscale_reliability.json").write_text(
        json.dumps(reliability, indent=2),
        encoding="utf-8",
    )
    (settings.labels_dir / "thresholds.json").write_text(
        json.dumps(label_outputs["thresholds_payload"], indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(label_outputs["prevalence_rows"]).to_csv(
        settings.labels_dir / "weighted_prevalence_by_year.csv",
        index=False,
    )

    final_items = _build_measurement_items_final(
        attitude_map=attitude_map,
        measurement=measurement,
        settings=settings,
    )
    final_items.to_csv(
        settings.measurement_tables_dir / "measurement_items_final.csv",
        index=False,
    )

    if measurement.factor_loading_rows:
        pd.DataFrame(measurement.factor_loading_rows).to_csv(
            settings.diagnostics_dir / "measurement_pca_factor_loadings.csv",
            index=False,
        )
    if measurement.factor_summary_rows:
        factor_summary_df = pd.DataFrame(measurement.factor_summary_rows)
        factor_summary_df.to_csv(
            settings.measurement_tables_dir / "measurement_pca_factor_summary.csv",
            index=False,
        )
        _write_measurement_table_figure(
            df=_build_measurement_component_summary_figure_df(factor_summary_df),
            title=with_profile_prefix(
                settings.profile_label, "Retained Measurement Components"
            ),
            out_path=settings.measurement_figures_dir / "measurement_component_summary_table.png",
        )

    used_items_df = final_items[final_items["used"]].copy()
    if not used_items_df.empty:
        _write_measurement_table_figure(
            df=_build_measurement_question_figure_df(
                used_items_df,
                holdout_years=settings.holdout_years,
            ),
            title=with_profile_prefix(
                settings.profile_label, "Retained Measurement Questions"
            ),
            out_path=settings.measurement_figures_dir / "measurement_component_questions_table.png",
        )


def _build_measurement_items_final(
    *,
    attitude_map: dict,
    measurement: MeasurementModel,
    settings: ScoringSettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    live_years = {int(wave["year"]) for wave in settings.waves}
    for construct, subscales in measurement.subscale_defs.items():
        for subscale, item_cols in subscales.items():
            used_cols = set(item_cols)
            for item_col in item_cols:
                source_construct = construct
                item_name = item_col
                if "__" in item_col:
                    source_construct, item_name = item_col.split("__", 1)
                item_map = (
                    attitude_map.get("constructs", {})
                    .get(source_construct, {})
                    .get("items", {})
                    .get(str(item_name), {})
                )
                wave_years = sorted(
                    int(year)
                    for year in item_map.get("waves", {}).keys()
                    if int(year) in live_years
                )
                rows.append(
                    {
                        "construct": construct,
                        "source_construct": source_construct,
                        "subscale": str(subscale),
                        "subscale_display": _subscale_display_name(
                            str(subscale), settings.subscale_display_names
                        ),
                        "item_key": str(item_name),
                        "item_column": item_col,
                        "used": item_col in used_cols,
                        "wave_years": ";".join(str(year) for year in wave_years),
                        "n_waves": len(wave_years),
                    }
                )
    return pd.DataFrame(rows)


def _build_measurement_component_summary_figure_df(
    factor_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    out = factor_summary_df.copy()
    if "component_label" in out.columns:
        out = out.drop(columns=["component_label"])
    if "component_label_display" in out.columns:
        out = out.rename(columns={"component_label_display": "component_label"})
    if "assigned_item_list" in out.columns:
        out["items"] = out["assigned_item_list"].fillna("").apply(
            lambda value: sum(1 for part in str(value).split(";") if part)
        )
        out = out.drop(columns=["assigned_item_list"])
    keep = [
        "construct",
        "component_label",
        "items",
        "minimum_answered_share",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out.loc[:, keep].copy()
    if "minimum_answered_share" in out.columns:
        out = out.rename(columns={"minimum_answered_share": "min_answered_share"})
    for column in ("construct", "component_label"):
        if column in out.columns:
            out[column] = out[column].apply(_format_display_label)
    return out


def _build_measurement_question_figure_df(
    used_items_df: pd.DataFrame,
    *,
    holdout_years: set[int],
) -> pd.DataFrame:
    out = used_items_df.copy()
    if "construct" in out.columns:
        out["construct"] = out["construct"].apply(_format_display_label)
    if "subscale_display" in out.columns:
        out["subscale_display"] = out["subscale_display"].apply(_format_display_label)
    if "item_key" in out.columns:
        out["item_key"] = out["item_key"].apply(_format_item_key_for_display)
    if "wave_years" in out.columns:
        out["wave_years"] = out["wave_years"].apply(
            lambda value: _format_wave_years_for_display(value, holdout_years=holdout_years)
        )
    keep = ["construct", "subscale_display", "item_key", "wave_years"]
    keep = [col for col in keep if col in out.columns]
    out = out.loc[:, keep].copy()
    out = out.rename(columns={"subscale_display": "component", "item_key": "question"})
    return out


def _write_measurement_table_figure(
    *,
    df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    if df.empty:
        return

    display_df = df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].apply(_format_measurement_table_value)

    rows = display_df.values.tolist()
    headers = [
        " ".join(str(col).replace("_", " ").split()).title() for col in display_df.columns
    ]
    row_line_counts = []
    for row in rows:
        row_line_counts.append(
            max(
                1,
                *[
                    str(value).count("\n") + 1
                    for value in row
                ],
            )
        )
    total_line_units = sum(row_line_counts) + 1.4
    fig_height = max(3.8, 0.34 * total_line_units)

    columns = list(display_df.columns)
    has_question_column = "question" in columns
    if has_question_column:
        fig_width = 13.5
    elif len(headers) >= 4:
        fig_width = 11.0
    else:
        fig_width = 10.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=10)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        bbox=[0.0, 0.0, 1.0, 0.94],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.0)

    col_widths = _measurement_table_col_widths(columns)
    for col_idx, width in enumerate(col_widths):
        for row_idx in range(len(rows) + 1):
            table[(row_idx, col_idx)].set_width(width)

    header_height = 0.046
    for col_idx in range(len(headers)):
        table[(0, col_idx)].set_height(header_height)

    for row_idx, line_count in enumerate(row_line_counts, start=1):
        row_height = max(0.044, 0.028 * line_count)
        for col_idx in range(len(headers)):
            table[(row_idx, col_idx)].set_height(row_height)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["edge"])
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor(PALETTE["bg_alt"])
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        else:
            cell.set_facecolor("#FAFCFF" if row_idx % 2 == 0 else "white")
            cell.set_text_props(color=PALETTE["ink"])

    finalize_figure(fig, out_path)


def _pca_correlation_diagnostics(
    source: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    construct: str,
    settings: ScoringSettings,
) -> dict[str, object]:
    source = source.loc[:, list(corr.index)]
    valid = source.notna().astype(int)
    pairwise_n = valid.T.dot(valid)
    pairwise_n = pairwise_n.loc[list(corr.index), list(corr.columns)]
    pairwise_n.to_csv(settings.diagnostics_dir / f"pca_pairwise_n_{construct}.csv")

    n_matrix = pairwise_n.to_numpy(dtype=float)
    off_diag_mask = ~np.eye(n_matrix.shape[0], dtype=bool)
    off_diag_values = n_matrix[off_diag_mask] if n_matrix.size else np.asarray([])
    if off_diag_values.size:
        pairwise_n_min = int(np.nanmin(off_diag_values))
        pairwise_n_median = float(np.nanmedian(off_diag_values))
        pairwise_n_max = int(np.nanmax(off_diag_values))
    else:
        pairwise_n_min = pairwise_n_median = pairwise_n_max = None

    corr_arr = corr.to_numpy(dtype=float)
    nonfinite_count = int((~np.isfinite(corr_arr)).sum())
    corr_for_eig = np.nan_to_num(corr_arr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_for_eig = 0.5 * (corr_for_eig + corr_for_eig.T)
    np.fill_diagonal(corr_for_eig, 1.0)
    eigvals = np.linalg.eigvalsh(corr_for_eig)
    eigvals_sorted = np.sort(eigvals)
    min_eigenvalue = float(eigvals_sorted[0]) if eigvals_sorted.size else None
    negative_count = int((eigvals_sorted < -1e-8).sum())

    diagnostics = {
        "construct": construct,
        "item_count": int(len(corr.index)),
        "pairwise_n_min": pairwise_n_min,
        "pairwise_n_median": pairwise_n_median,
        "pairwise_n_max": pairwise_n_max,
        "nonfinite_correlation_count": nonfinite_count,
        "min_eigenvalue": min_eigenvalue,
        "negative_eigenvalue_count": negative_count,
        "eigenvalues_pre_clamp": [float(value) for value in eigvals_sorted.tolist()],
    }
    diag_path = settings.diagnostics_dir / f"pca_correlation_diagnostics_{construct}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def _write_measurement_correlation_outputs(
    *,
    corr: pd.DataFrame,
    construct: str,
    settings: ScoringSettings,
) -> None:
    corr = corr.copy()
    corr = corr.loc[list(corr.index), list(corr.columns)]
    corr.to_csv(settings.diagnostics_dir / f"pca_reference_correlation_matrix_{construct}.csv")
    _write_measurement_correlation_figure(
        corr=corr,
        title=with_profile_prefix(
            settings.profile_label,
            f"Retained Item Correlations ({construct.upper()})",
        ),
        out_path=settings.measurement_figures_dir / f"pca_correlation_matrix_{construct}.png",
    )


def _write_measurement_component_correlation_outputs(
    *,
    corr: pd.DataFrame,
    subscale_map: dict[str, list[str]],
    settings: ScoringSettings,
) -> None:
    extras_dir = settings.measurement_figures_dir / "extras"
    for subscale, items in sorted(subscale_map.items()):
        kept_items = [item for item in items if item in corr.index]
        if len(kept_items) < 2:
            continue
        display_name = _subscale_display_name(subscale, settings.subscale_display_names)
        _write_measurement_correlation_figure(
            corr=corr.loc[kept_items, kept_items],
            title=with_profile_prefix(
                settings.profile_label, f"{display_name.title()} Item Correlations"
            ),
            out_path=extras_dir / f"pca_correlation_matrix_{display_name}.png",
        )


def _write_measurement_correlation_figure(
    *,
    corr: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    if corr.empty:
        return

    labels = [
        _format_item_key_for_display(str(item).split("__", 1)[1] if "__" in str(item) else str(item))
        for item in corr.index
    ]
    matrix = corr.to_numpy(dtype=float)
    upper_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    masked_matrix = np.ma.array(matrix, mask=upper_mask)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(12.6, 10.8))
    im = ax.imshow(
        masked_matrix,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=48,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.2,
    )
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            if upper_mask[row_idx, col_idx]:
                continue
            value = float(matrix[row_idx, col_idx])
            text_color = "white" if abs(value) >= 0.55 else PALETTE["ink"]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color=text_color,
            )

    style_axes(
        ax,
        title=title,
        xlabel="Retained items",
        ylabel="Retained items",
        grid_axis="",
        title_loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Correlation")
    cbar.outline.set_edgecolor(PALETTE["edge"])
    cbar.outline.set_linewidth(0.8)

    fig.subplots_adjust(left=0.28, bottom=0.22, right=0.9, top=0.9)
    finalize_figure(fig, out_path)


def _format_measurement_table_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    text = str(value)
    wrap_width = 54
    if "," in text and len(text) > 48:
        wrap_width = 42
    if ";" in text and len(text) > 28:
        wrap_width = 24
    if len(text) > wrap_width:
        return textwrap.fill(
            text,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    return text


def _format_wave_years_for_display(value, *, holdout_years: set[int]) -> str:
    if value is None or pd.isna(value):
        return ""
    years = []
    for part in str(value).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            year = int(part)
        except ValueError:
            continue
        if year in holdout_years:
            continue
        years.append(str(year))
    return ", ".join(years)


def _format_display_label(value) -> str:
    text = " ".join(str(value).replace("_", " ").split())
    return text.title()


def _format_item_key_for_display(value) -> str:
    token_map = {
        "gov": "government",
        "mp": "MP",
        "mps": "MPs",
        "eu": "EU",
    }
    parts = []
    for token in str(value).split("_"):
        mapped = token_map.get(token.lower(), token)
        parts.append(mapped)
    text = " ".join(parts)
    if text:
        text = text[0].upper() + text[1:]
    return text


def _measurement_table_col_widths(columns: list[str]) -> list[float]:
    if columns == ["construct", "component_label", "items", "min_answered_share"]:
        return [0.18, 0.28, 0.16, 0.20]
    if columns == ["construct", "component", "question", "wave_years"]:
        return [0.14, 0.19, 0.47, 0.20]
    width = 1.0 / max(1, len(columns))
    return [width] * len(columns)

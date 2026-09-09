# threshold helpers for turning scores into labels
# used by the scoring stage and does not write outputs itself

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .recoding import weighted_quantile
from .scoring_scores import _weighted_mean_std


def _mixture_threshold(series: pd.Series, seed: int) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size < 50:
        return None
    values = values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=seed)
    gmm.fit(values)
    means = gmm.means_.flatten()
    idx_high = int(np.argmax(means))
    grid = np.linspace(values.min(), values.max(), 512).reshape(-1, 1)
    probs = gmm.predict_proba(grid)[:, idx_high]
    above = np.where(probs >= 0.5)[0]
    if above.size == 0:
        return float(np.mean(means))
    return float(grid[above[0], 0])


def _thresholds_for_rule(
    rule: str,
    scores_df: pd.DataFrame,
    subscales: list[str],
    weight_col: str,
    weighted: bool,
    seed: int,
    eligibility_masks: dict[str, pd.Series] | None = None,
) -> dict:
    quantile_map = {
        "top_quartile": 0.75,
        "top_20": 0.8,
        "top_tertile": 2.0 / 3.0,
    }
    if rule in quantile_map:
        q = quantile_map[rule]
        thresholds = {}
        for col in subscales:
            series = scores_df[col]
            mask = (
                eligibility_masks.get(col)
                if eligibility_masks is not None and col in eligibility_masks
                else series.notna()
            )
            series_use = series[mask]
            if series_use.empty:
                thresholds[col] = None
                continue
            if weighted:
                weights_use = pd.to_numeric(
                    scores_df.loc[mask, weight_col], errors="coerce"
                )
                thresholds[col] = weighted_quantile(
                    series_use,
                    weights_use,
                    [q],
                )[0]
            else:
                thresholds[col] = float(series_use.quantile(q))
        return thresholds

    if rule == "z_gt_1":
        thresholds = {}
        for col in subscales:
            mask = (
                eligibility_masks.get(col)
                if eligibility_masks is not None and col in eligibility_masks
                else scores_df[col].notna()
            )
            series_use = scores_df.loc[mask, col]
            weights_use = scores_df.loc[mask, weight_col] if weighted else None
            mean, std = _weighted_mean_std(series_use, weights_use)
            if mean is None or std in (None, 0.0):
                thresholds[col] = None
            else:
                thresholds[col] = float(mean + std)
        return thresholds

    if rule == "mixture":
        thresholds = {}
        for col in subscales:
            mask = (
                eligibility_masks.get(col)
                if eligibility_masks is not None and col in eligibility_masks
                else scores_df[col].notna()
            )
            series_use = scores_df.loc[mask, col]
            mix_thr = _mixture_threshold(series_use, seed)
            if mix_thr is None:
                if series_use.empty:
                    mix_thr = None
                elif weighted:
                    weights_use = pd.to_numeric(
                        scores_df.loc[mask, weight_col], errors="coerce"
                    )
                    mix_thr = weighted_quantile(series_use, weights_use, [0.8])[0]
                else:
                    mix_thr = float(series_use.quantile(0.8))
            thresholds[col] = mix_thr
        return thresholds

    raise ValueError(f"Unknown threshold rule: {rule}")


def _threshold_eligibility_masks(
    scores_df: pd.DataFrame,
    cols: list[str],
    *,
    weight_col: str,
    weighted: bool,
) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    weights = pd.to_numeric(scores_df[weight_col], errors="coerce") if weighted else None
    for col in cols:
        if col not in scores_df.columns:
            continue
        mask = scores_df[col].notna()
        if weights is not None:
            mask = mask & weights.notna()
        masks[col] = mask
    return masks

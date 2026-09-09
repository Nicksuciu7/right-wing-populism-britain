from __future__ import annotations

import math

import numpy as np
import pandas as pd


def weighted_quantile(
    values: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    quantiles: list[float],
) -> list[float]:
    quantiles = [float(q) for q in quantiles]
    if not quantiles:
        return []

    values_arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    weights_arr = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not mask.any():
        return [math.nan for _ in quantiles]

    values_use = values_arr[mask]
    weights_use = weights_arr[mask]
    order = np.argsort(values_use, kind="mergesort")
    values_use = values_use[order]
    weights_use = weights_use[order]

    cum_weights = np.cumsum(weights_use)
    total_weight = float(cum_weights[-1])
    if total_weight <= 0:
        return [math.nan for _ in quantiles]

    probs = cum_weights / total_weight
    return [
        float(np.interp(min(max(q, 0.0), 1.0), probs, values_use))
        for q in quantiles
    ]

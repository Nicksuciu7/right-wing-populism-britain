# shared score helpers for the scoring stage
# provides weighted summary stats and subscales

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _weighted_mean_std(values: pd.Series, weights: pd.Series | None) -> tuple[float | None, float | None]:
    series = pd.to_numeric(values, errors="coerce")
    if weights is None:
        mean = series.mean(skipna=True)
        std = series.std(skipna=True, ddof=0)
        return (float(mean) if pd.notna(mean) else None, float(std) if pd.notna(std) else None)

    weights = pd.to_numeric(weights, errors="coerce")
    mask = series.notna() & weights.notna()
    if not mask.any():
        return None, None
    v = series[mask].to_numpy()
    w = weights[mask].to_numpy()
    mean = float(np.average(v, weights=w))
    var = float(np.average((v - mean) ** 2, weights=w))
    std = math.sqrt(var)
    return mean, std


def _subscale_display_name(
    subscale_name: str | None, display_names: dict[str, str]
) -> str | None:
    if subscale_name is None:
        return None
    if subscale_name in display_names:
        return str(display_names[subscale_name])
    if subscale_name.startswith("populism_"):
        return subscale_name[len("populism_"):]
    if subscale_name.startswith("right_"):
        return subscale_name[len("right_"):]
    if subscale_name.startswith("rwp_"):
        return subscale_name[len("rwp_"):]
    return subscale_name

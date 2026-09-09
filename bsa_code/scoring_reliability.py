# reliability helpers for scored subscales
# used by scoring stage to calculate build diagnostics

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _cronbach_alpha(df: pd.DataFrame, cols: list[str]) -> float | None:
    if len(cols) < 2:
        return None
    data = df[cols].dropna()
    if data.shape[0] < 2:
        return None
    item_vars = data.var(ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0 or item_vars.isna().any():
        return None
    k = data.shape[1]
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def _omega_total_approx(df: pd.DataFrame, cols: list[str]) -> float | None:
    if len(cols) < 2:
        return None
    data = df[cols].dropna()
    if data.shape[0] < 2:
        return None
    try:
        X = data.to_numpy(dtype=float)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std
        fa = PCA(n_components=1)
        fa.fit(X)
        loadings = fa.components_[0]
        if np.sum(loadings) < 0:
            loadings = -loadings
        psi = np.var(X - np.outer(X @ loadings, loadings), axis=0)
        numerator = float(np.sum(loadings) ** 2)
        denom = numerator + float(np.sum(psi))
        if denom == 0:
            return None
        return float(numerator / denom)
    except Exception:
        return None


def _construct_reliability(df: pd.DataFrame, cols: list[str]) -> dict[str, float | int | None]:
    complete_cases = df[cols].dropna()
    return {
        "n_items": len(cols),
        "n_complete_cases": int(complete_cases.shape[0]),
        "cronbach_alpha": _cronbach_alpha(df, cols),
        "omega_total_approx": _omega_total_approx(df, cols),
    }

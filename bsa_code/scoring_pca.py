# pca helpers for the scoring stage
# fits rotated component structures, assigns items, and writes scree plots

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import PALETTE, finalize_figure, style_axes, style_legend


def _detect_scree_elbow(
    explained_ratio: np.ndarray,
    *,
    max_components: int | None = None,
) -> int | None:
    ratio = np.asarray(explained_ratio, dtype=float).reshape(-1)
    if ratio.size == 0:
        return None
    if max_components is not None:
        max_components = int(max_components)
        if max_components > 0:
            ratio = ratio[:max_components]
    if ratio.size <= 2:
        return int(np.argmax(ratio) + 1)

    x = np.arange(ratio.size, dtype=float)
    start = np.array([x[0], ratio[0]], dtype=float)
    end = np.array([x[-1], ratio[-1]], dtype=float)
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm <= 0:
        return int(np.argmax(ratio) + 1)

    pts = np.column_stack((x, ratio))
    distances = np.abs(
        line[0] * (start[1] - pts[:, 1]) - (start[0] - pts[:, 0]) * line[1]
    ) / norm
    distances[0] = -np.inf
    distances[-1] = -np.inf
    return int(np.argmax(distances) + 1)


def _save_scree_plot(
    explained_ratio: np.ndarray | list[float],
    out_path: Path,
    construct: str,
    *,
    weighted: bool = False,
    retained_components: int | None = None,
    max_components: int | None = None,
    show_elbow: bool = True,
) -> None:
    ratio = np.asarray(explained_ratio, dtype=float).reshape(-1)
    if ratio.size == 0:
        return
    if max_components is None:
        plot_components = ratio.size
    else:
        try:
            plot_components = int(max_components)
        except Exception:
            plot_components = ratio.size
        if plot_components <= 0:
            plot_components = ratio.size
        plot_components = min(plot_components, ratio.size)
    ratio = ratio[:plot_components]
    comps = np.arange(1, ratio.size + 1)
    elbow_component = _detect_scree_elbow(ratio) if show_elbow else None

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["bg"])

    ax.bar(
        comps,
        ratio,
        width=0.72,
        color=PALETTE["blue_light"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.6,
        label="Explained variance",
        zorder=2,
    )
    ax.plot(
        comps,
        ratio,
        color=PALETTE["blue"],
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        label="Scree",
        zorder=3,
    )

    if elbow_component is not None:
        y_elbow = float(ratio[elbow_component - 1])
        ax.axvline(
            elbow_component,
            color=PALETTE["red"],
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label="Elbow",
            zorder=1,
        )
        ax.scatter([elbow_component], [y_elbow], color=PALETTE["red"], s=26, zorder=5)
        y_offset = max(0.01, float(ratio.max()) * 0.14)
        y_annot = min(0.98, y_elbow + y_offset)
        ax.annotate(
            "Elbow",
            xy=(elbow_component, y_elbow),
            xytext=(elbow_component + 0.28, y_annot),
            textcoords="data",
            fontsize=8,
            color=PALETTE["ink"],
            arrowprops={"arrowstyle": "-", "color": PALETTE["gray"], "lw": 0.8},
            zorder=6,
        )

    if retained_components is not None:
        k = int(max(1, min(retained_components, ratio.size)))
        ax.axvline(
            k,
            color=PALETTE["gray"],
            linewidth=1.1,
            linestyle=":",
            alpha=0.9,
            label=f"Retained (k={k})",
            zorder=1,
        )
        ax.axvspan(0.5, k + 0.5, color=PALETTE["blue_light"], alpha=0.10, zorder=0)
    xticks = list(range(1, ratio.size + 1))
    ax.set_xticks(xticks)
    ax.set_xlim(0.4, ratio.size + 0.6)
    y_top = min(1.0, max(0.06, float(ratio.max()) * 1.35))
    ax.set_ylim(0.0, y_top)
    title_suffix = " (weighted)" if weighted else ""
    style_axes(
        ax,
        title=f"Scree plot{title_suffix}: {construct} (top {ratio.size})",
        xlabel="Component",
        ylabel="Variance",
        grid_axis="y",
        percent_y=True,
        integer_x=True,
        title_loc="center",
    )
    finalize_figure(fig, out_path)


def _varimax(loadings: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> np.ndarray:
    p, k = loadings.shape
    rotation = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        rotated = loadings @ rotation
        tmp = rotated ** 3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))
        u, s, vt = np.linalg.svd(loadings.T @ tmp)
        rotation = u @ vt
        d = float(s.sum())
        if d_old != 0 and d / d_old < 1.0 + tol:
            break
    return loadings @ rotation


def _rotate_loadings(loadings: np.ndarray, method: str) -> np.ndarray:
    method = (method or "none").lower()
    if method == "none":
        return loadings
    if method == "varimax":
        return _varimax(loadings)
    raise ValueError(f"Unsupported rotation: {method}")


def _align_component_signs(
    items: list[str],
    loadings: np.ndarray,
    *,
    anchor_items: list[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    aligned = np.array(loadings, dtype=float, copy=True)
    if aligned.ndim != 2 or aligned.size == 0:
        return aligned, []

    item_to_idx = {item: idx for idx, item in enumerate(items)}
    sign_meta: list[dict[str, object]] = []
    n_items = len(items)
    for comp_idx in range(aligned.shape[1]):
        column = aligned[:, comp_idx]
        top_idx = int(np.argmax(np.abs(column)))
        top_item = items[top_idx] if top_idx < n_items else f"item_{top_idx}"
        top_abs_loading = float(np.abs(column[top_idx]))

        anchor_source = "max_abs_loading"
        anchor_item = top_item
        if anchor_items is not None and comp_idx < len(anchor_items):
            candidate = str(anchor_items[comp_idx])
            if candidate in item_to_idx:
                anchor_item = candidate
                anchor_source = "fixed_anchor_item"
        anchor_idx = item_to_idx.get(anchor_item, top_idx)
        anchor_loading_before = float(column[anchor_idx])
        flip = anchor_loading_before < 0
        if flip:
            aligned[:, comp_idx] = -aligned[:, comp_idx]
        sign_meta.append(
            {
                "component": int(comp_idx + 1),
                "anchor_item": anchor_item,
                "anchor_source": anchor_source,
                "anchor_loading_before": anchor_loading_before,
                "anchor_loading_after": float(aligned[anchor_idx, comp_idx]),
                "top_abs_item": top_item,
                "top_abs_loading": top_abs_loading,
                "flipped": bool(flip),
            }
        )
    return aligned, sign_meta


def _corr_from_frame(df: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    valid_cols = [col for col in cols if col in df.columns and df[col].notna().any()]
    if len(valid_cols) < 2:
        return None
    data = df.loc[:, valid_cols].apply(pd.to_numeric, errors="coerce")
    corr = data.corr(min_periods=2)
    if corr.shape[0] < 2:
        return None
    return corr


def _pca_from_corr(
    corr: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corr_arr = np.asarray(corr, dtype=float)
    corr_arr = np.nan_to_num(corr_arr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_arr = 0.5 * (corr_arr + corr_arr.T)
    np.fill_diagonal(corr_arr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr_arr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0)
    total = float(eigvals.sum())
    ratio = eigvals / total if total > 0 else np.zeros_like(eigvals)
    n_components = max(1, min(n_components, corr_arr.shape[0]))
    loadings = eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])
    return eigvals, ratio, loadings


def _assign_items_to_factors(
    items: list[str],
    loadings: np.ndarray,
    loading_min: float,
    loading_gap: float,
) -> dict[str, dict[str, float | int | None]]:
    assignments: dict[str, dict[str, float | int | None]] = {}
    for idx, item in enumerate(items):
        row = loadings[idx]
        abs_vals = np.abs(row)
        order = np.argsort(abs_vals)[::-1]
        best = int(order[0]) if order.size else None
        second = int(order[1]) if order.size > 1 else None
        best_val = float(abs_vals[best]) if best is not None else 0.0
        second_val = float(abs_vals[second]) if second is not None else 0.0
        gap = best_val - second_val
        assigned = None
        if best is not None and best_val >= loading_min and gap >= loading_gap:
            assigned = best
        assignments[item] = {
            "factor": assigned,
            "loading": float(row[best]) if best is not None else None,
            "abs_loading": best_val,
            "second_abs_loading": second_val,
            "loading_gap": float(gap),
        }
    return assignments


def _label_factors(
    construct: str,
    items: list[str],
    assignments: dict[str, dict[str, float | int | None]],
    block_items: dict[str, list[str]],
    n_components: int,
) -> dict[int, str]:
    factor_labels: dict[int, str] = {}
    used_labels: set[str] = set()
    for factor_idx in range(n_components):
        assigned_items = [
            item for item in items if assignments.get(item, {}).get("factor") == factor_idx
        ]
        block_scores = []
        for block, cols in block_items.items():
            count = sum(1 for item in assigned_items if item in cols)
            loading_sum = sum(
                assignments[item].get("abs_loading", 0.0)
                for item in assigned_items
                if item in cols
            )
            block_scores.append((block, count, loading_sum))
        block_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        if block_scores and block_scores[0][1] > 0:
            label = f"{construct}_{block_scores[0][0]}"
        else:
            label = f"{construct}_factor{factor_idx + 1}"





        if label in used_labels and construct == "populism" and label == "populism_anti_elite":
            assigned_set = set(assigned_items)
            cynicism_markers = {
                "populism__corrupt_get_to_top",
                "populism__trust_politicians",
            }
            if assigned_set & cynicism_markers:
                label = "populism_elite_cynicism"

        if label in used_labels:
            label = f"{label}_factor{factor_idx + 1}"

        factor_labels[factor_idx] = label
        used_labels.add(label)
    return factor_labels

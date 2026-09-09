"""Render portfolio figures from reviewed aggregate tables; no survey access needed.

These figures reproduce saved estimates. They do not fit models, estimate
uncertainty or incorporate the separate post-submission corrections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import PALETTE, apply_plot_theme, finalize_figure, style_axes

ROOT = Path(__file__).resolve().parents[1]


def _heading(fig, title: str, subtitle: str, note: str) -> None:
    fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=19, fontweight="semibold")
    fig.text(0.07, 0.905, subtitle, fontsize=10.5, color=PALETTE["gray"])
    fig.text(0.07, 0.025, note, fontsize=8.5, color=PALETTE["gray"])


def prevalence(tables: Path, destination: Path) -> None:
    frame = pd.read_csv(tables / "trends/rwp_prevalence_by_year.csv").sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.80)
    _heading(fig, "A non-monotonic pattern across survey waves",
             "Share above one fixed, pooled weighted 80th-percentile score threshold",
             "BSA 2011, 2013, 2019 and 2024 · Submitted estimates · 2023 reserved for validation\n"
             "Points are observed waves; connecting lines do not estimate the intervening years. No confidence intervals estimated.")
    ax.plot(frame.year, frame.weighted_prevalence, marker="o", color=PALETTE["blue"], markersize=8)
    ax.axhline(0.2, color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax.text(2010.3, 0.207, "20% reference", ha="left", fontsize=9, color=PALETTE["gray"])
    for row in frame.itertuples():
        ax.annotate(f"{row.weighted_prevalence:.1%}", (row.year, row.weighted_prevalence),
                    xytext=(0, 13), textcoords="offset points", ha="center", fontweight="semibold", fontsize=12)
    ax.set_xticks(frame.year, [f"{y}\nn = {n:,}" for y, n in zip(frame.year, frame.n)])
    style_axes(ax, ylabel="Weighted share above the threshold", y_lim=(0, 0.5), x_lim=(2010, 2025), percent_y=True)
    finalize_figure(fig, destination / "prevalence.png")


def performance(tables: Path, destination: Path) -> None:
    comparison = pd.read_csv(tables / "modelling/model_comparison.csv").set_index("model")
    ridge = pd.read_csv(tables / "modelling/regression_metrics.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.24, top=0.73, wspace=0.58)
    _heading(fig, "Demographics add modest predictive information",
             "Survey-weighted ridge regression · continuous 0–1 attitude score · submitted estimates",
             "Internal test n = 3,617; 2023 holdout n = 5,559. Training includes 2024: the holdout tests wave transfer, not forecasting.\n"
             "Baseline comparison is available for the internal test only. Most score variation remains unexplained.")
    models = ["mean_baseline", "year_only_baseline", "ridge_regression"]
    values = comparison.loc[models, "weighted_rmse"].to_numpy()
    axes[0].barh([2, 1, 0], values, color=["#AAB8C6", "#6D98C4", PALETTE["blue"]], height=0.53)
    axes[0].set_yticks([2, 1, 0], ["Training mean", "Year only", "Ridge · 13 predictors"])
    for y, value in zip([2, 1, 0], values):
        axes[0].text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=11)
    style_axes(axes[0], title="Internal-test error", xlabel="Weighted RMSE · lower is better", grid_axis="x", x_lim=(0, 0.195))
    scores = [float(ridge.weighted_r2), float(ridge.holdout_weighted_r2)]
    axes[1].bar([0, 1], scores, width=0.5, color=[PALETTE["blue"], PALETTE["orange"]])
    axes[1].set_xticks([0, 1], ["Internal test", "2023 holdout"])
    for x, score in enumerate(scores):
        axes[1].text(x, score + 0.012, f"{score:.3f}", ha="center", fontsize=12, fontweight="semibold")
    style_axes(axes[1], title="Ridge performance", ylabel="Weighted R² · higher is better", y_lim=(0, 1))
    finalize_figure(fig, destination / "model-performance.png")


def importance(tables: Path, destination: Path) -> None:
    frame = pd.read_csv(tables / "trends/predictor_trends.csv")
    data = frame.pivot(index="predictor", columns="year", values="permutation_r2_drop_mean")
    data = data.loc[data.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 7.2))
    fig.subplots_adjust(left=0.23, right=0.89, bottom=0.20, top=0.80)
    _heading(fig, "Education is a recurring descriptive correlate",
             "Grouped permutation drop in weighted R² · separate within-wave linear models",
             "All available trend predictors shown, ordered by average importance across waves. 20 permutations per predictor.\n"
             "In-sample descriptive importance, not causal effects or out-of-sample performance.\n"
             "Submitted education mappings differ in 2024; see the separately evaluated correction profile in docs/audit.md.")
    limit = max(float(np.nanmax(np.abs(data.to_numpy()))), 0.01)
    im = ax.imshow(data, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(data)), [p.replace("_", " ").title() for p in data.index])
    ax.set_xticks(range(len(data.columns)), data.columns)
    for (y, x), value in np.ndenumerate(data.to_numpy()):
        ax.text(x, y, f"{value:.3f}", va="center", ha="center", fontsize=10,
                color="white" if abs(value) > limit * 0.65 else PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)
    colorbar.set_label("Drop in weighted R²")
    finalize_figure(fig, destination / "predictor-importance.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "outputs/main/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    apply_plot_theme()
    for render in (prevalence, performance, importance):
        render(args.tables_dir, args.output_dir)
    print(f"Rendered three aggregate-only figures in {args.output_dir}")


if __name__ == "__main__":
    main()

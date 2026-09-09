# shared plotting style and save helpers
# used by scoring, modelling, and trend figures

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

_APPLIED = False

PALETTE = {
    "blue": "#2E5B87",
    "blue_light": "#6D98C4",
    "orange": "#E67E22",
    "green": "#2E8B57",
    "red": "#C54B4B",
    "gray": "#6B7280",
    "ink": "#1F2937",
    "grid": "#D6DCE5",
    "edge": "#CCD5E1",
    "bg": "#F7F9FC",
    "bg_alt": "#EEF3FA",
}


def format_profile_label(profile_name: str | None) -> str:
    if not profile_name:
        return "Results"
    return " ".join(str(profile_name).replace("_", " ").split()).title()


def with_profile_prefix(profile_label: str | None, title: str) -> str:
    clean_label = " ".join(str(profile_label or "").replace("_", " ").split()).strip()
    if not clean_label or clean_label.lower() in {"main", "results"}:
        return title
    return f"{clean_label}: {title}"


def apply_plot_theme() -> None:
    global _APPLIED
    if _APPLIED:
        return
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "axes.edgecolor": PALETTE["edge"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "legend.facecolor": "white",
            "legend.edgecolor": PALETTE["edge"],
            "legend.framealpha": 0.98,
            "legend.borderpad": 0.35,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "patch.edgecolor": "white",
            "patch.linewidth": 0.6,
            "font.size": 10,
            "text.color": PALETTE["ink"],
            "axes.prop_cycle": mpl.cycler(
                color=[
                    PALETTE["blue"],
                    PALETTE["orange"],
                    PALETTE["green"],
                    "#8B5CF6",
                    "#D97706",
                    "#0EA5A7",
                    "#DB2777",
                ]
            ),
        }
    )
    _APPLIED = True


def style_axes(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid_axis: str = "y",
    percent_y: bool = False,
    percent_x: bool = False,
    integer_x: bool = False,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    title_loc: str = "left",
) -> None:
    if title:
        ax.set_title(title, loc=title_loc, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.45, linewidth=0.8)
        if grid_axis != "both":
            ax.grid(axis="x" if grid_axis == "y" else "y", alpha=0.16, linewidth=0.7)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if percent_x:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.margins(x=0.02)
    ax.spines["left"].set_color(PALETTE["edge"])
    ax.spines["bottom"].set_color(PALETTE["edge"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, direction="out")


def style_legend(ax, **kwargs):
    leg = ax.legend(**kwargs)
    if leg is None:
        return None
    frame = leg.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor(PALETTE["edge"])
    frame.set_facecolor("white")
    frame.set_alpha(0.98)
    return leg


def finalize_figure(fig, out_path: Path | str, *, dpi: int = 240) -> None:
    """Export legible raster figures with consistent margins and no open handles."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

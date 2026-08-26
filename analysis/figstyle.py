"""Shared matplotlib style for the paper's figures.

PolicyEngine palette, publication defaults, and a handful of helpers used by
:mod:`analysis.figures`. Import for side effects (``apply()`` is called on
import) or call :func:`apply` explicitly.

Design rules
------------
* One accent per meaning: teal = loss, blue = compensation/policy, grey =
  context, dark = text and reference lines.
* No chartjunk: no gridlines on the value axis heavier than a hairline, no
  boxes, no shadows, no redundant legends.
* Every axis carries units; every figure carries an informative title.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------

TEAL = "#39C6C0"
BLUE = "#2C6496"
GREY = "#616161"
DARK = "#0C1A27"

TEAL_LIGHT = "#A9E5E2"
TEAL_DARK = "#1F8C88"
BLUE_LIGHT = "#7FA9CC"
BLUE_DARK = "#1B3F5E"
GREY_LIGHT = "#BDBDBD"

#: Ordered categorical cycle for up to five series (policies, fuels).
CYCLE = (BLUE, TEAL, GREY, BLUE_LIGHT, TEAL_DARK)

#: Fuel colours, fixed across figures.
FUEL_COLORS = {
    "gas": BLUE,
    "electricity": TEAL,
    "motor_fuel": GREY,
}
FUEL_LABELS = {
    "gas": "Gas",
    "electricity": "Electricity",
    "motor fuel": "Motor fuel",
    "motor_fuel": "Motor fuel",
}

#: Scenario colours, fixed across figures.
SCENARIO_COLORS = {
    "niesr_baseline": GREY,
    "niesr_adverse": BLUE,
    "realised_2026": TEAL_DARK,
}

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def apply() -> None:
    """Set the global rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlepad": 9,
            "axes.labelsize": 10,
            "axes.labelcolor": DARK,
            "axes.edgecolor": GREY_LIGHT,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=list(CYCLE)),
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.handlelength": 1.6,
            "text.color": DARK,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
        }
    )


apply()


# --------------------------------------------------------------------------
# formatters and helpers
# --------------------------------------------------------------------------


def gbp(value: float, _pos: int | None = None) -> str:
    """£-formatted tick label with thousands separators."""
    return f"£{value:,.0f}"


def pct(value: float, _pos: int | None = None) -> str:
    """Percent tick label for values already expressed in percent."""
    if value == 0:
        return "0%"
    if abs(value - round(value)) < 1e-9 and abs(value) >= 1:
        return f"{value:.0f}%"
    if abs(value) < 0.1:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


GBP_FMT = FuncFormatter(gbp)
PCT_FMT = FuncFormatter(pct)


def only_y_grid(ax) -> None:
    ax.grid(axis="y", which="major")
    ax.grid(axis="x", visible=False)


def only_x_grid(ax) -> None:
    ax.grid(axis="x", which="major")
    ax.grid(axis="y", visible=False)


def panel_label(ax, text: str) -> None:
    """Bold panel letter, e.g. 'A', above the axes on the left."""
    ax.text(
        -0.02,
        1.10,
        text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=DARK,
    )


def label_bars(ax, bars, values, fmt="{:.0f}", *, dy=0.01, fontsize=8, color=DARK):
    """Write a value above each bar, offset by a fraction of the y-range."""
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * dy
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def label_hbars(ax, bars, values, fmt="{:.0f}", *, dx=0.01, fontsize=8, color=DARK):
    """Write a value to the right of each horizontal bar, haloed in white so it
    stays legible where it crosses a reference line."""
    import matplotlib.patheffects as pe  # noqa: PLC0415

    lo, hi = ax.get_xlim()
    pad = (hi - lo) * dx
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=fontsize,
            color=color,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
        )


def note(fig, text: str, *, y: float = -0.02) -> None:
    """Source/footnote line under the figure."""
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="top",
        fontsize=8,
        color=GREY,
        wrap=True,
    )


def save(fig, name: str) -> Path:
    """Save ``fig`` into results/figures and close it. Returns the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def pretty_region(name: str) -> str:
    """'YORKSHIRE' -> 'Yorkshire', 'NORTH_EAST' -> 'North East'."""
    special = {
        "NORTHERN_IRELAND": "Northern Ireland",
        "EAST_OF_ENGLAND": "East of England",
        "YORKSHIRE": "Yorkshire & Humber",
    }
    if name in special:
        return special[name]
    return " ".join(part.capitalize() for part in name.split("_"))

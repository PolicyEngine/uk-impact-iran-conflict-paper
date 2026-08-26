"""Shared figure style for the working paper.

One palette, one type scale, one axis-title convention across every figure in
``results/figures``. Presentation only: nothing here touches data or weights.

Palette and typography follow the PolicyEngine house style — primary blue
(#2C6496), teal accent (#39C6C0) and a grey ladder; a blue sequential ramp for
single-direction magnitudes and a grey-white-blue diverging map (PolicyEngine
convention: **negative in grey, positive in blue**, white midpoint) for the
constituency maps, where almost every seat is a loss.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- PolicyEngine brand palette (canonical hexes from policyengine-core) ---
BLUE = BLUE_PRIMARY = "#2C6496"
BLUE_LIGHT = "#D8E6F3"
BLUE_PRESSED = "#17354F"
BLUE_98 = "#F7FAFD"
TEAL = TEAL_ACCENT = "#39C6C0"
TEAL_PRESSED = "#227773"
DARKEST_BLUE = "#0C1A27"
GREEN = DARK_GREEN = "#558B2F"
DARK_GRAY = "#616161"
GRAY = "#808080"
MEDIUM_DARK_GRAY = "#D2D2D2"
LIGHT_GRAY = "#F2F2F2"

#: Categorical slots — fixed order, never cycled. Five slots for the five
#: scored policy responses, all mutually separable PolicyEngine hues.
SERIES = [BLUE, TEAL, GREEN, DARK_GRAY, DARKEST_BLUE, GRAY]

#: Stable colour per policy response, so a reader who learns the JRF block is
#: teal in Figure 4 does not have to relearn it in Figure 6.
POLICY_COLORS = {
    "shock": DARK_GRAY,
    "social_tariff": BLUE,
    "jrf_block": TEAL,
    "whd_expansion": GREEN,
    "vat_zero": DARKEST_BLUE,
    "ippr_rebate": TEAL_PRESSED,
}

#: Intra-decile bands, ordered worst-off to best-off; the two loss bands take
#: the grey end of the diverging ramp so "uncompensated losers" reads at a
#: glance.
INTRA_DECILE_COLORS = {
    "lose_more_5": DARK_GRAY,
    "lose_less_5": MEDIUM_DARK_GRAY,
    "no_change": LIGHT_GRAY,
    "gain_less_5": BLUE_LIGHT,
    "gain_more_5": BLUE,
}
INTRA_DECILE_LABELS = {
    "lose_more_5": "Lose more than 5%",
    "lose_less_5": "Lose less than 5%",
    "no_change": "No change",
    "gain_less_5": "Gain less than 5%",
    "gain_more_5": "Gain more than 5%",
}

# Ink / chrome
INK = DARKEST_BLUE
INK2 = DARK_GRAY
MUTED = GRAY
GRID = LIGHT_GRAY
BASELINE = MEDIUM_DARK_GRAY
NEUTRAL = LIGHT_GRAY
LIGHT_BLUE = BLUE_LIGHT

#: Single-direction magnitudes (the £-loss map).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pe_seq", [BLUE_98, BLUE_LIGHT, BLUE, BLUE_PRESSED]
)

#: Grey (negative) — white — blue (positive).
DIVERGING = LinearSegmentedColormap.from_list(
    "pe_div",
    [DARK_GRAY, GRAY, MEDIUM_DARK_GRAY, "#FFFFFF", BLUE_LIGHT, BLUE, BLUE_PRESSED],
)

DECILE_AXIS = "Income decile (equivalised household disposable income, HBAI)"

# Canonical figure sizes (inches)
SINGLE = (8.0, 4.5)
TWOPANEL = (11.0, 4.5)
MAP = (8.5, 10.5)
TWOMAP = (12.0, 9.0)
DPI = 200

_SERIF = ["Roboto Serif", "Roboto Slab", "Source Serif Pro", "DejaVu Serif"]


def apply_style() -> None:
    """Set the house rcParams. Idempotent; call at the top of every figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _SERIF,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def decile_ax(ax, ylabel: str, xlabel: str = DECILE_AXIS) -> None:
    """Common decile-chart chrome: x ticks 1-10, y grid only."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)


def zero_line(ax) -> None:
    """A baseline rule at zero — every chart here spans losses and gains."""
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)


def legend_below(ax, ncol: int) -> None:
    """Legend centred below the axes (house style: legends never sit on data)."""
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.18))


def save(fig, path) -> None:
    """Tight-layout, save at house DPI, close."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("wrote", path)

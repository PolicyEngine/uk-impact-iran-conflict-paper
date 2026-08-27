#!/usr/bin/env python3
"""Render the scenario-grid figure from ``results/grid/grid.csv``.

Presentation only — no microdata, no simulation. Run ``analysis/run_grid.py``
first.

The figure is a 2x2 panel of heatmaps over the (wholesale gas %, oil %) grid:

A. mean household loss, £
B. aggregate cost, £bn
C. the decile-1 / decile-10 burden ratio (the distributional gradient)
D. motor fuel's share of the loss, %

Panel D is the reason the figure exists. The paper's headline is that motor fuel
is 67.8% of the loss (``docs/FINDINGS.md`` §2), which is what limits every
domestic-bill compensation instrument. The grid shows that this is a property of
*this* shock's gas/oil mix rather than of energy shocks in general: it is
diverging around 50%, so the reader can see exactly where the shock stops being
a pump-price event and becomes a domestic-bill one.

Usage: ``python analysis/figures_grid.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:  # `python analysis/figures_grid.py` or `-m analysis.figures_grid`
    import figstyle as fs
except ImportError:  # pragma: no cover
    from analysis import figstyle as fs

import matplotlib.pyplot as plt

from uk_iran_conflict import scenarios as scen

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "grid"

#: Sequential ramp for magnitudes: paper white through the teal accent into the
#: dark ink, so a printed greyscale copy still reads monotonically.
SEQ = LinearSegmentedColormap.from_list(
    "pe_seq", ["#FFFFFF", fs.TEAL_LIGHT, fs.TEAL, fs.TEAL_DARK, fs.DARK]
)
#: Sequential ramp in the blue accent, for the distributional gradient.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "pe_seq_blue", ["#FFFFFF", fs.BLUE_LIGHT, fs.BLUE, fs.BLUE_DARK]
)
#: Diverging ramp for motor fuel's share of the loss. The midpoint is not
#: cosmetic: 50% is where the shock stops being a pump-price event (teal) and
#: becomes a domestic-bill event (blue).
DIV = LinearSegmentedColormap.from_list(
    "pe_div", [fs.BLUE_DARK, fs.BLUE, fs.BLUE_LIGHT, "#F7F7F7", fs.TEAL, fs.TEAL_DARK]
)

#: The three named scenarios are marked as numbered points rather than spelt
#: out on every panel: a full label sits on top of the cell values it is meant
#: to help the reader read. The key is carried in the figure note.
NAMED_TAGS = {
    "niesr_baseline": ("1", "NIESR baseline"),
    "niesr_adverse": ("2", "NIESR adverse"),
    "realised_2026": ("3", "Realised 2026"),
    # The variant calibrations are marked too, since the fuel-share frontier in
    # panel D is exactly what separates them; without these two the figure
    # rendered them as unlabelled "?" markers.
    "realised_2026_symmetric": ("4", "Symmetric damping"),
    "realised_2026_peak_fuel": ("5", "Peak-fuel bound"),
}


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Oil on the rows (descending, so it rises up the page), gas on the columns."""
    return df.pivot(index="oil_pct", columns="gas_pct", values=col).sort_index(
        ascending=False
    )


def _heatmap(ax, piv, cmap, fmt, *, vmin=None, vmax=None, center=None):
    """One annotated imshow panel over the grid, mirroring the sister paper."""
    values = piv.values.astype(float)
    finite = values[np.isfinite(values)]
    if center is not None:
        lim = max(abs(finite - center).max(), 1e-9)
        vmin, vmax = center - lim, center + lim
    else:
        vmin = float(finite.min()) if vmin is None else vmin
        vmax = float(finite.max()) if vmax is None else vmax
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    im.cmap.set_bad("#EEEEEE")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=fs.GREY)
                continue
            colour = (
                "white"
                if abs(v - (center if center is not None else vmin))
                > 0.62 * (vmax - (center if center is not None else vmin))
                else fs.DARK
            )
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=6.4,
                color=colour,
            )
    ax.grid(visible=False)
    ax.set_xticks(range(len(piv.columns)), [f"{c:.0f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{r:.0f}" for r in piv.index])
    ax.tick_params(labelsize=7.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


#: Columns holding the **damped-equivalent** coordinates written by
#: ``analysis/run_grid.py``. Every grid cell is run undamped
#: (``sustained_fraction = 1.0``), so a scenario that damps its own wholesale
#: move must be plotted at the sustained move that reproduces it, not at its
#: headline peak (docs/FIXES.md D25). Plotting the headline coordinates put the
#: realised 2026 point ABOVE the 50% motor-fuel frontier by the grid's own
#: numbers, while the paper claimed all three named scenarios sat below it: the
#: figure and the CSV disagreed. The headline columns are used only as a
#: fallback for a named_points.csv written before that fix.
DAMPED_COORDS = ("gas_pct_damped", "oil_pct_damped")


def _named_coords(row) -> tuple[float, float]:
    """(gas %, oil %) to plot for one named scenario: damped where available."""
    gas_col, oil_col = DAMPED_COORDS
    gas = row.get(gas_col)
    oil = row.get(oil_col)
    if gas is None or oil is None or not np.isfinite([gas, oil]).all():
        return float(row["gas_pct"]), float(row["oil_pct"])
    return float(gas), float(oil)


def _mark_named(ax, piv, named: pd.DataFrame) -> None:
    """Plot the named scenarios at their damped-equivalent grid coordinates."""
    gas_axis = np.asarray(piv.columns, dtype=float)
    oil_axis = np.asarray(piv.index, dtype=float)  # descending
    for _, row in named.iterrows():
        gas_pct, oil_pct = _named_coords(row)
        x = np.interp(gas_pct, gas_axis, np.arange(len(gas_axis)))
        y = np.interp(oil_pct, oil_axis[::-1], np.arange(len(oil_axis))[::-1])
        ax.plot(
            x,
            y,
            marker="o",
            ms=5.5,
            mfc="none",
            mec=fs.DARK,
            mew=1.4,
            zorder=5,
            clip_on=False,
        )
        ax.plot(x, y, marker="o", ms=1.8, color=fs.DARK, zorder=6, clip_on=False)
        tag = NAMED_TAGS.get(row["scenario"], ("?", row["scenario"]))[0]
        # Keep the tag inside the axes: flip it left near the right edge.
        right = x > len(gas_axis) - 1.3
        ax.annotate(
            tag,
            (x, y),
            textcoords="offset points",
            xytext=(-7 if right else 7, 3),
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=fs.DARK,
            zorder=7,
            annotation_clip=False,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )


def scenario_grid() -> Path:
    df = pd.read_csv(GRID / "grid.csv")
    named = pd.read_csv(GRID / "named_points.csv")

    # The zero/zero cell has no shock, so its shares and ratio are undefined
    # (docs/FIXES.md E35). run_grid.py now flags it and writes NaN; this repeats
    # the blanking so an older grid.csv renders identically rather than showing
    # the degenerate corner's spurious 0%/100% fuel split.
    dead = df.get("is_degenerate")
    if dead is None:
        dead = (df["gas_pct"] == 0) & (df["oil_pct"] == 0)
    dead = dead.astype(bool)
    for col in ("motor_fuel_share_pct", "d1_d10_ratio"):
        df.loc[dead, col] = np.nan

    #: Half-width of panel C's colour range, in ratio units.
    #:
    #: The round-3 rebuild made the gradient invariant across the whole grid --
    #: every cell is 9.31 -- because the ratio is carried by the domestic leg,
    #: which is a common scaling of one consumption vector. That invariance IS
    #: the finding. But auto-scaling a colour map to the surviving spread, which
    #: is under two thousandths, renders floating-point residue as a bold blue
    #: gradient and invites the reader to interpret it. A fixed range wide
    #: enough to contain any variation worth seeing leaves a genuinely flat
    #: field looking flat.
    RATIO_PANEL_PAD = 0.5
    ratio_finite = df["d1_d10_ratio"].dropna()
    ratio_mid = float(ratio_finite.mean()) if len(ratio_finite) else 0.0
    ratio_range = (ratio_mid - RATIO_PANEL_PAD, ratio_mid + RATIO_PANEL_PAD)
    ratio_spread = (
        float(ratio_finite.max() - ratio_finite.min()) if len(ratio_finite) else 0.0
    )

    panels = [
        ("mean_loss_gbp", "A", "Mean household loss (£/yr)", SEQ, "{:.0f}", None),
        ("aggregate_cost_bn", "B", "Aggregate cost (£bn/yr)", SEQ, "{:.1f}", None),
        (
            "d1_d10_ratio",
            "C",
            "Burden ratio, decile 1 ÷ decile 10",
            SEQ_BLUE,
            "{:.2f}",
            None,
        ),  # colour range fixed below: see RATIO_PANEL_PAD
        (
            "motor_fuel_share_pct",
            "D",
            "Motor fuel's share of the loss (%)",
            DIV,
            "{:.0f}",
            50.0,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.9))
    for (col, letter, title, cmap, fmt, center), ax in zip(
        panels, axes.ravel(), strict=True
    ):
        piv = _pivot(df, col)
        limits = ratio_range if col == "d1_d10_ratio" else (None, None)
        im = _heatmap(ax, piv, cmap, fmt, center=center, vmin=limits[0], vmax=limits[1])
        _mark_named(ax, piv, named)
        ax.set_title(f"{letter}. {title}", fontsize=8.6, pad=5)
        if col == "d1_d10_ratio":
            # Say the flat field is flat, and by how much, so the panel is read
            # as the invariance result it is rather than as a failed heatmap.
            ax.set_xlabel(
                f"invariant across the grid (spread {ratio_spread:.4f})",
                fontsize=6.8,
                labelpad=2,
            )
        cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.03)
        cb.ax.tick_params(labelsize=6.5, length=0)
        cb.outline.set_visible(False)

    # Panel C carries its own x-label (the invariance note), so only panel D
    # gets the axis name on the bottom row; C's is set in the loop above.
    axes[1][1].set_xlabel("Wholesale gas change (%)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Brent oil change (%)", fontsize=8)

    fig.suptitle(
        "Household incidence across the gas-oil scenario grid",
        fontsize=10.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    fs.note(
        fig,
        "Each cell is a sustained wholesale move versus the pre-war reference, "
        "run through the same cap path and pump pass-through as the named "
        "scenarios and scored on the same PolicyEngine UK baseline (2026). "
        "Panel D diverges at 50%: teal cells are pump-price shocks, blue cells "
        "domestic-bill shocks. Marked points are the named scenarios "
        "(1 NIESR baseline, 2 NIESR adverse, 3 realised 2026), plotted at "
        "damped-equivalent coordinates: every cell runs undamped, so a "
        "scenario that damps its peak wholesale move (realised 2026 sustains "
        f"{scen.REALISED_CAP_CALIBRATION.sustained_fraction:.2f} of its gas "
        "peak) is placed at the sustained move that reproduces its own retail "
        "and pump prices, not at its headline peak. The zero-shock corner is "
        "undefined and left blank. Panel C is flat by construction: the "
        "gradient is carried by the domestic leg, which is a common scaling of "
        "one consumption vector, so it does not move with the gas/oil mix. "
        "That invariance is the result; the colour range is fixed so a flat "
        "field is not shaded as if it varied.",
        y=0.035,
    )
    return fs.save(fig, "fig_scenario_grid.png")


if __name__ == "__main__":
    print(scenario_grid())

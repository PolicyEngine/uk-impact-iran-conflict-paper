#!/usr/bin/env python3
"""Figure builders for the paper, off cached results/ artifacts.

No microsimulation runs here — presentation only, from the JSON/CSV written by
``analysis/run_all.py``.

The headline figure is :func:`constituency_two_maps`: **two side-by-side
constituency choropleths of the same shock** — loss as a share of income, and
loss in pounds. They rank the 650 seats almost oppositely, which is the paper's
Step 3 contribution: Fetzer, Gazze & Bishop (2024) find affluent areas more
exposed in £; the budget-share literature finds poor households more exposed in
%. Both are true, and no existing UK analysis shows them at seat level.

Usage:
    python analysis/figures.py [--results results] [--out results/figures]
                               [--scenario niesr_adverse]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

#: 2024 Westminster constituency boundaries (GSS-coded), staged in data/.
BOUNDARIES = ROOT / "data" / "uk_constituencies_2024.geojson"

DECILES = list(range(1, 11))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_cell(results: Path, scenario: str, policy: str = "shock") -> dict:
    """Load one (scenario, policy) result JSON."""
    return json.loads((results / scenario / f"{policy}.json").read_text())


def decile_series(cell: Mapping, block: str) -> np.ndarray:
    """Pull a 10-vector out of a result block (JSON keys are strings)."""
    values = cell[block]
    return np.array(
        [float(values[str(d)] if str(d) in values else values[d]) for d in DECILES]
    )


# --------------------------------------------------------------------------
# the headline figure
# --------------------------------------------------------------------------


def load_constituency_frame(results: Path) -> pd.DataFrame:
    """Join the cached constituency impacts onto 2024 boundary polygons."""
    import geopandas as gpd  # noqa: PLC0415 — heavy optional dependency

    impacts = pd.read_csv(results / "geo" / "constituency_impacts.csv")
    polygons = gpd.read_file(BOUNDARIES)[["GSScode", "geometry"]]
    merged = polygons.merge(impacts, left_on="GSScode", right_on="code", how="inner")
    dropped = len(impacts) - len(merged)
    if dropped:
        print(f"warning: {dropped} constituencies did not join to a boundary")
    # The published GeoJSON is mislabelled EPSG:4326 but carries British
    # National Grid eastings/northings; assign the true CRS (no reprojection).
    return merged.set_crs(27700, allow_override=True)


def _map_panel(ax, gdf, column: str, cmap, norm, title: str) -> ScalarMappable:
    gdf.plot(
        column=column, cmap=cmap, norm=norm, ax=ax, edgecolor="white", linewidth=0.12
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=figstyle.INK, pad=8)
    return ScalarMappable(norm=norm, cmap=cmap)


def constituency_two_maps(results: Path, out: Path, clip_pct: float = 95.0) -> Path:
    """THE headline figure: % of income lost and £ lost, side by side.

    Left panel: mean household loss as a share of net income (diverging, grey
    losses). Right panel: mean household loss in pounds (sequential magnitude).
    Both scales are clipped at the ``clip_pct`` percentile of |value| so a
    handful of extreme seats does not crush the gradient across the rest.

    The two panels are the same shock. Their geographies are close to
    opposites — that is the figure's whole argument, and the caption should
    say so.
    """
    figstyle.apply_style()
    gdf = load_constituency_frame(results)

    pct = gdf["relative_change"].to_numpy(dtype=float) * 100
    cash = gdf["average_change"].to_numpy(dtype=float)

    pct_max = float(np.nanpercentile(np.abs(pct), clip_pct)) or float(
        np.nanmax(np.abs(pct))
    )
    pct_norm = TwoSlopeNorm(vcenter=0.0, vmin=-pct_max, vmax=pct_max)
    # Losses are negative; plot the magnitude on the sequential ramp so
    # "darker = worse hit" reads the same way in both panels.
    magnitude = np.abs(cash)
    cash_norm = Normalize(
        vmin=float(np.nanpercentile(magnitude, 100 - clip_pct)),
        vmax=float(np.nanpercentile(magnitude, clip_pct)),
    )
    gdf = gdf.assign(_pct=pct, _cash_magnitude=magnitude)

    fig, axes = plt.subplots(1, 2, figsize=figstyle.TWOMAP)
    sm_pct = _map_panel(
        axes[0],
        gdf,
        "_pct",
        figstyle.DIVERGING,
        pct_norm,
        "Loss as a share of net income (%)",
    )
    sm_cash = _map_panel(
        axes[1],
        gdf,
        "_cash_magnitude",
        figstyle.SEQUENTIAL,
        cash_norm,
        "Loss per household (£/year)",
    )

    for ax, sm in ((axes[0], sm_pct), (axes[1], sm_cash)):
        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, extend="both"
        )
        cbar.outline.set_visible(False)

    path = out / "fig_constituency_two_maps.png"
    figstyle.save(fig, path)
    return path


# --------------------------------------------------------------------------
# decile charts
# --------------------------------------------------------------------------


def decile_impact_chart(results: Path, out: Path, scenario: str) -> Path:
    """Two-panel decile chart of the bare shock: £ change and % of income.

    The same crossing as the maps, in the conventional decile form: the cash
    loss rises with income while the relative loss falls with it (the Deaton
    first-order incidence of Step 2).
    """
    figstyle.apply_style()
    cell = load_cell(results, scenario, "shock")
    cash = decile_series(cell, "decile_average_change")
    relative = decile_series(cell, "decile_relative_change") * 100

    fig, axes = plt.subplots(1, 2, figsize=figstyle.TWOPANEL)
    axes[0].bar(DECILES, cash, color=figstyle.BLUE)
    figstyle.decile_ax(axes[0], "Change in net income (£/year)")
    figstyle.zero_line(axes[0])

    axes[1].bar(DECILES, relative, color=figstyle.TEAL)
    figstyle.decile_ax(axes[1], "Change in net income (% of baseline)")
    figstyle.zero_line(axes[1])

    path = out / f"fig_decile_impact_{scenario}.png"
    figstyle.save(fig, path)
    return path


def intra_decile_chart(
    results: Path, out: Path, scenario: str, policy: str = "shock"
) -> Path:
    """Stacked winners/losers within each decile — the uncompensated-loser figure.

    PolicyEngine's ``IntraDecileImpact`` bands. The point (Cronin, Fullerton &
    Sexton 2019; Douenne 2020; Sallee 2019): even where a policy makes a decile
    better off on average, a large share of that decile still loses, and no
    observable-based transfer closes the gap.
    """
    figstyle.apply_style()
    cell = load_cell(results, scenario, policy)
    bands = cell["intra_decile"]

    fig, ax = plt.subplots(figsize=figstyle.SINGLE)
    bottom = np.zeros(len(DECILES))
    for key, color in figstyle.INTRA_DECILE_COLORS.items():
        values = np.array(
            [
                float((bands[str(d)] if str(d) in bands else bands[d])[key]) * 100
                for d in DECILES
            ]
        )
        ax.bar(
            DECILES,
            values,
            bottom=bottom,
            color=color,
            label=figstyle.INTRA_DECILE_LABELS[key],
            width=0.75,
        )
        bottom += values
    figstyle.decile_ax(ax, "Share of households (%)")
    ax.set_ylim(0, 100)
    figstyle.legend_below(ax, ncol=3)

    path = out / f"fig_intra_decile_{scenario}_{policy}.png"
    figstyle.save(fig, path)
    return path


# --------------------------------------------------------------------------
# policy comparison
# --------------------------------------------------------------------------


def policy_comparison_chart(
    results: Path, out: Path, scenario: str, policies: Sequence[str] | None = None
) -> Path:
    """Compare the five responses on the three metrics the paper scores.

    Panels: (1) exchequer cost, £bn; (2) share of spend reaching deciles 1-3;
    (3) share of decile 1 still left worse off than pre-shock — the metric
    everyone misses, and the one on which the means-tested options do worst
    (~40% of households struggling to heat their home are not on means-tested
    benefits).
    """
    figstyle.apply_style()
    if policies is None:
        policies = [p for p in figstyle.POLICY_COLORS if p != "shock"]

    cost, bottom_three, losers, labels, colors = [], [], [], [], []
    for policy in policies:
        try:
            cell = load_cell(results, scenario, policy)
        except FileNotFoundError:
            print(f"warning: no result for {scenario}/{policy}; skipping")
            continue
        gains = decile_series(cell, "decile_average_change")
        positive = np.clip(gains, 0.0, None)
        total = positive.sum()
        bands = cell["intra_decile"]
        d1 = bands["1"] if "1" in bands else bands[1]
        cost.append(-cell["exchequer_cost"] / 1e9)
        bottom_three.append(100 * positive[:3].sum() / total if total else 0.0)
        losers.append(100 * (d1["lose_less_5"] + d1["lose_more_5"]))
        labels.append(policy.replace("_", " "))
        colors.append(figstyle.POLICY_COLORS.get(policy, figstyle.BLUE))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    for ax, values, title in (
        (axes[0], cost, "Exchequer cost (£bn)"),
        (axes[1], bottom_three, "Share of gains to deciles 1-3 (%)"),
        (axes[2], losers, "Decile 1 left worse off (%)"),
    ):
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(title)
        ax.grid(axis="x", visible=False)
        figstyle.zero_line(ax)

    path = out / f"fig_policy_comparison_{scenario}.png"
    figstyle.save(fig, path)
    return path


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--scenario", default="niesr_adverse")
    args = parser.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    builders = (
        lambda: constituency_two_maps(results, out),
        lambda: decile_impact_chart(results, out, args.scenario),
        lambda: intra_decile_chart(results, out, args.scenario, "shock"),
        lambda: policy_comparison_chart(results, out, args.scenario),
    )
    for build in builders:
        try:
            build()
        except FileNotFoundError as exc:
            print(f"skipped (missing input): {exc}")


if __name__ == "__main__":
    main()

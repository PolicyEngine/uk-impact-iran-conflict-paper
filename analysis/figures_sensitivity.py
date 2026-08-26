#!/usr/bin/env python3
"""Appendix sensitivity figures, built from ``results/sensitivity/*.csv``.

These were previously placeholders. They are generated here rather than in
``analysis/figures.py`` because they read the sweep CSVs, which are produced by
``analysis/run_sensitivity.py`` and need no microdata — so this script runs in
CI without a token.

Usage::

    python analysis/figures_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT / "results" / "sensitivity"
FIGS = ROOT / "results" / "figures"

TEAL = "#39C6C0"
BLUE = "#2C6496"
GREY = "#616161"
DARK = "#0C1A27"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def elasticity_figure() -> Path:
    """Aggregate cost and the decile gradient across the elasticity sweep.

    The main specification is zero elasticity — an explicit upper bound — so the
    figure is drawn to show how much of that bound each credible short-run
    estimate shaves off, not to suggest a preferred value.
    """
    df = pd.read_csv(SENS / "elasticity.csv")
    # The sweep CSV labels the uniform grid ``grid`` and the literature
    # specifications ``named``; an earlier filter looked for ``flat`` and
    # silently drew an empty line.
    flat = df[df["kind"] == "grid"].sort_values("epsilon_mean", ascending=False)
    named = df[df["kind"] == "named"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

    ax = axes[0]
    ax.plot(
        -flat["epsilon_mean"],
        flat["aggregate_loss_bn"],
        marker="o",
        color=BLUE,
        linewidth=1.8,
        markersize=4,
        label="Constant elasticity",
    )
    for _, r in named.iterrows():
        ax.scatter(
            -r["epsilon_mean"],
            r["aggregate_loss_bn"],
            color=TEAL,
            zorder=5,
            s=42,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.annotate(
            str(r["spec"]).split("(")[0].strip()[:22],
            (-r["epsilon_mean"], r["aggregate_loss_bn"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=7,
            color=DARK,
        )
    ax.set_xlabel("Elasticity of demand (absolute value)", fontsize=9, color=DARK)
    ax.set_ylabel("Aggregate cost (£bn)", fontsize=9, color=DARK)
    ax.set_title(
        "Aggregate cost falls near-linearly in the demand response",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    _style(ax)

    ax = axes[1]
    ratio = flat["decile_gradient_pp"]
    ax.plot(
        -flat["epsilon_mean"],
        ratio,
        marker="o",
        color=BLUE,
        linewidth=1.8,
        markersize=4,
    )
    for _, r in named.iterrows():
        ax.scatter(
            -r["epsilon_mean"],
            r["decile_gradient_pp"],
            color=TEAL,
            zorder=5,
            s=42,
            edgecolor="white",
            linewidth=1.0,
        )
    ax.set_xlabel("Elasticity of demand (absolute value)", fontsize=9, color=DARK)
    ax.set_ylabel("Decile 1 − decile 10 burden (pp)", fontsize=9, color=DARK)
    ax.set_title(
        "The distributional gradient survives every specification",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    _style(ax)

    fig.tight_layout()
    out = FIGS / "figA1_elasticity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def welfare_bounds_figure() -> Path:
    """The correction: welfare bounds against the old spend-change measure.

    The appendix used to report the change in *expenditure* as the loss, so a
    strong demand response looked as though it removed four fifths of the shock.
    It does not: it removes four fifths of the spending, and about a tenth of the
    welfare loss, because the rest is foregone consumption that the household
    valued. Panel A draws the spending path inside the Paasche--Laspeyres band
    on the compensating variation; panel B draws the two "shaved" measures
    against each other, which is where the inversion is visible.
    """
    df = pd.read_csv(SENS / "elasticity.csv")
    flat = df[df["kind"] == "grid"].sort_values("epsilon_mean", ascending=False)
    named = df[df["kind"] == "named"]
    x = -flat["epsilon_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    ax = axes[0]
    ax.fill_between(
        x,
        flat["cv_lower_bn"],
        flat["cv_upper_bn"],
        color=TEAL,
        alpha=0.28,
        linewidth=0,
        label="Compensating variation (Paasche–Laspeyres bounds)",
    )
    ax.plot(x, flat["cv_lower_bn"], color=TEAL, linewidth=1.6, marker="o", markersize=3)
    ax.plot(
        x,
        flat["cv_upper_bn"],
        color=TEAL,
        linewidth=1.6,
        linestyle=(0, (4, 3)),
    )
    ax.plot(
        x,
        flat["aggregate_loss_bn"],
        color=BLUE,
        linewidth=2.0,
        marker="s",
        markersize=4,
        label="Change in expenditure (the superseded measure)",
    )
    for _, r in named.iterrows():
        ax.scatter(
            -r["epsilon_mean"],
            r["cv_lower_bn"],
            color=DARK,
            zorder=6,
            s=26,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_ylim(0, float(flat["cv_upper_bn"].max()) * 1.15)
    ax.set_xlabel("Elasticity of demand (absolute value)", fontsize=9, color=DARK)
    ax.set_ylabel("Aggregate loss (£bn)", fontsize=9, color=DARK)
    ax.set_title(
        "A the welfare loss barely moves; only spending does",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _style(ax)

    ax = axes[1]
    ax.plot(
        x,
        100 * flat["share_of_upper_bound_shaved"],
        color=BLUE,
        linewidth=2.0,
        marker="s",
        markersize=4,
        label="Share of spending shaved",
    )
    ax.plot(
        x,
        100 * flat["welfare_share_shaved"],
        color=TEAL,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Share of the welfare loss shaved",
    )
    worst = flat.iloc[-1]
    ax.annotate(
        f"{100 * float(worst['share_of_upper_bound_shaved']):.0f}% of spending\n"
        f"but only {100 * float(worst['welfare_share_shaved']):.1f}% of welfare",
        xy=(float(-worst["epsilon_mean"]), 100 * float(worst["welfare_share_shaved"])),
        xytext=(-118, 34),
        textcoords="offset points",
        fontsize=8,
        color=DARK,
        arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 0.9},
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel("Elasticity of demand (absolute value)", fontsize=9, color=DARK)
    ax.set_ylabel(
        "Share of the zero-elasticity bound removed (%)", fontsize=9, color=DARK
    )
    ax.set_title(
        "B the same sweep, on the two measures",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _style(ax)

    fig.tight_layout()
    out = FIGS / "figA2_welfare_bounds.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def cap_lag_figure() -> Path:
    """Annualised versus cumulative loss across the wholesale-to-retail lag.

    The point of the figure is that the cumulative burden is invariant: only the
    calendar year in which it is booked moves. Drawing both series together is
    the clearest way to show that the annual number is an attribution, not an
    effect.
    """
    df = pd.read_csv(SENS / "cap_lag.csv").sort_values("lag_quarters")
    x = df["lag_quarters"].astype(int)
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

    ax = axes[0]
    ax.bar(
        x - width / 2,
        df["annualised_mean_loss_gbp"],
        width,
        color=BLUE,
        label="Annualised (2026)",
    )
    ax.bar(
        x + width / 2,
        df["cumulative_mean_loss_gbp"],
        width,
        color=TEAL,
        label="Cumulative",
    )
    ax.set_xticks(list(x))
    ax.set_xlabel("Wholesale-to-retail lag (quarters)", fontsize=9, color=DARK)
    ax.set_ylabel("Mean household loss (£)", fontsize=9, color=DARK)
    ax.set_title(
        "Cumulative burden is invariant; only its timing moves",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    ax.plot(
        x,
        df["annualised_domestic_bn"],
        marker="o",
        color=BLUE,
        linewidth=1.8,
        markersize=4,
        label="Domestic energy",
    )
    ax.plot(
        x,
        df["motor_fuel_bn"],
        marker="s",
        color=TEAL,
        linewidth=1.8,
        markersize=4,
        label="Motor fuel",
    )
    ax.set_xticks(list(x))
    ax.set_xlabel("Wholesale-to-retail lag (quarters)", fontsize=9, color=DARK)
    ax.set_ylabel("Booked in 2026 (£bn)", fontsize=9, color=DARK)
    ax.set_title(
        "Motor fuel dominates 2026: the bill shock lands in 2027",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    fig.tight_layout()
    out = FIGS / "figA3_cap_lag.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (elasticity_figure, welfare_bounds_figure, cap_lag_figure):
        path = fn()
        size = path.stat().st_size / 1e3
        print(f"  {path.name:32} {size:7.1f} kB  ok")


if __name__ == "__main__":
    main()

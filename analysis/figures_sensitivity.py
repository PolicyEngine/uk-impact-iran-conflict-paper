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
#: Added for the fourth envelope row type (the instrument's own ceiling),
#: which needs to read as a darker sibling of BLUE rather than a new hue.
BLUE_DARK = "#1B3F5E"

#: Small-integer -> English word, so a title reads "three of five", not "3 of five".
NUMBER_WORD = (
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
)


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

    # Wider and taller than the three-bar version: four row types per
    # instrument, and a legend that no longer fits on one short row.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

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
    """The lag sweep on both anchoring rules, and the two legs behind it.

    The rebuilt sweep runs each lag twice: ``anchored`` re-solves the sustained
    fraction so the Cornwall cap anchor still binds, ``unanchored`` holds it
    fixed and so varies the timing alone. Drawing both is the point of the
    figure now — they agree out to two quarters and separate after, so the
    long-lag divergence is the anchoring rule rather than the lag. The old
    annualised-versus-cumulative pair cannot be drawn: the sweep no longer
    carries cumulative columns, and the cumulative burden is very nearly
    invariant along the unanchored series, so it would be a flat line.

    Three of the five anchored lags do not solve at all after the round-3
    rebuild — at those lags the two published caps' observation windows either
    fail to separate them or imply a pre-war counterfactual above the observed
    July cap — so their rows are blank. Blank rows are drawn as an explicit
    "not identified" mark rather than left as a gap, which is what they looked
    like before: three missing bars and no reason given.
    """
    df = pd.read_csv(SENS / "cap_lag.csv").sort_values("lag_quarters")
    lags = sorted(df["lag_quarters"].unique())
    width = 0.18 * (lags[1] - lags[0]) / 0.5 if len(lags) > 1 else 0.18

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

    ax = axes[0]
    for offset, (anchor, colour) in enumerate(
        (("anchored", BLUE), ("unanchored", TEAL))
    ):
        sub = df[df["anchor"] == anchor].sort_values("lag_quarters")
        values = pd.to_numeric(sub["mean_loss_gbp"], errors="coerce")
        positions = sub["lag_quarters"] + (offset - 0.5) * width
        ax.bar(
            positions,
            values.fillna(0.0),
            width,
            color=colour,
            label=anchor.capitalize(),
        )
        # Say why a bar is absent. An unexplained gap reads as a broken
        # pipeline; "not identified" is the actual finding.
        for pos, value in zip(positions, values, strict=True):
            if pd.isna(value):
                ax.annotate(
                    "not\nidentified",
                    (pos, 0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.5,
                    color=colour,
                )
    central = df[df["is_central_specification"].astype(str).str.lower() == "true"]
    if not central.empty:
        ax.axvline(
            float(central["lag_quarters"].iloc[0]),
            color=GREY,
            linewidth=0.9,
            linestyle=":",
        )
        ax.annotate(
            "paper",
            (float(central["lag_quarters"].iloc[0]), ax.get_ylim()[1]),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7.5,
            color=GREY,
        )
    ax.set_xticks(list(lags))
    ax.set_xticklabels([f"{lag:g}" for lag in lags])
    ax.set_xlabel("Wholesale-to-retail lag (quarters)", fontsize=9, color=DARK)
    ax.set_ylabel("Mean loss booked in the year (£)", fontsize=9, color=DARK)
    ax.set_title(
        "The lag moves the attribution, not the shock",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    # The UNANCHORED series, which is defined at every lag.
    #
    # This panel used to plot the anchored one, and after the rebuild that left
    # two points out of five joined by a line across an axis running to four
    # quarters. The unanchored series is also the right one for this panel on
    # the merits: it varies when the shock reaches the bill and holds how big it
    # is fixed, which is what "the lag moves the attribution, not the shock"
    # means. The anchored series re-solves the shock's size at every lag.
    series = df[df["anchor"] == "unanchored"].sort_values("lag_quarters")
    ax.plot(
        series["lag_quarters"],
        pd.to_numeric(series["domestic_loss_bn"], errors="coerce"),
        marker="o",
        color=BLUE,
        linewidth=1.8,
        markersize=4,
        label="Domestic energy",
    )
    ax.plot(
        series["lag_quarters"],
        pd.to_numeric(series["motor_fuel_loss_bn"], errors="coerce"),
        marker="s",
        color=TEAL,
        linewidth=1.8,
        markersize=4,
        label="Motor fuel",
    )
    ax.set_xticks(list(lags))
    ax.set_xticklabels([f"{lag:g}" for lag in lags])
    ax.set_xlabel("Wholesale-to-retail lag (quarters)", fontsize=9, color=DARK)
    ax.set_ylabel("Booked in the modelled year (£bn)", fontsize=9, color=DARK)
    ax.set_title(
        "Motor fuel dominates: the bill shock lands later",
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


#: Row type -> (display label, colour). The order is the order of the argument:
#: what the instrument can actually do, how much of the envelope that lets it
#: absorb, what proportional scaling pretends it can do, and what the same money
#: buys through eligibility instead.
#:
#: The first bar used to be ``common_capped`` labelled "At feasible max", which
#: is the mislabel the round-3 referees flagged: that row is
#: ``min(envelope, feasible-max cost)``, so for an instrument costing more than
#: the envelope it is the BUDGET, not the ceiling. The genuine ceiling is its
#: own row and its own bar, and the two are labelled apart.
ENVELOPE_VARIANTS = (
    ("feasible_max", "Own ceiling (no envelope)", BLUE_DARK),
    ("common_capped", "Absorbs envelope", BLUE),
    ("common_scaled", "Scaled to envelope", GREY),
    ("common_eligibility", "Wider eligibility", TEAL),
)

ENVELOPE_LABELS = {
    "social_tariff": "Social\ntariff",
    "jrf_block": "JRF\nblock",
    "whd_expansion": "WHD\nexpansion",
    "vat_zero": "VAT\nzero-rate",
    "ippr_rebate": "Flat\nrebate",
}


def policy_envelope_figure() -> Path:
    """What each instrument can absorb, and what it offsets when it does.

    The withdrawn claim — "VAT zero-rating wins at a common envelope" — is an
    artefact of the scaled bar. Zero-rating absorbs about £2bn and stops; the
    scaled row reaches £5bn only by removing more VAT points than the tax has.
    Hatching marks every infeasible bar, so the reader can see at a glance that
    the tallest offsets in the scaled series are not available.

    The first two bars answer different questions and the pre-revision figure
    drew only the second while labelling it as the first. "Own ceiling" is the
    instrument at its own feasible maximum with no envelope applied; "absorbs
    envelope" is the smaller of that ceiling and the budget.
    """
    df = pd.read_csv(SENS / "policy_envelope.csv")
    order = [key for key, _ in ENVELOPE_LABELS.items() if (df["policy"] == key).any()]
    x = list(range(len(order)))
    # Four row types rather than three.
    width = 0.20

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    def bars(ax, column: str, scale: float) -> None:
        for offset, (variant, label, colour) in enumerate(ENVELOPE_VARIANTS):
            sub = df[df["envelope"] == variant].set_index("policy")
            values, feasible, positions = [], [], []
            for i, key in enumerate(order):
                if key not in sub.index:
                    continue
                positions.append(
                    i + (offset - (len(ENVELOPE_VARIANTS) - 1) / 2) * width
                )
                values.append(float(sub.loc[key, column]) * scale)
                feasible.append(str(sub.loc[key, "is_feasible"]).lower() == "true")
            drawn = ax.bar(positions, values, width, color=colour, label=label)
            for patch, ok in zip(drawn, feasible, strict=True):
                if not ok:
                    patch.set_hatch("///")
                    patch.set_edgecolor("white")
                    patch.set_linewidth(0.0)
        ax.set_xticks(x)
        ax.set_xticklabels([ENVELOPE_LABELS[key] for key in order], fontsize=8)
        _style(ax)

    envelope = float(
        df.loc[df["envelope"] == "common_capped", "envelope_bn"].dropna().iloc[0]
    )

    ax = axes[0]
    bars(ax, "cost_bn", 1.0)
    ax.axhline(envelope, color=DARK, linewidth=0.9, linestyle="--")
    # The own-ceiling bars run far above the envelope — the JRF block reaches
    # more than four times it — so a limit pinned to 1.18x the envelope would
    # crop the very bars that make the point. A log scale keeps £2bn and £22bn
    # on the same axis without the small bars vanishing.
    ceilings = pd.to_numeric(
        df.loc[df["envelope"] == "feasible_max", "cost_bn"], errors="coerce"
    ).dropna()
    ax.set_yscale("log")
    ax.set_ylim(
        0.5, max(float(ceilings.max()) if len(ceilings) else envelope, envelope) * 1.7
    )
    ax.set_ylabel("Exchequer cost (£bn, log scale)", fontsize=9, color=DARK)
    saturating = int((ceilings < envelope).sum())
    # Two short lines rather than one long one: at this width a single-line
    # title ran straight into the right-hand panel's.
    ax.set_title(
        "The ceiling is not the budget\n"
        f"({NUMBER_WORD[saturating]} of five cannot absorb "
        f"the £{envelope:.0f}bn, dashed)",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )

    ax = axes[1]
    bars(ax, "share_of_aggregate_loss_offset", 100.0)
    ax.set_ylabel("Share of aggregate loss offset (%)", fontsize=9, color=DARK)
    ax.set_title(
        "What each actually offsets\n(hatched: settings that cannot exist)",
        fontsize=9.5,
        color=DARK,
        loc="left",
    )

    # One legend for both panels, below them, so neither bar chart has to give
    # up headroom to it.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    # Reserve the strip the legend occupies before tight_layout packs the axes,
    # or the four-entry legend lands on top of the instrument labels.
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = FIGS / "figA4_policy_envelope.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for fn in (
        elasticity_figure,
        welfare_bounds_figure,
        cap_lag_figure,
        policy_envelope_figure,
    ):
        path = fn()
        size = path.stat().st_size / 1e3
        print(f"  {path.name:32} {size:7.1f} kB  ok")


if __name__ == "__main__":
    main()

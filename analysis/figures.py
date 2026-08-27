#!/usr/bin/env python3
"""Build every figure in ``results/figures/`` from the committed results.

Usage::

    uv run python analysis/figures.py            # all figures
    uv run python analysis/figures.py fig2 fig4  # a subset, by prefix
    uv run python analysis/figures.py --refresh  # recompute the micro cache

Most figures read only ``results/<scenario>/*.json`` and
``uk_iran_conflict.scenarios`` (no microdata needed). Three of them —
the fuel decomposition, the policy-gain-by-decile panel and the
benefit-status figure — need household-level cuts that the committed JSON
does not carry, so they are recomputed from the PolicyEngine UK microdata
using exactly the same code path as ``analysis/run_incidence.py`` and cached
as JSON outside the repo. If the microdata is unavailable those three
figures are skipped with a loud message rather than faked.

Geography note: the dataset carries **region** (12) and **country** (4).
Its ``local_authority`` column is degenerate and there is no constituency
weight matrix in this dataset, so the constituency figure promised in the
brief is not producible here; ``fig4_region.png`` is the honest
substitute and ``fig5_fuel_decomposition.png`` replaces the constituency
scatter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uk_iran_conflict import scenarios as scen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CENTRAL = "realised_2026"
#: The undamped-pump run, used only as a visual upper-bound reference.
PEAK_FUEL = "realised_2026_peak_fuel"
SCENARIO_ORDER = ("niesr_baseline", "realised_2026", "niesr_adverse")
POLICY_ORDER = (
    "social_tariff",
    "jrf_block",
    "whd_expansion",
    "vat_zero",
    "ippr_rebate",
)
POLICY_SHORT = {
    "social_tariff": "Social tariff\n(means-tested)",
    "jrf_block": "JRF discounted\nblock",
    "whd_expansion": "Warm Home\nDiscount £150",
    "vat_zero": "VAT zero-rate\non fuel",
    "ippr_rebate": "IPPR flat\nrebate £183",
}
POLICY_COLORS = {
    "social_tariff": fs.BLUE,
    "jrf_block": fs.TEAL,
    "whd_expansion": fs.BLUE_LIGHT,
    "vat_zero": fs.GREY,
    "ippr_rebate": fs.TEAL_DARK,
}
CACHE = Path(tempfile.gettempdir()) / "uk_iran_conflict_micro_cache.json"


# --------------------------------------------------------------------------
# committed results
# --------------------------------------------------------------------------


def load(scenario: str, name: str) -> dict:
    return json.loads((RESULTS / scenario / f"{name}.json").read_text())


def shock(scenario: str = CENTRAL) -> dict:
    return load(scenario, "shock")


def deciles(scenario: str = CENTRAL) -> list[dict]:
    return sorted(shock(scenario)["decile"], key=lambda r: r["decile"])


# --------------------------------------------------------------------------
# micro cuts not present in the committed JSON
# --------------------------------------------------------------------------


def _micro_compute() -> dict:
    """Recompute the household-level cuts the figures need, from microdata."""
    from run_incidence import _load_env, dataset_path  # noqa: PLC0415

    from uk_iran_conflict import policies as pol  # noqa: PLC0415
    from uk_iran_conflict.incidence import (  # noqa: PLC0415
        income_for_ratio,
        load_baseline,
        shock_cost,
        wmean,
    )

    _load_env()
    path = dataset_path()
    base = load_baseline(path, 2026)
    mt = pol.means_tested_flag(path, 2026)
    cost = shock_cost(base, scen.SCENARIOS[CENTRAL])

    w = base.weight
    loss = cost.total
    # Every percentage-of-income cut in the paper is on the equivalised AHC
    # denominator (docs/FIXES.md D1); the figures must use the same concept, or
    # the ">5% of income" cut in fig8 would answer a different question from the
    # decile tables. Non-positive incomes are dropped from the ratio.
    income = income_for_ratio(base)
    pos = income > 0
    burden = np.zeros_like(loss)
    burden[pos] = 100 * loss[pos] / income[pos]

    out: dict = {"fuel_by_decile": [], "gain_by_decile": {}, "benefit": {}}

    for d in range(1, 11):
        sel = base.decile == d
        out["fuel_by_decile"].append(
            {
                "decile": d,
                "gas": wmean(cost.gas[sel], w[sel]),
                "electricity": wmean(cost.electricity[sel], w[sel]),
                "motor_fuel": wmean(cost.motor_fuel[sel], w[sel]),
            }
        )

    gains: dict[str, np.ndarray] = {}
    for key in POLICY_ORDER:
        gain = pol.POLICIES[key].gain(base, cost, mt)
        gains[key] = gain
        out["gain_by_decile"][key] = [
            wmean(gain[base.decile == d], w[base.decile == d]) for d in range(1, 11)
        ]

    # benefit-status cut
    groups = {"means_tested": mt, "not_means_tested": ~mt}
    for gname, sel in groups.items():
        wl = w[sel]
        entry = {
            "households_m": float(wl.sum() / 1e6),
            "share_of_households": float(wl.sum() / w.sum()),
            "mean_loss_gbp": wmean(loss[sel], wl),
            "mean_loss_pct": float(
                100
                * (loss[sel & pos] * w[sel & pos]).sum()
                / (income[sel & pos] * w[sel & pos]).sum()
            ),
            "mean_gain_gbp": {key: wmean(gains[key][sel], wl) for key in POLICY_ORDER},
            "uncompensated_share": {},
        }
        for key in POLICY_ORDER:
            losers = sel & (loss > 0)
            wln = w[losers]
            net = loss[losers] - gains[key][losers]
            entry["uncompensated_share"][key] = (
                float(wln[net > 0].sum() / wln.sum()) if wln.sum() > 0 else float("nan")
            )
        out["benefit"][gname] = entry

    # the anchor fact: of households carrying a heavy energy burden, how many
    # are outside the means-tested system the social tariff can reach?
    for threshold in (3.0, 5.0, 10.0):
        heavy = pos & (burden > threshold)
        wh = w[heavy]
        out["benefit"][f"heavy_burden_gt{threshold:.0f}pct"] = {
            "households_m": float(wh.sum() / 1e6),
            "share_not_means_tested": (
                float(w[heavy & ~mt].sum() / wh.sum()) if wh.sum() > 0 else float("nan")
            ),
        }
    # same cut, by decile, for the panel
    out["benefit"]["heavy_by_decile"] = [
        {
            "decile": d,
            "share_not_means_tested": (
                float(
                    w[(base.decile == d) & pos & (burden > 5) & ~mt].sum()
                    / max(w[(base.decile == d) & pos & (burden > 5)].sum(), 1e-9)
                )
            ),
            "households_m": float(
                w[(base.decile == d) & pos & (burden > 5)].sum() / 1e6
            ),
        }
        for d in range(1, 11)
    ]
    return out


def micro(refresh: bool = False) -> dict | None:
    """Cached micro cuts, or ``None`` if the microdata is unavailable."""
    if not refresh and CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            pass
    try:
        data = _micro_compute()
    except Exception as exc:  # noqa: BLE001 — absence of microdata is expected
        print(f"  !! microdata unavailable ({type(exc).__name__}: {exc})")
        return None
    CACHE.write_text(json.dumps(data, indent=1))
    return data


# ==========================================================================
# fig1 — scenario price paths and the gas/electricity asymmetry
# ==========================================================================


def fig1_price_path() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), width_ratios=[1.45, 1])

    quarters = scen.SCENARIOS[CENTRAL].quarter_labels
    baseline_cap = scen.SCENARIOS[CENTRAL].baseline_cap_gbp
    x = np.arange(len(quarters) + 1)
    xlabels = ["2026Q3\n(actual)", *[q.replace("Q", "\nQ") for q in quarters]]

    ax1.axhline(baseline_cap, color=fs.GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax1.text(
        x[-1] + 0.6,
        baseline_cap + 6,
        f"Jul–Sep 2026 cap £{baseline_cap:,.0f}",
        ha="right",
        va="bottom",
        fontsize=8,
        color=fs.GREY,
    )

    for key in SCENARIO_ORDER:
        sc = scen.SCENARIOS[key]
        caps = [baseline_cap, *[step.cap_gbp for step in sc.cap_path]]
        color = fs.SCENARIO_COLORS[key]
        ax1.step(
            x,
            caps,
            where="post",
            color=color,
            lw=2.4 if key == CENTRAL else 1.8,
            zorder=3,
            label=sc.label,
        )
        ax1.scatter(x, caps, s=18, color=color, zorder=4)
        ax1.annotate(
            f"£{caps[-1]:,.0f}",
            (x[-1], caps[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_xlim(-0.3, len(quarters) + 0.75)
    ax1.set_ylabel("Ofgem default tariff cap, £/yr (typical dual fuel)")
    ax1.yaxis.set_major_formatter(fs.GBP_FMT)
    ax1.set_title("A implied cap path: lagged, quantised into quarterly steps")
    ax1.legend(loc="upper left")
    fs.only_y_grid(ax1)

    # panel B — the asymmetry
    from uk_iran_conflict import reforms  # noqa: PLC0415

    keys = list(SCENARIO_ORDER)
    gas = []
    elec = []
    for key in keys:
        g, e = reforms.retail_factors(scen.SCENARIOS[key])
        gas.append(100 * (g - 1))
        elec.append(100 * (e - 1))

    idx = np.arange(len(keys))
    width = 0.36
    bg = ax2.bar(idx - width / 2, gas, width, color=fs.BLUE, label="Gas")
    be = ax2.bar(idx + width / 2, elec, width, color=fs.TEAL, label="Electricity")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(
        [scen.SCENARIOS[k].label.replace(" (", "\n(") for k in keys], fontsize=8.5
    )
    ax2.set_ylabel("Steady-state retail unit-price rise, %")
    ax2.yaxis.set_major_formatter(fs.PCT_FMT)
    ax2.set_ylim(0, max(gas) * 1.28)
    fs.label_bars(ax2, bg, gas, fmt="+{:.1f}%")
    fs.label_bars(ax2, be, elec, fmt="+{:.1f}%")
    ax2.set_title(
        "B gas and electricity move asymmetrically\n"
        "(gas sets the power price ~85% of the time)"
    )
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fs.only_y_grid(ax2)

    fig.suptitle(
        "Figure 1. From wholesale to the bill: scenario price paths, 2026Q4–2027Q3",
        y=1.06,
    )
    fs.note(
        fig,
        "Wholesale is ~45% of a dual-fuel bill and the cap lags the forward curve by "
        "2–3 quarters; electricity is attenuated by the marginal-pricing share and its "
        "smaller wholesale cost share.\nSource: authors' calculations, "
        "uk_iran_conflict.scenarios; Ofgem cap (Jul 2026 TDCV basis); "
        "Cornwall Insight 19 Aug 2026.",
        y=-0.06,
    )
    return fs.save(fig, "fig1_price_path.png")


# ==========================================================================
# fig2 — THE headline: £ vs % of income by decile
# ==========================================================================


def fig2_decile_dual() -> Path:
    rows = deciles()
    d = np.array([r["decile"] for r in rows])
    gbp = np.array([r["mean_loss_gbp"] for r in rows])
    pct = np.array([r["mean_loss_pct"] for r in rows])
    s = shock()

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 5.0), sharex=True)

    # A — cash
    bars = axl.bar(d, gbp, color=fs.BLUE, width=0.72)
    bars[0].set_color(fs.BLUE_LIGHT)
    bars[-1].set_color(fs.BLUE_LIGHT)
    axl.axhline(s["mean_loss_gbp"], color=fs.DARK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    axl.text(
        10.45,
        s["mean_loss_gbp"] + gbp.max() * 0.015,
        f"mean £{s['mean_loss_gbp']:,.0f}",
        fontsize=8,
        va="bottom",
        ha="right",
        color=fs.DARK,
    )
    axl.set_ylim(0, gbp.max() * 1.16)
    fs.label_bars(axl, bars, gbp, fmt="£{:,.0f}")
    axl.set_ylabel("Mean annual loss, £ per household")
    axl.yaxis.set_major_formatter(fs.GBP_FMT)
    axl.set_title("A cash: the rich lose more")
    fs.only_y_grid(axl)

    # B — share of income
    bars = axr.bar(d, pct, color=fs.TEAL, width=0.72)
    bars[0].set_color(fs.TEAL_DARK)
    axr.axhline(s["mean_loss_pct"], color=fs.DARK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    # On the left: the right-hand end of the mean line runs into the decile-9
    # and decile-10 value labels, which sit close to it.
    axr.text(
        0.55,
        s["mean_loss_pct"] + pct.max() * 0.015,
        f"mean {s['mean_loss_pct']:.2f}%",
        fontsize=8,
        va="bottom",
        ha="left",
        color=fs.DARK,
    )
    axr.set_ylim(0, pct.max() * 1.16)
    fs.label_bars(axr, bars, pct, fmt="{:.2f}%")
    axr.set_ylabel("Mean annual loss, % of equivalised income")
    axr.yaxis.set_major_formatter(fs.PCT_FMT)
    axr.set_title("B burden: the poor lose more")
    fs.only_y_grid(axr)

    ratio_gbp = gbp[-1] / gbp[0]
    ratio_pct = pct[0] / pct[-1]
    axl.annotate(
        f"D10 loses {ratio_gbp:.1f}× D1 in £",
        xy=(0.03, 0.93),
        xycoords="axes fraction",
        fontsize=9,
        color=fs.BLUE_DARK,
        fontweight="bold",
    )
    axr.annotate(
        f"D1 loses {ratio_pct:.1f}× D10 as a share of income",
        xy=(0.03, 0.93),
        xycoords="axes fraction",
        fontsize=9,
        color=fs.TEAL_DARK,
        fontweight="bold",
    )

    for ax in (axl, axr):
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Equivalised net income decile (1 = poorest)")
        ax.set_xlim(0.4, 10.6)

    fig.suptitle(
        "Figure 2. The same shock, two orderings: £ loss rises with income, "
        "burden falls",
        y=1.02,
    )
    fs.note(
        fig,
        f"Realised 2026, main specification; £{s['aggregate_cost_bn']:.2f}bn total, "
        f"mean £{s['mean_loss_gbp']:,.0f} per household "
        f"({s['mean_loss_pct']:.2f}% of income). First-order (Deaton) incidence: "
        "quantities fixed, so an upper bound. Income is equivalised AHC net income "
        "(HBAI scale); percent is the aggregate ratio of weighted loss to weighted "
        "income within the decile. "
        f"{s['coverage']['households_m']:.2f}m households "
        f"({100 * s['coverage']['share_of_households']:.1f}%, "
        f"{100 * s['coverage']['share_of_loss']:.1f}% of the loss) fall outside "
        "deciles 1-10 and are excluded from the decile columns; all of them have "
        "non-positive equivalised income.\nSource: authors' calculations on "
        "PolicyEngine UK (enhanced FRS 2023-24, uprated to 2026).",
        y=-0.03,
    )
    return fs.save(fig, "fig2_decile_dual.png")


# ==========================================================================
# fig3 — within-decile dispersion
# ==========================================================================


def fig3_within_decile() -> Path:
    rows = sorted(shock()["intra_decile"], key=lambda r: r["decile"])
    d = np.array([r["decile"] for r in rows])
    p10 = np.array([r["p10_loss_pct"] for r in rows])
    p50 = np.array([r["p50_loss_pct"] for r in rows])
    p90 = np.array([r["p90_loss_pct"] for r in rows])
    above5 = np.array([100 * r["share_above_5pct"] for r in rows])

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.4, 5.0), width_ratios=[1.5, 1], sharex=True
    )

    for xi, lo, hi in zip(d, p10, p90, strict=False):
        axl.plot([xi, xi], [lo, hi], color=fs.TEAL_LIGHT, lw=9, solid_capstyle="butt")
    axl.scatter(d, p10, s=26, color=fs.GREY, zorder=3, label="10th percentile")
    axl.scatter(d, p90, s=26, color=fs.BLUE, zorder=3, label="90th percentile")
    axl.plot(
        d,
        p50,
        color=fs.TEAL_DARK,
        lw=2.0,
        zorder=4,
        marker="o",
        ms=5,
        label="Median household",
    )
    axl.set_yscale("log")
    axl.set_yticks([0.03, 0.1, 0.3, 1, 3, 10])
    axl.set_yticklabels(["0.03%", "0.1%", "0.3%", "1%", "3%", "10%"])
    axl.set_ylabel("Loss as % of equivalised income (log scale)")
    axl.set_title("A within-decile spread dwarfs the between-decile gradient")
    axl.legend(loc="upper right", ncols=3, fontsize=8.5)
    fs.only_y_grid(axl)

    span_within = p90 / np.maximum(p10, 1e-9)
    axl.annotate(
        f"p90/p10 within a decile: {span_within.min():.0f}×–{span_within.max():.0f}×\n"
        f"median across deciles varies only "
        f"{p50.max() / p50.min():.1f}×",
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        fontsize=8.5,
        color=fs.DARK,
    )

    bars = axr.bar(d, above5, color=fs.BLUE, width=0.72)
    bars[0].set_color(fs.TEAL_DARK)
    axr.set_ylim(0, above5.max() * 1.2)
    fs.label_bars(axr, bars, above5, fmt="{:.1f}%")
    axr.set_ylabel("Share of households losing >5% of equivalised income")
    axr.yaxis.set_major_formatter(fs.PCT_FMT)
    axr.set_title("B heavy losers exist in every decile")
    fs.only_y_grid(axr)

    for ax in (axl, axr):
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Equivalised net income decile (1 = poorest)")
        ax.set_xlim(0.4, 10.6)

    fig.suptitle(
        "Figure 3. Dispersion of the loss within income deciles",
        y=1.02,
    )
    fs.note(
        fig,
        "Realised 2026, main specification. Bands span the weighted 10th–90th "
        "percentile of the household loss as a share of equivalised AHC net income; "
        "households with non-positive equivalised income are excluded from the "
        "ratio. The mean within-decile range exceeds the between-decile range, "
        "but the comparison is carried entirely by decile one: excluding it, "
        "the mean within-decile range falls below the between-decile one. The "
        "case for reporting within-decile outcomes (Cronin, Fullerton & Sexton "
        "2019) does not rest on that inequality.\nSource: authors' "
        "calculations on PolicyEngine UK.",
        y=-0.03,
    )
    return fs.save(fig, "fig3_within_decile.png")


# ==========================================================================
# fig4 — region panel (replaces the impossible constituency map)
# ==========================================================================


def fig4_region() -> Path:
    rows = shock()["region"]
    names = [fs.pretty_region(r["name"]) for r in rows]
    gbp = np.array([r["mean_loss_gbp"] for r in rows])
    pct = np.array([r["mean_loss_pct"] for r in rows])
    s = shock()

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.6, 5.2))

    order = np.argsort(gbp)
    y = np.arange(len(order))
    bars = axl.barh(y, gbp[order], color=fs.BLUE, height=0.68)
    bars[-1].set_color(fs.BLUE_DARK)
    axl.set_yticks(y)
    axl.set_yticklabels([names[i] for i in order])
    axl.set_xlim(0, gbp.max() * 1.2)
    axl.xaxis.set_major_formatter(fs.GBP_FMT)
    axl.set_xlabel("Mean annual loss, £ per household")
    axl.axvline(s["mean_loss_gbp"], color=fs.DARK, lw=1.0, ls=(0, (4, 3)))
    axl.text(
        s["mean_loss_gbp"],
        len(order) - 0.2,
        f"  UK mean £{s['mean_loss_gbp']:,.0f}",
        fontsize=8,
        color=fs.DARK,
        va="top",
    )
    fs.label_hbars(axl, bars, gbp[order], fmt="£{:,.0f}")
    axl.set_title("A cash loss")
    fs.only_x_grid(axl)

    order_p = np.argsort(pct)
    bars = axr.barh(y, pct[order_p], color=fs.TEAL, height=0.68)
    bars[-1].set_color(fs.TEAL_DARK)
    axr.set_yticks(y)
    axr.set_yticklabels([names[i] for i in order_p])
    axr.set_xlim(0, pct.max() * 1.2)
    axr.xaxis.set_major_formatter(fs.PCT_FMT)
    axr.set_xlabel("Mean annual loss, % of equivalised income")
    axr.axvline(s["mean_loss_pct"], color=fs.DARK, lw=1.0, ls=(0, (4, 3)))
    axr.text(
        s["mean_loss_pct"],
        len(order) - 0.2,
        f"  UK mean {s['mean_loss_pct']:.2f}%",
        fontsize=8,
        color=fs.DARK,
        va="top",
    )
    fs.label_hbars(axr, bars, pct[order_p], fmt="{:.2f}%")
    axr.set_title("B burden")
    fs.only_x_grid(axr)

    fig.suptitle(
        "Figure 4. Where the shock lands: mean household loss by region",
        y=1.03,
    )
    fs.note(
        fig,
        "Realised 2026 scenario, 12 ONS regions/nations. Unlike the decile view the "
        "two orderings largely agree: the North East is worst on both margins, London "
        "least exposed on both. The dataset carries region and country only, so no "
        "constituency-level cut is possible here.\nSource: authors' calculations on "
        "PolicyEngine UK.",
        y=-0.03,
    )
    return fs.save(fig, "fig4_region.png")


# ==========================================================================
# fig5 — fuel decomposition by decile
# ==========================================================================


def _spec_fuel_shares() -> list[tuple[str, float, float]]:
    """(label, motor-fuel share %, domestic share %) for each specification.

    Read from ``results/robustness/comparison.csv`` so the figure and
    ``tab_specifications`` cannot disagree.
    """
    import csv  # noqa: PLC0415

    path = RESULTS / "robustness" / "comparison.csv"
    with path.open(newline="") as fh:
        rows = {r["variant"]: r for r in csv.DictReader(fh)}
    order = (
        ("main", "Main"),
        ("steady_state", "Steady\nstate"),
        ("symmetric_damping", "Symmetric\ndamping"),
        ("peak_fuel", "Peak fuel\n(upper bound)"),
        ("ons_shape", "ONS\nshape"),
        ("ons_both_levels", "ONS both\nlevels"),
        ("unequivalised", "Unequiv-\nalised"),
    )
    out = []
    for key, label in order:
        r = rows.get(key)
        if r is None:
            continue
        motor = 100 * float(r["motor_fuel_share_of_loss"])
        domestic = 100 * (
            float(r["gas_share_of_loss"]) + float(r["electricity_share_of_loss"])
        )
        out.append((label, motor, domestic))
    return out


def _fuel_by_decile_rows() -> list[dict]:
    """Petrol/diesel split by decile, from ``results/sensitivity``."""
    import csv  # noqa: PLC0415

    path = RESULTS / "sensitivity" / "fuel_by_decile.csv"
    with path.open(newline="") as fh:
        return sorted(csv.DictReader(fh), key=lambda r: int(r["decile"]))


def fig5_fuel_decomposition(cache: dict) -> Path:
    """What the loss is made of — and how far that depends on the calibration.

    Panel C is the change the referee response requires: the motor-fuel share is
    no longer stated as a single number, because it is calibration-dependent.
    The range across the seven specifications, with the 50% line drawn, is the
    honest statement of the paper's channel-composition result.
    """
    rows = cache["fuel_by_decile"]
    d = np.array([r["decile"] for r in rows])
    gas = np.array([r["gas"] for r in rows])
    elec = np.array([r["electricity"] for r in rows])
    motor = np.array([r["motor_fuel"] for r in rows])
    # Split motor fuel into its two legs; diesel took a 21.6% uplift against
    # petrol's 12.0%, which is the likelier explanation of the decile-eight cash
    # spike than mileage (docs/FIXES.md D26).
    fuel_rows = _fuel_by_decile_rows()
    petrol = np.array([float(r["petrol_loss_gbp"]) for r in fuel_rows])
    diesel = np.array([float(r["diesel_loss_gbp"]) for r in fuel_rows])
    # Keep the panel's motor-fuel total exactly the cache's, splitting it in the
    # CSV's proportions, so rounding differences never open a visible gap.
    with np.errstate(divide="ignore", invalid="ignore"):
        diesel_share = np.where(petrol + diesel > 0, diesel / (petrol + diesel), 0.0)
    diesel = motor * diesel_share
    petrol = motor - diesel
    total = gas + elec + motor
    s = shock()

    fig, (axl, axr, axc) = plt.subplots(
        1, 3, figsize=(15.2, 5.2), width_ratios=[1.3, 1.3, 1.05]
    )

    axl.bar(d, gas, width=0.72, color=fs.FUEL_COLORS["gas"], label="Gas")
    axl.bar(
        d,
        elec,
        bottom=gas,
        width=0.72,
        color=fs.FUEL_COLORS["electricity"],
        label="Electricity",
    )
    axl.bar(
        d,
        petrol,
        bottom=gas + elec,
        width=0.72,
        color=fs.GREY_LIGHT,
        label="Petrol",
    )
    axl.bar(
        d,
        diesel,
        bottom=gas + elec + petrol,
        width=0.72,
        color=fs.FUEL_COLORS["motor_fuel"],
        label="Diesel",
    )
    for xi, t in zip(d, total, strict=False):
        axl.text(xi, t * 1.02, f"£{t:,.0f}", ha="center", va="bottom", fontsize=8)
    axl.set_ylim(0, total.max() * 1.18)
    axl.yaxis.set_major_formatter(fs.GBP_FMT)
    axl.set_ylabel("Mean annual loss, £ per household")
    axl.set_title("A composition of the cash loss")
    axl.legend(loc="upper left", fontsize=8.5)
    fs.only_y_grid(axl)

    share_gas = 100 * gas / total
    share_elec = 100 * elec / total
    share_motor = 100 * motor / total
    axr.bar(d, share_gas, width=0.72, color=fs.FUEL_COLORS["gas"])
    axr.bar(
        d, share_elec, bottom=share_gas, width=0.72, color=fs.FUEL_COLORS["electricity"]
    )
    axr.bar(
        d,
        share_motor,
        bottom=share_gas + share_elec,
        width=0.72,
        color=fs.FUEL_COLORS["motor_fuel"],
    )
    for xi, sg, se, sm in zip(d, share_gas, share_elec, share_motor, strict=False):
        axr.text(
            xi,
            sg / 2,
            f"{sg:.0f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
        )
        axr.text(
            xi,
            sg + se / 2,
            f"{se:.0f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=fs.DARK,
        )
        axr.text(
            xi,
            sg + se + sm / 2,
            f"{sm:.0f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
        )
    axr.set_ylim(0, 100)
    axr.yaxis.set_major_formatter(fs.PCT_FMT)
    axr.set_ylabel("Share of the household's loss, %")
    axr.set_title("B the composition is flat across deciles\n(main specification)")
    fs.only_y_grid(axr)

    for ax in (axl, axr):
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Equivalised net income decile (1 = poorest)")
        ax.set_xlim(0.4, 10.6)

    # C — the range the majority claim actually rests on
    specs = _spec_fuel_shares()
    labels = [label for label, _m, _d in specs]
    motor_shares = np.array([m for _l, m, _d in specs])
    y = np.arange(len(specs))[::-1]
    colors = [
        fs.TEAL_DARK if label.startswith("Main") else fs.GREY_LIGHT for label in labels
    ]
    bars = axc.barh(y, motor_shares, color=colors, height=0.62)
    axc.axvline(50, color=fs.DARK, lw=1.2, ls=(0, (4, 3)), zorder=4)
    # Sits under the bottom bar, where there is white space, rather than on top
    # of the leading bar's value label.
    axc.text(
        50.0,
        y.min() - 0.62,
        "50%: motor fuel stops\nbeing the majority",
        fontsize=7.5,
        color=fs.DARK,
        ha="center",
        va="top",
    )
    axc.set_ylim(y.min() - 1.5, y.max() + 0.6)
    fs.label_hbars(axc, bars, motor_shares, fmt="{:.1f}%")
    axc.set_yticks(y)
    axc.set_yticklabels(labels, fontsize=7.5)
    axc.set_xlim(0, 100)
    axc.xaxis.set_major_formatter(fs.PCT_FMT)
    axc.set_xlabel("Motor fuel, % of the aggregate loss")
    axc.set_title(
        "C the majority is calibration-dependent:\n"
        f"{motor_shares.min():.0f}–{motor_shares.max():.0f}% across specifications"
    )
    fs.only_x_grid(axc)

    fig.suptitle(
        "Figure 5. What the loss is made of: gas, electricity and motor fuel, "
        "and how far that depends on the calibration",
        y=1.02,
    )
    fs.note(
        fig,
        f"Panels A and B: realised 2026, main specification (equivalised AHC "
        f"denominator, Step 1 phase-in). Motor fuel is "
        f"{100 * s['motor_fuel_share_of_loss']:.0f}% of the aggregate loss here, gas "
        f"{100 * s['gas_share_of_loss']:.0f}% and electricity "
        f"{100 * s['electricity_share_of_loss']:.0f}%, so a domestic-bill instrument "
        f"can reach at most "
        f"{100 * (s['gas_share_of_loss'] + s['electricity_share_of_loss']):.0f}% of "
        f"the shock. Panel C: the same statistic under all "
        f"{len(specs)} specifications. The share depends only on the ratio of the gas "
        "and pump damping fractions, so it is a calibration result, not a robust "
        "fact.\nSource: authors' calculations on PolicyEngine UK (LCFS-imputed "
        "spend, NEED-calibrated quantities); results/robustness/comparison.csv.",
        y=-0.03,
    )
    return fs.save(fig, "fig5_fuel_decomposition.png")


# ==========================================================================
# fig6 — uncompensated losers by decile, by policy
# ==========================================================================


def fig6_uncompensated() -> Path:
    """Two views of the same compensation question.

    Panel A is the knife-edge measure the paper used to report on its own: any
    shortfall, however small, counts as uncompensated. Panel B is the continuous
    counterpart the referees asked for — the loss actually left on the table —
    and it ranks the instruments differently, which is the point of showing
    both. VAT zero-rating leaves 100% of losers formally uncompensated while
    leaving the *smallest* residual loss of the five.
    """
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.4, 5.0), sharex=True)
    d = np.arange(1, 11)

    for key in POLICY_ORDER:
        score = load(CENTRAL, key)
        y = np.array([100 * score["uncompensated_by_decile"][str(i)] for i in d])
        axl.plot(
            d,
            y,
            marker="o",
            ms=5,
            lw=2.2 if key in ("social_tariff", "ippr_rebate") else 1.6,
            color=POLICY_COLORS[key],
            label=(
                f"{POLICY_SHORT[key].replace(chr(10), ' ')} (£{score['cost_bn']:.1f}bn)"
            ),
        )

    axl.set_ylim(0, 105)
    axl.yaxis.set_major_formatter(fs.PCT_FMT)
    axl.set_ylabel("Losing households still worse off after the policy, %")
    axl.axhline(100, color=fs.GREY_LIGHT, lw=0.9)
    axl.text(
        10.2, 101, "nobody fully compensated", ha="right", fontsize=8, color=fs.GREY
    )
    axl.set_title("A the knife-edge measure: any shortfall counts")
    axl.legend(loc="lower right", ncols=2, fontsize=8)
    fs.only_y_grid(axl)

    loss = np.array([r["mean_loss_gbp"] for r in deciles()])
    axr.plot(
        d,
        loss,
        color=fs.DARK,
        lw=2.0,
        ls=(0, (4, 3)),
        zorder=5,
        label="Loss before any policy",
    )
    for key in POLICY_ORDER:
        score = load(CENTRAL, key)
        y = np.array([score["mean_residual_loss_by_decile"][str(i)] for i in d])
        axr.plot(
            d,
            y,
            marker="o",
            ms=5,
            lw=2.2 if key in ("vat_zero", "jrf_block") else 1.6,
            color=POLICY_COLORS[key],
            label=(
                f"{POLICY_SHORT[key].replace(chr(10), ' ')} — "
                f"{100 * score['share_of_aggregate_loss_offset']:.0f}% of the loss "
                "offset"
            ),
        )
    axr.set_ylim(0, loss.max() * 1.24)
    axr.yaxis.set_major_formatter(fs.GBP_FMT)
    axr.set_ylabel("Mean residual loss after the policy, £ per household")
    axr.set_title("B the continuous measure: what is actually left over")
    axr.legend(loc="upper left", ncols=1, fontsize=8)
    fs.only_y_grid(axr)

    for ax in (axl, axr):
        ax.set_xticks(d)
        ax.set_xlim(0.7, 10.3)
        ax.set_xlabel("Equivalised net income decile (1 = poorest)")

    fig.suptitle(
        "Figure 6. Compensation, measured two ways: the knife-edge share and "
        "the residual loss",
        y=1.02,
    )
    fs.note(
        fig,
        "Realised 2026, main specification, each instrument at its sponsor's own "
        "stated design. A household counts as uncompensated in panel A if its loss "
        "exceeds its gain, however narrowly; panel B shows the mean loss remaining "
        "after the gain. VAT zero-rating leaves every loser formally uncompensated — "
        "it returns only the 5% VAT on a domestic bill — yet leaves a smaller residual "
        "loss than the means-tested social tariff, because it reaches everyone. The "
        "two panels are the same policies in a different order, which is why the "
        "knife-edge share should not be reported alone.\nSource: authors' "
        "calculations on PolicyEngine UK.",
        y=-0.03,
    )
    return fs.save(fig, "fig6_uncompensated.png")


# ==========================================================================
# fig7 — mean gain by decile per policy, against the loss
# ==========================================================================


def fig7_policy_deciles(cache: dict) -> Path:
    d = np.arange(1, 11)
    loss = np.array([r["mean_loss_gbp"] for r in deciles()])

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    width = 0.16
    offsets = (np.arange(len(POLICY_ORDER)) - (len(POLICY_ORDER) - 1) / 2) * width

    for off, key in zip(offsets, POLICY_ORDER, strict=False):
        gains = np.array(cache["gain_by_decile"][key])
        score = load(CENTRAL, key)
        ax.bar(
            d + off,
            gains,
            width,
            color=POLICY_COLORS[key],
            label=(
                f"{POLICY_SHORT[key].replace(chr(10), ' ')} — "
                f"£{score['cost_bn']:.1f}bn, "
                f"{100 * score['share_to_bottom_three']:.0f}% to D1–3"
            ),
        )

    ax.plot(
        d,
        loss,
        color=fs.DARK,
        lw=2.2,
        marker="D",
        ms=5,
        zorder=5,
        label="Loss from the shock",
    )
    ax.set_xticks(d)
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, loss.max() * 1.22)
    ax.yaxis.set_major_formatter(fs.GBP_FMT)
    ax.set_xlabel("Equivalised net income decile (1 = poorest)")
    ax.set_ylabel("Mean annual amount, £ per household")
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    fs.only_y_grid(ax)
    ax.set_title(
        "Figure 7. Every option is small against the loss it is meant to offset",
        pad=12,
    )
    fs.note(
        fig,
        "Realised 2026 scenario; bars are the mean gain per household in the decile "
        "(including non-recipients), the line is the mean loss. Costs are simulated, "
        "not sponsors' stated costings.\nSource: authors' calculations on "
        "PolicyEngine UK.",
        y=-0.02,
    )
    return fs.save(fig, "fig7_policy_deciles.png")


# ==========================================================================
# fig8 — the crux: means-tested status
# ==========================================================================


def fig8_benefit_status(cache: dict) -> Path:
    b = cache["benefit"]
    mt = b["means_tested"]
    nmt = b["not_means_tested"]
    heavy = b["heavy_burden_gt5pct"]
    heavy_rows = b["heavy_by_decile"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), width_ratios=[0.85, 1.5, 1.2])
    axa, axb, axc = axes

    # A — the loss, by status
    groups = ["On a means-tested\nbenefit", "Not on a means-tested\nbenefit"]
    xs = np.arange(2)
    colors = [fs.TEAL_DARK, fs.BLUE]
    gbp_vals = [mt["mean_loss_gbp"], nmt["mean_loss_gbp"]]
    pct_vals = [mt["mean_loss_pct"], nmt["mean_loss_pct"]]

    bars = axa.bar(xs, gbp_vals, width=0.6, color=colors)
    axa.set_ylim(0, max(gbp_vals) * 1.25)
    fs.label_bars(axa, bars, gbp_vals, fmt="£{:,.0f}")
    for xi, p in zip(xs, pct_vals, strict=False):
        axa.text(
            xi,
            gbp_vals[xi] * 0.5,
            f"{p:.2f}%\nof income",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    axa.set_xticks(xs)
    axa.set_xticklabels(
        [
            f"{g}\n{h:.1f}m households"
            for g, h in zip(
                groups, [mt["households_m"], nmt["households_m"]], strict=False
            )
        ],
        fontsize=8.5,
    )
    axa.yaxis.set_major_formatter(fs.GBP_FMT)
    axa.set_ylabel("Mean annual loss, £ per household")
    axa.set_title("A the loss is larger outside\nthe means-tested system")
    fs.only_y_grid(axa)

    # B — compensation reaching each group
    width = 0.36
    idx = np.arange(len(POLICY_ORDER))
    g_mt = [mt["mean_gain_gbp"][k] for k in POLICY_ORDER]
    g_nmt = [nmt["mean_gain_gbp"][k] for k in POLICY_ORDER]
    b1 = axb.bar(
        idx - width / 2,
        g_mt,
        width,
        color=fs.TEAL_DARK,
        label="On a means-tested benefit",
    )
    b2 = axb.bar(
        idx + width / 2,
        g_nmt,
        width,
        color=fs.BLUE,
        label="Not on a means-tested benefit",
    )
    axb.set_ylim(0, max(g_mt + g_nmt) * 1.24)
    fs.label_bars(axb, b1, g_mt, fmt="£{:.0f}")
    fs.label_bars(axb, b2, g_nmt, fmt="£{:.0f}")
    axb.set_xticks(idx)
    axb.set_xticklabels([POLICY_SHORT[k] for k in POLICY_ORDER], fontsize=8)
    axb.yaxis.set_major_formatter(fs.GBP_FMT)
    axb.set_ylabel("Mean gain, £ per household per year")
    axb.set_title("B what each option pays\nto each group")
    axb.legend(loc="upper left", fontsize=8.5)
    fs.only_y_grid(axb)
    axb.annotate(
        "means-tested instruments\npay the other group nothing",
        xy=(0.30, 0.60),
        xycoords="axes fraction",
        fontsize=8.5,
        color=fs.GREY,
    )

    # C — the anchor fact, by decile
    d = np.array([r["decile"] for r in heavy_rows])
    share = np.array([100 * r["share_not_means_tested"] for r in heavy_rows])
    bars = axc.bar(d, share, width=0.72, color=fs.BLUE)
    for i in range(3):
        bars[i].set_color(fs.BLUE_DARK)
    axc.axhline(
        100 * heavy["share_not_means_tested"], color=fs.DARK, lw=1.2, ls=(0, (4, 3))
    )
    axc.text(
        10.4,
        100 * heavy["share_not_means_tested"] + 2,
        f"all deciles: {100 * heavy['share_not_means_tested']:.0f}%",
        ha="right",
        fontsize=8.5,
        color=fs.DARK,
        fontweight="bold",
    )
    axc.set_ylim(0, 105)
    axc.set_xticks(range(1, 11))
    axc.set_xlim(0.4, 10.6)
    axc.yaxis.set_major_formatter(fs.PCT_FMT)
    axc.set_xlabel("Equivalised net income decile")
    axc.set_ylabel("Share outside the means-tested system, %")
    axc.set_title(
        "C of households losing >5% of income,\nthe share a social tariff cannot reach"
    )
    fs.only_y_grid(axc)

    fig.suptitle(
        "Figure 8. The reach problem: the hardest-hit households are largely "
        "outside the means-tested system",
        y=1.05,
    )
    fs.note(
        fig,
        f"Realised 2026 scenario. {heavy['households_m']:.1f}m households lose more "
        f"than 5% of net income; "
        f"{100 * heavy['share_not_means_tested']:.0f}% of them receive no means-tested "
        "benefit, the counterpart of JRF's anchor statistic that ~40% of households "
        "struggling to heat their home are not on means-tested benefits. Means-tested "
        "receipt is any of UC, Pension Credit, tax credits, Housing Benefit, Income "
        "Support, income-based JSA/ESA.\nSource: authors' calculations on "
        "PolicyEngine UK.",
        y=-0.04,
    )
    return fs.save(fig, "fig8_benefit_status.png")


# ==========================================================================
# fig9 — the cash profile under each calibration (a withdrawn claim)
# ==========================================================================

#: ``results/robustness/cash_profiles.json`` key -> (legend label, colour).
CASH_PROFILE_STYLE = (
    ("raw", "Uncalibrated imputation", fs.GREY),
    ("ons_fuel_shape", "ONS motor-fuel shape", fs.BLUE),
    ("ons_both_levels", "ONS both levels", fs.TEAL),
)


def fig9_cash_profiles() -> Path:
    """Mean cash loss by decile under each calibration.

    The paper claimed the non-monotone cash profile — the decile-eight hump —
    was common to every specification. It is not. Both ONS calibrations are
    strictly monotone in cash and peak at decile ten; only the uncalibrated
    PolicyEngine imputation humps, and it humps in exactly the leg the ONS
    calibration corrects. Drawing all three together is the honest way to show
    that the hump is a property of one imputation rather than of the shock.
    """
    path = RESULTS / "robustness" / "cash_profiles.json"
    profiles = json.loads(path.read_text())["profiles"]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for key, label, colour in CASH_PROFILE_STYLE:
        profile = profiles.get(key)
        if profile is None:
            continue
        monotone = bool(profile["is_monotone_increasing"])
        ax.plot(
            profile["decile"],
            profile["mean_loss_gbp"],
            marker="o" if monotone else "s",
            markersize=4.5,
            linewidth=1.9,
            color=colour,
            linestyle="-" if monotone else "--",
            label=f"{label} ({'monotone' if monotone else 'non-monotone'})",
        )
        peak = int(profile["peak_decile"])
        ax.scatter(
            [peak],
            [profile["mean_loss_gbp"][peak - 1]],
            s=90,
            facecolors="none",
            edgecolors=colour,
            linewidths=1.4,
            zorder=5,
        )

    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Equivalised net income decile (1 = poorest)")
    ax.set_ylabel("Mean annual loss (£)")
    ax.set_title("Rings mark each profile's peak decile")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fs.only_y_grid(ax)

    fig.suptitle(
        "Figure 9. The decile-eight cash hump is an artefact of one imputation",
        y=1.0,
    )
    fs.note(
        fig,
        "Realised 2026. The decile-eight cash hump appears only in the "
        "uncalibrated PolicyEngine UK motor-fuel imputation; under both ONS "
        "calibrations the cash loss rises monotonically and peaks at decile "
        "ten. The claim that the hump is common to every specification is "
        "withdrawn. The percentage-of-income gradient, which is what the "
        "paper's distributional argument rests on, is unaffected: it is "
        "downward-sloping in all three.\nSource: authors' calculations on "
        "PolicyEngine UK; results/robustness/cash_profiles.json.",
        y=-0.04,
    )
    return fs.save(fig, "fig9_cash_profiles.png")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

NO_MICRO = {
    "fig1": fig1_price_path,
    "fig2": fig2_decile_dual,
    "fig3": fig3_within_decile,
    "fig4": fig4_region,
    "fig6": fig6_uncompensated,
    "fig9": fig9_cash_profiles,
}
NEEDS_MICRO = {
    "fig5": fig5_fuel_decomposition,
    "fig7": fig7_policy_deciles,
    "fig8": fig8_benefit_status,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="*", help="figure prefixes, e.g. fig2 fig5")
    parser.add_argument(
        "--refresh", action="store_true", help="recompute the micro cache"
    )
    args = parser.parse_args()

    wanted = set(args.which) or set(NO_MICRO) | set(NEEDS_MICRO)
    made: list[Path] = []

    for key, fn in NO_MICRO.items():
        if key in wanted:
            made.append(fn())
            print(f"  {made[-1].name}")

    micro_keys = wanted & set(NEEDS_MICRO)
    if micro_keys:
        cache = micro(refresh=args.refresh)
        if cache is None:
            print(
                f"  skipped {sorted(micro_keys)}: these need the PolicyEngine UK "
                "microdata (set HUGGING_FACE_TOKEN in .env)"
            )
        else:
            for key in sorted(micro_keys):
                made.append(NEEDS_MICRO[key](cache))
                print(f"  {made[-1].name}")

    print(f"\n{len(made)} figures written to {fs.FIGURES_DIR}")
    for path in made:
        size = path.stat().st_size
        status = "ok" if size > 10_000 else "SUSPICIOUSLY SMALL"
        print(f"  {path.name:34} {size / 1024:7.1f} kB  {status}")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()

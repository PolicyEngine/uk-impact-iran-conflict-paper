#!/usr/bin/env python3
"""Mechanically emit paper/values_generated.tex from canonical results files.

Every headline number in the manuscript enters prose as a macro from
``paper/values_generated.tex``; nothing is hand-transcribed. This script derives
each macro from the canonical files under ``results/`` and writes them as LaTeX
``\\newcommand``s. It does not edit the ``.tex`` prose itself.

Usage:  python analysis/emit_tex_values.py  ->  paper/values_generated.tex

If a source file or key is missing (a stale tree, a mid-refactor rename), the
macro is emitted as ``\\GENMISSING`` **with a comment naming the missing
source**, so an incomplete tree fails visibly at LaTeX build time rather than
silently keeping old numbers.

Macros emitted (grouped by source):

  the bare shock, per scenario (results/<scenario>/shock.json):
    \\gen<Scenario>ShockCostBn \\gen<Scenario>ShockDoneAvg
    \\gen<Scenario>ShockDoneRelPct \\gen<Scenario>ShockDtenAvg
    \\gen<Scenario>ShockDtenRelPct \\gen<Scenario>ShockPovRelAhcPp
    \\gen<Scenario>ShockGiniChangePp \\gen<Scenario>ShockLosersDoneePct
    \\gen<Scenario>ShockLosersDfivePct
    for <Scenario> in Baseline, Adverse, Realised

  policy responses, central scenario (results/<central>/<policy>.json):
    \\gen<Policy>CostBn \\gen<Policy>BottomThreeSharePct
    \\gen<Policy>CostPerPoundDone \\gen<Policy>PovRelAhcPp
    \\gen<Policy>LosersDoneePct \\gen<Policy>LosersDfivePct
    for <Policy> in SocialTariff, JrfBlock, WhdExpansion, VatZero, IpprRebate

  cross-cell spans (results/summary.csv):
    \\genCellCount \\genPolicyCostMinBn \\genPolicyCostMaxBn
    \\genLosersDoneMinPct \\genLosersDoneMaxPct

  constituency geography (results/geo/constituency_impacts.csv):
    \\genSeatCount \\genSeatWorstPctName \\genSeatWorstPctValue
    \\genSeatWorstCashName \\genSeatWorstCashValue \\genSeatRankCorrelation

  inequality, central scenario shock:
    \\genBaselineGini \\genBaselineTopOnePct \\genBaselineTopTenPct
    \\genBaselineBottomFiftyPct

  elasticity sensitivity (results/sensitivity/elasticity.csv):
    \\genElasticityCostMinBn \\genElasticityCostMaxBn
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "values_generated.tex"

#: The scenario whose cells carry the paper's headline numbers.
#: TODO(contract): must be a key of uk_iran_conflict.scenarios.SCENARIOS.
CENTRAL_SCENARIO = "niesr_adverse"

SCENARIO_MACRO = {
    "niesr_baseline": "Baseline",
    "niesr_adverse": "Adverse",
    "realised_2026": "Realised",
}
POLICY_MACRO = {
    "social_tariff": "SocialTariff",
    "jrf_block": "JrfBlock",
    "whd_expansion": "WhdExpansion",
    "vat_zero": "VatZero",
    "ippr_rebate": "IpprRebate",
}

lines: list[str] = []
missing: list[str] = []


#: Macro names already written, so a name emitted twice cannot reach the .tex.
#: LaTeX rejects a duplicate \newcommand outright, which would break the build
#: for a reason unrelated to the results — the paper-facing aliases below
#: deliberately overlap the results-shaped names for a few policy costs.
_emitted: set[str] = set()


def emit(name: str, value, fmt: str = "{:.1f}", source: str = "") -> None:
    """Emit one macro; on any failure emit a loud placeholder.

    First definition wins: a repeated ``name`` is skipped rather than emitted
    twice, since ``\\newcommand`` errors on redefinition.
    """
    if name in _emitted:
        return
    _emitted.add(name)
    try:
        if callable(value):
            value = value()
        text = fmt.format(value) if not isinstance(value, str) else value
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    except Exception as exc:  # noqa: BLE001 — every miss must surface, not abort
        missing.append(f"{name} ({source or 'unknown source'}: {exc})")
        lines.append(
            f"\\newcommand{{\\{name}}}{{\\GENMISSING}} % MISSING: {source} -> {exc}"
        )


def jload(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def _decile(cell: dict, block: str, decile: int) -> float:
    """Read one decile out of a result block (JSON keys come back as strings)."""
    values = cell[block]
    return float(values[str(decile)] if str(decile) in values else values[decile])


def _losers(cell: dict, decile: int) -> float:
    """Share of a decile left worse off — losers of any size."""
    bands = cell["intra_decile"]
    band = bands[str(decile)] if str(decile) in bands else bands[decile]
    return float(band["lose_less_5"] + band["lose_more_5"])


def main(draft: bool = False) -> None:
    # --- the bare shock, per scenario --------------------------------------
    for scenario, macro in SCENARIO_MACRO.items():
        rel = f"{scenario}/shock.json"
        emit(
            f"gen{macro}ShockCostBn",
            lambda rel=rel: jload(rel)["exchequer_cost"] / 1e9,
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{macro}ShockDoneAvg",
            lambda rel=rel: _decile(jload(rel), "decile_average_change", 1),
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{macro}ShockDoneRelPct",
            lambda rel=rel: _decile(jload(rel), "decile_relative_change", 1) * 100,
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{macro}ShockDtenAvg",
            lambda rel=rel: _decile(jload(rel), "decile_average_change", 10),
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{macro}ShockDtenRelPct",
            lambda rel=rel: _decile(jload(rel), "decile_relative_change", 10) * 100,
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{macro}ShockPovRelAhcPp",
            lambda rel=rel: jload(rel)["poverty_change"]["relative_ahc"] * 100,
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{macro}ShockGiniChangePp",
            lambda rel=rel: (
                (jload(rel)["gini_reform"] - jload(rel)["gini_baseline"]) * 100
            ),
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{macro}ShockLosersDoneePct",
            lambda rel=rel: _losers(jload(rel), 1) * 100,
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{macro}ShockLosersDfivePct",
            lambda rel=rel: _losers(jload(rel), 5) * 100,
            "{:.0f}",
            rel,
        )

    # --- policy responses, central scenario --------------------------------
    for policy, macro in POLICY_MACRO.items():
        rel = f"{CENTRAL_SCENARIO}/{policy}.json"
        emit(
            f"gen{macro}CostBn",
            lambda rel=rel: -jload(rel)["exchequer_cost"] / 1e9,
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{macro}PovRelAhcPp",
            lambda rel=rel: jload(rel)["poverty_change"]["relative_ahc"] * 100,
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{macro}LosersDoneePct",
            lambda rel=rel: _losers(jload(rel), 1) * 100,
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{macro}LosersDfivePct",
            lambda rel=rel: _losers(jload(rel), 5) * 100,
            "{:.0f}",
            rel,
        )

        def bottom_three(rel=rel) -> float:
            cell = jload(rel)
            gains = {
                int(k): v
                for k, v in cell["decile_average_change"].items()
                if float(v) > 0
            }
            total = sum(gains.values())
            return 100 * sum(gains.get(d, 0.0) for d in (1, 2, 3)) / total

        emit(f"gen{macro}BottomThreeSharePct", bottom_three, "{:.0f}", rel)
        emit(
            f"gen{macro}CostPerPoundDone",
            lambda rel=rel: (
                -jload(rel)["exchequer_cost"]
                / _decile(jload(rel), "decile_average_change", 1)
                / 1e6
            ),
            "{:.2f}",
            rel,
        )

    # --- cross-cell spans ---------------------------------------------------
    sm = "summary.csv"
    try:
        summary = pd.read_csv(R / sm)
        policies_only = summary[summary["policy"] != "shock"]
    except Exception:  # noqa: BLE001
        summary = policies_only = None
    emit("genCellCount", lambda: len(summary), "{:d}", sm)
    emit(
        "genPolicyCostMinBn",
        lambda: -float(policies_only["exchequer_cost_bn"].max()),
        "{:.1f}",
        sm,
    )
    emit(
        "genPolicyCostMaxBn",
        lambda: -float(policies_only["exchequer_cost_bn"].min()),
        "{:.1f}",
        sm,
    )
    emit(
        "genLosersDoneMinPct",
        lambda: 100 * float(policies_only["uncompensated_losers_decile1"].min()),
        "{:.0f}",
        sm,
    )
    emit(
        "genLosersDoneMaxPct",
        lambda: 100 * float(policies_only["uncompensated_losers_decile1"].max()),
        "{:.0f}",
        sm,
    )

    # --- constituency geography (the two-map figure) ------------------------
    geo = "geo/constituency_impacts.csv"
    try:
        seats = pd.read_csv(R / geo)
    except Exception:  # noqa: BLE001
        seats = None
    emit("genSeatCount", lambda: len(seats), "{:d}", geo)
    emit(
        "genSeatWorstPctName",
        lambda: str(seats.loc[seats["relative_change"].idxmin(), "name"]),
        "{}",
        geo,
    )
    emit(
        "genSeatWorstPctValue",
        lambda: 100 * float(seats["relative_change"].min()),
        "{:.2f}",
        geo,
    )
    emit(
        "genSeatWorstCashName",
        lambda: str(seats.loc[seats["average_change"].idxmin(), "name"]),
        "{}",
        geo,
    )
    emit(
        "genSeatWorstCashValue",
        lambda: float(seats["average_change"].min()),
        "{:.0f}",
        geo,
    )
    # The paper's Step 3 point: the £ and % maps rank seats almost oppositely.
    emit(
        "genSeatRankCorrelation",
        lambda: float(
            seats["average_change"].corr(seats["relative_change"], method="spearman")
        ),
        "{:.2f}",
        geo,
    )

    # --- baseline inequality ------------------------------------------------
    base = f"{CENTRAL_SCENARIO}/shock.json"
    emit("genBaselineGini", lambda: jload(base)["gini_baseline"], "{:.3f}", base)
    emit(
        "genBaselineTopOnePct",
        lambda: jload(base)["top_one_percent_share_baseline"] * 100,
        "{:.1f}",
        base,
    )
    emit(
        "genBaselineTopTenPct",
        lambda: jload(base)["top_ten_percent_share_baseline"] * 100,
        "{:.1f}",
        base,
    )
    emit(
        "genBaselineBottomFiftyPct",
        lambda: jload(base)["bottom_fifty_percent_share_baseline"] * 100,
        "{:.1f}",
        base,
    )

    # --- elasticity sensitivity (#1114: PolicyEngine has none of its own) ---
    el = "sensitivity/elasticity.csv"
    try:
        elast = pd.read_csv(R / el)
    except Exception:  # noqa: BLE001
        elast = None
    emit(
        "genElasticityCostMinBn",
        lambda: float(elast["exchequer_cost_bn"].min()),
        "{:.1f}",
        el,
    )
    emit(
        "genElasticityCostMaxBn",
        lambda: float(elast["exchequer_cost_bn"].max()),
        "{:.1f}",
        el,
    )

    # --- paper-facing macro names ------------------------------------------
    # PAPER-FACING MACROS. The prose in paper/sections/*.tex was drafted
    # against these names; the blocks above emit the results-shaped names.
    # Both are emitted from the SAME canonical files so the two schemes can
    # never disagree. If a name here is unused, LaTeX simply ignores it; if a
    # name used in prose is absent, the build fails on \GENMISSING by design.
    central = f"{CENTRAL_SCENARIO}/shock.json"
    realised = "realised_2026/shock.json"
    adverse = "niesr_adverse/shock.json"

    def _scen(rel, *keys, scale=1.0):
        cell = jload(rel)
        for k in keys:
            cell = cell[k]
        return float(cell) * scale

    # headline loss
    emit(
        "genCentralLossMean",
        lambda: -_scen(central, "mean_household_change"),
        "{:.0f}",
        central,
    )
    emit(
        "genCentralLossPctIncome",
        lambda: -_scen(central, "mean_relative_change", scale=100),
        "{:.2f}",
        central,
    )

    # decile losses, both metrics — the paper's central contrast
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"genDecile{tag}LossPct",
            lambda d=d: -_decile(jload(central), "decile_relative_change", d) * 100,
            "{:.2f}",
            central,
        )
        emit(
            f"genDecile{tag}LossGbp",
            lambda d=d: -_decile(jload(central), "decile_average_change", d),
            "{:.0f}",
            central,
        )
    emit(
        "genDecileRatioPct",
        lambda: (
            _decile(jload(central), "decile_relative_change", 1)
            / _decile(jload(central), "decile_relative_change", 10)
        ),
        "{:.1f}",
        central,
    )
    emit(
        "genBetweenDecileRangePct",
        lambda: (
            100
            * abs(
                _decile(jload(central), "decile_relative_change", 1)
                - _decile(jload(central), "decile_relative_change", 10)
            )
        ),
        "{:.2f}",
        central,
    )
    emit(
        "genWithinDecileRangePct",
        lambda: _scen(central, "within_decile_range", scale=100),
        "{:.2f}",
        central,
    )

    # constituency geography — the headline figure
    emit(
        "genConstituencyRankCorr",
        lambda: _scen(central, "constituency", "rank_correlation_gbp_vs_pct"),
        "{:.2f}",
        central,
    )
    emit(
        "genTailOverlapCount",
        lambda: _scen(central, "constituency", "tail_overlap_count"),
        "{:.0f}",
        central,
    )

    # price path
    emit(
        "genGasPeakPence",
        lambda: _scen(central, "prices", "gas_peak_pence"),
        "{:.0f}",
        central,
    )
    emit(
        "genOilPeakUsd",
        lambda: _scen(central, "prices", "oil_peak_usd"),
        "{:.0f}",
        central,
    )
    emit(
        "genOilPeakPct",
        lambda: _scen(central, "prices", "oil_peak_pct", scale=100),
        "{:.0f}",
        central,
    )
    emit(
        "genGasUnitRatePct",
        lambda: _scen(central, "prices", "gas_unit_rate_pct", scale=100),
        "{:.1f}",
        central,
    )
    emit(
        "genElecUnitRatePct",
        lambda: _scen(central, "prices", "electricity_unit_rate_pct", scale=100),
        "{:.1f}",
        central,
    )
    emit(
        "genCapAnnualised",
        lambda: _scen(central, "prices", "cap_annualised_gbp"),
        "{:.0f}",
        central,
    )
    emit(
        "genCapBaselineLevel",
        lambda: _scen(central, "prices", "cap_baseline_gbp"),
        "{:.0f}",
        central,
    )

    # aggregate spend by scenario
    emit(
        "genAggSpendRealisedBn",
        lambda: _scen(realised, "aggregate_energy_spend", scale=1 / 1e9),
        "{:.1f}",
        realised,
    )
    emit(
        "genAggSpendAdverseBn",
        lambda: _scen(adverse, "aggregate_energy_spend", scale=1 / 1e9),
        "{:.1f}",
        adverse,
    )

    # heating-regime heterogeneity
    emit(
        "genOffGridLossGbp",
        lambda: -_scen(central, "heating", "off_gas_grid_change"),
        "{:.0f}",
        central,
    )
    emit(
        "genOnGridLossGbp",
        lambda: -_scen(central, "heating", "on_gas_grid_change"),
        "{:.0f}",
        central,
    )

    # policy scorecard, paper-facing names
    _POLICY_FILE = {
        "SocialTariff": "social_tariff",
        "Jrf": "jrf_block",
        "Whd": "whd_expansion",
        "VatCut": "vat_cut",
        "Rebate": "ippr_rebate",
    }
    for tag, fname in _POLICY_FILE.items():
        prel = f"{CENTRAL_SCENARIO}/{fname}.json"
        emit(
            f"gen{tag}CostBn",
            lambda prel=prel: -jload(prel)["exchequer_cost"] / 1e9,
            "{:.1f}",
            prel,
        )
        emit(
            f"gen{tag}ShareBottomThree",
            lambda prel=prel: _scen(prel, "share_to_bottom_three", scale=100),
            "{:.0f}",
            prel,
        )
        emit(
            f"gen{tag}UncompensatedShare",
            lambda prel=prel: _scen(prel, "uncompensated_loser_share", scale=100),
            "{:.0f}",
            prel,
        )
    # the prose uses these two shorter aliases
    emit(
        "genJrfCostBn",
        lambda: -jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["exchequer_cost"] / 1e9,
        "{:.1f}",
        f"{CENTRAL_SCENARIO}/jrf_block.json",
    )
    emit(
        "genNonMeansTestedStruggling",
        lambda: _scen(central, "struggling_not_means_tested_share", scale=100),
        "{:.0f}",
        central,
    )

    # best/worst cost per £ of bottom-decile gain, across the five policies
    def _cost_per_pound(select):
        vals = []
        for fname in _POLICY_FILE.values():
            try:
                cell = jload(f"{CENTRAL_SCENARIO}/{fname}.json")
                gain = _decile(cell, "decile_average_change", 1)
                if gain > 0:
                    vals.append(-cell["exchequer_cost"] / 1e9 / gain)
            except Exception:  # noqa: BLE001
                continue
        if not vals:
            raise ValueError("no policy results available")
        return select(vals)

    emit(
        "genBestCostPerPound",
        lambda: _cost_per_pound(min),
        "{:.2f}",
        f"{CENTRAL_SCENARIO}/<policies>.json",
    )
    emit(
        "genWorstCostPerPound",
        lambda: _cost_per_pound(max),
        "{:.2f}",
        f"{CENTRAL_SCENARIO}/<policies>.json",
    )

    # appendix sensitivities
    ela = "sensitivity/elasticity.csv"
    emit(
        "genElasticityMeanLossHigh",
        lambda: -float(pd.read_csv(R / ela).iloc[-1]["mean_household_change"]),
        "{:.0f}",
        ela,
    )
    emit(
        "genElasticityRankCorrHigh",
        lambda: float(pd.read_csv(R / ela).iloc[-1]["constituency_rank_corr"]),
        "{:.2f}",
        ela,
    )
    lag = "sensitivity/cap_lag.csv"
    emit(
        "genLagFourAnnualisedShare",
        lambda: 100 * float(pd.read_csv(R / lag).iloc[-1]["annualised_share"]),
        "{:.0f}",
        lag,
    )
    emit(
        "genLagFourCumulativeShare",
        lambda: 100 * float(pd.read_csv(R / lag).iloc[-1]["cumulative_share"]),
        "{:.0f}",
        lag,
    )

    # --- write --------------------------------------------------------------
    header = [
        "% values_generated.tex — machine-generated by analysis/emit_tex_values.py",
        f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        "from canonical files under results/. DO NOT EDIT BY HAND.",
        "% \\GENMISSING marks values whose canonical source was absent at emit "
        "time; it errors at LaTeX build time by design.",
        (
            "% DRAFT MODE: \\GENMISSING renders as a visible marker instead of "
            "erroring. Never commit a submitted draft built this way."
            if draft
            else "% \\GENMISSING errors at build time; run without --draft."
        ),
        (
            "\\newcommand{\\GENMISSING}{\\textcolor{red}{\\textbf{[?]}}}"
            if draft
            else "\\newcommand{\\GENMISSING}{\\errmessage{emit_tex_values: "
            "missing canonical result}}"
        ),
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} macros)")
    if missing:
        print(f"WARNING: {len(missing)} macros missing canonical sources:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument(
        "--draft",
        action="store_true",
        help=(
            "Render missing values as a visible [?] marker instead of erroring, "
            "so an incomplete tree still compiles for reading. Never use for a "
            "submitted draft."
        ),
    )
    main(**vars(_parser.parse_args()))

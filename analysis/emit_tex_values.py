#!/usr/bin/env python3
r"""Mechanically emit ``paper/values_generated.tex`` from the canonical results.

Every headline number in the manuscript enters prose as a macro defined here;
nothing is hand-transcribed. This script derives each macro from the files under
``results/`` (plus the pure-data price paths in
:mod:`uk_iran_conflict.scenarios`, which need no microdata) and writes them as
LaTeX ``\newcommand``\ s. It never edits the prose.

Usage
-----
``python analysis/emit_tex_values.py``            -> ``paper/values_generated.tex``
``python analysis/emit_tex_values.py --draft``    -> ``\GENMISSING`` renders as a
red ``[?]`` instead of erroring, so an incomplete tree still compiles.

Safety design
-------------
* :func:`emit` never raises. A missing file or key produces
  ``\newcommand{\x}{\GENMISSING} % MISSING: <source> -> <error>`` and is listed
  on stdout at the end, so a stale tree fails **visibly** at LaTeX build time
  rather than silently keeping old numbers.
* First definition wins: a repeated macro name is skipped, because
  ``\newcommand`` errors on redefinition.

The real schema
---------------
``results/<scenario>/shock.json`` (dataclass ``incidence.ScenarioResult`` plus
one appended key)::

    scenario, aggregate_cost_bn, mean_loss_gbp, mean_loss_pct,
    gas_share_of_loss, electricity_share_of_loss, motor_fuel_share_of_loss,
    decile[10]       {decile, mean_loss_gbp, mean_loss_pct,
                      share_of_total_loss, households_m}
    intra_decile[10] {decile, p10_loss_pct, p50_loss_pct, p90_loss_pct,
                      share_above_5pct, share_above_10pct}
    region[12]       {name, mean_loss_gbp, mean_loss_pct, households_m}
    gini_baseline, gini_after, poverty_bhc_baseline, poverty_ahc_baseline,
    means_tested_share

``results/<scenario>/<policy>.json`` (dataclass ``policies.PolicyScore``)::

    policy, label, cost_bn, stated_cost_bn, share_to_bottom_three,
    cost_per_pound_decile_one, mean_gain_gbp, uncompensated_share_overall,
    uncompensated_by_decile{"1".."10"}, net_loss_after_policy_gbp,
    fully_compensated_share

``results/<scenario>/aggregates.json``::

    aggregate_energy_spend_bn, aggregate_fuel_spend_bn

Note on ``cost_per_pound_decile_one``: as stored it is *£bn of total cost per £1
of mean decile-one gain*, which is not a readable unit. The paper's
"cost per pound of bottom-decile gain" is recovered here as total cost divided
by the total gain accruing to decile one, i.e.

    cost_bn * 1e9 / (mean decile-one gain * decile-one households).

Sensitivity sources
-------------------
``results/sensitivity/elasticity.csv``  (14 rows: five named specs then the flat
grid) -- ``spec, epsilon_mean, mean_loss_gbp, aggregate_loss_bn,
share_of_upper_bound_shaved, decile1_loss_pct, decile10_loss_pct, ...``

``results/sensitivity/cap_lag.csv``  (4 rows, one per lag) --
``lag_quarters, annualised_mean_loss_gbp, cumulative_mean_loss_gbp,
cumulative_loss_bn, motor_fuel_bn, ...``

``results/sensitivity/asymmetry.csv``  (3 rows: 0.70 / 0.85 paper / 1.00) --
``marginal_pricing_share, mean_loss_gbp, decile_ratio_pct,
gas_share_of_domestic_loss, ...``

Values not yet in ``results/``
-----------------------------
Three prose numbers are computed inside ``analysis/run_incidence.py`` but not
persisted, and this emitter must run in CI without the private microdata token,
so they are carried here as documented constants with a TODO naming the fix.
See :data:`_PENDING_PERSIST`.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "values_generated.tex"

#: The observed shock is the paper's headline. The two NIESR paths are the
#: forward-looking bounds around it (methodology.tex, "Scenarios").
CENTRAL_SCENARIO = "realised_2026"

SCENARIOS = ("niesr_baseline", "niesr_adverse", "realised_2026")

#: Policy key -> the macro stem the prose uses. The prose names are shorter
#: than the file names (\genJrfCostBn, not \genJrfBlockCostBn).
POLICY_MACRO = {
    "social_tariff": "SocialTariff",
    "jrf_block": "Jrf",
    "whd_expansion": "Whd",
    "vat_zero": "VatCut",
    "ippr_rebate": "Rebate",
}

lines: list[str] = []
missing: list[str] = []

#: Macro names already written; a repeat is skipped rather than emitted twice.
_emitted: set[str] = set()


def emit(name: str, value, fmt: str = "{:.1f}", source: str = "") -> None:
    r"""Emit one macro; on any failure emit a loud ``\GENMISSING`` placeholder."""
    if name in _emitted:
        return
    _emitted.add(name)
    try:
        if callable(value):
            value = value()
        if value is None:
            raise ValueError("value is None")
        text = value if isinstance(value, str) else fmt.format(value)
        if text.strip() in {"", "nan", "inf", "-inf"}:
            raise ValueError(f"non-finite value {text!r}")
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    except Exception as exc:  # noqa: BLE001 — every miss must surface, not abort
        missing.append(f"{name} ({source or 'unknown source'}: {exc})")
        lines.append(
            f"\\newcommand{{\\{name}}}{{\\GENMISSING}} % MISSING: {source} -> {exc}"
        )


def jload(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def decile_row(shock: dict, block: str, decile: int) -> dict:
    """One row of ``decile`` or ``intra_decile``, keyed by decile number."""
    for row in shock[block]:
        if int(row["decile"]) == decile:
            return row
    raise KeyError(f"{block} has no decile {decile}")


def decile_one_households_m(shock: dict) -> float:
    return float(decile_row(shock, "decile", 1)["households_m"])


def cost_per_pound_decile_one(policy: dict, shock: dict) -> float:
    """Total cost, in £, per £1 of gain reaching decile one.

    ``cost_per_pound_decile_one`` in the JSON is £bn per £1 of *mean* decile-one
    gain, so the mean gain is recovered from it and multiplied by the number of
    decile-one households to get the gain actually delivered to that decile.
    """
    stored = float(policy["cost_per_pound_decile_one"])
    cost_bn = float(policy["cost_bn"])
    if not stored:
        raise ValueError("no decile-one gain")
    mean_gain_d1 = cost_bn / stored
    total_d1_gain = mean_gain_d1 * decile_one_households_m(shock) * 1e6
    if total_d1_gain <= 0:
        raise ValueError("non-positive decile-one gain")
    return cost_bn * 1e9 / total_d1_gain


# ---------------------------------------------------------------------------
# sensitivity CSV readers
# ---------------------------------------------------------------------------

SENS = R / "sensitivity"


def sload(name: str) -> list[dict]:
    """Rows of one ``results/sensitivity/*.csv`` as dicts of strings."""
    with (SENS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def srow(name: str, key: str, value: str) -> dict:
    """The single row of ``name`` whose column ``key`` equals ``value``."""
    for row in sload(name):
        if row[key].strip() == value:
            return row
    raise KeyError(f"{name}: no row with {key}={value}")


def snum(row: dict, column: str) -> float:
    return float(row[column])


#: The paper's calibrated lag, mirrored from
#: ``uk_iran_conflict.scenarios.CAP_LAG_QUARTERS`` but written literally so this
#: emitter stays importable without the package (CI runs it with no microdata).
PAPER_CAP_LAG = "3"

#: The marginal-pricing sweep endpoints and the paper's central value.
ASYM_LOW, ASYM_CENTRAL, ASYM_HIGH = "0.7", "0.85", "1.0"

# ---------------------------------------------------------------------------
# values the pipeline computes but does not yet persist
# ---------------------------------------------------------------------------

#: TODO(results): ``analysis/run_incidence.py`` should persist a ``medians``
#: block in ``results/<scenario>/shock.json`` carrying, per decile, the median
#: equivalised disposable income, the share of households with zero or negative
#: income, and the share of large losers outside the means-tested system. Until
#: it does, these three prose numbers are carried here as constants: they come
#: from the same verified run as the committed JSON (see ``docs/FINDINGS.md``
#: sections 4 and 8) and cannot be recomputed here, because this emitter must
#: run in CI without a Hugging Face token for the private microdata.
_PENDING_PERSIST = {
    "genDecileOneMedianIncomeGbp": (16000.0, "{:,.0f}"),
    "genDecileOneZeroIncomeShare": (0.57, "{:.2f}"),
    "genLargeLoserOutsideMeansTest": (98.0, "{:.0f}"),
}


def main(draft: bool = False) -> None:
    central = f"{CENTRAL_SCENARIO}/shock.json"

    # ------------------------------------------------------------------
    # headline loss, central (realised) scenario
    # ------------------------------------------------------------------
    emit(
        "genCentralLossMean", lambda: jload(central)["mean_loss_gbp"], "{:.0f}", central
    )
    emit(
        "genCentralLossPctIncome",
        lambda: jload(central)["mean_loss_pct"],
        "{:.2f}",
        central,
    )

    # decile contrast: £ rises across deciles, % falls. The paper's core point.
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"genDecile{tag}LossGbp",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )
        emit(
            f"genDecile{tag}LossPct",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            central,
        )
    emit(
        "genDecileRatioPct",
        lambda: (
            decile_row(jload(central), "decile", 1)["mean_loss_pct"]
            / decile_row(jload(central), "decile", 10)["mean_loss_pct"]
        ),
        "{:.1f}",
        central,
    )

    # between- vs within-decile dispersion (Cronin, Fullerton & Sexton 2019)
    emit(
        "genBetweenDecileRangePct",
        lambda: abs(
            decile_row(jload(central), "decile", 1)["mean_loss_pct"]
            - decile_row(jload(central), "decile", 10)["mean_loss_pct"]
        ),
        "{:.2f}",
        central,
    )

    def within_decile_range() -> float:
        """Mean across deciles of the within-decile p90-p10 loss-share range."""
        rows = jload(central)["intra_decile"]
        spreads = [r["p90_loss_pct"] - r["p10_loss_pct"] for r in rows]
        return sum(spreads) / len(spreads)

    emit("genWithinDecileRangePct", within_decile_range, "{:.2f}", central)

    # ------------------------------------------------------------------
    # aggregate additional spend, by scenario
    # ------------------------------------------------------------------
    for tag, scenario in (
        ("Realised", "realised_2026"),
        ("Adverse", "niesr_adverse"),
        ("Baseline", "niesr_baseline"),
    ):
        rel = f"{scenario}/shock.json"
        emit(
            f"genAggSpend{tag}Bn",
            lambda rel=rel: jload(rel)["aggregate_cost_bn"],
            "{:.1f}",
            rel,
        )

    # ------------------------------------------------------------------
    # price path (pure scenario data — no microdata needed)
    # ------------------------------------------------------------------
    def scenario_obj():
        from uk_iran_conflict import scenarios as scen  # noqa: PLC0415

        return scen.SCENARIOS[CENTRAL_SCENARIO]

    src = "uk_iran_conflict.scenarios.SCENARIOS[realised_2026]"
    emit(
        "genGasPeakPence",
        lambda: scenario_obj().gas.change_pence_per_therm,
        "{:.0f}",
        src,
    )
    emit("genOilPeakUsd", lambda: scenario_obj().oil.change_usd_per_bbl, "{:.0f}", src)
    emit("genOilPeakPct", lambda: 100 * scenario_obj().oil.pct_change, "{:.0f}", src)
    emit(
        "genGasUnitRatePct",
        lambda: 100 * scenario_obj().retail_shock.gas_pct_change,
        "{:.1f}",
        src,
    )
    emit(
        "genElecUnitRatePct",
        lambda: 100 * scenario_obj().retail_shock.electricity_pct_change,
        "{:.1f}",
        src,
    )
    emit("genCapAnnualised", lambda: scenario_obj().peak_cap_gbp, "{:.0f}", src)
    emit("genCapBaselineLevel", lambda: scenario_obj().baseline_cap_gbp, "{:.0f}", src)

    # ------------------------------------------------------------------
    # policy scorecard, central scenario
    # ------------------------------------------------------------------
    for policy, tag in POLICY_MACRO.items():
        rel = f"{CENTRAL_SCENARIO}/{policy}.json"
        emit(f"gen{tag}CostBn", lambda rel=rel: jload(rel)["cost_bn"], "{:.1f}", rel)
        emit(
            f"gen{tag}ShareBottomThree",
            lambda rel=rel: 100 * jload(rel)["share_to_bottom_three"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}UncompensatedShare",
            lambda rel=rel: 100 * jload(rel)["uncompensated_share_overall"],
            "{:.0f}",
            rel,
        )

    def cost_per_pound(select) -> float:
        shock = jload(central)
        values = []
        for policy in POLICY_MACRO:
            values.append(
                cost_per_pound_decile_one(
                    jload(f"{CENTRAL_SCENARIO}/{policy}.json"), shock
                )
            )
        return select(values)

    emit(
        "genBestCostPerPound",
        lambda: cost_per_pound(min),
        "{:.2f}",
        f"{CENTRAL_SCENARIO}/<policies>.json",
    )
    emit(
        "genWorstCostPerPound",
        lambda: cost_per_pound(max),
        "{:.2f}",
        f"{CENTRAL_SCENARIO}/<policies>.json",
    )

    # ------------------------------------------------------------------
    # composition of the loss: motor fuel is two thirds of it
    # ------------------------------------------------------------------
    for tag, scenario in (("", CENTRAL_SCENARIO), ("Adverse", "niesr_adverse")):
        rel = f"{scenario}/shock.json"
        for stem, key in (
            ("MotorFuel", "motor_fuel_share_of_loss"),
            ("Gas", "gas_share_of_loss"),
            ("Elec", "electricity_share_of_loss"),
        ):
            emit(
                f"gen{stem}ShareOfLoss{tag}",
                lambda rel=rel, key=key: 100 * jload(rel)[key],
                "{:.1f}",
                rel,
            )
        # every domestic-bill instrument is confined to this share by construction
        emit(
            f"genDomesticShareOfLoss{tag}",
            lambda rel=rel: (
                100
                * (
                    jload(rel)["gas_share_of_loss"]
                    + jload(rel)["electricity_share_of_loss"]
                )
            ),
            "{:.1f}",
            rel,
        )

    # ------------------------------------------------------------------
    # more of the decile profile: the cash peak at eight, and medians
    # ------------------------------------------------------------------
    for tag, d in (("Eight", 8), ("Nine", 9)):
        emit(
            f"genDecile{tag}LossGbp",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )
        emit(
            f"genDecile{tag}LossPct",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            central,
        )

    # The %-gradient survives on medians, so it is not a small-income artefact.
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"genDecile{tag}MedianLossPct",
            lambda d=d: decile_row(jload(central), "intra_decile", d)["p50_loss_pct"],
            "{:.2f}",
            central,
        )
    # Computed from the medians *as printed* (2dp), so a reader dividing the two
    # figures in the prose reproduces this ratio exactly.
    emit(
        "genMedianDecileRatioPct",
        lambda: (
            round(decile_row(jload(central), "intra_decile", 1)["p50_loss_pct"], 2)
            / round(decile_row(jload(central), "intra_decile", 10)["p50_loss_pct"], 2)
        ),
        "{:.1f}",
        central,
    )

    # decile one's tail: the widest within-decile spread in the distribution
    emit(
        "genDecileOnePNinetyLossPct",
        lambda: decile_row(jload(central), "intra_decile", 1)["p90_loss_pct"],
        "{:.2f}",
        central,
    )
    for tag, key in (
        ("Five", "share_above_5pct"),
        ("Ten", "share_above_10pct"),
    ):
        emit(
            f"genDecileOneShareAbove{tag}Pct",
            lambda key=key: 100 * decile_row(jload(central), "intra_decile", 1)[key],
            "{:.1f}",
            central,
        )

    # ------------------------------------------------------------------
    # region: the finest real geography this dataset supports
    # ------------------------------------------------------------------
    def region_extreme(select):
        return select(jload(central)["region"], key=lambda r: r["mean_loss_pct"])

    for tag, select in (("Max", max), ("Min", min)):
        emit(
            f"genRegion{tag}LossPct",
            lambda select=select: region_extreme(select)["mean_loss_pct"],
            "{:.2f}",
            central,
        )
        emit(
            f"genRegion{tag}LossGbp",
            lambda select=select: region_extreme(select)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )

    # ------------------------------------------------------------------
    # means-tested coverage (a declared limitation, not a code failure)
    # ------------------------------------------------------------------
    for name in ("genMeansTestedShareHouseholds", "genMeansTestedSharePct"):
        emit(
            name, lambda: 100 * jload(central)["means_tested_share"], "{:.1f}", central
        )
    emit(
        "genMeansTestedHouseholdsM",
        lambda: (
            jload(central)["means_tested_share"]
            * sum(r["households_m"] for r in jload(central)["decile"])
        ),
        "{:.1f}",
        central,
    )

    emit(
        "genRebateFullyCompensatedShare",
        lambda: (
            100
            * jload(f"{CENTRAL_SCENARIO}/ippr_rebate.json")["fully_compensated_share"]
        ),
        "{:.0f}",
        f"{CENTRAL_SCENARIO}/ippr_rebate.json",
    )

    # Removing a 5% reduced rate is a 1 - 1/1.05 cut in the VAT-inclusive bill:
    # an arithmetic ceiling on what zero-rating can offset.
    emit(
        "genVatCutMaxOffsetPct",
        lambda: 100 * (1 - 1 / 1.05),
        "{:.2f}",
        "arithmetic: 5% reduced rate removed from a VAT-inclusive bill",
    )

    # ------------------------------------------------------------------
    # sensitivity 1: demand response, the sweep that dominates
    # ------------------------------------------------------------------
    ela = "sensitivity/elasticity.csv"
    emit(
        "genElasticityAggZeroBn",
        lambda: snum(srow("elasticity.csv", "spec", "flat_0.0"), "aggregate_loss_bn"),
        "{:.2f}",
        ela,
    )
    emit(
        "genElasticityAggHighBn",
        lambda: snum(srow("elasticity.csv", "spec", "flat_-0.8"), "aggregate_loss_bn"),
        "{:.2f}",
        ela,
    )
    for tag, spec in (
        ("Labandeira", "labandeira_short_run"),
        ("Priesmann", "priesmann_short_run"),
    ):
        emit(
            f"gen{tag}ShortShaved",
            lambda spec=spec: (
                100
                * snum(
                    srow("elasticity.csv", "spec", spec), "share_of_upper_bound_shaved"
                )
            ),
            "{:.0f}",
            ela,
        )
    # income-varying elasticities flatten the gradient — the "heat or eat" effect
    emit(
        "genPriesmannDecileRatio",
        lambda: (
            snum(
                srow("elasticity.csv", "spec", "priesmann_short_run"),
                "decile1_loss_pct",
            )
            / snum(
                srow("elasticity.csv", "spec", "priesmann_short_run"),
                "decile10_loss_pct",
            )
        ),
        "{:.1f}",
        ela,
    )

    # ------------------------------------------------------------------
    # sensitivity 2: cap lag — cumulative is invariant, annualised is windowing
    # ------------------------------------------------------------------
    lag = "sensitivity/cap_lag.csv"
    emit(
        "genCapLagCumulativeMean",
        lambda: snum(
            srow("cap_lag.csv", "lag_quarters", PAPER_CAP_LAG),
            "cumulative_mean_loss_gbp",
        ),
        "{:.0f}",
        lag,
    )
    emit(
        "genCapLagCumulativeBn",
        lambda: snum(
            srow("cap_lag.csv", "lag_quarters", PAPER_CAP_LAG), "cumulative_loss_bn"
        ),
        "{:.2f}",
        lag,
    )
    for tag, which in (("LagOne", "1"), ("LagFour", "4"), ("Paper", PAPER_CAP_LAG)):
        emit(
            f"genCapLagAnnualised{tag}",
            lambda which=which: snum(
                srow("cap_lag.csv", "lag_quarters", which), "annualised_mean_loss_gbp"
            ),
            "{:.0f}",
            lag,
        )
    # the annualised 2026 total is almost purely the fast pump channel
    emit(
        "genMotorFuelAnnualisedBn",
        lambda: snum(
            srow("cap_lag.csv", "lag_quarters", PAPER_CAP_LAG), "motor_fuel_bn"
        ),
        "{:.1f}",
        lag,
    )

    # ------------------------------------------------------------------
    # sensitivity 3: marginal-pricing share — composition, not incidence
    # ------------------------------------------------------------------
    asym = "sensitivity/asymmetry.csv"

    def arow(share: str) -> dict:
        return srow("asymmetry.csv", "marginal_pricing_share", share)

    for tag, share in (("Low", ASYM_LOW), ("High", ASYM_HIGH)):
        emit(
            f"genAsymMeanLoss{tag}",
            lambda share=share: snum(arow(share), "mean_loss_gbp"),
            "{:.0f}",
            asym,
        )
        emit(
            f"genAsymRatio{tag}",
            lambda share=share: snum(arow(share), "decile_ratio_pct"),
            "{:.2f}",
            asym,
        )
    # As with the median ratio: taken off the mean losses as printed (whole
    # pounds), so the prose's "\pounds 459 to \pounds 483" implies this figure.
    emit(
        "genAsymMeanShiftPct",
        lambda: (
            100
            * (
                round(snum(arow(ASYM_HIGH), "mean_loss_gbp"))
                / round(snum(arow(ASYM_LOW), "mean_loss_gbp"))
                - 1
            )
        ),
        "{:.1f}",
        asym,
    )
    for tag, share in (
        ("Low", ASYM_LOW),
        ("Central", ASYM_CENTRAL),
        ("High", ASYM_HIGH),
    ):
        emit(
            f"genGasShareDomestic{tag}",
            lambda share=share: 100 * snum(arow(share), "gas_share_of_domestic_loss"),
            "{:.1f}",
            asym,
        )

    # ------------------------------------------------------------------
    # carried constants: see _PENDING_PERSIST
    # ------------------------------------------------------------------
    for name, (value, fmt) in _PENDING_PERSIST.items():
        emit(name, value, fmt, "carried constant — see _PENDING_PERSIST TODO")

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    header = [
        "% values_generated.tex — machine-generated by analysis/emit_tex_values.py",
        f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"from canonical files under results/ (central scenario: {CENTRAL_SCENARIO}).",
        "% DO NOT EDIT BY HAND.",
        "% \\GENMISSING marks values whose canonical source was absent or is not "
        "computable from this dataset; it errors at LaTeX build time by design.",
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
    resolved = len(lines) - len(missing)
    print(f"wrote {OUT} ({len(lines)} macros, {resolved} resolved)")
    if missing:
        print(f"WARNING: {len(missing)} macros emitted as \\GENMISSING:")
        for m in missing:
            print("  " + m)
    else:
        print("all macros resolved to real values")


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(ROOT))

    _parser = argparse.ArgumentParser(description="emit paper/values_generated.tex")
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

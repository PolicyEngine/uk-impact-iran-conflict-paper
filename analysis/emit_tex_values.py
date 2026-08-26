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

#: The undamped-pump run, retained as an explicit **upper bound** on the
#: motor-fuel channel: peak pump prices charged for a full twelve months. The
#: main specification damps them on the same logic as the wholesale gas peak
#: (``docs/VALIDATION.md`` Check 2b), so every headline in the paper is a range
#: whose top end comes from here.
PEAK_FUEL_SCENARIO = "realised_2026_peak_fuel"

#: The main specification re-scored on ONS-calibrated motor-fuel decile shares
#: (``docs/VALIDATION.md`` Check 2d). Robustness only; the national motor-fuel
#: total is preserved and only the decile profile is replaced.
ONS_FUEL_DIR = "robustness/ons_fuel"

SCENARIOS = ("niesr_baseline", "niesr_adverse", "realised_2026")

#: The seven specifications the referee response requires the paper to report
#: side by side. ``stem`` is the macro infix (no digits — LaTeX forbids them in
#: a control sequence), ``rel`` the ``shock.json`` under ``results/``, and
#: ``variant`` the matching row key in ``results/robustness/comparison.csv``.
#:
#: The first entry is the main specification (equivalised AHC denominator, D1,
#: plus the Step 1 consumption-weighted quarterly cap average, D2). Everything
#: else is an alternative or a robustness line, never a headline.
SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("Main", "realised_2026", "main", "Main (equivalised, phase-in)"),
    ("SteadyState", "robustness/steady_state", "steady_state", "Steady state"),
    (
        "SymDamp",
        "robustness/symmetric_damping",
        "symmetric_damping",
        "Symmetric damping",
    ),
    (
        "PeakFuel",
        "realised_2026_peak_fuel",
        "peak_fuel",
        "Peak fuel (upper bound)",
    ),
    ("OnsShape", "robustness/ons_fuel", "ons_shape", "ONS motor-fuel shape"),
    ("OnsLevels", "robustness/ons_levels", "ons_both_levels", "ONS both levels"),
    (
        "Unequiv",
        "robustness/unequivalised",
        "unequivalised",
        "Unequivalised (robustness)",
    ),
)

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


def note(text: str) -> None:
    """Write a comment line into the generated file (never a macro)."""
    lines.append(f"% {text}")


#: The three sweep CSVs under ``results/sensitivity/`` and ``results/grid/`` were
#: produced **before** the central scenario was re-specified, so every macro
#: derived from them describes the peak-fuel upper bound (undamped pump prices),
#: not the damped main specification. They cannot be recomputed here — the
#: sweeps need the private microdata and are written by
#: ``analysis/run_sensitivity.py`` / ``analysis/run_grid.py``. The macros are
#: emitted with a loud comment rather than suppressed, so the prose keeps
#: compiling while the provenance stays visible in the generated file.
STALE_SWEEP_WARNING = (
    "WARNING: the block below comes from a sweep CSV generated on the "
    "PEAK-FUEL (undamped pump) specification, not the damped main "
    "specification. Re-run analysis/run_sensitivity.py and analysis/run_grid.py "
    "to refresh, then re-run this emitter."
)


def sweep_is_stale() -> bool:
    """True if ``elasticity.csv``'s zero-elasticity row is not the main spec.

    The zero-elasticity row *is* the main specification by construction, so if
    its aggregate does not match ``results/realised_2026/shock.json`` the sweeps
    predate the re-specification.
    """
    try:
        zero = snum(srow("elasticity.csv", "spec", "flat_0.0"), "aggregate_loss_bn")
        main_agg = float(jload(f"{CENTRAL_SCENARIO}/shock.json")["aggregate_cost_bn"])
        return abs(zero - main_agg) > 0.05
    except Exception:  # noqa: BLE001 — a missing sweep is reported by emit()
        return False


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


def cload(rel: str) -> list[dict]:
    """Rows of any CSV under ``results/`` as dicts of strings."""
    with (R / rel).open(newline="") as fh:
        return list(csv.DictReader(fh))


def crow(rel: str, key: str, value: str) -> dict:
    for row in cload(rel):
        if row[key].strip() == value:
            return row
    raise KeyError(f"{rel}: no row with {key}={value}")


COMPARISON = "robustness/comparison.csv"
ENVELOPE = "sensitivity/policy_envelope.csv"
DOMESTIC_LEG = "sensitivity/domestic_leg.csv"
FUEL_BY_DECILE = "sensitivity/fuel_by_decile.csv"


def comparison_rows() -> dict[str, dict]:
    return {r["variant"]: r for r in cload(COMPARISON)}


def envelope_row(policy: str, envelope: str) -> dict:
    """One row of the common-envelope scorecard (``stated`` or ``common``)."""
    for row in cload(ENVELOPE):
        if row["policy"].strip() == policy and row["envelope"].strip() == envelope:
            return row
    raise KeyError(f"{ENVELOPE}: no {policy}/{envelope} row")


def leg_rows(parameter: str) -> list[dict]:
    rows = [r for r in cload(DOMESTIC_LEG) if r["parameter"].strip() == parameter]
    if not rows:
        raise KeyError(f"{DOMESTIC_LEG}: no rows for parameter {parameter!r}")
    return rows


def leg_span(parameter: str, column: str, select) -> float:
    return select(float(r[column]) for r in leg_rows(parameter))


def fuel_decile(decile: int) -> dict:
    return crow(FUEL_BY_DECILE, "decile", str(decile))


#: The paper's calibrated lag, mirrored from
#: ``uk_iran_conflict.scenarios.CAP_LAG_QUARTERS`` but written literally so this
#: emitter stays importable without the package (CI runs it with no microdata).
PAPER_CAP_LAG = "3"

#: The marginal-pricing sweep endpoints and the paper's central value.
ASYM_LOW, ASYM_CENTRAL, ASYM_HIGH = "0.7", "0.85", "1.0"

# ---------------------------------------------------------------------------
# persisted prose values (docs/FIXES.md C12)
# ---------------------------------------------------------------------------
#
# These were previously hardcoded literals in this file, which contradicted the
# appendix's guarantee that every number is emitted mechanically. They now come
# from ``results/persisted_values.json``, written by the incidence run, so a
# missing or stale tree produces ``\GENMISSING`` like everything else.


def persisted() -> dict:
    return jload("persisted_values.json")


def audit() -> dict:
    return jload("means_tested_audit.json")


#: The only surviving carried constant: the share of *large losers* outside the
#: means-tested system is a figure-cache statistic (``analysis/figures.py``
#: ``heavy_burden_gt5pct``), not a results-tree one. Kept explicit so the gap is
#: visible rather than disguised.
_CARRIED = {
    "genLargeLoserOutsideMeansTest": (98.0, "{:.0f}"),
}

# ---------------------------------------------------------------------------
# ONS benchmarks — the real published values, not our own rescaled output
# ---------------------------------------------------------------------------
#
# docs/FIXES.md C10: the paper attributed £521/£2,230 to ONS Family Spending.
# Those are *our model's* motor-fuel decile means after the ONS-shape rescaling,
# which preserves the microdata's national total. The published ONS numbers are
# £318 (decile 1) and £1,362 (decile 10), with an all-household mean of £960 for
# motor fuel and £1,780 for domestic energy. Both sets are emitted, under names
# that say which is which.
ONS_BENCH_FUEL = 960.0
ONS_BENCH_DOMESTIC = 1780.0

ONS_BENCH = {
    "genOnsBenchFuelDecileOneGbp": (318.0, "{:,.0f}"),
    "genOnsBenchFuelDecileTenGbp": (1362.0, "{:,.0f}"),
    "genOnsBenchFuelMeanGbp": (ONS_BENCH_FUEL, "{:,.0f}"),
    "genOnsBenchDomesticMeanGbp": (ONS_BENCH_DOMESTIC, "{:,.0f}"),
    "genOnsBenchFuelRatio": (1362.0 / 318.0, "{:.1f}"),
}


def _variant_macros(rel: str, stem: str) -> None:
    """Emit the standard headline block for one alternative specification.

    ``rel`` is a ``shock.json`` under ``results/``; ``stem`` is the macro
    infix (``PeakFuel``, ``OnsFuel``). Digits are never used in a macro name —
    LaTeX forbids them — so deciles stay spelled out as One and Ten.
    """
    emit(f"gen{stem}AggBn", lambda: jload(rel)["aggregate_cost_bn"], "{:.1f}", rel)
    emit(f"gen{stem}LossMean", lambda: jload(rel)["mean_loss_gbp"], "{:.0f}", rel)
    emit(f"gen{stem}LossPctIncome", lambda: jload(rel)["mean_loss_pct"], "{:.2f}", rel)
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"gen{stem}Decile{tag}LossGbp",
            lambda d=d: decile_row(jload(rel), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{stem}Decile{tag}LossPct",
            lambda d=d: decile_row(jload(rel), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            rel,
        )
    emit(
        f"gen{stem}DecileRatioPct",
        lambda: (
            decile_row(jload(rel), "decile", 1)["mean_loss_pct"]
            / decile_row(jload(rel), "decile", 10)["mean_loss_pct"]
        ),
        "{:.1f}",
        rel,
    )
    for label, key in (
        ("MotorFuel", "motor_fuel_share_of_loss"),
        ("Gas", "gas_share_of_loss"),
        ("Elec", "electricity_share_of_loss"),
    ):
        emit(
            f"gen{stem}{label}ShareOfLoss",
            lambda key=key: 100 * jload(rel)[key],
            "{:.1f}",
            rel,
        )
    emit(
        f"gen{stem}DomesticShareOfLoss",
        lambda: (
            100
            * (
                jload(rel)["gas_share_of_loss"]
                + jload(rel)["electricity_share_of_loss"]
            )
        ),
        "{:.1f}",
        rel,
    )


def main(draft: bool = False) -> None:
    central = f"{CENTRAL_SCENARIO}/shock.json"
    stale = sweep_is_stale()

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

    # the same statistic per instrument, so a sentence can name one directly
    # rather than only the best and worst (docs/FIXES.md D18)
    for policy, tag in POLICY_MACRO.items():
        emit(
            f"gen{tag}CostPerPound",
            lambda policy=policy: cost_per_pound_decile_one(
                jload(f"{CENTRAL_SCENARIO}/{policy}.json"), jload(central)
            ),
            "{:.2f}",
            f"{CENTRAL_SCENARIO}/{policy}.json",
        )
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
    # the two alternative specifications the audit requires
    #
    # The paper reports the fuel share and everything that hangs off it as a
    # *range*: main specification (pump peak damped like the gas peak) to
    # peak-fuel upper bound (pump peak charged for a full year). The
    # ONS-calibrated run is a separate axis — same shock, corrected motor-fuel
    # decile profile — and is reported alongside the main gradient, not inside
    # the range.
    # ------------------------------------------------------------------
    for stem, rel, _variant, _label in SPECS:
        _variant_macros(f"{rel}/shock.json", stem)
    #: ``OnsFuel`` is the name the prose used before the ONS run split into a
    #: shape-only and a both-levels specification; kept as an alias for the
    #: shape-only run so no existing sentence silently loses its number.
    _variant_macros(f"{ONS_FUEL_DIR}/shock.json", "OnsFuel")
    emit("genSpecCount", len(SPECS), "{:.0f}", "SPECS")
    # The main aggregate to two decimals, for sentences that compare
    # specifications whose totals differ in the second place (8.96 vs 8.93).
    emit(
        "genCentralAggBn",
        lambda: jload(central)["aggregate_cost_bn"],
        "{:.2f}",
        central,
    )
    for stem, rel, _variant, _label in SPECS:
        emit(
            f"gen{stem}AggPreciseBn",
            lambda rel=rel: jload(f"{rel}/shock.json")["aggregate_cost_bn"],
            "{:.2f}",
            rel,
        )

    # How much of the observed peak each channel is charged for over the year.
    emit(
        "genPumpSustainedFraction",
        lambda: 100 * scenario_obj().pass_through.pump_sustained_fraction,
        "{:.0f}",
        src,
    )
    emit(
        "genGasSustainedFraction",
        lambda: 100 * scenario_obj().pass_through.sustained_fraction,
        "{:.0f}",
        src,
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
    if stale:
        note(STALE_SWEEP_WARNING)
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
    if stale:
        note(STALE_SWEEP_WARNING)
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
    # The spread of the annualised figure across the plausible 1-4 quarter lag
    # range: the paper's "roughly £40" understates it (docs/FIXES.md D17).
    emit(
        "genCapLagRangeGbp",
        lambda: (
            max(float(r["annualised_mean_loss_gbp"]) for r in sload("cap_lag.csv"))
            - min(float(r["annualised_mean_loss_gbp"]) for r in sload("cap_lag.csv"))
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
    if stale:
        note(STALE_SWEEP_WARNING)
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
    # sensitivity 1b: the CV welfare bounds (docs/FIXES.md A5)
    #
    # ``aggregate_loss_bn`` in elasticity.csv is the change in *expenditure*.
    # Quoting it as the loss counts foregone heating as costless — the "heat or
    # eat" fallacy. The money-metric statement is the pair of bounds on the
    # compensating variation: Laspeyres (q0.dp, the zero-elasticity upper bound)
    # and Paasche (q1.dp, the lower bound). Demand response shaves the *welfare*
    # loss by far less than it shaves spending, which inverts the appendix's
    # ranking of uncertainties.
    # ------------------------------------------------------------------
    def erow(spec: str) -> dict:
        return srow("elasticity.csv", "spec", spec)

    emit(
        "genCvUpperBn",
        lambda: snum(erow("flat_0.0"), "cv_upper_bn"),
        "{:.2f}",
        ela,
    )
    emit(
        "genCvUpperMeanGbp",
        lambda: snum(erow("flat_0.0"), "cv_upper_mean_gbp"),
        "{:.0f}",
        ela,
    )
    emit(
        "genCvLowerHighEpsBn",
        lambda: snum(erow("flat_-0.8"), "cv_lower_bn"),
        "{:.2f}",
        ela,
    )
    emit(
        "genCvLowerHighEpsMeanGbp",
        lambda: snum(erow("flat_-0.8"), "cv_lower_mean_gbp"),
        "{:.0f}",
        ela,
    )
    # the correction itself: welfare shaved versus spending shaved, at eps=-0.8
    emit(
        "genWelfareShavedHighEpsPct",
        lambda: 100 * snum(erow("flat_-0.8"), "welfare_share_shaved"),
        "{:.1f}",
        ela,
    )
    emit(
        "genSpendShavedHighEpsPct",
        lambda: 100 * snum(erow("flat_-0.8"), "share_of_upper_bound_shaved"),
        "{:.0f}",
        ela,
    )
    # the largest welfare shave anywhere in the sweep — the honest ceiling on
    # how much demand response can matter for the paper's headline
    emit(
        "genWelfareShavedMaxPct",
        lambda: (
            100 * max(float(r["welfare_share_shaved"]) for r in sload("elasticity.csv"))
        ),
        "{:.1f}",
        ela,
    )
    emit(
        "genCvLowerMinBn",
        lambda: min(float(r["cv_lower_bn"]) for r in sload("elasticity.csv")),
        "{:.2f}",
        ela,
    )
    for tag, spec in (
        ("Labandeira", "labandeira_short_run"),
        ("Priesmann", "priesmann_short_run"),
    ):
        emit(
            f"gen{tag}ShortWelfareShaved",
            lambda spec=spec: 100 * snum(erow(spec), "welfare_share_shaved"),
            "{:.1f}",
            ela,
        )
        emit(
            f"gen{tag}ShortSpendShaved",
            lambda spec=spec: 100 * snum(erow(spec), "share_of_upper_bound_shaved"),
            "{:.0f}",
            ela,
        )
        emit(
            f"gen{tag}ShortCvLowerBn",
            lambda spec=spec: snum(erow(spec), "cv_lower_bn"),
            "{:.2f}",
            ela,
        )

    # ------------------------------------------------------------------
    # the seven specifications: ranges across them (docs/FIXES.md A3)
    #
    # The motor-fuel majority claim is calibration-dependent, so every headline
    # derived from it is reported as a range over the specifications rather than
    # as a single share.
    # ------------------------------------------------------------------
    def spec_values(column: str) -> list[float]:
        rows = comparison_rows()
        out = []
        for _stem, _rel, variant, _label in SPECS:
            out.append(float(rows[variant][column]))
        return out

    for name, column, fmt, scale in (
        ("genSpecAggMinBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genSpecAggMaxBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genSpecLossMeanMin", "mean_loss_gbp", "{:.0f}", 1),
        ("genSpecLossMeanMax", "mean_loss_gbp", "{:.0f}", 1),
        ("genSpecFuelShareMinPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genSpecFuelShareMaxPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genSpecDecileRatioMin", "d1_d10_ratio_pct", "{:.2f}", 1),
        ("genSpecDecileRatioMax", "d1_d10_ratio_pct", "{:.2f}", 1),
    ):
        select = (
            min if name.endswith(("MinBn", "MeanMin", "MinPct", "RatioMin")) else max
        )
        emit(
            name,
            lambda column=column, select=select, scale=scale: (
                scale * select(spec_values(column))
            ),
            fmt,
            COMPARISON,
        )
    # how far the specification choice moves the aggregate, as a share of the
    # main specification — the number the uncertainty ranking is built on
    emit(
        "genSpecAggSpreadPct",
        lambda: (
            100
            * (
                max(spec_values("aggregate_cost_bn"))
                - min(spec_values("aggregate_cost_bn"))
            )
            / float(jload(central)["aggregate_cost_bn"])
        ),
        "{:.0f}",
        COMPARISON,
    )
    # the fuel share excluding the explicit peak-fuel upper bound: the range a
    # reader should use when the upper bound is not the quantity of interest
    emit(
        "genSpecFuelShareMaxExPeakPct",
        lambda: (
            100
            * max(
                float(comparison_rows()[variant]["motor_fuel_share_of_loss"])
                for _s, _r, variant, _l in SPECS
                if variant != "peak_fuel"
            )
        ),
        "{:.1f}",
        COMPARISON,
    )
    # phase-in weights actually applied to the domestic leg (decision D2)
    for name, column in (
        ("genAnnualPhaseInGasPct", "annual_phase_in_gas"),
        ("genAnnualPhaseInElecPct", "annual_phase_in_electricity"),
    ):
        emit(
            name,
            lambda column=column: 100 * float(jload(central)[column]),
            "{:.1f}",
            central,
        )

    # ------------------------------------------------------------------
    # decile coverage and the dropped weight (docs/FIXES.md A6)
    # ------------------------------------------------------------------
    def cov() -> dict:
        return jload(central)["coverage"]

    emit(
        "genCoverageExcludedHouseholdsM",
        lambda: cov()["households_m"],
        "{:.2f}",
        central,
    )
    emit(
        "genCoverageExcludedSharePct",
        lambda: 100 * cov()["share_of_households"],
        "{:.2f}",
        central,
    )
    emit(
        "genCoverageExcludedLossSharePct",
        lambda: 100 * cov()["share_of_loss"],
        "{:.2f}",
        central,
    )
    emit(
        "genCoverageExcludedZeroIncomeSharePct",
        lambda: 100 * cov()["zero_or_negative_income_share"],
        "{:.0f}",
        central,
    )
    emit(
        "genCoveredHouseholdsM",
        lambda: cov()["covered_households_m"],
        "{:.2f}",
        central,
    )
    emit("genCoveredLossBn", lambda: cov()["covered_loss_bn"], "{:.2f}", central)
    emit(
        "genZeroOrNegIncomeSharePct",
        lambda: 100 * jload(central)["zero_or_negative_income_share"],
        "{:.2f}",
        central,
    )

    # ------------------------------------------------------------------
    # persisted prose values (docs/FIXES.md C12), including the corrected
    # decile-one zero-income share: 20.05% equivalised, not 0.57%
    # ------------------------------------------------------------------
    pv = "persisted_values.json"
    emit(
        "genDecileOneZeroIncomeShare",
        lambda: (
            100 * persisted()["decile1_zero_or_negative_income_share_equivalised_ahc"]
        ),
        "{:.2f}",
        pv,
    )
    emit(
        "genDecileOneZeroIncomeShareUnequiv",
        lambda: (
            100 * persisted()["decile1_zero_or_negative_income_share_unequivalised"]
        ),
        "{:.2f}",
        pv,
    )
    emit(
        "genDecileOneMedianIncomeGbp",
        lambda: persisted()["decile1_median_income_gbp_equivalised_ahc"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genDecileOneMeanIncomeGbp",
        lambda: persisted()["decile1_mean_income_gbp_equivalised_ahc"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genDecileOneMedianIncomeUnequivGbp",
        lambda: persisted()["decile1_median_income_gbp_unequivalised"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genModelDomesticMeanGbp",
        lambda: persisted()["mean_domestic_energy_spend_gbp"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genModelFuelMeanGbp",
        lambda: persisted()["mean_motor_fuel_spend_gbp"],
        "{:,.0f}",
        pv,
    )
    # the raw imputation, and the profile after the ONS-shape rescaling. The
    # rescaled numbers are OURS, not ONS's: see ONS_BENCH for the published ones.
    for tag, index in (("One", 0), ("Ten", 9)):
        emit(
            f"genFuelSpendDecile{tag}RawGbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"]["raw"][index],
            "{:,.0f}",
            pv,
        )
        emit(
            f"genFuelSpendDecile{tag}OnsGbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"]["ons_shape"][
                index
            ],
            "{:,.0f}",
            pv,
        )
        emit(
            f"genRescaledFuelDecile{tag}Gbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"]["ons_shape"][
                index
            ],
            "{:,.0f}",
            pv,
        )
        emit(
            f"genOnsLevelsFuelDecile{tag}Gbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"][
                "ons_both_levels"
            ][index],
            "{:,.0f}",
            pv,
        )

    # the two disclosed imputation defects, in both directions (C11)
    for name, (value, fmt) in ONS_BENCH.items():
        emit(name, value, fmt, "ONS Family Spending FYE 2025 (published)")
    emit(
        "genDomesticUnderImputationPct",
        lambda: (
            100
            * (1 - persisted()["mean_domestic_energy_spend_gbp"] / ONS_BENCH_DOMESTIC)
        ),
        "{:.0f}",
        pv,
    )
    emit(
        "genFuelOverImputationPct",
        lambda: 100 * (persisted()["mean_motor_fuel_spend_gbp"] / ONS_BENCH_FUEL - 1),
        "{:.0f}",
        pv,
    )

    # ------------------------------------------------------------------
    # continuous compensation measures at each sponsor's stated design (B9)
    # ------------------------------------------------------------------
    for policy, tag in POLICY_MACRO.items():
        rel = f"{CENTRAL_SCENARIO}/{policy}.json"
        emit(
            f"gen{tag}OffsetSharePct",
            lambda rel=rel: 100 * jload(rel)["share_of_aggregate_loss_offset"],
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualMeanGbp",
            lambda rel=rel: jload(rel)["mean_residual_loss_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualMedianGbp",
            lambda rel=rel: jload(rel)["median_residual_loss_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualDecileOneGbp",
            lambda rel=rel: jload(rel)["mean_residual_loss_by_decile"]["1"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualDecileTenGbp",
            lambda rel=rel: jload(rel)["mean_residual_loss_by_decile"]["10"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}OffsetDecileOnePct",
            lambda rel=rel: 100 * jload(rel)["share_of_loss_offset_by_decile"]["1"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}MeanGainGbp",
            lambda rel=rel: jload(rel)["mean_gain_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}FullyCompensatedShare",
            lambda rel=rel: 100 * jload(rel)["fully_compensated_share"],
            "{:.0f}",
            rel,
        )

    # The JRF block, re-specified as a level subsidy, now costs more than JRF's
    # own stated £5bn rather than a third of it (B7).
    emit(
        "genJrfStatedCostBn",
        lambda: jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["stated_cost_bn"],
        "{:.1f}",
        f"{CENTRAL_SCENARIO}/jrf_block.json",
    )
    emit(
        "genJrfCostOverStated",
        lambda: (
            jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["cost_bn"]
            / jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["stated_cost_bn"]
        ),
        "{:.1f}",
        f"{CENTRAL_SCENARIO}/jrf_block.json",
    )

    # ------------------------------------------------------------------
    # the common-envelope scorecard (B7): all five instruments at £5bn
    # ------------------------------------------------------------------
    emit(
        "genEnvelopeBn",
        lambda: snum(envelope_row("vat_zero", "common"), "envelope_bn"),
        "{:.0f}",
        ENVELOPE,
    )
    for policy, tag in POLICY_MACRO.items():
        emit(
            f"gen{tag}EnvelopeOffsetPct",
            lambda policy=policy: (
                100
                * snum(envelope_row(policy, "common"), "share_of_aggregate_loss_offset")
            ),
            "{:.1f}",
            ENVELOPE,
        )
        emit(
            f"gen{tag}EnvelopeResidualMeanGbp",
            lambda policy=policy: snum(
                envelope_row(policy, "common"), "mean_residual_loss_gbp"
            ),
            "{:.0f}",
            ENVELOPE,
        )
        emit(
            f"gen{tag}EnvelopeResidualMedianGbp",
            lambda policy=policy: snum(
                envelope_row(policy, "common"), "median_residual_loss_gbp"
            ),
            "{:.0f}",
            ENVELOPE,
        )
        emit(
            f"gen{tag}EnvelopeResidualDecileOneGbp",
            lambda policy=policy: snum(
                envelope_row(policy, "common"), "mean_residual_loss_d1"
            ),
            "{:.0f}",
            ENVELOPE,
        )
        emit(
            f"gen{tag}EnvelopeMeanGainGbp",
            lambda policy=policy: snum(envelope_row(policy, "common"), "mean_gain_gbp"),
            "{:.0f}",
            ENVELOPE,
        )
        emit(
            f"gen{tag}EnvelopeUncompensatedPct",
            lambda policy=policy: (
                100
                * snum(envelope_row(policy, "common"), "uncompensated_share_overall")
            ),
            "{:.0f}",
            ENVELOPE,
        )
        emit(
            f"gen{tag}EnvelopeScale",
            lambda policy=policy: snum(
                envelope_row(policy, "common"), "envelope_scale"
            ),
            "{:.2f}",
            ENVELOPE,
        )

    def envelope_best(select, column: str) -> dict:
        rows = [r for r in cload(ENVELOPE) if r["envelope"].strip() == "common"]
        return select(rows, key=lambda r: float(r[column]))

    #: Short, LaTeX-safe prose names. The CSV ``label`` carries "%" and "->",
    #: which would break the build, so it is never emitted verbatim.
    envelope_names = {
        "social_tariff": "the means-tested social tariff",
        "jrf_block": "the JRF discounted block",
        "whd_expansion": "the Warm Home Discount expansion",
        "vat_zero": "VAT zero-rating",
        "ippr_rebate": "the flat rebate",
    }
    emit(
        "genEnvelopeBestLabel",
        lambda: envelope_names[
            envelope_best(max, "share_of_aggregate_loss_offset")["policy"].strip()
        ],
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeBestResidualLabel",
        lambda: envelope_names[
            envelope_best(min, "mean_residual_loss_gbp")["policy"].strip()
        ],
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeBestOffsetPct",
        lambda: (
            100
            * float(
                envelope_best(max, "share_of_aggregate_loss_offset")[
                    "share_of_aggregate_loss_offset"
                ]
            )
        ),
        "{:.1f}",
        ENVELOPE,
    )
    emit(
        "genEnvelopeBestResidualMeanGbp",
        lambda: float(
            envelope_best(min, "mean_residual_loss_gbp")["mean_residual_loss_gbp"]
        ),
        "{:.0f}",
        ENVELOPE,
    )
    emit(
        "genEnvelopeWorstOffsetPct",
        lambda: (
            100
            * float(
                envelope_best(min, "share_of_aggregate_loss_offset")[
                    "share_of_aggregate_loss_offset"
                ]
            )
        ),
        "{:.1f}",
        ENVELOPE,
    )

    # ------------------------------------------------------------------
    # domestic-leg parameter sweep (A4): only the product is identified
    # ------------------------------------------------------------------
    for tag, parameter in (
        ("Split", "sustained_fraction_split"),
        ("Nbp", "prewar_nbp_pence_per_therm"),
        ("GasShare", "wholesale_share_gas_bill"),
        ("ElecShare", "wholesale_share_electricity_bill"),
    ):
        for suffix, select in (("Min", min), ("Max", max)):
            emit(
                f"genDomesticLeg{tag}Agg{suffix}Bn",
                lambda parameter=parameter, select=select: leg_span(
                    parameter, "aggregate_cost_bn", select
                ),
                "{:.2f}",
                DOMESTIC_LEG,
            )
            emit(
                f"genDomesticLeg{tag}Mean{suffix}Gbp",
                lambda parameter=parameter, select=select: leg_span(
                    parameter, "mean_loss_gbp", select
                ),
                "{:.0f}",
                DOMESTIC_LEG,
            )
            emit(
                f"genDomesticLeg{tag}FuelShare{suffix}Pct",
                lambda parameter=parameter, select=select: (
                    100 * leg_span(parameter, "motor_fuel_share_of_loss", select)
                ),
                "{:.1f}",
                DOMESTIC_LEG,
            )
    emit(
        "genDomesticLegAnchorProduct",
        lambda: float(leg_rows("sustained_fraction_split")[0]["anchor_product"]),
        "{:.3f}",
        DOMESTIC_LEG,
    )
    for name, select in (
        ("genDomesticLegAggMinBn", min),
        ("genDomesticLegAggMaxBn", max),
    ):
        emit(
            name,
            lambda select=select: select(
                float(r["aggregate_cost_bn"]) for r in cload(DOMESTIC_LEG)
            ),
            "{:.2f}",
            DOMESTIC_LEG,
        )
    for name, select in (
        ("genDomesticLegFuelShareMinPct", min),
        ("genDomesticLegFuelShareMaxPct", max),
    ):
        emit(
            name,
            lambda select=select: (
                100
                * select(
                    float(r["motor_fuel_share_of_loss"]) for r in cload(DOMESTIC_LEG)
                )
            ),
            "{:.1f}",
            DOMESTIC_LEG,
        )
    emit(
        "genDomesticLegSpreadPct",
        lambda: (
            100
            * (
                max(float(r["aggregate_cost_bn"]) for r in cload(DOMESTIC_LEG))
                - min(float(r["aggregate_cost_bn"]) for r in cload(DOMESTIC_LEG))
            )
            / float(jload(central)["aggregate_cost_bn"])
        ),
        "{:.0f}",
        DOMESTIC_LEG,
    )

    # ------------------------------------------------------------------
    # petrol versus diesel by decile (D26): the decile-eight cash spike
    # ------------------------------------------------------------------
    emit(
        "genPetrolUpliftPct",
        lambda: 100 * (float(fuel_decile(1)["petrol_price_factor"]) - 1),
        "{:.1f}",
        FUEL_BY_DECILE,
    )
    emit(
        "genDieselUpliftPct",
        lambda: 100 * (float(fuel_decile(1)["diesel_price_factor"]) - 1),
        "{:.1f}",
        FUEL_BY_DECILE,
    )
    for tag, d in (("One", 1), ("Eight", 8), ("Ten", 10)):
        emit(
            f"genDieselShareSpendDecile{tag}Pct",
            lambda d=d: 100 * float(fuel_decile(d)["diesel_share_of_fuel_spend"]),
            "{:.1f}",
            FUEL_BY_DECILE,
        )
        emit(
            f"genDieselShareLossDecile{tag}Pct",
            lambda d=d: 100 * float(fuel_decile(d)["diesel_share_of_fuel_loss"]),
            "{:.1f}",
            FUEL_BY_DECILE,
        )
        emit(
            f"genFuelLossDecile{tag}Gbp",
            lambda d=d: float(fuel_decile(d)["motor_fuel_loss_gbp"]),
            "{:.0f}",
            FUEL_BY_DECILE,
        )
        emit(
            f"genFuelSpendShareDecile{tag}Pct",
            lambda d=d: 100 * float(fuel_decile(d)["share_with_any_fuel_spend"]),
            "{:.1f}",
            FUEL_BY_DECILE,
        )

    # ------------------------------------------------------------------
    # means-tested audit (C13): the resolved variable set, logged not swallowed
    # ------------------------------------------------------------------
    aud = "means_tested_audit.json"
    emit(
        "genMeansTestedVarCount",
        lambda: len(audit()["resolved_variables"]),
        "{:.0f}",
        aud,
    )
    emit(
        "genMeansTestedMissingCount",
        lambda: len(audit()["missing_required"]),
        "{:.0f}",
        aud,
    )
    emit(
        "genUniversalCreditHouseholdsM",
        lambda: audit()["by_variable"]["universal_credit"]["households_m"],
        "{:.2f}",
        aud,
    )
    emit(
        "genPensionCreditHouseholdsM",
        lambda: audit()["by_variable"]["pension_credit"]["households_m"],
        "{:.2f}",
        aud,
    )
    emit(
        "genTotalHouseholdsM",
        lambda: audit()["total_households_m"],
        "{:.1f}",
        aud,
    )

    # ------------------------------------------------------------------
    # macros the prose asks for that no results file supports
    #
    # These names appear in the manuscript but describe *combinations* of
    # specifications that were never run: there is no symmetric-damping run on
    # the ONS both-levels calibration, and no symmetric-damping run on the
    # steady-state basis. They are emitted as \GENMISSING rather than guessed at,
    # so the build fails visibly and the sentence gets repointed at a
    # specification that exists (\genSymDampMotorFuelShareOfLoss,
    # \genOnsLevelsMotorFuelShareOfLoss, \genSteadyStateMotorFuelShareOfLoss).
    # ------------------------------------------------------------------
    def _no_such_run(name: str):
        def fail():
            raise KeyError(
                "no such specification: this macro asks for a combination of two "
                "specifications that results/ does not contain"
            )

        emit(name, fail, source="results/robustness/comparison.csv")

    # The two combination specifications: symmetric damping crossed with the
    # other accounting choices. These are the calibrations under which the
    # motor-fuel majority fails, so the abstract cites them and they are real
    # runs (analysis/run_combinations.py), not arithmetic.
    def _combo(key: str) -> float:
        cell = json.loads((R / "robustness" / "combinations.json").read_text())
        return float(cell[key]["motor_fuel_share_pct"])

    emit(
        "genSymDampSteadyMotorFuelShare",
        lambda: _combo("symmetric_steady_state"),
        "{:.1f}",
        "robustness/combinations.json",
    )
    emit(
        "genSymDampOnsLevelsMotorFuelShare",
        lambda: _combo("symmetric_ons_levels"),
        "{:.1f}",
        "robustness/combinations.json",
    )
    for _name in ("genSymDampSteadyAggBn", "genSymDampOnsLevelsAggBn"):
        _k = "symmetric_steady_state" if "Steady" in _name else "symmetric_ons_levels"
        emit(
            _name,
            lambda k=_k: json.loads(
                (R / "robustness" / "combinations.json").read_text()
            )[k]["aggregate_cost_bn"],
            "{:.2f}",
            "robustness/combinations.json",
        )

    # ------------------------------------------------------------------
    # remaining carried constant
    # ------------------------------------------------------------------
    for name, (value, fmt) in _CARRIED.items():
        emit(name, value, fmt, "carried constant — see _CARRIED")

    # ------------------------------------------------------------------
    # --- scenario grid (results/grid/grid.csv) ---------------------------
    if stale:
        note(STALE_SWEEP_WARNING)

    # The grid shows the paper's gradient is invariant to the gas/oil mix, and
    # locates the frontier at which motor fuel stops dominating the loss.
    def _grid() -> list[dict]:
        with open(R / "grid" / "grid.csv", newline="") as fh:
            return list(csv.DictReader(fh))

    emit(
        "genGridRatioMin",
        lambda: min(
            float(r["d1_d10_ratio"])
            for r in _grid()
            if r["d1_d10_ratio"] and float(r["d1_d10_ratio"]) > 0
        ),
        "{:.2f}",
        "grid/grid.csv",
    )
    emit(
        "genGridRatioMax",
        lambda: max(float(r["d1_d10_ratio"]) for r in _grid() if r["d1_d10_ratio"]),
        "{:.2f}",
        "grid/grid.csv",
    )

    def _frontier() -> float:
        """Mean gas/oil ratio at which motor fuel's share crosses 50 per cent.

        Interpolated within each oil row, then averaged over rows that cross.
        """
        by_oil: dict[float, list[dict]] = {}
        for r in _grid():
            oil = float(r["oil_pct"])
            if oil > 0:
                by_oil.setdefault(oil, []).append(r)
        slopes = []
        for oil, rows in by_oil.items():
            rows = sorted(rows, key=lambda r: float(r["gas_pct"]))
            rows = [r for r in rows if r["motor_fuel_share_pct"]]
            share = [float(r["motor_fuel_share_pct"]) for r in rows]
            gas = [float(r["gas_pct"]) for r in rows]
            for i in range(len(share) - 1):
                if share[i] >= 50 >= share[i + 1]:
                    t = (share[i] - 50) / (share[i] - share[i + 1])
                    slopes.append((gas[i] + t * (gas[i + 1] - gas[i])) / oil)
                    break
        if not slopes:
            raise ValueError("no 50 per cent crossing found in grid")
        return sum(slopes) / len(slopes)

    emit("genGridFrontierSlope", _frontier, "{:.2f}", "grid/grid.csv")

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
    n_macros = sum(1 for line in lines if line.startswith("\\newcommand"))
    resolved = n_macros - len(missing)
    print(f"wrote {OUT} ({n_macros} macros, {resolved} resolved)")
    if stale:
        print(
            "WARNING: results/sensitivity/*.csv and results/grid/grid.csv were "
            "generated on the peak-fuel specification; the macros derived from "
            "them do not describe the damped main spec. Re-run "
            "analysis/run_sensitivity.py and analysis/run_grid.py."
        )
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

#!/usr/bin/env python3
"""Run every realised-2026 specification the referee round requires, off one load.

Usage::

    python analysis/run_variants.py
    python analysis/run_variants.py --period 2026

``docs/FIXES.md`` records two decisions that change the headline, and several
robustness runs that have to be reported next to it. All of them are produced
here, from a **single** load of the microdata (loading it is the slow step):

Specifications
--------------
``main``
    Equivalised AHC denominator (decision D1) and the Step 1 consumption-weighted
    quarterly cap average on the domestic leg (decision D2). This is the headline.
    Overwrites ``results/realised_2026/``.
``steady_state``
    Identical but charging full steady-state pass-through for twelve months —
    what the code did before D2. Kept as a labelled alternative, never the
    headline.
``symmetric_damping``
    A single common peak-to-annual-average damping fraction on **both** the gas
    and the pump legs (``scenarios.SYMMETRIC_SUSTAINED_FRACTION``). The paper's
    motor-fuel majority depends only on the ratio of the two fractions, so this
    is a first-class specification and may overturn the headline claim (A3).
``peak_fuel``
    The undamped-pump upper bound, retained from the earlier audit.
``ons_shape``
    Motor fuel put on the ONS Family Spending decile *shape*, national total
    preserved (the pre-existing robustness run).
``ons_both_levels``
    Both imputation *levels* corrected against ONS — domestic energy up from
    £1,330 to £1,780, motor fuel down to £960 (C11). The two imputation errors
    run in opposite directions and compound on the central claim, so this is the
    test that can settle it.
``unequivalised``
    The main specification with the pre-D1 unequivalised denominator, so the
    effect of the denominator change is visible rather than silent (A1).

Also written
------------
* ``results/robustness/comparison.csv`` — headline statistics for every
  specification side by side.
* ``results/robustness/policy_comparison.csv`` — the five instruments scored
  under each.
* ``results/sensitivity/domestic_leg.csv`` — the A4 sweep of the four parameters
  that scale the domestic channel one-for-one, including the
  ``sustained_fraction`` x ``phase_in[0]`` split that the Cornwall anchor pins
  only as a product.
* ``results/persisted_values.json`` — values the emitter previously hardcoded
  (C12).

Needs the PolicyEngine UK microdata, which is private: set ``HUGGING_FACE_TOKEN``
(the repo reads a gitignored ``.env``). Handling mirrors
``analysis/run_incidence.py``, which remains the script that produces the NIESR
scenarios.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.run_incidence import RESULTS, _asdict, _load_env, dataset_path
from uk_iran_conflict import incidence as inc
from uk_iran_conflict import policies as pol
from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import (
    Baseline,
    ScenarioResult,
    apply_calibration,
    decile_concept_audit,
    load_baseline,
    motor_fuel_margins,
    run_scenario,
    wmean,
    wquantile,
    wsum,
)

#: Index of the cap quarter the Ofgem October-2026 anchor prices.
ANCHOR_INDEX = scen.CAP_QUARTER_LABELS.index(scen.CAP_ANCHOR_QUARTER)

#: The product the October cap identifies **given the counterfactual**:
#: ``sustained_fraction`` x the phase-in weight in the anchor quarter, 0.714.
#:
#: Round-3 changed what this is. The counterfactual cap is no longer a shocked
#: observation, so the two published caps jointly identify the counterfactual
#: *and* the sustained fraction (``scenarios.solve_cap_calibration``); holding
#: this product fixed and sweeping its split is now the *conditional*
#: identification question — how much of the cap move is a large sustained
#: fraction arriving slowly versus a small one arriving fast — with the
#: counterfactual held at the solved value.
CAP_ANCHOR_PRODUCT = (
    scen.REALISED_SUSTAINED_FRACTION * scen.CAP_PHASE_IN_PROFILE[ANCHOR_INDEX]
)


def _relinear_ramp() -> scen.Scenario:
    """``realised_2026`` on the linear-ramp phase-in, re-solved end to end.

    The point of this variant is to *separate* the two things round-3 changed.
    The baseline fix (a constructed pre-war counterfactual instead of a shocked
    observation) and the phase-in fix (observation windows instead of a ramp in
    calendar time) both move the domestic leg, and a reader is entitled to know
    which did the work. Here the ramp comes back — with its partial-quarter bug
    fixed — and the counterfactual cap and sustained fraction are re-solved on
    *its* profile, so the only difference from the main specification is the
    phase-in construction.

    The ramp puts 2026Q3 and 2026Q4 both at 1.0, which is exactly the degeneracy
    that made the old specification incoherent: two identical windows cannot
    separate two different observed caps.
    :func:`~uk_iran_conflict.scenarios.solve_cap_calibration` raises on it
    rather than returning a number, so this variant is reported as
    **not identified** if the ramp is used unmodified. That refusal is the
    finding, not a failure of the run.
    """
    central = scen.get_scenario("realised_2026")
    profile = scen.LINEAR_RAMP_PHASE_IN_PROFILE
    calibration = scen.solve_cap_calibration(
        central.gas.pct_change,
        profile[scen.CAP_QUARTER_LABELS.index(scen.CAP_BASE_QUARTER)],
        profile[ANCHOR_INDEX],
    )
    return dataclasses.replace(
        central,
        key="realised_2026_linear_ramp",
        baseline_cap_gbp=calibration.prewar_cap_gbp,
        pass_through=dataclasses.replace(
            central.pass_through,
            phase_in_profile=profile,
            sustained_fraction=calibration.sustained_fraction,
        ),
    )


def derived_scenarios() -> dict[str, scen.Scenario]:
    """Scenarios built from the registry rather than registered in it.

    Kept here rather than in ``uk_iran_conflict.scenarios`` because each is a
    *presentation* of a registered scenario — a different annualisation window
    or a different phase-in construction — not a different state of the world.
    """
    out: dict[str, scen.Scenario] = {}
    for key in ("realised_2026", "realised_2026_peak_fuel"):
        name = f"{key}_calendar_2026"
        out[name] = dataclasses.replace(
            scen.on_window(scen.get_scenario(key), scen.CALENDAR_2026_MONTHS),
            key=name,
        )
    try:
        out["realised_2026_linear_ramp"] = _relinear_ramp()
    except ValueError as exc:  # the ramp cannot identify the counterfactual
        print(f"linear_ramp variant is NOT IDENTIFIED and is skipped: {exc}")
    return out


def scenario_for(key: str, derived: dict[str, scen.Scenario]) -> scen.Scenario | None:
    """Registered scenario, derived scenario, or ``None`` if unavailable."""
    if key in scen.SCENARIOS:
        return scen.SCENARIOS[key]
    return derived.get(key)


@dataclass(frozen=True)
class Variant:
    """One specification to score."""

    name: str
    scenario_key: str
    out_dir: Path
    label: str
    calibration: str = "raw"
    income_basis: str = "equivalised_ahc"
    domestic_basis: str = "annual"


def variants() -> tuple[Variant, ...]:
    return (
        Variant(
            name="main",
            scenario_key="realised_2026",
            out_dir=RESULTS / "realised_2026",
            label=(
                "Main: equivalised AHC denominator (D1) and the Step 1 "
                "consumption-weighted quarterly cap average (D2)"
            ),
        ),
        Variant(
            name="steady_state",
            scenario_key="realised_2026",
            out_dir=RESULTS / "robustness" / "steady_state",
            domestic_basis="steady_state",
            label=(
                "Alternative: full steady-state pass-through charged for twelve "
                "months, i.e. the phase-in not applied"
            ),
        ),
        Variant(
            name="symmetric_damping",
            scenario_key="realised_2026_symmetric",
            out_dir=RESULTS / "robustness" / "symmetric_damping",
            label=(
                "Symmetric damping: one common peak-to-annual fraction "
                f"({scen.SYMMETRIC_SUSTAINED_FRACTION:.3f}) on both the gas and "
                "the pump legs"
            ),
        ),
        Variant(
            name="peak_fuel",
            scenario_key="realised_2026_peak_fuel",
            out_dir=RESULTS / "realised_2026_peak_fuel",
            label="Upper bound: peak pump prices applied undamped for a full year",
        ),
        Variant(
            name="ons_shape",
            scenario_key="realised_2026",
            out_dir=RESULTS / "robustness" / "ons_fuel",
            calibration="ons_fuel_shape",
            label=(
                "Robustness: ONS motor-fuel decile shape, national fuel total preserved"
            ),
        ),
        Variant(
            name="ons_both_levels",
            scenario_key="realised_2026",
            out_dir=RESULTS / "robustness" / "ons_levels",
            calibration="ons_both_levels",
            label=(
                "Robustness: both imputation levels corrected against ONS "
                "(domestic energy to £1,780, motor fuel to £960)"
            ),
        ),
        Variant(
            name="mt_fuel_parity",
            scenario_key="realised_2026",
            out_dir=RESULTS / "robustness" / "mt_fuel_parity",
            calibration="mt_fuel_parity",
            label=(
                "Round-3: means-tested motor-fuel spend equalised to "
                "non-means-tested within decile (decile totals preserved). The "
                "specification that tests the paper's central policy finding: "
                "rescale_motor_fuel_to_ons applies one factor per decile and so "
                "cannot move the 6.2x means-tested fuel margin at all."
            ),
        ),
        Variant(
            name="nts_participation",
            scenario_key="realised_2026",
            out_dir=RESULTS / "robustness" / "nts_participation",
            calibration="nts_participation",
            label=(
                "Round-3: DfT car-availability gradient imposed on the fuel "
                "participation rate (England-only comparator; decile totals "
                "preserved). Tests the margin the imputation cannot produce - "
                "a zero-fuel share identical in deciles one and ten."
            ),
        ),
        Variant(
            name="linear_ramp",
            scenario_key="realised_2026_linear_ramp",
            out_dir=RESULTS / "robustness" / "linear_ramp",
            label=(
                "Alternative cap phase-in: the pre-round-3 linear ramp in "
                "calendar time (with the partial-quarter fix), instead of the "
                "observation-window construction. The counterfactual cap is "
                "re-solved on the ramp's own profile, so this isolates the "
                "phase-in construction from the baseline fix."
            ),
        ),
        Variant(
            name="calendar_2026",
            scenario_key="realised_2026_calendar_2026",
            out_dir=RESULTS / "robustness" / "calendar_2026",
            label=(
                "Main specification re-annualised onto calendar 2026, the "
                "Resolution Foundation's £11bn window."
            ),
        ),
        Variant(
            name="peak_fuel_calendar_2026",
            scenario_key="realised_2026_peak_fuel_calendar_2026",
            out_dir=RESULTS / "robustness" / "peak_fuel_calendar_2026",
            label=(
                "Round-3 finding 10: the peak-fuel UPPER BOUND on calendar 2026, "
                "so the Resolution Foundation bracketing is like-for-like. "
                "Previously only the central case was re-annualised onto the RF "
                "window, so the published bracket mixed two different windows."
            ),
        ),
        Variant(
            name="unequivalised",
            scenario_key="realised_2026",
            out_dir=RESULTS / "robustness" / "unequivalised",
            income_basis="unequivalised",
            label=(
                "Robustness: main specification on the pre-D1 unequivalised "
                "net-income denominator"
            ),
        ),
    )


def comparison_row(
    variant: Variant,
    result: ScenarioResult,
    base: Baseline,
    scenario: scen.Scenario | None = None,
) -> dict:
    """The statistics every specification must be able to show side by side."""
    d1 = result.decile[0]
    d10 = result.decile[-1]
    coverage = result.coverage
    return {
        "variant": variant.name,
        "scenario": variant.scenario_key,
        "calibration": variant.calibration,
        "income_basis": variant.income_basis,
        "domestic_basis": variant.domestic_basis,
        "label": variant.label,
        "aggregate_cost_bn": result.aggregate_cost_bn,
        "mean_loss_gbp": result.mean_loss_gbp,
        "mean_loss_pct": result.mean_loss_pct,
        "mean_loss_pct_equivalised_both": result.mean_loss_pct_equivalised_both,
        "mean_income_gbp": result.mean_income_gbp,
        "median_income_gbp": result.median_income_gbp,
        "decile1_loss_gbp": d1.mean_loss_gbp,
        "decile1_loss_pct": d1.mean_loss_pct,
        "decile10_loss_gbp": d10.mean_loss_gbp,
        "decile10_loss_pct": d10.mean_loss_pct,
        "d1_d10_ratio_pct": d1.mean_loss_pct / d10.mean_loss_pct,
        "d1_d10_ratio_gbp": d1.mean_loss_gbp / d10.mean_loss_gbp,
        "gas_share_of_loss": result.gas_share_of_loss,
        "electricity_share_of_loss": result.electricity_share_of_loss,
        "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
        "annual_phase_in_gas": result.annual_phase_in_gas,
        "annual_phase_in_electricity": result.annual_phase_in_electricity,
        # Round-2 disclosures. Table 1 is not like-for-like without the two
        # damping fractions: realised_2026 damps the pump leg while both NIESR
        # scenarios use 1.0 (defensible — their paths are sustained levels, not
        # peaks — but it has to be in the results before the prose can say so).
        "modelled_window": result.modelled_window,
        "gas_sustained_fraction": result.gas_sustained_fraction,
        "pump_sustained_fraction": result.pump_sustained_fraction,
        "cap_lag_quarters": result.cap_lag_quarters,
        "legs_damped_symmetrically": (
            result.gas_sustained_fraction == result.pump_sustained_fraction
        ),
        # The domestic-energy-only gradient: VALIDATION.md's most informative
        # available robustness anchor, and the one channel whose imputation is
        # not contradicted by DfT car-access data.
        "domestic_only_d1_d10_ratio_pct": result.domestic_only_d1_d10_ratio_pct,
        "domestic_only_d1_d10_ratio_gbp": result.domestic_only_d1_d10_ratio_gbp,
        "domestic_only_decile1_loss_pct": (
            result.decile_domestic_only[0].mean_loss_pct
            if result.decile_domestic_only
            else float("nan")
        ),
        "domestic_only_decile10_loss_pct": (
            result.decile_domestic_only[-1].mean_loss_pct
            if result.decile_domestic_only
            else float("nan")
        ),
        # Within- vs between-decile dispersion, with and without decile one.
        "mean_within_decile_range_pp": (
            result.dispersion.mean_within_decile_range_pp
            if result.dispersion
            else float("nan")
        ),
        "mean_within_decile_range_excl_d1_pp": (
            result.dispersion.mean_within_decile_range_excl_d1_pp
            if result.dispersion
            else float("nan")
        ),
        "median_within_decile_range_pp": (
            result.dispersion.median_within_decile_range_pp
            if result.dispersion
            else float("nan")
        ),
        "between_decile_range_pp": (
            result.dispersion.between_decile_range_pp
            if result.dispersion
            else float("nan")
        ),
        "within_exceeds_between": (
            result.dispersion.within_exceeds_between if result.dispersion else None
        ),
        "within_exceeds_between_excl_d1": (
            result.dispersion.within_exceeds_between_excl_d1
            if result.dispersion
            else None
        ),
        "decile_ranking_concept": (
            result.decile_concept.best_match if result.decile_concept else ""
        ),
        "decile_ranking_documented_truth": (
            result.decile_concept.documented_truth if result.decile_concept else ""
        ),
        "decile_ranking_person_weighted": (
            result.decile_concept.person_weighted if result.decile_concept else None
        ),
        # Round-3 finding 9: the gradient from decile two, and with the
        # non-positive tail winsorised back in rather than dropped.
        "d2_d10_ratio_pct": result.d2_d10_ratio_pct,
        "d2_d10_ratio_gbp": result.d2_d10_ratio_gbp,
        "d1_d10_ratio_pct_winsorised": result.d1_d10_ratio_pct_winsorised,
        "d2_d10_ratio_pct_winsorised": result.d2_d10_ratio_pct_winsorised,
        "decile1_loss_pct_winsorised": result.decile1_loss_pct_winsorised,
        "mean_loss_pct_winsorised": result.mean_loss_pct_winsorised,
        # Round-3 finding 2: the margin that carries the central policy finding.
        "means_tested_share_of_loss": (
            result.motor_fuel_margins.means_tested_share_of_loss
            if result.motor_fuel_margins
            else float("nan")
        ),
        "means_tested_fuel_ratio": (
            result.motor_fuel_margins.means_tested_fuel_ratio
            if result.motor_fuel_margins
            else float("nan")
        ),
        "zero_fuel_share_overall": (
            result.motor_fuel_margins.zero_fuel_share_overall
            if result.motor_fuel_margins
            else float("nan")
        ),
        "zero_fuel_share_d1_minus_d10_pp": (
            result.motor_fuel_margins.zero_share_d1_minus_d10_pp
            if result.motor_fuel_margins
            else float("nan")
        ),
        # The cap calibration this specification actually ran on.
        "prewar_counterfactual_cap_gbp": float(
            getattr(scenario, "baseline_cap_gbp", float("nan"))
        ),
        "decile_ranking_matches_denominator": (
            result.decile_concept.matches_burden_denominator
            if result.decile_concept
            else None
        ),
        "excluded_households_m": coverage.households_m if coverage else float("nan"),
        "excluded_share_of_loss": coverage.share_of_loss if coverage else float("nan"),
        "aggregate_energy_spend_bn": wsum(base.energy, base.weight) / 1e9,
        "aggregate_fuel_spend_bn": wsum(base.motor_fuel, base.weight) / 1e9,
        "mean_energy_spend_gbp": wmean(base.energy, base.weight),
        "mean_fuel_spend_gbp": wmean(base.motor_fuel, base.weight),
    }


# --------------------------------------------------------------------------
# A4: the parameters that scale the domestic leg one-for-one
# --------------------------------------------------------------------------


def _with_pass_through(scenario: scen.Scenario, **changes) -> scen.Scenario:
    return dataclasses.replace(
        scenario,
        pass_through=dataclasses.replace(scenario.pass_through, **changes),
    )


def domestic_leg_sweep(base: Baseline) -> list[dict]:
    """Sweep everything that scales the domestic channel proportionally (A4).

    Four parameters, each currently unswept and each entering the domestic leg
    linearly:

    ``sustained_fraction`` x the anchor quarter's phase-in weight
        Ofgem's confirmed October-2026 cap identifies only the **product** of
        these two (0.199 x 1.00 = 0.199), and it constrains only the *anchor*
        quarter. The sweep holds that product — so every row still reproduces
        the anchor exactly — and for each candidate ``sustained_fraction`` sets
        the anchor quarter's phase-in weight to whatever the anchor requires,
        leaving the other quarters at the lag-derived profile. A lower sustained
        fraction therefore means a *steeper* climb to full pass-through and a
        larger window-average domestic factor. Every one of these specifications
        is observationally equivalent at the anchor and none is equivalent for
        the headline, which is exactly the identification problem.
    ``PREWAR_NBP_PENCE_PER_THERM``
        The realised gas percentage change is +78p on this base, so the domestic
        leg scales inversely with it.
    ``WHOLESALE_SHARE_GAS_BILL`` and ``WHOLESALE_SHARE_ELECTRICITY_BILL``
        Wholesale cost shares of each bill; the retail shock is linear in them.
    """
    central = scen.get_scenario("realised_2026")
    rows: list[dict] = []

    def record(parameter: str, value: object, scenario: scen.Scenario) -> None:
        result, _ = run_scenario(base, scenario)
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "prewar_counterfactual_cap_gbp": scenario.baseline_cap_gbp,
                "sustained_fraction": scenario.pass_through.sustained_fraction,
                "phase_in_at_anchor_quarter": (
                    scenario.pass_through.phase_in_profile[ANCHOR_INDEX]
                ),
                "anchor_product": (
                    scenario.pass_through.sustained_fraction
                    * scenario.pass_through.phase_in_profile[ANCHOR_INDEX]
                ),
                "annual_phase_in_gas": scenario.pass_through.annual_phase_in_gas,
                "gas_pct_change": scenario.annual_retail_shock.gas_pct_change,
                "aggregate_cost_bn": result.aggregate_cost_bn,
                "mean_loss_gbp": result.mean_loss_gbp,
                "mean_loss_pct": result.mean_loss_pct,
                "decile1_loss_pct": result.decile[0].mean_loss_pct,
                "decile10_loss_pct": result.decile[-1].mean_loss_pct,
                "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
                "domestic_share_of_loss": (
                    result.gas_share_of_loss + result.electricity_share_of_loss
                ),
            }
        )

    # 1. The split of the anchored product. The grid starts at the product
    # itself: below it the anchor quarter would need a phase-in weight above
    # 1.0, which is not a phase-in weight.
    for fraction in (0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
        anchor_weight = CAP_ANCHOR_PRODUCT / fraction
        if anchor_weight > 1.0:
            continue  # a phase-in weight above 1.0 is not a phase-in weight
        profile = list(scen.CAP_PHASE_IN_PROFILE)
        profile[ANCHOR_INDEX] = anchor_weight
        record(
            "sustained_fraction_split",
            fraction,
            _with_pass_through(
                central,
                sustained_fraction=fraction,
                phase_in_profile=tuple(profile),
            ),
        )

    # 2. The pre-war NBP reference the gas percentage is measured against.
    for prewar in (70.0, 80.0, 90.0, 100.0, 110.0, 130.0):
        record(
            "prewar_nbp_pence_per_therm",
            prewar,
            dataclasses.replace(
                central,
                gas=dataclasses.replace(central.gas, prewar_pence_per_therm=prewar),
            ),
        )

    # 3-4. The two wholesale cost shares.
    for share in (0.35, 0.40, 0.45, 0.50, 0.55):
        record(
            "wholesale_share_gas_bill",
            share,
            _with_pass_through(central, wholesale_share_gas_bill=share),
        )
    for share in (0.25, 0.30, 0.35, 0.40, 0.45):
        record(
            "wholesale_share_electricity_bill",
            share,
            _with_pass_through(central, wholesale_share_electricity_bill=share),
        )

    # 5. The monthly wholesale gas path, which is what identifies the pre-war
    # counterfactual cap. See scenarios.solve_cap_calibration on why this is the
    # sweep that matters most after the baseline fix itself.
    for label, kwargs in [
        (f"shift{k:+d}m", {"shift_months": k}) for k in (-2, -1, 0, 1, 2)
    ] + [(f"flatten{f:g}", {"flatten": f}) for f in (0.6, 0.8, 1.0, 1.2, 1.4)]:
        profile = scen.gas_profile_variant(**kwargs)
        phase_in = scen.cap_phase_in_profile(
            central.pass_through.lag_quarters, central.quarter_labels, profile=profile
        )
        try:
            calibration = scen.solve_cap_calibration(
                central.gas.pct_change,
                phase_in[scen.CAP_QUARTER_LABELS.index(scen.CAP_BASE_QUARTER)],
                phase_in[ANCHOR_INDEX],
            )
        except ValueError as exc:
            rows.append(
                {
                    "parameter": "gas_peak_monthly_profile",
                    "value": label,
                    "identified": False,
                    "note": str(exc),
                }
            )
            continue
        variant = dataclasses.replace(
            central,
            baseline_cap_gbp=calibration.prewar_cap_gbp,
            pass_through=dataclasses.replace(
                central.pass_through,
                phase_in_profile=phase_in,
                sustained_fraction=calibration.sustained_fraction,
            ),
        )
        before = len(rows)
        record("gas_peak_monthly_profile", label, variant)
        rows[before].update(
            {
                "identified": True,
                "prewar_counterfactual_cap_gbp": calibration.prewar_cap_gbp,
                "phase_in_profile": ";".join(f"{v:.4f}" for v in phase_in),
            }
        )
    return rows


# --------------------------------------------------------------------------
# C12: values the emitter previously hardcoded
# --------------------------------------------------------------------------


def _decile_mean(b: Baseline, values, d: int) -> float:
    sel = b.decile == d
    return wmean(values[sel], b.weight[sel])


def persisted_values(base: Baseline) -> dict:
    """Every prose literal the emitter carried with no results backing (C12)."""
    ons_shape = apply_calibration(base, "ons_fuel_shape")
    ons_levels = apply_calibration(base, "ons_both_levels")
    out: dict = {
        "note": (
            "Values the LaTeX emitter previously hardcoded. Persisted here so "
            "the appendix's guarantee that every number is emitted mechanically "
            "holds (docs/FIXES.md C12)."
        ),
        "motor_fuel_decile_mean_gbp": {},
    }
    for label, b in (
        ("raw", base),
        ("ons_shape", ons_shape),
        ("ons_both_levels", ons_levels),
    ):
        out["motor_fuel_decile_mean_gbp"][label] = [
            _decile_mean(b, b.motor_fuel, d) for d in range(1, 11)
        ]
    sel = base.decile == 1
    w = base.weight[sel]
    for label, income in (
        ("equivalised_ahc", base.equiv_income_ahc),
        ("unequivalised", base.net_income),
    ):
        inc = income[sel]
        out[f"decile1_median_income_gbp_{label}"] = wquantile(inc, w, 0.5)
        out[f"decile1_mean_income_gbp_{label}"] = wmean(inc, w)
        out[f"decile1_zero_or_negative_income_share_{label}"] = float(
            w[inc <= 0].sum() / w.sum()
        )
    out["mean_domestic_energy_spend_gbp"] = wmean(base.energy, base.weight)
    out["mean_motor_fuel_spend_gbp"] = wmean(base.motor_fuel, base.weight)
    return out


#: Calibrations the full decile cash profile is reported under (round-2
#: finding 4).
CASH_PROFILE_CALIBRATIONS: tuple[tuple[str, str], ...] = (
    ("raw", "Uncalibrated PolicyEngine UK imputation"),
    ("ons_fuel_shape", "ONS motor-fuel decile shape, national total preserved"),
    ("ons_both_levels", "Both imputation levels corrected against ONS"),
)


def cash_profiles(base: Baseline, scenario) -> dict:
    """Persist the FULL decile cash profile under every calibration.

    Round-2 finding: the paper says the cash profile's non-monotonicity "is
    common to every specification". Two of three referees checked, and it is
    not: under **both** ONS calibrations the profile is strictly monotone and
    peaks at decile ten. Only the uncalibrated imputation — the one whose
    decile-one motor-fuel spend is £1,073 against ONS's £318, and which gives
    deciles one and ten an identical fuel-purchasing rate against DfT's 60% and
    86% car access — produces the kink the claim rests on.

    Two decile-eight-shaped things were being confused, so the profile is
    persisted in full, per calibration, with the monotonicity and the peak
    decile computed here rather than eyeballed: ``is_monotone_increasing``,
    ``peak_decile`` and the list of deciles at which the profile falls. The
    prose has to be written against this file.
    """
    out: dict = {
        "note": (
            "Full decile cash-loss profile under each calibration. The paper's "
            "claim that the non-monotone cash profile is common to every "
            "specification is false under both ONS calibrations, which are "
            "strictly monotone and peak at decile ten (round-2 referees)."
        ),
        "scenario": getattr(scenario, "key", ""),
        "profiles": {},
    }
    for calibration, label in CASH_PROFILE_CALIBRATIONS:
        result, _ = run_scenario(base, scenario, calibration=calibration)
        cash = [row.mean_loss_gbp for row in result.decile]
        pct = [row.mean_loss_pct for row in result.decile]
        falls = [
            row.decile
            for row, prev in zip(result.decile[1:], result.decile[:-1], strict=True)
            if row.mean_loss_gbp < prev.mean_loss_gbp
        ]
        out["profiles"][calibration] = {
            "label": label,
            "decile": [row.decile for row in result.decile],
            "mean_loss_gbp": cash,
            "mean_loss_pct": pct,
            "is_monotone_increasing": not falls,
            "deciles_where_cash_falls": falls,
            "peak_decile": (
                result.decile[int(max(range(len(cash)), key=cash.__getitem__))].decile
                if cash
                else None
            ),
            "decile1_over_decile10_gbp": (
                cash[0] / cash[-1] if cash and cash[-1] else float("nan")
            ),
        }
    return out


#: Fields that must agree between the ``motor_fuel_margins`` and
#: ``means_tested_fuel`` blocks of ``results/round3_findings.json``.
#:
#: **Round-4 finding 4.** The published file carried bit-identical values —
#: ``means_tested_mean_fuel_gbp = 223.6003875732422`` for all five calibrations —
#: in ``motor_fuel_margins``, while ``means_tested_fuel`` in the SAME file gave
#: 223.60 / 176.26 / 139.65 / 1001.20. The margins sub-block had been written by
#: an earlier version that computed the margins off the uncalibrated baseline
#: and was never regenerated. Nothing in the pipeline or the test suite could
#: see it, because the tests run on synthetic fixtures and never look at
#: ``results/``. These two blocks are now cross-checked at write time and by
#: ``tests/test_artefacts.py`` against the file on disk.
CROSS_BLOCK_FIELDS: tuple[str, ...] = ("means_tested_mean_fuel_gbp",)

#: Relative tolerance for that cross-check. The two blocks are computed by
#: different call paths off the same calibrated baseline, so they should agree
#: to floating point; this leaves room for nothing but rounding.
CROSS_BLOCK_RTOL: float = 1e-9


def check_fuel_blocks_agree(findings: dict, rtol: float = CROSS_BLOCK_RTOL) -> dict:
    """Cross-check ``motor_fuel_margins`` against ``means_tested_fuel``.

    Both blocks report the means-tested mean motor-fuel spend under each
    calibration, by different routes: ``motor_fuel_margins`` calls
    :func:`~uk_iran_conflict.incidence.motor_fuel_margins` on the calibrated
    baseline directly, ``means_tested_fuel`` reads it off the
    :class:`~uk_iran_conflict.incidence.ScenarioResult`. If they disagree, one
    of them is stale or one of the two call paths is not applying the
    calibration — which is exactly what happened (round-4 finding 4).

    Raises rather than returning a flag on divergence: a results tree that
    contradicts itself must not be written.
    """
    margins = findings.get("motor_fuel_margins", {})
    headline = findings.get("means_tested_fuel", {})
    checked: dict[str, dict] = {}
    bad: list[str] = []
    for calibration in inc.CALIBRATIONS:
        left = margins.get(calibration, {})
        right = headline.get(calibration, {})
        for field_name in CROSS_BLOCK_FIELDS:
            a = left.get(field_name)
            b = right.get(field_name)
            agree = (
                a is not None
                and b is not None
                and abs(float(a) - float(b)) <= rtol * max(abs(float(b)), 1.0)
            )
            checked[f"{calibration}.{field_name}"] = {
                "motor_fuel_margins": a,
                "means_tested_fuel": b,
                "agree": bool(agree),
            }
            if not agree:
                bad.append(f"{calibration}.{field_name}: {a!r} vs {b!r}")
    # A block whose values are identical across every calibration is the stale
    # signature itself: the calibrations exist precisely to move this number.
    values = [
        margins.get(c, {}).get("means_tested_mean_fuel_gbp") for c in inc.CALIBRATIONS
    ]
    present = [v for v in values if v is not None]
    all_identical = len(present) > 1 and len(set(present)) == 1
    if all_identical:
        bad.append(
            "motor_fuel_margins.means_tested_mean_fuel_gbp is identical across "
            f"every calibration ({present[0]!r}); the calibrations exist to "
            "move it, so the block was not regenerated"
        )
    if bad:
        raise SystemExit(
            "round3_findings blocks disagree — the results tree is stale or the "
            "calibration is not reaching both call paths:\n  " + "\n  ".join(bad)
        )
    return {
        "fields": list(CROSS_BLOCK_FIELDS),
        "rtol": rtol,
        "checked": checked,
        "all_agree": True,
        "distinct_values_across_calibrations": len(set(present)),
        "note": (
            "Round-4 finding 4. `motor_fuel_margins` and `means_tested_fuel` "
            "report the same quantity by two call paths; if they ever diverge "
            "this pipeline raises and refuses to write the file."
        ),
    }


def round3_findings(base: Baseline, scenario) -> dict:
    """Persist the round-3 facts the prose has to be written against.

    Four things, none of which the paper could previously state from
    ``results/``:

    ``cap_calibration``
        The pre-war counterfactual cap, the sustained fraction solved with it,
        and the modelled-versus-observed caps in both published quarters. The
        headline moves with these and nothing else in the domestic leg does.
    ``decile_concept``
        What ``household_income_decile`` actually ranks on, measured. The paper
        asserts the opposite of the truth: the ranking is equivalised **BHC**,
        person-weighted, while every burden is household-weighted equivalised
        **AHC**. The ``-1`` sentinel share is the out-of-range households the
        paper already discusses, now explained rather than described.
    ``motor_fuel_margins``
        The means-tested fuel margin and the participation margin, under every
        calibration, so the effect of each specification on each margin is
        visible in one table.
    ``means_tested_fuel``
        The headline consequence: the means-tested share of the aggregate loss
        under the raw imputation and under within-decile parity.
    """
    out: dict = {
        "note": (
            "Round-3 findings, persisted so the paper can be written against "
            "measured facts rather than assertions."
        ),
        "cap_calibration": {
            "prewar_counterfactual_cap_gbp": scen.PREWAR_COUNTERFACTUAL_CAP_GBP,
            "sustained_fraction": scen.REALISED_SUSTAINED_FRACTION,
            "phase_in_profile": list(scen.CAP_PHASE_IN_PROFILE),
            "quarter_labels": list(scen.CAP_QUARTER_LABELS),
            "cap_anchor_pct": scen.CAP_ANCHOR_PCT,
            "cap_base_pct": scen.CAP_BASE_PCT,
            "validation": dict(scen.CAP_VALIDATION),
            "bill_level_pass_through": scen.BILL_LEVEL_PASS_THROUGH,
            "electricity_to_gas_pass_through_ratio": (
                scen.ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO
            ),
            "annual_phase_in_gas": scenario.pass_through.annual_phase_in_gas,
            "annual_applied_gas_factor": (scenario.annual_retail_shock.gas_factor),
            "annual_applied_electricity_factor": (
                scenario.annual_retail_shock.electricity_factor
            ),
            "steady_state_gas_factor": scenario.retail_shock.gas_factor,
            "steady_state_electricity_factor": (
                scenario.retail_shock.electricity_factor
            ),
            "superseded": {
                "baseline_cap_gbp": scen.OFGEM_CAP_JUL_2026_GBP,
                "cap_anchor_pct": (
                    (
                        scen.OFGEM_CAP_OCT_2026_GBP
                        + scen.OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP
                    )
                    / scen.OFGEM_CAP_JUL_2026_GBP
                    - 1.0
                ),
                "why": (
                    "The observed 1 Jul - 30 Sep 2026 cap was used as the "
                    "un-shocked baseline while 2026Q3 carried a phase-in of "
                    "1.0. One quarter cannot be both the unshocked denominator "
                    "and the fully shocked numerator."
                ),
            },
        },
        "decile_concept": _asdict(decile_concept_audit(base)),
        "motor_fuel_margins": {},
        "means_tested_fuel": {},
    }
    out["decile_concept"]["household_weighted_agreement"] = _asdict(
        decile_concept_audit(base, person_weighted=False)
    )["agreement"]
    for calibration in inc.CALIBRATIONS:
        run_base = apply_calibration(base, calibration)
        result, _ = run_scenario(base, scenario, calibration=calibration)
        out["motor_fuel_margins"][calibration] = _asdict(
            motor_fuel_margins(run_base, None)
        )
        out["means_tested_fuel"][calibration] = {
            "means_tested_share_of_loss": (
                result.motor_fuel_margins.means_tested_share_of_loss
                if result.motor_fuel_margins
                else float("nan")
            ),
            "means_tested_mean_fuel_gbp": (
                result.motor_fuel_margins.means_tested_mean_fuel_gbp
                if result.motor_fuel_margins
                else float("nan")
            ),
            "non_means_tested_mean_fuel_gbp": (
                result.motor_fuel_margins.non_means_tested_mean_fuel_gbp
                if result.motor_fuel_margins
                else float("nan")
            ),
            "aggregate_cost_bn": result.aggregate_cost_bn,
            "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
            "d1_d10_ratio_pct": result.all_channel_d1_d10_ratio_pct,
        }
    out["fuel_block_consistency"] = check_fuel_blocks_agree(out)
    raw = out["means_tested_fuel"]["raw"]["means_tested_share_of_loss"]
    parity = out["means_tested_fuel"]["mt_fuel_parity"]["means_tested_share_of_loss"]
    out["means_tested_fuel"]["implied_targeting_multiple"] = {
        "note": (
            "The paper's 'seven times' claim scales inversely with the "
            "means-tested share of the loss, so the ratio of these two shares "
            "is the factor by which the claim moves under within-decile parity."
        ),
        "raw_share": raw,
        "parity_share": parity,
        "factor": (parity / raw if raw else float("nan")),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=int, default=2026)
    args = parser.parse_args()

    _load_env()
    path = dataset_path()
    print(f"dataset: {path}")

    # The microdata is loaded exactly once; every calibration works on a
    # recalibrated copy of the same arrays.
    base = load_baseline(path, args.period)
    print(f"baseline: {base.n:,} households, {base.weight.sum() / 1e6:.1f}m weighted")
    mt = pol.means_tested_flag(path, args.period)
    base = dataclasses.replace(base, means_tested=mt)
    means_tested_share = float(wmean(mt.astype(float), base.weight))
    print(f"means-tested: {100 * means_tested_share:.1f}% of households")

    round3 = round3_findings(base, scen.get_scenario("realised_2026"))
    (RESULTS / "round3_findings.json").write_text(json.dumps(round3, indent=2))
    print(
        "cap calibration: pre-war counterfactual "
        f"£{scen.PREWAR_COUNTERFACTUAL_CAP_GBP:,.0f}, sustained fraction "
        f"{scen.REALISED_SUSTAINED_FRACTION:.3f}, October move "
        f"{100 * scen.CAP_ANCHOR_PCT:.2f}% (was 6.31% against the July cap)"
    )
    print(
        "decile ranking concept: "
        f"{round3['decile_concept']['best_match']} "
        f"(documented truth {round3['decile_concept']['documented_truth']}), "
        f"matches the burden denominator: "
        f"{round3['decile_concept']['matches_burden_denominator']}"
    )
    _mt = round3["means_tested_fuel"]
    print(
        "means-tested share of loss: raw "
        f"{100 * _mt['raw']['means_tested_share_of_loss']:.2f}% -> parity "
        f"{100 * _mt['mt_fuel_parity']['means_tested_share_of_loss']:.2f}% "
        f"(x{_mt['implied_targeting_multiple']['factor']:.2f})"
    )

    persisted = persisted_values(base)
    (RESULTS / "persisted_values.json").write_text(json.dumps(persisted, indent=2))

    robustness_dir = RESULTS / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    profiles = cash_profiles(base, scen.SCENARIOS["realised_2026"])
    (robustness_dir / "cash_profiles.json").write_text(json.dumps(profiles, indent=2))
    for calibration, payload in profiles["profiles"].items():
        print(
            f"cash profile {calibration:16} monotone="
            f"{payload['is_monotone_increasing']!s:5} peak=D{payload['peak_decile']} "
            f"falls at {payload['deciles_where_cash_falls'] or 'nowhere'}"
        )
    print(
        "motor fuel D1/D10 £/yr: raw "
        f"{persisted['motor_fuel_decile_mean_gbp']['raw'][0]:.0f}/"
        f"{persisted['motor_fuel_decile_mean_gbp']['raw'][-1]:.0f}, ONS-shape "
        f"{persisted['motor_fuel_decile_mean_gbp']['ons_shape'][0]:.0f}/"
        f"{persisted['motor_fuel_decile_mean_gbp']['ons_shape'][-1]:.0f}, "
        "ONS-levels "
        f"{persisted['motor_fuel_decile_mean_gbp']['ons_both_levels'][0]:.0f}/"
        f"{persisted['motor_fuel_decile_mean_gbp']['ons_both_levels'][-1]:.0f}"
    )

    rows: list[dict] = []
    policy_rows: list[dict] = []
    derived = derived_scenarios()
    for variant in variants():
        scenario = scenario_for(variant.scenario_key, derived)
        if scenario is None:
            print(f"\n{variant.name}: scenario unavailable, skipped")
            continue
        run_base = apply_calibration(base, variant.calibration)
        result, cost = run_scenario(
            base,
            scenario,
            calibration=variant.calibration,
            income_basis=variant.income_basis,
            domestic_basis=variant.domestic_basis,
        )
        out = variant.out_dir
        out.mkdir(parents=True, exist_ok=True)

        payload = _asdict(result)
        payload["means_tested_share"] = means_tested_share
        payload["specification"] = variant.label
        payload["variant"] = variant.name
        payload["pump_sustained_fraction"] = (
            scenario.pass_through.pump_sustained_fraction
        )
        payload["sustained_fraction"] = scenario.pass_through.sustained_fraction
        (out / "shock.json").write_text(json.dumps(payload, indent=2))

        (out / "aggregates.json").write_text(
            json.dumps(
                {
                    "aggregate_energy_spend_bn": wsum(run_base.energy, run_base.weight)
                    / 1e9,
                    "aggregate_fuel_spend_bn": wsum(
                        run_base.motor_fuel, run_base.weight
                    )
                    / 1e9,
                },
                indent=2,
            )
        )

        coverage = result.coverage
        print(
            f"\n{variant.name}: £{result.aggregate_cost_bn:.2f}bn, "
            f"mean £{result.mean_loss_gbp:.0f} "
            f"({result.mean_loss_pct:.2f}% of income), motor fuel "
            f"{100 * result.motor_fuel_share_of_loss:.1f}% of loss"
        )
        print(
            f"   decile 1 {result.decile[0].mean_loss_pct:.2f}% "
            f"(£{result.decile[0].mean_loss_gbp:.0f}) vs "
            f"decile 10 {result.decile[-1].mean_loss_pct:.2f}% "
            f"(£{result.decile[-1].mean_loss_gbp:.0f}); "
            f"mean income £{result.mean_income_gbp:,.0f}"
        )
        if coverage is not None:
            print(
                f"   outside deciles 1-10: {coverage.households_m:.2f}m households "
                f"({100 * coverage.share_of_households:.2f}%), "
                f"{100 * coverage.share_of_loss:.2f}% of the loss, "
                f"{100 * coverage.zero_or_negative_income_share:.1f}% of them with "
                "zero or negative income"
            )

        for pkey, policy in pol.POLICIES.items():
            score, _gain = pol.score_policy(run_base, cost, mt, policy)
            (out / f"{pkey}.json").write_text(json.dumps(_asdict(score), indent=2))
            policy_rows.append(
                {
                    "variant": variant.name,
                    "scenario": variant.scenario_key,
                    "policy": pkey,
                    "label": policy.label,
                    "cost_bn": score.cost_bn,
                    "share_to_bottom_three": score.share_to_bottom_three,
                    "cost_per_pound_decile_one": score.cost_per_pound_decile_one,
                    "cost_per_pound_decile_one_units": (
                        score.cost_per_pound_decile_one_units
                    ),
                    "share_of_aggregate_loss_offset": (
                        score.share_of_aggregate_loss_offset
                    ),
                    "implied_parameter": score.implied_parameter,
                    "parameter_units": score.parameter_units,
                    "uncompensated_share": score.uncompensated_share_overall,
                    "fully_compensated_share": score.fully_compensated_share,
                    "mean_gain_gbp": score.mean_gain_gbp,
                    "net_loss_after_gbp": score.net_loss_after_policy_gbp,
                }
            )

        rows.append(comparison_row(variant, result, run_base, scenario))

    robustness = RESULTS / "robustness"
    robustness.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(robustness / "comparison.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(robustness / "policy_comparison.csv", index=False)
    print(f"\nwrote {robustness / 'comparison.csv'} ({len(rows)} rows)")

    sweep = domestic_leg_sweep(base)
    sensitivity = RESULTS / "sensitivity"
    sensitivity.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sweep).to_csv(sensitivity / "domestic_leg.csv", index=False)
    print(f"wrote {sensitivity / 'domestic_leg.csv'} ({len(sweep)} rows)")


if __name__ == "__main__":
    main()

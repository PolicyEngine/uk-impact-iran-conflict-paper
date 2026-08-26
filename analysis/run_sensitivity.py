#!/usr/bin/env python3
"""Robustness and sensitivity sweeps for the 2026 energy-shock incidence paper.

Three sweeps, each writing one CSV under ``results/sensitivity/``:

1. ``elasticity.csv`` — the demand-response sweep. The paper's MAIN
   specification is zero elasticity (first-order Deaton approximation, an
   explicit upper bound on the welfare loss). This sweeps a flat own-price
   elasticity from 0 to -0.8 and also runs the named
   :class:`uk_iran_conflict.elasticity.ElasticitySpec` variants.

   **Two different objects are reported, and only one of them is welfare**
   (docs/FIXES.md A5). ``aggregate_loss_bn`` is the change in *expenditure*;
   at eps = -0.8 it counts a household that stops heating its home as barely
   worse off, which is the "heat or eat" fallacy the appendix criticises three
   paragraphs later. ``cv_lower_bn`` and ``cv_upper_bn`` bracket the
   compensating variation between the Paasche term ``q1 . dp`` and the
   Laspeyres term ``q0 . dp``. Quote the bracket, never the spend change.

2. ``cap_lag.csv`` — the wholesale-to-retail lag sweep. ``CAP_LAG_QUARTERS``
   is the paper's most contestable modelling choice, so the lag is swept over
   1-4 quarters and BOTH the annualised (2026-window) and cumulative
   (whole-path) loss are reported, so a reader can separate a genuine effect
   from an artefact of which quarters fall inside the 2026 window.

3. ``asymmetry.csv`` — the gas/electricity asymmetry sweep.
   ``MARGINAL_PRICING_SHARE`` = 0.85 is the paper's central modelling claim;
   1.0 recovers the naive symmetric assumption. Swept over 0.7/0.85/1.0.

All sweeps use the ``realised_2026`` scenario (the paper's realised central
case) and the same Baseline object, loaded once.

Usage::

    python analysis/run_sensitivity.py

Needs the private PolicyEngine UK microdata: set ``HUGGING_FACE_TOKEN`` in the
gitignored ``.env``, exactly as ``analysis/run_incidence.py`` does.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from uk_iran_conflict import elasticity as ela
from uk_iran_conflict import policies as pol
from uk_iran_conflict import reforms
from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import (
    Baseline,
    load_baseline,
    run_scenario,
    sustained_pump_factors,
    wmean,
    wshare,
    wsum,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sensitivity"

DATASET_REPO = "policyengine/populace-uk-private"
DATASET_FILE = "populace_uk_2023.h5"

#: Every sweep is run on the paper's realised central case.
SCENARIO_KEY = "realised_2026"

#: Flat elasticities swept in sweep 1 (NEGATIVE epsilon convention).
EPSILON_GRID: tuple[float, ...] = (
    0.0,
    -0.1,
    -0.2,
    -0.3,
    -0.4,
    -0.5,
    -0.6,
    -0.7,
    -0.8,
)

#: Quarter labels used to place the phase-in profile on a calendar in sweep 2.
#: The war starts ~28 Feb 2026, so the shock quarter is 2026Q1.
SHOCK_QUARTER_INDEX = 0
CALENDAR = ("2026Q1", "2026Q2", "2026Q3", "2026Q4", "2027Q1", "2027Q2", "2027Q3")


# --------------------------------------------------------------------------
# environment / data, mirroring analysis/run_incidence.py
# --------------------------------------------------------------------------


def _load_env() -> None:
    """Read the gitignored .env so the token never has to be exported by hand."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def dataset_path() -> str:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "No Hugging Face token. Set HUGGING_FACE_TOKEN in .env — the "
            "PolicyEngine UK microdata is a private dataset."
        )
    return hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset", token=token)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _decile_stats(base: Baseline, cost: np.ndarray, d: int) -> tuple[float, float]:
    """(mean loss £, loss as % of income) for one income decile.

    The percentage is the AGGREGATE ratio (``wshare``) — weighted sum of loss
    over weighted sum of income — never a mean of household-level ratios.
    """
    sel = base.decile == d
    w = base.weight[sel]
    income = np.clip(base.net_income[sel], 1, None)
    return wmean(cost[sel], w), 100 * wshare(cost[sel], income, w)


def _headline(base: Baseline, cost: np.ndarray) -> dict[str, float]:
    """Mean loss, aggregate loss and the decile-1/decile-10 cells."""
    w = base.weight
    income = np.clip(base.net_income, 1, None)
    d1_gbp, d1_pct = _decile_stats(base, cost, 1)
    d10_gbp, d10_pct = _decile_stats(base, cost, 10)
    return {
        "mean_loss_gbp": wmean(cost, w),
        "mean_loss_pct": 100 * wshare(cost, income, w),
        "aggregate_loss_bn": wsum(cost, w) / 1e9,
        "decile1_loss_gbp": d1_gbp,
        "decile1_loss_pct": d1_pct,
        "decile10_loss_gbp": d10_gbp,
        "decile10_loss_pct": d10_pct,
    }


def _price_ratios(scenario) -> dict[str, float]:
    """Carrier price ratios (p1/p0) implied by a scenario."""
    gas_factor, elec_factor = reforms.retail_factors(scenario)
    # Must use the damped pump factors, not the raw peaks: ``reforms`` returns
    # the quoted peak moves, and the peak-to-year damping lives in ``incidence``
    # alongside the gas damping. Reading the raw peaks here silently ran the
    # elasticity and cap-lag sweeps on the peak-fuel upper bound rather than the
    # main specification.
    petrol_factor, diesel_factor = sustained_pump_factors(scenario)
    return {
        "gas": gas_factor,
        "electricity": elec_factor,
        "petrol": petrol_factor,
        "diesel": diesel_factor,
    }


def _annual_factors(base: Baseline, scenario) -> dict[str, float]:
    """Per-carrier factor reconciling a raw ``q0 x dp`` leg with ``shock_cost``.

    Defensive coupling. ``uk_iran_conflict.incidence.shock_cost`` is the single
    definition of the headline cost, and it is acquiring a phase-in-weighted
    annual factor (docs/FIXES.md D2): the annual domestic price is the
    consumption-weighted average of the quarterly cap levels, not the
    steady-state level. Anything computed here from ``base.gas * (ratio - 1)``
    would otherwise silently keep charging the steady state and the sweep would
    stop agreeing with the headline it is a sweep *around*.

    So each leg is scaled by the ratio of the aggregate ``shock_cost`` leg to
    the aggregate raw leg. If ``shock_cost`` applies no annual factor the ratio
    is 1.0 and nothing changes; if it applies one, the sweep follows it without
    needing to know what it is. Elasticity enters multiplicatively and so is
    unaffected by the rescaling.
    """
    from uk_iran_conflict.incidence import shock_cost  # noqa: PLC0415

    ratios = _price_ratios(scenario)
    cost = shock_cost(base, scenario)
    w = base.weight
    raw = {
        "gas": wsum(base.gas * (ratios["gas"] - 1.0), w),
        "electricity": wsum(base.electricity * (ratios["electricity"] - 1.0), w),
        "motor_fuel": wsum(
            base.petrol * (ratios["petrol"] - 1.0)
            + base.diesel * (ratios["diesel"] - 1.0),
            w,
        ),
    }
    modelled = {
        "gas": wsum(cost.gas, w),
        "electricity": wsum(cost.electricity, w),
        "motor_fuel": wsum(cost.motor_fuel, w),
    }
    return {k: (modelled[k] / raw[k] if raw[k] else 1.0) for k in raw}


def _elastic_cost(
    base: Baseline, scenario, spec: ela.ElasticitySpec
) -> dict[str, np.ndarray]:
    """Per-household spend change by carrier under an elasticity spec.

    Uses the constant-elasticity spend factor ``(p1/p0)**(1+eps)`` from
    :mod:`uk_iran_conflict.elasticity`, never the linear approximation. For an
    income-varying spec the elasticity is resolved per household from its
    income decile; households with an out-of-range decile fall back to the
    spec's own decile-table mean (the module's documented behaviour).
    """
    ratios = _price_ratios(scenario)
    deciles = np.asarray(base.decile).astype(int)
    present = sorted({int(d) for d in deciles})

    def factor(carrier: str, ratio: float) -> np.ndarray:
        """Spend multiplier minus one, per household."""
        if spec.flat is not None:
            eps = ela.elasticity_for(spec, carrier)
            return np.full(base.n, ela.spend_factor(ratio, eps) - 1.0)
        out = np.zeros(base.n)
        for d in present:
            eps = ela.elasticity_for(spec, carrier, d)
            out[deciles == d] = ela.spend_factor(ratio, eps) - 1.0
        return out

    annual = _annual_factors(base, scenario)
    return {
        "gas": annual["gas"] * base.gas * factor("gas", ratios["gas"]),
        "electricity": annual["electricity"]
        * base.electricity
        * factor("electricity", ratios["electricity"]),
        "motor_fuel": annual["motor_fuel"]
        * (
            base.petrol * factor("motor_fuel", ratios["petrol"])
            + base.diesel * factor("motor_fuel", ratios["diesel"])
        ),
    }


def _cv_cost(
    base: Baseline, scenario, spec: ela.ElasticitySpec
) -> tuple[np.ndarray, np.ndarray]:
    """Per-household (lower, upper) money-metric bounds on the welfare loss.

    Upper is the Laspeyres term ``q0 . dp`` — the paper's zero-elasticity
    headline, and invariant to ``spec`` by construction. Lower is the Paasche
    term ``q1 . dp``: post-adjustment quantities valued at the price change.
    The true compensating variation lies between them.

    This is the fix for docs/FIXES.md A5. The elasticity enters only through
    ``q1``, so its whole effect on the *welfare* number is the factor
    ``(p1/p0)**eps`` — a few per cent, not the four-fifths that the change in
    expenditure suggests.
    """
    ratios = _price_ratios(scenario)
    deciles = np.asarray(base.decile).astype(int)
    present = sorted({int(d) for d in deciles})

    def quantity(carrier: str, ratio: float) -> np.ndarray:
        """Post-shock quantity factor ``(p1/p0)**eps``, per household."""
        if spec.flat is not None:
            eps = ela.elasticity_for(spec, carrier)
            return np.full(base.n, ela.quantity_factor(ratio, eps))
        out = np.ones(base.n)
        for d in present:
            eps = ela.elasticity_for(spec, carrier, d)
            out[deciles == d] = ela.quantity_factor(ratio, eps)
        return out

    annual = _annual_factors(base, scenario)
    legs = (
        (base.gas, "gas", ratios["gas"]),
        (base.electricity, "electricity", ratios["electricity"]),
        (base.petrol, "motor_fuel", ratios["petrol"]),
        (base.diesel, "motor_fuel", ratios["diesel"]),
    )
    lower = np.zeros(base.n)
    upper = np.zeros(base.n)
    for spend, carrier, ratio in legs:
        static = annual[carrier] * spend * (ratio - 1.0)
        upper += static
        lower += static * quantity(carrier, ratio)
    return lower, upper


def _flat_spec(epsilon: float) -> ela.ElasticitySpec:
    """A flat, all-carrier spec at ``epsilon`` — the sweep grid."""
    if epsilon == 0.0:
        return ela.ElasticitySpec.main()
    return ela.ElasticitySpec(
        name=f"flat_{epsilon:+.2f}",
        source="Uniform own-price elasticity sweep (appendix robustness grid).",
        flat={c: epsilon for c in ela.CARRIERS},
    )


def _mean_epsilon(base: Baseline, spec: ela.ElasticitySpec) -> float:
    """Spend-weighted mean elasticity implied by a spec, for reporting."""
    spends = {
        "gas": base.gas,
        "electricity": base.electricity,
        "motor_fuel": base.motor_fuel,
    }
    num = 0.0
    den = 0.0
    deciles = np.asarray(base.decile).astype(int)
    for carrier, spend in spends.items():
        total = wsum(spend, base.weight)
        if spec.flat is not None:
            num += ela.elasticity_for(spec, carrier) * total
            den += total
            continue
        for d in sorted({int(x) for x in deciles}):
            sel = deciles == d
            s = wsum(spend[sel], base.weight[sel])
            num += ela.elasticity_for(spec, carrier, d) * s
            den += s
    return num / den if den else float("nan")


# --------------------------------------------------------------------------
# sweep 1 — elasticity
# --------------------------------------------------------------------------


def sweep_elasticity(base: Baseline, scenario) -> pd.DataFrame:
    """Sweep the own-price elasticity from 0 to -0.8, plus the named specs."""
    # Named specs first, then the grid in increasing elasticity magnitude, so
    # the LAST row of the CSV is the strongest-response variant — the row
    # analysis/emit_tex_values.py reads for its "high elasticity" macros.
    specs: list[tuple[str, str, ela.ElasticitySpec]] = [
        ("labandeira_short_run", "named", ela.ElasticitySpec.labandeira_flat()),
        (
            "labandeira_long_run",
            "named",
            ela.ElasticitySpec.labandeira_flat("long_run"),
        ),
        (
            "priesmann_short_run",
            "named",
            ela.ElasticitySpec.priesmann_income_varying(),
        ),
        (
            "priesmann_long_run",
            "named",
            ela.ElasticitySpec.priesmann_income_varying("long_run"),
        ),
        (
            "prior_repo_replication",
            "named",
            ela.ElasticitySpec.prior_repo_replication(),
        ),
    ]
    specs += [(f"flat_{e:.1f}", "grid", _flat_spec(e)) for e in EPSILON_GRID]

    # The upper bound is the zero-elasticity main specification, computed
    # first so the "shaved" column is well defined whatever the row order.
    zero = _elastic_cost(base, scenario, ela.ElasticitySpec.main())
    upper = _headline(base, zero["gas"] + zero["electricity"] + zero["motor_fuel"])
    w = base.weight

    rows: list[dict] = []
    print("\n=== sweep 1: elasticity (realised_2026) ===")
    print(
        f"{'spec':26} {'eps':>6} {'mean £':>8} {'% inc':>7} "
        f"{'D1 £':>7} {'D1 %':>6} {'D10 £':>7} {'D10 %':>6} "
        f"{'spend £bn':>9} {'CV lo':>7} {'CV hi':>7} {'sp shv':>7} {'w shv':>6}"
    )
    for label, kind, spec in specs:
        cost = _elastic_cost(base, scenario, spec)
        total = cost["gas"] + cost["electricity"] + cost["motor_fuel"]
        h = _headline(base, total)
        shaved = 1.0 - h["aggregate_loss_bn"] / upper["aggregate_loss_bn"]
        # Money-metric bounds (A5). The upper bound is elasticity-invariant, so
        # cv_upper_bn is the same in every row and equals the headline.
        cv_lo, cv_hi = _cv_cost(base, scenario, spec)
        cv_lower_bn = wsum(cv_lo, w) / 1e9
        cv_upper_bn = wsum(cv_hi, w) / 1e9
        welfare_shaved = (
            1.0 - cv_lower_bn / cv_upper_bn if cv_upper_bn else float("nan")
        )
        row = {
            "spec": label,
            "kind": kind,
            "is_main_specification": bool(spec.is_main_specification),
            "epsilon_mean": _mean_epsilon(base, spec),
            **h,
            # aliases consumed by analysis/emit_tex_values.py
            "exchequer_cost_bn": h["aggregate_loss_bn"],
            "mean_household_change": -h["mean_loss_gbp"],
            "share_of_upper_bound_shaved": shaved,
            # --- money-metric welfare (docs/FIXES.md A5) ------------------
            # aggregate_loss_bn above is a change in EXPENDITURE and is not a
            # welfare measure under a non-zero elasticity. These are.
            "aggregate_spend_change_bn": h["aggregate_loss_bn"],
            "cv_lower_bn": cv_lower_bn,
            "cv_upper_bn": cv_upper_bn,
            "cv_lower_mean_gbp": wmean(cv_lo, w),
            "cv_upper_mean_gbp": wmean(cv_hi, w),
            "welfare_share_shaved": welfare_shaved,
            "measure_note": (
                "aggregate_loss_bn = change in expenditure; "
                "cv_lower_bn/cv_upper_bn = Paasche/Laspeyres bounds on the "
                "compensating variation. Quote the bounds."
            ),
            "decile_gradient_pp": h["decile1_loss_pct"] - h["decile10_loss_pct"],
            "source": spec.source,
        }
        rows.append(row)
        print(
            f"{label:26} {row['epsilon_mean']:6.2f} {h['mean_loss_gbp']:8.0f} "
            f"{h['mean_loss_pct']:7.2f} {h['decile1_loss_gbp']:7.0f} "
            f"{h['decile1_loss_pct']:6.2f} {h['decile10_loss_gbp']:7.0f} "
            f"{h['decile10_loss_pct']:6.2f} {h['aggregate_loss_bn']:9.2f} "
            f"{cv_lower_bn:7.2f} {cv_upper_bn:7.2f} "
            f"{100 * shaved:6.1f}% {100 * welfare_shaved:5.1f}%"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# sweep 2 — wholesale-to-retail lag
# --------------------------------------------------------------------------


def sweep_cap_lag(base: Baseline, scenario) -> pd.DataFrame:
    """Sweep ``lag_quarters`` over 1-4, reporting annualised AND cumulative loss.

    The scenario's phase-in profile is placed on a calendar starting at the
    shock quarter (2026Q1). With a lag of L quarters the profile's first
    element lands in 2026Q1+L. Two very different quantities follow:

    * **annualised** — the loss inside the 2026 calendar year only. This falls
      mechanically as the lag lengthens, because fewer phase-in quarters fit
      inside the window. It is largely a windowing artefact.
    * **cumulative** — the loss over the whole phase-in path, wherever it
      falls. This is invariant to the lag **by construction, as an identity**:
      ``cumulative_weight = sum(profile) / 4`` contains no lag term at all
      (docs/FIXES.md D24). It is not an empirical finding that the cumulative
      loss survives a re-timing of the cap; it is arithmetic, and the sweep
      asserts it rather than discovering it.

    The domestic-energy (cap) channel is lagged; motor fuel is not, since pump
    prices pass through in weeks. Motor fuel therefore enters both columns at
    its full annual value.
    """
    pt = scenario.pass_through
    profile = tuple(pt.phase_in_profile)
    ratios = _price_ratios(scenario)

    # Steady-state (full-pass-through) annual costs.
    domestic = base.gas * (ratios["gas"] - 1.0) + base.electricity * (
        ratios["electricity"] - 1.0
    )
    fuel = base.petrol * (ratios["petrol"] - 1.0) + base.diesel * (
        ratios["diesel"] - 1.0
    )
    w = base.weight
    income = np.clip(base.net_income, 1, None)
    steady_domestic_bn = wsum(domestic, w) / 1e9

    rows: list[dict] = []
    print("\n=== sweep 2: wholesale-to-retail cap lag (realised_2026) ===")
    print(
        f"{'lag':>3} {'first qtr':>9} {'qtrs in 2026':>12} "
        f"{'ann £':>7} {'ann £bn':>8} {'ann %':>6} "
        f"{'cum £':>7} {'cum £bn':>8} {'ann share':>9} {'cum share':>9}"
    )
    for lag in (1, 2, 3, 4):
        start = SHOCK_QUARTER_INDEX + lag
        in_2026 = [
            profile[i - start]
            for i in range(4)  # 2026Q1..2026Q4
            if 0 <= i - start < len(profile)
        ]
        # Quarterly weights: each quarter carries a quarter of an annual bill.
        annual_weight = sum(in_2026) / 4.0
        # D24: no lag term appears on the right-hand side. The "exact
        # invariance" of the cumulative loss is this identity, not a result.
        cumulative_weight = sum(profile) / 4.0
        assert cumulative_weight == sum(profile) / 4.0

        ann_cost = domestic * annual_weight + fuel
        cum_cost = domestic * cumulative_weight + fuel

        first_quarter = CALENDAR[start] if start < len(CALENDAR) else "2027Q4+"
        row = {
            "lag_quarters": lag,
            "first_cap_quarter": first_quarter,
            "phase_in_quarters_in_2026": len(in_2026),
            "phase_in_weight_2026": annual_weight,
            "phase_in_weight_cumulative": cumulative_weight,
            "annualised_mean_loss_gbp": wmean(ann_cost, w),
            "annualised_mean_loss_pct": 100 * wshare(ann_cost, income, w),
            "annualised_loss_bn": wsum(ann_cost, w) / 1e9,
            "cumulative_mean_loss_gbp": wmean(cum_cost, w),
            "cumulative_mean_loss_pct": 100 * wshare(cum_cost, income, w),
            "cumulative_loss_bn": wsum(cum_cost, w) / 1e9,
            # Shares are of the domestic channel's full-pass-through year:
            # the quantity the lag actually acts on.
            "annualised_share": annual_weight,
            "cumulative_share": cumulative_weight,
            # True by construction: cumulative_weight is a function of the
            # phase-in profile alone (docs/FIXES.md D24).
            "cumulative_is_lag_invariant_by_construction": True,
            "steady_state_domestic_bn": steady_domestic_bn,
            "annualised_domestic_bn": steady_domestic_bn * annual_weight,
            "cumulative_domestic_bn": steady_domestic_bn * cumulative_weight,
            "motor_fuel_bn": wsum(fuel, w) / 1e9,
        }
        rows.append(row)
        print(
            f"{lag:3d} {first_quarter:>9} {len(in_2026):12d} "
            f"{row['annualised_mean_loss_gbp']:7.0f} "
            f"{row['annualised_loss_bn']:8.2f} "
            f"{row['annualised_mean_loss_pct']:6.2f} "
            f"{row['cumulative_mean_loss_gbp']:7.0f} "
            f"{row['cumulative_loss_bn']:8.2f} "
            f"{100 * row['annualised_share']:8.0f}% "
            f"{100 * row['cumulative_share']:8.0f}%"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# sweep 3 — gas/electricity asymmetry
# --------------------------------------------------------------------------


def sweep_asymmetry(base: Baseline, scenario) -> pd.DataFrame:
    """Sweep ``marginal_pricing_share`` over 0.7 / 0.85 / 1.0.

    1.0 is the naive symmetric assumption: gas sets the power price all the
    time, so the electricity shock is as large as the wholesale share of an
    electricity bill allows.
    """
    rows: list[dict] = []
    print("\n=== sweep 3: gas/electricity asymmetry (realised_2026) ===")
    print(
        f"{'share':>6} {'gas x':>7} {'elec x':>7} {'mean £':>8} {'% inc':>7} "
        f"{'D1 %':>6} {'D10 %':>6} {'grad':>6} "
        f"{'gas%loss':>9} {'elec%loss':>9} {'£bn':>7}"
    )
    for share in (0.70, 0.85, 1.00):
        pt = dataclasses.replace(scenario.pass_through, marginal_pricing_share=share)
        variant = dataclasses.replace(scenario, pass_through=pt)
        result, cost = run_scenario(base, variant)
        d1 = result.decile[0]
        d10 = result.decile[-1]
        gas_f, elec_f = reforms.retail_factors(variant)
        # Split within domestic energy only, so the fast motor-fuel channel
        # does not dilute the comparison.
        w = base.weight
        dom = wsum(cost.domestic, w)
        row = {
            "marginal_pricing_share": share,
            "is_paper_central": share == scen.MARGINAL_PRICING_SHARE,
            "gas_price_factor": gas_f,
            "electricity_price_factor": elec_f,
            "mean_loss_gbp": result.mean_loss_gbp,
            "mean_loss_pct": result.mean_loss_pct,
            "aggregate_loss_bn": result.aggregate_cost_bn,
            "decile1_loss_gbp": d1.mean_loss_gbp,
            "decile1_loss_pct": d1.mean_loss_pct,
            "decile10_loss_gbp": d10.mean_loss_gbp,
            "decile10_loss_pct": d10.mean_loss_pct,
            "decile_gradient_pp": d1.mean_loss_pct - d10.mean_loss_pct,
            "decile_ratio_pct": (
                d1.mean_loss_pct / d10.mean_loss_pct
                if d10.mean_loss_pct
                else float("nan")
            ),
            "gas_share_of_loss": result.gas_share_of_loss,
            "electricity_share_of_loss": result.electricity_share_of_loss,
            "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
            "gas_share_of_domestic_loss": wsum(cost.gas, w) / dom
            if dom
            else float("nan"),
            "electricity_share_of_domestic_loss": (
                wsum(cost.electricity, w) / dom if dom else float("nan")
            ),
            "domestic_loss_bn": dom / 1e9,
        }
        rows.append(row)
        print(
            f"{share:6.2f} {gas_f:7.4f} {elec_f:7.4f} "
            f"{row['mean_loss_gbp']:8.0f} {row['mean_loss_pct']:7.2f} "
            f"{row['decile1_loss_pct']:6.2f} {row['decile10_loss_pct']:6.2f} "
            f"{row['decile_gradient_pp']:6.2f} "
            f"{100 * row['gas_share_of_domestic_loss']:8.1f}% "
            f"{100 * row['electricity_share_of_domestic_loss']:8.1f}% "
            f"{row['aggregate_loss_bn']:7.2f}"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# sweep 4 — the policy scorecard at stated cost AND at a common envelope
# --------------------------------------------------------------------------


def sweep_policy_envelope(
    base: Baseline, scenario, mt: np.ndarray, envelope_bn: float
) -> pd.DataFrame:
    """Score all five instruments twice: at own cost, and at a common envelope.

    Two fixes land here (docs/FIXES.md B7, B9).

    **B7.** At the sponsors' own stated costs the scorecard sets a £1.3bn
    instrument against a £5.4bn one and then reads the difference as
    *targeting*. Targeting is a statement about the shape of a spend, and the
    shape is only comparable at a common size, so every instrument is also
    rescaled to ``envelope_bn``.

    **B9.** "Share of losers uncompensated" is knife-edge — a household short
    by £1 counts the same as one short by £900 — and it is applied to a loss
    that is itself an upper bound, which is how VAT zero-rating came to be
    described as compensating nobody while delivering the second-highest mean
    gain and a lower mean residual loss than the social tariff. The continuous
    measures (share of aggregate loss offset, mean and median residual loss,
    overall and by decile) are written out beside it.
    """
    from uk_iran_conflict.incidence import shock_cost  # noqa: PLC0415

    cost = shock_cost(base, scenario)
    rows: list[dict] = []
    print("\n=== sweep 4: policy scorecard, stated cost vs common envelope ===")
    print(
        f"{'policy':16} {'envelope':>9} {'£bn':>6} {'D1-3':>6} {'offset':>7} "
        f"{'mean res':>9} {'med res':>8} {'unc':>6} {'£/£ D1':>7}"
    )
    for score in pol.scorecard(base, cost, mt, envelope_bn=envelope_bn):
        row = {
            "scenario": getattr(scenario, "key", ""),
            "policy": score.policy,
            "label": score.label,
            "envelope": score.envelope,
            "envelope_bn": score.envelope_bn,
            "envelope_scale": score.envelope_scale,
            "cost_bn": score.cost_bn,
            "stated_cost_bn": score.stated_cost_bn,
            "share_to_bottom_three": score.share_to_bottom_three,
            "cost_per_pound_decile_one": score.cost_per_pound_decile_one,
            "mean_gain_gbp": score.mean_gain_gbp,
            "uncompensated_share_overall": score.uncompensated_share_overall,
            "fully_compensated_share": score.fully_compensated_share,
            "net_loss_after_policy_gbp": score.net_loss_after_policy_gbp,
            "share_of_aggregate_loss_offset": score.share_of_aggregate_loss_offset,
            "mean_residual_loss_gbp": score.mean_residual_loss_gbp,
            "median_residual_loss_gbp": score.median_residual_loss_gbp,
        }
        for d in range(1, 11):
            row[f"mean_residual_loss_d{d}"] = score.mean_residual_loss_by_decile.get(d)
            row[f"median_residual_loss_d{d}"] = (
                score.median_residual_loss_by_decile.get(d)
            )
            offset_d = score.share_of_loss_offset_by_decile
            row[f"share_of_loss_offset_d{d}"] = offset_d.get(d)
            row[f"mean_gain_d{d}"] = score.mean_gain_by_decile.get(d)
            row[f"uncompensated_d{d}"] = score.uncompensated_by_decile.get(d)
        rows.append(row)
        print(
            f"{score.policy:16} {score.envelope:>9} {score.cost_bn:6.2f} "
            f"{100 * score.share_to_bottom_three:5.1f}% "
            f"{100 * score.share_of_aggregate_loss_offset:6.1f}% "
            f"{score.mean_residual_loss_gbp:9.0f} "
            f"{score.median_residual_loss_gbp:8.0f} "
            f"{100 * score.uncompensated_share_overall:5.1f}% "
            f"{score.cost_per_pound_decile_one:7.2f}"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# sweep 5 — petrol, diesel and diesel share by decile
# --------------------------------------------------------------------------


def sweep_fuel_by_decile(base: Baseline, scenario) -> pd.DataFrame:
    """Petrol spend, diesel spend and diesel share, by income decile.

    Settles docs/FIXES.md D26. The decile-8 cash spike is narrated in the text
    as a mileage story, but the diesel uplift in this scenario is far larger
    than the petrol one (+21.6% against +12.0% after peak-to-year damping), so
    a decile that simply holds more diesel cars will show a bigger cash loss at
    identical mileage. The two explanations make different predictions about
    the diesel *share*, which is what this table reports. No narration here:
    the columns are the evidence.
    """
    ratios = _price_ratios(scenario)
    annual = _annual_factors(base, scenario)["motor_fuel"]
    w = base.weight
    rows: list[dict] = []
    print("\n=== sweep 5: motor fuel composition by decile ===")
    print(
        f"{'decile':>6} {'petrol £':>9} {'diesel £':>9} {'fuel £':>8} "
        f"{'diesel share':>13} {'fuel loss £':>12} {'diesel % of loss':>17}"
    )
    for d in range(1, 11):
        sel = base.decile == d
        wd = w[sel]
        if wd.sum() <= 0:
            continue
        petrol = wmean(base.petrol[sel], wd)
        diesel = wmean(base.diesel[sel], wd)
        petrol_loss = annual * petrol * (ratios["petrol"] - 1.0)
        diesel_loss = annual * diesel * (ratios["diesel"] - 1.0)
        fuel = petrol + diesel
        loss = petrol_loss + diesel_loss
        row = {
            "decile": d,
            "petrol_spend_gbp": petrol,
            "diesel_spend_gbp": diesel,
            "motor_fuel_spend_gbp": fuel,
            "diesel_share_of_fuel_spend": diesel / fuel if fuel else float("nan"),
            "petrol_price_factor": ratios["petrol"],
            "diesel_price_factor": ratios["diesel"],
            "petrol_loss_gbp": petrol_loss,
            "diesel_loss_gbp": diesel_loss,
            "motor_fuel_loss_gbp": loss,
            "diesel_share_of_fuel_loss": diesel_loss / loss if loss else float("nan"),
            "share_with_any_fuel_spend": float(
                wd[base.motor_fuel[sel] > 0].sum() / wd.sum()
            ),
            "households_m": float(wd.sum() / 1e6),
        }
        rows.append(row)
        print(
            f"{d:6d} {petrol:9.0f} {diesel:9.0f} {fuel:8.0f} "
            f"{100 * row['diesel_share_of_fuel_spend']:12.1f}% "
            f"{loss:12.0f} {100 * row['diesel_share_of_fuel_loss']:16.1f}%"
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--scenario", default=SCENARIO_KEY)
    parser.add_argument(
        "--envelope-bn",
        type=float,
        default=pol.COMMON_ENVELOPE_BN,
        help="common exchequer envelope for the policy scorecard, £bn",
    )
    args = parser.parse_args()

    _load_env()
    path = dataset_path()
    print(f"dataset: {path}")

    base = load_baseline(path, args.period)
    print(f"baseline: {base.n:,} households, {base.weight.sum() / 1e6:.1f}m weighted")

    scenario = scen.SCENARIOS[args.scenario]
    gas_f, elec_f = reforms.retail_factors(scenario)
    print(
        f"scenario: {scenario.label} — retail gas x{gas_f:.4f}, "
        f"electricity x{elec_f:.4f}"
    )

    # C13: resolve the means-tested variable set loudly and record it, so the
    # paper's 15.7% coverage claim is checkable variable by variable against
    # the DWP caseloads in docs/VALIDATION.md Check 4. This also caches the
    # real household child count the JRF block needs (B8).
    audit_path = ROOT / "results" / "means_tested_audit.json"
    mt = pol.means_tested_flag(path, args.period, audit_path=audit_path)
    print(f"wrote {audit_path}")
    print(
        f"means-tested: {100 * wmean(mt.astype(float), base.weight):.1f}% of "
        f"households ({base.weight[mt].sum() / 1e6:.2f}m)"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("elasticity", sweep_elasticity(base, scenario)),
        ("cap_lag", sweep_cap_lag(base, scenario)),
        ("asymmetry", sweep_asymmetry(base, scenario)),
        (
            "policy_envelope",
            sweep_policy_envelope(base, scenario, mt, args.envelope_bn),
        ),
        ("fuel_by_decile", sweep_fuel_by_decile(base, scenario)),
    ):
        dest = OUT / f"{name}.csv"
        frame.to_csv(dest, index=False)
        print(f"\nwrote {dest} ({len(frame)} rows)")


if __name__ == "__main__":
    main()

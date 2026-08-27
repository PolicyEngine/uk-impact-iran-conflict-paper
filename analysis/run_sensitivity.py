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

2. ``cap_lag.csv`` — the wholesale-to-retail lag sweep, and the **headline
   reconciliation**. ``CAP_LAG_QUARTERS`` is the paper's most contestable
   modelling choice, and before the round-2 revision the headline and this
   appendix reported the same specification as £304 and £205 because they
   applied unrelated arithmetic over unrelated windows. Both now derive the
   phase-in profile from the lag through the single function
   ``scenarios.cap_phase_in_profile`` and average it over the single window
   ``scenarios.MODELLED_WINDOW_LABEL``, so the row at the central lag *is* the
   headline and the sweep raises ``AssertionError`` if it ever stops being.
   Each lag is run twice: with the sustained fraction re-solved to hold Ofgem's
   confirmed October-2026 cap anchor, and with it held fixed so the anchor
   breaks.

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

#: Exact dataset commit the paper's results are produced from.
#:
#: Round-2 referee 3: ``hf_hub_download`` was called with no ``revision=``, so
#: it resolved the repository's ``main`` branch at whatever it happened to point
#: to on the day. The *model* (``policyengine-uk``) is pinned exactly in
#: ``pyproject.toml`` and the *data* it consumes was not, which is the harder
#: half of replication to get right and the easier half to get wrong: a
#: recalibration of the enhanced FRS moves every number in the paper with no
#: diff anywhere in this repository. Pinned to the commit the published results
#: were produced from. To move to a newer release, change this line, re-run
#: every pipeline, and say so.
DATASET_REVISION = "f0baa351d94f666e3b4b8f4e270ade45f02a89cc"

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

#: The shock's onset month and the modelled window both live in
#: :mod:`uk_iran_conflict.scenarios` now; sweep 2 reads them from there rather
#: than keeping a second calendar that could drift out of step with the
#: headline. That drift is exactly what round-2 finding 1 was.


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
    return hf_hub_download(
        DATASET_REPO,
        DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
        token=token,
    )


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


#: Lags swept in sweep 2, in quarters. 1.5 is the central case
#: (``scenarios.CAP_LAG_QUARTERS``); 3 is the pre-revision value, kept so the
#: published specification stays reproducible.
CAP_LAG_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)


def sweep_cap_lag(base: Baseline, scenario) -> pd.DataFrame:
    """Sweep the wholesale-to-retail cap lag, on the paper's own window.

    **This sweep is the round-2 reconciliation** (finding 1, and ``docs/FIXES.md``
    A2, which asked for it and did not get it). Before this revision the headline
    and this appendix computed different things and disagreed by 48%:

    * the headline damped the domestic leg by
      ``PassThroughAssumptions.annual_phase_in_gas``, the consumption-weighted
      average of a hand-written ``(0.35, 0.85, 1.00, 0.90)`` phase-in tuple that
      had **no functional relation to** ``CAP_LAG_QUARTERS`` at all;
    * this sweep slid that same tuple along a calendar and counted how much of it
      landed inside calendar 2026 — a *different window* from the one the
      headline averaged over (2026Q4-2027Q3) and a different arithmetic.

    So the two could not agree, and changing the headline from £343 to £304
    narrowed the gap without touching its cause. Both now go through one
    function of one parameter: ``scenarios.cap_phase_in_profile(lag)`` produces
    the profile, the scenario's own consumption weights average it over
    ``scenarios.MODELLED_WINDOW_LABEL``, and each row is a full
    :func:`~uk_iran_conflict.incidence.run_scenario` on that scenario. The row at
    ``lag_quarters == scenarios.CAP_LAG_QUARTERS`` **is** the headline, exactly,
    and ``reconciles_with_headline`` asserts it rather than hoping.

    Two columns per lag, because the cap anchor and the lag are not independent:

    ``anchored``
        The sustained fraction is re-solved at each lag so the modelled
        October-2026 cap still reproduces Ofgem's confirmed £1,723
        (:func:`~uk_iran_conflict.scenarios.sustained_fraction_for_cap_anchor`).
        This is the specification the paper uses. The domestic leg is nearly
        lag-invariant here — and that near-invariance is now a real result about
        an externally anchored cap, not the identity
        ``cumulative_weight = sum(profile) / 4`` that the previous version
        asserted and correctly flagged as arithmetic (``docs/FIXES.md`` D24).
    ``unanchored``
        The sustained fraction is held at the scenario's own value while the lag
        moves, so the cap anchor breaks. Reported to show how much of the
        invariance is the anchor doing the work.
    """
    rows: list[dict] = []
    anchor_index = scen.CAP_QUARTER_LABELS.index(scen.CAP_ANCHOR_QUARTER)
    headline, _ = run_scenario(base, scenario)
    print("\n=== sweep 2: wholesale-to-retail cap lag (realised_2026) ===")
    print(
        f"headline: mean £{headline.mean_loss_gbp:.2f}, "
        f"window {scen.MODELLED_WINDOW_LABEL}"
    )
    print(
        f"{'lag':>5} {'anchor':>10} {'phase-in':>9} {'sustained':>10} "
        f"{'gas %':>7} {'mean £':>8} {'£bn':>7} {'fuel %':>7} {'cap Q4 £':>9}"
    )
    for lag in CAP_LAG_GRID:
        profile = scen.cap_phase_in_profile(lag)
        for anchored in (True, False):
            if anchored:
                sustained = scen.sustained_fraction_for_cap_anchor(
                    scenario.gas.pct_change,
                    profile[anchor_index],
                    wholesale_share_gas_bill=(
                        scenario.pass_through.wholesale_share_gas_bill
                    ),
                    wholesale_share_electricity_bill=(
                        scenario.pass_through.wholesale_share_electricity_bill
                    ),
                    marginal_pricing_share=(
                        scenario.pass_through.marginal_pricing_share
                    ),
                    gas_share_of_dual_fuel_bill=(
                        scenario.pass_through.gas_share_of_dual_fuel_bill
                    ),
                )
            else:
                sustained = scenario.pass_through.sustained_fraction
            variant = dataclasses.replace(
                scenario,
                pass_through=dataclasses.replace(
                    scenario.pass_through,
                    lag_quarters=lag,
                    phase_in_profile=profile,
                    sustained_fraction=min(1.0, sustained),
                ),
            )
            result, cost = run_scenario(base, variant)
            w = base.weight
            anchor_step = variant.cap_step(scen.CAP_ANCHOR_QUARTER)
            is_central = anchored and abs(lag - scen.CAP_LAG_QUARTERS) < 1e-9
            row = {
                "lag_quarters": lag,
                "anchor": "anchored" if anchored else "unanchored",
                "phase_in_profile": ";".join(f"{v:.4f}" for v in profile),
                "phase_in_at_anchor_quarter": profile[anchor_index],
                "annual_phase_in_gas": variant.pass_through.annual_phase_in_gas,
                "annual_phase_in_electricity": (
                    variant.pass_through.annual_phase_in_electricity
                ),
                "sustained_fraction": variant.pass_through.sustained_fraction,
                "anchor_product": (
                    variant.pass_through.sustained_fraction * profile[anchor_index]
                ),
                "retail_gas_pct_change": (
                    100 * variant.annual_retail_shock.gas_pct_change
                ),
                "cap_anchor_quarter_gbp": anchor_step.cap_gbp,
                "cap_anchor_quarter_pct": 100 * anchor_step.cap_pct_change,
                "mean_loss_gbp": result.mean_loss_gbp,
                "mean_loss_pct": result.mean_loss_pct,
                "aggregate_loss_bn": result.aggregate_cost_bn,
                "domestic_loss_bn": wsum(cost.domestic, w) / 1e9,
                "motor_fuel_loss_bn": wsum(cost.motor_fuel, w) / 1e9,
                "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
                "decile1_loss_pct": result.decile[0].mean_loss_pct,
                "decile10_loss_pct": result.decile[-1].mean_loss_pct,
                "is_central_specification": is_central,
                "headline_mean_loss_gbp": headline.mean_loss_gbp,
                "reconciles_with_headline": (
                    abs(result.mean_loss_gbp - headline.mean_loss_gbp) < 1e-6
                    if is_central
                    else None
                ),
            }
            if is_central and not row["reconciles_with_headline"]:
                raise AssertionError(
                    "cap-lag sweep no longer reproduces the headline at the "
                    f"central lag: £{result.mean_loss_gbp:.4f} against "
                    f"£{headline.mean_loss_gbp:.4f}. The headline and this "
                    "appendix must be the same function of the same parameter."
                )
            rows.append(row)
            print(
                f"{lag:5.1f} {row['anchor']:>10} "
                f"{row['annual_phase_in_gas']:9.4f} "
                f"{row['sustained_fraction']:10.4f} "
                f"{row['retail_gas_pct_change']:7.2f} "
                f"{row['mean_loss_gbp']:8.2f} {row['aggregate_loss_bn']:7.2f} "
                f"{100 * row['motor_fuel_share_of_loss']:6.1f}% "
                f"{row['cap_anchor_quarter_gbp']:9.0f}"
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
        f"{'policy':16} {'envelope':>18} {'£bn':>6} {'param':>9} {'max':>8} "
        f"{'ok':>3} {'D1-3':>6} {'offset':>7} {'mean res':>9} {'£/£ D1':>7}"
    )
    for score in pol.scorecard(base, cost, mt, envelope_bn=envelope_bn):
        row = {
            "scenario": getattr(scenario, "key", ""),
            "policy": score.policy,
            "label": score.label,
            "envelope": score.envelope,
            "envelope_bn": score.envelope_bn,
            "envelope_scale": score.envelope_scale,
            # Round-2 finding 2: the implied parameter is now on every scaled
            # row, so a 138% bill discount or a negative VAT rate is visible in
            # the CSV instead of hiding inside a scale factor.
            "label_used": score.label_used,
            "parameter": score.parameter,
            "parameter_units": score.parameter_units,
            "stated_parameter": score.stated_parameter,
            "implied_parameter": score.implied_parameter,
            "feasible_max_parameter": score.feasible_max_parameter,
            "is_feasible": score.is_feasible,
            "absorbable_envelope_bn": score.absorbable_envelope_bn,
            "eligible_share": score.eligible_share,
            "cost_per_pound_decile_one_units": (score.cost_per_pound_decile_one_units),
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
            f"{score.policy:16} {score.envelope:>18} {score.cost_bn:6.2f} "
            f"{score.implied_parameter:9.2f} {score.feasible_max_parameter:8.2f} "
            f"{'y' if score.is_feasible else 'NO':>3} "
            f"{100 * score.share_to_bottom_three:5.1f}% "
            f"{100 * score.share_of_aggregate_loss_offset:6.1f}% "
            f"{score.mean_residual_loss_gbp:9.0f} "
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

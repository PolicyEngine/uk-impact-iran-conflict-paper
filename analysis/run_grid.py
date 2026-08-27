#!/usr/bin/env python3
"""Sweep the two independent macro drivers of the shock and score every cell.

The paper's three named scenarios are single points in a two-dimensional space.
The shock has exactly two genuinely independent drivers:

* **wholesale gas**, which reaches households through the Ofgem default tariff
  cap and therefore through ``gas_consumption`` and ``electricity_consumption``;
* **crude oil**, which reaches households at the pump through ``petrol_spending``
  and ``diesel_spending``.

They are independent because the UK gas price is an NBP/LNG story and the pump
price is a Brent/refining story; a Hormuz closure moves both, but a European
storage or Norwegian-outage shock moves only the first and an OPEC+ decision
only the second. Sweeping them against each other shows which combinations make
this a pump-price event and which make it a domestic-bill event — the question
raised by the headline finding that motor fuel is 67.8% of the loss
(``docs/FINDINGS.md`` §2).

Every cell is built with the existing :mod:`uk_iran_conflict.scenarios`
machinery (``PassThroughAssumptions`` -> ``retail_shock`` -> ``cap_path``) and
scored with :func:`uk_iran_conflict.incidence.run_scenario` against a **single**
loaded :class:`~uk_iran_conflict.incidence.Baseline`; the microdata load is the
slow step and happens once.

Usage::

    python analysis/run_grid.py               # 6x6 grid, period 2026
    python analysis/run_grid.py --steps 7

Needs the private PolicyEngine UK microdata exactly as ``run_incidence.py``
does: ``HUGGING_FACE_TOKEN`` in the gitignored ``.env``.

Writes ``results/grid/grid.csv``, one row per cell.

Degenerate cells (docs/FIXES.md E35). The ``(0%, 0%)`` corner has no shock at
all: every fuel share is 0/0 and the decile ratio is an empty division. It is
kept in the CSV for completeness but flagged ``is_degenerate`` and written as
NaN, and consumers must filter on that column. The two *axes* are not
degenerate — a cell with a gas move and no oil move really is 0% motor fuel,
and one with an oil move and no gas move really is 100% — so those cells are
documented here rather than filtered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # works whether run as `python analysis/run_grid.py` or `-m analysis.run_grid`
    from run_incidence import _load_env, dataset_path
except ImportError:  # pragma: no cover
    from analysis.run_incidence import _load_env, dataset_path

from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import (
    decile_table,
    load_baseline,
    run_scenario,
    wmean,
    wsum,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "grid"

#: Oil-to-pump pass-through, fitted to the three named scenarios rather than
#: invented: each of them sets petrol at ~0.35x and diesel at ~0.55-0.63x the
#: proportional oil move (realised 2026: +57% oil -> +20% petrol, +36% diesel;
#: NIESR baseline: +39.8% -> +14%/+22%; NIESR adverse: +90% -> +32%/+50%).
#: The means across the three are used, so each named scenario's own pump path
#: is reproduced to within a couple of percentage points at its own oil move.
#: Diesel passes through harder than petrol because the distillate market is the
#: one exposed to Gulf refining and Hormuz shipping.
PETROL_PASS_THROUGH: float = 0.353
DIESEL_PASS_THROUGH: float = 0.580

#: The grid is specified as a **sustained** wholesale move, so pass-through is
#: full (``sustained_fraction = 1.0``), matching both NIESR scenarios.
#:
#: This is why :func:`named_points` reports **damped-equivalent** coordinates
#: as well as headline ones (docs/FIXES.md D25). The realised 2026 scenario
#: damps its peak gas move to 0.36 and its peak pump moves to their own
#: sustained fraction; plotting it at its *headline* +217% gas / +57% oil put
#: it in a cell far more severe than the scenario it is labelled with — and by
#: the grid's own numbers that cell sits ABOVE the motor-fuel frontier, while
#: the paper claims all three named scenarios sit below it. The figure and the
#: CSV contradicted each other. The damped-equivalent coordinates are the
#: (gas %, oil %) a *sustained* cell would need in order to deliver the
#: scenario's own retail and pump moves, so a named point now lands on the cell
#: that actually reproduces it.
SUSTAINED_FRACTION: float = 1.0


def build_scenario(gas_pct: float, oil_pct: float) -> scen.Scenario:
    """A grid cell as a real :class:`~uk_iran_conflict.scenarios.Scenario`.

    ``gas_pct`` and ``oil_pct`` are proportional moves versus the pre-war
    reference (0.40 == +40%). The cap-path derivation is *not* reimplemented
    here: it comes free with the ``Scenario``.
    """
    return scen.Scenario(
        key=f"grid_gas{gas_pct * 100:.0f}_oil{oil_pct * 100:.0f}",
        label=f"gas +{gas_pct * 100:.0f}%, oil +{oil_pct * 100:.0f}%",
        description=(
            "Synthetic scenario-grid cell: a sustained wholesale gas move of "
            f"+{gas_pct * 100:.0f}% and a Brent move of +{oil_pct * 100:.0f}% "
            "versus the pre-war reference."
        ),
        oil=scen.OilPath(
            level_usd_per_bbl=scen.PREWAR_BRENT_USD_PER_BBL * (1.0 + oil_pct)
        ),
        gas=scen.GasPath(
            change_pence_per_therm=scen.PREWAR_NBP_PENCE_PER_THERM * gas_pct
        ),
        pump=scen.PumpPricePath(
            petrol_pct_change=PETROL_PASS_THROUGH * oil_pct,
            diesel_pct_change=DIESEL_PASS_THROUGH * oil_pct,
        ),
        pass_through=scen.PassThroughAssumptions(sustained_fraction=SUSTAINED_FRACTION),
        source=(
            "Synthetic sweep around the named scenarios; pre-war references and "
            "pass-through assumptions as in uk_iran_conflict.scenarios."
        ),
        notes=(
            "Pump prices are mapped from the oil move with the petrol/diesel "
            f"pass-through coefficients {PETROL_PASS_THROUGH}/"
            f"{DIESEL_PASS_THROUGH}, fitted to the three named scenarios."
        ),
    )


def _damped_equivalent(s: scen.Scenario) -> dict[str, float]:
    """The sustained (gas %, oil %) cell that reproduces ``s``'s own shock.

    A grid cell runs undamped, so a scenario that damps its wholesale move
    belongs at the *smaller* move that, run undamped, delivers the same retail
    and pump prices:

    * **gas** — the cap channel is linear in
      ``sustained_fraction x gas.pct_change``, so the equivalent sustained gas
      move is exactly that product.
    * **oil** — invert the grid's own fitted pump pass-through on the scenario's
      *damped* pump factors (from
      :func:`uk_iran_conflict.incidence.sustained_pump_factors`, which applies
      ``pump_sustained_fraction``). Petrol and diesel give two implied oil
      moves; their mean is reported, along with both legs, since the fitted
      coefficients reproduce each named scenario only to within a point or two.

    Coded defensively against the sibling change to ``shock_cost``: this reads
    only the scenario's own pass-through fields and the public
    ``sustained_pump_factors`` helper, never the annualised cost.
    """
    from uk_iran_conflict.incidence import sustained_pump_factors  # noqa: PLC0415

    pt = getattr(s, "pass_through", None)
    gas_fraction = float(getattr(pt, "sustained_fraction", 1.0))
    petrol_factor, diesel_factor = sustained_pump_factors(s)
    oil_from_petrol = (petrol_factor - 1.0) / PETROL_PASS_THROUGH
    oil_from_diesel = (diesel_factor - 1.0) / DIESEL_PASS_THROUGH
    return {
        "gas_pct_damped": 100 * gas_fraction * s.gas.pct_change,
        "oil_pct_damped": 100 * 0.5 * (oil_from_petrol + oil_from_diesel),
        "oil_pct_damped_from_petrol": 100 * oil_from_petrol,
        "oil_pct_damped_from_diesel": 100 * oil_from_diesel,
        "gas_sustained_fraction": gas_fraction,
        "pump_sustained_fraction": float(getattr(pt, "pump_sustained_fraction", 1.0)),
        "damped_petrol_pct_change": 100 * (petrol_factor - 1.0),
        "damped_diesel_pct_change": 100 * (diesel_factor - 1.0),
    }


def named_points() -> pd.DataFrame:
    """Where the three named scenarios sit in (gas %, oil %) space.

    Two coordinate pairs per scenario. ``gas_pct``/``oil_pct`` are the headline
    wholesale moves; ``gas_pct_damped``/``oil_pct_damped`` are the
    damped-equivalent coordinates described in :func:`_damped_equivalent`.
    **Plot the damped pair** — every grid cell is undamped, so only the damped
    pair puts a scenario on the cell that reproduces it (docs/FIXES.md D25).
    """
    rows = []
    for key, s in scen.SCENARIOS.items():
        rows.append(
            {
                "scenario": key,
                "label": s.label,
                "gas_pct": 100 * s.gas.pct_change,
                "oil_pct": 100 * s.oil.pct_change,
                **_damped_equivalent(s),
            }
        )
    return pd.DataFrame(rows)


def cell_row(base, gas_pct: float, oil_pct: float) -> dict:
    """Score one cell off the already-loaded baseline."""
    scenario = build_scenario(gas_pct, oil_pct)
    result, cost = run_scenario(base, scenario)
    w = base.weight
    d1 = result.decile[0]
    d10 = result.decile[-1]
    peak_cap = scenario.peak_cap_gbp
    total = cost.total
    losers = total > 0
    # E35: the (0%, 0%) corner has no shock at all, so every share is 0/0 and
    # the burden ratio is an empty division. Left in the CSV for completeness
    # but flagged and NaN-ed, rather than plotted as a spurious 0%/100% fuel
    # split. Any consumer of this file must filter on ``is_degenerate``.
    degenerate = not bool(np.any(losers)) or wsum(total, w) <= 0

    def _nan_if_degenerate(value: float) -> float:
        return float("nan") if degenerate else value

    return {
        "is_degenerate": degenerate,
        "gas_pct": round(100 * gas_pct, 4),
        "oil_pct": round(100 * oil_pct, 4),
        "gas_change_pence_per_therm": scenario.gas.change_pence_per_therm,
        "brent_usd_per_bbl": scenario.oil.level_usd_per_bbl,
        "petrol_pct_change": 100 * scenario.pump.petrol_pct_change,
        "diesel_pct_change": 100 * scenario.pump.diesel_pct_change,
        "retail_gas_pct_change": 100 * scenario.retail_shock.gas_pct_change,
        "retail_electricity_pct_change": (
            100 * scenario.retail_shock.electricity_pct_change
        ),
        "peak_cap_gbp": peak_cap,
        "peak_cap_pct_change": 100 * (peak_cap / scenario.baseline_cap_gbp - 1.0),
        "aggregate_cost_bn": result.aggregate_cost_bn,
        "mean_loss_gbp": result.mean_loss_gbp,
        "mean_loss_pct": result.mean_loss_pct,
        "motor_fuel_share_pct": _nan_if_degenerate(
            100 * result.motor_fuel_share_of_loss
        ),
        "gas_share_pct": _nan_if_degenerate(100 * result.gas_share_of_loss),
        "electricity_share_pct": _nan_if_degenerate(
            100 * result.electricity_share_of_loss
        ),
        "domestic_share_pct": _nan_if_degenerate(
            100 * (result.gas_share_of_loss + result.electricity_share_of_loss)
        ),
        "decile1_loss_pct": d1.mean_loss_pct,
        "decile10_loss_pct": d10.mean_loss_pct,
        "d1_d10_ratio": _nan_if_degenerate(
            d1.mean_loss_pct / d10.mean_loss_pct if d10.mean_loss_pct else float("nan")
        ),
        # Round-2 finding 7: the grid's decile ratios were reported (11.77-12.73)
        # outside the range of every specification in the paper (4.41-9.25), with
        # no reconciliation. The ratio is scale-invariant — ``mean_loss_pct`` is
        # an aggregate ratio — so an undamped cell cannot move it, and a
        # weighted sum of two channels must lie between the two channels' own
        # ratios. The published grid therefore could not be reconciled with the
        # published headline because it was a stale artefact of the pre-D1
        # unequivalised denominator. Both channel ratios are now emitted beside
        # the total so the bracketing is checkable cell by cell, and
        # ``reconcile_named_scenarios`` asserts it.
        "d1_d10_ratio_domestic_only": _nan_if_degenerate(
            result.domestic_only_d1_d10_ratio_pct
        ),
        "domestic_share_of_loss": _nan_if_degenerate(
            result.gas_share_of_loss + result.electricity_share_of_loss
        ),
        "decile1_loss_gbp": d1.mean_loss_gbp,
        "decile10_loss_gbp": d10.mean_loss_gbp,
        "share_losing_over_5pct": float(
            w[losers][
                (total[losers] / np.clip(base.net_income[losers], 1, None)) > 0.05
            ].sum()
            / w.sum()
        ),
        "gini_after": result.gini_after,
        "gini_change_pp": 100 * (result.gini_after - result.gini_baseline),
    }


#: The four sub-channels every shock in this paper is built from. Each one's
#: D1/D10 burden ratio is a fixed property of the *baseline* — ``mean_loss_pct``
#: is an aggregate ratio, so scaling a channel's price move cannot change its own
#: decile gradient — and every scenario's all-channel ratio is an exact convex
#: combination of the four, weighted by each channel's share of decile ten's
#: loss. That is what makes the check below able to fail.
SUB_CHANNELS: tuple[str, ...] = ("gas", "electricity", "petrol", "diesel")


def _sub_channel_costs(base, scenario) -> dict[str, np.ndarray]:
    """Per-household cost of each of the four sub-channels, £/yr."""
    from uk_iran_conflict.incidence import (  # noqa: PLC0415
        domestic_retail_factors,
        sustained_pump_factors,
    )

    gas_factor, elec_factor = domestic_retail_factors(scenario)
    petrol_factor, diesel_factor = sustained_pump_factors(scenario)
    return {
        "gas": base.gas * (gas_factor - 1.0),
        "electricity": base.electricity * (elec_factor - 1.0),
        "petrol": base.petrol * (petrol_factor - 1.0),
        "diesel": base.diesel * (diesel_factor - 1.0),
    }


def _sub_channel_decomposition(base, scenario, ratio: float) -> dict:
    """Decompose a scenario's D1/D10 ratio onto the four sub-channels.

    **Round-4 finding 5.** The two-channel identity in
    :func:`reconcile_named_scenarios` holds by construction and is worth
    asserting, but it cannot explain why every named scenario sits *outside* the
    grid's range. This does.

    The grid varies exactly two things — the gas move and the oil move — and
    maps the oil move to the pump with **fixed** coefficients
    (:data:`PETROL_PASS_THROUGH`, :data:`DIESEL_PASS_THROUGH`). So across all
    thirty-six cells the gas:electricity mix and the petrol:diesel mix are both
    held constant, and only the domestic-versus-fuel mix moves. Petrol and
    diesel have materially different decile gradients in this imputation, and
    the named scenarios each carry their own petrol/diesel mix, so their
    all-channel ratios land off the one-dimensional curve the grid traces. The
    named scenarios are not outside the grid because anything is inconsistent;
    they are outside it because the grid does not sweep the margin that
    separates them from it.

    Returns each sub-channel's own ratio, its share of decile-ten loss, the
    convex combination they imply, and ``inside_sub_channel_span`` — a **real**
    bracketing test: an aggregate ratio that is not between the smallest and
    largest of its own components' ratios is arithmetically impossible and means
    the pipeline is broken.
    """
    w = base.weight
    d10 = base.decile == 10
    costs = _sub_channel_costs(base, scenario)
    ratios = {name: _decile_ratio(decile_table(base, c)) for name, c in costs.items()}
    d10_loss = {name: wsum(c[d10], w[d10]) for name, c in costs.items()}
    total_d10 = sum(d10_loss.values())
    shares = {
        name: (value / total_d10 if total_d10 else float("nan"))
        for name, value in d10_loss.items()
    }
    implied = sum(shares[name] * ratios[name] for name in SUB_CHANNELS)
    finite = [ratios[name] for name in SUB_CHANNELS if np.isfinite(ratios[name])]
    lo = min(finite) if finite else float("nan")
    hi = max(finite) if finite else float("nan")
    return {
        "channels": list(SUB_CHANNELS),
        "d1_d10_ratio_by_channel": ratios,
        "share_of_decile10_loss_by_channel": shares,
        "d1_d10_ratio_implied": implied,
        "identity_residual": abs(ratio - implied),
        "sub_channel_ratio_min": lo,
        "sub_channel_ratio_max": hi,
        "inside_sub_channel_span": bool(
            np.isfinite(lo) and np.isfinite(hi) and lo - 1e-9 <= ratio <= hi + 1e-9
        ),
        "petrol_share_of_motor_fuel_loss": (
            d10_loss["petrol"] / (d10_loss["petrol"] + d10_loss["diesel"])
            if (d10_loss["petrol"] + d10_loss["diesel"])
            else float("nan")
        ),
    }


def _grid_scope(live: pd.DataFrame, spread: float, informative: float) -> dict:
    """What the grid does and does not show, stated plainly. Round-4 finding 5.

    The appendix advertised that this script "checks every named scenario
    against the live grid's range and raises if one falls outside it". It never
    did: range membership was recorded and never enforced, and
    ``inside_grid_range`` is in fact ``false`` for all five named scenarios
    while ``check_is_informative`` is ``false`` too. A guarantee that does not
    run must not be advertised, and this block exists so the paper cannot
    advertise it: ``range_check_enforced`` is ``false``, permanently and by
    design, with the reason attached.

    What *is* enforced is the sub-channel bracketing and the convex-combination
    identity, both of which can fail and neither of which is a range-membership
    test.
    """
    petrol_diesel_ratio = DIESEL_PASS_THROUGH / PETROL_PASS_THROUGH
    return {
        "range_check_enforced": False,
        "why_the_range_check_is_not_enforced": (
            "It carries no information on this grid. The D1/D10 burden ratio "
            f"spans {spread:.6f} across all live cells, against the "
            f"{informative:.2f} it would need for membership to discriminate "
            "between pipelines. A test that every value passes is not a test, "
            "and enforcing it would give the paper a guarantee it does not "
            "have. The paper must not claim this script raises on range "
            "membership: it does not, and it should not."
        ),
        "what_the_grid_shows": [
            "the aggregate cost, mean loss and channel split of the shock "
            "across a 6x6 sweep of the two independent wholesale drivers",
            "that the modelled D1/D10 burden gradient is INVARIANT to the "
            "domestic-versus-motor-fuel mix, because in this imputation the "
            "two channels have almost the same decile gradient — which is the "
            "motor-fuel imputation defect seen from a third direction",
            "where in (gas, oil) space motor fuel stops being the majority of the loss",
        ],
        "what_the_grid_does_not_show": [
            "any test of mix-invariance with power: the range is degenerate, "
            "so invariance is measured, not tested, and the paper may cite the "
            "measured spread but not a passed test",
            "the petrol-versus-diesel margin. Every cell maps the oil move to "
            f"the pump with the SAME fixed coefficients ({PETROL_PASS_THROUGH} "
            f"petrol, {DIESEL_PASS_THROUGH} diesel, ratio "
            f"{petrol_diesel_ratio:.3f}), so the petrol:diesel mix is constant "
            "across all cells. The named scenarios carry their own mixes, "
            "which is why their ratios fall outside the grid's range — not an "
            "inconsistency, a margin the grid does not sweep",
            "the gas-versus-electricity margin, which is likewise fixed by the "
            "cap pass-through assumptions in every cell",
            "anything about damped scenarios at their headline coordinates: "
            "every cell is undamped, which is why named_points reports "
            "damped-equivalent coordinates",
        ],
        "enforced_checks": [
            "the convex-combination identity on the domestic/motor-fuel split, "
            "at 1e-6 — raises",
            "sub-channel bracketing: every named scenario's all-channel ratio "
            "must lie between the smallest and largest of its own four "
            "sub-channel ratios — raises",
        ],
        "live_cells": int(len(live)),
    }


def reconcile_named_scenarios(base, live: pd.DataFrame) -> dict:
    """Reconcile each named scenario's decile ratio with the grid, meaningfully.

    **Round-3 finding 5: the previous check could not fail.** It asserted that
    every named scenario's D1/D10 burden ratio lay within ±5% of the live grid's
    range. Across all thirty-six cells that range was 9.311577 to 9.312363 — a
    spread of 0.0008 — because the two *pure-channel* ratios are themselves
    9.3116 and 9.3124. The ratio is a convex combination of those two, so every
    convex combination lands inside a band five per cent wide by arithmetic, and
    the check passed for reasons that had nothing to do with the pipeline.

    Two things replace it.

    **An honest statement about the grid.** The grid's near-degeneracy is a
    *result*, not a nuisance: it says the modelled decile gradient is invariant
    to the channel mix, because in this imputation the domestic and motor-fuel
    channels have almost the same decile gradient. That, in turn, is the
    motor-fuel imputation defect showing up from a third direction — a fuel
    channel whose gradient matches the domestic channel's is a fuel channel that
    has not reproduced car ownership. ``grid_shows_invariance`` and the measured
    spread say so directly, and ``check_is_informative`` records that a
    range-membership test on this grid carries no information.

    **A reconciliation that can actually fail.** The burden ratio decomposes
    exactly. Writing ``a`` for the share of decile ten's loss that is domestic::

        ratio_all = a x ratio_domestic + (1 - a) x ratio_fuel

    That is an identity in the pipeline's own arithmetic, so it holds to
    floating-point if the grid and the named scenarios come from the same code
    and fails immediately if they do not — which is what the check was supposed
    to be testing. It is asserted at 1e-6, not 5%.
    """
    lo = float(live["d1_d10_ratio"].min())
    hi = float(live["d1_d10_ratio"].max())
    spread = hi - lo
    # Below this the grid is degenerate and a range-membership test is vacuous.
    informative_spread = 0.05 * max(abs(lo), 1e-12)
    identity_tol = 1e-6
    scenarios: dict[str, dict] = {}
    for key, scenario in scen.SCENARIOS.items():
        result, cost = run_scenario(base, scenario)
        w = base.weight
        d10 = base.decile == 10
        domestic_d10 = wsum(cost.domestic[d10], w[d10])
        fuel_d10 = wsum(cost.motor_fuel[d10], w[d10])
        total_d10 = domestic_d10 + fuel_d10
        share = domestic_d10 / total_d10 if total_d10 else float("nan")
        fuel_rows = decile_table(base, cost.motor_fuel)
        fuel_ratio = _decile_ratio(fuel_rows)
        ratio = result.all_channel_d1_d10_ratio_pct
        domestic_ratio = result.domestic_only_d1_d10_ratio_pct
        implied = share * domestic_ratio + (1.0 - share) * fuel_ratio
        sub = _sub_channel_decomposition(base, scenario, ratio)
        scenarios[key] = {
            "d1_d10_ratio": ratio,
            "d1_d10_ratio_domestic_only": domestic_ratio,
            "d1_d10_ratio_motor_fuel_only": fuel_ratio,
            "domestic_share_of_decile10_loss": share,
            "d1_d10_ratio_implied_by_channel_mix": implied,
            "identity_residual": abs(ratio - implied),
            "identity_holds": bool(abs(ratio - implied) < identity_tol),
            "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
            "inside_grid_range": bool(lo <= ratio <= hi),
            # Round-4 finding 5: WHY it falls outside, decomposed on the four
            # sub-channels the grid holds partly fixed.
            "sub_channels": sub,
        }
    broken = [k for k, v in scenarios.items() if not v["identity_holds"]]
    unbracketed = [
        k
        for k, v in scenarios.items()
        if not v["sub_channels"]["inside_sub_channel_span"]
    ]
    outside_range = [k for k, v in scenarios.items() if not v["inside_grid_range"]]
    return {
        "note": (
            "Round-3 finding 5. The previous ±5% range-membership check could "
            "not fail: the grid's D1/D10 ratio spans 0.0008 across 36 cells "
            "because both pure-channel ratios are ~9.31 and every mix is a "
            "convex combination of them. Replaced by (a) an explicit statement "
            "that the grid shows INVARIANCE of the decile gradient to the "
            "channel mix, which is itself evidence about the fuel imputation, "
            "and (b) the exact convex-combination identity, asserted at 1e-6."
        ),
        "grid_d1_d10_ratio_min": lo,
        "grid_d1_d10_ratio_max": hi,
        "grid_d1_d10_ratio_spread": spread,
        "grid_shows_invariance": bool(spread < informative_spread),
        "check_is_informative": bool(spread >= informative_spread),
        "informative_spread_threshold": informative_spread,
        "identity_tolerance": identity_tol,
        "scenarios": scenarios,
        "all_named_scenarios_inside_grid_range": all(
            v["inside_grid_range"] for v in scenarios.values()
        ),
        "channel_mix_identity_holds": not broken,
        "identity_broken": broken,
        "outside": broken,
        # --- round-4 finding 5 -------------------------------------------
        "grid_scope": _grid_scope(live, spread, informative_spread),
        "named_scenarios_outside_grid_range": outside_range,
        "why_named_scenarios_fall_outside_grid_range": (
            "Not an inconsistency. Every grid cell maps the oil move to the "
            f"pump with the same fixed coefficients ({PETROL_PASS_THROUGH} "
            f"petrol, {DIESEL_PASS_THROUGH} diesel), so the petrol:diesel mix "
            "is identical in all thirty-six cells, and the cap pass-through "
            "fixes the gas:electricity mix likewise. The grid therefore traces "
            "a one-dimensional curve through a four-channel space. The named "
            "scenarios carry their own petrol/diesel mixes and sit off that "
            "curve. See each scenario's `sub_channels` block, and "
            "`grid_scope.what_the_grid_does_not_show`."
        ),
        "sub_channel_bracketing_holds": not unbracketed,
        "sub_channel_bracketing_broken": unbracketed,
    }


def _decile_ratio(rows) -> float:
    """Decile-one over decile-ten ``mean_loss_pct`` from a decile table."""
    top = next((r for r in rows if r.decile == 10), None)
    bottom = next((r for r in rows if r.decile == 1), None)
    if top is None or bottom is None or not top.mean_loss_pct:
        return float("nan")
    return float(bottom.mean_loss_pct / top.mean_loss_pct)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument(
        "--steps", type=int, default=6, help="points per axis (6 -> 36 cells)"
    )
    parser.add_argument(
        "--max-pct", type=float, default=100.0, help="top of both axes, in %"
    )
    args = parser.parse_args()

    _load_env()
    path = dataset_path()
    print(f"dataset: {path}")

    base = load_baseline(path, args.period)
    print(f"baseline: {base.n:,} households, {base.weight.sum() / 1e6:.1f}m weighted")
    print(
        f"baseline spend: domestic £{wsum(base.energy, base.weight) / 1e9:.1f}bn, "
        f"motor fuel £{wsum(base.motor_fuel, base.weight) / 1e9:.1f}bn "
        f"(mean £{wmean(base.motor_fuel, base.weight):.0f})"
    )

    axis = np.linspace(0.0, args.max_pct / 100.0, args.steps)
    rows = [cell_row(base, gas_pct, oil_pct) for oil_pct in axis for gas_pct in axis]
    df = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "grid.csv", index=False)
    named_points().to_csv(OUT / "named_points.csv", index=False)
    print(f"\nwrote {OUT / 'grid.csv'} ({len(df)} cells)")

    live = df[~df["is_degenerate"]]
    for col, fmt in [
        ("mean_loss_gbp", "£{:.0f}"),
        ("aggregate_cost_bn", "£{:.1f}bn"),
        ("d1_d10_ratio", "{:.2f}x"),
        ("motor_fuel_share_pct", "{:.1f}%"),
    ]:
        lo, hi = live[col].min(), live[col].max()
        print(f"  {col:22} {fmt.format(lo)} to {fmt.format(hi)}")

    reconciliation = reconcile_named_scenarios(base, live)
    (OUT / "reconciliation.json").write_text(json.dumps(reconciliation, indent=2))
    print(
        f"\ngrid D1/D10 ratio spans {reconciliation['grid_d1_d10_ratio_min']:.4f}x "
        f"to {reconciliation['grid_d1_d10_ratio_max']:.4f}x "
        f"(spread {reconciliation['grid_d1_d10_ratio_spread']:.6f}); "
        + (
            "the grid shows INVARIANCE of the decile gradient to the channel "
            "mix, so a range-membership test on it carries no information"
            if reconciliation["grid_shows_invariance"]
            else "the range is wide enough for membership to be informative"
        )
    )
    for key, payload in reconciliation["scenarios"].items():
        print(
            f"  {key:26} {payload['d1_d10_ratio']:7.4f}x "
            f"(domestic {payload['d1_d10_ratio_domestic_only']:7.4f}x, fuel "
            f"{payload['d1_d10_ratio_motor_fuel_only']:7.4f}x) "
            f"channel-mix identity residual {payload['identity_residual']:.2e} "
            f"{'ok' if payload['identity_holds'] else 'BROKEN'}"
        )
    if reconciliation["sub_channel_bracketing_broken"]:
        raise SystemExit(
            "the all-channel D1/D10 ratio falls outside the span of its own "
            "four sub-channel ratios for "
            f"{reconciliation['sub_channel_bracketing_broken']}. That is "
            "arithmetically impossible for a weighted sum of those channels, "
            "so the decile table and the shock cost are not being computed off "
            "the same baseline."
        )
    if reconciliation["identity_broken"]:
        raise SystemExit(
            "the decile ratio is not the convex combination of its own channel "
            f"ratios for {reconciliation['identity_broken']}; the grid and the "
            "named scenarios are not the same pipeline. Re-run "
            "analysis/run_variants.py and analysis/run_grid.py together."
        )

    scope = reconciliation["grid_scope"]
    print(
        "\nrange check: "
        + ("ENFORCED" if scope["range_check_enforced"] else "NOT enforced, by design")
        + f" — {len(reconciliation['named_scenarios_outside_grid_range'])} of "
        f"{len(reconciliation['scenarios'])} named scenarios fall outside the "
        "grid's range, because the grid holds the petrol:diesel and "
        "gas:electricity mixes fixed in every cell"
    )
    print(
        "  enforced instead: sub-channel bracketing "
        + ("ok" if reconciliation["sub_channel_bracketing_holds"] else "BROKEN")
        + ", convex-combination identity "
        + ("ok" if reconciliation["channel_mix_identity_holds"] else "BROKEN")
    )

    flip = live[live["motor_fuel_share_pct"] < 50]
    print(
        "\nmotor fuel stops dominating in "
        f"{len(flip)} of {len(live)} live cells; "
        + (
            f"smallest gas move that flips a cell: +{flip['gas_pct'].min():.0f}% gas"
            if len(flip)
            else "never within the swept range"
        )
    )


if __name__ == "__main__":
    main()

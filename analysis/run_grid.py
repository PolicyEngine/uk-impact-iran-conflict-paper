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
from pathlib import Path

import numpy as np
import pandas as pd

try:  # works whether run as `python analysis/run_grid.py` or `-m analysis.run_grid`
    from run_incidence import _load_env, dataset_path
except ImportError:  # pragma: no cover
    from analysis.run_incidence import _load_env, dataset_path

from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import load_baseline, run_scenario, wmean, wsum

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

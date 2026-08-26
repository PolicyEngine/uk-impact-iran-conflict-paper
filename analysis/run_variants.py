#!/usr/bin/env python3
"""Run the three realised-2026 specifications the validation audit requires.

Usage::

    python analysis/run_variants.py
    python analysis/run_variants.py --period 2026

``docs/VALIDATION.md`` raises two defects in the realised-2026 results. This
script produces the corrected main specification and the two runs that bound it,
on a **single** load of the microdata (loading it is the slow step):

(a) ``realised_2026`` — the main specification. The pump-price peak is now damped
    on the same logic as the wholesale gas peak
    (``scenarios.REALISED_PUMP_SUSTAINED_FRACTION``), fixing the Check 2b
    inconsistency in which gas was damped to 0.36 while peak pump prices were
    charged undamped for twelve months. Overwrites ``results/realised_2026/``,
    which is now this specification.
(b) ``realised_2026_peak_fuel`` — the undamped-pump run, retained as an explicit
    **upper bound** on the motor-fuel channel and on its share of the loss.
    Written to ``results/realised_2026_peak_fuel/``.
(c) ``realised_2026`` scored on an **ONS-calibrated** motor-fuel baseline
    (Check 2d): household motor-fuel spend is reweighted so weighted decile means
    follow the ONS Family Spending profile, preserving the national total and
    within-decile relative variation. A robustness run only — it does not touch
    the main specification. Written to ``results/robustness/ons_fuel/``.

Also writes ``results/robustness/comparison.csv`` with the headline statistics
for all three.

Needs the PolicyEngine UK microdata, which is private: set ``HUGGING_FACE_TOKEN``
(the repo reads a gitignored ``.env``). Handling mirrors
``analysis/run_incidence.py``, which remains the script that produces the NIESR
scenarios.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.run_incidence import RESULTS, _asdict, _load_env, dataset_path
from uk_iran_conflict import policies as pol
from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import (
    Baseline,
    ScenarioResult,
    load_baseline,
    rescale_motor_fuel_to_ons,
    run_scenario,
    wmean,
    wsum,
)


@dataclass(frozen=True)
class Variant:
    """One specification to score."""

    name: str
    scenario_key: str
    ons_fuel: bool
    out_dir: Path
    label: str


def variants() -> tuple[Variant, ...]:
    return (
        Variant(
            name="main_damped_pump",
            scenario_key="realised_2026",
            ons_fuel=False,
            out_dir=RESULTS / "realised_2026",
            label="Main: realised 2026, pump peak damped consistently with gas",
        ),
        Variant(
            name="upper_bound_peak_fuel",
            scenario_key="realised_2026_peak_fuel",
            ons_fuel=False,
            out_dir=RESULTS / "realised_2026_peak_fuel",
            label="Upper bound: peak pump prices applied undamped for a full year",
        ),
        Variant(
            name="robustness_ons_fuel",
            scenario_key="realised_2026",
            ons_fuel=True,
            out_dir=RESULTS / "robustness" / "ons_fuel",
            label="Robustness: main spec on ONS-calibrated motor-fuel deciles",
        ),
    )


def comparison_row(variant: Variant, result: ScenarioResult, base: Baseline) -> dict:
    """The statistics the audit asks to see side by side for all three runs."""
    d1 = result.decile[0]
    d10 = result.decile[-1]
    return {
        "variant": variant.name,
        "scenario": variant.scenario_key,
        "ons_fuel_calibration": variant.ons_fuel,
        "label": variant.label,
        "aggregate_cost_bn": result.aggregate_cost_bn,
        "mean_loss_gbp": result.mean_loss_gbp,
        "mean_loss_pct": result.mean_loss_pct,
        "decile1_loss_gbp": d1.mean_loss_gbp,
        "decile1_loss_pct": d1.mean_loss_pct,
        "decile10_loss_gbp": d10.mean_loss_gbp,
        "decile10_loss_pct": d10.mean_loss_pct,
        "d1_d10_ratio_pct": d1.mean_loss_pct / d10.mean_loss_pct,
        "d1_d10_ratio_gbp": d1.mean_loss_gbp / d10.mean_loss_gbp,
        "gas_share_of_loss": result.gas_share_of_loss,
        "electricity_share_of_loss": result.electricity_share_of_loss,
        "motor_fuel_share_of_loss": result.motor_fuel_share_of_loss,
        "aggregate_energy_spend_bn": wsum(base.energy, base.weight) / 1e9,
        "aggregate_fuel_spend_bn": wsum(base.motor_fuel, base.weight) / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=int, default=2026)
    args = parser.parse_args()

    _load_env()
    path = dataset_path()
    print(f"dataset: {path}")

    # The microdata is loaded exactly once; the ONS variant works on a
    # reweighted copy of the same arrays.
    base = load_baseline(path, args.period)
    print(f"baseline: {base.n:,} households, {base.weight.sum() / 1e6:.1f}m weighted")
    mt = pol.means_tested_flag(path, args.period)
    means_tested_share = float(wmean(mt.astype(float), base.weight))
    print(f"means-tested: {100 * means_tested_share:.1f}% of households")

    ons_base = rescale_motor_fuel_to_ons(base)

    def _fuel_mean(b: Baseline, d: int) -> float:
        sel = b.decile == d
        return wmean(b.motor_fuel[sel], b.weight[sel])

    print(
        "ONS-calibrated fuel: "
        f"D1 £{_fuel_mean(base, 1):.0f} -> £{_fuel_mean(ons_base, 1):.0f}, "
        f"D10 £{_fuel_mean(base, 10):.0f} -> £{_fuel_mean(ons_base, 10):.0f}"
    )

    rows: list[dict] = []
    policy_rows: list[dict] = []
    for variant in variants():
        scenario = scen.SCENARIOS[variant.scenario_key]
        run_base = ons_base if variant.ons_fuel else base
        result, cost = run_scenario(run_base, scenario)
        out = variant.out_dir
        out.mkdir(parents=True, exist_ok=True)

        payload = _asdict(result)
        payload["means_tested_share"] = means_tested_share
        payload["specification"] = variant.label
        payload["ons_fuel_calibration"] = variant.ons_fuel
        payload["pump_sustained_fraction"] = (
            scenario.pass_through.pump_sustained_fraction
        )
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
            f"(£{result.decile[-1].mean_loss_gbp:.0f})"
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
                    "uncompensated_share": score.uncompensated_share_overall,
                    "fully_compensated_share": score.fully_compensated_share,
                    "mean_gain_gbp": score.mean_gain_gbp,
                    "net_loss_after_gbp": score.net_loss_after_policy_gbp,
                }
            )
            print(
                f"   {pkey:15} £{score.cost_bn:5.1f}bn | "
                f"{100 * score.share_to_bottom_three:4.1f}% to D1-3 | "
                f"{100 * score.uncompensated_share_overall:4.1f}% uncompensated"
            )

        rows.append(comparison_row(variant, result, run_base))

    robustness = RESULTS / "robustness"
    robustness.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(robustness / "comparison.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(robustness / "policy_comparison.csv", index=False)
    print(f"\nwrote {robustness / 'comparison.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

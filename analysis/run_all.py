#!/usr/bin/env python3
"""Run every macro scenario x policy response and write results/.

Usage:
    python analysis/run_all.py [--data-dir data] [--period 2026]
                               [--scenarios niesr_baseline ...]
                               [--policies social_tariff ...]
                               [--elasticity main]
                               [--no-constituencies]

Writes one JSON per cell:

    results/{scenario}/shock.json              — the bare shock
    results/{scenario}/{policy}.json           — shock + one policy response

plus results/summary.csv, the flat table the figures and
``analysis/emit_tex_values.py`` read.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uk_iran_conflict.reforms import POLICY_REFORMS  # noqa: E402
from uk_iran_conflict.runner import (  # noqa: E402
    ScenarioResult,
    run_scenario,
    uncompensated_loser_share,
    write_result,
)

try:  # TODO(contract): expects uk_iran_conflict.scenarios.SCENARIOS
    from uk_iran_conflict.scenarios import SCENARIOS
except ImportError:  # pragma: no cover
    SCENARIOS = {}

SUMMARY_COLUMNS = (
    "scenario",
    "policy",
    "exchequer_cost_bn",
    "decile1_average_change",
    "decile1_relative_change",
    "decile10_average_change",
    "bottom_three_decile_share",
    "cost_per_pound_of_bottom_decile_gain",
    "uncompensated_losers_decile1",
    "uncompensated_losers_decile5",
    "poverty_relative_bhc_pp",
    "poverty_relative_ahc_pp",
    "poverty_absolute_bhc_pp",
    "poverty_absolute_ahc_pp",
    "gini_change_pp",
)


def summary_row(result: ScenarioResult) -> dict[str, object]:
    """Flatten one result into the summary-table row the paper reads."""
    decile_avg = {int(k): v for k, v in result.decile_average_change.items()}
    decile_rel = {int(k): v for k, v in result.decile_relative_change.items()}
    losers = uncompensated_loser_share(
        {int(k): v for k, v in result.intra_decile.items()}
    )
    total_gain = sum(v for v in decile_avg.values() if v > 0) or float("nan")
    bottom_three = sum(decile_avg.get(d, 0.0) for d in (1, 2, 3))
    cost_bn = result.exchequer_cost / 1e9
    return {
        "scenario": result.scenario,
        "policy": result.policy or "shock",
        "exchequer_cost_bn": cost_bn,
        "decile1_average_change": decile_avg.get(1, 0.0),
        "decile1_relative_change": decile_rel.get(1, 0.0),
        "decile10_average_change": decile_avg.get(10, 0.0),
        "bottom_three_decile_share": bottom_three / total_gain if total_gain else 0.0,
        "cost_per_pound_of_bottom_decile_gain": (
            result.exchequer_cost / decile_avg[1] if decile_avg.get(1) else 0.0
        ),
        "uncompensated_losers_decile1": losers.get(1, 0.0),
        "uncompensated_losers_decile5": losers.get(5, 0.0),
        "poverty_relative_bhc_pp": result.poverty_change.get("relative_bhc", 0.0) * 100,
        "poverty_relative_ahc_pp": result.poverty_change.get("relative_ahc", 0.0) * 100,
        "poverty_absolute_bhc_pp": result.poverty_change.get("absolute_bhc", 0.0) * 100,
        "poverty_absolute_ahc_pp": result.poverty_change.get("absolute_ahc", 0.0) * 100,
        "gini_change_pp": (result.gini_reform - result.gini_baseline) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default=None, help="explicit dataset .h5 path")
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument("--scenarios", nargs="*", default=sorted(SCENARIOS))
    parser.add_argument(
        "--policies",
        nargs="*",
        default=sorted(POLICY_REFORMS),
        help="policy responses to score on top of each shock",
    )
    parser.add_argument(
        "--elasticity",
        default="main",
        help="elasticity spec: main (zero, the headline) or a robustness variant",
    )
    parser.add_argument("--no-constituencies", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if not args.scenarios:
        parser.error(
            "no scenarios: uk_iran_conflict.scenarios.SCENARIOS is empty or absent"
        )

    dataset = (
        Path(args.dataset) if args.dataset else Path(args.data_dir) / "frs_2023_24.h5"
    )
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name in args.scenarios:
        if name not in SCENARIOS:
            raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
        scenario = SCENARIOS[name]
        for policy in [None, *args.policies]:
            result = run_scenario(
                dataset,
                scenario,
                policy=policy,
                period=args.period,
                elasticity=args.elasticity,
                include_constituencies=not args.no_constituencies,
            )
            filename = f"{policy or 'shock'}.json"
            write_result(result, results / name / filename)
            row = summary_row(result)
            rows.append(row)
            print(
                f"{name}/{row['policy']}: "
                f"exchequer £{row['exchequer_cost_bn']:.1f}bn, "
                f"D1 {row['decile1_relative_change'] * 100:+.2f}%, "
                f"losers in D1 {row['uncompensated_losers_decile1'] * 100:.0f}%, "
                f"rel-AHC poverty {row['poverty_relative_ahc_pp']:+.2f}pp"
            )

    with (results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {results / 'summary.csv'} ({len(rows)} cells)")


if __name__ == "__main__":
    main()

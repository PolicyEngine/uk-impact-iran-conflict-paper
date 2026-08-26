#!/usr/bin/env python3
"""Produce every canonical result file under ``results/``.

Usage::

    python analysis/run_incidence.py            # all scenarios and policies
    python analysis/run_incidence.py --period 2026

Needs the PolicyEngine UK microdata, which is private: set ``HUGGING_FACE_TOKEN``
(the repo reads a gitignored ``.env``). The dataset is downloaded once and
cached by ``huggingface_hub``.

Writes, per scenario, ``results/<scenario>/shock.json`` plus one JSON per policy,
and the flat ``results/summary.csv`` the paper's scorecard table is built from.

Every scenario is scored on the two decisions in ``docs/FIXES.md``: the
equivalised AHC denominator (D1) and the Step 1 consumption-weighted quarterly
cap average on the domestic leg (D2), which are the defaults in
:func:`uk_iran_conflict.incidence.run_scenario`. The realised-2026 family — the
main specification and the alternatives that bound it — is *also* produced by
``analysis/run_variants.py``, off a single load of the microdata and with the
specification labels attached. Run this script first and ``run_variants.py``
second: the numbers agree, and the variants script then rewrites
``results/realised_2026/shock.json`` with the labelled payload. ``summary.csv``
stays the flat, all-scenarios policy table it has always been.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from uk_iran_conflict import policies as pol
from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import load_baseline, run_scenario, wmean, wsum

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASET_REPO = "policyengine/populace-uk-private"
DATASET_FILE = "populace_uk_2023.h5"


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


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", type=int, default=2026)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(scen.SCENARIOS),
        help="Scenarios to score. Defaults to every registered scenario.",
    )
    args = parser.parse_args()

    _load_env()
    path = dataset_path()
    print(f"dataset: {path}")

    base = load_baseline(path, args.period)
    print(f"baseline: {base.n:,} households, {base.weight.sum() / 1e6:.1f}m weighted")
    mt = pol.means_tested_flag(path, args.period)
    print(
        f"means-tested: {100 * wmean(mt.astype(float), base.weight):.1f}% of households"
    )

    rows: list[dict] = []
    for key in args.scenarios:
        scenario = scen.SCENARIOS[key]
        result, cost = run_scenario(base, scenario)
        out = RESULTS / key
        out.mkdir(parents=True, exist_ok=True)

        payload = _asdict(result)
        payload["means_tested_share"] = float(wmean(mt.astype(float), base.weight))
        payload["sustained_fraction"] = scenario.pass_through.sustained_fraction
        payload["pump_sustained_fraction"] = (
            scenario.pass_through.pump_sustained_fraction
        )
        (out / "shock.json").write_text(json.dumps(payload, indent=2))
        print(
            f"\n{key}: £{result.aggregate_cost_bn:.1f}bn, "
            f"mean £{result.mean_loss_gbp:.0f} ({result.mean_loss_pct:.2f}% of income)"
        )
        print(
            f"   income basis {result.income_basis}, domestic basis "
            f"{result.domestic_basis}, mean income "
            f"£{result.mean_income_gbp:,.0f}"
        )
        print(
            f"   decile 1 {result.decile[0].mean_loss_pct:.2f}% "
            f"(£{result.decile[0].mean_loss_gbp:.0f}) vs "
            f"decile 10 {result.decile[-1].mean_loss_pct:.2f}% "
            f"(£{result.decile[-1].mean_loss_gbp:.0f})"
        )

        for pkey, policy in pol.POLICIES.items():
            score, gain = pol.score_policy(base, cost, mt, policy)
            (out / f"{pkey}.json").write_text(json.dumps(_asdict(score), indent=2))
            rows.append(
                {
                    "scenario": key,
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

        # aggregate spend, for the scenario table
        rows_extra = {
            "aggregate_energy_spend_bn": wsum(base.energy, base.weight) / 1e9,
            "aggregate_fuel_spend_bn": wsum(base.motor_fuel, base.weight) / 1e9,
        }
        (out / "aggregates.json").write_text(json.dumps(rows_extra, indent=2))

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False)
    print(f"\nwrote {RESULTS / 'summary.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

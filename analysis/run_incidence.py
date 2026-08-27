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
    """Resolve the pinned microdata file, reading ``.env`` for the token.

    Round-3 finding 7: only ``main()`` called :func:`_load_env`, so importing
    this module and calling ``dataset_path()`` directly — which every other
    analysis script does — raised "No Hugging Face token" despite a valid
    ``.env`` sitting next to it. :func:`_load_env` uses ``setdefault``, so
    calling it here as well is idempotent and never overrides an exported value.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    _load_env()
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
    # Attach the means-tested flag so the fuel-margin diagnostics and the
    # ``mt_fuel_parity`` calibration can see it (round-3 finding 2).
    base = dataclasses.replace(base, means_tested=mt)
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

        # The policy block's non-scorecard diagnostics — feasible maxima
        # uncapped by the envelope, what each envelope arm actually spends, the
        # JRF reference quantities and the large-loser statistic — written once,
        # off the paper's central scenario. ``analysis/emit_tex_values.py``
        # reads it so no policy number in the prose is a carried literal.
        if key == "realised_2026":
            diagnostics = pol.write_policy_diagnostics(
                base, cost, mt, pol.COMMON_ENVELOPE_BN
            )
            print(f"   wrote {diagnostics}")
            # Round-4 findings 1, 2, 3 and 6, printed so a run that changes
            # them is visible in the log and not only in the JSON.
            payload = json.loads(Path(diagnostics).read_text())
            identity = payload["feasible_max_identity"]
            for group in identity["identical_groups"]:
                print(
                    "   feasible-maximum IDENTITY: "
                    + " == ".join(group["policies"])
                    + f" at £{group['feasible_max_cost_bn']:.2f}bn — ONE result, "
                    "not two"
                )
            ceilings = payload["flat_payment_ceilings"]
            bill = ceilings["mean_eligible_domestic_bill_gbp"]
            loss = ceilings["mean_eligible_loss_gbp"]
            print(
                f"   WHD ceiling rule {ceilings['rule_used']}: £{bill:.0f} "
                f"(mean eligible LOSS is £{loss:.0f}, "
                f"x{ceilings['bill_over_loss']:.1f})"
            )
            gap = payload["jrf_costing_gap"]
            implied = gap["single_parameter_reconciliations"]["implied_discount"]
            print(
                f"   JRF gap: modelled £{gap['modelled_cost_bn']:.2f}bn vs "
                f"£{gap['sponsor_cost_bn']:.1f}bn (x{gap['ratio']:.2f}); "
                f"sponsor's total implies a {100 * implied:.1f}% discount on "
                "the same block"
            )
            for pkey in ("social_tariff", "whd_expansion"):
                rules = payload["by_policy"][pkey]["admission_rules"]["range"]
                off = rules["share_of_aggregate_loss_offset"]
                bot = rules["share_to_bottom_three"]
                print(
                    f"   {pkey} eligibility arm across admission rules: "
                    f"offset {100 * off['min']:.1f}-{100 * off['max']:.1f}%, "
                    f"D1-3 {100 * bot['min']:.1f}-{100 * bot['max']:.1f}% "
                    f"(upper bound {100 * bot['upper_bound_rule_value']:.1f}%)"
                )

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False)
    print(f"\nwrote {RESULTS / 'summary.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

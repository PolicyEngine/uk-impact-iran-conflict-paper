#!/usr/bin/env python3
"""The two combination specifications that overturn the motor-fuel majority.

The seven headline specifications in ``analysis/run_variants.py`` vary one thing
at a time. Referee 2's point was sharper than that: the motor-fuel share depends
on the *ratio* of the gas and pump damping fractions, so the calibrations that
matter are the ones that combine a symmetric damping ratio with the other
accounting choices. Those combinations are what the abstract cites as the cases
in which the majority fails, so they need to be real runs rather than arithmetic
in a report.

Two combinations are run here:

* symmetric damping on the steady-state basis (no phase-in)
* symmetric damping with both imputation levels corrected against ONS

Usage::

    python analysis/run_combinations.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.incidence import load_baseline, run_scenario

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_incidence import _load_env, dataset_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "robustness"


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _asdict(v) for k, v in obj.items()}
    return obj


#: (directory, scenario key, run_scenario kwargs, human label)
COMBINATIONS = (
    (
        "symmetric_steady_state",
        "realised_2026_symmetric",
        {"domestic_basis": "steady_state"},
        "Symmetric damping on the steady-state basis (phase-in not applied)",
    ),
    (
        "symmetric_ons_levels",
        "realised_2026_symmetric",
        {"calibration": "ons_both_levels"},
        "Symmetric damping with both imputation levels corrected against ONS",
    ),
)


def main() -> None:
    _load_env()
    path = dataset_path()
    base = load_baseline(path)
    print(f"baseline: {base.n:,} households\n")

    summary = {}
    for name, key, kwargs, label in COMBINATIONS:
        scenario = scen.SCENARIOS[key]
        try:
            result, _ = run_scenario(base, scenario, **kwargs)
        except TypeError as exc:  # the option names are the sibling's to define
            raise SystemExit(
                f"run_scenario does not accept {kwargs!r}: {exc}. "
                "Check the option names in uk_iran_conflict/incidence.py."
            ) from exc
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        payload = _asdict(result)
        payload["label"] = label
        (out / "shock.json").write_text(json.dumps(payload, indent=2))
        summary[name] = {
            "label": label,
            "aggregate_cost_bn": result.aggregate_cost_bn,
            "mean_loss_gbp": result.mean_loss_gbp,
            "mean_loss_pct": result.mean_loss_pct,
            "motor_fuel_share_pct": 100 * result.motor_fuel_share_of_loss,
        }
        print(
            f"{name:24} £{result.aggregate_cost_bn:5.2f}bn  "
            f"£{result.mean_loss_gbp:4.0f}  "
            f"motor fuel {100 * result.motor_fuel_share_of_loss:4.1f}%"
        )

    (OUT / "combinations.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'combinations.json'}")


if __name__ == "__main__":
    main()

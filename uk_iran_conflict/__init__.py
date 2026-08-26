"""From Hormuz to the Household: macro-to-micro incidence of the 2026 UK energy shock.

Vahid Ahmadi (PolicyEngine). A macro -> microsimulation link for the UK: NIESR/OBR
macro scenarios for the 2026 Iran conflict are mapped into an Ofgem cap path and a
pump-price path, run through PolicyEngine UK, and reported at income-decile and
650-constituency level — including the within-decile uncompensated-loser counts that
the horizontal-incidence literature (Cronin, Fullerton & Sexton 2019; Douenne 2020;
Sallee 2019) says dominate the vertical story.

Modules:

* :mod:`uk_iran_conflict.scenarios` — the macro scenario registry (``SCENARIOS``)
* :mod:`uk_iran_conflict.elasticity` — constant-elasticity consumption responses
* :mod:`uk_iran_conflict.reforms` — the price shock and the five policy responses
* :mod:`uk_iran_conflict.runner` — baseline / shocked / shocked+policy runs

``scenarios`` and ``elasticity`` are re-exported defensively so this package
imports cleanly during incremental development and without microdata access.
"""

from __future__ import annotations

from uk_iran_conflict.reforms import (
    POLICY_REFORMS,
    PolicyResponse,
    build_blunt_energy_bills_reform,
    build_policy_reform,
    build_shock_reform,
    compose,
)
from uk_iran_conflict.runner import (
    ScenarioResult,
    intra_decile_breakdown,
    read_result,
    run_scenario,
    uncompensated_loser_share,
    write_result,
)

try:  # pragma: no cover — sibling module, may land after this one
    from uk_iran_conflict.scenarios import SCENARIOS
except ImportError:  # TODO(contract): expects uk_iran_conflict.scenarios.SCENARIOS
    SCENARIOS = {}

__all__ = [
    "POLICY_REFORMS",
    "SCENARIOS",
    "PolicyResponse",
    "ScenarioResult",
    "build_blunt_energy_bills_reform",
    "build_policy_reform",
    "build_shock_reform",
    "compose",
    "intra_decile_breakdown",
    "read_result",
    "run_scenario",
    "uncompensated_loser_share",
    "write_result",
]

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
* :mod:`uk_iran_conflict.incidence` — the consumption-side first-order shock
* :mod:`uk_iran_conflict.policies` — the five scored policy responses

``scenarios`` and ``elasticity`` are re-exported defensively so this package
imports cleanly during incremental development and without microdata access.

There is deliberately **one** pipeline. The former
``uk_iran_conflict.runner`` / ``analysis/run_all.py`` pair implemented the
shock as a PolicyEngine *parameter reform* and aggregated to 650
constituencies — both of which the paper itself shows this data release cannot
support — and silently returned different numbers from the consumption-side
pipeline in :mod:`uk_iran_conflict.incidence`. They were deleted
(docs/FIXES.md C14). ``resolve_elasticity_spec`` moved to
:mod:`uk_iran_conflict.elasticity`.
"""

from __future__ import annotations

from uk_iran_conflict.elasticity import ElasticitySpec, resolve_elasticity_spec
from uk_iran_conflict.reforms import (
    POLICY_REFORMS,
    PolicyResponse,
    build_blunt_energy_bills_reform,
    build_policy_reform,
    build_shock_reform,
    compose,
)

try:  # pragma: no cover — sibling module, may land after this one
    from uk_iran_conflict.scenarios import SCENARIOS
except ImportError:  # TODO(contract): expects uk_iran_conflict.scenarios.SCENARIOS
    SCENARIOS = {}

__all__ = [
    "POLICY_REFORMS",
    "SCENARIOS",
    "ElasticitySpec",
    "PolicyResponse",
    "build_blunt_energy_bills_reform",
    "build_policy_reform",
    "build_shock_reform",
    "compose",
    "resolve_elasticity_spec",
]

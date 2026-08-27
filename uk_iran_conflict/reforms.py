"""Price-path helpers: reading a scenario's retail, pump and cap levels.

What this module is now
-----------------------
Three pure functions — :func:`retail_factors`, :func:`pump_price_factors` and
:func:`cap_levels` — that read a :class:`~uk_iran_conflict.scenarios.Scenario`
and return the price multipliers and cap levels it implies. Nothing here
imports PolicyEngine, touches microdata or builds a reform.

What this module used to be, and why it is gone
-----------------------------------------------
It was a **second, parallel implementation of the whole paper** — a
``build_shock_reform`` that moved the shock through PolicyEngine *parameters*,
and five ``build_*`` policy-response builders — and every part of it
contradicted the paper it belonged to (round-2 referee 3, ``docs/FIXES.md``
C14/C9):

* The shock reform set ``gov.ofgem.energy_price_cap``. The paper's §3.1 shows
  that parameter is read by exactly one variable, EPG *subsidy* machinery, and
  is not a household price channel; and it scaled the DESNZ pump-price
  parameters, which — because ``petrol_spending`` is an *input* variable — cuts
  litres at fixed spend rather than raising spend at fixed litres. That is the
  opposite sign of a first-order price shock, and it also moves fuel duty.
* ``build_whd_expansion`` was built on ``gov.dwp.warm_home_discount.amount``.
  The paper states categorically, and the installed release confirms, that
  **there is no Warm Home Discount in PolicyEngine UK at all** — no parameter,
  no variable. The reform could not have done what its docstring said.
* The five builders carried **different parameters** from the five instruments
  actually scored in :mod:`uk_iran_conflict.policies` (a different social-tariff
  discount, a different block share, a different rebate), so a replicator who
  ran them got different numbers with no warning.
* ``quarterly_periods`` emitted the period strings ``2026-04-01.2026-06-31`` and
  ``2026-10-01.2026-12-31`` — 31 June and 31 September. Two of the four cap
  quarters were unreachable dates, latent because nothing exercised them
  end-to-end.

None of it was imported by any pipeline that produces a number in the paper.
Deleting it is the fix; the surviving three helpers are the only ones
``analysis/`` and :mod:`uk_iran_conflict.incidence` ever called.

The single pipeline is: :mod:`uk_iran_conflict.scenarios` (prices) ->
:mod:`uk_iran_conflict.incidence` (first-order consumption-side incidence) ->
:mod:`uk_iran_conflict.policies` (transfers scored against that baseline).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "cap_levels",
    "pump_price_factors",
    "retail_factors",
]


def cap_levels(scenario: Any) -> tuple[float, ...]:
    """Quarterly Ofgem cap levels (£/yr) implied by ``scenario``.

    Reads :attr:`uk_iran_conflict.scenarios.Scenario.cap_path`, a tuple of
    ``CapStep`` records carrying ``cap_gbp``. A plain sequence of floats is
    also accepted so callers can hand-build a path in tests.

    Reporting only: the cap path is what the modelled quarterly steps *look
    like*, and the anchor in
    :data:`uk_iran_conflict.scenarios.CAP_ANCHOR_PCT` is imposed on it. It is
    never fed to PolicyEngine as a parameter.
    """
    path = getattr(scenario, "cap_path", None)
    if path is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes no "
            "cap_path; expected the uk_iran_conflict.scenarios contract"
        )
    return tuple(float(getattr(step, "cap_gbp", step)) for step in path)


def retail_factors(scenario: Any) -> tuple[float, float]:
    """The **steady-state** (gas, electricity) retail shock multipliers.

    Reads :attr:`Scenario.retail_shock`, whose ``gas_factor`` and
    ``electricity_factor`` are full-pass-through multipliers on PolicyEngine
    UK's separate ``gas_consumption`` and ``electricity_consumption``
    variables. **Their inequality is the point** — gas sets the marginal
    electricity price only about 85% of the time, so the two must never be
    collapsed onto ``domestic_energy_consumption``.

    These are *not* the multipliers the headline charges households. The
    modelled year damps them by the window-average phase-in
    (:attr:`Scenario.annual_retail_shock`); see
    :func:`uk_iran_conflict.incidence.domestic_retail_factors`, which is the
    single definition of what the incidence run applies. This function exists
    for reporting the steady state and for the sweeps that vary around it.

    Raises rather than falling back to a silently unshocked 1.0
    (``docs/FIXES.md`` E33).
    """
    shock = getattr(scenario, "retail_shock", None)
    if shock is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes no "
            "retail_shock; refusing to fall back to an unshocked 1.0"
        )
    return float(shock.gas_factor), float(shock.electricity_factor)


def pump_price_factors(scenario: Any) -> tuple[float, float]:
    """The **quoted** (petrol, diesel) pump-price multipliers for ``scenario``.

    For the realised path these are the observed *peaks*. The peak-to-window
    damping that the modelled year actually applies lives in
    :func:`uk_iran_conflict.incidence.sustained_pump_factors`; use that for
    anything that has to agree with the headline.

    Raises rather than falling back to a silently unshocked 1.0
    (``docs/FIXES.md`` E33).
    """
    pump = getattr(scenario, "pump", None)
    if pump is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes no pump "
            "path; refusing to fall back to an unshocked 1.0"
        )
    return 1.0 + float(pump.petrol_pct_change), 1.0 + float(pump.diesel_pct_change)

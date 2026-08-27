"""Tests for the surviving price-path helpers in ``uk_iran_conflict.reforms``.

The module used to be a second, parallel implementation of the whole paper —
a shock built as a PolicyEngine parameter reform plus five policy-response
builders on parameters the paper states do not exist. It was deleted in the
round-2 revision (referee 3, ``docs/FIXES.md`` C14); these tests pin what is
left, and pin the deletion itself so it cannot creep back.

Nothing here needs microdata, a Hugging Face token or PolicyEngine at all,
which is the point: the module now imports nothing.
"""

from __future__ import annotations

import pytest

from uk_iran_conflict import reforms
from uk_iran_conflict.scenarios import get_scenario

# --- the deletion ---------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        # The parallel shock implementation.
        "build_shock_reform",
        "build_relative_price_reform",
        "build_blunt_energy_bills_reform",
        "compose",
        # The five duplicate policy builders and their registry.
        "POLICY_REFORMS",
        "PolicyResponse",
        "build_policy_reform",
        "build_social_tariff",
        "build_jrf_discounted_block",
        "build_whd_expansion",
        "build_vat_zero_rating",
        "build_ippr_rebate",
        # The date-buggy period machinery (2026-06-31, 2026-09-31).
        "quarterly_periods",
        "_cap_changes",
        # The Warm Home Discount parameter that does not exist in the model.
        "WHD_AMOUNT",
        "OFGEM_CAP",
        "ENERGY_BILLS_LEVER",
    ],
)
def test_the_dead_parallel_implementation_is_gone(name):
    assert not hasattr(reforms, name), (
        f"{name!r} is back. It is part of the second implementation the paper "
        "contradicts; the single pipeline is scenarios -> incidence -> policies."
    )


def test_module_surface_is_only_the_three_price_path_helpers():
    assert set(reforms.__all__) == {
        "cap_levels",
        "pump_price_factors",
        "retail_factors",
    }


def test_module_imports_nothing_from_policyengine():
    """A regression guard: these helpers must stay pure arithmetic.

    The deleted builders imported ``policyengine_core.reforms`` lazily; any
    import statement here means a reform builder has come back.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(reforms))
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any("policyengine" in name for name in imported), imported


# --- the helpers ----------------------------------------------------------


def test_retail_factors_read_the_steady_state_shock():
    scenario = get_scenario("realised_2026")
    gas, elec = reforms.retail_factors(scenario)
    assert (gas, elec) == (
        scenario.retail_shock.gas_factor,
        scenario.retail_shock.electricity_factor,
    )
    # The asymmetry is the paper's central modelling claim.
    assert gas > elec > 1.0


def test_retail_factors_are_the_steady_state_not_the_annualised_one():
    """The helper reports the steady state; the headline applies the window one."""
    scenario = get_scenario("realised_2026")
    gas, _ = reforms.retail_factors(scenario)
    assert gas > scenario.annual_retail_shock.gas_factor


def test_pump_price_factors_are_the_quoted_peaks_not_the_damped_moves():
    scenario = get_scenario("realised_2026")
    petrol, diesel = reforms.pump_price_factors(scenario)
    assert petrol == pytest.approx(1.20)
    assert diesel == pytest.approx(1.36)


def test_cap_levels_match_the_scenario_cap_path():
    scenario = get_scenario("realised_2026")
    assert reforms.cap_levels(scenario) == tuple(
        step.cap_gbp for step in scenario.cap_path
    )


def test_cap_levels_accept_a_hand_built_path():
    class Bare:
        cap_path = (1663.0, 1700.0, 1768.0)

    assert reforms.cap_levels(Bare()) == (1663.0, 1700.0, 1768.0)


# --- E33: no silent unshocked fallbacks -----------------------------------


@pytest.mark.parametrize(
    "helper",
    [reforms.retail_factors, reforms.pump_price_factors, reforms.cap_levels],
)
def test_helpers_raise_rather_than_returning_a_silent_unit_factor(helper):
    """``docs/FIXES.md`` E33: a zero shock must be unrepresentable by accident.

    The old implementations fell back to ``1.0`` for a scenario that exposed
    nothing, which is indistinguishable in the results from a correctly
    modelled zero shock.
    """

    class Empty:
        key = "empty"

    with pytest.raises(AttributeError):
        helper(Empty())

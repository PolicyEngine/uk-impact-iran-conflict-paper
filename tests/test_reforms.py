"""Reform construction tests.

These run without microdata: they exercise the parameter-path bookkeeping and
the gas/electricity asymmetry, and only touch PolicyEngine where the reform
object itself is built (skipped if policyengine-uk is not installed).
"""

from __future__ import annotations

import dataclasses

import pytest

from uk_iran_conflict import reforms


def _needs_pe():
    """Skip a test when policyengine-core is unavailable (no data needed)."""
    return pytest.importorskip("policyengine_core.reforms")


@dataclasses.dataclass(frozen=True)
class FakeScenario:
    """Stand-in matching the uk_iran_conflict.scenarios contract."""

    name: str = "test"
    oil_price_change: float = 0.57
    gas_wholesale_change: float = 0.65
    ofgem_cap_path: tuple[float, ...] = (1717.0, 1720.0, 1663.0, 1729.0)
    petrol_price_change: float = 0.20
    diesel_price_change: float = 0.36
    gas_shock_factor: float = 1.30
    electricity_shock_factor: float = 1.12
    source: str = "test fixture"


# --- parameter paths ------------------------------------------------------


def test_parameter_paths_match_policyengine_uk():
    """The paths the paper documents are the paths the code shocks."""
    assert reforms.OFGEM_CAP == "gov.ofgem.energy_price_cap"
    assert reforms.PETROL_PRICE == "household.consumption.fuel.prices.petrol"
    assert reforms.DIESEL_PRICE == "household.consumption.fuel.prices.diesel"
    assert reforms.ENERGY_BILLS_LEVER == (
        "gov.contrib.policyengine.economy.energy_bills"
    )
    assert reforms.EPG_SUBSIDY == "gov.treasury.price_cap_subsidy"
    assert reforms.EPG_CONSUMPTION_LEVEL.endswith("monthly_epg_consumption_level")


def test_quarterly_periods_are_four_and_ordered():
    periods = reforms.quarterly_periods(2026)
    assert len(periods) == 4
    assert all(p.startswith("2026-") for p in periods)
    assert periods == tuple(sorted(periods))


def test_cap_path_length_is_validated():
    with pytest.raises(ValueError, match="expected 4"):
        reforms._cap_changes([1.0, 2.0], 2026)


# --- gas / electricity asymmetry ------------------------------------------


def test_blend_is_gas_weighted():
    """Gas is 62% of UK household final energy — the highest in the G7."""
    assert reforms.GAS_SHARE_OF_HOUSEHOLD_ENERGY == pytest.approx(0.62)
    blended = reforms._blend_cap_factor(1.30, 1.10)
    assert blended == pytest.approx(0.62 * 1.30 + 0.38 * 1.10)
    # strictly between the two fuel factors, and closer to gas
    assert 1.10 < blended < 1.30
    assert abs(blended - 1.30) < abs(blended - 1.10)


def test_asymmetric_factors_change_the_answer():
    """A symmetric shock and an asymmetric shock of the same mean differ."""
    symmetric = reforms._blend_cap_factor(1.20, 1.20)
    asymmetric = reforms._blend_cap_factor(1.30, 1.10)
    assert symmetric != pytest.approx(asymmetric)


def test_blend_is_identity_when_factors_equal():
    assert reforms._blend_cap_factor(1.25, 1.25) == pytest.approx(1.25)


# --- scenario field access ------------------------------------------------


def test_scenario_field_reads_the_contract():
    scenario = FakeScenario()
    assert reforms._scenario_field(scenario, "gas_shock_factor") == 1.30
    assert reforms._scenario_field(scenario, "missing", default=7.0) == 7.0
    assert reforms._scenario_name(scenario) == "test"


def test_scenario_field_raises_on_unknown_without_default():
    with pytest.raises(AttributeError, match="contract"):
        reforms._scenario_field(FakeScenario(), "not_a_field")


# --- policy registry ------------------------------------------------------


def test_five_policy_responses_registered():
    assert set(reforms.POLICY_REFORMS) == {
        "social_tariff",
        "jrf_block",
        "whd_expansion",
        "vat_zero",
        "ippr_rebate",
    }


@pytest.mark.parametrize("name,response", sorted(reforms.POLICY_REFORMS.items()))
def test_every_policy_documents_source_and_cost(name, response):
    """Each reform carries provenance and a real-world cost estimate."""
    assert response.name == name
    assert response.source
    assert response.cost_estimate_bn > 0
    doc = response.builder.__doc__ or ""
    assert "Source" in doc or "source" in doc
    assert "cost estimate" in doc.lower()


def test_jrf_block_cost_matches_the_published_figure():
    """JRF (Moore, Cook, 9 Apr 2026) cost the discounted block at ~£5bn."""
    assert reforms.POLICY_REFORMS["jrf_block"].cost_estimate_bn == pytest.approx(5.0)


def test_ippr_rebate_amount_is_the_published_183():
    import inspect

    signature = inspect.signature(reforms.build_ippr_rebate)
    assert signature.parameters["amount"].default == pytest.approx(183.0)


def test_whd_amount_is_150():
    import inspect

    signature = inspect.signature(reforms.build_whd_expansion)
    assert signature.parameters["amount"].default == pytest.approx(150.0)


def test_jrf_block_defaults_to_half_of_typical_consumption():
    import inspect

    signature = inspect.signature(reforms.build_jrf_discounted_block)
    assert signature.parameters["block_share"].default == pytest.approx(0.50)
    assert signature.parameters["per_child_allowance"].default > 0


def test_build_policy_reform_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown policy"):
        reforms.build_policy_reform("no_such_policy")


# --- reform objects (need policyengine-core, not microdata) ---------------


def test_shock_reform_builds():
    _needs_pe()
    reform = reforms.build_shock_reform(FakeScenario())
    assert reform.gas_shock_factor == pytest.approx(1.30)
    assert reform.electricity_shock_factor == pytest.approx(1.12)


@pytest.mark.parametrize("name", sorted(reforms.POLICY_REFORMS))
def test_policy_reforms_build(name):
    _needs_pe()
    assert reforms.build_policy_reform(name) is not None


def test_compose_is_a_noop_on_a_single_reform():
    sentinel = object()
    assert reforms.compose(sentinel) is sentinel
    assert reforms.compose(None) is None


# --- the real scenario contract (no PolicyEngine needed) ------------------


def test_real_scenario_accessors_prefer_the_derived_properties():
    """``Scenario.retail_shock`` / ``.cap_path`` win over the bare fallbacks."""
    from uk_iran_conflict.scenarios import get_scenario

    scenario = get_scenario("realised_2026")
    gas, elec = reforms.retail_factors(scenario)
    assert (gas, elec) == (
        scenario.retail_shock.gas_factor,
        scenario.retail_shock.electricity_factor,
    )
    assert reforms.cap_levels(scenario) == tuple(
        step.cap_gbp for step in scenario.cap_path
    )
    petrol, diesel = reforms.pump_price_factors(scenario)
    assert petrol == pytest.approx(1.20)
    assert diesel == pytest.approx(1.36)


def test_fallback_accessors_still_work_for_a_bare_scenario():
    """A hand-built scenario without the derived properties still resolves."""
    fake = FakeScenario()
    assert reforms.retail_factors(fake) == (1.30, 1.12)
    assert reforms.cap_levels(fake) == fake.ofgem_cap_path
    assert reforms.pump_price_factors(fake) == (
        pytest.approx(1.20),
        pytest.approx(1.36),
    )

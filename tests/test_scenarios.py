"""Scenario-registry integrity tests.

Every macro scenario must be internally consistent, sourced, and consistent
with the published figures in docs/RESEARCH_BRIEF.md. These run without
microdata.
"""

from __future__ import annotations

import dataclasses

import pytest

from uk_iran_conflict import reforms
from uk_iran_conflict.scenarios import (
    REALISED_PUMP_SUSTAINED_FRACTION,
    REALISED_SUSTAINED_FRACTION,
    SCENARIOS,
    PassThroughAssumptions,
    PumpPricePath,
    Scenario,
    get_scenario,
    scenario_keys,
)

ALL = sorted(SCENARIOS.items())


def test_registry_is_non_empty_and_keyed_consistently():
    assert SCENARIOS
    for key, scenario in ALL:
        assert isinstance(scenario, Scenario)
        assert scenario.key == key
    assert set(scenario_keys()) == set(SCENARIOS)


def test_get_scenario_round_trips_and_rejects_typos():
    for key, scenario in ALL:
        assert get_scenario(key) is scenario
    with pytest.raises(KeyError, match="unknown scenario"):
        get_scenario("no_such_scenario")


@pytest.mark.parametrize("key,scenario", ALL)
def test_scenario_is_frozen(key, scenario):
    """Scenarios are calibration constants; a run must not mutate them."""
    assert dataclasses.is_dataclass(scenario)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        scenario.label = "mutated"


@pytest.mark.parametrize("key,scenario", ALL)
def test_scenario_is_sourced_and_described(key, scenario):
    """No scenario enters the paper without a citation a referee can check."""
    assert len(scenario.source.strip()) > 40, f"{key} lacks a real citation"
    assert len(scenario.description.strip()) > 40, f"{key} lacks a description"
    assert scenario.label


@pytest.mark.parametrize("key,scenario", ALL)
def test_empty_source_is_rejected_at_construction(key, scenario):
    """__post_init__ enforces the citation requirement."""
    with pytest.raises(ValueError, match="source citation"):
        dataclasses.replace(scenario, source="   ")


# --- price paths ----------------------------------------------------------


@pytest.mark.parametrize("key,scenario", ALL)
def test_oil_and_gas_moves_are_upward_and_bounded(key, scenario):
    assert scenario.oil.level_usd_per_bbl > 0, key
    assert scenario.gas.change_pence_per_therm >= 0, key
    # 2022 peaked at +300p/therm; nothing in 2026 should exceed that.
    assert scenario.gas.change_pence_per_therm <= 300.0, key


@pytest.mark.parametrize("key,scenario", ALL)
def test_pump_price_changes_are_fractions_not_percentages(key, scenario):
    assert 0.0 <= scenario.pump.petrol_pct_change <= 1.0, f"{key}: petrol"
    assert 0.0 <= scenario.pump.diesel_pct_change <= 1.0, f"{key}: diesel"


@pytest.mark.parametrize("key,scenario", ALL)
def test_diesel_rises_at_least_as_much_as_petrol(key, scenario):
    """Observed 2026 peak: petrol +20%, diesel +36% — diesel is more exposed."""
    assert scenario.pump.diesel_pct_change >= scenario.pump.petrol_pct_change, key


def test_realised_scenario_reproduces_the_published_pump_moves():
    """The brief's realised 2026 peak: petrol +20%, diesel +36%."""
    realised = get_scenario("realised_2026")
    assert realised.pump.petrol_pct_change == pytest.approx(0.20)
    assert realised.pump.diesel_pct_change == pytest.approx(0.36)


# --- the gas/electricity asymmetry (the paper's Step 1 claim) -------------


@pytest.mark.parametrize("key,scenario", ALL)
def test_retail_factors_are_multipliers_at_or_above_one(key, scenario):
    shock = scenario.retail_shock
    assert shock.gas_factor >= 1.0, f"{key}: gas"
    assert shock.electricity_factor >= 1.0, f"{key}: electricity"
    assert shock.gas_factor == pytest.approx(1.0 + shock.gas_pct_change)
    assert shock.electricity_factor == pytest.approx(1.0 + shock.electricity_pct_change)


@pytest.mark.parametrize("key,scenario", ALL)
def test_gas_is_shocked_harder_than_electricity(key, scenario):
    """Gas sets the marginal power price ~85% of the time, not 100%.

    So a wholesale gas move passes through to gas bills strictly more strongly
    than to electricity bills. This asymmetry is the modelling choice that
    makes it a UK paper; if it ever collapses, the headline result is wrong.
    """
    shock = scenario.retail_shock
    if shock.gas_pct_change == 0.0:
        pytest.skip(f"{key} has no gas shock")
    assert shock.gas_factor > shock.electricity_factor, key
    assert scenario.pass_through.marginal_pricing_share < 1.0, key


def test_full_marginal_pricing_recovers_symmetry():
    """marginal_pricing_share = 1.0 is documented as the naive symmetric case."""
    scenario = get_scenario("realised_2026")
    naive = dataclasses.replace(
        scenario,
        pass_through=dataclasses.replace(
            scenario.pass_through, marginal_pricing_share=1.0
        ),
    )
    assert naive.retail_shock.electricity_pct_change > (
        scenario.retail_shock.electricity_pct_change
    )


# --- the cap path ---------------------------------------------------------


@pytest.mark.parametrize("key,scenario", ALL)
def test_cap_path_is_quarterly_and_labelled(key, scenario):
    path = scenario.cap_path
    assert len(path) == len(scenario.quarter_labels), key
    assert [s.quarter for s in path] == list(scenario.quarter_labels), key


@pytest.mark.parametrize("key,scenario", ALL)
def test_cap_levels_are_plausible_annual_bills(key, scenario):
    """£/yr on the Ofgem TDCV basis — not p/kWh, and not 2022 territory."""
    for step in scenario.cap_path:
        assert 800.0 < step.cap_gbp < 6000.0, f"{key} {step.quarter}: {step.cap_gbp}"
        assert step.cap_gbp >= step.baseline_cap_gbp, f"{key}: cap falls below baseline"
        assert step.cap_pct_change == pytest.approx(
            step.cap_change_gbp / step.baseline_cap_gbp
        )


@pytest.mark.parametrize("key,scenario", ALL)
def test_cap_path_is_lagged_not_instant(key, scenario):
    """The cap lags forward wholesale 6-9 months, so quarter 1 is damped."""
    profile = scenario.pass_through.phase_in_profile
    assert profile[0] < max(profile), f"{key}: no phase-in lag"
    assert scenario.pass_through.lag_quarters >= 2, key


@pytest.mark.parametrize("key,scenario", ALL)
def test_peak_cap_is_the_max_of_the_path(key, scenario):
    assert scenario.peak_cap_gbp == max(s.cap_gbp for s in scenario.cap_path)


@pytest.mark.parametrize("key,scenario", ALL)
def test_cap_step_lookup_matches_the_path(key, scenario):
    for step in scenario.cap_path:
        assert scenario.cap_step(step.quarter) == step
    with pytest.raises(KeyError):
        scenario.cap_step("1999Q9")


@pytest.mark.parametrize("key,scenario", ALL)
def test_wholesale_share_of_bill_is_forty_to_fifty_percent(key, scenario):
    """The brief: wholesale is ~40-50% of a dual-fuel bill."""
    pt = scenario.pass_through
    assert 0.30 <= pt.wholesale_share_gas_bill <= 0.60, key
    assert 0.20 <= pt.wholesale_share_electricity_bill <= 0.60, key
    assert 0.0 < pt.gas_share_of_dual_fuel_bill < 1.0, key


# --- registry-level properties -------------------------------------------


def test_scenarios_are_distinct_calibrations():
    signatures = {
        (
            s.oil.level_usd_per_bbl,
            s.gas.change_pence_per_therm,
            s.pump.petrol_pct_change,
            s.pass_through.sustained_fraction,
            s.pass_through.pump_sustained_fraction,
        )
        for s in SCENARIOS.values()
    }
    assert len(signatures) == len(SCENARIOS), "duplicate scenario calibrations"


def test_a_bigger_gas_shock_gives_a_bigger_steady_state_retail_shock():
    """Monotonicity of the wholesale-to-retail map, holding damping fixed.

    ``sustained_fraction`` is compared at a common value because it is a
    separate assumption: ``realised_2026`` damps a short-lived peak, so its
    headline +78p/therm can imply a *smaller* sustained retail shock than a
    smaller-but-sustained NIESR path. That is intended, and is exactly why the
    comparison has to hold the damping constant.
    """
    common = dataclasses.replace(
        get_scenario("realised_2026").pass_through, sustained_fraction=1.0
    )
    ordered = sorted(SCENARIOS.values(), key=lambda s: s.gas.change_pence_per_therm)
    shocks = [
        dataclasses.replace(s, pass_through=common).retail_shock.gas_pct_change
        for s in ordered
    ]
    assert shocks == sorted(shocks), "gas wholesale and retail orderings disagree"


def test_sustained_fraction_damps_the_retail_shock():
    """A short-lived peak passes through less than a sustained one."""
    scenario = get_scenario("realised_2026")
    sustained = dataclasses.replace(
        scenario,
        pass_through=dataclasses.replace(scenario.pass_through, sustained_fraction=1.0),
    )
    assert sustained.retail_shock.gas_pct_change >= scenario.retail_shock.gas_pct_change


# --- the contract with reforms.py ----------------------------------------


@pytest.mark.parametrize("key,scenario", ALL)
def test_reforms_can_read_every_scenario(key, scenario):
    """The registry and the reform builder agree on the field contract."""
    levels = reforms.cap_levels(scenario)
    assert len(levels) == len(scenario.cap_path)
    assert all(v > 0 for v in levels)

    gas, elec = reforms.retail_factors(scenario)
    assert gas >= elec >= 1.0

    petrol, diesel = reforms.pump_price_factors(scenario)
    assert diesel >= petrol >= 1.0


@pytest.mark.parametrize("key,scenario", ALL)
def test_cap_changes_map_onto_four_policyengine_periods(key, scenario):
    changes = reforms._cap_changes(reforms.cap_levels(scenario), 2026)
    assert len(changes) == 4, key
    assert all(period.startswith("2026-") for period in changes)


# --- pump damping (VALIDATION.md Check 2b) --------------------------------


def test_pump_sustained_fraction_defaults_to_one_and_is_the_identity():
    """The new parameter must leave every pre-existing scenario untouched."""
    default = PassThroughAssumptions()
    assert default.pump_sustained_fraction == 1.0
    pump = PumpPricePath(petrol_pct_change=0.20, diesel_pct_change=0.36)
    assert default.sustained_pump_changes(pump) == pytest.approx((0.20, 0.36))
    for key in ("niesr_baseline", "niesr_adverse"):
        scenario = get_scenario(key)
        assert scenario.pass_through.pump_sustained_fraction == 1.0
        assert scenario.sustained_pump_changes == pytest.approx(
            (scenario.pump.petrol_pct_change, scenario.pump.diesel_pct_change)
        )


def test_pump_sustained_fraction_scales_both_fuels_linearly():
    pump = PumpPricePath(petrol_pct_change=0.20, diesel_pct_change=0.36)
    half = PassThroughAssumptions(pump_sustained_fraction=0.5)
    assert half.sustained_pump_changes(pump) == pytest.approx((0.10, 0.18))
    # Damping is proportional, so the diesel/petrol ratio is preserved.
    petrol, diesel = half.sustained_pump_changes(pump)
    assert diesel / petrol == pytest.approx(0.36 / 0.20)


def test_pump_sustained_fraction_is_bounded():
    with pytest.raises(ValueError, match="pump_sustained_fraction"):
        PassThroughAssumptions(pump_sustained_fraction=1.5)
    with pytest.raises(ValueError, match="pump_sustained_fraction"):
        PassThroughAssumptions(pump_sustained_fraction=-0.1)


def test_pump_damping_does_not_touch_the_cap_path():
    """The pump parameter is a fuel-channel object; the cap must not move."""
    main = get_scenario("realised_2026")
    bound = get_scenario("realised_2026_peak_fuel")
    assert [s.cap_gbp for s in main.cap_path] == [s.cap_gbp for s in bound.cap_path]
    assert main.retail_shock == bound.retail_shock


def test_realised_main_damps_the_pump_peak_and_the_bound_does_not():
    main = get_scenario("realised_2026")
    bound = get_scenario("realised_2026_peak_fuel")
    assert main.pass_through.pump_sustained_fraction == pytest.approx(
        REALISED_PUMP_SUSTAINED_FRACTION
    )
    assert bound.pass_through.pump_sustained_fraction == 1.0
    # The bound applies the published peaks verbatim...
    assert bound.sustained_pump_changes == pytest.approx((0.20, 0.36))
    # ...and the main specification strictly less.
    assert main.sustained_pump_changes[0] < bound.sustained_pump_changes[0]
    assert main.sustained_pump_changes[1] < bound.sustained_pump_changes[1]


def test_pump_damping_is_not_the_cap_damping():
    """0.36 is calibrated to the Ofgem cap window; reusing it here would be wrong."""
    assert REALISED_PUMP_SUSTAINED_FRACTION != REALISED_SUSTAINED_FRACTION
    assert REALISED_PUMP_SUSTAINED_FRACTION > REALISED_SUSTAINED_FRACTION
    assert 0.0 < REALISED_PUMP_SUSTAINED_FRACTION <= 1.0


def test_the_two_realised_scenarios_differ_only_in_pump_damping():
    main = get_scenario("realised_2026")
    bound = get_scenario("realised_2026_peak_fuel")
    for field_name in ("oil", "gas", "pump", "baseline_cap_gbp", "quarter_labels"):
        assert getattr(main, field_name) == getattr(bound, field_name)
    assert (
        dataclasses.replace(main.pass_through, pump_sustained_fraction=1.0)
        == bound.pass_through
    )


def test_incidence_applies_the_pump_damping():
    """The damping must actually reach the cost calculation, not just the dataclass."""
    from uk_iran_conflict.incidence import sustained_pump_factors

    main = get_scenario("realised_2026")
    bound = get_scenario("realised_2026_peak_fuel")
    assert sustained_pump_factors(bound) == pytest.approx((1.20, 1.36))
    petrol, diesel = sustained_pump_factors(main)
    assert petrol == pytest.approx(1.0 + REALISED_PUMP_SUSTAINED_FRACTION * 0.20)
    assert diesel == pytest.approx(1.0 + REALISED_PUMP_SUSTAINED_FRACTION * 0.36)
    # Undamped raw factors still available for the parameter reform.
    assert reforms.pump_price_factors(main) == pytest.approx((1.20, 1.36))

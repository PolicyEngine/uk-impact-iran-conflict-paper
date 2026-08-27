"""Scenario-registry integrity tests.

Every macro scenario must be internally consistent, sourced, and consistent
with the published figures in docs/RESEARCH_BRIEF.md. These run without
microdata.
"""

from __future__ import annotations

import dataclasses

import pytest

from uk_iran_conflict import reforms
from uk_iran_conflict import scenarios as scen
from uk_iran_conflict.scenarios import (
    CAP_PHASE_IN_PROFILE,
    QUARTERLY_CONSUMPTION_WEIGHTS_ELECTRICITY,
    QUARTERLY_CONSUMPTION_WEIGHTS_GAS,
    REALISED_PUMP_SUSTAINED_FRACTION,
    REALISED_SUSTAINED_FRACTION,
    SCENARIOS,
    SYMMETRIC_SUSTAINED_FRACTION,
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
    """The cap lags, so the shock quarter is damped relative to full pass-through.

    The lag is now **1.5 quarters**, not 3. Ofgem's observation window closes
    about seven weeks before the charge period and covers the preceding
    quarter, so midpoint-to-midpoint is four to five months. Three round-2
    referees flagged 3 as roughly double the institutional lag, and it was
    doing real work: it pushed the cap move out of the modelled year and
    produced the claim that 2026 is almost purely a pump-price event.
    """
    profile = scenario.pass_through.phase_in_profile
    assert profile[0] < max(profile), f"{key}: no phase-in lag"
    assert 1.0 <= scenario.pass_through.lag_quarters <= 2.0, key
    # Round-3: the profile is NOT monotone any more, and must not be. It is the
    # average of a monthly wholesale path over a moving three-month observation
    # window, so it rises as the spike enters the window and falls as the spike
    # leaves it. The old assertion was true only because the old profile was a
    # ramp in calendar time, which is not how the cap works.
    peak = profile.index(max(profile))
    assert list(profile[: peak + 1]) == sorted(profile[: peak + 1]), key
    assert profile[-1] <= max(profile), key


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
def test_cap_levels_cover_the_modelled_window(key, scenario):
    """One cap level per quarter the modelled year touches.

    The period-string machinery this test used to exercise
    (``reforms._cap_changes``) is gone with the rest of the parallel parameter
    -reform implementation; two of the four period strings it emitted were
    unreachable dates (``2026-06-31``, ``2026-09-31``).
    """
    levels = reforms.cap_levels(scenario)
    assert len(levels) == len(scen.CAP_QUARTER_LABELS), key
    assert scenario.quarter_labels == scen.CAP_QUARTER_LABELS, key
    assert all(v > 0 for v in levels), key


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
    """The cap fraction is a cap-window share; reusing it at the pump is wrong."""
    assert REALISED_PUMP_SUSTAINED_FRACTION != REALISED_SUSTAINED_FRACTION
    assert 0.0 < REALISED_PUMP_SUSTAINED_FRACTION <= 1.0
    assert 0.0 < REALISED_SUSTAINED_FRACTION <= 1.0
    # Round-3: once the baseline error is out of it, the two are close rather
    # than differing by a factor of three. Most of the ratio was the bug.
    assert 0.5 < REALISED_PUMP_SUSTAINED_FRACTION / REALISED_SUSTAINED_FRACTION < 2.0


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
    # The quoted peaks stay available for reporting.
    assert reforms.pump_price_factors(main) == pytest.approx((1.20, 1.36))


# --- D2: the consumption-weighted annual domestic factor ------------------


def test_consumption_weights_are_shares_and_winter_heavy():
    for weights in (
        QUARTERLY_CONSUMPTION_WEIGHTS_GAS,
        QUARTERLY_CONSUMPTION_WEIGHTS_ELECTRICITY,
    ):
        assert len(weights) == len(CAP_PHASE_IN_PROFILE)
        assert sum(weights) == pytest.approx(1.0)
        assert all(w > 0 for w in weights)
    # On the shock-year window the quarters are 2026Q1 (March only), Q2, Q3,
    # Q4 and 2027Q1 (Jan-Feb). The two heating blocks are Q4 and 2027Q1.
    gas, elec = (
        QUARTERLY_CONSUMPTION_WEIGHTS_GAS,
        QUARTERLY_CONSUMPTION_WEIGHTS_ELECTRICITY,
    )
    assert gas[3] + gas[4] > 0.5, "Oct-Feb should carry most of the gas year"
    assert max(gas) - min(gas) > max(elec) - min(elec), "gas is more seasonal"
    # The window is a full annual heating cycle, so the monthly weights it is
    # built from need no renormalisation.
    assert sum(scen.MONTHLY_CONSUMPTION_WEIGHTS_GAS) == pytest.approx(1.0)
    assert sum(scen.MONTHLY_CONSUMPTION_WEIGHTS_ELECTRICITY) == pytest.approx(1.0)


def test_annual_phase_in_is_the_consumption_weighted_average_of_the_profile():
    pt = PassThroughAssumptions()
    expected = sum(
        w * p
        for w, p in zip(
            QUARTERLY_CONSUMPTION_WEIGHTS_GAS, CAP_PHASE_IN_PROFILE, strict=True
        )
    )
    assert pt.annual_phase_in_gas == pytest.approx(expected)
    assert pt.annual_phase_in_gas == pytest.approx(0.5296, abs=1e-4)
    assert pt.annual_phase_in_electricity == pytest.approx(0.5166, abs=1e-4)
    # It is an average of the profile, so it lies inside the profile's range and
    # strictly below the peak: the paper's "not the peak".
    assert (
        min(CAP_PHASE_IN_PROFILE) < pt.annual_phase_in_gas < max(CAP_PHASE_IN_PROFILE)
    )


def test_annual_phase_in_is_flat_when_the_profile_is_flat():
    pt = PassThroughAssumptions(phase_in_profile=(0.5,) * len(CAP_PHASE_IN_PROFILE))
    assert pt.annual_phase_in_gas == pytest.approx(0.5)
    assert pt.annual_phase_in_electricity == pytest.approx(0.5)


def test_consumption_weights_are_validated():
    with pytest.raises(ValueError, match="same length"):
        PassThroughAssumptions(consumption_weights_gas=(1.0, 1.0))
    n = len(CAP_PHASE_IN_PROFILE)
    with pytest.raises(ValueError, match="non-negative"):
        PassThroughAssumptions(consumption_weights_gas=(-1.0,) + (1.0,) * (n - 1))
    with pytest.raises(ValueError, match="sum to zero"):
        PassThroughAssumptions(consumption_weights_electricity=(0.0,) * n)


@pytest.mark.parametrize("key,scenario", ALL)
def test_annual_retail_shock_is_the_damped_steady_state(key, scenario):
    annual, steady = scenario.annual_retail_shock, scenario.retail_shock
    pt = scenario.pass_through
    assert annual.gas_pct_change == pytest.approx(
        pt.annual_phase_in_gas * steady.gas_pct_change
    )
    assert annual.electricity_pct_change == pytest.approx(
        pt.annual_phase_in_electricity * steady.electricity_pct_change
    )
    assert abs(annual.gas_pct_change) <= abs(steady.gas_pct_change)
    # Electricity stays strictly below gas: the asymmetry survives the damping.
    assert annual.electricity_pct_change < annual.gas_pct_change


# --- A3: the symmetric-damping specification ------------------------------


def test_symmetric_scenario_damps_both_legs_by_one_fraction():
    sym = get_scenario("realised_2026_symmetric")
    asym = get_scenario("realised_2026")
    assert sym.pass_through.sustained_fraction == SYMMETRIC_SUSTAINED_FRACTION
    assert sym.pass_through.pump_sustained_fraction == SYMMETRIC_SUSTAINED_FRACTION
    # Identical underlying paths — only the damping differs.
    assert sym.gas == asym.gas
    assert sym.pump == asym.pump
    assert sym.oil == asym.oil
    # The pump leg is the same as the main specification's; the gas leg is not.
    assert sym.sustained_pump_changes == pytest.approx(asym.sustained_pump_changes)
    # Round-3: the solved gas fraction (0.765) is now ABOVE the common pump
    # fraction (0.650), where before it was 0.199 and far below. Imposing
    # symmetry therefore *lowers* the gas leg instead of raising it — the
    # specification still breaks the cap calibration, in the other direction.
    assert sym.retail_shock.gas_pct_change != asym.retail_shock.gas_pct_change
    assert sym.retail_shock.gas_pct_change < asym.retail_shock.gas_pct_change


def test_symmetric_scenario_breaks_the_cap_anchor_as_documented():
    """It buys symmetry at the price of the external anchor. Both are reported."""
    asym = get_scenario("realised_2026").cap_step(scen.CAP_ANCHOR_QUARTER)
    sym = get_scenario("realised_2026_symmetric").cap_step(scen.CAP_ANCHOR_QUARTER)
    assert asym.cap_pct_change == pytest.approx(scen.CAP_ANCHOR_PCT)
    assert sym.cap_pct_change != pytest.approx(scen.CAP_ANCHOR_PCT)
    assert sym.cap_gbp != pytest.approx(1768.0)
    assert "cap calibration" in get_scenario("realised_2026_symmetric").notes


def test_symmetric_fraction_is_the_pump_profile_arithmetic_not_the_cap_anchor():
    assert SYMMETRIC_SUSTAINED_FRACTION == REALISED_PUMP_SUSTAINED_FRACTION
    assert SYMMETRIC_SUSTAINED_FRACTION != REALISED_SUSTAINED_FRACTION


# --------------------------------------------------------------------------
# Round-2 finding 1: ONE window, both legs
# --------------------------------------------------------------------------


def test_the_modelled_window_is_twelve_months_from_the_shock():
    assert scen.MODELLED_WINDOW_MONTHS[0] == scen.SHOCK_ONSET_MONTH
    assert len(scen.MODELLED_WINDOW_MONTHS) == 12
    assert scen.MODELLED_WINDOW_MONTHS[-1] == "2027-02"
    assert scen.MODELLED_WINDOW_START == "2026-03"


def test_months_between_is_inclusive_and_rolls_the_year():
    assert scen.months_between("2026-11", "2027-02") == (
        "2026-11",
        "2026-12",
        "2027-01",
        "2027-02",
    )
    assert scen.months_between("2026-05", "2026-05") == ("2026-05",)
    with pytest.raises(ValueError, match="precedes"):
        scen.months_between("2026-05", "2026-04")


def test_quarter_helpers():
    assert scen.quarter_of("2026-01") == "2026Q1"
    assert scen.quarter_of("2026-03") == "2026Q1"
    assert scen.quarter_of("2026-04") == "2026Q2"
    assert scen.quarter_of("2026-12") == "2026Q4"
    assert scen.quarters_of(scen.MODELLED_WINDOW_MONTHS) == (
        "2026Q1",
        "2026Q2",
        "2026Q3",
        "2026Q4",
        "2027Q1",
    )


def test_both_legs_are_derived_from_the_same_window():
    """The round-2 fix in one assertion.

    The cap phase-in weights and the pump damping fraction must be functions of
    the *same* month list. Before the revision the domestic leg averaged over
    2026Q4-2027Q3 while the pump fraction was derived over calendar 2026, and
    the two were summed and labelled "2026".
    """
    assert scen.CAP_QUARTER_LABELS == scen.quarters_of(scen.MODELLED_WINDOW_MONTHS)
    assert scen.QUARTERLY_CONSUMPTION_WEIGHTS_GAS == scen.quarterly_consumption_weights(
        scen.MODELLED_WINDOW_MONTHS, scen.MONTHLY_CONSUMPTION_WEIGHTS_GAS
    )
    assert scen.REALISED_PUMP_SUSTAINED_FRACTION == scen.pump_sustained_fraction(
        scen.MODELLED_WINDOW_MONTHS
    )
    # Every month of the window is priced by exactly one of the cap quarters.
    for month in scen.MODELLED_WINDOW_MONTHS:
        assert scen.quarter_of(month) in scen.CAP_QUARTER_LABELS


def test_the_window_choice_actually_moves_the_pump_fraction():
    """If it did not, the finding would be cosmetic. It is not."""
    assert scen.PUMP_SUSTAINED_FRACTION_CALENDAR_2026 == pytest.approx(0.5917, abs=1e-4)
    assert scen.REALISED_PUMP_SUSTAINED_FRACTION == pytest.approx(0.65, abs=1e-4)
    assert (
        scen.REALISED_PUMP_SUSTAINED_FRACTION
        > scen.PUMP_SUSTAINED_FRACTION_CALENDAR_2026
    )


def test_pump_profile_is_zero_before_the_shock():
    for month in ("2026-01", "2026-02"):
        assert scen.PUMP_PEAK_MONTHLY_PROFILE[month] == 0.0
    assert scen.PUMP_PEAK_MONTHLY_PROFILE[scen.SHOCK_ONSET_MONTH] > 0.0
    assert max(scen.PUMP_PEAK_MONTHLY_PROFILE.values()) == 1.0


def test_pump_sustained_fraction_raises_on_a_month_it_cannot_price():
    with pytest.raises(KeyError, match="2028-01"):
        scen.pump_sustained_fraction(("2028-01",))


# --- the phase-in profile is a function of the lag ------------------------


def test_phase_in_profile_is_derived_from_the_lag():
    assert scen.CAP_PHASE_IN_PROFILE == scen.cap_phase_in_profile(scen.CAP_LAG_QUARTERS)
    assert scen.CAP_PHASE_IN_PROFILE[0] == 0.0  # 2026Q1's window closed pre-war
    assert scen.CAP_PHASE_IN_PROFILE[1] == 0.0  # 2026Q2's closes mid-February


def test_2026q3_is_not_fully_shocked_which_is_the_round3_root_cause():
    """The contradiction three round-3 referees found, asserted away.

    2026Q3 cannot be both the un-shocked baseline and the fully-shocked
    numerator. It is no longer either: it is a partly-shocked quarter, and the
    un-shocked baseline is a constructed counterfactual.
    """
    index = scen.CAP_QUARTER_LABELS.index(scen.CAP_BASE_QUARTER)
    assert 0.0 < scen.CAP_PHASE_IN_PROFILE[index] < 1.0
    assert scen.PREWAR_COUNTERFACTUAL_CAP_GBP < scen.OFGEM_CAP_JUL_2026_GBP
    assert scen.BASELINE_CAP_GBP == scen.PREWAR_COUNTERFACTUAL_CAP_GBP
    # The July cap already carries a material part of the conflict.
    assert scen.CAP_BASE_PCT > 0.10


def test_the_profile_is_not_monotone_because_the_spike_rolls_out_of_the_window():
    """2027Q1's window sits on the decay, so it prices less than 2026Q4's.

    The old linear ramp was monotone by construction and could not represent
    this. It is a property of an averaging observation window, not a bug.
    """
    profile = scen.CAP_PHASE_IN_PROFILE
    assert profile[-1] < profile[-2]
    assert list(profile) != sorted(profile) or profile[-1] == profile[-2]


@pytest.mark.parametrize("lag", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
def test_phase_in_profile_is_bounded_at_every_lag(lag):
    profile = scen.cap_phase_in_profile(lag)
    assert len(profile) == len(scen.CAP_QUARTER_LABELS)
    assert all(0.0 <= v <= 1.0 for v in profile)
    # No quarter whose observation window closes before the shock may carry any
    # of it, at any lag.
    assert profile[0] == 0.0


def test_observation_window_is_three_months_closing_before_the_charge_period():
    weights = scen.observation_window_weights("2026Q3")
    assert sum(weights.values()) == pytest.approx(1.0)
    # At the central lag the window is mid-February to mid-May 2026.
    assert set(weights) == {"2026-02", "2026-03", "2026-04", "2026-05"}
    assert weights["2026-03"] == pytest.approx(1 / 3)
    assert weights["2026-02"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="positive"):
        scen.observation_window_weights("2026Q3", 0.0)


def test_the_linear_ramp_fixes_the_partial_quarter_misalignment():
    """Round-3: 2026Q1 carries March only and March is post-onset.

    Evaluated at the full quarter's midpoint (mid-February, pre-onset) it
    returned 0.0; evaluated at March's own midpoint it returns 0.111, and the
    annual gas phase-in on the ramp rises from 0.797 to 0.809.
    """
    ramp = scen.LINEAR_RAMP_PHASE_IN_PROFILE
    assert ramp == scen.linear_ramp_phase_in_profile()
    assert ramp[0] == pytest.approx(1 / 9, abs=1e-6)
    pt = PassThroughAssumptions(phase_in_profile=ramp)
    assert pt.annual_phase_in_gas == pytest.approx(0.8089, abs=1e-4)
    # The unfixed version evaluated every quarter at the full-quarter midpoint.
    unfixed = scen.linear_ramp_phase_in_profile(months=("2026-01",))
    assert unfixed[0] == 0.0
    assert PassThroughAssumptions(
        phase_in_profile=unfixed
    ).annual_phase_in_gas == pytest.approx(0.7972, abs=1e-4)


def test_a_longer_lag_pushes_the_shock_out_of_the_window():
    """The mechanism by which lag = 3 made 2026 look like a pump-price event."""
    short = PassThroughAssumptions(phase_in_profile=scen.cap_phase_in_profile(1.5))
    long = PassThroughAssumptions(phase_in_profile=scen.cap_phase_in_profile(3.0))
    assert long.annual_phase_in_gas < short.annual_phase_in_gas


def test_cap_phase_in_profile_rejects_a_non_positive_lag():
    with pytest.raises(ValueError, match="positive"):
        scen.cap_phase_in_profile(0.0)
    with pytest.raises(ValueError, match="positive"):
        scen.linear_ramp_phase_in_profile(0.0)


def test_cap_phase_in_profile_raises_on_a_month_the_gas_profile_cannot_price():
    with pytest.raises(KeyError, match="GAS_PEAK_MONTHLY_PROFILE"):
        scen.cap_phase_in_profile(1.5, ("2030Q1",))


# --- the cap anchor is solved, not written down ---------------------------


def test_cap_anchor_uses_the_confirmed_ofgem_cap_net_of_the_vat_relief():
    assert scen.OFGEM_CAP_OCT_2026_GBP == 1723.0
    assert scen.CAP_ANCHOR_PCT == pytest.approx(
        (1723.0 + 45.0) / scen.PREWAR_COUNTERFACTUAL_CAP_GBP - 1.0
    )
    # Ignoring the unrelated electricity VAT relief would attribute a tax cut
    # to the war and understate the domestic leg.
    assert scen.CAP_ANCHOR_PCT > 1723.0 / scen.PREWAR_COUNTERFACTUAL_CAP_GBP - 1.0
    # And it is measured against the counterfactual, not the observed July cap:
    # that difference is the round-3 root-cause fix and it is worth 17pp.
    old = (1723.0 + 45.0) / scen.OFGEM_CAP_JUL_2026_GBP - 1.0
    assert old == pytest.approx(0.0631, abs=1e-4)
    assert scen.CAP_ANCHOR_PCT > 3 * old


def test_the_calibration_reproduces_both_published_caps_exactly():
    """The identity that replaces the single-anchor fit."""
    v = scen.CAP_VALIDATION
    assert v["modelled_jul_2026_gbp"] == pytest.approx(v["observed_jul_2026_gbp"])
    assert v["modelled_oct_2026_gbp"] == pytest.approx(
        v["observed_oct_2026_vat_adjusted_gbp"]
    )
    scenario = get_scenario("realised_2026")
    assert scenario.cap_step("2026Q3").cap_gbp == pytest.approx(1663.0)
    assert scenario.cap_step("2026Q4").cap_gbp == pytest.approx(1768.0)
    assert 0.0 < scen.REALISED_SUSTAINED_FRACTION <= 1.0


def test_the_solver_refuses_a_degenerate_pair_of_observation_windows():
    """Two identical windows cannot separate two different caps.

    This is exactly what the pre-round-3 profile did — 2026Q3 and 2026Q4 both at
    1.0 — and it is why the counterfactual had to be assumed rather than solved.
    """
    with pytest.raises(ValueError, match="not identified"):
        scen.solve_cap_calibration(scen.REALISED_GAS_PCT_CHANGE, 1.0, 1.0)


def test_the_solver_refuses_an_implied_fraction_above_one():
    with pytest.raises(ValueError, match="sustained fraction"):
        # A tiny gas move cannot produce the observed cap step.
        scen.solve_cap_calibration(0.01, 0.6333, 0.9333)


def test_the_implemented_pass_through_coefficients_are_named():
    """Round-3 finding 4: the paper quotes coefficients the code does not use."""
    assert scen.BILL_LEVEL_PASS_THROUGH == pytest.approx(
        0.45 * 0.45 + 0.55 * 0.85 * 0.35
    )
    assert scen.BILL_LEVEL_PASS_THROUGH == pytest.approx(0.3661, abs=1e-4)
    # Equation (1) states phi ~ 0.45, which is the *gas-bill* share, not this.
    assert scen.BILL_LEVEL_PASS_THROUGH != scen.WHOLESALE_SHARE_GAS_BILL
    assert scen.ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO == pytest.approx(
        0.85 * 0.35 / 0.45
    )
    assert scen.ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO == pytest.approx(0.6611, abs=1e-4)
    # The stated reduced form phi_elec = psi . phi_gas would give 0.85.
    assert scen.ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO != scen.MARGINAL_PRICING_SHARE
    # And they are what the scenario actually applies.
    scenario = get_scenario("realised_2026")
    shock = scenario.retail_shock
    assert shock.electricity_pct_change / shock.gas_pct_change == pytest.approx(
        scen.ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO
    )
    with pytest.raises(ValueError):
        scen.electricity_to_gas_pass_through_ratio(wholesale_share_gas_bill=0.0)


def test_the_gas_profile_sweep_moves_the_counterfactual_and_stays_honest():
    """The identification the counterfactual rests on, swept."""
    default = scen.gas_profile_variant()
    assert default == dict(scen.GAS_PEAK_MONTHLY_PROFILE)
    caps = []
    for kwargs in ({"shift_months": 1}, {"flatten": 0.8}, {"flatten": 1.2}):
        profile = scen.gas_profile_variant(**kwargs)
        phase_in = scen.cap_phase_in_profile(profile=profile)
        calibration = scen.solve_cap_calibration(
            scen.REALISED_GAS_PCT_CHANGE, phase_in[2], phase_in[3]
        )
        caps.append(calibration.prewar_cap_gbp)
        assert calibration.prewar_cap_gbp < scen.OFGEM_CAP_JUL_2026_GBP
    # It genuinely moves — the sweep is not decorative.
    assert max(caps) - min(caps) > 50.0
    # A flatter path implies a lower counterfactual and a bigger conflict move.
    flat = scen.cap_phase_in_profile(profile=scen.gas_profile_variant(flatten=0.8))
    steep = scen.cap_phase_in_profile(profile=scen.gas_profile_variant(flatten=1.2))
    lo = scen.solve_cap_calibration(scen.REALISED_GAS_PCT_CHANGE, flat[2], flat[3])
    hi = scen.solve_cap_calibration(scen.REALISED_GAS_PCT_CHANGE, steep[2], steep[3])
    assert lo.prewar_cap_gbp < hi.prewar_cap_gbp


def test_the_realised_scenario_reproduces_the_cap_anchor_exactly():
    step = get_scenario("realised_2026").cap_step(scen.CAP_ANCHOR_QUARTER)
    assert step.cap_pct_change == pytest.approx(scen.CAP_ANCHOR_PCT)
    assert step.cap_gbp == pytest.approx(
        scen.OFGEM_CAP_OCT_2026_GBP + scen.OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP
    )


@pytest.mark.parametrize("lag", [1.0, 1.5, 2.0, 3.0, 4.0])
def test_the_anchor_holds_at_every_lag_when_the_fraction_is_re_solved(lag):
    """``docs/FIXES.md`` A4: the anchor pins the product, not either factor.

    Writing the sustained fraction as a literal while the phase-in profile
    changed underneath it is how the anchor silently broke. Solving for it makes
    the anchor hold by construction at any lag, window or bill-share
    assumption — which is what lets the cap-lag appendix and the headline be the
    same function of the same parameter.
    """
    profile = scen.cap_phase_in_profile(lag)
    index = scen.CAP_QUARTER_LABELS.index(scen.CAP_ANCHOR_QUARTER)
    if profile[index] <= 0:
        pytest.skip(f"lag {lag} puts no pass-through in the anchor quarter")
    fraction = scen.sustained_fraction_for_cap_anchor(
        scen.REALISED_GAS_PCT_CHANGE, profile[index]
    )
    scenario = dataclasses.replace(
        get_scenario("realised_2026"),
        pass_through=dataclasses.replace(
            get_scenario("realised_2026").pass_through,
            lag_quarters=lag,
            phase_in_profile=profile,
            sustained_fraction=min(1.0, fraction),
        ),
    )
    step = scenario.cap_step(scen.CAP_ANCHOR_QUARTER)
    if fraction <= 1.0:
        assert step.cap_pct_change == pytest.approx(scen.CAP_ANCHOR_PCT)
        assert fraction * profile[index] == pytest.approx(
            scen.REALISED_SUSTAINED_FRACTION * scen.CAP_PHASE_IN_PROFILE[index]
        )


def test_sustained_fraction_solver_refuses_a_quarter_with_no_pass_through():
    with pytest.raises(ValueError, match="no pass-through"):
        scen.sustained_fraction_for_cap_anchor(scen.REALISED_GAS_PCT_CHANGE, 0.0)


def test_quarterly_consumption_weights_renormalise_a_partial_window():
    weights = scen.quarterly_consumption_weights(
        ("2026-01", "2026-02"), scen.MONTHLY_CONSUMPTION_WEIGHTS_GAS
    )
    assert sum(weights) == pytest.approx(1.0)
    assert len(weights) == 1
    with pytest.raises(ValueError, match="sum to zero"):
        scen.quarterly_consumption_weights(("2026-01",), (0.0,) * 12)


# --------------------------------------------------------------------------
# Round-3 finding 10: any scenario can be re-annualised onto any window
# --------------------------------------------------------------------------


def test_on_window_rebuilds_everything_the_window_determines():
    main = get_scenario("realised_2026")
    rf = scen.on_window(main, scen.CALENDAR_2026_MONTHS)
    assert rf.quarter_labels == ("2026Q1", "2026Q2", "2026Q3", "2026Q4")
    assert len(rf.pass_through.phase_in_profile) == 4
    assert len(rf.pass_through.consumption_weights_gas) == 4
    assert sum(rf.pass_through.consumption_weights_gas) == pytest.approx(1.0)
    # The pump fraction is re-derived on the RF window (0.592, not 0.650).
    assert rf.pass_through.pump_sustained_fraction == pytest.approx(
        scen.PUMP_SUSTAINED_FRACTION_CALENDAR_2026
    )
    # The calibration is a property of the wholesale path, not of the window,
    # so it is carried over and both published caps still reproduce.
    assert rf.pass_through.sustained_fraction == main.pass_through.sustained_fraction
    assert rf.cap_step("2026Q4").cap_gbp == pytest.approx(1768.0)


def test_the_peak_fuel_bound_can_be_stated_on_the_rf_window_like_for_like():
    """Round-3 finding 10: the published bracket mixed two different windows."""
    bound = scen.on_window(
        get_scenario("realised_2026_peak_fuel"), scen.CALENDAR_2026_MONTHS
    )
    main = scen.on_window(get_scenario("realised_2026"), scen.CALENDAR_2026_MONTHS)
    # The bound is still undamped on the fuel leg, on this window too.
    assert bound.pass_through.pump_sustained_fraction == 1.0
    assert main.pass_through.pump_sustained_fraction < 1.0
    assert bound.sustained_pump_changes == pytest.approx((0.20, 0.36))
    # And both are now annualised over the SAME window as each other.
    assert bound.quarter_labels == main.quarter_labels
    assert (
        bound.pass_through.consumption_weights_gas
        == main.pass_through.consumption_weights_gas
    )


def test_the_module_no_longer_points_at_a_module_that_does_not_exist():
    """Round-3 finding 11: the docstring referenced ``uk_iran_conflict.shocks``."""
    assert "uk_iran_conflict.shocks" not in scen.__doc__
    assert "uk_iran_conflict.incidence" in scen.__doc__


def test_the_peak_fuel_notes_no_longer_quote_a_stale_fraction_or_window():
    """Round-3 finding 11: "damping the gas peak to 0.36", "full calendar year"."""
    notes = get_scenario("realised_2026_peak_fuel").notes
    description = get_scenario("realised_2026_peak_fuel").description
    assert "0.36 is internally inconsistent" not in notes
    assert "full calendar year" not in description
    assert "MODELLED_WINDOW_LABEL" in description

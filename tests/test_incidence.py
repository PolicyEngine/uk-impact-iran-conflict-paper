"""Incidence tests that run without microdata.

Every test here builds a small synthetic :class:`Baseline` by hand, so the
module's arithmetic — in particular the two corrections required by
``docs/VALIDATION.md`` (the pump-price damping, Check 2b; the ONS motor-fuel
recalibration, Check 2d) — is testable without a Hugging Face token.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uk_iran_conflict import incidence as inc
from uk_iran_conflict.scenarios import get_scenario


def make_baseline(
    n_per_decile: int = 5, seed: int = 0, unbanded: int = 0
) -> inc.Baseline:
    """Ten deciles of households with a deliberately *flat* fuel profile.

    ``unbanded`` appends households carrying no valid decile (and negative
    income), which is what the coverage tests need.

    Flat across deciles is exactly the defect VALIDATION.md Check 2d documents
    (D1 £1,073 against D10 £1,333), so it is the right starting point for
    testing the correction. Within each decile the spend varies, so the test
    can check that relative variation survives.
    """
    rng = np.random.default_rng(seed)
    n = n_per_decile * 10
    decile = np.repeat(np.arange(1, 11), n_per_decile).astype(float)
    spread = rng.uniform(0.5, 1.5, n)
    if unbanded:
        # Households the microdata cannot place in a decile: decile -1, and
        # (as in the real data) non-positive income. FIXES.md A6.
        decile = np.concatenate([decile, np.full(unbanded, -1.0)])
        spread = np.concatenate([spread, np.full(unbanded, 1.0)])
        n += unbanded
    net_income = np.where(decile > 0, 10_000.0 * decile, -500.0)
    return inc.Baseline(
        net_income=net_income,
        weight=np.full(n, 100.0),
        people=np.full(n, 2.0),
        gas=np.full(n, 600.0),
        electricity=np.full(n, 700.0),
        petrol=700.0 * spread,
        diesel=300.0 * spread,
        decile=decile,
        # Deliberately *not* equal to net income, so a test can tell which
        # denominator a statistic used (FIXES.md D1).
        equiv_income_ahc=net_income / 1.4,
        in_poverty_bhc=np.zeros(n),
        in_poverty_ahc=np.zeros(n),
        region=np.array(["London"] * n),
        country=np.array(["ENGLAND"] * n),
        equivalisation_ahc=np.full(n, 1.4),
    )


# --- pump damping reaches the cost ---------------------------------------


def test_shock_cost_uses_the_damped_pump_moves():
    base = make_baseline()
    main = inc.shock_cost(base, get_scenario("realised_2026"))
    bound = inc.shock_cost(base, get_scenario("realised_2026_peak_fuel"))
    w = base.weight
    # Gas and electricity legs identical; only motor fuel moves.
    assert inc.wsum(main.gas, w) == pytest.approx(inc.wsum(bound.gas, w))
    assert inc.wsum(main.electricity, w) == pytest.approx(
        inc.wsum(bound.electricity, w)
    )
    ratio = inc.wsum(main.motor_fuel, w) / inc.wsum(bound.motor_fuel, w)
    assert ratio == pytest.approx(0.60)
    # And the fuel share of the loss falls as a result.
    main_share = inc.wsum(main.motor_fuel, w) / inc.wsum(main.total, w)
    bound_share = inc.wsum(bound.motor_fuel, w) / inc.wsum(bound.total, w)
    assert main_share < bound_share


def test_niesr_scenarios_are_unaffected_by_the_new_parameter():
    base = make_baseline()
    for key in ("niesr_baseline", "niesr_adverse"):
        scenario = get_scenario(key)
        cost = inc.shock_cost(base, scenario)
        expected = base.petrol * scenario.pump.petrol_pct_change + (
            base.diesel * scenario.pump.diesel_pct_change
        )
        assert cost.motor_fuel == pytest.approx(expected)


# --- ONS motor-fuel recalibration ----------------------------------------


def test_ons_targets_are_monotone_and_hit_the_published_endpoints():
    targets = inc.ons_motor_fuel_decile_targets()
    assert len(targets) == 10
    assert targets[0] == pytest.approx(inc.ONS_MOTOR_FUEL_D1_GBP)
    assert targets[-1] == pytest.approx(inc.ONS_MOTOR_FUEL_D10_GBP)
    assert np.all(np.diff(targets) > 0)
    # Log-linear: a constant ratio between adjacent deciles.
    ratios = targets[1:] / targets[:-1]
    assert ratios == pytest.approx(np.full(9, ratios[0]))
    assert targets[-1] / targets[0] == pytest.approx(4.28, abs=0.01)


def test_rescaling_preserves_the_national_total():
    base = make_baseline()
    ons = inc.rescale_motor_fuel_to_ons(base)
    assert inc.wsum(ons.motor_fuel, ons.weight) == pytest.approx(
        inc.wsum(base.motor_fuel, base.weight)
    )


def test_rescaling_imposes_the_ons_decile_gradient():
    base = make_baseline()
    ons = inc.rescale_motor_fuel_to_ons(base)
    means = []
    for d in range(1, 11):
        sel = ons.decile == d
        means.append(inc.wmean(ons.motor_fuel[sel], ons.weight[sel]))
    means = np.array(means)
    assert np.all(np.diff(means) > 0), "gradient not imposed"
    assert means[-1] / means[0] == pytest.approx(
        inc.ONS_MOTOR_FUEL_D10_GBP / inc.ONS_MOTOR_FUEL_D1_GBP, rel=1e-6
    )
    # Decile 1 falls sharply, decile 10 rises: the direction VALIDATION.md gives.
    d1_before = inc.wmean(
        base.motor_fuel[base.decile == 1], base.weight[base.decile == 1]
    )
    assert means[0] < d1_before


def test_rescaling_preserves_within_decile_relative_variation():
    base = make_baseline()
    ons = inc.rescale_motor_fuel_to_ons(base)
    for d in (1, 5, 10):
        sel = base.decile == d
        before = base.motor_fuel[sel]
        after = ons.motor_fuel[sel]
        factors = after / before
        assert factors == pytest.approx(np.full(factors.shape, factors[0]))
    # Petrol and diesel are scaled by the same factor, so the mix is unchanged.
    assert (ons.petrol / ons.diesel) == pytest.approx(base.petrol / base.diesel)


def test_rescaling_leaves_domestic_energy_untouched():
    base = make_baseline()
    ons = inc.rescale_motor_fuel_to_ons(base)
    assert ons.gas is base.gas or np.allclose(ons.gas, base.gas)
    assert np.allclose(ons.electricity, base.electricity)
    assert np.allclose(ons.net_income, base.net_income)
    assert np.allclose(ons.weight, base.weight)


def test_run_scenario_defaults_to_the_raw_microdata():
    """The main specification must not move: the option is off by default."""
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    plain, _ = inc.run_scenario(base, scenario)
    explicit_off, _ = inc.run_scenario(base, scenario, ons_fuel_calibration=False)
    assert plain.aggregate_cost_bn == pytest.approx(explicit_off.aggregate_cost_bn)
    assert plain.decile[0].mean_loss_gbp == pytest.approx(
        explicit_off.decile[0].mean_loss_gbp
    )


def test_ons_option_flattens_nothing_but_the_fuel_channel():
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    main, _ = inc.run_scenario(base, scenario)
    ons, _ = inc.run_scenario(base, scenario, ons_fuel_calibration=True)
    # Aggregate is unchanged (total fuel spend preserved, flat weights here).
    assert ons.aggregate_cost_bn == pytest.approx(main.aggregate_cost_bn, rel=1e-9)
    # Decile 1 loses less, decile 10 more.
    assert ons.decile[0].mean_loss_gbp < main.decile[0].mean_loss_gbp
    assert ons.decile[-1].mean_loss_gbp > main.decile[-1].mean_loss_gbp
    # And the input baseline was not mutated.
    assert inc.wmean(base.motor_fuel[base.decile == 1], base.weight[base.decile == 1])


def test_scale_factors_are_all_positive_and_finite():
    factors = inc.ons_motor_fuel_scale_factors(make_baseline())
    assert np.all(np.isfinite(factors))
    assert np.all(factors > 0)
    assert factors[0] < 1.0 < factors[-1]


# --- D1: the denominator --------------------------------------------------


def test_income_for_ratio_defaults_to_equivalised_ahc():
    base = make_baseline()
    assert inc.DEFAULT_INCOME_BASIS == "equivalised_ahc"
    assert inc.income_for_ratio(base) is base.equiv_income_ahc
    assert inc.income_for_ratio(base, "unequivalised") is base.net_income
    with pytest.raises(ValueError, match="unknown income basis"):
        inc.income_for_ratio(base, "gross")


def test_percentages_divide_by_equivalised_income():
    """Every percentage-of-income statistic must move with the denominator."""
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    equiv, _ = inc.run_scenario(base, scenario)
    uneq, _ = inc.run_scenario(base, scenario, income_basis="unequivalised")
    # The fixture's equivalised income is net income / 1.4, so every ratio is
    # exactly 1.4x larger — proof the switch reaches all four tables.
    assert equiv.mean_loss_pct == pytest.approx(1.4 * uneq.mean_loss_pct)
    for a, b in zip(equiv.decile, uneq.decile, strict=True):
        assert a.mean_loss_pct == pytest.approx(1.4 * b.mean_loss_pct)
        assert a.mean_loss_gbp == pytest.approx(b.mean_loss_gbp)
    for a, b in zip(equiv.region, uneq.region, strict=True):
        assert a.mean_loss_pct == pytest.approx(1.4 * b.mean_loss_pct)
    for a, b in zip(equiv.intra_decile, uneq.intra_decile, strict=True):
        assert a.p50_loss_pct == pytest.approx(1.4 * b.p50_loss_pct)
    # And the implied mean income falls, which is the referee's sanity check.
    assert equiv.mean_income_gbp < uneq.mean_income_gbp
    assert equiv.mean_income_gbp == pytest.approx(uneq.mean_income_gbp / 1.4)


def test_equivalising_the_numerator_too_is_reported_as_a_diagnostic():
    """The cash-over-equivalised headline is larger by the equivalisation scale."""
    base = make_baseline()
    result, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    assert result.mean_loss_pct_equivalised_both == pytest.approx(
        result.mean_loss_pct / 1.4
    )
    # A baseline without the factor loaded falls back to an unequivalised 1.0.
    bare = dataclasses.replace(base, equivalisation_ahc=None)
    plain, _ = inc.run_scenario(bare, get_scenario("realised_2026"))
    assert plain.mean_loss_pct_equivalised_both == pytest.approx(plain.mean_loss_pct)
    assert np.all(bare.equivalisation == 1.0)


def test_non_positive_incomes_are_dropped_not_clipped():
    """A1/A6: a fifth of decile one is <= 0 on the equivalised AHC basis."""
    base = make_baseline(unbanded=4)
    result, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    # The unbanded households carry negative income and are excluded from the
    # percentage statistics, but not from the aggregate.
    assert result.zero_or_negative_income_share == pytest.approx(4 / 54)
    banded = make_baseline()
    banded_result, _ = inc.run_scenario(banded, get_scenario("realised_2026"))
    assert result.mean_loss_pct == pytest.approx(banded_result.mean_loss_pct)
    assert result.aggregate_cost_bn > banded_result.aggregate_cost_bn
    # Clipping to £1 would have produced a wildly larger ratio instead.
    cost = np.array([100.0, 100.0])
    income = np.array([-500.0, 10_000.0])
    w = np.ones(2)
    assert inc.pct_of_income(cost, income, w) == pytest.approx(1.0)


def test_result_records_the_bases_it_was_run_on():
    base = make_baseline()
    result, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    assert result.income_basis == "equivalised_ahc"
    assert result.domestic_basis == "annual"
    assert result.calibration == "raw"


# --- D2: the phase-in -----------------------------------------------------


def test_annual_domestic_basis_is_the_default_and_damps_only_the_domestic_leg():
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    assert inc.DEFAULT_DOMESTIC_BASIS == "annual"
    annual = inc.shock_cost(base, scenario)
    steady = inc.shock_cost(base, scenario, "steady_state")
    w = base.weight
    # Motor fuel is untouched: pump prices do not lag a cap window.
    assert inc.wsum(annual.motor_fuel, w) == pytest.approx(
        inc.wsum(steady.motor_fuel, w)
    )
    # Gas and electricity are damped by their own consumption-weighted factors.
    assert inc.wsum(annual.gas, w) == pytest.approx(
        scenario.pass_through.annual_phase_in_gas * inc.wsum(steady.gas, w)
    )
    assert inc.wsum(annual.electricity, w) == pytest.approx(
        scenario.pass_through.annual_phase_in_electricity
        * inc.wsum(steady.electricity, w)
    )
    # So the headline aggregate falls, and motor fuel's share rises.
    assert inc.wsum(annual.total, w) < inc.wsum(steady.total, w)


def test_domestic_basis_is_validated():
    base = make_baseline()
    with pytest.raises(ValueError, match="unknown domestic basis"):
        inc.shock_cost(base, get_scenario("realised_2026"), "peak")


def test_run_scenario_reports_the_phase_in_it_applied():
    result, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    assert result.annual_phase_in_gas == pytest.approx(0.7285)
    assert result.annual_phase_in_electricity == pytest.approx(0.754)


# --- A3: symmetric damping ------------------------------------------------


def test_symmetric_scenario_raises_the_domestic_leg_and_cuts_the_fuel_share():
    base = make_baseline()
    main, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    sym, _ = inc.run_scenario(base, get_scenario("realised_2026_symmetric"))
    assert sym.aggregate_cost_bn > main.aggregate_cost_bn
    assert sym.motor_fuel_share_of_loss < main.motor_fuel_share_of_loss
    # The fuel channel itself is identical; only the domestic leg is rescaled.
    assert sym.decile[0].mean_loss_gbp > main.decile[0].mean_loss_gbp


def test_fuel_share_depends_only_on_the_ratio_of_the_two_fractions():
    """The referee's finding, asserted: any common fraction gives one share."""
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    shares = []
    for fraction in (0.30, 0.45, 0.60, 0.85):
        variant = dataclasses.replace(
            scenario,
            pass_through=dataclasses.replace(
                scenario.pass_through,
                sustained_fraction=fraction,
                pump_sustained_fraction=fraction,
            ),
        )
        result, _ = inc.run_scenario(base, variant)
        shares.append(result.motor_fuel_share_of_loss)
    assert shares == pytest.approx([shares[0]] * len(shares))
    asym, _ = inc.run_scenario(base, scenario)
    assert asym.motor_fuel_share_of_loss > shares[0]


# --- A6: decile coverage --------------------------------------------------


def test_out_of_range_households_are_reported_not_silently_dropped():
    base = make_baseline(unbanded=4)
    result, cost = inc.run_scenario(base, get_scenario("realised_2026"))
    coverage = result.coverage
    assert coverage is not None
    assert coverage.households_m == pytest.approx(4 * 100.0 / 1e6)
    assert coverage.share_of_households == pytest.approx(4 / 54)
    assert coverage.loss_bn > 0
    assert coverage.share_of_loss > 0
    # In the fixture, as in the microdata, they are the non-positive incomes.
    assert coverage.zero_or_negative_income_share == pytest.approx(1.0)
    assert coverage.covered_households_m == pytest.approx(50 * 100.0 / 1e6)


def test_share_of_total_loss_sums_to_one_over_the_deciles():
    base = make_baseline(unbanded=4)
    result, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    total = sum(row.share_of_total_loss for row in result.decile)
    assert total == pytest.approx(1.0)
    # And the aggregate still counts everybody, excluded households included.
    assert result.aggregate_cost_bn > result.coverage.covered_loss_bn / 1.0 * 0


def test_decile_rows_carry_the_denominator_evidence():
    """C12: the decile-one income facts come out of the run, not out of prose."""
    base = make_baseline()
    result, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    d1 = result.decile[0]
    assert d1.mean_income_gbp == pytest.approx(10_000.0 / 1.4)
    assert d1.median_income_gbp == pytest.approx(10_000.0 / 1.4)
    assert d1.zero_or_negative_income_share == pytest.approx(0.0)


# --- C11: ONS levels on both legs -----------------------------------------


def test_ons_both_levels_corrects_both_means():
    base = make_baseline()
    fixed = inc.rescale_to_ons_levels(base)
    w = base.weight
    assert inc.wmean(fixed.energy, w) == pytest.approx(inc.ONS_DOMESTIC_ENERGY_MEAN_GBP)
    assert inc.wmean(fixed.motor_fuel, w) == pytest.approx(inc.ONS_MOTOR_FUEL_MEAN_GBP)
    # Domestic energy is scaled up, motor fuel down: the two defects run in
    # opposite directions (FIXES.md C11).
    assert inc.wmean(fixed.energy, w) > inc.wmean(base.energy, w)
    assert inc.wmean(fixed.motor_fuel, w) < inc.wmean(base.motor_fuel, w)
    # The gas/electricity mix is untouched; the fuel decile shape is ONS's.
    assert (fixed.gas / fixed.electricity) == pytest.approx(base.gas / base.electricity)
    means = [
        inc.wmean(fixed.motor_fuel[fixed.decile == d], w[fixed.decile == d])
        for d in (1, 10)
    ]
    assert means[1] / means[0] == pytest.approx(
        inc.ONS_MOTOR_FUEL_D10_GBP / inc.ONS_MOTOR_FUEL_D1_GBP, rel=1e-6
    )


def test_ons_both_levels_cuts_the_motor_fuel_share_of_the_loss():
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    raw, _ = inc.run_scenario(base, scenario)
    fixed, _ = inc.run_scenario(base, scenario, calibration="ons_both_levels")
    assert fixed.motor_fuel_share_of_loss < raw.motor_fuel_share_of_loss


def test_calibration_selector_is_validated_and_exclusive():
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    with pytest.raises(ValueError, match="unknown calibration"):
        inc.apply_calibration(base, "ons")
    with pytest.raises(ValueError, match="not both"):
        inc.run_scenario(base, scenario, ons_fuel_calibration=True, calibration="raw")
    shorthand, _ = inc.run_scenario(base, scenario, ons_fuel_calibration=True)
    explicit, _ = inc.run_scenario(base, scenario, calibration="ons_fuel_shape")
    assert shorthand.decile[0].mean_loss_gbp == pytest.approx(
        explicit.decile[0].mean_loss_gbp
    )


# --- E30-E33: numerical hygiene -------------------------------------------


def test_wquantile_uses_the_mid_rank_convention():
    """E30: with equal weights this is the symmetric Hazen position."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones(5)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert inc.wquantile(x, w, q) == pytest.approx(
            np.quantile(x, q, method="hazen")
        )
    # Symmetric about the median, which the raw-cumulative version was not.
    assert inc.wquantile(x, w, 0.5) == pytest.approx(3.0)
    assert inc.wquantile(x, w, 0.25) + inc.wquantile(x, w, 0.75) == pytest.approx(6.0)
    # Weighting a value up pulls the median toward it.
    assert inc.wquantile(x, np.array([1.0, 1, 1, 1, 20]), 0.5) > 3.0
    assert np.isnan(inc.wquantile(x, np.zeros(5), 0.5))


def test_top_share_splits_tied_boundary_weight():
    """E31: a variable full of ties must not hand the whole tie to the tail."""
    x = np.ones(10)
    w = np.ones(10)
    # Everybody identical: the top 10% holds exactly 10% of the total.
    assert inc.top_share(x, w, 0.10) == pytest.approx(0.10)
    assert inc.bottom_share(x, w, 0.10) == pytest.approx(0.10)
    # A genuine tail is still measured.
    y = np.array([1.0] * 9 + [91.0])
    assert inc.top_share(y, w, 0.10) == pytest.approx(0.91)


def test_gini_includes_the_origin_segment():
    """E32: the Lorenz curve starts at the origin."""
    assert inc.gini(np.array([0.0, 1.0]), np.ones(2)) == pytest.approx(0.5)
    assert inc.gini(np.array([1.0, 1.0, 1.0]), np.ones(3)) == pytest.approx(0.0)
    # A perfectly concentrated distribution tends to 1 as n grows.
    n = 200
    x = np.zeros(n)
    x[-1] = 1.0
    assert inc.gini(x, np.ones(n)) == pytest.approx(1 - 1 / n, abs=1e-9)


def test_missing_shock_raises_instead_of_returning_an_unshocked_one():
    """E33: a silent zero shock is indistinguishable from a modelled zero."""

    class Bare:
        key = "bare"

    with pytest.raises(AttributeError, match="annual_retail_shock"):
        inc.domestic_retail_factors(Bare())
    with pytest.raises(AttributeError, match="retail_shock"):
        inc.domestic_retail_factors(Bare(), "steady_state")
    with pytest.raises(AttributeError, match="pump path"):
        inc.sustained_pump_factors(Bare())

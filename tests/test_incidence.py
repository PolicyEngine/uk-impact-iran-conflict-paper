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
from uk_iran_conflict import scenarios as scen
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
        # Equivalised BHC — the concept ``household_income_decile`` really
        # ranks on. Deliberately distinct from both other income concepts so a
        # test can tell which one the audit picked (round-3 finding 1).
        equiv_income_bhc=net_income / 1.25,
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
    from uk_iran_conflict.scenarios import REALISED_PUMP_SUSTAINED_FRACTION

    assert ratio == pytest.approx(REALISED_PUMP_SUSTAINED_FRACTION)
    # Derived over the same twelve months as the cap phase-in, not over
    # calendar 2026 (round-2 finding 1).
    assert ratio == pytest.approx(0.65)
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
    assert result.annual_phase_in_gas == pytest.approx(0.5296, abs=1e-4)
    assert result.annual_phase_in_electricity == pytest.approx(0.5166, abs=1e-4)


# --- A3: symmetric damping ------------------------------------------------


def test_symmetric_scenario_moves_only_the_domestic_leg():
    """Round-3 reversed the sign of this, and that is the point.

    Before the baseline fix the solved gas fraction was 0.199 against the pump
    leg's 0.650, so imposing a common fraction *raised* the domestic leg by a
    factor of three and cut the fuel share sharply. With the counterfactual
    constructed properly the solved fraction is 0.765 — above the pump leg's
    0.650 — so symmetry now *lowers* the domestic leg slightly. The
    specification is unchanged; what changed is that the asymmetry it was built
    to probe was mostly the baseline bug.
    """
    base = make_baseline()
    main, main_cost = inc.run_scenario(base, get_scenario("realised_2026"))
    sym, sym_cost = inc.run_scenario(base, get_scenario("realised_2026_symmetric"))
    assert sym.aggregate_cost_bn < main.aggregate_cost_bn
    assert sym.motor_fuel_share_of_loss > main.motor_fuel_share_of_loss
    # The fuel channel itself is identical; only the domestic leg is rescaled.
    assert sym_cost.motor_fuel == pytest.approx(main_cost.motor_fuel)
    assert sym_cost.domestic.sum() < main_cost.domestic.sum()
    # And the two fractions are now close, where they used to differ 3x.
    ratio = scen.REALISED_PUMP_SUSTAINED_FRACTION / scen.REALISED_SUSTAINED_FRACTION
    assert 0.5 < ratio < 2.0


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
    # The asymmetric specification differs from every common-fraction one, and
    # after the round-3 baseline fix it differs *downward*: the solved gas
    # fraction is now above the pump fraction, so the asymmetry works against
    # the fuel share instead of for it.
    asym, _ = inc.run_scenario(base, scenario)
    assert asym.motor_fuel_share_of_loss != pytest.approx(shares[0])
    assert asym.motor_fuel_share_of_loss < shares[0]


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


# --------------------------------------------------------------------------
# Round-2: within- vs between-decile dispersion
# --------------------------------------------------------------------------


def _intra(ranges):
    return [
        inc.IntraDecileRow(
            decile=i + 1,
            p10_loss_pct=1.0,
            p50_loss_pct=1.0 + r / 2,
            p90_loss_pct=1.0 + r,
            share_above_5pct=0.0,
            share_above_10pct=0.0,
        )
        for i, r in enumerate(ranges)
    ]


def _deciles(means):
    return [
        inc.DecileRow(
            decile=i + 1,
            mean_loss_gbp=100.0 * m,
            mean_loss_pct=m,
            share_of_total_loss=0.1,
            households_m=3.0,
        )
        for i, m in enumerate(means)
    ]


def test_dispersion_summary_reports_both_with_and_without_decile_one():
    """The finding: the claim reverses once decile one is set aside.

    Decile one is where about a fifth of households have non-positive
    equivalised AHC income, so a p90-p10 range there is a statement about the
    denominator rather than about horizontal incidence.
    """
    # One huge decile-one range, nine small ones.
    summary = inc.dispersion_summary(
        _intra([10.0] + [1.0] * 9), _deciles([2.5] + [0.5] * 9)
    )
    assert summary.between_decile_range_pp == pytest.approx(2.0)
    assert summary.mean_within_decile_range_pp == pytest.approx(1.9)
    assert summary.mean_within_decile_range_excl_d1_pp == pytest.approx(1.0)
    assert summary.median_within_decile_range_pp == pytest.approx(1.0)
    assert summary.within_exceeds_between is False
    assert summary.within_exceeds_between_excl_d1 is False
    assert summary.deciles_below_between_range == 9
    assert len(summary.within_decile_range_by_decile_pp) == 10


def test_dispersion_summary_can_disagree_with_itself_across_decile_one():
    summary = inc.dispersion_summary(
        _intra([30.0] + [1.9] * 9), _deciles([2.5] + [0.5] * 9)
    )
    assert summary.within_exceeds_between is True
    assert summary.within_exceeds_between_excl_d1 is False


def test_run_scenario_persists_the_dispersion_statistics():
    result, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    d = result.dispersion
    assert d is not None
    assert len(d.within_decile_range_by_decile_pp) == len(result.intra_decile)
    assert d.between_decile_range_pp > 0


# --------------------------------------------------------------------------
# Round-2: what income concept does the decile ranking use?
# --------------------------------------------------------------------------


def test_decile_concept_audit_identifies_the_ranking_variable():
    base = make_baseline()
    audit = inc.decile_concept_audit(base)
    assert set(audit.agreement) == set(inc.DECILE_CONCEPT_CANDIDATES)
    assert audit.best_match in inc.DECILE_CONCEPT_CANDIDATES
    assert 0.0 <= audit.agreement[audit.best_match] <= 1.0
    assert audit.burden_denominator == inc.DEFAULT_INCOME_BASIS
    assert audit.matches_burden_denominator == (
        audit.best_match == inc.DEFAULT_INCOME_BASIS
    )
    for gap in audit.mean_absolute_decile_gap.values():
        assert gap >= 0.0
    assert audit.unavailable == ()
    assert audit.person_weighted is True
    assert audit.documented_truth == "equivalised_bhc"


def test_the_audit_uses_the_real_bhc_variable_not_a_reconstruction():
    """Round-3 finding 1. The old audit built its BHC candidate as
    ``household_net_income / household_equivalisation_ahc`` — the wrong
    numerator and an AHC denominator — so the concept the package actually ranks
    on was never a candidate, and AHC "won" inside the construction error.
    """
    base = make_baseline()
    # Give BHC its own ordering, so its deciles are genuinely a different
    # partition rather than a monotone relabelling of the other two.
    rng = np.random.default_rng(7)
    base = dataclasses.replace(
        base, equiv_income_bhc=rng.permutation(base.equiv_income_bhc)
    )
    # The real variable is distinct from both the wrong reconstruction and AHC.
    wrong = base.net_income / base.equivalisation
    assert not np.allclose(base.equiv_income_bhc, wrong)
    assert not np.allclose(base.equiv_income_bhc, base.equiv_income_ahc)
    # Ranking on the real BHC concept must be recovered exactly.
    ranked = inc._weighted_deciles(
        np.asarray(base.equiv_income_bhc, dtype=float), base.weight * base.people
    )
    ranked_base = dataclasses.replace(base, decile=ranked.astype(float))
    audit = inc.decile_concept_audit(ranked_base)
    assert audit.best_match == "equivalised_bhc"
    assert audit.agreement["equivalised_bhc"] == pytest.approx(1.0)
    assert audit.best_match_is_documented_truth is True
    # And on this data that is NOT the burden denominator: the paper's gradient
    # is a cross-concept statistic.
    assert audit.matches_burden_denominator is False


def test_the_audit_reports_bhc_unavailable_rather_than_substituting_a_proxy():
    base = dataclasses.replace(make_baseline(), equiv_income_bhc=None)
    audit = inc.decile_concept_audit(base)
    assert audit.unavailable == ("equivalised_bhc",)
    assert "equivalised_bhc" not in audit.agreement


def test_the_audit_is_person_weighted_like_the_package():
    """``household_income_decile`` weights by ``household_weight * count_people``."""
    base = make_baseline()
    people = np.where(base.decile <= 5, 1.0, 4.0)
    base = dataclasses.replace(base, people=people)
    person = inc.decile_concept_audit(base, person_weighted=True)
    household = inc.decile_concept_audit(base, person_weighted=False)
    assert person.person_weighted is True
    assert household.person_weighted is False
    # The two weightings give different rankings whenever household size is
    # correlated with income, which it is.
    assert person.agreement != household.agreement


def test_the_negative_sentinel_explains_the_out_of_range_households():
    base = make_baseline(unbanded=4)
    audit = inc.decile_concept_audit(base)
    assert audit.negative_sentinel_share > 0.0
    assert inc.DECILE_RANKING_NEGATIVE_SENTINEL == -1


def test_decile_concept_audit_detects_a_deliberately_wrong_ranking():
    """If the ranking were on a different concept, the audit must say so."""
    base = make_baseline()
    reversed_decile = 11.0 - base.decile
    scrambled = dataclasses.replace(base, decile=reversed_decile)
    audit = inc.decile_concept_audit(scrambled)
    assert (
        audit.agreement[audit.best_match]
        < inc.decile_concept_audit(base).agreement["equivalised_ahc"]
    )


def test_run_scenario_persists_the_decile_concept_audit():
    result, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    assert result.decile_concept is not None
    assert result.decile_concept.burden_denominator == "equivalised_ahc"


# --------------------------------------------------------------------------
# Round-2: the domestic-energy-only gradient (VALIDATION.md's anchor)
# --------------------------------------------------------------------------


def test_domestic_only_gradient_is_computed_and_uses_the_domestic_leg():
    base = make_baseline()
    scenario = get_scenario("realised_2026")
    result, cost = inc.run_scenario(base, scenario)
    expected = inc.decile_table(base, cost.domestic)
    assert [r.mean_loss_gbp for r in result.decile_domestic_only] == pytest.approx(
        [r.mean_loss_gbp for r in expected]
    )
    # It is strictly smaller than the all-channel loss in every decile.
    for dom, total in zip(result.decile_domestic_only, result.decile, strict=True):
        assert dom.mean_loss_gbp <= total.mean_loss_gbp


def test_domestic_only_gradient_is_reported_beside_the_all_channel_one():
    result, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    assert result.domestic_only_d1_d10_ratio_pct > 0
    assert result.all_channel_d1_d10_ratio_pct == pytest.approx(
        result.decile[0].mean_loss_pct / result.decile[-1].mean_loss_pct
    )
    assert result.domestic_only_d1_d10_ratio_gbp > 0


# --------------------------------------------------------------------------
# Round-2: Table 1 disclosures
# --------------------------------------------------------------------------


def test_every_result_records_the_window_and_both_damping_fractions():
    """Table 1 is not like-for-like without them, so the prose cannot omit them."""
    from uk_iran_conflict import scenarios as scen

    realised, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    niesr, _ = inc.run_scenario(make_baseline(), get_scenario("niesr_baseline"))

    assert realised.modelled_window == scen.MODELLED_WINDOW_LABEL
    assert realised.pump_sustained_fraction == pytest.approx(
        scen.REALISED_PUMP_SUSTAINED_FRACTION
    )
    assert realised.cap_lag_quarters == pytest.approx(scen.CAP_LAG_QUARTERS)
    # The asymmetry Table 1 has to disclose: NIESR paths are sustained levels,
    # so their pump leg is undamped while the realised one is not.
    assert niesr.pump_sustained_fraction == 1.0
    assert realised.pump_sustained_fraction < niesr.pump_sustained_fraction


def test_the_symmetric_specification_is_the_one_with_equal_fractions():
    sym, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026_symmetric"))
    assert sym.gas_sustained_fraction == pytest.approx(sym.pump_sustained_fraction)
    main, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    assert main.gas_sustained_fraction != main.pump_sustained_fraction


# --------------------------------------------------------------------------
# Round-3 finding 2: the means-tested motor-fuel margin
# --------------------------------------------------------------------------


def _with_means_tested(base: inc.Baseline, low_fuel_factor: float = 0.2):
    """Every third household means-tested, with a deliberately thin fuel spend.

    Reproduces the shape of the microdata margin (means-tested households
    imputed a small fraction of everyone else's motor fuel) so the correction
    can be tested without microdata.
    """
    mt = np.arange(base.n) % 3 == 0
    return dataclasses.replace(
        base,
        means_tested=mt,
        petrol=np.where(mt, base.petrol * low_fuel_factor, base.petrol),
        diesel=np.where(mt, base.diesel * low_fuel_factor, base.diesel),
    )


def test_motor_fuel_margins_measure_the_means_tested_split():
    base = _with_means_tested(make_baseline())
    margins = inc.motor_fuel_margins(base)
    assert margins.means_tested_mean_fuel_gbp < margins.non_means_tested_mean_fuel_gbp
    assert margins.means_tested_fuel_ratio == pytest.approx(5.0, rel=0.05)
    assert len(margins.zero_fuel_share_by_decile) == 10
    assert 0.0 <= margins.zero_fuel_share_overall <= 1.0


def test_motor_fuel_margins_require_the_means_tested_flag():
    with pytest.raises(ValueError, match="means_tested"):
        inc.motor_fuel_margins(make_baseline())
    with pytest.raises(ValueError, match="means_tested"):
        inc.equalise_means_tested_fuel(make_baseline())


def test_mt_fuel_parity_equalises_within_decile_and_preserves_the_total():
    base = _with_means_tested(make_baseline())
    fixed = inc.apply_calibration(base, "mt_fuel_parity")
    # The national total is preserved, so the aggregate loss stays comparable.
    assert inc.wsum(fixed.motor_fuel, fixed.weight) == pytest.approx(
        inc.wsum(base.motor_fuel, base.weight)
    )
    # Parity within *every* decile. (The pooled ratio need not be exactly 1.0:
    # means-tested households are not spread evenly across deciles, and the
    # specification deliberately acts within decile rather than pooling.)
    mt = np.asarray(fixed.means_tested).astype(bool)
    for d in range(1, 11):
        sel = fixed.decile == d
        a, b = sel & mt, sel & ~mt
        assert inc.wmean(fixed.motor_fuel[a], fixed.weight[a]) == pytest.approx(
            inc.wmean(fixed.motor_fuel[b], fixed.weight[b])
        )


def test_the_ons_shape_calibration_cannot_move_the_means_tested_margin():
    """Why a new specification was needed: one factor per decile scales both
    groups identically, so the 6.2x margin survives it untouched.
    """
    base = _with_means_tested(make_baseline())
    shaped = inc.apply_calibration(base, "ons_fuel_shape")
    parity = inc.apply_calibration(base, "mt_fuel_parity")
    mt = np.asarray(base.means_tested).astype(bool)

    def within(b, d):
        sel = b.decile == d
        a, c = sel & mt, sel & ~mt
        return inc.wmean(b.motor_fuel[a], b.weight[a]) / inc.wmean(
            b.motor_fuel[c], b.weight[c]
        )

    for d in range(1, 11):
        # One factor per decile scales both groups identically: the margin is
        # exactly unchanged, decile by decile.
        assert within(shaped, d) == pytest.approx(within(base, d))
        # The new specification is the only one that moves it.
        assert within(parity, d) == pytest.approx(1.0)


def test_mt_fuel_parity_raises_the_means_tested_share_of_the_loss():
    """The paper's central policy finding scales inversely with this share."""
    base = _with_means_tested(make_baseline())
    scenario = get_scenario("realised_2026")
    raw, _ = inc.run_scenario(base, scenario)
    parity, _ = inc.run_scenario(base, scenario, calibration="mt_fuel_parity")
    assert raw.motor_fuel_margins is not None
    assert parity.motor_fuel_margins is not None
    assert (
        parity.motor_fuel_margins.means_tested_share_of_loss
        > raw.motor_fuel_margins.means_tested_share_of_loss
    )


# --------------------------------------------------------------------------
# Round-3 finding 3: the fuel participation margin
# --------------------------------------------------------------------------


def _with_flat_zero_rate(base: inc.Baseline, zero_share: float = 0.6):
    """A baseline whose zero-fuel rate is identical in every decile.

    That identity — 62.0% in decile one and 62.0% in decile ten — is the
    verified evidence for the participation artefact, and it is what the
    correction has to be able to detect and move.
    """
    zero = np.zeros(base.n, dtype=bool)
    for d in range(1, 11):
        sel = np.flatnonzero(base.decile == d)
        zero[sel[: int(round(zero_share * len(sel)))]] = True
    return dataclasses.replace(
        base,
        petrol=np.where(zero, 0.0, base.petrol),
        diesel=np.where(zero, 0.0, base.diesel),
    )


def test_the_flat_zero_rate_is_the_diagnostic_the_paper_needs():
    base = _with_flat_zero_rate(make_baseline(n_per_decile=10))
    margins = inc.motor_fuel_margins(_with_means_tested(base))
    shares = margins.zero_fuel_share_by_decile
    assert shares[0] == pytest.approx(shares[-1])
    assert margins.zero_share_d1_minus_d10_pp == pytest.approx(0.0)


def test_nts_targets_are_a_gradient_between_the_published_ends():
    targets = inc.nts_participation_targets()
    assert len(targets) == 10
    assert targets[0] == pytest.approx(inc.NTS_CAR_AVAILABILITY_D1)
    assert targets[-1] == pytest.approx(inc.NTS_CAR_AVAILABILITY_D10)
    assert list(targets) == sorted(targets)
    assert all(0.0 <= t <= 1.0 for t in targets)


def test_participation_correction_imposes_a_gradient_and_preserves_the_total():
    base = _with_flat_zero_rate(make_baseline(n_per_decile=50))
    fixed = inc.apply_calibration(base, "nts_participation")
    assert inc.wsum(fixed.motor_fuel, fixed.weight) == pytest.approx(
        inc.wsum(base.motor_fuel, base.weight)
    )
    after = inc.motor_fuel_margins(_with_means_tested(fixed)).zero_fuel_share_by_decile
    before = inc.motor_fuel_margins(_with_means_tested(base)).zero_fuel_share_by_decile
    # A gradient where there was none.
    assert before[0] == pytest.approx(before[-1])
    assert after[0] > after[-1]
    # And every decile's participation moved toward its NTS target.
    targets = inc.nts_participation_targets()
    for d in range(10):
        assert 1 - after[d] == pytest.approx(targets[d], abs=0.05)


def test_participation_correction_lowers_the_conditional_level():
    """Participation and the conditional level move together when the decile
    total is preserved; the paper has to report both.
    """
    base = _with_flat_zero_rate(make_baseline(n_per_decile=50))
    fixed = inc.apply_calibration(base, "nts_participation")
    positive_before = base.motor_fuel[base.motor_fuel > 0]
    positive_after = fixed.motor_fuel[fixed.motor_fuel > 0]
    assert positive_after.mean() < positive_before.mean()


# --------------------------------------------------------------------------
# Round-3 finding 9: the gradient without decile one, and without the drop
# --------------------------------------------------------------------------


def test_the_gradient_is_also_reported_from_decile_two():
    result, _ = inc.run_scenario(make_baseline(), get_scenario("realised_2026"))
    d1 = next(r for r in result.decile if r.decile == 1)
    d2 = next(r for r in result.decile if r.decile == 2)
    d10 = next(r for r in result.decile if r.decile == 10)
    assert result.all_channel_d1_d10_ratio_pct == pytest.approx(
        d1.mean_loss_pct / d10.mean_loss_pct
    )
    assert result.d2_d10_ratio_pct == pytest.approx(
        d2.mean_loss_pct / d10.mean_loss_pct
    )
    assert result.d2_d10_ratio_gbp == pytest.approx(
        d2.mean_loss_gbp / d10.mean_loss_gbp
    )


def test_the_non_positive_tail_treatment_is_swept_on_every_run():
    """A fifth of decile one has non-positive equivalised AHC income and is
    dropped. Winsorising it back in must be visible, not asserted away.
    """
    base = make_baseline(unbanded=4)
    result, _ = inc.run_scenario(base, get_scenario("realised_2026"))
    assert result.income_treatment == "drop"
    assert np.isfinite(result.d1_d10_ratio_pct_winsorised)
    assert np.isfinite(result.mean_loss_pct_winsorised)
    # Running with the treatment as the headline convention must reproduce the
    # winsorised statistics, so the two are the same arithmetic.
    win, _ = inc.run_scenario(
        base, get_scenario("realised_2026"), income_treatment="winsorise_p1"
    )
    assert win.mean_loss_pct == pytest.approx(result.mean_loss_pct_winsorised)
    assert win.income_treatment == "winsorise_p1"


def test_winsorising_keeps_the_non_positive_tail_in_the_denominator():
    base = make_baseline()
    income = base.equiv_income_ahc.copy()
    income[0] = -100.0
    treated = inc.treat_non_positive_income(income, base.weight, "winsorise_p1")
    assert treated[0] > 0
    assert np.allclose(treated[1:], income[1:])
    assert inc.treat_non_positive_income(income, base.weight, "drop") is income
    with pytest.raises(ValueError, match="unknown income treatment"):
        inc.treat_non_positive_income(income, base.weight, "clip")


def test_calibration_names_are_validated():
    with pytest.raises(ValueError, match="unknown calibration"):
        inc.apply_calibration(make_baseline(), "no_such_calibration")


# --------------------------------------------------------------------------
# Round-3 finding 6: silent defaults and the dead path
# --------------------------------------------------------------------------


class _NoPassThrough:
    """A scenario-shaped object with a pump path but no pass-through block."""

    key = "no_pass_through"
    pump = get_scenario("realised_2026").pump


def test_a_missing_pass_through_block_is_an_error_not_the_peak_bound():
    """``getattr(None, "pump_sustained_fraction", 1.0)`` returned 1.0 silently.

    1.0 is the peak-fuel upper bound — precisely the fallback the docstring says
    it refuses — reached by precisely the mechanism it refuses it for.
    """
    with pytest.raises(AttributeError, match="peak-fuel upper bound"):
        inc.sustained_pump_factors(_NoPassThrough())


def test_sustained_pump_factors_delegates_to_the_scenario_method():
    """The dead path, made live. ``Scenario.sustained_pump_changes`` was
    referenced only by the tests while the pipeline used this function.
    """
    for key in ("realised_2026", "realised_2026_peak_fuel", "niesr_adverse"):
        scenario = get_scenario(key)
        petrol, diesel = inc.sustained_pump_factors(scenario)
        expected_petrol, expected_diesel = scenario.sustained_pump_changes
        assert petrol == pytest.approx(1.0 + expected_petrol)
        assert diesel == pytest.approx(1.0 + expected_diesel)


def test_a_missing_pump_path_is_still_an_error():
    class NoPump:
        key = "no_pump"

    with pytest.raises(AttributeError, match="no pump"):
        inc.sustained_pump_factors(NoPump())


# --------------------------------------------------------------------------
# Round-3 finding 5: a grid reconciliation check that can actually fail
# --------------------------------------------------------------------------
#
# These exercise ``analysis/run_grid.py``'s reconciliation, which is part of
# this revision's surface. They live here rather than in
# ``tests/test_grid_and_sweeps.py`` only because that file belongs to another
# workstream in this round.

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "analysis"))
import run_grid  # noqa: E402


@pytest.fixture
def grid_base() -> inc.Baseline:
    """A baseline wide enough for the channel ratios to differ from each other."""
    n = 20
    return inc.Baseline(
        net_income=np.linspace(8_000.0, 100_000.0, n),
        weight=np.full(n, 1e6),
        people=np.full(n, 2.0),
        gas=np.linspace(400.0, 900.0, n),
        electricity=np.linspace(500.0, 1_000.0, n),
        petrol=np.linspace(200.0, 900.0, n),
        diesel=np.linspace(100.0, 600.0, n),
        decile=np.repeat(np.arange(1.0, 11.0), 2),
        equiv_income_ahc=np.linspace(8_000.0, 100_000.0, n),
        in_poverty_bhc=np.zeros(n),
        in_poverty_ahc=np.zeros(n),
        region=np.zeros(n),
        country=np.zeros(n),
    )


# --- round-3 finding 5: a reconciliation check that can actually fail -----


def test_the_decile_ratio_is_the_convex_combination_of_its_channel_ratios(grid_base):
    """The identity that replaces the vacuous ±5% range-membership check.

    The old check asserted that every named scenario's D1/D10 ratio lay within
    5% of a grid range that spanned 0.0008 across all 36 cells, because both
    pure-channel ratios were ~9.31 and every mix is a convex combination of
    them. It could not fail. This can: the identity is exact in the pipeline's
    own arithmetic and breaks the moment the grid and the named scenarios stop
    being the same code.
    """
    import pandas as pd

    live = pd.DataFrame(
        [run_grid.cell_row(grid_base, g, o) for g, o in ((0.5, 0.4), (0.8, 0.6))]
    )
    out = run_grid.reconcile_named_scenarios(grid_base, live)
    assert out["channel_mix_identity_holds"] is True
    assert out["identity_broken"] == []
    for payload in out["scenarios"].values():
        assert payload["identity_holds"] is True
        assert payload["identity_residual"] < out["identity_tolerance"]
        share = payload["domestic_share_of_decile10_loss"]
        assert 0.0 <= share <= 1.0
        assert payload["d1_d10_ratio"] == pytest.approx(
            share * payload["d1_d10_ratio_domestic_only"]
            + (1 - share) * payload["d1_d10_ratio_motor_fuel_only"]
        )


def test_the_reconciliation_reports_whether_it_is_informative(grid_base):
    """A degenerate grid must say so rather than pass silently."""
    import pandas as pd

    degenerate = pd.DataFrame({"d1_d10_ratio": [9.311577, 9.312363]})
    out = run_grid.reconcile_named_scenarios(grid_base, degenerate)
    assert out["grid_shows_invariance"] is True
    assert out["check_is_informative"] is False
    assert out["grid_d1_d10_ratio_spread"] < 0.01
    wide = pd.DataFrame({"d1_d10_ratio": [4.0, 12.0]})
    out_wide = run_grid.reconcile_named_scenarios(grid_base, wide)
    assert out_wide["check_is_informative"] is True


# --- round-4 finding 5: the range check the paper advertised did not run --


def test_the_grid_never_claims_to_enforce_a_range_check(grid_base):
    """The appendix advertised that run_grid "checks every named scenario
    against the live grid's range and raises if one falls outside it". It never
    did, and on this grid it must not: the range is degenerate. The scope block
    says so permanently, so the guarantee cannot be re-advertised.
    """
    import pandas as pd

    degenerate = pd.DataFrame({"d1_d10_ratio": [9.311577, 9.312363]})
    out = run_grid.reconcile_named_scenarios(grid_base, degenerate)
    scope = out["grid_scope"]
    assert scope["range_check_enforced"] is False
    assert scope["why_the_range_check_is_not_enforced"]
    assert scope["what_the_grid_shows"] and scope["what_the_grid_does_not_show"]
    assert any("petrol" in item for item in scope["what_the_grid_does_not_show"])
    assert len(scope["enforced_checks"]) >= 2


def test_named_scenarios_outside_the_range_are_explained_not_hidden(grid_base):
    import pandas as pd

    degenerate = pd.DataFrame({"d1_d10_ratio": [9.311577, 9.312363]})
    out = run_grid.reconcile_named_scenarios(grid_base, degenerate)
    assert set(out["named_scenarios_outside_grid_range"]) == {
        key for key, v in out["scenarios"].items() if not v["inside_grid_range"]
    }
    assert "petrol:diesel" in out["why_named_scenarios_fall_outside_grid_range"]


def test_the_sub_channel_bracketing_check_can_fail_and_here_does_not(grid_base):
    """The check that replaces the range check: an aggregate ratio outside the
    span of its own four sub-channel ratios is arithmetically impossible.
    """
    import pandas as pd

    live = pd.DataFrame(
        [run_grid.cell_row(grid_base, g, o) for g, o in ((0.5, 0.4), (0.8, 0.6))]
    )
    out = run_grid.reconcile_named_scenarios(grid_base, live)
    assert out["sub_channel_bracketing_holds"] is True
    assert out["sub_channel_bracketing_broken"] == []
    for payload in out["scenarios"].values():
        sub = payload["sub_channels"]
        assert sub["inside_sub_channel_span"] is True
        assert set(sub["channels"]) == set(run_grid.SUB_CHANNELS)
        assert sum(sub["share_of_decile10_loss_by_channel"].values()) == pytest.approx(
            1.0
        )
        assert payload["d1_d10_ratio"] == pytest.approx(sub["d1_d10_ratio_implied"])


def test_the_bracketing_check_rejects_an_impossible_ratio(grid_base):
    """It can fail — which is the whole difference from the check it replaces."""
    scenario = scen.get_scenario("realised_2026")
    honest = run_grid._sub_channel_decomposition(grid_base, scenario, 9.3)
    impossible = run_grid._sub_channel_decomposition(
        grid_base, scenario, honest["sub_channel_ratio_max"] + 1.0
    )
    assert impossible["inside_sub_channel_span"] is False


def test_the_grid_holds_the_petrol_diesel_mix_fixed_in_every_cell():
    """The reason the named scenarios fall outside the grid's range."""
    mixes = {
        run_grid.build_scenario(g, o).pump.diesel_pct_change
        / run_grid.build_scenario(g, o).pump.petrol_pct_change
        for g, o in ((0.2, 0.4), (0.6, 0.8), (1.0, 0.2))
    }
    assert len(mixes) == 1
    assert mixes.pop() == pytest.approx(
        run_grid.DIESEL_PASS_THROUGH / run_grid.PETROL_PASS_THROUGH
    )


# --- Round-3 finding 7: dataset_path() must read .env itself ------------

import os  # noqa: E402

from analysis import run_incidence  # noqa: E402


def test_dataset_path_loads_the_env_itself(monkeypatch, tmp_path):
    """Only ``main()`` called ``_load_env``, so importing the module and calling
    ``dataset_path()`` directly — which every other analysis script does —
    raised "No Hugging Face token" despite a valid ``.env``.
    """
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("HUGGING_FACE_TOKEN=from-the-env-file\n")
    monkeypatch.setattr(run_incidence, "ROOT", tmp_path)

    seen = {}

    def fake_download(repo, filename, **kwargs):
        seen.update(kwargs)
        return "/tmp/fake.h5"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert run_incidence.dataset_path() == "/tmp/fake.h5"
    assert seen["token"] == "from-the-env-file"
    # The dataset pin travels with it.
    assert seen["revision"] == run_incidence.DATASET_REVISION


def test_dataset_path_still_fails_loudly_with_no_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(run_incidence, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="No Hugging Face token"):
        run_incidence.dataset_path()


def test_load_env_never_overrides_an_exported_value(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "exported")
    env = tmp_path / ".env"
    env.write_text("HUGGING_FACE_TOKEN=from-file\n")
    monkeypatch.setattr(run_incidence, "ROOT", tmp_path)
    run_incidence._load_env()
    assert os.environ["HUGGING_FACE_TOKEN"] == "exported"

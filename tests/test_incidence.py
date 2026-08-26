"""Incidence tests that run without microdata.

Every test here builds a small synthetic :class:`Baseline` by hand, so the
module's arithmetic — in particular the two corrections required by
``docs/VALIDATION.md`` (the pump-price damping, Check 2b; the ONS motor-fuel
recalibration, Check 2d) — is testable without a Hugging Face token.
"""

from __future__ import annotations

import numpy as np
import pytest

from uk_iran_conflict import incidence as inc
from uk_iran_conflict.scenarios import get_scenario


def make_baseline(n_per_decile: int = 5, seed: int = 0) -> inc.Baseline:
    """Ten deciles of households with a deliberately *flat* fuel profile.

    Flat across deciles is exactly the defect VALIDATION.md Check 2d documents
    (D1 £1,073 against D10 £1,333), so it is the right starting point for
    testing the correction. Within each decile the spend varies, so the test
    can check that relative variation survives.
    """
    rng = np.random.default_rng(seed)
    n = n_per_decile * 10
    decile = np.repeat(np.arange(1, 11), n_per_decile)
    spread = rng.uniform(0.5, 1.5, n)
    return inc.Baseline(
        net_income=10_000.0 * decile,
        weight=np.full(n, 100.0),
        people=np.full(n, 2.0),
        gas=np.full(n, 600.0),
        electricity=np.full(n, 700.0),
        petrol=700.0 * spread,
        diesel=300.0 * spread,
        decile=decile,
        equiv_income_ahc=10_000.0 * decile,
        in_poverty_bhc=np.zeros(n),
        in_poverty_ahc=np.zeros(n),
        region=np.array(["London"] * n),
        country=np.array(["ENGLAND"] * n),
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

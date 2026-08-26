"""Consumption-elasticity behaviour tests.

The paper's main specification is **zero elasticity** — the Deaton first-order
approximation, in which Delta-cost = q0 x Delta-p and the measured loss is an
explicit upper bound, not a behavioural prediction. Everything else in
:mod:`uk_iran_conflict.elasticity` is a robustness variant, because
PolicyEngine UK has no consumption elasticity or price pass-through of its own
(open issue #1114, which notes UKMOD's TCO uses 0.8).

These tests run without microdata.
"""

from __future__ import annotations

import pytest

from uk_iran_conflict.elasticity import (
    CARRIERS,
    LABANDEIRA_2017_LONG_RUN,
    LABANDEIRA_2017_SHORT_RUN,
    ZERO_ELASTICITY,
    ElasticitySpec,
    consumption_reduction,
    cv_bounds,
    deadweight_share,
    elasticity_for,
    laspeyres_cv,
    paasche_cv,
    quantity_factor,
    resolve_elasticity_spec,
    spend_change,
    spend_factor,
    static_spend_change,
    welfare_shaved_share,
)

SPECS = {
    "main": ElasticitySpec.main(),
    "labandeira_short_run": ElasticitySpec.labandeira_flat("short_run"),
    "labandeira_long_run": ElasticitySpec.labandeira_flat("long_run"),
    "priesmann_short_run": ElasticitySpec.priesmann_income_varying("short_run"),
    "priesmann_long_run": ElasticitySpec.priesmann_income_varying("long_run"),
    "prior_repo": ElasticitySpec.prior_repo_replication(),
}
ALL_SPECS = sorted(SPECS.items())
RATIOS = (1.0, 1.1, 1.3, 1.6, 2.61)


# --- the main specification -----------------------------------------------


def test_three_carriers_are_modelled():
    assert set(CARRIERS) == {"gas", "electricity", "motor_fuel"}
    assert set(ZERO_ELASTICITY) == set(CARRIERS)


def test_exactly_one_spec_is_the_main_specification():
    flagged = [n for n, s in ALL_SPECS if s.is_main_specification]
    assert flagged == ["main"]


def test_main_spec_is_zero_for_every_carrier():
    spec = ElasticitySpec.main()
    for carrier in CARRIERS:
        assert spec.epsilon(carrier) == 0.0


@pytest.mark.parametrize("ratio", RATIOS)
def test_zero_elasticity_leaves_quantity_unchanged(ratio):
    """The main spec must not move quantities at all."""
    assert quantity_factor(ratio, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", RATIOS)
def test_zero_elasticity_spend_factor_is_the_price_ratio(ratio):
    """At eps = 0, spend scales exactly with price: the Deaton first order."""
    assert spend_factor(ratio, 0.0) == pytest.approx(ratio)


def test_static_spend_change_is_q0_delta_p():
    assert static_spend_change(1000.0, 1.6) == pytest.approx(600.0)
    assert static_spend_change(1000.0, 1.6) == pytest.approx(
        spend_change(1000.0, 1.6, 0.0)
    )


def test_main_spec_bounds_every_robustness_variant():
    """The headline loss is an upper bound on every elastic alternative."""
    ratio = 1.6
    baseline = 1000.0
    headline = static_spend_change(baseline, ratio)
    for name, spec in ALL_SPECS:
        for carrier in CARRIERS:
            eps = elasticity_for(spec, carrier, decile=1)
            assert spend_change(baseline, ratio, eps) <= headline + 1e-9, (
                f"{name}/{carrier} exceeds the zero-elasticity upper bound"
            )


# --- sign and shape of every spec ----------------------------------------


@pytest.mark.parametrize("name,spec", ALL_SPECS)
def test_every_spec_is_sourced_and_named(name, spec):
    assert spec.name
    assert len(spec.source.strip()) > 30, f"{name} lacks a citation"


@pytest.mark.parametrize("name,spec", ALL_SPECS)
def test_elasticities_are_non_positive(name, spec):
    """Energy demand does not rise when its price rises."""
    for carrier in CARRIERS:
        for decile in range(1, 11):
            assert elasticity_for(spec, carrier, decile) <= 0.0, f"{name}/{carrier}"


@pytest.mark.parametrize("name,spec", ALL_SPECS)
def test_elasticities_are_plausible_magnitudes(name, spec):
    """Residential own-price elasticities sit well inside (-1, 0].

    Labandeira et al. (2017) short-run range is [-0.803, 0.066]; anything past
    -1 would imply spend *falling* when price rises.
    """
    for carrier in CARRIERS:
        for decile in range(1, 11):
            assert -1.0 < elasticity_for(spec, carrier, decile) <= 0.0


@pytest.mark.parametrize("name,spec", ALL_SPECS)
def test_spend_still_rises_when_price_rises(name, spec):
    """eps > -1 guarantees a price rise raises spend — never a saving."""
    for carrier in CARRIERS:
        eps = elasticity_for(spec, carrier, decile=1)
        assert spend_change(1000.0, 1.6, eps) > 0.0, f"{name}/{carrier}"


def test_spec_rejects_both_flat_and_by_decile():
    with pytest.raises(ValueError, match="exactly one"):
        ElasticitySpec(
            name="bad",
            source="x" * 40,
            flat=ZERO_ELASTICITY,
            by_decile={"gas": {1: -0.1}},
        )


def test_spec_rejects_neither_flat_nor_by_decile():
    with pytest.raises(ValueError, match="exactly one"):
        ElasticitySpec(name="bad", source="x" * 40)


# --- closed forms ---------------------------------------------------------


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("eps", (0.0, -0.2, -0.64, -0.803))
def test_quantity_and_spend_factors_are_consistent(ratio, eps):
    """spend factor = quantity factor x price ratio, identically."""
    assert spend_factor(ratio, eps) == pytest.approx(
        quantity_factor(ratio, eps) * ratio
    )


@pytest.mark.parametrize("eps", (0.0, -0.2, -0.64))
def test_spend_factor_stays_positive_at_extreme_shocks(eps):
    """The constant-elasticity form, not the linear one.

    The linear ``(1 + r)(1 + eps*r)`` approximation returns a negative
    (physically impossible) factor at eps = -0.64, r = +1.61.
    """
    assert spend_factor(2.61, eps) > 0.0


def test_published_doctest_values():
    assert quantity_factor(1.6, -0.64) == pytest.approx(0.7402, abs=1e-4)
    assert spend_factor(2.61, -0.64) == pytest.approx(1.4125, abs=1e-4)
    assert spend_change(1000.0, 1.6, -0.64) == pytest.approx(184.4, abs=0.1)


@pytest.mark.parametrize("eps", (0.0, -0.2, -0.64))
def test_consumption_reduction_is_a_bounded_positive_fraction(eps):
    """The prior repo's linear version exceeded 1.0 — negative consumption."""
    for ratio in RATIOS:
        cut = consumption_reduction(ratio, eps)
        assert 0.0 <= cut < 1.0


def test_consumption_reduction_is_zero_under_the_main_spec():
    for ratio in RATIOS:
        assert consumption_reduction(ratio, 0.0) == pytest.approx(0.0)


def test_deadweight_share_is_zero_under_the_main_spec():
    assert deadweight_share(1.6, 0.0) == pytest.approx(0.0)


def test_deadweight_share_grows_with_the_elasticity():
    assert deadweight_share(1.6, -0.64) > deadweight_share(1.6, -0.2) > 0.0


# --- monotonicity ---------------------------------------------------------


@pytest.mark.parametrize("eps", (0.0, -0.2, -0.64))
def test_a_larger_price_rise_cuts_quantity_by_more(eps):
    assert quantity_factor(1.5, eps) <= quantity_factor(1.1, eps) + 1e-12


@pytest.mark.parametrize("ratio", (1.3, 1.6, 2.61))
def test_a_more_elastic_household_cuts_more(ratio):
    assert quantity_factor(ratio, -0.64) < quantity_factor(ratio, -0.2)
    assert quantity_factor(ratio, -0.2) < quantity_factor(ratio, 0.0)


def test_long_run_is_at_least_as_elastic_as_short_run():
    """Labandeira et al.: demand responds more over longer horizons."""
    for carrier in CARRIERS:
        assert LABANDEIRA_2017_LONG_RUN[carrier] <= LABANDEIRA_2017_SHORT_RUN[carrier]


# --- income-varying specs -------------------------------------------------


def test_priesmann_gas_responsiveness_falls_with_income():
    """The Priesmann gradient is the reason the income-varying spec exists."""
    spec = ElasticitySpec.priesmann_income_varying("short_run")
    assert abs(spec.epsilon("gas", 1)) > abs(spec.epsilon("gas", 10))


def test_priesmann_electricity_responsiveness_rises_with_income():
    spec = ElasticitySpec.priesmann_income_varying("short_run")
    assert abs(spec.epsilon("electricity", 10)) > abs(spec.epsilon("electricity", 1))


def test_income_varying_spec_requires_a_decile():
    spec = ElasticitySpec.priesmann_income_varying()
    with pytest.raises(ValueError, match="decile must be supplied"):
        elasticity_for(spec, "gas")


def test_out_of_range_decile_never_silently_becomes_the_main_spec():
    """A top-coded or missing decile must not resolve to zero elasticity."""
    spec = ElasticitySpec.priesmann_income_varying()
    fallback = elasticity_for(spec, "gas", decile=99)
    assert fallback < 0.0
    assert fallback == pytest.approx(
        sum(spec.by_decile["gas"].values()) / len(spec.by_decile["gas"])
    )


def test_explicit_fallback_is_honoured():
    spec = ElasticitySpec.priesmann_income_varying()
    assert elasticity_for(spec, "gas", 99, missing_decile_fallback=-0.3) == -0.3


@pytest.mark.parametrize("name,spec", ALL_SPECS)
def test_unknown_carrier_raises(name, spec):
    with pytest.raises(KeyError, match="carrier"):
        elasticity_for(spec, "coal", decile=1)  # type: ignore[arg-type]


# --- input validation -----------------------------------------------------


@pytest.mark.parametrize("bad", (0.0, -0.5))
def test_non_positive_price_ratio_is_rejected(bad):
    with pytest.raises(ValueError, match="price_ratio must be positive"):
        spend_factor(bad, 0.0)


def test_negative_baseline_spend_is_rejected():
    with pytest.raises(ValueError, match="baseline_spend"):
        spend_change(-1.0, 1.6, 0.0)


def test_elasticity_past_minus_one_is_rejected():
    """eps <= -1 implies a price rise reduces spend — no estimate supports it."""
    with pytest.raises(ValueError):
        spend_factor(1.6, -1.5)


# --- spec resolution ------------------------------------------------------
#
# ``resolve_elasticity_spec`` used to live in ``uk_iran_conflict.runner``,
# deleted with the second pipeline (docs/FIXES.md C14). It is now part of the
# elasticity module and needs neither PolicyEngine nor microdata.


@pytest.mark.parametrize(
    "name",
    ["main", "zero", "labandeira_short_run", "priesmann_short_run", "prior_repo"],
)
def test_resolves_every_named_spec(name):
    spec = resolve_elasticity_spec(name)
    assert isinstance(spec, ElasticitySpec)
    assert spec.is_main_specification == (name in ("main", "zero"))


def test_rejects_an_unknown_spec_name():
    with pytest.raises(KeyError, match="unknown elasticity spec"):
        resolve_elasticity_spec("not_a_spec")


def test_passes_through_a_spec_object():
    spec = ElasticitySpec.main()
    assert resolve_elasticity_spec(spec) is spec


# --- A5: money-metric welfare bounds --------------------------------------
#
# The bug this guards: ``spend_change`` is a change in EXPENDITURE. Reporting
# it as the cost of the shock counts foregone heating as costless. The CV is
# bracketed by the Paasche and Laspeyres terms, and that bracket is an order of
# magnitude narrower than the spend change implies.


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("eps", [0.0, -0.1, -0.3, -0.5, -0.8])
def test_cv_bounds_bracket_and_order(ratio, eps):
    lo, hi = cv_bounds(1000.0, ratio, eps)
    assert lo <= hi + 1e-12
    assert hi == pytest.approx(laspeyres_cv(1000.0, ratio))
    assert lo == pytest.approx(paasche_cv(1000.0, ratio, eps))


@pytest.mark.parametrize("ratio", RATIOS)
def test_zero_elasticity_collapses_the_bracket_to_the_headline(ratio):
    lo, hi = cv_bounds(1000.0, ratio, 0.0)
    assert lo == pytest.approx(hi)
    assert hi == pytest.approx(static_spend_change(1000.0, ratio))


@pytest.mark.parametrize("eps", [-0.1, -0.3, -0.5, -0.8])
def test_upper_bound_is_elasticity_invariant(eps):
    """The Laspeyres bound is the static headline whatever epsilon is."""
    assert laspeyres_cv(1000.0, 1.14) == pytest.approx(140.0)
    assert cv_bounds(1000.0, 1.14, eps)[1] == pytest.approx(140.0)


@pytest.mark.parametrize("eps", [-0.1, -0.3, -0.5, -0.8])
def test_spend_change_understates_the_welfare_loss(eps):
    """The whole of fix A5, as an assertion."""
    lo, _ = cv_bounds(1000.0, 1.14, eps)
    assert spend_change(1000.0, 1.14, eps) < lo


def test_the_uncertainty_ranking_inverts_at_the_realised_gas_move():
    """At +14% and eps = -0.8 demand response can shave at most ~10%."""
    assert welfare_shaved_share(1.14, -0.8) == pytest.approx(0.0995, abs=5e-4)
    assert deadweight_share(1.14, -0.8) == pytest.approx(0.8103, abs=5e-4)
    # An order of magnitude apart: the spending measure overstates the
    # uncertainty demand response contributes by a factor of eight.
    assert deadweight_share(1.14, -0.8) > 8 * welfare_shaved_share(1.14, -0.8)


def test_welfare_shaving_is_identically_the_consumption_cut():
    for ratio in RATIOS:
        for eps in (0.0, -0.2, -0.64):
            assert welfare_shaved_share(ratio, eps) == pytest.approx(
                consumption_reduction(ratio, eps)
            )


def test_cv_bounds_reject_negative_spend():
    with pytest.raises(ValueError, match="non-negative"):
        cv_bounds(-1.0, 1.1, -0.2)

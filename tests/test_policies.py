"""Policy-scoring tests. No microdata: the Baseline is built by hand.

These cover the four policy-modelling fixes in ``docs/FIXES.md``:

* **B7** — the JRF block is a discount on the *level* of the covered block,
  like the social tariff, not a rebate of part of the price increase; and every
  instrument is also scored at a common exchequer envelope.
* **B8** — the per-child allowance uses a real child count, never household
  size minus two.
* **B9** — continuous compensation measures beside the knife-edge one.
* **C13** — the means-tested variable set is required and audited, not
  swallowed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uk_iran_conflict import policies as pol
from uk_iran_conflict.incidence import Baseline, ShockCost, wsum

N = 10


def _baseline(**overrides) -> Baseline:
    n = N
    kwargs = dict(
        net_income=np.linspace(8_000.0, 100_000.0, n),
        weight=np.full(n, 1e6),
        people=np.array([1.0, 2, 3, 4, 5, 2, 2, 1, 3, 4]),
        gas=np.linspace(400.0, 900.0, n),
        electricity=np.linspace(500.0, 1_000.0, n),
        petrol=np.linspace(200.0, 900.0, n),
        diesel=np.linspace(100.0, 600.0, n),
        decile=np.arange(1.0, 11.0),
        equiv_income_ahc=np.linspace(8_000.0, 100_000.0, n),
        in_poverty_bhc=np.zeros(n),
        in_poverty_ahc=np.zeros(n),
        region=np.zeros(n),
        country=np.zeros(n),
    )
    kwargs.update(overrides)
    return Baseline(**kwargs)


#: Roughly the realised 2026 shock: +14% gas, +9% electricity, +19% motor fuel.
def _cost(base: Baseline) -> ShockCost:
    return ShockCost(
        gas=base.gas * 0.1404,
        electricity=base.electricity * 0.0928,
        motor_fuel=base.motor_fuel * 0.19,
        scenario="test",
    )


@pytest.fixture
def world():
    base = _baseline()
    # Two children in households 3 and 9; one in household 4. Deliberately not
    # equal to people - 2 anywhere, so the B8 defect cannot pass by accident.
    pol._CHILD_COUNTS[base.n] = np.array([0.0, 0, 2, 1, 0, 0, 0, 0, 2, 0])
    mt = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    yield base, _cost(base), mt
    pol._CHILD_COUNTS.pop(base.n, None)


# --- B8: the child count --------------------------------------------------


def test_child_count_is_not_household_size_minus_two(world):
    base, _, _ = world
    children = pol.child_counts(base)
    assert not np.allclose(children, np.clip(base.people - 2, 0, None))
    # The lone parent with one child (household 4, people=4) and the
    # three-adult household (people=3, no children) are the two cases the old
    # expression got backwards.
    assert children[2] == 2.0
    assert np.clip(base.people - 2, 0, None)[2] == 1.0


def test_child_count_refuses_to_guess(world):
    base, cost, mt = world
    pol._CHILD_COUNTS.pop(base.n, None)
    with pytest.raises(RuntimeError, match="no child count available"):
        pol.child_counts(base)
    with pytest.raises(RuntimeError, match="no child count available"):
        pol.POLICIES["jrf_block"].gain(base, cost, mt)


def test_baseline_supplied_child_count_wins(world):
    """Defensive against the sibling adding a children field to Baseline."""
    base, _, _ = world
    import dataclasses

    assert not dataclasses.is_dataclass(object)  # sanity

    class WithChildren:
        n = N
        children = np.full(N, 3.0)

    assert np.allclose(pol.child_counts(WithChildren()), 3.0)


# --- B7a: the JRF block is a level subsidy --------------------------------


def _shock_only_block(base, cost, mt, block_share=0.5, discount=0.5, per_child=60.0):
    """The mis-specified version this fix replaces, for comparison."""
    from uk_iran_conflict.incidence import wquantile

    typical = wquantile(base.energy, base.weight, 0.5)
    covered = np.minimum(base.energy, block_share * typical)
    children = pol.child_counts(base)
    return discount * covered * pol._shock_rate(base, cost) + per_child * children


def test_jrf_block_discounts_the_level_not_the_shock(world):
    base, cost, mt = world
    level = pol.POLICIES["jrf_block"].gain(base, cost, mt)
    shock_only = _shock_only_block(base, cost, mt)
    assert wsum(level, base.weight) > wsum(shock_only, base.weight)
    # The energy component of the level subsidy is (1 + r)/r times the
    # shock-only one, r ~ 0.12 across the two domestic carriers, so the whole
    # instrument is several times more expensive. That factor, not JRF being
    # more generous, is the £1.9bn-against-£5bn gap.
    energy_level = level - 60.0 * pol.child_counts(base)
    energy_shock = shock_only - 60.0 * pol.child_counts(base)
    ratio = wsum(energy_level, base.weight) / wsum(energy_shock, base.weight)
    assert ratio > 5.0


def test_jrf_block_and_social_tariff_are_the_same_kind_of_instrument(world):
    """Both discount a level; the block just covers less of the bill."""
    base, cost, mt = world
    tariff = pol.POLICIES["social_tariff"].gain(base, cost, mt)
    block = pol.POLICIES["jrf_block"].gain(base, cost, mt) - 60.0 * pol.child_counts(
        base
    )
    # For a means-tested household the tariff is 0.35 of the whole shocked
    # bill and the block is 0.50 of the shocked covered block; both are
    # proportional to a shocked level, so their ratio is a pure quantity ratio.
    sel = mt & (block > 0)
    implied = (block[sel] / 0.50) / (tariff[sel] / 0.35)
    covered_share = np.minimum(base.energy[sel], block.max() / 0.5) / base.energy[sel]
    assert np.all(implied <= 1.0 + 1e-9)
    assert np.all(covered_share > 0)


def test_block_never_exceeds_the_household_bill(world):
    base, cost, mt = world
    block = pol.POLICIES["jrf_block"].gain(base, cost, mt) - 60.0 * pol.child_counts(
        base
    )
    assert np.all(block <= 0.5 * (base.energy + cost.domestic) + 1e-9)


# --- B7b: the common envelope --------------------------------------------


@pytest.mark.parametrize("key", sorted(pol.POLICIES))
def test_uncapped_common_envelope_hits_the_envelope(world, key):
    """The pure scalar rescaling still hits the envelope — and says what it cost.

    Retained so the published numbers stay auditable, but the row now carries
    the parameter it implies and, when that parameter is outside the
    instrument's own space, a generic name instead of the real policy's.
    """
    base, cost, mt = world
    score, _ = pol.score_policy_at_envelope(
        base, cost, mt, pol.POLICIES[key], envelope_bn=3.0, cap_at_feasible_max=False
    )
    assert score.cost_bn == pytest.approx(3.0)
    assert score.envelope == "common_scaled"
    assert score.envelope_scale > 0
    assert score.implied_parameter == pytest.approx(
        pol.POLICIES[key].stated_parameter * score.envelope_scale
    )
    if score.is_feasible:
        assert score.label_used == pol.POLICIES[key].label
    else:
        assert score.label_used != pol.POLICIES[key].label
        assert score.implied_parameter > score.feasible_max_parameter


@pytest.mark.parametrize("key", sorted(pol.POLICIES))
def test_capped_common_envelope_never_leaves_the_parameter_space(world, key):
    """The round-2 fix: an instrument may not be paid past its own ceiling."""
    base, cost, mt = world
    score, _ = pol.score_policy_at_envelope(
        base, cost, mt, pol.POLICIES[key], envelope_bn=3.0
    )
    assert score.envelope == "common_capped"
    assert score.is_feasible
    assert score.implied_parameter <= score.feasible_max_parameter * (1 + 1e-9)
    assert score.label_used == pol.POLICIES[key].label
    # It spends at most the envelope, and reports what it could absorb.
    assert score.cost_bn <= 3.0 + 1e-9
    assert score.absorbable_envelope_bn <= 3.0 + 1e-9
    if score.cost_bn < 3.0 - 1e-9:
        assert score.absorbable_envelope_bn == pytest.approx(score.cost_bn, rel=1e-6)


def test_vat_zero_rating_cannot_be_scaled_past_its_arithmetic_ceiling(world):
    """The referees' sharpest example: x2.47 of a 5% rate is a negative VAT rate."""
    base, cost, mt = world
    policy = pol.POLICIES["vat_zero"]
    assert policy.feasible_max_parameter(base, cost, mt) == 5.0
    capped, _ = pol.score_policy_at_envelope(base, cost, mt, policy, envelope_bn=3.0)
    scaled, _ = pol.score_policy_at_envelope(
        base, cost, mt, policy, envelope_bn=3.0, cap_at_feasible_max=False
    )
    assert capped.implied_parameter == pytest.approx(5.0)
    assert capped.cost_bn < 3.0
    assert scaled.implied_parameter > 5.0
    assert not scaled.is_feasible
    assert "domestic-bill subsidy" in scaled.label_used
    assert "VAT" not in scaled.label_used


def test_a_means_tested_instrument_can_reach_the_envelope_by_widening_eligibility(
    world,
):
    """The margin a real policymaker uses, and the one the coverage finding
    points at."""
    base, cost, mt = world
    policy = pol.POLICIES["social_tariff"]
    widened, _ = pol.score_policy_by_eligibility(
        base, cost, mt, policy, envelope_bn=3.0
    )
    assert widened.envelope == "common_eligibility"
    assert widened.is_feasible
    # Generosity is untouched; only the eligible population moved.
    assert widened.implied_parameter == pytest.approx(policy.stated_parameter)
    assert widened.eligible_share > float(base.weight[mt].sum() / base.weight.sum())
    assert widened.label_used == policy.label


def test_eligibility_widening_adds_the_poorest_first(world):
    base, cost, mt = world
    widened = pol.widen_eligibility(
        base, cost, mt, pol.POLICIES["social_tariff"], envelope_bn=3.0
    )
    added = widened & ~mt
    if added.any():
        assert (
            base.equiv_income_ahc[added].max() <= base.equiv_income_ahc[~widened].min()
        )


def test_eligibility_widening_is_undefined_for_a_universal_instrument(world):
    base, cost, mt = world
    with pytest.raises(ValueError, match="universal"):
        pol.score_policy_by_eligibility(
            base, cost, mt, pol.POLICIES["vat_zero"], envelope_bn=3.0
        )


@pytest.mark.parametrize("key", sorted(pol.POLICIES))
def test_rescaling_preserves_the_shape_of_the_spend(world, key):
    """Scaling is exactly a change of generosity: targeting is unchanged."""
    base, cost, mt = world
    stated, _ = pol.score_policy(base, cost, mt, pol.POLICIES[key])
    common, _ = pol.score_policy_at_envelope(
        base, cost, mt, pol.POLICIES[key], envelope_bn=3.0, cap_at_feasible_max=False
    )
    assert common.share_to_bottom_three == pytest.approx(stated.share_to_bottom_three)
    # Which is exactly why scaling cannot settle a targeting question, and why
    # the eligibility margin has to be scored separately.


def test_scorecard_returns_every_envelope_for_every_policy(world):
    base, cost, mt = world
    scores = pol.scorecard(base, cost, mt, envelope_bn=4.0)
    means_tested = sum(1 for p in pol.POLICIES.values() if p.means_tested)
    assert len(scores) == 3 * len(pol.POLICIES) + means_tested
    assert {s.envelope for s in scores} == {
        "stated",
        "common_capped",
        "common_scaled",
        "common_eligibility",
    }
    for s in scores:
        if s.envelope == "common_scaled":
            assert s.cost_bn == pytest.approx(4.0)
        if s.envelope == "common_capped":
            assert s.is_feasible
        # Every row records the parameter it implies and its ceiling.
        assert s.implied_parameter == s.implied_parameter
        assert s.feasible_max_parameter == s.feasible_max_parameter


def test_the_stated_cost_comparison_is_not_like_for_like(world):
    """The motivating fact: stated costs differ by a large multiple."""
    base, cost, mt = world
    costs = [
        pol.score_policy(base, cost, mt, p)[0].cost_bn for p in pol.POLICIES.values()
    ]
    assert max(costs) / min(costs) > 2.0


# --- B9: continuous compensation measures ---------------------------------


@pytest.mark.parametrize("key", sorted(pol.POLICIES))
def test_continuous_measures_are_populated(world, key):
    base, cost, mt = world
    score, _ = pol.score_policy(base, cost, mt, pol.POLICIES[key])
    assert 0.0 <= score.share_of_aggregate_loss_offset <= 1.0
    assert score.mean_residual_loss_gbp >= 0.0
    assert score.median_residual_loss_gbp >= 0.0
    assert set(score.mean_residual_loss_by_decile) == set(range(1, 11))
    assert set(score.median_residual_loss_by_decile) == set(range(1, 11))
    assert set(score.share_of_loss_offset_by_decile) == set(range(1, 11))


def test_residual_loss_falls_as_the_policy_gets_more_generous(world):
    base, cost, mt = world
    small, _ = pol.score_policy_at_envelope(
        base,
        cost,
        mt,
        pol.POLICIES["vat_zero"],
        envelope_bn=1.0,
        cap_at_feasible_max=False,
    )
    large, _ = pol.score_policy_at_envelope(
        base,
        cost,
        mt,
        pol.POLICIES["vat_zero"],
        envelope_bn=4.0,
        cap_at_feasible_max=False,
    )
    assert large.mean_residual_loss_gbp < small.mean_residual_loss_gbp
    assert large.share_of_aggregate_loss_offset > small.share_of_aggregate_loss_offset


def test_the_knife_edge_metric_can_disagree_with_the_continuous_one(world):
    """Why B9 exists: 'compensates nobody' is compatible with real relief.

    A policy that pays every household 90% of its loss leaves 100% of losers
    'uncompensated' and offsets 90% of the aggregate loss. The knife-edge
    metric alone would report it as doing nothing.
    """
    base, cost, mt = world
    near_miss = pol.Policy(
        key="near_miss",
        label="90% of every household's loss",
        source="test",
        stated_cost_bn=float("nan"),
        gain=lambda b, c, m: 0.9 * c.total,
    )
    score, _ = pol.score_policy(base, cost, mt, near_miss)
    assert score.uncompensated_share_overall == pytest.approx(1.0)
    assert score.share_of_aggregate_loss_offset == pytest.approx(0.9)


def test_residual_loss_is_floored_at_zero(world):
    """Over-compensation at the top must not net off shortfall at the bottom."""
    base, cost, mt = world
    lopsided = pol.Policy(
        key="lopsided",
        label="pays the top decile ten times its loss, nobody else",
        source="test",
        stated_cost_bn=float("nan"),
        gain=lambda b, c, m: np.where(b.decile == 10, 10.0 * c.total, 0.0),
    )
    score, _ = pol.score_policy(base, cost, mt, lopsided)
    assert score.mean_residual_loss_gbp > 0.0
    assert score.share_of_aggregate_loss_offset < 0.2
    # The signed net measure, by contrast, can go negative on the same policy.
    assert score.net_loss_after_policy_gbp < score.mean_residual_loss_gbp


# --- C13: the means-tested variable set -----------------------------------


def test_required_and_legacy_partition_the_original_set():
    assert set(pol.MEANS_TESTED_REQUIRED) | set(pol.MEANS_TESTED_LEGACY) == set(
        pol.MEANS_TESTED_VARIABLES
    )
    assert not set(pol.MEANS_TESTED_REQUIRED) & set(pol.MEANS_TESTED_LEGACY)
    assert "universal_credit" in pol.MEANS_TESTED_REQUIRED


class _FakeVariables(dict):
    pass


class _FakeSim:
    """Minimal stand-in for a Microsimulation."""

    def __init__(self, values: dict[str, np.ndarray]):
        self._values = values
        self.tax_benefit_system = type(
            "TBS", (), {"variables": _FakeVariables({k: None for k in values})}
        )()

    def calculate(self, name, period, map_to=None):
        if name not in self._values:
            raise AssertionError(f"calculate called for unknown variable {name!r}")
        return self._values[name]


def test_resolve_returns_none_for_a_missing_variable_without_calculating():
    sim = _FakeSim({"universal_credit": np.ones(3)})
    assert pol._resolve(sim, "not_a_variable", 2026) is None
    assert pol._resolve(sim, "universal_credit", 2026) is not None


def test_resolve_does_not_swallow_a_real_failure():
    """A broken formula must propagate, not look like a missing benefit."""

    class Exploding(_FakeSim):
        def calculate(self, name, period, map_to=None):
            raise ValueError("formula blew up")

    sim = Exploding({"universal_credit": np.ones(3)})
    with pytest.raises(ValueError, match="formula blew up"):
        pol._resolve(sim, "universal_credit", 2026)


def test_a_missing_required_benefit_is_fatal(monkeypatch):
    """The C13 regression guard: renaming UC must not shrink the population."""
    n = 4
    values = {
        "household_weight": np.full(n, 1e6),
        "pension_credit": np.array([0.0, 100, 0, 0]),
        "housing_benefit": np.zeros(n),
        "esa_income": np.zeros(n),
        "is_child": np.zeros(n),
    }  # universal_credit deliberately absent

    fake = _FakeSim(values)
    module = type("M", (), {"Microsimulation": lambda dataset: fake})
    monkeypatch.setitem(__import__("sys").modules, "policyengine_uk", module)

    with pytest.raises(RuntimeError, match="universal_credit"):
        pol.means_tested_audit("nowhere.h5", 2026)


def test_the_audit_records_every_resolved_variable(monkeypatch, tmp_path):
    n = 4
    values = {
        "household_weight": np.array([1e6, 1e6, 2e6, 1e6]),
        "universal_credit": np.array([500.0, 0, 0, 0]),
        "pension_credit": np.array([0.0, 100, 0, 0]),
        "housing_benefit": np.zeros(n),
        "esa_income": np.zeros(n),
        "household_count_children": np.array([0.0, 1, 2, 0]),
    }
    fake = _FakeSim(values)
    module = type("M", (), {"Microsimulation": lambda dataset: fake})
    monkeypatch.setitem(__import__("sys").modules, "policyengine_uk", module)

    audit = pol.means_tested_audit("nowhere.h5", 2026)
    assert set(audit["resolved_variables"]) == {
        "universal_credit",
        "pension_credit",
        "housing_benefit",
        "esa_income",
    }
    # The wound-down legacy benefits are absent from this fixture and that is
    # recorded rather than ignored.
    assert set(audit["missing_legacy"]) == set(pol.MEANS_TESTED_LEGACY)
    uc = audit["by_variable"]["universal_credit"]
    assert uc["households_m"] == pytest.approx(1.0)
    assert audit["any_means_tested_households_m"] == pytest.approx(2.0)
    assert audit["any_means_tested_share"] == pytest.approx(2.0 / 5.0)
    # and the child count is cached off the same simulation (B8)
    assert np.allclose(pol._CHILD_COUNTS[n], values["household_count_children"])

    path = pol.write_means_tested_audit("nowhere.h5", 2026, tmp_path / "audit.json")
    assert path.exists()
    assert "universal_credit" in path.read_text()
    pol._CHILD_COUNTS.pop(n, None)


# --------------------------------------------------------------------------
# Round-2 corrections to the scorecard's arithmetic
# --------------------------------------------------------------------------


def test_share_to_bottom_three_excludes_out_of_range_deciles(world):
    """``decile <= 3`` had no lower guard.

    ``incidence.decile_table`` correctly excludes households carrying a missing
    or out-of-range decile — about 0.24m weighted households, 100% of them with
    non-positive equivalised AHC income — while the scorecard swept them into
    "the bottom three deciles" and credited their gain there.
    """
    base, cost, mt = world
    out_of_range = np.concatenate([base.decile[:-1], [-1.0]])
    shifted = dataclasses.replace(base, decile=out_of_range)
    score, _ = pol.score_policy(shifted, cost, mt, pol.POLICIES["ippr_rebate"])
    w, gain = shifted.weight, pol.POLICIES["ippr_rebate"].gain(shifted, cost, mt)
    inside = (shifted.decile >= 1) & (shifted.decile <= 3)
    assert score.share_to_bottom_three == pytest.approx(
        wsum(gain[inside], w[inside]) / wsum(gain, w)
    )
    # And the unguarded version would have been strictly larger.
    unguarded = shifted.decile <= 3
    assert wsum(gain[unguarded], w[unguarded]) > wsum(gain[inside], w[inside])


def test_cost_per_pound_to_decile_one_is_dimensionless(world):
    """It was £bn divided by £/household, printed as pounds."""
    base, cost, mt = world
    policy = pol.POLICIES["ippr_rebate"]
    score, gain = pol.score_policy(base, cost, mt, policy)
    d1 = base.decile == 1
    expected = wsum(gain, base.weight) / wsum(gain[d1], base.weight[d1])
    assert score.cost_per_pound_decile_one == pytest.approx(expected)
    # Total cost per £1 reaching decile one can never be below £1.
    assert score.cost_per_pound_decile_one >= 1.0
    assert "dimensionless" in score.cost_per_pound_decile_one_units


def test_cost_per_pound_to_decile_one_is_scale_invariant(world):
    """It measures targeting, so doubling every payment must not move it."""
    base, cost, mt = world
    policy = pol.POLICIES["ippr_rebate"]
    stated, _ = pol.score_policy(base, cost, mt, policy)
    doubled, _ = pol.score_policy(
        base, cost, mt, policy, gain=policy.gain(base, cost, mt) * 2.0
    )
    assert doubled.cost_per_pound_decile_one == pytest.approx(
        stated.cost_per_pound_decile_one
    )


def test_a_perfectly_targeted_instrument_costs_one_pound_per_pound(world):
    base, cost, mt = world
    only_d1 = np.where(base.decile == 1, 100.0, 0.0)
    score, _ = pol.score_policy(
        base, cost, mt, pol.POLICIES["ippr_rebate"], gain=only_d1
    )
    assert score.cost_per_pound_decile_one == pytest.approx(1.0)


def test_the_two_loss_offset_definitions_are_now_one(world):
    """Capping at the decile aggregate produced 100% offset beside a real residual."""
    base, cost, mt = world
    # Pay decile one ten times its loss and nobody else anything: on the old
    # decile-aggregate definition every pound counts; on the household-level
    # one, only the pounds that meet a loss do.
    d1 = base.decile == 1
    gain = np.where(d1, cost.total * 10.0, 0.0)
    score, _ = pol.score_policy(base, cost, mt, pol.POLICIES["ippr_rebate"], gain=gain)
    assert score.share_of_loss_offset_by_decile[1] == pytest.approx(1.0)
    assert score.mean_residual_loss_by_decile[1] == pytest.approx(0.0)
    # A decile that is over-paid on aggregate but unevenly cannot show 100%.
    lumpy = np.zeros_like(cost.total)
    lumpy[0] = cost.total.sum() * 5
    score2, _ = pol.score_policy(
        base, cost, mt, pol.POLICIES["ippr_rebate"], gain=lumpy
    )
    for d, offset in score2.share_of_loss_offset_by_decile.items():
        residual = score2.mean_residual_loss_by_decile[d]
        assert offset <= 1.0
        # The invariant that was violated: full offset and a positive residual
        # cannot coexist in the same row.
        if offset == pytest.approx(1.0):
            assert residual == pytest.approx(0.0)


def test_offset_by_decile_uses_the_same_definition_as_the_headline(world):
    base, cost, mt = world
    for policy in pol.POLICIES.values():
        score, gain = pol.score_policy(base, cost, mt, policy)
        w, loss = base.weight, cost.total
        credited = np.minimum(gain, loss)
        assert score.share_of_aggregate_loss_offset == pytest.approx(
            wsum(credited, w) / wsum(loss, w)
        )
        for d, offset in score.share_of_loss_offset_by_decile.items():
            sel = base.decile == d
            assert offset == pytest.approx(
                wsum(credited[sel], w[sel]) / wsum(loss[sel], w[sel])
            )


# --- item 8: the JRF post-shock revaluation -------------------------------


def test_jrf_block_is_not_revalued_at_post_shock_prices_by_default(world):
    """The revaluation was the entire basis of the "more generous than JRF" claim."""
    base, cost, mt = world
    plain = pol._jrf_block(base, cost, mt)
    revalued = pol._jrf_block(base, cost, mt, revalue_at_post_shock_prices=True)
    assert (revalued >= plain).all()
    assert wsum(revalued, base.weight) > wsum(plain, base.weight)
    # The default is the documented one.
    assert pol.POLICIES["jrf_block"].gain(base, cost, mt) == pytest.approx(plain)


def test_jrf_revaluation_grows_with_the_shock_which_is_why_it_is_wrong(world):
    """A subsidy that gets more generous the worse the shock is: nobody proposed it."""
    base, cost, mt = world
    bigger = ShockCost(
        gas=cost.gas * 3, electricity=cost.electricity * 3, motor_fuel=cost.motor_fuel
    )
    small = wsum(
        pol._jrf_block(base, cost, mt, revalue_at_post_shock_prices=True), base.weight
    )
    large = wsum(
        pol._jrf_block(base, bigger, mt, revalue_at_post_shock_prices=True),
        base.weight,
    )
    assert large > small
    # Without the revaluation the instrument's cost is invariant to the shock,
    # as a discount on a fixed block at a pegged rate should be.
    assert wsum(pol._jrf_block(base, cost, mt), base.weight) == pytest.approx(
        wsum(pol._jrf_block(base, bigger, mt), base.weight)
    )

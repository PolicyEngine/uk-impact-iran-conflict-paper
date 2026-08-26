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
def test_common_envelope_hits_the_envelope(world, key):
    base, cost, mt = world
    score, _ = pol.score_policy_at_envelope(
        base, cost, mt, pol.POLICIES[key], envelope_bn=3.0
    )
    assert score.cost_bn == pytest.approx(3.0)
    assert score.envelope == "common"
    assert score.envelope_scale > 0


@pytest.mark.parametrize("key", sorted(pol.POLICIES))
def test_rescaling_preserves_the_shape_of_the_spend(world, key):
    """Scaling is exactly a change of generosity: targeting is unchanged."""
    base, cost, mt = world
    stated, _ = pol.score_policy(base, cost, mt, pol.POLICIES[key])
    common, _ = pol.score_policy_at_envelope(
        base, cost, mt, pol.POLICIES[key], envelope_bn=3.0
    )
    assert common.share_to_bottom_three == pytest.approx(stated.share_to_bottom_three)


def test_scorecard_returns_both_envelopes_for_every_policy(world):
    base, cost, mt = world
    scores = pol.scorecard(base, cost, mt, envelope_bn=4.0)
    assert len(scores) == 2 * len(pol.POLICIES)
    assert {s.envelope for s in scores} == {"stated", "common"}
    for s in scores:
        if s.envelope == "common":
            assert s.cost_bn == pytest.approx(4.0)


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
        base, cost, mt, pol.POLICIES["vat_zero"], envelope_bn=1.0
    )
    large, _ = pol.score_policy_at_envelope(
        base, cost, mt, pol.POLICIES["vat_zero"], envelope_bn=4.0
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

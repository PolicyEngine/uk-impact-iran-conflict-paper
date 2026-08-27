"""Grid coordinates, the degenerate cell, and the sensitivity sweeps.

No microdata: the sweeps are exercised against a hand-built ``Baseline``.
Covers ``docs/FIXES.md`` A5 (welfare bounds in the elasticity sweep), D24 (the
cap-lag identity), D25 (damped-equivalent named-scenario coordinates), D26
(fuel composition by decile) and E35 (the degenerate grid corner).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import run_grid  # noqa: E402
import run_sensitivity as rs  # noqa: E402

from uk_iran_conflict import policies as pol  # noqa: E402
from uk_iran_conflict import scenarios as scen  # noqa: E402
from uk_iran_conflict.incidence import Baseline  # noqa: E402

N = 20


def _baseline() -> Baseline:
    rng = np.random.default_rng(0)
    return Baseline(
        net_income=np.linspace(8_000.0, 100_000.0, N),
        weight=np.full(N, 1e6),
        people=rng.integers(1, 5, N).astype(float),
        gas=np.linspace(400.0, 900.0, N),
        electricity=np.linspace(500.0, 1_000.0, N),
        petrol=np.linspace(200.0, 900.0, N),
        diesel=np.linspace(100.0, 600.0, N),
        decile=np.repeat(np.arange(1.0, 11.0), 2),
        equiv_income_ahc=np.linspace(8_000.0, 100_000.0, N),
        in_poverty_bhc=np.zeros(N),
        in_poverty_ahc=np.zeros(N),
        region=np.zeros(N),
        country=np.zeros(N),
    )


@pytest.fixture
def base():
    return _baseline()


@pytest.fixture
def scenario():
    return scen.SCENARIOS["realised_2026"]


# --- D25: named scenarios must be plotted where the grid can reproduce them


def test_named_points_carry_damped_equivalent_coordinates():
    named = run_grid.named_points()
    for col in ("gas_pct", "oil_pct", "gas_pct_damped", "oil_pct_damped"):
        assert col in named.columns
    assert named["gas_pct_damped"].notna().all()
    assert named["oil_pct_damped"].notna().all()


def test_a_damped_scenario_moves_and_an_undamped_one_does_not():
    named = run_grid.named_points().set_index("scenario")
    realised = named.loc["realised_2026"]
    # The realised path damps its peak gas move to 0.36, so its grid cell is
    # materially to the left of its headline coordinate.
    assert realised["gas_pct_damped"] < realised["gas_pct"]
    assert realised["gas_pct_damped"] == pytest.approx(
        realised["gas_pct"] * realised["gas_sustained_fraction"]
    )
    # NIESR's scenarios are specified as sustained levels, so they do not move
    # on the gas axis at all.
    for key in ("niesr_baseline", "niesr_adverse"):
        row = named.loc[key]
        assert row["gas_pct_damped"] == pytest.approx(row["gas_pct"])


def test_damping_moves_the_realised_point_to_the_correct_side_of_the_frontier():
    """The contradiction D25 records, as a regression test.

    Motor fuel's share of the loss in a grid cell depends on the *ratio* of the
    oil move to the gas move. Damping the gas leg by 0.36 while damping the
    pump leg by much less raises that ratio, so the damped-equivalent cell is
    the more fuel-heavy one — the same side of the 50% frontier as the realised
    scenario's own result (motor fuel is the majority of its loss). The
    headline coordinates put it on the other side.
    """
    row = run_grid.named_points().set_index("scenario").loc["realised_2026"]
    headline_ratio = row["oil_pct"] / row["gas_pct"]
    damped_ratio = row["oil_pct_damped"] / row["gas_pct_damped"]
    assert damped_ratio > headline_ratio


# --- E35: the degenerate corner ------------------------------------------


def test_the_zero_shock_cell_is_flagged_and_blanked(base):
    row = run_grid.cell_row(base, 0.0, 0.0)
    assert row["is_degenerate"] is True
    for col in (
        "motor_fuel_share_pct",
        "gas_share_pct",
        "electricity_share_pct",
        "domestic_share_pct",
        "d1_d10_ratio",
    ):
        assert np.isnan(row[col]), col


def test_a_live_cell_is_not_flagged(base):
    row = run_grid.cell_row(base, 0.4, 0.4)
    assert row["is_degenerate"] is False
    assert np.isfinite(row["motor_fuel_share_pct"])
    assert 0.0 <= row["motor_fuel_share_pct"] <= 100.0


# --- A5: the elasticity sweep reports welfare, not just spending ----------


def test_elasticity_sweep_reports_cv_bounds(base, scenario):
    frame = rs.sweep_elasticity(base, scenario)
    for col in ("cv_lower_bn", "cv_upper_bn", "welfare_share_shaved"):
        assert col in frame.columns
    # The Laspeyres bound does not depend on the elasticity, so it is the same
    # in every row and equals the zero-elasticity headline.
    assert frame["cv_upper_bn"].nunique() == 1
    main = frame[frame["is_main_specification"]].iloc[0]
    assert main["cv_upper_bn"] == pytest.approx(main["aggregate_loss_bn"])
    assert main["cv_lower_bn"] == pytest.approx(main["cv_upper_bn"])


def test_the_uncertainty_ranking_inverts(base, scenario):
    """The spend measure overstates demand-response uncertainty ~8-fold."""
    frame = rs.sweep_elasticity(base, scenario).set_index("spec")
    row = frame.loc["flat_-0.8"]
    assert row["cv_lower_bn"] > 4 * row["aggregate_spend_change_bn"]
    assert row["welfare_share_shaved"] < 0.15
    assert row["share_of_upper_bound_shaved"] > 0.75
    assert row["cv_lower_bn"] <= row["cv_upper_bn"]


@pytest.mark.parametrize("spec_name", ["flat_-0.3", "flat_-0.8"])
def test_cv_bounds_bracket_the_spend_change(base, scenario, spec_name):
    frame = rs.sweep_elasticity(base, scenario).set_index("spec")
    row = frame.loc[spec_name]
    assert row["aggregate_spend_change_bn"] < row["cv_lower_bn"]


# --- D24: the cap-lag invariance is an identity ---------------------------


def test_cap_lag_sweep_reconciles_with_the_headline(base, scenario):
    """Round-2 finding 1 / ``docs/FIXES.md`` A2, and the whole point of the sweep.

    Before the revision the headline and this appendix reported the same
    specification as £304 and £205 — a 48% gap — because they applied unrelated
    arithmetic (a hand-written phase-in tuple versus a sliding calendar count)
    over unrelated windows. Both now go through
    ``scenarios.cap_phase_in_profile``, so the central row *is* the headline.
    """
    from uk_iran_conflict import scenarios as scen
    from uk_iran_conflict.incidence import run_scenario

    frame = rs.sweep_cap_lag(base, scenario)
    headline, _ = run_scenario(base, scenario)

    central = frame[frame["is_central_specification"]]
    assert len(central) == 1
    assert central["lag_quarters"].iloc[0] == pytest.approx(scen.CAP_LAG_QUARTERS)
    assert central["mean_loss_gbp"].iloc[0] == pytest.approx(headline.mean_loss_gbp)
    assert bool(central["reconciles_with_headline"].iloc[0])


def test_cap_lag_sweep_holds_the_cap_anchor_when_anchored(base, scenario):
    from uk_iran_conflict import scenarios as scen

    frame = rs.sweep_cap_lag(base, scenario)
    anchored = frame[frame["anchor"] == "anchored"]
    assert len(anchored) == len(rs.CAP_LAG_GRID)
    assert np.allclose(anchored["cap_anchor_quarter_pct"], 100 * scen.CAP_ANCHOR_PCT)
    # The anchored domestic leg is nearly lag-invariant, which is now a result
    # about an externally anchored cap rather than an identity. The unanchored
    # one is not: that is how much of the invariance the anchor is doing.
    anchored_spread = anchored["mean_loss_gbp"].max() - anchored["mean_loss_gbp"].min()
    unanchored = frame[frame["anchor"] == "unanchored"]
    unanchored_spread = (
        unanchored["mean_loss_gbp"].max() - unanchored["mean_loss_gbp"].min()
    )
    assert anchored_spread < unanchored_spread
    # And the whole sweep sits far inside the 48% gap it replaces.
    assert anchored_spread / anchored["mean_loss_gbp"].min() < 0.15


def test_cap_lag_sweep_derives_the_profile_from_the_lag(base, scenario):
    from uk_iran_conflict import scenarios as scen

    frame = rs.sweep_cap_lag(base, scenario)
    for lag, profile in zip(
        frame["lag_quarters"], frame["phase_in_profile"], strict=True
    ):
        expected = scen.cap_phase_in_profile(lag)
        assert profile == ";".join(f"{v:.4f}" for v in expected)
    # A longer lag pushes the cap move out of the modelled window.
    anchored = frame[frame["anchor"] == "anchored"].sort_values("lag_quarters")
    assert anchored["annual_phase_in_gas"].is_monotonic_decreasing


# --- D26: petrol, diesel and diesel share by decile -----------------------


def test_fuel_composition_table_lets_the_data_choose(base, scenario):
    frame = rs.sweep_fuel_by_decile(base, scenario)
    assert len(frame) == 10
    for col in (
        "petrol_spend_gbp",
        "diesel_spend_gbp",
        "diesel_share_of_fuel_spend",
        "diesel_share_of_fuel_loss",
    ):
        assert col in frame.columns
    assert frame["diesel_share_of_fuel_spend"].between(0, 1).all()
    # Diesel is uplifted harder than petrol, so diesel's share of the loss
    # exceeds its share of spend in every decile. That is the alternative
    # explanation for the decile-8 cash spike the narration attributes to
    # mileage.
    assert (
        frame["diesel_share_of_fuel_loss"] > frame["diesel_share_of_fuel_spend"]
    ).all()


# --- B7/B9: the scorecard sweep runs end to end ---------------------------


def test_policy_envelope_sweep(base, scenario):
    pol._CHILD_COUNTS[base.n] = np.tile([0.0, 1.0], base.n // 2)
    try:
        mt = np.arange(base.n) % 3 == 0
        frame = rs.sweep_policy_envelope(base, scenario, mt, envelope_bn=5.0)
    finally:
        pol._CHILD_COUNTS.pop(base.n, None)
    # stated + common_capped + common_scaled for every policy, plus a
    # common_eligibility row for each means-tested one.
    means_tested = sum(1 for p in pol.POLICIES.values() if p.means_tested)
    assert len(frame) == 3 * len(pol.POLICIES) + means_tested
    scaled = frame[frame["envelope"] == "common_scaled"]
    assert np.allclose(scaled["cost_bn"], 5.0)
    # Round-2 finding 2: the implied parameter is on every row, and any row
    # outside the instrument's parameter space is renamed.
    for _, row in frame.iterrows():
        assert row["implied_parameter"] == row["implied_parameter"]  # not NaN
        if not row["is_feasible"]:
            assert row["label_used"] != pol.POLICIES[row["policy"]].label
    capped = frame[frame["envelope"] == "common_capped"]
    assert (capped["is_feasible"]).all()
    assert (capped["cost_bn"] <= 5.0 + 1e-9).all()
    for col in (
        "share_of_aggregate_loss_offset",
        "mean_residual_loss_gbp",
        "median_residual_loss_gbp",
        "mean_residual_loss_d1",
        "share_of_loss_offset_d10",
    ):
        assert col in frame.columns

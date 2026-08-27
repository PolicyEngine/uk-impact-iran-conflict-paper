"""Consistency tests on the ``results/`` tree itself.

**Round-4 finding 4, and the reason it survived three rounds of refereeing.**
The other 500-odd tests in this suite run on synthetic fixtures built by hand.
Nothing microdata-dependent is tested, and nothing at all looks at what the
pipelines actually *wrote*. So when ``results/round3_findings.json`` came to
carry a ``motor_fuel_margins`` sub-block written by a superseded version of
``analysis/run_variants.py`` — bit-identical values for all five calibrations,
contradicting the ``means_tested_fuel`` block a few lines below it in the same
file — no test could see it, because no test read the file.

These tests read the file. They are not a substitute for the unit tests: they
check that the published artefacts agree **with each other** and with the code
that claims to have produced them, which is a different and previously
unguarded property.

Every test skips if ``results/`` is absent, so the suite still runs in a clean
checkout with no microdata and no Hugging Face token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uk_iran_conflict import incidence as inc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

#: Modules whose changes invalidate the results tree. Deliberately not the
#: emitters or the figure scripts: those consume ``results/`` and cannot make it
#: stale.
PRODUCING_MODULES: tuple[Path, ...] = (
    ROOT / "uk_iran_conflict",
    ROOT / "analysis" / "run_incidence.py",
    ROOT / "analysis" / "run_variants.py",
    ROOT / "analysis" / "run_grid.py",
    ROOT / "analysis" / "run_combinations.py",
    ROOT / "analysis" / "run_sensitivity.py",
)


def _load(relative: str) -> dict:
    path = RESULTS / relative
    if not path.exists():
        pytest.skip(f"{path} not present; run the pipelines first")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# The stale block itself
# --------------------------------------------------------------------------


def test_the_two_fuel_blocks_in_round3_findings_agree():
    """``motor_fuel_margins`` and ``means_tested_fuel`` report the same
    quantity by two call paths. The published file had them disagreeing by a
    factor of seven and nothing noticed.
    """
    findings = _load("round3_findings.json")
    margins = findings["motor_fuel_margins"]
    headline = findings["means_tested_fuel"]
    for calibration in inc.CALIBRATIONS:
        left = margins[calibration]["means_tested_mean_fuel_gbp"]
        right = headline[calibration]["means_tested_mean_fuel_gbp"]
        assert left == pytest.approx(right, rel=1e-9), (
            f"{calibration}: motor_fuel_margins says {left}, means_tested_fuel "
            f"says {right}. One of the two blocks is stale — re-run "
            "analysis/run_variants.py."
        )


def test_the_margins_block_actually_moves_with_the_calibration():
    """The stale block's signature: five identical numbers where the
    calibrations exist precisely to make them differ.
    """
    findings = _load("round3_findings.json")
    values = [
        findings["motor_fuel_margins"][c]["means_tested_mean_fuel_gbp"]
        for c in inc.CALIBRATIONS
    ]
    assert len(set(values)) > 1, (
        "every calibration reports the same means-tested mean motor-fuel spend "
        f"({values[0]}). The calibrations exist to move it; the block was "
        "written by an earlier version and never regenerated."
    )


def test_the_pipeline_persists_its_own_cross_check():
    findings = _load("round3_findings.json")
    assert findings["fuel_block_consistency"]["all_agree"] is True


# --------------------------------------------------------------------------
# A stale tree, generally
# --------------------------------------------------------------------------


def _newest_source_mtime() -> float:
    newest = 0.0
    for target in PRODUCING_MODULES:
        if target.is_dir():
            candidates = target.rglob("*.py")
        elif target.exists():
            candidates = [target]
        else:
            continue
        for path in candidates:
            newest = max(newest, path.stat().st_mtime)
    return newest


def test_the_results_tree_is_not_older_than_the_code_that_produced_it():
    """The general form of finding 4: an artefact older than its producer.

    This is the test that would have caught the stale sub-block without anyone
    having to guess which sub-block to look at. If it fails, re-run the
    pipelines; do not touch the timestamps.
    """
    if not RESULTS.exists():
        pytest.skip("no results/ tree")
    artefacts = sorted(RESULTS.rglob("*.json")) + sorted(RESULTS.rglob("*.csv"))
    if not artefacts:
        pytest.skip("no artefacts in results/")
    newest_source = _newest_source_mtime()
    stale = [
        str(path.relative_to(ROOT))
        for path in artefacts
        if path.stat().st_mtime < newest_source
    ]
    assert not stale, (
        f"{len(stale)} artefact(s) predate the newest producing module: "
        f"{stale[:8]}{' ...' if len(stale) > 8 else ''}. Re-run "
        "analysis/run_incidence.py, run_variants.py, run_combinations.py, "
        "run_grid.py and run_sensitivity.py."
    )


# --------------------------------------------------------------------------
# Round-4 findings 1, 2, 3 and 6 must be present in what was written
# --------------------------------------------------------------------------


def test_the_coincident_feasible_maxima_are_flagged_in_the_diagnostics():
    diagnostics = _load("policy_diagnostics.json")
    identity = diagnostics["feasible_max_identity"]
    assert identity["report_as_one_result"] is True
    groups = [set(g["policies"]) for g in identity["identical_groups"]]
    assert {"social_tariff", "whd_expansion"} in groups


def test_the_two_means_tested_feasible_max_costs_are_the_same_written_number():
    diagnostics = _load("policy_diagnostics.json")
    by_policy = diagnostics["by_policy"]
    assert by_policy["social_tariff"]["feasible_max_cost_bn"] == pytest.approx(
        by_policy["whd_expansion"]["feasible_max_cost_bn"], rel=1e-6
    )


def test_the_whd_ceiling_rule_is_written_down_in_the_diagnostics():
    diagnostics = _load("policy_diagnostics.json")
    ceilings = diagnostics["flat_payment_ceilings"]
    assert ceilings["rule_used"] == "mean_eligible_domestic_bill"
    assert ceilings["bill_over_loss"] > 1.0
    whd = diagnostics["by_policy"]["whd_expansion"]
    assert whd["feasible_max_parameter"] == pytest.approx(
        ceilings["mean_eligible_domestic_bill_gbp"], rel=1e-9
    )


def test_the_admission_rule_range_is_persisted():
    diagnostics = _load("policy_diagnostics.json")
    for key in ("social_tariff", "whd_expansion"):
        rules = diagnostics["by_policy"][key]["admission_rules"]
        assert set(rules["by_rule"]) == {
            "equivalised_ahc_income",
            "unequivalised_net_income",
            "highest_domestic_bill",
            "random",
        }
        assert rules["by_rule"]["equivalised_ahc_income"]["is_upper_bound"] is True
        spread = rules["range"]["share_of_aggregate_loss_offset"]
        assert spread["max"] >= spread["min"]


def test_the_jrf_gap_is_resolved_in_the_written_diagnostics():
    diagnostics = _load("policy_diagnostics.json")
    gap = diagnostics["jrf_costing_gap"]
    assert gap["comparable"] is True
    assert gap["resolution"].startswith("RESOLVED")
    # The decomposition must add up in the file, not only in the code.
    d = gap["decomposition"]
    assert d["modelled_block_bn"] + d["per_child_allowance_bn"] == pytest.approx(
        gap["modelled_cost_bn"], rel=1e-9
    )
    assert gap["modelled_cost_bn"] - gap["sponsor_cost_bn"] == pytest.approx(
        gap["gap_bn"], rel=1e-9
    )


def test_the_grid_does_not_advertise_a_check_it_does_not_run():
    recon = _load("grid/reconciliation.json")
    scope = recon["grid_scope"]
    assert scope["range_check_enforced"] is False
    assert recon["sub_channel_bracketing_holds"] is True
    assert recon["channel_mix_identity_holds"] is True
    # Whatever the range check says, the enforced checks must have passed for
    # the file to exist at all.
    assert recon["sub_channel_bracketing_broken"] == []
    assert recon["identity_broken"] == []

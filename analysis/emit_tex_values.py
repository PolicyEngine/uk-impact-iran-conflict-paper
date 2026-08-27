#!/usr/bin/env python3
r"""Mechanically emit ``paper/values_generated.tex`` from the canonical results.

Every headline number in the manuscript enters prose as a macro defined here;
nothing is hand-transcribed. This script derives each macro from the files under
``results/`` (plus the pure-data price paths in
:mod:`uk_iran_conflict.scenarios`, which need no microdata) and writes them as
LaTeX ``\newcommand``\ s. It never edits the prose.

Usage
-----
``python analysis/emit_tex_values.py``            -> ``paper/values_generated.tex``
``python analysis/emit_tex_values.py --draft``    -> ``\GENMISSING`` renders as a
red ``[?]`` instead of erroring, so an incomplete tree still compiles.

Safety design
-------------
* :func:`emit` never raises. A missing file or key produces
  ``\newcommand{\x}{\GENMISSING} % MISSING: <source> -> <error>`` and is listed
  on stdout at the end, so a stale tree fails **visibly** at LaTeX build time
  rather than silently keeping old numbers.
* First definition wins: a repeated macro name is skipped, because
  ``\newcommand`` errors on redefinition.

The real schema
---------------
``results/<scenario>/shock.json`` (dataclass ``incidence.ScenarioResult`` plus
one appended key)::

    scenario, aggregate_cost_bn, mean_loss_gbp, mean_loss_pct,
    gas_share_of_loss, electricity_share_of_loss, motor_fuel_share_of_loss,
    decile[10]       {decile, mean_loss_gbp, mean_loss_pct,
                      share_of_total_loss, households_m}
    intra_decile[10] {decile, p10_loss_pct, p50_loss_pct, p90_loss_pct,
                      share_above_5pct, share_above_10pct}
    region[12]       {name, mean_loss_gbp, mean_loss_pct, households_m}
    gini_baseline, gini_after, poverty_bhc_baseline, poverty_ahc_baseline,
    means_tested_share

``results/sensitivity/policy_envelope.csv`` -- up to **five** rows per policy,
keyed by ``envelope``: ``stated``, ``feasible_max``, ``common_capped``,
``common_scaled`` and ``common_eligibility``. Carries ``label_used, parameter,
parameter_units, stated_parameter, implied_parameter, feasible_max_parameter,
is_feasible, absorbable_envelope_bn, eligible_share``.

The round-3 referees flagged the ``common_capped`` label specifically, and the
correction is a distinction this emitter now enforces in every macro name:

``feasible_max``   the instrument at its OWN ceiling, **uncapped by any
                   envelope**. This is the true feasible maximum and the only
                   row down which feasible maxima are comparable. The JRF block
                   reaches £21.86bn here; nothing constrains it to £5bn.
``common_capped``  **envelope absorption**: ``min(envelope, feasible-max
                   cost)``. It answers "how much of the £5bn can this
                   instrument absorb?", which for an instrument that already
                   costs more than the envelope scales generosity *down*. It is
                   emphatically **not** a feasible maximum, and the pre-round-3
                   version of this file said it was.

The two coincide only for an instrument that saturates below the envelope
(``feasible_max_cost_bn < envelope_bn``), which is why the mislabel survived
three of the five instruments and broke on the other two.

``results/<scenario>/<policy>.json`` (dataclass ``policies.PolicyScore``)::

    policy, label, cost_bn, stated_cost_bn, share_to_bottom_three,
    cost_per_pound_decile_one, mean_gain_gbp, uncompensated_share_overall,
    uncompensated_by_decile{"1".."10"}, net_loss_after_policy_gbp,
    fully_compensated_share

``results/<scenario>/aggregates.json``::

    aggregate_energy_spend_bn, aggregate_fuel_spend_bn

Note on ``cost_per_pound_decile_one``: the rebuilt scorecard stores it
**dimensionless** — total exchequer cost divided by the aggregate gain reaching
decile one, so it is >= 1 by construction — and records that in a companion
``cost_per_pound_decile_one_units`` column. It is read straight through and must
never be rendered with a pound sign; the old reconstruction from the mean gain
and the decile-one household count is gone.

Post-rebuild additions read here
-------------------------------
* ``dispersion`` on every ``shock.json``: the within-decile p90-p10 range, its
  mean, that mean excluding decile one, and a median-based measure, against the
  between-decile range. The three disagree, and the prose now says so.
* ``decile_domestic_only`` and ``domestic_only_d1_d10_ratio_*``: the gradient of
  the domestic leg alone, which is invariant across specifications in a way the
  all-channel gradient is not.
* ``decile_concept``: how far ranking on unequivalised BHC income agrees with
  ranking on the equivalised AHC income the burden is measured against.
* ``results/robustness/cash_profiles.json``: the full ten-decile cash profile
  under each calibration.
* ``results/grid/reconciliation.json``: every named scenario against the grid's
  own range of the decile ratio.

Sensitivity sources
-------------------
``results/sensitivity/elasticity.csv``  (14 rows: five named specs then the flat
grid) -- ``spec, epsilon_mean, mean_loss_gbp, aggregate_loss_bn,
share_of_upper_bound_shaved, decile1_loss_pct, decile10_loss_pct, ...``

``results/sensitivity/cap_lag.csv``  (10 rows: five lags x two anchoring rules)
-- ``lag_quarters, anchor, mean_loss_gbp, aggregate_loss_bn, domestic_loss_bn,
motor_fuel_loss_bn, annual_phase_in_gas, annual_phase_in_electricity,
is_central_specification, ...``. ``lag_quarters`` is a float
(the central lag is 1.5) and there are no cumulative columns: the cumulative
burden is reconstructed by undoing the phase-in leg by leg, and is invariant
along the ``unanchored`` series.

``results/sensitivity/asymmetry.csv``  (3 rows: 0.70 / 0.85 paper / 1.00) --
``marginal_pricing_share, mean_loss_gbp, decile_ratio_pct,
gas_share_of_domestic_loss, ...``

Values not yet in ``results/``
-----------------------------
Three prose numbers are computed inside ``analysis/run_incidence.py`` but not
persisted, and this emitter must run in CI without the private microdata token,
so they are carried here as documented constants with a TODO naming the fix.
See :data:`_PENDING_PERSIST`.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "values_generated.tex"

#: The observed shock is the paper's headline. The two NIESR paths are the
#: forward-looking bounds around it (methodology.tex, "Scenarios").
CENTRAL_SCENARIO = "realised_2026"

#: The undamped-pump run, retained as an explicit **upper bound** on the
#: motor-fuel channel: peak pump prices charged for a full twelve months. The
#: main specification damps them on the same logic as the wholesale gas peak
#: (``docs/VALIDATION.md`` Check 2b), so every headline in the paper is a range
#: whose top end comes from here.
PEAK_FUEL_SCENARIO = "realised_2026_peak_fuel"

#: The main specification re-scored on ONS-calibrated motor-fuel decile shares
#: (``docs/VALIDATION.md`` Check 2d). Robustness only; the national motor-fuel
#: total is preserved and only the decile profile is replaced.
ONS_FUEL_DIR = "robustness/ons_fuel"

SCENARIOS = ("niesr_baseline", "niesr_adverse", "realised_2026")

#: The seven specifications the referee response requires the paper to report
#: side by side. ``stem`` is the macro infix (no digits — LaTeX forbids them in
#: a control sequence), ``rel`` the ``shock.json`` under ``results/``, and
#: ``variant`` the matching row key in ``results/robustness/comparison.csv``.
#:
#: The first entry is the main specification (equivalised AHC denominator, D1,
#: plus the Step 1 consumption-weighted quarterly cap average, D2). Everything
#: else is an alternative or a robustness line, never a headline.
SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("Main", "realised_2026", "main", "Main (equivalised, phase-in)"),
    ("SteadyState", "robustness/steady_state", "steady_state", "Steady state"),
    (
        "SymDamp",
        "robustness/symmetric_damping",
        "symmetric_damping",
        "Symmetric damping",
    ),
    (
        "PeakFuel",
        "realised_2026_peak_fuel",
        "peak_fuel",
        "Peak fuel (upper bound)",
    ),
    ("OnsShape", "robustness/ons_fuel", "ons_shape", "ONS motor-fuel shape"),
    ("OnsLevels", "robustness/ons_levels", "ons_both_levels", "ONS both levels"),
    (
        "Unequiv",
        "robustness/unequivalised",
        "unequivalised",
        "Unequivalised (robustness)",
    ),
)

#: Specifications added in round 3 that are **not** members of the range the
#: paper quotes over ``SPECS``.
#:
#: They are deliberately separate. ``SPECS`` varies the accounting choices at a
#: fixed twelve-month window from the shock, so a min/max over it is a range of
#: the same quantity. These four vary something else: two change the annualising
#: *window* (calendar 2026, to meet the Resolution Foundation on its own
#: ground), and two change the motor-fuel *margin* attributed to means-tested
#: households. Folding them into the same min/max would silently widen the
#: paper's headline range with quantities that are not comparable to it, which
#: is a version of the mistake round 2 caught. Every one of them still gets a
#: full macro block of its own.
EXTRA_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Calendar",
        "robustness/calendar_2026",
        "calendar_2026",
        "Calendar 2026 (both legs)",
    ),
    (
        "PeakFuelCalendar",
        "robustness/peak_fuel_calendar_2026",
        "peak_fuel_calendar_2026",
        "Calendar 2026, peak fuel",
    ),
    (
        "MtFuelParity",
        "robustness/mt_fuel_parity",
        "mt_fuel_parity",
        "Means-tested fuel parity",
    ),
    (
        "NtsParticipation",
        "robustness/nts_participation",
        "nts_participation",
        "NTS participation margin",
    ),
)

#: Policy key -> the macro stem the prose uses. The prose names are shorter
#: than the file names (\genJrfCostBn, not \genJrfBlockCostBn).
POLICY_MACRO = {
    "social_tariff": "SocialTariff",
    "jrf_block": "Jrf",
    "whd_expansion": "Whd",
    "vat_zero": "VatCut",
    "ippr_rebate": "Rebate",
}

#: Small-integer -> English word. LaTeX macro names cannot contain digits and
#: the prose reads better with the word anyway.
NUMBER_WORD = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    # The comparison table passed ten specifications in round 3, which indexed
    # this tuple out of range and turned \genSpecificationCount into a
    # \GENMISSING. Counting words are cheap; running out of them is not.
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)


def number_word(n: int) -> str:
    """``NUMBER_WORD[n]`` where that exists, else the digits.

    A count that outgrows the table degrades to "23" rather than to a
    ``\\GENMISSING``: the sentence still reads, and nothing is silently wrong.
    """
    n = int(n)
    return NUMBER_WORD[n] if 0 <= n < len(NUMBER_WORD) else str(n)


#: Decile index (0-based) -> the macro infix the paper already uses.
DECILE_WORD = (
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
)

lines: list[str] = []
missing: list[str] = []

#: Macro names already written; a repeat is skipped rather than emitted twice.
_emitted: set[str] = set()


def emit(name: str, value, fmt: str = "{:.1f}", source: str = "") -> None:
    r"""Emit one macro; on any failure emit a loud ``\GENMISSING`` placeholder."""
    if name in _emitted:
        return
    _emitted.add(name)
    try:
        if callable(value):
            value = value()
        if value is None:
            raise ValueError("value is None")
        text = value if isinstance(value, str) else fmt.format(value)
        if text.strip() in {"", "nan", "inf", "-inf"}:
            raise ValueError(f"non-finite value {text!r}")
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    except Exception as exc:  # noqa: BLE001 — every miss must surface, not abort
        missing.append(f"{name} ({source or 'unknown source'}: {exc})")
        lines.append(
            f"\\newcommand{{\\{name}}}{{\\GENMISSING}} % MISSING: {source} -> {exc}"
        )


def note(text: str) -> None:
    """Write a comment line into the generated file (never a macro)."""
    lines.append(f"% {text}")


#: The three sweep CSVs under ``results/sensitivity/`` and ``results/grid/`` were
#: produced **before** the central scenario was re-specified, so every macro
#: derived from them describes the peak-fuel upper bound (undamped pump prices),
#: not the damped main specification. They cannot be recomputed here — the
#: sweeps need the private microdata and are written by
#: ``analysis/run_sensitivity.py`` / ``analysis/run_grid.py``. The macros are
#: emitted with a loud comment rather than suppressed, so the prose keeps
#: compiling while the provenance stays visible in the generated file.
STALE_SWEEP_WARNING = (
    "WARNING: the block below comes from a sweep CSV generated on the "
    "PEAK-FUEL (undamped pump) specification, not the damped main "
    "specification. Re-run analysis/run_sensitivity.py and analysis/run_grid.py "
    "to refresh, then re-run this emitter."
)


def sweep_is_stale() -> bool:
    """True if ``elasticity.csv``'s zero-elasticity row is not the main spec.

    The zero-elasticity row *is* the main specification by construction, so if
    its aggregate does not match ``results/realised_2026/shock.json`` the sweeps
    predate the re-specification.
    """
    try:
        zero = snum(srow("elasticity.csv", "spec", "flat_0.0"), "aggregate_loss_bn")
        main_agg = float(jload(f"{CENTRAL_SCENARIO}/shock.json")["aggregate_cost_bn"])
        return abs(zero - main_agg) > 0.05
    except Exception:  # noqa: BLE001 — a missing sweep is reported by emit()
        return False


def scen_module():
    """The pure-data scenarios module, imported lazily.

    It needs no microdata, so it is safe to read in CI; importing it at module
    scope would still make this emitter depend on the package being installed,
    which it deliberately does not.
    """
    from uk_iran_conflict import scenarios as scen  # noqa: PLC0415

    return scen


def jload(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def decile_row(shock: dict, block: str, decile: int) -> dict:
    """One row of ``decile`` or ``intra_decile``, keyed by decile number."""
    for row in shock[block]:
        if int(row["decile"]) == decile:
            return row
    raise KeyError(f"{block} has no decile {decile}")


def decile_one_households_m(shock: dict) -> float:
    return float(decile_row(shock, "decile", 1)["households_m"])


def cost_per_pound_decile_one(policy: dict, shock: dict | None = None) -> float:
    """Total exchequer cost per £1 of gain reaching decile one.

    The rebuilt scorecard stores this **already dimensionless** — total cost
    divided by the aggregate gain accruing to decile one, so it is >= 1 by
    construction and carries no pound sign. ``cost_per_pound_decile_one_units``
    records that in the file itself. The old reconstruction from the mean gain
    and the decile-one household count is gone: it divided by the wrong thing
    against the new schema. ``shock`` is retained only so existing call sites
    keep working.
    """
    value = float(policy["cost_per_pound_decile_one"])
    if not (value == value) or value <= 0:  # NaN or non-positive
        raise ValueError(f"no usable decile-one cost ratio: {value!r}")
    return value


#: The units string the scorecard writes beside ``cost_per_pound_decile_one``.
#: Asserted rather than assumed, so a future re-dimensioning back to pounds is
#: caught here instead of silently reintroducing a "\pounds" in the prose.
COST_PER_POUND_IS_DIMENSIONLESS = "dimensionless"


# ---------------------------------------------------------------------------
# sensitivity CSV readers
# ---------------------------------------------------------------------------

SENS = R / "sensitivity"


def sload(name: str) -> list[dict]:
    """Rows of one ``results/sensitivity/*.csv`` as dicts of strings."""
    with (SENS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def srow(name: str, key: str, value: str) -> dict:
    """The single row of ``name`` whose column ``key`` equals ``value``."""
    for row in sload(name):
        if row[key].strip() == value:
            return row
    raise KeyError(f"{name}: no row with {key}={value}")


def snum(row: dict, column: str) -> float:
    return float(row[column])


def cload(rel: str) -> list[dict]:
    """Rows of any CSV under ``results/`` as dicts of strings."""
    with (R / rel).open(newline="") as fh:
        return list(csv.DictReader(fh))


def crow(rel: str, key: str, value: str) -> dict:
    for row in cload(rel):
        if row[key].strip() == value:
            return row
    raise KeyError(f"{rel}: no row with {key}={value}")


COMPARISON = "robustness/comparison.csv"
ENVELOPE = "sensitivity/policy_envelope.csv"
DOMESTIC_LEG = "sensitivity/domestic_leg.csv"
FUEL_BY_DECILE = "sensitivity/fuel_by_decile.csv"


def comparison_rows() -> dict[str, dict]:
    return {r["variant"]: r for r in cload(COMPARISON)}


#: The rebuilt scorecard carries up to **five** rows per policy, distinguished
#: by the ``envelope`` column. The distinction is load-bearing rather than
#: presentational: VAT zero-rating cannot physically absorb the common envelope
#: (removing 5 VAT points costs £2.11bn and there is no sixth point to remove),
#: and two instruments are infeasible once scaled proportionally.
#:
#: ``stated``             the sponsor's own design, at the sponsor's own cost
#: ``feasible_max``       the instrument at its OWN ceiling, uncapped by any
#:                        envelope. The true feasible maximum.
#: ``common_capped``      envelope absorption: ``min(envelope, feasible-max
#:                        cost)``. How much of the envelope the instrument can
#:                        absorb — **not** a feasible maximum.
#: ``common_scaled``      the old proportional scaling to the full envelope.
#: ``common_eligibility`` the envelope spent by widening *eligibility* at the
#:                        sponsor's own generosity, where that is defined.
ENVELOPE_STATED = "stated"
ENVELOPE_FEASIBLE_MAX = "feasible_max"
ENVELOPE_CAPPED = "common_capped"
ENVELOPE_SCALED = "common_scaled"
ENVELOPE_ELIGIBILITY = "common_eligibility"

#: Macro infix per row type.
#:
#: ``Envelope`` and ``Capped`` (no infix qualifier) are kept as names of the
#: **envelope-absorption** row, because that is the row the prose means when it
#: says "at a common envelope" and both names are already in the manuscript.
#: ``Absorbed`` is the new, self-describing alias, and ``FeasibleMax`` is the
#: separate row it used to be confused with.
ENVELOPE_ROW_TAGS = (
    (ENVELOPE_FEASIBLE_MAX, "FeasibleMax"),
    (ENVELOPE_CAPPED, "Capped"),
    (ENVELOPE_SCALED, "Scaled"),
    (ENVELOPE_ELIGIBILITY, "Eligibility"),
    (ENVELOPE_STATED, "Stated"),
)

#: Extra macro aliases per row type, beyond the tag itself.
ENVELOPE_ROW_ALIASES = {
    #: "at a common envelope" in the manuscript means envelope absorption.
    ENVELOPE_CAPPED: ("Envelope", "Absorbed"),
    ENVELOPE_FEASIBLE_MAX: ("AtFeasibleMax",),
}

#: One-line semantics per row type, emitted so the prose can name what a row
#: means without a hand-written gloss that can drift from the code.
ENVELOPE_ROW_SEMANTICS = {
    ENVELOPE_STATED: "the sponsor's own parameter, at whatever it costs",
    ENVELOPE_FEASIBLE_MAX: (
        "the instrument at its own ceiling, uncapped by any envelope"
    ),
    ENVELOPE_CAPPED: (
        "envelope absorption: the smaller of the envelope and the feasible-maximum cost"
    ),
    ENVELOPE_SCALED: "proportional rescaling to the envelope, feasible or not",
    ENVELOPE_ELIGIBILITY: (
        "the sponsor's own generosity, with eligibility widened until the "
        "envelope is spent"
    ),
}


def envelope_row(policy: str, envelope: str) -> dict:
    """One row of the common-envelope scorecard, by policy and row type."""
    for row in cload(ENVELOPE):
        if row["policy"].strip() == policy and row["envelope"].strip() == envelope:
            return row
    raise KeyError(f"{ENVELOPE}: no {policy}/{envelope} row")


def envelope_has(policy: str, envelope: str) -> bool:
    r"""True if this policy/row-type combination exists in the rebuilt file.

    Not every instrument has every row: only the two means-tested schemes have
    an eligibility-widening variant, and where scaling is already feasible the
    scaled row coincides with the capped one. Macros are emitted only for rows
    that exist, so an absent row is a silently-narrower macro set rather than a
    ``\GENMISSING`` the prose would then have to work around.
    """
    try:
        envelope_row(policy, envelope)
    except Exception:  # noqa: BLE001 — absence is the answer, not an error
        return False
    return True


def envelope_rows(envelope: str) -> list[dict]:
    """Every policy's row of one type."""
    return [r for r in cload(ENVELOPE) if r["envelope"].strip() == envelope]


def is_true(text: str) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes"}


def parameter_text(value: float) -> str:
    """Render an instrument parameter at a readable precision.

    The parameters span three orders of magnitude (5 VAT points, a 35 per cent
    discount, a £1,083 payment), so a single format string either loses the
    12.74 that makes the VAT scaling absurd or writes "35.0" for a round
    number. Large values get thousands separators and no decimal; small ones
    keep one decimal, trimmed when it is zero.
    """
    number = float(value)
    if abs(number) >= 100:
        return f"{number:,.0f}"
    return f"{number:.1f}".removesuffix(".0")


def parameter_suffix(row: dict) -> str:
    """``Gbp`` or ``Pct``, from the row's own ``parameter_units`` string.

    The instrument parameters are not commensurable — a bill discount in per
    cent, a payment in pounds, VAT points — so the macro name has to say which,
    and it is read off the file rather than hardcoded per policy.
    """
    units = row.get("parameter_units", "").strip()
    return "Gbp" if units.startswith("£") else "Pct"


def is_identified(row: dict) -> bool:
    """True if a sweep row produced numbers at all.

    Round 3 made non-identification a real outcome rather than an error. Two
    cells of the ``gas_peak_monthly_profile`` sweep, and three of the anchored
    ``cap_lag`` series, cannot be solved: the two cap observations do not pin a
    pre-war counterfactual, either because the later observation window prices
    no more of the shock than the earlier one, or because the implied sustained
    fraction falls outside ``(0, 1]``. Those rows are written with every
    numeric column empty and a ``note`` saying why.

    They are a finding, not a defect, so they are counted and reported rather
    than dropped silently -- but they must be excluded from any min/max, or a
    ``float("")`` takes the whole macro down with it. An ``identified`` column
    that is present and explicitly false excludes the row; an absent or empty
    one falls back to asking whether the row actually carries an aggregate,
    because the sweeps that cannot fail do not write the column at all.
    """
    flag = str(row.get("identified", "")).strip().lower()
    if flag in {"false", "0", "no"}:
        return False
    if flag in {"true", "1", "yes"}:
        return True
    return str(row.get("aggregate_cost_bn", "")).strip() != ""


def leg_rows(parameter: str, identified_only: bool = True) -> list[dict]:
    rows = [r for r in cload(DOMESTIC_LEG) if r["parameter"].strip() == parameter]
    if identified_only:
        rows = [r for r in rows if is_identified(r)]
    if not rows:
        raise KeyError(f"{DOMESTIC_LEG}: no usable rows for parameter {parameter!r}")
    return rows


def leg_span(parameter: str, column: str, select) -> float:
    return select(float(r[column]) for r in leg_rows(parameter))


def leg_all(identified_only: bool = True) -> list[dict]:
    """Every ``domestic_leg.csv`` row, by default only the identified ones."""
    rows = cload(DOMESTIC_LEG)
    return [r for r in rows if is_identified(r)] if identified_only else rows


def fuel_decile(decile: int) -> dict:
    return crow(FUEL_BY_DECILE, "decile", str(decile))


#: The paper's calibrated lag, mirrored from
#: ``uk_iran_conflict.scenarios.CAP_LAG_QUARTERS`` but written literally so this
#: emitter stays importable without the package (CI runs it with no microdata).
#: The rebuilt sweep re-anchors the cap at 1.5 quarters, not 3.
PAPER_CAP_LAG = 1.5

#: ``cap_lag.csv`` is now a *two-way* sweep: each lag appears once ``anchored``
#: (the sustained fraction is re-solved so the Cornwall cap anchor still binds)
#: and once ``unanchored`` (the sustained fraction is held at the central value).
#: The paper's specification is the anchored central row; the unanchored series
#: is the one over which the cumulative burden is invariant, because nothing
#: about the shock's size changes along it.
CAP_LAG_ANCHOR = "anchored"


def lag_rows(anchor: str | None = CAP_LAG_ANCHOR) -> list[dict]:
    """``cap_lag.csv`` rows, optionally restricted to one anchoring rule."""
    rows = sload("cap_lag.csv")
    if anchor is not None:
        rows = [r for r in rows if r["anchor"].strip() == anchor]
    # Three anchored lags are not identified at all (see :func:`is_identified`):
    # at a one-quarter lag the observation windows do not separate the two
    # published caps, and at three and four quarters the implied pre-war
    # counterfactual is not strictly below the observed July cap. Their rows are
    # blank, so they are excluded here and counted separately.
    rows = [r for r in rows if is_identified(r)]
    if not rows:
        raise KeyError(f"cap_lag.csv: no identified rows with anchor={anchor!r}")
    return rows


def lag_row(quarters: float, anchor: str | None = CAP_LAG_ANCHOR) -> dict:
    """The single ``cap_lag.csv`` row for one lag.

    ``lag_quarters`` is written as a float (``"1.0"``, ``"1.5"``) in the rebuilt
    sweep, so it is matched numerically rather than as a string — the old
    ``lag_quarters == "3"`` comparison silently matched nothing.
    """
    for row in sload("cap_lag.csv"):
        if anchor is not None and row["anchor"].strip() != anchor:
            continue
        if abs(float(row["lag_quarters"]) - quarters) < 1e-9:
            if not is_identified(row):
                # The row exists and is deliberately blank. Surface the sweep's
                # own explanation rather than a bare float("") failure, so the
                # MISSING comment in values_generated.tex says why this lag has
                # no anchored answer instead of looking like a broken read.
                raise ValueError(
                    f"lag {quarters:g} is not identified under the {anchor} "
                    f"rule: {row.get('note', '').strip() or 'no note given'}"
                )
            return row
    raise KeyError(f"cap_lag.csv: no {anchor} row with lag_quarters={quarters}")


def lag_identified(quarters: float, anchor: str | None = CAP_LAG_ANCHOR) -> bool:
    """True if this lag has an identified row under this anchoring rule."""
    try:
        lag_row(quarters, anchor)
    except Exception:  # noqa: BLE001 — non-identification is the answer
        return False
    return True


def lag_central() -> dict:
    """The row the sweep itself flags as the paper's central specification."""
    for row in sload("cap_lag.csv"):
        if row["is_central_specification"].strip().lower() == "true":
            return row
    raise KeyError("cap_lag.csv: no row flagged is_central_specification")


def lag_cumulative_bn(row: dict) -> float:
    """Cumulative (whole-shock) loss implied by one annualised cap-lag row.

    The annualised figure is the cumulative one multiplied by the share of full
    pass-through realised inside the modelled window, leg by leg. Undoing that
    per leg — gas and electricity have different phase-in fractions, motor fuel
    has none — recovers a quantity that is invariant along the *unanchored*
    series to within a thousandth of a billion, which is the identity the
    appendix asserts. It is not invariant along the anchored series, and should
    not be: re-anchoring changes the size of the shock, not just its timing.
    """
    shock = jload(f"{CENTRAL_SCENARIO}/shock.json")
    gas = float(shock["gas_share_of_loss"])
    elec = float(shock["electricity_share_of_loss"])
    if gas + elec <= 0:
        raise ValueError("no domestic leg in the central scenario")
    gas, elec = gas / (gas + elec), elec / (gas + elec)
    domestic = float(row["domestic_loss_bn"])
    phase_gas = float(row["annual_phase_in_gas"])
    phase_elec = float(row["annual_phase_in_electricity"])
    return domestic * (gas / phase_gas + elec / phase_elec) + float(
        row["motor_fuel_loss_bn"]
    )


#: The marginal-pricing sweep endpoints and the paper's central value.
ASYM_LOW, ASYM_CENTRAL, ASYM_HIGH = "0.7", "0.85", "1.0"

# ---------------------------------------------------------------------------
# persisted prose values (docs/FIXES.md C12)
# ---------------------------------------------------------------------------
#
# These were previously hardcoded literals in this file, which contradicted the
# appendix's guarantee that every number is emitted mechanically. They now come
# from ``results/persisted_values.json``, written by the incidence run, so a
# missing or stale tree produces ``\GENMISSING`` like everything else.


def persisted() -> dict:
    return jload("persisted_values.json")


def audit() -> dict:
    return jload("means_tested_audit.json")


def diagnostics() -> dict:
    """``results/policy_diagnostics.json``, written by ``run_incidence.py``.

    Everything about the policy block that is not a per-row scorecard field:
    the true feasible maxima uncapped by the envelope, what each arm of the
    common-envelope comparison actually spends, the JRF reference quantities on
    both bases, and the large-loser statistic with its ceiling.
    """
    return jload("policy_diagnostics.json")


#: No carried constants remain.
#:
#: ``genLargeLoserOutsideMeansTest`` was the last one: 98 per cent, hardcoded
#: here because the statistic lived only in the figure cache
#: (``analysis/figures.py``, ``heavy_burden_gt5pct``) and not in the results
#: tree. ``policies.large_loser_outside_means_test`` now computes it from the
#: baseline and the shock, ``run_incidence.py`` persists it in
#: ``results/policy_diagnostics.json``, and it is read from there like every
#: other number. The appendix's claim that every figure in the paper is emitted
#: mechanically is, as of round 3, true.
_CARRIED: dict[str, tuple[float, str]] = {}

# ---------------------------------------------------------------------------
# ONS benchmarks — the real published values, not our own rescaled output
# ---------------------------------------------------------------------------
#
# docs/FIXES.md C10: the paper attributed £521/£2,230 to ONS Family Spending.
# Those are *our model's* motor-fuel decile means after the ONS-shape rescaling,
# which preserves the microdata's national total. The published ONS numbers are
# £318 (decile 1) and £1,362 (decile 10), with an all-household mean of £960 for
# motor fuel and £1,780 for domestic energy. Both sets are emitted, under names
# that say which is which.
ONS_BENCH_FUEL = 960.0
ONS_BENCH_DOMESTIC = 1780.0

ONS_BENCH = {
    "genOnsBenchFuelDecileOneGbp": (318.0, "{:,.0f}"),
    "genOnsBenchFuelDecileTenGbp": (1362.0, "{:,.0f}"),
    "genOnsBenchFuelMeanGbp": (ONS_BENCH_FUEL, "{:,.0f}"),
    "genOnsBenchDomesticMeanGbp": (ONS_BENCH_DOMESTIC, "{:,.0f}"),
    "genOnsBenchFuelRatio": (1362.0 / 318.0, "{:.1f}"),
}


def _variant_macros(rel: str, stem: str) -> None:
    """Emit the standard headline block for one alternative specification.

    ``rel`` is a ``shock.json`` under ``results/``; ``stem`` is the macro
    infix (``PeakFuel``, ``OnsFuel``). Digits are never used in a macro name —
    LaTeX forbids them — so deciles stay spelled out as One and Ten.
    """
    emit(f"gen{stem}AggBn", lambda: jload(rel)["aggregate_cost_bn"], "{:.1f}", rel)
    emit(f"gen{stem}LossMean", lambda: jload(rel)["mean_loss_gbp"], "{:.0f}", rel)
    emit(f"gen{stem}LossPctIncome", lambda: jload(rel)["mean_loss_pct"], "{:.2f}", rel)
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"gen{stem}Decile{tag}LossGbp",
            lambda d=d: decile_row(jload(rel), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{stem}Decile{tag}LossPct",
            lambda d=d: decile_row(jload(rel), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            rel,
        )
    emit(
        f"gen{stem}DecileRatioPct",
        lambda: (
            decile_row(jload(rel), "decile", 1)["mean_loss_pct"]
            / decile_row(jload(rel), "decile", 10)["mean_loss_pct"]
        ),
        "{:.1f}",
        rel,
    )
    emit(
        f"gen{stem}DecileRatioPrecisePct",
        lambda: (
            decile_row(jload(rel), "decile", 1)["mean_loss_pct"]
            / decile_row(jload(rel), "decile", 10)["mean_loss_pct"]
        ),
        "{:.2f}",
        rel,
    )
    # The three gradient companions, on every specification rather than only on
    # the main one: measured from decile two, and both measured again on a
    # winsorised burden. A specification-dependent headline gradient with a
    # specification-invariant companion is a much more informative pair than
    # either alone.
    for name, key in (
        ("DecileRatioFromDecileTwoPct", "d2_d10_ratio_pct"),
        ("DecileRatioWinsorisedPct", "d1_d10_ratio_pct_winsorised"),
        ("DecileRatioFromDecileTwoWinsorisedPct", "d2_d10_ratio_pct_winsorised"),
    ):
        emit(f"gen{stem}{name}", lambda key=key: jload(rel)[key], "{:.2f}", rel)
    for label, key in (
        ("MotorFuel", "motor_fuel_share_of_loss"),
        ("Gas", "gas_share_of_loss"),
        ("Elec", "electricity_share_of_loss"),
    ):
        emit(
            f"gen{stem}{label}ShareOfLoss",
            lambda key=key: 100 * jload(rel)[key],
            "{:.1f}",
            rel,
        )
    emit(
        f"gen{stem}DomesticShareOfLoss",
        lambda: (
            100
            * (
                jload(rel)["gas_share_of_loss"]
                + jload(rel)["electricity_share_of_loss"]
            )
        ),
        "{:.1f}",
        rel,
    )


def main(draft: bool = False) -> None:
    central = f"{CENTRAL_SCENARIO}/shock.json"
    stale = sweep_is_stale()

    # ------------------------------------------------------------------
    # headline loss, central (realised) scenario
    # ------------------------------------------------------------------
    emit(
        "genCentralLossMean", lambda: jload(central)["mean_loss_gbp"], "{:.0f}", central
    )
    emit(
        "genCentralLossPctIncome",
        lambda: jload(central)["mean_loss_pct"],
        "{:.2f}",
        central,
    )

    # decile contrast: £ rises across deciles, % falls. The paper's core point.
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"genDecile{tag}LossGbp",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )
        emit(
            f"genDecile{tag}LossPct",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            central,
        )
    emit(
        "genDecileRatioPct",
        lambda: (
            decile_row(jload(central), "decile", 1)["mean_loss_pct"]
            / decile_row(jload(central), "decile", 10)["mean_loss_pct"]
        ),
        "{:.1f}",
        central,
    )

    # between- vs within-decile dispersion (Cronin, Fullerton & Sexton 2019)
    emit(
        "genBetweenDecileRangePct",
        lambda: abs(
            decile_row(jload(central), "decile", 1)["mean_loss_pct"]
            - decile_row(jload(central), "decile", 10)["mean_loss_pct"]
        ),
        "{:.2f}",
        central,
    )

    # The rebuilt runs carry a ``dispersion`` summary on every result, so the
    # within-decile statistics are read from it rather than recomputed here.
    # There are three of them and they disagree, which is the point: the mean
    # p90-p10 range exceeds the between-decile range, the same mean excluding
    # decile one does not, and a median-based measure is far below it. The
    # Cronin-style claim survives on exactly one of the three.
    def dispersion() -> dict:
        return jload(central)["dispersion"]

    # ROUND 4, item 6: the prose quotes 3.64, which is this macro -- the MEAN
    # p90-p10 range across the ten deciles, decile one included. It sat next to
    # \genWithinDecileRangeMaxExDOnePct (3.81, the largest range outside decile
    # one), which is unused and easy to reach for by mistake because its name is
    # longer and looks more specific. ``Mean`` is emitted as an explicit alias
    # so a sentence can name the statistic it means instead of relying on the
    # unqualified name being the mean.
    for name in ("genWithinDecileRangePct", "genWithinDecileRangeMeanPct"):
        emit(
            name,
            lambda: dispersion()["mean_within_decile_range_pp"],
            "{:.2f}",
            f"{central}:dispersion",
        )
    emit(
        "genWithinDecileRangeMeanExDOnePct",
        lambda: dispersion()["mean_within_decile_range_excl_d1_pp"],
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileRangeMaxPct",
        lambda: max(dispersion()["within_decile_range_by_decile_pp"]),
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileRangeMinPct",
        lambda: min(dispersion()["within_decile_range_by_decile_pp"]),
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileRangeExDOnePct",
        lambda: dispersion()["mean_within_decile_range_excl_d1_pp"],
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileMedianRangePct",
        lambda: dispersion()["median_based_within_pp"],
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileMedianOfRangesPct",
        lambda: dispersion()["median_within_decile_range_pp"],
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genDecileOneRangePct",
        lambda: dispersion()["within_decile_range_by_decile_pp"][0],
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileRangeMinExDOnePct",
        lambda: min(dispersion()["within_decile_range_by_decile_pp"][1:]),
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genWithinDecileRangeMaxExDOnePct",
        lambda: max(dispersion()["within_decile_range_by_decile_pp"][1:]),
        "{:.2f}",
        f"{central}:dispersion",
    )
    emit(
        "genDecilesBelowBetweenRange",
        lambda: number_word(int(dispersion()["deciles_below_between_range"])),
        source=f"{central}:dispersion",
    )
    emit(
        "genWithinExceedsBetween",
        lambda: "does" if dispersion()["within_exceeds_between"] else "does not",
        source=f"{central}:dispersion",
    )
    emit(
        "genWithinExceedsBetweenExDOne",
        lambda: (
            "does" if dispersion()["within_exceeds_between_excl_d1"] else "does not"
        ),
        source=f"{central}:dispersion",
    )
    # The same trio on the dispersion block's own between-decile range, so the
    # comparison in the prose is like for like (the macro above computes it
    # from the decile table; they agree, and this asserts that they do).
    emit(
        "genBetweenDecileRangeDispersionPct",
        lambda: dispersion()["between_decile_range_pp"],
        "{:.2f}",
        f"{central}:dispersion",
    )

    # ------------------------------------------------------------------
    # the domestic-only gradient: the ratio that does not move
    # ------------------------------------------------------------------
    #
    # The all-channel decile ratio moves with the gas/pump mix, because motor
    # fuel is the channel the mix moves. The *domestic-only* ratio does not: it
    # is 9.31 in every named scenario and every specification, to three
    # significant figures, because the domestic leg is a common scaling of one
    # consumption vector. That invariance is the finding, not the level.
    emit(
        "genDomesticOnlyDecileRatioPct",
        lambda: jload(central)["domestic_only_d1_d10_ratio_pct"],
        "{:.2f}",
        central,
    )
    emit(
        "genDomesticOnlyDecileRatioGbp",
        lambda: jload(central)["domestic_only_d1_d10_ratio_gbp"],
        "{:.2f}",
        central,
    )
    emit(
        "genAllChannelDecileRatioPct",
        lambda: jload(central)["all_channel_d1_d10_ratio_pct"],
        "{:.2f}",
        central,
    )
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"genDomesticOnlyDecile{tag}LossGbp",
            lambda d=d: decile_row(jload(central), "decile_domestic_only", d)[
                "mean_loss_gbp"
            ],
            "{:.0f}",
            central,
        )
        emit(
            f"genDomesticOnlyDecile{tag}LossPct",
            lambda d=d: decile_row(jload(central), "decile_domestic_only", d)[
                "mean_loss_pct"
            ],
            "{:.2f}",
            central,
        )

    def _domestic_only_span(select):
        """Span of the domestic-only ratio across the equivalised specifications.

        The unequivalised run is excluded: it re-ranks households on a different
        income concept, so its ratio is a different object rather than a
        sensitivity of the same one. It is emitted separately below, which is
        the honest way to show that the invariance is invariance *given the
        denominator*, not invariance to the denominator.
        """
        values = [
            float(r["domestic_only_d1_d10_ratio_pct"])
            for r in cload(COMPARISON)
            if r["domestic_only_d1_d10_ratio_pct"]
            and r["variant"].strip() != "unequivalised"
        ]
        if not values:
            raise ValueError("comparison.csv carries no domestic-only ratio")
        return select(values)

    emit(
        "genDomesticOnlyRatioMin",
        lambda: _domestic_only_span(min),
        "{:.2f}",
        COMPARISON,
    )
    emit(
        "genDomesticOnlyRatioMax",
        lambda: _domestic_only_span(max),
        "{:.2f}",
        COMPARISON,
    )
    emit(
        "genDomesticOnlyRatioSpread",
        lambda: _domestic_only_span(max) - _domestic_only_span(min),
        "{:.3f}",
        COMPARISON,
    )
    emit(
        "genDomesticOnlyRatioUnequiv",
        lambda: float(
            crow(COMPARISON, "variant", "unequivalised")[
                "domestic_only_d1_d10_ratio_pct"
            ]
        ),
        "{:.2f}",
        COMPARISON,
    )
    emit(
        "genDomesticOnlyRatioEquivSpecCount",
        lambda: number_word(
            sum(1 for r in cload(COMPARISON) if r["variant"].strip() != "unequivalised")
        ),
        source=COMPARISON,
    )
    emit(
        "genSpecificationCount",
        lambda: number_word(len(cload(COMPARISON))),
        source=COMPARISON,
    )

    # ------------------------------------------------------------------
    # which decile concept: the ranking is not the same object as the burden
    # ------------------------------------------------------------------
    #
    # Ranking households by unequivalised BHC income and by the equivalised AHC
    # income the burden is measured against puts barely half of them in the same
    # decile. The audit reports the agreement rate on each concept; the paper's
    # own denominator is the best-matching one, which is the check that matters.
    def concept() -> dict:
        return jload(central)["decile_concept"]

    # ROUND 4, item 6: these rendered at ``{:.0f}``, so the best-matching
    # concept -- 0.99896 agreement -- printed as "100" beside a 56.1 per cent
    # comparator, which reads as a typo rather than as a number. One decimal
    # place is the least that keeps a near-perfect match distinguishable from a
    # perfect one, and it costs nothing on the other two.
    for name, key in (
        ("genDecileRankAgreementPct", None),
        ("genDecileRankAgreementUnequivPct", "unequivalised_bhc"),
        ("genDecileRankAgreementEquivBhcPct", "equivalised_bhc"),
        ("genDecileRankAgreementEquivAhcPct", "equivalised_ahc"),
    ):
        emit(
            name,
            lambda key=key: (
                100 * concept()["agreement"][key or concept()["best_match"]]
            ),
            "{:.1f}",
            f"{central}:decile_concept",
        )
    emit(
        "genDecileRankMeanGap",
        lambda: concept()["mean_absolute_decile_gap"][concept()["best_match"]],
        "{:.2f}",
        f"{central}:decile_concept",
    )
    emit(
        "genDecileRankMatchesDenominator",
        lambda: "does" if concept()["matches_burden_denominator"] else "does not",
        source=f"{central}:decile_concept",
    )

    # ------------------------------------------------------------------
    # the cash profile under each calibration (the withdrawn non-monotonicity)
    # ------------------------------------------------------------------
    #
    # The paper claimed the non-monotone cash profile — the decile-eight hump —
    # was common to every specification. It is not: both ONS calibrations are
    # strictly monotone and peak at decile ten. Only the raw PolicyEngine
    # imputation humps, and the hump is in the leg the ONS calibration corrects.
    CASH_PROFILES = "robustness/cash_profiles.json"
    CASH_TAGS = (
        ("Raw", "raw"),
        ("OnsShape", "ons_fuel_shape"),
        ("OnsLevels", "ons_both_levels"),
    )

    def profile(key: str) -> dict:
        return jload(CASH_PROFILES)["profiles"][key]

    for tag, key in CASH_TAGS:
        for i, word in enumerate(DECILE_WORD):
            emit(
                f"gen{tag}CashDecile{word}Gbp",
                lambda key=key, i=i: profile(key)["mean_loss_gbp"][i],
                "{:.0f}",
                CASH_PROFILES,
            )
        emit(
            f"gen{tag}CashPeakDecile",
            lambda key=key: DECILE_WORD[int(profile(key)["peak_decile"]) - 1].lower(),
            source=CASH_PROFILES,
        )
        emit(
            f"gen{tag}CashPeakDecileNum",
            lambda key=key: int(profile(key)["peak_decile"]),
            "{:d}",
            CASH_PROFILES,
        )
        emit(
            f"gen{tag}CashMonotone",
            lambda key=key: (
                "is" if profile(key)["is_monotone_increasing"] else "is not"
            ),
            source=CASH_PROFILES,
        )
        emit(
            f"gen{tag}CashMonotoneWord",
            lambda key=key: (
                "monotone" if profile(key)["is_monotone_increasing"] else "non-monotone"
            ),
            source=CASH_PROFILES,
        )
        emit(
            f"gen{tag}CashDecileOneOverTen",
            lambda key=key: profile(key)["decile1_over_decile10_gbp"],
            "{:.2f}",
            CASH_PROFILES,
        )
        emit(
            f"gen{tag}CashFallsAtDecileCount",
            lambda key=key: number_word(len(profile(key)["deciles_where_cash_falls"])),
            source=CASH_PROFILES,
        )
    emit(
        "genCashProfileMonotoneCount",
        lambda: number_word(
            sum(1 for _, key in CASH_TAGS if profile(key)["is_monotone_increasing"])
        ),
        source=CASH_PROFILES,
    )

    # ------------------------------------------------------------------
    # grid reconciliation: every named scenario sits inside the grid's range
    # ------------------------------------------------------------------
    RECON = "grid/reconciliation.json"

    def recon() -> dict:
        return jload(RECON)

    emit(
        "genGridReconRatioMin",
        lambda: recon()["grid_d1_d10_ratio_min"],
        "{:.2f}",
        RECON,
    )
    emit(
        "genGridReconRatioMax",
        lambda: recon()["grid_d1_d10_ratio_max"],
        "{:.2f}",
        RECON,
    )
    # Round-3 finding 5 rewrote this file. The old check was "is every named
    # scenario inside the grid's D1/D10 range, plus or minus 5 per cent?", and
    # it could not fail: the grid's range was so wide that the tolerance band
    # swallowed everything. ``tolerance`` and ``accepted_band`` are gone with
    # it. What replaced them is a real test — the grid's own spread against the
    # spread it would need for the membership check to carry information — and
    # an identity check on the channel mix. Both are emitted; the old two are
    # not resurrected, because a macro that cannot fail is not evidence.
    emit(
        "genGridReconSpread",
        lambda: recon()["grid_d1_d10_ratio_spread"],
        "{:.4f}",
        RECON,
    )
    emit(
        "genGridReconInformativeThreshold",
        lambda: recon()["informative_spread_threshold"],
        "{:.2f}",
        RECON,
    )
    emit(
        "genGridReconShowsInvariance",
        lambda: "does" if recon()["grid_shows_invariance"] else "does not",
        source=RECON,
    )
    emit(
        "genGridReconCheckIsInformative",
        lambda: "is" if recon()["check_is_informative"] else "is not",
        source=RECON,
    )
    emit(
        "genGridReconIdentityHolds",
        lambda: "does" if recon()["channel_mix_identity_holds"] else "does not",
        source=RECON,
    )
    emit(
        "genGridReconIdentityBrokenCount",
        lambda: number_word(len(recon()["identity_broken"])),
        source=RECON,
    )
    emit(
        "genGridReconIdentityTolerance",
        lambda: recon()["identity_tolerance"],
        "{:g}",
        RECON,
    )
    emit(
        "genGridReconScenarioCount",
        lambda: number_word(len(recon()["scenarios"])),
        source=RECON,
    )
    emit(
        "genGridReconAllInside",
        lambda: (
            "every" if recon()["all_named_scenarios_inside_grid_range"] else "not every"
        ),
        source=RECON,
    )

    # Counted here rather than read from the file's own ``outside`` list, which
    # after the round-3 rewrite records identity breaks rather than range
    # misses. The two disagreed — ``all_named_scenarios_inside_grid_range`` is
    # false while ``outside`` is empty — and a prose sentence built on the pair
    # would have contradicted itself. The grid's range is now the *domestic-only*
    # ratio, which is invariant to four decimal places, so the all-channel
    # ratios sit outside it by construction: that is the finding, not a failure.
    def _outside_grid_range() -> list[str]:
        low = recon()["grid_d1_d10_ratio_min"]
        high = recon()["grid_d1_d10_ratio_max"]
        return [
            name
            for name, v in recon()["scenarios"].items()
            if not low <= float(v["d1_d10_ratio"]) <= high
        ]

    emit(
        "genGridReconOutsideCount",
        lambda: number_word(len(_outside_grid_range())),
        source=RECON,
    )
    emit(
        "genGridReconIdentityBreakCount",
        lambda: number_word(len(recon()["outside"])),
        source=RECON,
    )
    emit(
        "genGridReconDomesticOnlyMin",
        lambda: min(
            float(v["d1_d10_ratio_domestic_only"])
            for v in recon()["scenarios"].values()
        ),
        "{:.2f}",
        RECON,
    )

    # ROUND 4. Why the named scenarios sit outside the grid's range is now
    # diagnosed rather than guessed at: petrol and diesel have materially
    # different decile gradients, and every grid cell fixes the petrol:diesel
    # mix, so the grid traces a one-dimensional curve through a four-channel
    # space. The range check is permanently unenforced and the file says so;
    # two checks that CAN fail are enforced and are named here.
    def _sub_channel() -> dict:
        return recon()["sub_channel_gradients"]["d1_d10_ratio_by_channel"]

    for tag, channel in (
        ("Gas", "gas"),
        ("Elec", "electricity"),
        ("Petrol", "petrol"),
        ("Diesel", "diesel"),
    ):
        emit(
            f"genGridSubChannelRatio{tag}",
            lambda channel=channel: _sub_channel()[channel],
            "{:.2f}",
            RECON,
        )
    emit(
        "genGridRangeCheckEnforced",
        lambda: "is" if recon()["grid_scope"]["range_check_enforced"] else "is not",
        source=RECON,
    )
    emit(
        "genGridEnforcedCheckCount",
        lambda: number_word(len(recon()["grid_scope"]["enforced_checks"])),
        source=RECON,
    )
    emit(
        "genGridSubChannelBracketingHolds",
        lambda: "does" if recon()["sub_channel_bracketing_holds"] else "does not",
        source=RECON,
    )
    emit(
        "genGridSubChannelBracketingBrokenCount",
        lambda: number_word(len(recon()["sub_channel_bracketing_broken"])),
        source=RECON,
    )
    emit(
        "genGridLiveCellCount",
        lambda: number_word(recon()["grid_scope"]["live_cells"]),
        source=RECON,
    )
    emit(
        "genGridReconDomesticOnlyMax",
        lambda: max(
            float(v["d1_d10_ratio_domestic_only"])
            for v in recon()["scenarios"].values()
        ),
        "{:.2f}",
        RECON,
    )

    # ------------------------------------------------------------------
    # aggregate additional spend, by scenario
    # ------------------------------------------------------------------
    for tag, scenario in (
        ("Realised", "realised_2026"),
        ("Adverse", "niesr_adverse"),
        ("Baseline", "niesr_baseline"),
    ):
        rel = f"{scenario}/shock.json"
        emit(
            f"genAggSpend{tag}Bn",
            lambda rel=rel: jload(rel)["aggregate_cost_bn"],
            "{:.1f}",
            rel,
        )

    # ------------------------------------------------------------------
    # price path (pure scenario data — no microdata needed)
    # ------------------------------------------------------------------
    def scenario_obj():
        from uk_iran_conflict import scenarios as scen  # noqa: PLC0415

        return scen.SCENARIOS[CENTRAL_SCENARIO]

    src = "uk_iran_conflict.scenarios.SCENARIOS[realised_2026]"
    emit(
        "genGasPeakPence",
        lambda: scenario_obj().gas.change_pence_per_therm,
        "{:.0f}",
        src,
    )
    emit("genOilPeakUsd", lambda: scenario_obj().oil.change_usd_per_bbl, "{:.0f}", src)
    emit("genOilPeakPct", lambda: 100 * scenario_obj().oil.pct_change, "{:.0f}", src)
    emit(
        "genGasUnitRatePct",
        lambda: 100 * scenario_obj().retail_shock.gas_pct_change,
        "{:.1f}",
        src,
    )
    emit(
        "genElecUnitRatePct",
        lambda: 100 * scenario_obj().retail_shock.electricity_pct_change,
        "{:.1f}",
        src,
    )
    emit("genCapAnnualised", lambda: scenario_obj().peak_cap_gbp, "{:.0f}", src)
    emit("genCapBaselineLevel", lambda: scenario_obj().baseline_cap_gbp, "{:.0f}", src)

    # ------------------------------------------------------------------
    # policy scorecard, central scenario
    # ------------------------------------------------------------------
    for policy, tag in POLICY_MACRO.items():
        rel = f"{CENTRAL_SCENARIO}/{policy}.json"
        emit(f"gen{tag}CostBn", lambda rel=rel: jload(rel)["cost_bn"], "{:.1f}", rel)
        emit(
            f"gen{tag}ShareBottomThree",
            lambda rel=rel: 100 * jload(rel)["share_to_bottom_three"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}UncompensatedShare",
            lambda rel=rel: 100 * jload(rel)["uncompensated_share_overall"],
            "{:.0f}",
            rel,
        )

    def cost_per_pound(select) -> float:
        shock = jload(central)
        values = []
        for policy in POLICY_MACRO:
            values.append(
                cost_per_pound_decile_one(
                    jload(f"{CENTRAL_SCENARIO}/{policy}.json"), shock
                )
            )
        return select(values)

    # the same statistic per instrument, so a sentence can name one directly
    # rather than only the best and worst (docs/FIXES.md D18)
    for policy, tag in POLICY_MACRO.items():
        emit(
            f"gen{tag}CostPerPound",
            lambda policy=policy: cost_per_pound_decile_one(
                jload(f"{CENTRAL_SCENARIO}/{policy}.json"), jload(central)
            ),
            "{:.2f}",
            f"{CENTRAL_SCENARIO}/{policy}.json",
        )
    emit(
        "genBestCostPerPound",
        lambda: cost_per_pound(min),
        "{:.2f}",
        f"{CENTRAL_SCENARIO}/<policies>.json",
    )
    emit(
        "genWorstCostPerPound",
        lambda: cost_per_pound(max),
        "{:.2f}",
        f"{CENTRAL_SCENARIO}/<policies>.json",
    )

    # ------------------------------------------------------------------
    # composition of the loss: motor fuel is two thirds of it
    # ------------------------------------------------------------------
    for tag, scenario in (("", CENTRAL_SCENARIO), ("Adverse", "niesr_adverse")):
        rel = f"{scenario}/shock.json"
        for stem, key in (
            ("MotorFuel", "motor_fuel_share_of_loss"),
            ("Gas", "gas_share_of_loss"),
            ("Elec", "electricity_share_of_loss"),
        ):
            emit(
                f"gen{stem}ShareOfLoss{tag}",
                lambda rel=rel, key=key: 100 * jload(rel)[key],
                "{:.1f}",
                rel,
            )
        # every domestic-bill instrument is confined to this share by construction
        emit(
            f"genDomesticShareOfLoss{tag}",
            lambda rel=rel: (
                100
                * (
                    jload(rel)["gas_share_of_loss"]
                    + jload(rel)["electricity_share_of_loss"]
                )
            ),
            "{:.1f}",
            rel,
        )

    # ------------------------------------------------------------------
    # the two alternative specifications the audit requires
    #
    # The paper reports the fuel share and everything that hangs off it as a
    # *range*: main specification (pump peak damped like the gas peak) to
    # peak-fuel upper bound (pump peak charged for a full year). The
    # ONS-calibrated run is a separate axis — same shock, corrected motor-fuel
    # decile profile — and is reported alongside the main gradient, not inside
    # the range.
    # ------------------------------------------------------------------
    for stem, rel, _variant, _label in SPECS:
        _variant_macros(f"{rel}/shock.json", stem)
    #: ``OnsFuel`` is the name the prose used before the ONS run split into a
    #: shape-only and a both-levels specification; kept as an alias for the
    #: shape-only run so no existing sentence silently loses its number.
    _variant_macros(f"{ONS_FUEL_DIR}/shock.json", "OnsFuel")
    # The round-3 additions: the calendar-window runs and the two motor-fuel
    # margin corrections. Full macro blocks, outside the headline range.
    for stem, rel, _variant, _label in EXTRA_SPECS:
        _variant_macros(f"{rel}/shock.json", stem)
    emit("genSpecCount", len(SPECS), "{:.0f}", "SPECS")
    emit("genSpecCountWord", number_word(len(SPECS)), source="SPECS")
    emit("genExtraSpecCount", number_word(len(EXTRA_SPECS)), source="EXTRA_SPECS")
    emit(
        "genAllVariantCount",
        lambda: number_word(len(cload(COMPARISON))),
        source=COMPARISON,
    )
    # The main aggregate to two decimals, for sentences that compare
    # specifications whose totals differ in the second place (8.96 vs 8.93).
    emit(
        "genCentralAggBn",
        lambda: jload(central)["aggregate_cost_bn"],
        "{:.2f}",
        central,
    )
    for stem, rel, _variant, _label in SPECS:
        emit(
            f"gen{stem}AggPreciseBn",
            lambda rel=rel: jload(f"{rel}/shock.json")["aggregate_cost_bn"],
            "{:.2f}",
            rel,
        )

    # How much of the observed peak each channel is charged for over the year.
    emit(
        "genPumpSustainedFraction",
        lambda: 100 * scenario_obj().pass_through.pump_sustained_fraction,
        "{:.0f}",
        src,
    )
    emit(
        "genGasSustainedFraction",
        lambda: 100 * scenario_obj().pass_through.sustained_fraction,
        "{:.0f}",
        src,
    )

    # ------------------------------------------------------------------
    # more of the decile profile: the cash peak at eight, and medians
    # ------------------------------------------------------------------
    for tag, d in (("Eight", 8), ("Nine", 9)):
        emit(
            f"genDecile{tag}LossGbp",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )
        emit(
            f"genDecile{tag}LossPct",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            central,
        )

    # The %-gradient survives on medians, so it is not a small-income artefact.
    for tag, d in (("One", 1), ("Ten", 10)):
        emit(
            f"genDecile{tag}MedianLossPct",
            lambda d=d: decile_row(jload(central), "intra_decile", d)["p50_loss_pct"],
            "{:.2f}",
            central,
        )
    # Round-3 referees: this used to divide the medians *as printed* (2dp) so a
    # reader dividing the two figures in the prose reproduced the ratio exactly.
    # That is backwards. Rounding a 0.0579 denominator to two decimals moves it
    # by up to 0.9 per cent, and the quotient by as much again: the printed
    # ratio came out 9.7 where the quantity is 9.9. The statistic is now the
    # ratio of the unrounded medians, and it is the printed ratio that is
    # correct; a reader who reconstructs it from two rounded figures gets a
    # slightly different number, which is a property of rounding, not an error
    # in either figure.
    emit(
        "genMedianDecileRatioPct",
        lambda: (
            decile_row(jload(central), "intra_decile", 1)["p50_loss_pct"]
            / decile_row(jload(central), "intra_decile", 10)["p50_loss_pct"]
        ),
        "{:.1f}",
        central,
    )
    # The same ratio to two decimals, for a sentence that needs to show the
    # rounding is not doing the work.
    emit(
        "genMedianDecileRatioPrecisePct",
        lambda: (
            decile_row(jload(central), "intra_decile", 1)["p50_loss_pct"]
            / decile_row(jload(central), "intra_decile", 10)["p50_loss_pct"]
        ),
        "{:.2f}",
        central,
    )

    # decile one's tail: the widest within-decile spread in the distribution
    emit(
        "genDecileOnePNinetyLossPct",
        lambda: decile_row(jload(central), "intra_decile", 1)["p90_loss_pct"],
        "{:.2f}",
        central,
    )
    for tag, key in (
        ("Five", "share_above_5pct"),
        ("Ten", "share_above_10pct"),
    ):
        emit(
            f"genDecileOneShareAbove{tag}Pct",
            lambda key=key: 100 * decile_row(jload(central), "intra_decile", 1)[key],
            "{:.1f}",
            central,
        )

    # ------------------------------------------------------------------
    # region: the finest real geography this dataset supports
    # ------------------------------------------------------------------
    def region_extreme(select):
        return select(jload(central)["region"], key=lambda r: r["mean_loss_pct"])

    for tag, select in (("Max", max), ("Min", min)):
        emit(
            f"genRegion{tag}LossPct",
            lambda select=select: region_extreme(select)["mean_loss_pct"],
            "{:.2f}",
            central,
        )
        emit(
            f"genRegion{tag}LossGbp",
            lambda select=select: region_extreme(select)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )

    # ------------------------------------------------------------------
    # means-tested coverage (a declared limitation, not a code failure)
    # ------------------------------------------------------------------
    for name in ("genMeansTestedShareHouseholds", "genMeansTestedSharePct"):
        emit(
            name, lambda: 100 * jload(central)["means_tested_share"], "{:.1f}", central
        )
    emit(
        "genMeansTestedHouseholdsM",
        lambda: (
            jload(central)["means_tested_share"]
            * sum(r["households_m"] for r in jload(central)["decile"])
        ),
        "{:.1f}",
        central,
    )

    emit(
        "genRebateFullyCompensatedShare",
        lambda: (
            100
            * jload(f"{CENTRAL_SCENARIO}/ippr_rebate.json")["fully_compensated_share"]
        ),
        "{:.0f}",
        f"{CENTRAL_SCENARIO}/ippr_rebate.json",
    )

    # Removing a 5% reduced rate is a 1 - 1/1.05 cut in the VAT-inclusive bill:
    # an arithmetic ceiling on what zero-rating can offset.
    emit(
        "genVatCutMaxOffsetPct",
        lambda: 100 * (1 - 1 / 1.05),
        "{:.2f}",
        "arithmetic: 5% reduced rate removed from a VAT-inclusive bill",
    )

    # ------------------------------------------------------------------
    # sensitivity 1: demand response, the sweep that dominates
    # ------------------------------------------------------------------
    if stale:
        note(STALE_SWEEP_WARNING)
    # ------------------------------------------------------------------
    ela = "sensitivity/elasticity.csv"
    emit(
        "genElasticityAggZeroBn",
        lambda: snum(srow("elasticity.csv", "spec", "flat_0.0"), "aggregate_loss_bn"),
        "{:.2f}",
        ela,
    )
    emit(
        "genElasticityAggHighBn",
        lambda: snum(srow("elasticity.csv", "spec", "flat_-0.8"), "aggregate_loss_bn"),
        "{:.2f}",
        ela,
    )
    for tag, spec in (
        ("Labandeira", "labandeira_short_run"),
        ("Priesmann", "priesmann_short_run"),
    ):
        emit(
            f"gen{tag}ShortShaved",
            lambda spec=spec: (
                100
                * snum(
                    srow("elasticity.csv", "spec", spec), "share_of_upper_bound_shaved"
                )
            ),
            "{:.0f}",
            ela,
        )
    # income-varying elasticities flatten the gradient — the "heat or eat" effect
    emit(
        "genPriesmannDecileRatio",
        lambda: (
            snum(
                srow("elasticity.csv", "spec", "priesmann_short_run"),
                "decile1_loss_pct",
            )
            / snum(
                srow("elasticity.csv", "spec", "priesmann_short_run"),
                "decile10_loss_pct",
            )
        ),
        "{:.1f}",
        ela,
    )

    # ------------------------------------------------------------------
    # sensitivity 2: cap lag — cumulative is invariant, annualised is windowing
    # ------------------------------------------------------------------
    if stale:
        note(STALE_SWEEP_WARNING)
    # ------------------------------------------------------------------
    lag = "sensitivity/cap_lag.csv"

    # The cumulative burden is invariant along the unanchored series (the
    # phase-in weights only move the burden between calendar years), so it is
    # read off that series rather than off one row.
    #: How far the reconstructed cumulative burden may vary along the
    #: unanchored series before the identity is treated as broken, as a share
    #: of its own mean.
    #:
    #: This used to be an absolute £0.01bn, which the round-3 rebuild broke: the
    #: spread is now £0.06bn on £15.6bn, four tenths of one per cent. The
    #: identity is approximate by construction and always was. Undoing the
    #: phase-in leg by leg uses the *central* run's gas/electricity split for
    #: every lag, while each lag row has a slightly different split of its own,
    #: because a different phase-in profile reweights the two legs against each
    #: other. The old profile was a linear ramp and that second-order term was
    #: below the £0.01bn threshold; the observation-window profile is
    #: non-monotone and it is not. A relative tolerance is the right shape for
    #: the test — it still fails loudly if the identity genuinely stops holding —
    #: and the realised spread is emitted below so the paper states the residual
    #: rather than hiding behind the tolerance.
    CUMULATIVE_INVARIANCE_TOLERANCE = 0.01

    def _cumulative_values() -> list[float]:
        return [lag_cumulative_bn(r) for r in lag_rows("unanchored")]

    def _cumulative_bn() -> float:
        values = _cumulative_values()
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        if mean <= 0 or spread / mean > CUMULATIVE_INVARIANCE_TOLERANCE:
            raise ValueError(
                "cumulative loss is not invariant along the unanchored series "
                f"(spread {spread:.3f}bn on a mean of {mean:.3f}bn, "
                f"{100 * spread / mean:.2f}% > "
                f"{100 * CUMULATIVE_INVARIANCE_TOLERANCE:.0f}%)"
            )
        return mean

    emit("genCapLagCumulativeBn", _cumulative_bn, "{:.2f}", lag)
    emit(
        "genCapLagCumulativeSpreadBn",
        lambda: max(_cumulative_values()) - min(_cumulative_values()),
        "{:.3f}",
        lag,
    )
    emit(
        "genCapLagCumulativeSpreadPct",
        lambda: (
            100
            * (max(_cumulative_values()) - min(_cumulative_values()))
            / _cumulative_bn()
        ),
        "{:.2f}",
        lag,
    )
    emit(
        "genCapLagCumulativeTolerancePct",
        lambda: 100 * CUMULATIVE_INVARIANCE_TOLERANCE,
        "{:.0f}",
        lag,
    )
    # How many anchored lags the two cap observations actually identify. Three
    # of the five do not resolve at all, which is a much stronger statement
    # about the cap calibration than the spread of the ones that do.
    emit(
        "genCapLagAnchoredIdentifiedCount",
        lambda: number_word(len(lag_rows("anchored"))),
        source=lag,
    )
    emit(
        "genCapLagAnchoredNonIdentifiedCount",
        lambda: number_word(
            sum(
                1
                for r in sload("cap_lag.csv")
                if r["anchor"].strip() == "anchored" and not is_identified(r)
            )
        ),
        source=lag,
    )
    emit(
        "genCapLagCumulativeMean",
        lambda: (
            _cumulative_bn()
            * float(lag_central()["mean_loss_gbp"])
            / float(lag_central()["aggregate_loss_bn"])
        ),
        "{:.0f}",
        lag,
    )

    #: Lag -> macro infix. LaTeX macro names cannot contain digits, so the
    #: half-quarter lags are spelled out (``OnePointFive``).
    LAG_TAGS = (
        ("LagOne", 1.0),
        ("LagOnePointFive", 1.5),
        ("LagTwo", 2.0),
        ("LagThree", 3.0),
        ("LagFour", 4.0),
    )
    for tag, quarters in LAG_TAGS:
        # Read off the UNANCHORED series.
        #
        # This family used to read the anchored one, and after the round-3
        # rebuild three of its five lags stopped existing: re-anchoring at one,
        # three and four quarters does not identify a pre-war counterfactual at
        # all, so those rows are blank and the macros became \GENMISSING.
        #
        # The unanchored series is also the right series on the merits, which is
        # why the switch is a correction rather than a workaround. A lag
        # sensitivity is supposed to vary *when* the shock reaches the bill and
        # hold *how big it is* fixed. That is exactly what the unanchored series
        # does. The anchored one re-solves the sustained fraction at every lag,
        # so moving along it changes the size of the shock as well as its
        # timing, and the resulting spread is not a lag sensitivity. The two
        # coincide at the paper's own 1.5-quarter lag, so nothing about the
        # central specification moves.
        emit(
            f"genCapLagAnnualised{tag}",
            lambda quarters=quarters: snum(
                lag_row(quarters, "unanchored"), "mean_loss_gbp"
            ),
            "{:.0f}",
            lag,
        )
        # The anchored figure where it exists, under a name that says so.
        if lag_identified(quarters, "anchored"):
            emit(
                f"genCapLagAnnualised{tag}Anchored",
                lambda quarters=quarters: snum(
                    lag_row(quarters, "anchored"), "mean_loss_gbp"
                ),
                "{:.0f}",
                lag,
            )
        emit(
            f"genCapLagAnnualised{tag}Unanchored",
            lambda quarters=quarters: snum(
                lag_row(quarters, "unanchored"), "mean_loss_gbp"
            ),
            "{:.0f}",
            lag,
        )
        # Whether the anchored cell exists at all. Three of the five lags
        # cannot be anchored: re-solving the sustained fraction at those lags
        # either fails to separate the two published caps or drives the pre-war
        # counterfactual above the observed July cap. A sentence that quotes an
        # anchored lag has to be able to say which ones there are.
        emit(
            f"genCapLag{tag}Identified",
            lambda quarters=quarters: "is" if lag_identified(quarters) else "is not",
            source=lag,
        )
    emit(
        "genCapLagAnnualisedPaper",
        lambda: float(lag_central()["mean_loss_gbp"]),
        "{:.0f}",
        lag,
    )
    emit("genCapLagPaperQuarters", lambda: PAPER_CAP_LAG, "{:g}", lag)

    # The spread of the annualised figure across the plausible 1-4 quarter lag
    # range: the paper's "roughly £40" understates it (docs/FIXES.md D17).
    # Reported on both anchoring rules, because they disagree at the long lags.
    def _range(anchor: str) -> float:
        values = [float(r["mean_loss_gbp"]) for r in lag_rows(anchor)]
        return max(values) - min(values)

    emit("genCapLagRangeGbp", lambda: _range("anchored"), "{:.0f}", lag)
    emit("genCapLagRangeUnanchoredGbp", lambda: _range("unanchored"), "{:.0f}", lag)
    #: Anchored range under a name that says which series it is, so a sentence
    #: cannot pick the anchored spread and the unanchored endpoints. Round-4
    #: referees, item 2: the prose ran "£384 at l=1 to £221 at l=4 ... the
    #: spread is £114", which pairs the UNANCHORED endpoints with the ANCHORED
    #: spread (384 - 221 = 163, not 114). The two series are now emitted as
    #: closed sets -- endpoints, their lags, and how many lags each contains --
    #: so a sentence built out of these macros is internally consistent by
    #: construction.
    emit("genCapLagRangeAnchoredGbp", lambda: _range("anchored"), "{:.0f}", lag)

    def _endpoint(anchor: str, select) -> dict:
        return select(lag_rows(anchor), key=lambda r: float(r["mean_loss_gbp"]))

    def _lag_text(row: dict) -> str:
        return f"{float(row['lag_quarters']):g}"

    for series, anchor in (("Anchored", "anchored"), ("Unanchored", "unanchored")):
        for bound, select in (("High", max), ("Low", min)):
            emit(
                f"genCapLag{series}{bound}Gbp",
                lambda anchor=anchor, select=select: float(
                    _endpoint(anchor, select)["mean_loss_gbp"]
                ),
                "{:.0f}",
                lag,
            )
            emit(
                f"genCapLag{series}{bound}Quarters",
                lambda anchor=anchor, select=select: _lag_text(
                    _endpoint(anchor, select)
                ),
                source=lag,
            )
        # How many lags this series actually contains. The unanchored series has
        # all five; the anchored one has two, and a sentence that quotes an
        # anchored range has to be able to say so.
        emit(
            f"genCapLag{series}LagCount",
            lambda anchor=anchor: number_word(len(lag_rows(anchor))),
            source=lag,
        )
        emit(
            f"genCapLag{series}LagCountNumeric",
            lambda anchor=anchor: len(lag_rows(anchor)),
            "{:.0f}",
            lag,
        )
        # The identified lags, spelled out, so the prose can name them rather
        # than assert a count.
        emit(
            f"genCapLag{series}LagList",
            lambda anchor=anchor: ", ".join(
                _lag_text(r)
                for r in sorted(
                    lag_rows(anchor), key=lambda r: float(r["lag_quarters"])
                )
            ),
            source=lag,
        )
        emit(
            f"genCapLag{series}RangeGbp",
            lambda anchor=anchor: _range(anchor),
            "{:.0f}",
            lag,
        )

    # ------------------------------------------------------------------
    # ROUND 4, item 2: the lag is a small sensitivity for the LEVEL and a
    # large one for the COMPOSITION
    # ------------------------------------------------------------------
    #
    # Both identified anchored lags are read off here as a matched pair, so the
    # paper can make that statement with numbers rather than with an adjective.
    # Moving the anchor from the paper's 1.5 quarters to 2 leaves the mean loss
    # in the same order of magnitude but re-solves the pre-war counterfactual
    # cap, collapses the domestic leg, and hands the motor-fuel channel most of
    # the loss.
    COMPOSITION_LAGS = (("Paper", PAPER_CAP_LAG), ("LagTwo", 2.0))
    COMPOSITION_COLUMNS = (
        ("SolvedCapGbp", "prewar_counterfactual_cap_gbp", "{:,.0f}", 1),
        ("DomesticBn", "domestic_loss_bn", "{:.2f}", 1),
        ("MotorFuelBn", "motor_fuel_loss_bn", "{:.2f}", 1),
        ("MotorFuelSharePct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("AggregateBn", "aggregate_loss_bn", "{:.2f}", 1),
        ("MeanGbp", "mean_loss_gbp", "{:.0f}", 1),
        ("SustainedFractionPct", "sustained_fraction", "{:.1f}", 100),
    )
    for tag, quarters in COMPOSITION_LAGS:
        for name, column, fmt, scale in COMPOSITION_COLUMNS:
            emit(
                f"genCapLagAnchored{tag}{name}",
                lambda quarters=quarters, column=column, scale=scale: (
                    scale * snum(lag_row(quarters, "anchored"), column)
                ),
                fmt,
                lag,
            )
    #: The name the prose reached for before this block existed. Kept as an
    #: alias of \genCapLagAnchoredLagTwoMotorFuelSharePct so a sentence written
    #: against either name resolves, and so the \providecommand placeholder the
    #: prose was carrying is superseded by a real emitted value.
    emit(
        "genCapLagMotorFuelShareLagTwoAnchoredPct",
        lambda: 100 * snum(lag_row(2.0, "anchored"), "motor_fuel_share_of_loss"),
        "{:.1f}",
        lag,
    )
    emit(
        "genCapLagMotorFuelSharePaperAnchoredPct",
        lambda: (
            100 * snum(lag_row(PAPER_CAP_LAG, "anchored"), "motor_fuel_share_of_loss")
        ),
        "{:.1f}",
        lag,
    )

    # The moves themselves, so the prose states a delta it did not compute by
    # hand from two printed figures.
    def _anchored_move(column: str) -> tuple[float, float]:
        return (
            snum(lag_row(PAPER_CAP_LAG, "anchored"), column),
            snum(lag_row(2.0, "anchored"), column),
        )

    emit(
        "genCapLagCompositionMotorFuelShareMovePp",
        lambda: (
            100
            * (
                _anchored_move("motor_fuel_share_of_loss")[1]
                - _anchored_move("motor_fuel_share_of_loss")[0]
            )
        ),
        "{:.1f}",
        lag,
    )
    emit(
        "genCapLagCompositionDomesticFallBn",
        lambda: (
            _anchored_move("domestic_loss_bn")[0]
            - _anchored_move("domestic_loss_bn")[1]
        ),
        "{:.2f}",
        lag,
    )
    emit(
        "genCapLagCompositionDomesticFallPct",
        lambda: (
            100
            * (
                _anchored_move("domestic_loss_bn")[0]
                - _anchored_move("domestic_loss_bn")[1]
            )
            / _anchored_move("domestic_loss_bn")[0]
        ),
        "{:.0f}",
        lag,
    )
    emit(
        "genCapLagCompositionMeanFallPct",
        lambda: (
            100
            * (_anchored_move("mean_loss_gbp")[0] - _anchored_move("mean_loss_gbp")[1])
            / _anchored_move("mean_loss_gbp")[0]
        ),
        "{:.0f}",
        lag,
    )
    emit(
        "genCapLagCompositionCapMoveGbp",
        lambda: (
            _anchored_move("prewar_counterfactual_cap_gbp")[1]
            - _anchored_move("prewar_counterfactual_cap_gbp")[0]
        ),
        "{:,.0f}",
        lag,
    )

    # The one-word verdict, so the prose's "small for the level, large for the
    # composition" is a claim this emitter checks rather than an adjective. Both
    # sides are proportional moves of the same anchored pair, so they are
    # comparable: the domestic leg falls by a much larger fraction of itself
    # than the mean loss does.
    def _relative_move(column: str) -> float:
        before, after = _anchored_move(column)
        return abs(after - before) / abs(before)

    emit(
        "genCapLagCompositionMovesMoreThanLevel",
        lambda: (
            "does"
            if _relative_move("domestic_loss_bn") > _relative_move("mean_loss_gbp")
            else "does not"
        ),
        source=lag,
    )
    emit(
        "genCapLagLevelMovePct",
        lambda: 100 * _relative_move("mean_loss_gbp"),
        "{:.0f}",
        lag,
    )
    emit(
        "genCapLagCompositionMovePct",
        lambda: 100 * _relative_move("domestic_loss_bn"),
        "{:.0f}",
        lag,
    )
    # ... and as a share of the paper's own annualised mean, which is the form
    # the prose needs to say whether the windowing choice is material.
    emit(
        "genCapLagSpreadPct",
        lambda: 100 * _range("anchored") / float(lag_central()["mean_loss_gbp"]),
        "{:.0f}",
        lag,
    )
    emit(
        "genCapLagSpreadUnanchoredPct",
        lambda: 100 * _range("unanchored") / float(lag_central()["mean_loss_gbp"]),
        "{:.0f}",
        lag,
    )
    # the annualised 2026 total is almost purely the fast pump channel
    emit(
        "genMotorFuelAnnualisedBn",
        lambda: float(lag_central()["motor_fuel_loss_bn"]),
        "{:.1f}",
        lag,
    )
    emit(
        "genDomesticAnnualisedBn",
        lambda: float(lag_central()["domestic_loss_bn"]),
        "{:.1f}",
        lag,
    )

    # ------------------------------------------------------------------
    # the Resolution Foundation's window: calendar 2026, both legs
    # ------------------------------------------------------------------
    #
    # The paper's window is the twelve months from the shock (March 2026 to
    # February 2027). The RF £11bn comparator is calendar 2026, which opens two
    # months before the shock and closes before the cap has fully moved. Both
    # legs are therefore re-annualised on the RF window here, from the pure
    # scenario data (no microdata): the domestic leg by the ratio of the
    # consumption-weighted cap phase-in over the two windows, leg by leg, and
    # the motor-fuel leg by the ratio of the pump damping fractions. Every
    # component is linear in the price change, so rescaling the totals is exact.
    def _calendar_legs() -> tuple[float, float]:
        """(domestic £bn, motor fuel £bn) re-annualised on calendar 2026."""
        from uk_iran_conflict import scenarios as scen  # noqa: PLC0415

        shock = jload(central)
        agg = float(shock["aggregate_cost_bn"])
        gas_share = float(shock["gas_share_of_loss"])
        elec_share = float(shock["electricity_share_of_loss"])
        fuel_share = float(shock["motor_fuel_share_of_loss"])

        months = scen.CALENDAR_2026_MONTHS
        quarters = scen.quarters_of(months)
        profile = scen.cap_phase_in_profile(scen.CAP_LAG_QUARTERS, quarters)

        def annual_phase_in(weights_by_month) -> float:
            weights = scen.quarterly_consumption_weights(months, weights_by_month)
            return sum(p * w for p, w in zip(profile, weights, strict=True))

        cal_gas = annual_phase_in(scen.MONTHLY_CONSUMPTION_WEIGHTS_GAS)
        cal_elec = annual_phase_in(scen.MONTHLY_CONSUMPTION_WEIGHTS_ELECTRICITY)
        own_gas = float(shock["annual_phase_in_gas"])
        own_elec = float(shock["annual_phase_in_electricity"])

        domestic = agg * (
            gas_share * cal_gas / own_gas + elec_share * cal_elec / own_elec
        )
        fuel = (
            agg
            * fuel_share
            * scen.PUMP_SUSTAINED_FRACTION_CALENDAR_2026
            / scen.REALISED_PUMP_SUSTAINED_FRACTION
        )
        return domestic, fuel

    cal_src = f"{central} + uk_iran_conflict.scenarios (calendar 2026 window)"
    emit("genCalendarAggBn", lambda: sum(_calendar_legs()), "{:.1f}", cal_src)
    emit(
        "genCalendarMotorFuelShareOfLoss",
        lambda: 100 * _calendar_legs()[1] / sum(_calendar_legs()),
        "{:.0f}",
        cal_src,
    )
    emit("genCalendarDomesticBn", lambda: _calendar_legs()[0], "{:.1f}", cal_src)
    emit("genCalendarMotorFuelBn", lambda: _calendar_legs()[1], "{:.1f}", cal_src)
    emit(
        "genCalendarPumpFraction",
        lambda: (
            __import__(
                "uk_iran_conflict.scenarios", fromlist=["x"]
            ).PUMP_SUSTAINED_FRACTION_CALENDAR_2026
        ),
        "{:.2f}",
        "uk_iran_conflict.scenarios",
    )

    # ------------------------------------------------------------------
    # sensitivity 3: marginal-pricing share — composition, not incidence
    # ------------------------------------------------------------------
    if stale:
        note(STALE_SWEEP_WARNING)
    # ------------------------------------------------------------------
    asym = "sensitivity/asymmetry.csv"

    def arow(share: str) -> dict:
        return srow("asymmetry.csv", "marginal_pricing_share", share)

    for tag, share in (("Low", ASYM_LOW), ("High", ASYM_HIGH)):
        emit(
            f"genAsymMeanLoss{tag}",
            lambda share=share: snum(arow(share), "mean_loss_gbp"),
            "{:.0f}",
            asym,
        )
        emit(
            f"genAsymRatio{tag}",
            lambda share=share: snum(arow(share), "decile_ratio_pct"),
            "{:.2f}",
            asym,
        )
    # Round-3 referees: this used to divide the mean losses *as printed* (whole
    # pounds) so the prose's own two figures implied it. Rounding each mean to
    # the pound before dividing shifted the quotient by a tenth of a point --
    # the reported 4.1 against a true 4.0 per cent -- for no gain: a
    # round-then-divide statistic is not the statistic, it is an approximation
    # of it whose error depends on where the two operands happen to fall. Both
    # operands are now divided unrounded.
    emit(
        "genAsymMeanShiftPct",
        lambda: (
            100
            * (
                snum(arow(ASYM_HIGH), "mean_loss_gbp")
                / snum(arow(ASYM_LOW), "mean_loss_gbp")
                - 1
            )
        ),
        "{:.1f}",
        asym,
    )
    emit(
        "genAsymMeanLossCentral",
        lambda: snum(arow(ASYM_CENTRAL), "mean_loss_gbp"),
        "{:.0f}",
        asym,
    )
    emit(
        "genAsymRatioCentral",
        lambda: snum(arow(ASYM_CENTRAL), "decile_ratio_pct"),
        "{:.2f}",
        asym,
    )
    for tag, share in (
        ("Low", ASYM_LOW),
        ("Central", ASYM_CENTRAL),
        ("High", ASYM_HIGH),
    ):
        emit(
            f"genGasShareDomestic{tag}",
            lambda share=share: 100 * snum(arow(share), "gas_share_of_domestic_loss"),
            "{:.1f}",
            asym,
        )

    # ------------------------------------------------------------------
    # sensitivity 1b: the CV welfare bounds (docs/FIXES.md A5)
    #
    # ``aggregate_loss_bn`` in elasticity.csv is the change in *expenditure*.
    # Quoting it as the loss counts foregone heating as costless — the "heat or
    # eat" fallacy. The money-metric statement is the pair of bounds on the
    # compensating variation: Laspeyres (q0.dp, the zero-elasticity upper bound)
    # and Paasche (q1.dp, the lower bound). Demand response shaves the *welfare*
    # loss by far less than it shaves spending, which inverts the appendix's
    # ranking of uncertainties.
    # ------------------------------------------------------------------
    def erow(spec: str) -> dict:
        return srow("elasticity.csv", "spec", spec)

    emit(
        "genCvUpperBn",
        lambda: snum(erow("flat_0.0"), "cv_upper_bn"),
        "{:.2f}",
        ela,
    )
    emit(
        "genCvUpperMeanGbp",
        lambda: snum(erow("flat_0.0"), "cv_upper_mean_gbp"),
        "{:.0f}",
        ela,
    )
    emit(
        "genCvLowerHighEpsBn",
        lambda: snum(erow("flat_-0.8"), "cv_lower_bn"),
        "{:.2f}",
        ela,
    )
    emit(
        "genCvLowerHighEpsMeanGbp",
        lambda: snum(erow("flat_-0.8"), "cv_lower_mean_gbp"),
        "{:.0f}",
        ela,
    )
    # the correction itself: welfare shaved versus spending shaved, at eps=-0.8
    emit(
        "genWelfareShavedHighEpsPct",
        lambda: 100 * snum(erow("flat_-0.8"), "welfare_share_shaved"),
        "{:.1f}",
        ela,
    )
    emit(
        "genSpendShavedHighEpsPct",
        lambda: 100 * snum(erow("flat_-0.8"), "share_of_upper_bound_shaved"),
        "{:.0f}",
        ela,
    )
    # the largest welfare shave anywhere in the sweep — the honest ceiling on
    # how much demand response can matter for the paper's headline
    emit(
        "genWelfareShavedMaxPct",
        lambda: (
            100 * max(float(r["welfare_share_shaved"]) for r in sload("elasticity.csv"))
        ),
        "{:.1f}",
        ela,
    )
    emit(
        "genCvLowerMinBn",
        lambda: min(float(r["cv_lower_bn"]) for r in sload("elasticity.csv")),
        "{:.2f}",
        ela,
    )
    for tag, spec in (
        ("Labandeira", "labandeira_short_run"),
        ("Priesmann", "priesmann_short_run"),
    ):
        emit(
            f"gen{tag}ShortWelfareShaved",
            lambda spec=spec: 100 * snum(erow(spec), "welfare_share_shaved"),
            "{:.1f}",
            ela,
        )
        emit(
            f"gen{tag}ShortSpendShaved",
            lambda spec=spec: 100 * snum(erow(spec), "share_of_upper_bound_shaved"),
            "{:.0f}",
            ela,
        )
        emit(
            f"gen{tag}ShortCvLowerBn",
            lambda spec=spec: snum(erow(spec), "cv_lower_bn"),
            "{:.2f}",
            ela,
        )

    # ------------------------------------------------------------------
    # the seven specifications: ranges across them (docs/FIXES.md A3)
    #
    # The motor-fuel majority claim is calibration-dependent, so every headline
    # derived from it is reported as a range over the specifications rather than
    # as a single share.
    # ------------------------------------------------------------------
    def spec_values(column: str) -> list[float]:
        rows = comparison_rows()
        out = []
        for _stem, _rel, variant, _label in SPECS:
            out.append(float(rows[variant][column]))
        return out

    for name, column, fmt, scale in (
        ("genSpecAggMinBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genSpecAggMaxBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genSpecLossMeanMin", "mean_loss_gbp", "{:.0f}", 1),
        ("genSpecLossMeanMax", "mean_loss_gbp", "{:.0f}", 1),
        ("genSpecFuelShareMinPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genSpecFuelShareMaxPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genSpecDecileRatioMin", "d1_d10_ratio_pct", "{:.2f}", 1),
        ("genSpecDecileRatioMax", "d1_d10_ratio_pct", "{:.2f}", 1),
    ):
        select = (
            min if name.endswith(("MinBn", "MeanMin", "MinPct", "RatioMin")) else max
        )
        emit(
            name,
            lambda column=column, select=select, scale=scale: (
                scale * select(spec_values(column))
            ),
            fmt,
            COMPARISON,
        )

    # ------------------------------------------------------------------
    # ROUND 4, item 3: computed extrema, not hand-picked endpoints
    # ------------------------------------------------------------------
    #
    # The prose quoted the domestic share as "between 34.6 and 57.7 per cent
    # across the seven specifications" using \genPeakFuelDomesticShareOfLoss and
    # \genOnsLevelsDomesticShareOfLoss. The low end happened to be the true
    # minimum; the high end was not the maximum -- the ONS-levels row is 57.7
    # and the steady-state row is 60.8. Picking two named rows and calling them
    # a range is exactly the failure mode a min/max macro exists to prevent, so
    # the range now has its own computed pair and the named-row macros are left
    # for sentences that mean that named row.
    #
    # ``domestic`` is not a column of comparison.csv; it is the gas and
    # electricity shares summed, which is the definition every bill instrument
    # is bounded by.
    def spec_domestic_shares() -> list[float]:
        rows = comparison_rows()
        return [
            100
            * (
                float(rows[variant]["gas_share_of_loss"])
                + float(rows[variant]["electricity_share_of_loss"])
            )
            for _stem, _rel, variant, _label in SPECS
        ]

    def all_variant_domestic_shares() -> list[float]:
        return [
            100
            * (float(r["gas_share_of_loss"]) + float(r["electricity_share_of_loss"]))
            for r in cload(COMPARISON)
            if str(r["gas_share_of_loss"]).strip() != ""
        ]

    for name, values, select in (
        ("genDomesticShareMin", spec_domestic_shares, min),
        ("genDomesticShareMax", spec_domestic_shares, max),
        ("genAllVariantDomesticShareMin", all_variant_domestic_shares, min),
        ("genAllVariantDomesticShareMax", all_variant_domestic_shares, max),
    ):
        emit(
            name,
            lambda values=values, select=select: select(values()),
            "{:.1f}",
            COMPARISON,
        )

    # Which specification sits at each end, so a sentence naming the row cannot
    # name the wrong one.
    def _domestic_extreme_label(select) -> str:
        rows = comparison_rows()
        pairs = [
            (
                100
                * (
                    float(rows[variant]["gas_share_of_loss"])
                    + float(rows[variant]["electricity_share_of_loss"])
                ),
                label,
            )
            for _stem, _rel, variant, label in SPECS
        ]
        return select(pairs, key=lambda pair: pair[0])[1]

    for name, select in (
        ("genDomesticShareMinSpec", min),
        ("genDomesticShareMaxSpec", max),
    ):
        emit(
            name,
            lambda select=select: _domestic_extreme_label(select),
            source=COMPARISON,
        )

    # The same computed-extremum treatment for the other quantity the prose can
    # hand-pick: the within-decile dispersion range across specifications, whose
    # named rows sit either side of the true extremes in the same way.
    def spec_column_values(column: str) -> list[float]:
        rows = comparison_rows()
        out = []
        for _stem, _rel, variant, _label in SPECS:
            raw = str(rows[variant].get(column, "")).strip()
            if raw:
                out.append(float(raw))
        if not out:
            raise ValueError(f"comparison.csv carries no {column}")
        return out

    for name, column, fmt, scale in (
        ("genSpecMeanLossPctMin", "mean_loss_pct", "{:.2f}", 1),
        ("genSpecMeanLossPctMax", "mean_loss_pct", "{:.2f}", 1),
        ("genSpecDecileOneLossPctMin", "decile1_loss_pct", "{:.2f}", 1),
        ("genSpecDecileOneLossPctMax", "decile1_loss_pct", "{:.2f}", 1),
        ("genSpecWithinDecileRangeMinPp", "mean_within_decile_range_pp", "{:.2f}", 1),
        ("genSpecWithinDecileRangeMaxPp", "mean_within_decile_range_pp", "{:.2f}", 1),
        ("genSpecMtShareMinPct", "means_tested_share_of_loss", "{:.2f}", 100),
        ("genSpecMtShareMaxPct", "means_tested_share_of_loss", "{:.2f}", 100),
    ):
        select = min if name.endswith(("Min", "MinPp", "MinPct")) else max
        emit(
            name,
            lambda column=column, select=select, scale=scale: (
                scale * select(spec_column_values(column))
            ),
            fmt,
            COMPARISON,
        )

    # how far the specification choice moves the aggregate, as a share of the
    # main specification — the number the uncertainty ranking is built on
    emit(
        "genSpecAggSpreadPct",
        lambda: (
            100
            * (
                max(spec_values("aggregate_cost_bn"))
                - min(spec_values("aggregate_cost_bn"))
            )
            / float(jload(central)["aggregate_cost_bn"])
        ),
        "{:.0f}",
        COMPARISON,
    )

    # The same ranges over **every** variant in comparison.csv, window changes
    # and margin corrections included. This is the widest honest statement of
    # how far the channel-composition result depends on choices the paper makes
    # rather than on the shock, and it is the range the decomposition figure
    # draws. It is a different object from the ``Spec`` range above and is named
    # so.
    def all_variant_values(column: str) -> list[float]:
        values = [
            float(r[column]) for r in cload(COMPARISON) if str(r[column]).strip() != ""
        ]
        if not values:
            raise ValueError(f"comparison.csv carries no {column}")
        return values

    for name, column, fmt, scale in (
        ("genAllVariantAggMinBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genAllVariantAggMaxBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genAllVariantLossMeanMin", "mean_loss_gbp", "{:.0f}", 1),
        ("genAllVariantLossMeanMax", "mean_loss_gbp", "{:.0f}", 1),
        ("genAllVariantFuelShareMinPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genAllVariantFuelShareMaxPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genAllVariantDecileRatioMin", "d1_d10_ratio_pct", "{:.2f}", 1),
        ("genAllVariantDecileRatioMax", "d1_d10_ratio_pct", "{:.2f}", 1),
        ("genAllVariantDecileTwoRatioMin", "d2_d10_ratio_pct", "{:.2f}", 1),
        ("genAllVariantDecileTwoRatioMax", "d2_d10_ratio_pct", "{:.2f}", 1),
        ("genAllVariantMtShareMinPct", "means_tested_share_of_loss", "{:.2f}", 100),
        ("genAllVariantMtShareMaxPct", "means_tested_share_of_loss", "{:.2f}", 100),
    ):
        select = min if "Min" in name else max
        emit(
            name,
            lambda column=column, select=select, scale=scale: (
                scale * select(all_variant_values(column))
            ),
            fmt,
            COMPARISON,
        )
    # Does motor fuel stay the majority of the loss across every variant? It
    # does not, and this is the macro that says so without a hand-written claim.
    emit(
        "genFuelMajorityEveryVariant",
        lambda: (
            "does"
            if min(all_variant_values("motor_fuel_share_of_loss")) > 0.5
            else "does not"
        ),
        source=COMPARISON,
    )
    emit(
        "genFuelMajorityVariantCount",
        lambda: number_word(
            sum(1 for v in all_variant_values("motor_fuel_share_of_loss") if v > 0.5)
        ),
        source=COMPARISON,
    )
    # Every variant agrees the counterfactual cap is the same solved number:
    # the calibration is a property of the price path, not of the accounting.
    emit(
        "genPrewarCapInvariantAcrossVariants",
        lambda: (
            "is"
            if max(all_variant_values("prewar_counterfactual_cap_gbp"))
            - min(all_variant_values("prewar_counterfactual_cap_gbp"))
            < 0.01
            else "is not"
        ),
        source=COMPARISON,
    )

    # the fuel share excluding the explicit peak-fuel upper bound: the range a
    # reader should use when the upper bound is not the quantity of interest
    emit(
        "genSpecFuelShareMaxExPeakPct",
        lambda: (
            100
            * max(
                float(comparison_rows()[variant]["motor_fuel_share_of_loss"])
                for _s, _r, variant, _l in SPECS
                if variant != "peak_fuel"
            )
        ),
        "{:.1f}",
        COMPARISON,
    )
    # phase-in weights actually applied to the domestic leg (decision D2)
    for name, column in (
        ("genAnnualPhaseInGasPct", "annual_phase_in_gas"),
        ("genAnnualPhaseInElecPct", "annual_phase_in_electricity"),
    ):
        emit(
            name,
            lambda column=column: 100 * float(jload(central)[column]),
            "{:.1f}",
            central,
        )

    # ------------------------------------------------------------------
    # decile coverage and the dropped weight (docs/FIXES.md A6)
    # ------------------------------------------------------------------
    def cov() -> dict:
        return jload(central)["coverage"]

    emit(
        "genCoverageExcludedHouseholdsM",
        lambda: cov()["households_m"],
        "{:.2f}",
        central,
    )
    emit(
        "genCoverageExcludedSharePct",
        lambda: 100 * cov()["share_of_households"],
        "{:.2f}",
        central,
    )
    emit(
        "genCoverageExcludedLossSharePct",
        lambda: 100 * cov()["share_of_loss"],
        "{:.2f}",
        central,
    )
    emit(
        "genCoverageExcludedZeroIncomeSharePct",
        lambda: 100 * cov()["zero_or_negative_income_share"],
        "{:.0f}",
        central,
    )
    emit(
        "genCoveredHouseholdsM",
        lambda: cov()["covered_households_m"],
        "{:.2f}",
        central,
    )
    emit("genCoveredLossBn", lambda: cov()["covered_loss_bn"], "{:.2f}", central)
    emit(
        "genZeroOrNegIncomeSharePct",
        lambda: 100 * jload(central)["zero_or_negative_income_share"],
        "{:.2f}",
        central,
    )

    # ------------------------------------------------------------------
    # persisted prose values (docs/FIXES.md C12), including the corrected
    # decile-one zero-income share: 20.05% equivalised, not 0.57%
    # ------------------------------------------------------------------
    pv = "persisted_values.json"
    emit(
        "genDecileOneZeroIncomeShare",
        lambda: (
            100 * persisted()["decile1_zero_or_negative_income_share_equivalised_ahc"]
        ),
        "{:.2f}",
        pv,
    )
    emit(
        "genDecileOneZeroIncomeShareUnequiv",
        lambda: (
            100 * persisted()["decile1_zero_or_negative_income_share_unequivalised"]
        ),
        "{:.2f}",
        pv,
    )
    emit(
        "genDecileOneMedianIncomeGbp",
        lambda: persisted()["decile1_median_income_gbp_equivalised_ahc"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genDecileOneMeanIncomeGbp",
        lambda: persisted()["decile1_mean_income_gbp_equivalised_ahc"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genDecileOneMedianIncomeUnequivGbp",
        lambda: persisted()["decile1_median_income_gbp_unequivalised"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genModelDomesticMeanGbp",
        lambda: persisted()["mean_domestic_energy_spend_gbp"],
        "{:,.0f}",
        pv,
    )
    emit(
        "genModelFuelMeanGbp",
        lambda: persisted()["mean_motor_fuel_spend_gbp"],
        "{:,.0f}",
        pv,
    )
    # the raw imputation, and the profile after the ONS-shape rescaling. The
    # rescaled numbers are OURS, not ONS's: see ONS_BENCH for the published ones.
    for tag, index in (("One", 0), ("Ten", 9)):
        emit(
            f"genFuelSpendDecile{tag}RawGbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"]["raw"][index],
            "{:,.0f}",
            pv,
        )
        emit(
            f"genFuelSpendDecile{tag}OnsGbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"]["ons_shape"][
                index
            ],
            "{:,.0f}",
            pv,
        )
        emit(
            f"genRescaledFuelDecile{tag}Gbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"]["ons_shape"][
                index
            ],
            "{:,.0f}",
            pv,
        )
        emit(
            f"genOnsLevelsFuelDecile{tag}Gbp",
            lambda index=index: persisted()["motor_fuel_decile_mean_gbp"][
                "ons_both_levels"
            ][index],
            "{:,.0f}",
            pv,
        )

    # the two disclosed imputation defects, in both directions (C11)
    for name, (value, fmt) in ONS_BENCH.items():
        emit(name, value, fmt, "ONS Family Spending FYE 2025 (published)")
    emit(
        "genDomesticUnderImputationPct",
        lambda: (
            100
            * (1 - persisted()["mean_domestic_energy_spend_gbp"] / ONS_BENCH_DOMESTIC)
        ),
        "{:.0f}",
        pv,
    )
    emit(
        "genFuelOverImputationPct",
        lambda: 100 * (persisted()["mean_motor_fuel_spend_gbp"] / ONS_BENCH_FUEL - 1),
        "{:.0f}",
        pv,
    )

    # ------------------------------------------------------------------
    # continuous compensation measures at each sponsor's stated design (B9)
    # ------------------------------------------------------------------
    for policy, tag in POLICY_MACRO.items():
        rel = f"{CENTRAL_SCENARIO}/{policy}.json"
        emit(
            f"gen{tag}OffsetSharePct",
            lambda rel=rel: 100 * jload(rel)["share_of_aggregate_loss_offset"],
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualMeanGbp",
            lambda rel=rel: jload(rel)["mean_residual_loss_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualMedianGbp",
            lambda rel=rel: jload(rel)["median_residual_loss_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualDecileOneGbp",
            lambda rel=rel: jload(rel)["mean_residual_loss_by_decile"]["1"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}ResidualDecileTenGbp",
            lambda rel=rel: jload(rel)["mean_residual_loss_by_decile"]["10"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}OffsetDecileOnePct",
            lambda rel=rel: 100 * jload(rel)["share_of_loss_offset_by_decile"]["1"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}MeanGainGbp",
            lambda rel=rel: jload(rel)["mean_gain_gbp"],
            "{:.0f}",
            rel,
        )
        emit(
            f"gen{tag}FullyCompensatedShare",
            lambda rel=rel: 100 * jload(rel)["fully_compensated_share"],
            "{:.0f}",
            rel,
        )

    # The JRF block, re-specified as a level subsidy, now costs more than JRF's
    # own stated £5bn rather than a third of it (B7).
    emit(
        "genJrfStatedCostBn",
        lambda: jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["stated_cost_bn"],
        "{:.1f}",
        f"{CENTRAL_SCENARIO}/jrf_block.json",
    )
    emit(
        "genJrfCostOverStated",
        lambda: (
            jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["cost_bn"]
            / jload(f"{CENTRAL_SCENARIO}/jrf_block.json")["stated_cost_bn"]
        ),
        "{:.1f}",
        f"{CENTRAL_SCENARIO}/jrf_block.json",
    )

    # ------------------------------------------------------------------
    # the common-envelope scorecard (B7), rebuilt: five row types per policy
    # ------------------------------------------------------------------
    #
    # The headline of the rebuild is a withdrawal. "VAT zero-rating wins at a
    # common envelope" was an artefact of scaling every instrument
    # proportionally to £5bn regardless of whether the instrument could reach
    # that spend: zero-rating tops out at the cost of removing all five VAT
    # points, and the scaled row got there by removing more points than the tax
    # has. Two instruments are infeasible when scaled.
    #
    # Round 3 added the row the comparison was missing. ``feasible_max`` is the
    # instrument's own ceiling with no envelope applied at all;
    # ``common_capped`` is envelope *absorption*, ``min(envelope, feasible-max
    # cost)``. This file used to document the second as "the instrument at its
    # feasible maximum", which is true only for an instrument that saturates
    # below the envelope and false for the two that do not. Both rows are now
    # emitted under names that say which question they answer.
    emit(
        "genEnvelopeBn",
        lambda: snum(envelope_row("vat_zero", ENVELOPE_CAPPED), "envelope_bn"),
        "{:.0f}",
        ENVELOPE,
    )

    #: Short, LaTeX-safe prose names. The CSV ``label`` carries "%" and "->",
    #: which would break the build, so it is never emitted verbatim.
    envelope_names = {
        "social_tariff": "the means-tested social tariff",
        "jrf_block": "the JRF discounted block",
        "whd_expansion": "the Warm Home Discount expansion",
        "vat_zero": "VAT zero-rating",
        "ippr_rebate": "the flat rebate",
    }

    def env_pct(policy: str, envelope: str, column: str) -> float:
        return 100 * snum(envelope_row(policy, envelope), column)

    for policy, tag in POLICY_MACRO.items():
        # --- the instrument's own design parameters, whatever their units ---
        for name, column in (
            ("Stated", "stated_parameter"),
            ("FeasibleMax", "feasible_max_parameter"),
        ):
            if not envelope_has(policy, ENVELOPE_CAPPED):
                continue
            row = envelope_row(policy, ENVELOPE_CAPPED)
            suffix = parameter_suffix(row)
            emit(
                f"gen{tag}{name}Parameter{suffix}",
                lambda row=row, column=column: parameter_text(row[column]),
                source=ENVELOPE,
            )
            emit(
                f"gen{tag}{name}Parameter",
                lambda row=row, column=column: parameter_text(row[column]),
                source=ENVELOPE,
            )

        # --- the ceiling: what this instrument can actually absorb ---
        if envelope_has(policy, ENVELOPE_CAPPED):
            emit(
                f"gen{tag}AbsorbableBn",
                lambda policy=policy: snum(
                    envelope_row(policy, ENVELOPE_CAPPED), "absorbable_envelope_bn"
                ),
                "{:.2f}",
                ENVELOPE,
            )
            emit(
                f"gen{tag}AbsorbableEnvelopeBn",
                lambda policy=policy: snum(
                    envelope_row(policy, ENVELOPE_CAPPED), "absorbable_envelope_bn"
                ),
                "{:.2f}",
                ENVELOPE,
            )
            emit(
                f"gen{tag}AbsorbsFullEnvelope",
                lambda policy=policy: (
                    "yes"
                    if snum(
                        envelope_row(policy, ENVELOPE_CAPPED), "absorbable_envelope_bn"
                    )
                    >= snum(envelope_row(policy, ENVELOPE_CAPPED), "envelope_bn") - 1e-6
                    else "no"
                ),
                source=ENVELOPE,
            )

        # --- one block of outcome macros per row type that exists ---
        for envelope, row_tag in ENVELOPE_ROW_TAGS:
            if not envelope_has(policy, envelope):
                continue
            row = envelope_row(policy, envelope)
            suffix = parameter_suffix(row)

            #: ``Envelope`` with no row qualifier means the envelope-absorption
            #: row: the name predates the rebuild and the prose already carries
            #: it. ``Absorbed`` is the name that says what the row is.
            aliases = [row_tag, *ENVELOPE_ROW_ALIASES.get(envelope, ())]
            for alias in aliases:
                emit(
                    f"gen{tag}{alias}OffsetPct",
                    lambda policy=policy, envelope=envelope: env_pct(
                        policy, envelope, "share_of_aggregate_loss_offset"
                    ),
                    "{:.1f}",
                    ENVELOPE,
                )
                # ROUND 4, item 6: at one decimal place the social tariff's
                # stated and envelope-absorption offsets both print 5.2, so the
                # saturation sentence -- "moves the offset from 5.2 to 5.2 per
                # cent" -- said nothing. The saturation finding is that the move
                # is nearly nil, which needs a second decimal place to state:
                # 5.18 to 5.22. Emitted alongside rather than instead, so
                # sentences that want the round figure keep it.
                emit(
                    f"gen{tag}{alias}OffsetPrecisePct",
                    lambda policy=policy, envelope=envelope: env_pct(
                        policy, envelope, "share_of_aggregate_loss_offset"
                    ),
                    "{:.2f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}CostBn",
                    lambda row=row: float(row["cost_bn"]),
                    "{:.2f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}ResidualMeanGbp",
                    lambda row=row: float(row["mean_residual_loss_gbp"]),
                    "{:.0f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}ResidualMedianGbp",
                    lambda row=row: float(row["median_residual_loss_gbp"]),
                    "{:.0f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}ResidualDecileOneGbp",
                    lambda row=row: float(row["mean_residual_loss_d1"]),
                    "{:.0f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}MeanGainGbp",
                    lambda row=row: float(row["mean_gain_gbp"]),
                    "{:.0f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}UncompensatedPct",
                    lambda policy=policy, envelope=envelope: env_pct(
                        policy, envelope, "uncompensated_share_overall"
                    ),
                    "{:.0f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}ShareToBottomThreePct",
                    lambda policy=policy, envelope=envelope: env_pct(
                        policy, envelope, "share_to_bottom_three"
                    ),
                    "{:.1f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}CostPerPound",
                    lambda row=row: float(row["cost_per_pound_decile_one"]),
                    "{:.1f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}Scale",
                    lambda row=row: float(row["envelope_scale"]),
                    "{:.2f}",
                    ENVELOPE,
                )
                # The implied setting of the instrument's own dial. This is
                # where infeasibility becomes visible: 12.7 VAT points, a 143
                # per cent bill discount, a £1,083 WHD payment.
                emit(
                    f"gen{tag}{alias}Parameter{suffix}",
                    lambda row=row: parameter_text(row["implied_parameter"]),
                    source=ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}Parameter",
                    lambda row=row: parameter_text(row["implied_parameter"]),
                    source=ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}Feasible",
                    lambda row=row: "yes" if is_true(row["is_feasible"]) else "no",
                    source=ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}IsFeasible",
                    lambda row=row: (
                        "feasible" if is_true(row["is_feasible"]) else "infeasible"
                    ),
                    source=ENVELOPE,
                )

            if envelope == ENVELOPE_ELIGIBILITY:
                emit(
                    f"gen{tag}EligibleSharePct",
                    lambda policy=policy: env_pct(
                        policy, ENVELOPE_ELIGIBILITY, "eligible_share"
                    ),
                    "{:.0f}",
                    ENVELOPE,
                )

    # --- how many instruments the proportional scaling actually breaks -------
    def _infeasible_scaled() -> list[str]:
        return [
            r["policy"].strip()
            for r in envelope_rows(ENVELOPE_SCALED)
            if not is_true(r["is_feasible"])
        ]

    emit(
        "genEnvelopeInfeasibleScaledCount",
        lambda: number_word(len(_infeasible_scaled())),
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeInfeasibleScaledLabels",
        lambda: " and ".join(envelope_names[p] for p in _infeasible_scaled()),
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeFeasibleCappedCount",
        lambda: number_word(
            sum(1 for r in envelope_rows(ENVELOPE_CAPPED) if is_true(r["is_feasible"]))
        ),
        source=ENVELOPE,
    )

    # --- best and worst, scored on the *capped* rows only -------------------
    #
    # Restricting to the capped rows is the substantive change: on the scaled
    # rows VAT zero-rating offsets the most, but only by removing more VAT than
    # exists. The comparison below is between instruments that could be built.
    def envelope_best(select, column: str) -> dict:
        rows = envelope_rows(ENVELOPE_CAPPED)
        if not rows:
            raise ValueError(f"{ENVELOPE}: no {ENVELOPE_CAPPED} rows")
        return select(rows, key=lambda r: float(r[column]))

    emit(
        "genEnvelopeBestLabel",
        lambda: envelope_names[
            envelope_best(max, "share_of_aggregate_loss_offset")["policy"].strip()
        ],
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeBestResidualLabel",
        lambda: envelope_names[
            envelope_best(min, "mean_residual_loss_gbp")["policy"].strip()
        ],
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeBestOffsetPct",
        lambda: (
            100
            * float(
                envelope_best(max, "share_of_aggregate_loss_offset")[
                    "share_of_aggregate_loss_offset"
                ]
            )
        ),
        "{:.1f}",
        ENVELOPE,
    )
    emit(
        "genEnvelopeBestResidualMeanGbp",
        lambda: float(
            envelope_best(min, "mean_residual_loss_gbp")["mean_residual_loss_gbp"]
        ),
        "{:.0f}",
        ENVELOPE,
    )
    emit(
        "genEnvelopeWorstOffsetPct",
        lambda: (
            100
            * float(
                envelope_best(min, "share_of_aggregate_loss_offset")[
                    "share_of_aggregate_loss_offset"
                ]
            )
        ),
        "{:.1f}",
        ENVELOPE,
    )
    emit(
        "genEnvelopeWorstLabel",
        lambda: envelope_names[
            envelope_best(min, "share_of_aggregate_loss_offset")["policy"].strip()
        ],
        source=ENVELOPE,
    )
    # The eligibility rows are the ones that actually spend the envelope, and
    # they beat every capped row: at a fixed budget, who is *in* the scheme
    # dominates how generous the scheme is to those already in it.
    emit(
        "genEnvelopeBestEligibilityLabel",
        lambda: envelope_names[
            max(
                envelope_rows(ENVELOPE_ELIGIBILITY),
                key=lambda r: float(r["share_of_aggregate_loss_offset"]),
            )["policy"].strip()
        ],
        source=ENVELOPE,
    )
    emit(
        "genEnvelopeBestEligibilityOffsetPct",
        lambda: (
            100
            * max(
                float(r["share_of_aggregate_loss_offset"])
                for r in envelope_rows(ENVELOPE_ELIGIBILITY)
            )
        ),
        "{:.1f}",
        ENVELOPE,
    )

    # ------------------------------------------------------------------
    # domestic-leg parameter sweep (A4): only the product is identified
    # ------------------------------------------------------------------
    for tag, parameter in (
        ("Split", "sustained_fraction_split"),
        ("Nbp", "prewar_nbp_pence_per_therm"),
        ("GasShare", "wholesale_share_gas_bill"),
        ("ElecShare", "wholesale_share_electricity_bill"),
        # Added in round 3, and by some distance the widest block of the four.
        ("GasProfile", "gas_peak_monthly_profile"),
    ):
        for suffix, select in (("Min", min), ("Max", max)):
            emit(
                f"genDomesticLeg{tag}Agg{suffix}Bn",
                lambda parameter=parameter, select=select: leg_span(
                    parameter, "aggregate_cost_bn", select
                ),
                "{:.2f}",
                DOMESTIC_LEG,
            )
            emit(
                f"genDomesticLeg{tag}Mean{suffix}Gbp",
                lambda parameter=parameter, select=select: leg_span(
                    parameter, "mean_loss_gbp", select
                ),
                "{:.0f}",
                DOMESTIC_LEG,
            )
            emit(
                f"genDomesticLeg{tag}FuelShare{suffix}Pct",
                lambda parameter=parameter, select=select: (
                    100 * leg_span(parameter, "motor_fuel_share_of_loss", select)
                ),
                "{:.1f}",
                DOMESTIC_LEG,
            )
    emit(
        "genDomesticLegAnchorProduct",
        lambda: float(leg_rows("sustained_fraction_split")[0]["anchor_product"]),
        "{:.3f}",
        DOMESTIC_LEG,
    )
    for name, select in (
        ("genDomesticLegAggMinBn", min),
        ("genDomesticLegAggMaxBn", max),
    ):
        emit(
            name,
            lambda select=select: select(
                float(r["aggregate_cost_bn"]) for r in leg_all()
            ),
            "{:.2f}",
            DOMESTIC_LEG,
        )
    for name, select in (
        ("genDomesticLegFuelShareMinPct", min),
        ("genDomesticLegFuelShareMaxPct", max),
    ):
        emit(
            name,
            lambda select=select: (
                100 * select(float(r["motor_fuel_share_of_loss"]) for r in leg_all())
            ),
            "{:.1f}",
            DOMESTIC_LEG,
        )
    emit(
        "genDomesticLegSpreadPct",
        lambda: (
            100
            * (
                max(float(r["aggregate_cost_bn"]) for r in leg_all())
                - min(float(r["aggregate_cost_bn"]) for r in leg_all())
            )
            / float(jload(central)["aggregate_cost_bn"])
        ),
        "{:.0f}",
        DOMESTIC_LEG,
    )

    # ------------------------------------------------------------------
    # petrol versus diesel by decile (D26): the decile-eight cash spike
    # ------------------------------------------------------------------
    emit(
        "genPetrolUpliftPct",
        lambda: 100 * (float(fuel_decile(1)["petrol_price_factor"]) - 1),
        "{:.1f}",
        FUEL_BY_DECILE,
    )
    emit(
        "genDieselUpliftPct",
        lambda: 100 * (float(fuel_decile(1)["diesel_price_factor"]) - 1),
        "{:.1f}",
        FUEL_BY_DECILE,
    )
    for tag, d in (("One", 1), ("Eight", 8), ("Ten", 10)):
        emit(
            f"genDieselShareSpendDecile{tag}Pct",
            lambda d=d: 100 * float(fuel_decile(d)["diesel_share_of_fuel_spend"]),
            "{:.1f}",
            FUEL_BY_DECILE,
        )
        emit(
            f"genDieselShareLossDecile{tag}Pct",
            lambda d=d: 100 * float(fuel_decile(d)["diesel_share_of_fuel_loss"]),
            "{:.1f}",
            FUEL_BY_DECILE,
        )
        emit(
            f"genFuelLossDecile{tag}Gbp",
            lambda d=d: float(fuel_decile(d)["motor_fuel_loss_gbp"]),
            "{:.0f}",
            FUEL_BY_DECILE,
        )
        emit(
            f"genFuelSpendShareDecile{tag}Pct",
            lambda d=d: 100 * float(fuel_decile(d)["share_with_any_fuel_spend"]),
            "{:.1f}",
            FUEL_BY_DECILE,
        )

    # ------------------------------------------------------------------
    # means-tested audit (C13): the resolved variable set, logged not swallowed
    # ------------------------------------------------------------------
    aud = "means_tested_audit.json"
    emit(
        "genMeansTestedVarCount",
        lambda: len(audit()["resolved_variables"]),
        "{:.0f}",
        aud,
    )
    emit(
        "genMeansTestedMissingCount",
        lambda: len(audit()["missing_required"]),
        "{:.0f}",
        aud,
    )
    emit(
        "genUniversalCreditHouseholdsM",
        lambda: audit()["by_variable"]["universal_credit"]["households_m"],
        "{:.2f}",
        aud,
    )
    emit(
        "genPensionCreditHouseholdsM",
        lambda: audit()["by_variable"]["pension_credit"]["households_m"],
        "{:.2f}",
        aud,
    )
    emit(
        "genTotalHouseholdsM",
        lambda: audit()["total_households_m"],
        "{:.1f}",
        aud,
    )

    # ------------------------------------------------------------------
    # macros the prose asks for that no results file supports
    #
    # These names appear in the manuscript but describe *combinations* of
    # specifications that were never run: there is no symmetric-damping run on
    # the ONS both-levels calibration, and no symmetric-damping run on the
    # steady-state basis. They are emitted as \GENMISSING rather than guessed at,
    # so the build fails visibly and the sentence gets repointed at a
    # specification that exists (\genSymDampMotorFuelShareOfLoss,
    # \genOnsLevelsMotorFuelShareOfLoss, \genSteadyStateMotorFuelShareOfLoss).
    # ------------------------------------------------------------------
    def _no_such_run(name: str):
        def fail():
            raise KeyError(
                "no such specification: this macro asks for a combination of two "
                "specifications that results/ does not contain"
            )

        emit(name, fail, source="results/robustness/comparison.csv")

    # The two combination specifications: symmetric damping crossed with the
    # other accounting choices. These are the calibrations under which the
    # motor-fuel majority fails, so the abstract cites them and they are real
    # runs (analysis/run_combinations.py), not arithmetic.
    def _combo(key: str) -> float:
        cell = json.loads((R / "robustness" / "combinations.json").read_text())
        return float(cell[key]["motor_fuel_share_pct"])

    emit(
        "genSymDampSteadyMotorFuelShare",
        lambda: _combo("symmetric_steady_state"),
        "{:.1f}",
        "robustness/combinations.json",
    )
    emit(
        "genSymDampOnsLevelsMotorFuelShare",
        lambda: _combo("symmetric_ons_levels"),
        "{:.1f}",
        "robustness/combinations.json",
    )
    for _name in ("genSymDampSteadyAggBn", "genSymDampOnsLevelsAggBn"):
        _k = "symmetric_steady_state" if "Steady" in _name else "symmetric_ons_levels"
        emit(
            _name,
            lambda k=_k: json.loads(
                (R / "robustness" / "combinations.json").read_text()
            )[k]["aggregate_cost_bn"],
            "{:.2f}",
            "robustness/combinations.json",
        )

    # ==================================================================
    # ROUND 3: the price path rebuilt from a solved counterfactual
    # ==================================================================
    #
    # The pre-war baseline used to be the observed 1 Jul - 30 Sep 2026 cap. That
    # is a post-shock cap: the same quarter was serving as the un-shocked
    # denominator and, with a phase-in of 1.0, as the fully shocked numerator.
    # It is replaced by a counterfactual solved jointly with the sustained
    # fraction from BOTH published caps, and the solution reproduces both to
    # floating point. The old value was not merely imprecise; it was absorbing
    # the whole calibration error, which is why the solved sustained fraction
    # moves from 0.199 to 0.765.
    FINDINGS = "round3_findings.json"

    def cap() -> dict:
        return jload(FINDINGS)["cap_calibration"]

    def capval() -> dict:
        return cap()["validation"]

    emit(
        "genPrewarCounterfactualCapGbp",
        lambda: cap()["prewar_counterfactual_cap_gbp"],
        "{:,.0f}",
        FINDINGS,
    )
    emit(
        "genPrewarCounterfactualCapPreciseGbp",
        lambda: cap()["prewar_counterfactual_cap_gbp"],
        "{:,.2f}",
        FINDINGS,
    )
    # The two observations the solve is exactly identified on, and the modelled
    # values beside them. Emitting both halves is the point: a calibration that
    # claims to reproduce its targets should print the targets and the fit.
    emit(
        "genObservedCapJulGbp",
        lambda: capval()["observed_jul_2026_gbp"],
        "{:,.0f}",
        FINDINGS,
    )
    emit(
        "genModelledCapJulGbp",
        lambda: capval()["modelled_jul_2026_gbp"],
        "{:,.0f}",
        FINDINGS,
    )
    emit(
        "genObservedCapOctVatAdjustedGbp",
        lambda: capval()["observed_oct_2026_vat_adjusted_gbp"],
        "{:,.0f}",
        FINDINGS,
    )
    emit(
        "genModelledCapOctGbp",
        lambda: capval()["modelled_oct_2026_gbp"],
        "{:,.0f}",
        FINDINGS,
    )
    # The published October cap and the VAT relief that has to be added back
    # before it is comparable with July: the relief is a tax change, not a
    # wholesale one, so leaving it in would have the shock look smaller.
    emit(
        "genObservedCapOctPublishedGbp",
        lambda: scen_module().OFGEM_CAP_OCT_2026_GBP,
        "{:,.0f}",
        "uk_iran_conflict.scenarios",
    )
    emit(
        "genOctElectricityVatReliefGbp",
        lambda: scen_module().OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP,
        "{:,.0f}",
        "uk_iran_conflict.scenarios",
    )

    def _cap_fit_error() -> float:
        """Largest absolute miss across the two cap observations, in pounds."""
        v = capval()
        return max(
            abs(v["modelled_jul_2026_gbp"] - v["observed_jul_2026_gbp"]),
            abs(v["modelled_oct_2026_gbp"] - v["observed_oct_2026_vat_adjusted_gbp"]),
        )

    emit("genCapFitMaxErrorGbp", _cap_fit_error, "{:.2f}", FINDINGS)
    emit(
        "genCapFitReproducesBoth",
        lambda: "does" if _cap_fit_error() < 1e-6 else "does not",
        source=FINDINGS,
    )
    # How far each published cap sits above the solved counterfactual. The
    # October move is the anchor; the July cap already carried a large part of
    # the shock, which is exactly why it could never have been the baseline.
    emit(
        "genCapAnchorMovePct", lambda: 100 * cap()["cap_anchor_pct"], "{:.1f}", FINDINGS
    )
    emit("genCapBaseMovePct", lambda: 100 * cap()["cap_base_pct"], "{:.1f}", FINDINGS)
    emit(
        "genSteadyStateDualFuelPct",
        lambda: 100 * capval()["steady_state_dual_fuel_pct"],
        "{:.1f}",
        FINDINGS,
    )
    emit(
        "genGasSustainedFractionPrecisePct",
        lambda: 100 * cap()["sustained_fraction"],
        "{:.1f}",
        FINDINGS,
    )
    emit(
        "genGasSustainedFractionValue",
        lambda: cap()["sustained_fraction"],
        "{:.3f}",
        FINDINGS,
    )
    # The superseded calibration, named so the paper can show its own working.
    emit(
        "genSupersededBaselineCapGbp",
        lambda: cap()["superseded"]["baseline_cap_gbp"],
        "{:,.0f}",
        FINDINGS,
    )
    emit(
        "genSupersededCapAnchorPct",
        lambda: 100 * cap()["superseded"]["cap_anchor_pct"],
        "{:.1f}",
        FINDINGS,
    )

    # --- the phase-in profile: real observation windows, not a linear ramp ---
    #
    # ``cap_phase_in_profile`` used to slide a linear ramp across the quarters.
    # It now builds each cap period's actual forward-curve observation window
    # and averages the monthly gas-peak profile over it. The result is
    # non-monotone -- it peaks at 2026Q4 and falls back in 2027Q1, because the
    # window that sets the 2027Q1 cap is already past the peak -- and a linear
    # ramp cannot produce that shape at all.
    def phase_profile() -> list[float]:
        return list(cap()["phase_in_profile"])

    def quarter_labels() -> list[str]:
        return list(cap()["quarter_labels"])

    for i, word in enumerate(DECILE_WORD[:5]):
        emit(
            f"genCapPhaseInQuarter{word}",
            lambda i=i: phase_profile()[i],
            "{:.3f}",
            FINDINGS,
        )
        emit(
            f"genCapPhaseInQuarter{word}Label",
            lambda i=i: quarter_labels()[i],
            source=FINDINGS,
        )
    emit(
        "genCapPhaseInProfileText",
        lambda: ", ".join(f"{p:.3f}" for p in phase_profile()),
        source=FINDINGS,
    )
    emit(
        "genCapPhaseInQuarterCount",
        lambda: number_word(len(phase_profile())),
        source=FINDINGS,
    )
    emit("genCapPhaseInPeak", lambda: max(phase_profile()), "{:.3f}", FINDINGS)
    emit(
        "genCapPhaseInPeakQuarter",
        lambda: quarter_labels()[phase_profile().index(max(phase_profile()))],
        source=FINDINGS,
    )
    emit(
        "genCapPhaseInIsMonotone",
        lambda: "is" if phase_profile() == sorted(phase_profile()) else "is not",
        source=FINDINGS,
    )
    emit(
        "genCapPhaseInMonotoneWord",
        lambda: (
            "monotone" if phase_profile() == sorted(phase_profile()) else "non-monotone"
        ),
        source=FINDINGS,
    )
    emit(
        "genCapPhaseInFallsAtQuarter",
        lambda: next(
            (
                quarter_labels()[i + 1]
                for i in range(len(phase_profile()) - 1)
                if phase_profile()[i + 1] < phase_profile()[i]
            ),
            None,
        ),
        source=FINDINGS,
    )
    emit(
        "genCapPhaseInZeroQuarterCount",
        lambda: number_word(sum(1 for p in phase_profile() if p == 0)),
        source=FINDINGS,
    )

    # --- the named pass-through coefficients ---------------------------
    #
    # The manuscript states a wholesale share of the bill of about 0.45 and a
    # marginal-pricing share of 0.85, and calls them phi and psi. Neither is the
    # coefficient the model actually applies. The bill-level pass-through is the
    # product of the wholesale share and the share of the wholesale cost the cap
    # reprices; the electricity-to-gas ratio is what the marginal-pricing share
    # becomes once electricity's smaller wholesale cost share is folded in. Both
    # are computed, not asserted, and both are emitted here under names that say
    # what they are rather than which Greek letter used to stand for them.
    emit(
        "genBillLevelPassThrough",
        lambda: cap()["bill_level_pass_through"],
        "{:.4f}",
        FINDINGS,
    )
    emit(
        "genBillLevelPassThroughPct",
        lambda: 100 * cap()["bill_level_pass_through"],
        "{:.2f}",
        FINDINGS,
    )
    emit(
        "genElecToGasPassThroughRatio",
        lambda: cap()["electricity_to_gas_pass_through_ratio"],
        "{:.4f}",
        FINDINGS,
    )
    emit(
        "genElecToGasPassThroughRatioPct",
        lambda: 100 * cap()["electricity_to_gas_pass_through_ratio"],
        "{:.1f}",
        FINDINGS,
    )

    # --- applied versus steady-state price changes ----------------------
    #
    # Round-2 referees, item 5: the paper quoted the steady-state retail unit
    # rate changes and applied smaller ones. Both are now emitted, under names
    # that cannot be confused. The applied factors are what multiplies the
    # household's bill over the modelled window; the steady-state ones are what
    # the cap would settle at if the peak were sustained indefinitely.
    for tag, key in (
        ("Gas", "annual_applied_gas_factor"),
        ("Elec", "annual_applied_electricity_factor"),
    ):
        emit(f"genApplied{tag}Factor", lambda key=key: cap()[key], "{:.4f}", FINDINGS)
        emit(
            f"genApplied{tag}UnitRatePct",
            lambda key=key: 100 * (cap()[key] - 1),
            "{:.1f}",
            FINDINGS,
        )
    for tag, key in (
        ("Gas", "steady_state_gas_factor"),
        ("Elec", "steady_state_electricity_factor"),
    ):
        emit(
            f"genSteadyState{tag}Factor", lambda key=key: cap()[key], "{:.4f}", FINDINGS
        )
        emit(
            f"genSteadyState{tag}UnitRatePct",
            lambda key=key: 100 * (cap()[key] - 1),
            "{:.1f}",
            FINDINGS,
        )
    emit(
        "genAnnualPhaseInGasFraction",
        lambda: cap()["annual_phase_in_gas"],
        "{:.3f}",
        FINDINGS,
    )

    # ==================================================================
    # ROUND 3: the gradient, and which gradient
    # ==================================================================
    #
    # The headline D1/D10 ratio rests on decile one, where a fifth of households
    # have non-positive equivalised income. Three companions are now emitted on
    # every run so no sentence has to choose one silently: the ratio measured
    # from decile two, and both ratios again with the burden winsorised.
    emit(
        "genDecileRatioFromDecileTwoPct",
        lambda: jload(central)["d2_d10_ratio_pct"],
        "{:.2f}",
        central,
    )
    emit(
        "genDecileRatioFromDecileTwoGbp",
        lambda: jload(central)["d2_d10_ratio_gbp"],
        "{:.2f}",
        central,
    )
    emit(
        "genDecileRatioWinsorisedPct",
        lambda: jload(central)["d1_d10_ratio_pct_winsorised"],
        "{:.2f}",
        central,
    )
    emit(
        "genDecileRatioFromDecileTwoWinsorisedPct",
        lambda: jload(central)["d2_d10_ratio_pct_winsorised"],
        "{:.2f}",
        central,
    )
    for tag, key in (
        ("One", "decile1_loss_pct_winsorised"),
        ("Two", "decile2_loss_pct_winsorised"),
        ("Ten", "decile10_loss_pct_winsorised"),
    ):
        emit(
            f"genDecile{tag}LossPctWinsorised",
            lambda key=key: jload(central)[key],
            "{:.2f}",
            central,
        )
    emit(
        "genMeanLossPctWinsorised",
        lambda: jload(central)["mean_loss_pct_winsorised"],
        "{:.2f}",
        central,
    )
    emit(
        "genIncomeTreatment",
        lambda: str(jload(central)["income_treatment"]),
        source=central,
    )
    for tag, d in (("Two", 2),):
        emit(
            f"genDecile{tag}LossGbp",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_gbp"],
            "{:.0f}",
            central,
        )
        emit(
            f"genDecile{tag}LossPct",
            lambda d=d: decile_row(jload(central), "decile", d)["mean_loss_pct"],
            "{:.2f}",
            central,
        )
    # How far the tail treatment moves the headline gradient, in both
    # directions: winsorising *raises* the D1/D10 ratio and barely touches the
    # D2/D10 one, which is the cleanest statement that the gradient is a
    # decile-one phenomenon rather than an outlier artefact.
    emit(
        "genDecileRatioWinsorisedShiftPct",
        lambda: (
            100
            * (
                jload(central)["d1_d10_ratio_pct_winsorised"]
                / jload(central)["all_channel_d1_d10_ratio_pct"]
                - 1
            )
        ),
        "{:.0f}",
        central,
    )

    # --- which income concept the ranking is on ------------------------
    #
    # Round-2 referee 3, item 11, resolved: the ranking variable is equivalised
    # BHC income, person-weighted, and the burden denominator is equivalised
    # AHC. They agree 99.9 per cent of the time with the former and 56.1 per
    # cent with the latter, so the paper's gradient is a cross-concept
    # statistic: households are sorted on one income concept and their burden is
    # measured against another. That is a normal convention and it is also a
    # thing the paper has to say out loud.
    def concept_findings() -> dict:
        return jload(FINDINGS)["decile_concept"]

    def _pretty_concept(key: str) -> str:
        return {
            "equivalised_bhc": "equivalised BHC",
            "equivalised_ahc": "equivalised AHC",
            "unequivalised_bhc": "unequivalised BHC",
        }.get(str(key), str(key).replace("_", " "))

    emit(
        "genDecileRankConcept",
        lambda: _pretty_concept(concept_findings()["documented_truth"]),
        source=FINDINGS,
    )
    emit(
        "genDecileRankBestMatch",
        lambda: _pretty_concept(concept_findings()["best_match"]),
        source=FINDINGS,
    )
    emit(
        "genDecileRankBurdenDenominator",
        lambda: _pretty_concept(concept_findings()["burden_denominator"]),
        source=FINDINGS,
    )
    emit(
        "genDecileRankMatchesBurdenDenominator",
        lambda: (
            "does" if concept_findings()["matches_burden_denominator"] else "does not"
        ),
        source=FINDINGS,
    )
    emit(
        "genDecileRankIsCrossConcept",
        lambda: "is not" if concept_findings()["matches_burden_denominator"] else "is",
        source=FINDINGS,
    )
    emit(
        "genDecileRankWeighting",
        lambda: (
            "person-weighted"
            if concept_findings()["person_weighted"]
            else "household-weighted"
        ),
        source=FINDINGS,
    )
    emit(
        "genDecileRankDocumentedTruthConfirmed",
        lambda: (
            "does"
            if concept_findings()["best_match_is_documented_truth"]
            else "does not"
        ),
        source=FINDINGS,
    )
    for tag, key in (
        ("EquivBhc", "equivalised_bhc"),
        ("EquivAhc", "equivalised_ahc"),
        ("Unequiv", "unequivalised_bhc"),
    ):
        emit(
            f"genDecileRankAgreement{tag}PrecisePct",
            lambda key=key: 100 * concept_findings()["agreement"][key],
            "{:.1f}",
            FINDINGS,
        )
        emit(
            f"genDecileRankAgreement{tag}HouseholdPct",
            lambda key=key: (
                100 * concept_findings()["household_weighted_agreement"][key]
            ),
            "{:.1f}",
            FINDINGS,
        )
        emit(
            f"genDecileRankGap{tag}",
            lambda key=key: concept_findings()["mean_absolute_decile_gap"][key],
            "{:.2f}",
            FINDINGS,
        )
    emit(
        "genDecileRankNegativeSentinelSharePct",
        lambda: 100 * concept_findings()["negative_sentinel_share"],
        "{:.2f}",
        FINDINGS,
    )

    # ==================================================================
    # ROUND 3: what the means-tested system reaches
    # ==================================================================
    #
    # The paper's targeting claim scales inversely with the means-tested share
    # of the aggregate loss, and that share is the single most calibration-
    # dependent quantity in the policy section. Under the raw PolicyEngine
    # imputation, means-tested households buy a sixth as much motor fuel as
    # everyone else, which is the imputation's artefact rather than a fact about
    # driving. Two corrections are run: within-decile fuel parity, and an
    # NTS-calibrated participation margin. Both roughly double the share.
    def mt() -> dict:
        return jload(FINDINGS)["means_tested_fuel"]

    MT_SPECS = (
        ("Raw", "raw"),
        ("Parity", "mt_fuel_parity"),
        ("Participation", "nts_participation"),
        ("OnsShape", "ons_fuel_shape"),
        ("OnsLevels", "ons_both_levels"),
    )
    for tag, key in MT_SPECS:
        emit(
            f"genMeansTestedShareOfLoss{tag}Pct",
            lambda key=key: 100 * mt()[key]["means_tested_share_of_loss"],
            "{:.2f}",
            FINDINGS,
        )
        emit(
            f"genMeansTestedFuelSpend{tag}Gbp",
            lambda key=key: mt()[key]["means_tested_mean_fuel_gbp"],
            "{:,.0f}",
            FINDINGS,
        )
        emit(
            f"genNonMeansTestedFuelSpend{tag}Gbp",
            lambda key=key: mt()[key]["non_means_tested_mean_fuel_gbp"],
            "{:,.0f}",
            FINDINGS,
        )
        emit(
            f"genMeansTestedFuelRatio{tag}",
            lambda key=key: (
                mt()[key]["non_means_tested_mean_fuel_gbp"]
                / mt()[key]["means_tested_mean_fuel_gbp"]
            ),
            "{:.2f}",
            FINDINGS,
        )
    # The plain names, for the main specification.
    emit(
        "genMeansTestedShareOfLossPct",
        lambda: 100 * mt()["raw"]["means_tested_share_of_loss"],
        "{:.2f}",
        FINDINGS,
    )
    emit(
        "genMeansTestedFuelRatio",
        lambda: (
            mt()["raw"]["non_means_tested_mean_fuel_gbp"]
            / mt()["raw"]["means_tested_mean_fuel_gbp"]
        ),
        "{:.2f}",
        FINDINGS,
    )

    def targeting() -> dict:
        return mt()["implied_targeting_multiple"]

    emit(
        "genMeansTestedTargetingFactor",
        lambda: targeting()["factor"],
        "{:.2f}",
        FINDINGS,
    )
    # ------------------------------------------------------------------
    # ROUND 4: the eligibility-over-generosity multiple, computed
    # ------------------------------------------------------------------
    #
    # This block used to open with ``STATED_TARGETING_MULTIPLE = 7.0`` and a
    # source string reading "the pre-round-3 manuscript's own targeting claim".
    # Every corrected multiple below was that literal rescaled by the ratio of
    # means-tested loss shares, so the paper's central policy claim -- and the
    # range in its abstract -- was anchored on a number no run produced. It also
    # falsified the appendix's guarantee that every headline is emitted
    # mechanically from ``results/``. Round-4 referees, item 1.
    #
    # The multiple is now read straight off the live envelope table. It is the
    # ratio the eligibility argument actually rests on:
    #
    #     (loss offset when the envelope buys ELIGIBILITY, at the sponsor's own
    #      generosity)  /  (loss offset when the instrument is run to its own
    #      feasible ceiling on the population it can already see)
    #
    # Both rows are outputs of the same run, so the multiple moves whenever the
    # run does. It is defined only for the two means-tested schemes, because
    # only they have an eligibility-widening row at all.
    TARGETING_POLICIES = (
        ("SocialTariff", "social_tariff"),
        ("Whd", "whd_expansion"),
    )

    def _targeting_multiple(policy: str) -> float:
        """Eligibility offset over feasible-maximum offset, from the live rows."""
        widened = snum(
            envelope_row(policy, ENVELOPE_ELIGIBILITY),
            "share_of_aggregate_loss_offset",
        )
        ceiling = snum(
            envelope_row(policy, ENVELOPE_FEASIBLE_MAX),
            "share_of_aggregate_loss_offset",
        )
        if ceiling <= 0:
            raise ValueError(f"{policy}: feasible-maximum offset is not positive")
        return widened / ceiling

    def _targeting_multiples() -> list[float]:
        return [_targeting_multiple(policy) for _tag, policy in TARGETING_POLICIES]

    for tag, policy in TARGETING_POLICIES:
        emit(
            f"genTargetingMultiple{tag}",
            lambda policy=policy: _targeting_multiple(policy),
            "{:.1f}",
            ENVELOPE,
        )
    # The range across the instruments that have the comparison, which is what
    # a sentence quoting "a factor of" should carry. The old literal 7 sat
    # outside this range at both ends.
    for name, select in (
        ("genTargetingMultipleMin", min),
        ("genTargetingMultipleMax", max),
    ):
        emit(
            name,
            lambda select=select: select(_targeting_multiples()),
            "{:.1f}",
            ENVELOPE,
        )
    #: The name the manuscript already carries for "the multiple before any
    #: fuel-imputation correction". It is now the *computed* main-specification
    #: multiple rather than the literal 7, so every sentence built on it moves
    #: with the run. The social tariff is the instrument the claim is made
    #: about, so it is the one this name resolves to.
    emit(
        "genTargetingMultipleStated",
        lambda: _targeting_multiple("social_tariff"),
        "{:.1f}",
        ENVELOPE,
    )
    # The two fuel-imputation corrections. The rescaling itself is unchanged and
    # is the right operation -- the multiple scales inversely with the share of
    # the aggregate loss the means-tested population is imputed to carry -- but
    # it is now applied to a computed multiple instead of to a literal.
    for tag, key in (
        ("Parity", "mt_fuel_parity"),
        ("Participation", "nts_participation"),
    ):
        emit(
            f"genTargetingMultipleUnder{tag}",
            lambda key=key: (
                _targeting_multiple("social_tariff")
                * mt()["raw"]["means_tested_share_of_loss"]
                / mt()[key]["means_tested_share_of_loss"]
            ),
            "{:.1f}",
            f"{ENVELOPE} + {FINDINGS}",
        )
        # The same correction on each instrument, so a sentence that quotes the
        # Warm Home Discount's multiple can quote its correction too.
        for ptag, policy in TARGETING_POLICIES:
            emit(
                f"genTargetingMultiple{ptag}Under{tag}",
                lambda key=key, policy=policy: (
                    _targeting_multiple(policy)
                    * mt()["raw"]["means_tested_share_of_loss"]
                    / mt()[key]["means_tested_share_of_loss"]
                ),
                "{:.1f}",
                f"{ENVELOPE} + {FINDINGS}",
            )
        # ... and the range of the corrected multiple across the instruments.
        for suffix, select in (("Min", min), ("Max", max)):
            emit(
                f"genTargetingMultipleUnder{tag}{suffix}",
                lambda key=key, select=select: select(
                    m
                    * mt()["raw"]["means_tested_share_of_loss"]
                    / mt()[key]["means_tested_share_of_loss"]
                    for m in _targeting_multiples()
                ),
                "{:.1f}",
                f"{ENVELOPE} + {FINDINGS}",
            )

    # The participation margin: how much of the fuel gap is households that do
    # not drive at all, rather than households that drive less.
    def margins() -> dict:
        return jload(FINDINGS)["motor_fuel_margins"]

    for tag, key in (("Raw", "raw"), ("Participation", "nts_participation")):
        emit(
            f"genZeroFuelShareOverall{tag}Pct",
            lambda key=key: 100 * margins()[key]["zero_fuel_share_overall"],
            "{:.1f}",
            FINDINGS,
        )
        emit(
            f"genZeroFuelShareMeansTested{tag}Pct",
            lambda key=key: 100 * margins()[key]["means_tested_zero_share"],
            "{:.1f}",
            FINDINGS,
        )
        # ROUND 4, item 6: the raw gradient is 0.0157 percentage points and
        # rendered as "0.0" at one decimal place, which reads as an absent
        # number rather than a tiny one. It is recomputed here from the
        # by-decile array rather than read off the stored scalar, so the units
        # are unambiguous (the array is a share; the difference is scaled to
        # percentage points here), and printed at a precision that survives the
        # raw specification as well as the participation one.
        emit(
            f"genZeroFuelShareGradient{tag}Pp",
            lambda key=key: (
                100
                * (
                    margins()[key]["zero_fuel_share_by_decile"][0]
                    - margins()[key]["zero_fuel_share_by_decile"][9]
                )
            ),
            "{:.2f}",
            FINDINGS,
        )
        # The same gradient as stored, so a discrepancy between the stored
        # scalar and the array it summarises would surface as two macros that
        # disagree rather than as silence.
        emit(
            f"genZeroFuelShareGradient{tag}StoredPp",
            lambda key=key: margins()[key]["zero_share_d1_minus_d10_pp"],
            "{:.2f}",
            FINDINGS,
        )
        # Decile one and decile ten themselves, so a sentence quoting a
        # near-zero gradient can show the two levels it is the difference of.
        for dtag, index in (("DecileOne", 0), ("DecileTen", 9)):
            emit(
                f"genZeroFuelShare{dtag}{tag}Pct",
                lambda key=key, index=index: (
                    100 * margins()[key]["zero_fuel_share_by_decile"][index]
                ),
                "{:.1f}",
                FINDINGS,
            )

    # ==================================================================
    # ROUND 3: the Resolution Foundation comparison, like for like
    # ==================================================================
    #
    # Round-2 referees, item 1: the two legs were annualised over different
    # windows and the sum was labelled 2026, which broke the comparison with the
    # Resolution Foundation's calendar-2026 £11bn. Both legs now have a real
    # calendar-2026 run, and so does the peak-fuel upper bound, so the paper can
    # quote a bracket on the same window rather than a point on a different one.
    RF_WINDOW = (
        ("RfBracketLow", "calendar_2026"),
        ("RfBracketHigh", "peak_fuel_calendar_2026"),
    )
    for tag, variant in RF_WINDOW:
        emit(
            f"gen{tag}Bn",
            lambda variant=variant: float(
                crow(COMPARISON, "variant", variant)["aggregate_cost_bn"]
            ),
            "{:.2f}",
            COMPARISON,
        )
        emit(
            f"gen{tag}MeanGbp",
            lambda variant=variant: float(
                crow(COMPARISON, "variant", variant)["mean_loss_gbp"]
            ),
            "{:.0f}",
            COMPARISON,
        )
        emit(
            f"gen{tag}FuelSharePct",
            lambda variant=variant: (
                100
                * float(
                    crow(COMPARISON, "variant", variant)["motor_fuel_share_of_loss"]
                )
            ),
            "{:.1f}",
            COMPARISON,
        )
    # The Resolution Foundation's published calendar-2026 figure. An external
    # source, carried like the ONS benchmarks and cited as such.
    RF_COMPARATOR_BN = 11.0
    emit(
        "genRfComparatorBn",
        RF_COMPARATOR_BN,
        "{:.0f}",
        "Resolution Foundation, calendar 2026 (published)",
    )

    def _rf_bracket() -> tuple[float, float]:
        low = float(crow(COMPARISON, "variant", "calendar_2026")["aggregate_cost_bn"])
        high = float(
            crow(COMPARISON, "variant", "peak_fuel_calendar_2026")["aggregate_cost_bn"]
        )
        return low, high

    emit(
        "genRfBracketContainsComparator",
        lambda: (
            "does"
            if _rf_bracket()[0] <= RF_COMPARATOR_BN <= _rf_bracket()[1]
            else "does not"
        ),
        source=COMPARISON,
    )
    emit(
        "genRfBracketWidthBn",
        lambda: _rf_bracket()[1] - _rf_bracket()[0],
        "{:.2f}",
        COMPARISON,
    )
    emit(
        "genRfComparatorPositionPct",
        lambda: (
            100
            * (RF_COMPARATOR_BN - _rf_bracket()[0])
            / (_rf_bracket()[1] - _rf_bracket()[0])
        ),
        "{:.0f}",
        COMPARISON,
    )
    # The calendar-window run also replaces the arithmetic re-annualisation the
    # emitter used to do in-process. Both are kept: the run is the number, the
    # arithmetic is the check.
    emit(
        "genCalendarRunAggBn",
        lambda: float(
            crow(COMPARISON, "variant", "calendar_2026")["aggregate_cost_bn"]
        ),
        "{:.2f}",
        COMPARISON,
    )
    emit(
        "genCalendarRunMotorFuelSharePct",
        lambda: (
            100
            * float(
                crow(COMPARISON, "variant", "calendar_2026")["motor_fuel_share_of_loss"]
            )
        ),
        "{:.1f}",
        COMPARISON,
    )

    # ==================================================================
    # ROUND 3: the identification fragility that actually matters
    # ==================================================================
    #
    # ``GAS_PEAK_MONTHLY_PROFILE`` is the monthly shape of the wholesale gas
    # peak. It is a calibration, it was never swept, and it turns out to carry
    # the whole domestic leg: shifting it by a month either way, or flattening
    # it, moves the solved counterfactual cap over a £230 range, the aggregate
    # over £4bn, and the motor-fuel share from a bare majority to four fifths.
    # Two of the ten cells do not identify a counterfactual at all. This, not
    # the elasticity sweep, is the paper's real fragility.
    GAS_PROFILE = "gas_peak_monthly_profile"

    def profile_rows(identified_only: bool = True) -> list[dict]:
        rows = [r for r in cload(DOMESTIC_LEG) if r["parameter"].strip() == GAS_PROFILE]
        if identified_only:
            rows = [r for r in rows if is_identified(r)]
        if not rows:
            raise KeyError(f"{DOMESTIC_LEG}: no {GAS_PROFILE} rows")
        return rows

    def profile_span(column: str, select, scale: float = 1.0):
        return scale * select(float(r[column]) for r in profile_rows())

    for name, column, fmt, scale in (
        ("genGasProfileCapMinGbp", "prewar_counterfactual_cap_gbp", "{:,.0f}", 1),
        ("genGasProfileCapMaxGbp", "prewar_counterfactual_cap_gbp", "{:,.0f}", 1),
        ("genGasProfileAggMinBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genGasProfileAggMaxBn", "aggregate_cost_bn", "{:.2f}", 1),
        ("genGasProfileMeanMinGbp", "mean_loss_gbp", "{:.0f}", 1),
        ("genGasProfileMeanMaxGbp", "mean_loss_gbp", "{:.0f}", 1),
        ("genGasProfileFuelShareMinPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genGasProfileFuelShareMaxPct", "motor_fuel_share_of_loss", "{:.1f}", 100),
        ("genGasProfileSustainedMin", "sustained_fraction", "{:.3f}", 1),
        ("genGasProfileSustainedMax", "sustained_fraction", "{:.3f}", 1),
        ("genGasProfileDecileOnePctMin", "decile1_loss_pct", "{:.2f}", 1),
        ("genGasProfileDecileOnePctMax", "decile1_loss_pct", "{:.2f}", 1),
    ):
        select = min if "Min" in name else max
        emit(
            name,
            lambda column=column, select=select, scale=scale: profile_span(
                column, select, scale
            ),
            fmt,
            DOMESTIC_LEG,
        )
    emit(
        "genGasProfileCellCount",
        lambda: number_word(len(profile_rows(identified_only=False))),
        source=DOMESTIC_LEG,
    )
    emit(
        "genGasProfileIdentifiedCount",
        lambda: number_word(len(profile_rows())),
        source=DOMESTIC_LEG,
    )
    emit(
        "genGasProfileNonIdentifiedCount",
        lambda: number_word(
            len(profile_rows(identified_only=False)) - len(profile_rows())
        ),
        source=DOMESTIC_LEG,
    )
    emit(
        "genGasProfileAggSpreadPct",
        lambda: (
            100
            * (
                profile_span("aggregate_cost_bn", max)
                - profile_span("aggregate_cost_bn", min)
            )
            / float(jload(central)["aggregate_cost_bn"])
        ),
        "{:.0f}",
        DOMESTIC_LEG,
    )
    emit(
        "genGasProfileCapSpreadGbp",
        lambda: (
            profile_span("prewar_counterfactual_cap_gbp", max)
            - profile_span("prewar_counterfactual_cap_gbp", min)
        ),
        "{:,.0f}",
        DOMESTIC_LEG,
    )
    # Does the motor-fuel majority survive the sweep in both directions? It
    # does not: the majority is not a robust fact about the shock, it is a fact
    # about this monthly profile.
    emit(
        "genGasProfileFuelMajorityAlways",
        lambda: (
            "does"
            if profile_span("motor_fuel_share_of_loss", min) > 0.5
            else "does not"
        ),
        source=DOMESTIC_LEG,
    )
    # The whole sweep, not only the gas-profile block, now that the other four
    # parameters are read past their blank cells.
    emit(
        "genDomesticLegNonIdentifiedCount",
        lambda: number_word(
            sum(1 for r in cload(DOMESTIC_LEG) if not is_identified(r))
        ),
        source=DOMESTIC_LEG,
    )
    emit(
        "genDomesticLegCellCount",
        lambda: number_word(len(cload(DOMESTIC_LEG))),
        source=DOMESTIC_LEG,
    )

    # ==================================================================
    # ROUND 3: the scorecard's fifth row type and its diagnostics
    # ==================================================================
    emit(
        "genEnvelopeRowKindCount",
        lambda: number_word(len(diagnostics()["row_semantics"])),
        source="policy_diagnostics.json",
    )
    for row_kind, tag in ENVELOPE_ROW_TAGS:
        emit(
            f"genEnvelopeSemantics{tag}",
            lambda row_kind=row_kind: ENVELOPE_ROW_SEMANTICS[row_kind],
            source="analysis/emit_tex_values.py:ENVELOPE_ROW_SEMANTICS",
        )

    def diag(policy: str) -> dict:
        return diagnostics()["by_policy"][policy]

    for policy, tag in POLICY_MACRO.items():
        # The instrument's OWN ceiling, with no envelope applied. This is the
        # number the old "feasible maximum" column claimed to be and was not:
        # the JRF block reaches £21.9bn, more than four times the envelope.
        emit(
            f"gen{tag}FeasibleMaxCostBn",
            lambda policy=policy: diag(policy)["feasible_max_cost_bn"],
            "{:.2f}",
            "policy_diagnostics.json",
        )
        emit(
            f"gen{tag}AbsorbableWithinEnvelopeBn",
            lambda policy=policy: diag(policy)["absorbable_within_envelope_bn"],
            "{:.2f}",
            "policy_diagnostics.json",
        )
        emit(
            f"gen{tag}SaturatesBelowEnvelope",
            lambda policy=policy: (
                "does" if diag(policy)["saturates_below_envelope"] else "does not"
            ),
            source="policy_diagnostics.json",
        )
        # What the instrument leaves unspent when it cannot reach the envelope.
        emit(
            f"gen{tag}EnvelopeShortfallBn",
            lambda policy=policy: max(
                0.0,
                diagnostics()["envelope_bn"]
                - diag(policy)["absorbable_within_envelope_bn"],
            ),
            "{:.2f}",
            "policy_diagnostics.json",
        )
        emit(
            f"gen{tag}FeasibleMaxOverEnvelope",
            lambda policy=policy: (
                diag(policy)["feasible_max_cost_bn"] / diagnostics()["envelope_bn"]
            ),
            "{:.2f}",
            "policy_diagnostics.json",
        )
        emit(
            f"gen{tag}StatedCostSimulatedBn",
            lambda policy=policy: diag(policy)["stated_cost_simulated_bn"],
            "{:.2f}",
            "policy_diagnostics.json",
        )
        emit(
            f"gen{tag}IsMeansTested",
            lambda policy=policy: "is" if diag(policy)["means_tested"] else "is not",
            source="policy_diagnostics.json",
        )
        # The two arms of the common-envelope comparison, and whether they in
        # fact spend the same. For three of the five they do not, so a sentence
        # that says "at the same £5bn" is wrong about them.
        arms = lambda policy=policy: diag(policy)["envelope_arms"]  # noqa: E731
        emit(
            f"gen{tag}GenerosityArmSpendBn",
            lambda arms=arms: arms()["generosity_arm_spend_bn"],
            "{:.2f}",
            "policy_diagnostics.json",
        )
        emit(
            f"gen{tag}ArmsSpendTheSame",
            lambda arms=arms: "do" if arms()["arms_spend_the_same"] else "do not",
            source="policy_diagnostics.json",
        )
        # The eligibility arm exists only where eligibility is a margin at all:
        # widening the eligibility of a universal instrument is not a thing you
        # can do. Emitting these only for the means-tested schemes gives a
        # narrower macro set rather than three \GENMISSING placeholders the
        # prose would then have to route around — the same convention
        # :func:`envelope_has` follows for the absent CSV rows.
        if diag(policy)["means_tested"]:
            emit(
                f"gen{tag}EligibilityArmSpendBn",
                lambda arms=arms: arms()["eligibility_arm_spend_bn"],
                "{:.2f}",
                "policy_diagnostics.json",
            )
            emit(
                f"gen{tag}EligibilityArmEligibleSharePct",
                lambda arms=arms: 100 * arms()["eligibility_arm_eligible_share"],
                "{:.1f}",
                "policy_diagnostics.json",
            )
            # Widening eligibility until the envelope is spent can take a
            # "means-tested" scheme all the way to universal. Where it does, the
            # comparison is no longer between a targeted and an untargeted
            # instrument, and the table has to say so.
            emit(
                f"gen{tag}EligibilityArmIsUniversal",
                lambda arms=arms: (
                    "is" if arms()["eligibility_arm_is_universal"] else "is not"
                ),
                source="policy_diagnostics.json",
            )
            emit(
                f"gen{tag}EligibilityArmUniversalWord",
                lambda arms=arms: (
                    "universal"
                    if arms()["eligibility_arm_is_universal"]
                    else "targeted"
                ),
                source="policy_diagnostics.json",
            )

        # --- the overcompensation family, at the sponsor's stated design ---
        #
        # The saturation result the referees called a scaling artefact is, at
        # bottom, about paying households more than they lost. These make that
        # testable rather than rhetorical.
        rel = f"{CENTRAL_SCENARIO}/{policy}.json"
        emit(
            f"gen{tag}OvercompensatedHouseholdsM",
            lambda rel=rel: jload(rel)["overcompensated_households_m"],
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{tag}OvercompensatedShareOfRecipientsPct",
            lambda rel=rel: 100 * jload(rel)["overcompensated_share_of_recipients"],
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{tag}OvercompensatedShareOfHouseholdsPct",
            lambda rel=rel: 100 * jload(rel)["overcompensated_share_of_households"],
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{tag}OvercompensatedExcessBn",
            lambda rel=rel: jload(rel)["overcompensated_excess_bn"],
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{tag}OvercompensatedShareOfSpendPct",
            lambda rel=rel: 100 * jload(rel)["overcompensated_share_of_spend"],
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{tag}GainToLossRatioRecipients",
            lambda rel=rel: jload(rel)["gain_to_loss_ratio_recipients"],
            "{:.2f}",
            rel,
        )
        emit(
            f"gen{tag}EligibleSharePctStated",
            lambda rel=rel: 100 * jload(rel)["eligible_share"],
            "{:.1f}",
            rel,
        )
        emit(
            f"gen{tag}EligibilityIsUniversalStated",
            lambda rel=rel: (
                "is" if jload(rel)["eligibility_is_universal"] else "is not"
            ),
            source=rel,
        )

    # How many instruments cannot reach the envelope at all, and how many go
    # universal once eligibility is the margin.
    def _saturating() -> list[str]:
        return [
            p
            for p in POLICY_MACRO
            if diagnostics()["by_policy"][p]["saturates_below_envelope"]
        ]

    def _universal_eligibility() -> list[str]:
        return [
            p
            for p in POLICY_MACRO
            if diagnostics()["by_policy"][p]["envelope_arms"].get(
                "eligibility_arm_is_universal"
            )
        ]

    emit(
        "genEnvelopeSaturatingCount",
        lambda: number_word(len(_saturating())),
        source="policy_diagnostics.json",
    )
    emit(
        "genEnvelopeSaturatingLabels",
        lambda: " and ".join(envelope_names[p] for p in _saturating()),
        source="policy_diagnostics.json",
    )
    emit(
        "genEnvelopeUniversalEligibilityCount",
        lambda: number_word(len(_universal_eligibility())),
        source="policy_diagnostics.json",
    )
    emit(
        "genEnvelopeUniversalEligibilityLabels",
        lambda: (
            " and ".join(envelope_names[p] for p in _universal_eligibility())
            or "none of them"
        ),
        source="policy_diagnostics.json",
    )
    emit(
        "genEnvelopeFeasibleMaxTotalBn",
        lambda: sum(
            diagnostics()["by_policy"][p]["feasible_max_cost_bn"] for p in POLICY_MACRO
        ),
        "{:.1f}",
        "policy_diagnostics.json",
    )
    # Best on the feasible-max rows: the comparison the paper should be making
    # when it asks which instrument can do most, as opposed to which can absorb
    # most of a fixed budget.
    if envelope_rows(ENVELOPE_FEASIBLE_MAX):
        emit(
            "genEnvelopeBestFeasibleMaxLabel",
            lambda: envelope_names[
                max(
                    envelope_rows(ENVELOPE_FEASIBLE_MAX),
                    key=lambda r: float(r["share_of_aggregate_loss_offset"]),
                )["policy"].strip()
            ],
            source=ENVELOPE,
        )
        emit(
            "genEnvelopeBestFeasibleMaxOffsetPct",
            lambda: (
                100
                * max(
                    float(r["share_of_aggregate_loss_offset"])
                    for r in envelope_rows(ENVELOPE_FEASIBLE_MAX)
                )
            ),
            "{:.1f}",
            ENVELOPE,
        )

    # --- the large-loser statistic, no longer a carried literal ---------
    def large_loser() -> dict:
        return diagnostics()["large_loser_outside_means_test"]

    emit(
        "genLargeLoserOutsideMeansTest",
        lambda: large_loser()["share_outside_means_test_pct"],
        "{:.0f}",
        "policy_diagnostics.json",
    )
    emit(
        "genLargeLoserOutsideMeansTestPrecise",
        lambda: large_loser()["share_outside_means_test_pct"],
        "{:.1f}",
        "policy_diagnostics.json",
    )
    # The statistic's ceiling, and the part of it that is actually about
    # targeting. It cannot fall below 100 minus the means-tested share of
    # households, so quoted alone it is very nearly a tautology.
    emit(
        "genLargeLoserCeilingPct",
        lambda: large_loser()["ceiling_pct"],
        "{:.1f}",
        "policy_diagnostics.json",
    )
    emit(
        "genLargeLoserHeadroomPct",
        lambda: large_loser()["headroom_pct"],
        "{:.1f}",
        "policy_diagnostics.json",
    )
    emit(
        "genLargeLosersM",
        lambda: large_loser()["large_losers_m"],
        "{:.2f}",
        "policy_diagnostics.json",
    )
    emit(
        "genLargeLoserThresholdPct",
        lambda: large_loser()["threshold_pct"],
        "{:.0f}",
        "policy_diagnostics.json",
    )
    emit(
        "genLargeLoserShareOfHouseholdsPct",
        lambda: 100 * large_loser()["large_loser_share_of_households"],
        "{:.1f}",
        "policy_diagnostics.json",
    )

    # --- the JRF block's reference quantity, on both bases --------------
    def jrf_ref() -> dict:
        return diagnostics()["jrf_reference_quantities"]

    emit(
        "genJrfReferenceBasis",
        lambda: str(jrf_ref()["basis_used"]).replace("_", " "),
        source="policy_diagnostics.json",
    )
    emit(
        "genJrfOfgemTypicalGbp",
        lambda: jrf_ref()["ofgem_typical_consumption_gbp"],
        "{:,.0f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfOfgemBlockGbp",
        lambda: jrf_ref()["ofgem_block_gbp"],
        "{:,.0f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfModelledMedianBillGbp",
        lambda: jrf_ref()["modelled_median_domestic_bill_gbp"],
        "{:,.0f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfModelledBlockGbp",
        lambda: jrf_ref()["modelled_median_block_gbp"],
        "{:,.0f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfModelledOverOfgemPct",
        lambda: 100 * jrf_ref()["modelled_over_ofgem"],
        "{:.1f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfBlockSharePct",
        lambda: 100 * jrf_ref()["block_share"],
        "{:.0f}",
        "policy_diagnostics.json",
    )

    # --- the JRF costing gap, decomposed --------------------------------
    #
    # ROUND 4. The gap used to be reported as unexplained. It is not: the
    # microdata can only make the block CHEAPER (a household whose whole bill
    # is under the block cannot use all of it), and the gap survives that
    # channel whole. What closes it is the discount RATE, which JRF do not
    # publish and this paper chose. Everything below is read from
    # ``jrf_costing_gap``; nothing here is a constant.
    DIAG = "policy_diagnostics.json"

    def jrf_gap() -> dict:
        return diagnostics()["jrf_costing_gap"]

    emit("genJrfGapModelledBn", lambda: jrf_gap()["modelled_cost_bn"], "{:.2f}", DIAG)
    emit("genJrfGapSponsorBn", lambda: jrf_gap()["sponsor_cost_bn"], "{:.1f}", DIAG)
    emit("genJrfGapBn", lambda: jrf_gap()["gap_bn"], "{:.2f}", DIAG)
    emit("genJrfGapRatio", lambda: jrf_gap()["ratio"], "{:.2f}", DIAG)
    emit(
        "genJrfUniversalCeilingBn",
        lambda: jrf_gap()["decomposition"]["universal_ceiling_total_bn"],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genJrfBlockTruncationBn",
        lambda: jrf_gap()["decomposition"]["block_truncation_bn"],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genJrfBlockTruncationSharePct",
        lambda: 100 * jrf_gap()["decomposition"]["block_truncation_share_of_ceiling"],
        "{:.1f}",
        DIAG,
    )
    emit(
        "genJrfHouseholdsM",
        lambda: jrf_gap()["per_household"]["households_m"],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genJrfSponsorPerHouseholdGbp",
        lambda: jrf_gap()["per_household"]["sponsor_implied_gbp"],
        "{:,.0f}",
        DIAG,
    )
    emit(
        "genJrfMechanicalPerHouseholdGbp",
        lambda: jrf_gap()["per_household"]["mechanical_full_block_gbp"],
        "{:,.0f}",
        DIAG,
    )
    emit(
        "genJrfModelledPerHouseholdGbp",
        lambda: jrf_gap()["per_household"]["modelled_gbp"],
        "{:,.0f}",
        DIAG,
    )
    emit(
        "genJrfImpliedDiscountPct",
        lambda: 100 * jrf_gap()["single_parameter_reconciliations"]["implied_discount"],
        "{:.1f}",
        DIAG,
    )
    emit(
        "genJrfModelledDiscountPct",
        lambda: (
            100 * jrf_gap()["single_parameter_reconciliations"]["modelled_discount"]
        ),
        "{:.0f}",
        DIAG,
    )
    emit(
        "genJrfImpliedBlockSharePct",
        lambda: (
            100 * jrf_gap()["single_parameter_reconciliations"]["implied_block_share"]
        ),
        "{:.1f}",
        DIAG,
    )
    emit(
        "genJrfImpliedEligibleSharePct",
        lambda: (
            100
            * jrf_gap()["single_parameter_reconciliations"][
                "implied_eligible_share_poorest_first"
            ]
        ),
        "{:.1f}",
        DIAG,
    )
    emit(
        "genJrfNettingWhdBn",
        lambda: jrf_gap()["netting_off_existing_support"][
            "warm_home_discount_modelled_bn"
        ],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genJrfNettingShareOfGapPct",
        lambda: (
            100
            * jrf_gap()["netting_off_existing_support"]["share_of_gap_it_could_close"]
        ),
        "{:.1f}",
        DIAG,
    )
    emit(
        "genJrfNettingClosesGap",
        lambda: (
            "does"
            if jrf_gap()["netting_off_existing_support"]["can_close_gap"]
            else "does not"
        ),
        source=DIAG,
    )

    # --- the two flat-payment ceilings, side by side --------------------
    #
    # The scorecard uses the bill-exhausting rule; an earlier draft described
    # the loss-exhausting one. Both are emitted so the prose can name the rule
    # it quotes and the number can be checked against it.
    def ceilings() -> dict:
        return diagnostics()["flat_payment_ceilings"]

    emit(
        "genWhdCeilingBillRuleGbp",
        lambda: ceilings()["mean_eligible_domestic_bill_gbp"],
        "{:,.0f}",
        DIAG,
    )
    emit(
        "genWhdCeilingLossRuleGbp",
        lambda: ceilings()["mean_eligible_loss_gbp"],
        "{:,.0f}",
        DIAG,
    )
    emit(
        "genWhdCeilingRuleRatio",
        lambda: ceilings()["bill_over_loss"],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genWhdCeilingBillRuleCostBn",
        lambda: ceilings()["cost_at_bill_rule_bn"],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genWhdCeilingLossRuleCostBn",
        lambda: ceilings()["cost_at_loss_rule_bn"],
        "{:.2f}",
        DIAG,
    )
    emit(
        "genWhdCeilingRuleUsed",
        lambda: str(ceilings()["rule_used"]).replace("_", " "),
        source=DIAG,
    )

    # --- admission rules for the widened-eligibility arm ----------------
    #
    # ROUND 4. The eligibility arm's default rule ranks every non-claimant on
    # equivalised AHC income, which is the observability this paper argues does
    # not exist. Four rules are now scored. The default is the WORST of them on
    # the offset, so the finding does not rest on the assumption; what the
    # assumption buys is targeting. Macro names carry no digits.
    ADMISSION_TAGS = (
        ("Ahc", "equivalised_ahc_income"),
        ("NetIncome", "unequivalised_net_income"),
        ("Bill", "highest_domestic_bill"),
        ("Random", "random"),
    )
    ADMISSION_METRICS = (
        ("EligibleSharePct", "eligible_share", "{:.1f}"),
        ("OffsetPct", "share_of_aggregate_loss_offset", "{:.1f}"),
        ("BottomThreePct", "share_to_bottom_three", "{:.1f}"),
        ("UncompensatedPct", "uncompensated_share_overall", "{:.1f}"),
    )

    def admission(policy: str) -> dict:
        return diagnostics()["by_policy"][policy]["admission_rules"]

    for policy_tag, policy_key in (
        ("SocialTariff", "social_tariff"),
        ("Whd", "whd_expansion"),
    ):
        for rule_tag, rule_key in ADMISSION_TAGS:
            for metric_tag, metric_key, fmt in ADMISSION_METRICS:
                emit(
                    f"gen{policy_tag}Admission{rule_tag}{metric_tag}",
                    lambda p=policy_key, r=rule_key, m=metric_key: (
                        100 * float(admission(p)["by_rule"][r][m])
                    ),
                    fmt,
                    DIAG,
                )
        emit(
            f"gen{policy_tag}AdmissionRuleCount",
            lambda p=policy_key: number_word(len(admission(p)["rules"])),
            source=DIAG,
        )
        emit(
            f"gen{policy_tag}AdmissionOffsetMinPct",
            lambda p=policy_key: (
                100 * admission(p)["range"]["share_of_aggregate_loss_offset"]["min"]
            ),
            "{:.1f}",
            DIAG,
        )
        emit(
            f"gen{policy_tag}AdmissionOffsetMaxPct",
            lambda p=policy_key: (
                100 * admission(p)["range"]["share_of_aggregate_loss_offset"]["max"]
            ),
            "{:.1f}",
            DIAG,
        )
        emit(
            f"gen{policy_tag}AdmissionBottomThreeMinPct",
            lambda p=policy_key: (
                100 * admission(p)["range"]["share_to_bottom_three"]["min"]
            ),
            "{:.1f}",
            DIAG,
        )
        emit(
            f"gen{policy_tag}AdmissionBottomThreeMaxPct",
            lambda p=policy_key: (
                100 * admission(p)["range"]["share_to_bottom_three"]["max"]
            ),
            "{:.1f}",
            DIAG,
        )
        emit(
            f"gen{policy_tag}AdmissionDefaultIsWorstOffset",
            lambda p=policy_key: (
                "is"
                if "share_of_aggregate_loss_offset"
                in admission(p)["default_rule_does_not_maximise"]
                else "is not"
            ),
            source=DIAG,
        )
        emit(
            f"gen{policy_tag}AdmissionUniversalUnderEveryRule",
            lambda p=policy_key: (
                "does"
                if all(
                    float(v["eligible_share"]) >= 0.999
                    for v in admission(p)["by_rule"].values()
                )
                else "does not"
            ),
            source=DIAG,
        )

    emit(
        "genSocialTariffAdmissionRandomSeedCount",
        lambda: number_word(
            len(
                admission("social_tariff")["by_rule"]["random"]["random_draws"]["seeds"]
            )
        ),
        source=DIAG,
    )
    emit(
        "genSocialTariffAdmissionRandomOffsetMinPct",
        lambda: (
            100 * admission("social_tariff")["by_rule"]["random"]["random_draws"]["min"]
        ),
        "{:.1f}",
        DIAG,
    )
    emit(
        "genSocialTariffAdmissionRandomOffsetMaxPct",
        lambda: (
            100 * admission("social_tariff")["by_rule"]["random"]["random_draws"]["max"]
        ),
        "{:.1f}",
        DIAG,
    )

    # ==================================================================
    # ROUND 3: quantities the prose asks for that are derived, not stored
    # ==================================================================
    #
    # Each is a ratio of two numbers that ARE stored, computed here rather than
    # transcribed. Nothing below introduces a constant.

    def total_households_m() -> float:
        """Households behind the headline mean, from the mean itself.

        ``mean_loss_gbp`` is the aggregate over the weighted household count, so
        inverting it recovers that count exactly. Summing the decile table
        instead undercounts by the households the decile ranking drops (the
        ``coverage`` block), which would inflate every per-household figure
        below by about one per cent.
        """
        shock = jload(central)
        mean = float(shock["mean_loss_gbp"])
        if mean <= 0:
            raise ValueError("no mean loss to invert")
        return float(shock["aggregate_cost_bn"]) * 1e9 / mean / 1e6

    emit("genTotalHouseholdsMFromMean", total_households_m, "{:.2f}", central)

    # --- the mean loss split by channel --------------------------------
    #
    # The channel shares are shares of the AGGREGATE, and the aggregate is the
    # mean times the household count, so the mean splits in exactly the same
    # proportions. £377 = £169 domestic + £208 motor fuel.
    def _channel_mean(*keys: str) -> float:
        shock = jload(central)
        return float(shock["mean_loss_gbp"]) * sum(float(shock[k]) for k in keys)

    emit(
        "genDomesticMeanLossGbp",
        lambda: _channel_mean("gas_share_of_loss", "electricity_share_of_loss"),
        "{:.0f}",
        central,
    )
    emit(
        "genMotorFuelMeanLossGbp",
        lambda: _channel_mean("motor_fuel_share_of_loss"),
        "{:.0f}",
        central,
    )
    emit(
        "genGasMeanLossGbp",
        lambda: _channel_mean("gas_share_of_loss"),
        "{:.0f}",
        central,
    )
    emit(
        "genElecMeanLossGbp",
        lambda: _channel_mean("electricity_share_of_loss"),
        "{:.0f}",
        central,
    )

    # --- the mean loss among means-tested households --------------------
    #
    # Their share of the loss over their share of households, times the overall
    # mean. Means-tested households are 15.7 per cent of households and carry
    # 5.2 per cent of the loss, so their mean loss is about a third of the
    # national one -- which is the whole reason the targeting claim is
    # fragile, and is emitted under every fuel-margin calibration for the same
    # reason.
    def _mt_mean_loss(variant: str = "main") -> float:
        shock = jload(central)
        mt_households = float(shock["means_tested_share"])
        if mt_households <= 0:
            raise ValueError("no means-tested households")
        row = crow(COMPARISON, "variant", variant)
        return (
            float(row["mean_loss_gbp"])
            * float(row["means_tested_share_of_loss"])
            / mt_households
        )

    emit("genMeansTestedMeanLossGbp", lambda: _mt_mean_loss(), "{:.0f}", COMPARISON)
    for tag, variant in (
        ("Parity", "mt_fuel_parity"),
        ("Participation", "nts_participation"),
    ):
        emit(
            f"genMeansTestedMeanLoss{tag}Gbp",
            lambda variant=variant: _mt_mean_loss(variant),
            "{:.0f}",
            COMPARISON,
        )
    emit(
        "genMeansTestedMeanLossRatio",
        lambda: _mt_mean_loss() / float(jload(central)["mean_loss_gbp"]),
        "{:.2f}",
        COMPARISON,
    )
    emit(
        "genMeansTestedHouseholdsMFromShare",
        lambda: float(jload(central)["means_tested_share"]) * total_households_m(),
        "{:.2f}",
        central,
    )

    # --- gain per ELIGIBLE household, not per household -----------------
    #
    # ``mean_gain_gbp`` on every scorecard row is averaged over ALL households,
    # including the ones the instrument never pays. For a scheme reaching a
    # sixth of the population that understates the payment by a factor of six,
    # and it is the payment -- not its dilution across non-recipients -- that
    # has to be compared with the loss to see the overcompensation. Both are
    # emitted, under names that say which denominator they carry.
    for policy, tag in POLICY_MACRO.items():
        for envelope, row_tag in ENVELOPE_ROW_TAGS:
            if not envelope_has(policy, envelope):
                continue
            row = envelope_row(policy, envelope)
            aliases = [row_tag, *ENVELOPE_ROW_ALIASES.get(envelope, ())]
            for alias in aliases:
                emit(
                    f"gen{tag}{alias}GainPerEligibleGbp",
                    lambda row=row: (
                        float(row["mean_gain_gbp"]) / float(row["eligible_share"])
                    ),
                    "{:.0f}",
                    ENVELOPE,
                )
                emit(
                    f"gen{tag}{alias}EligibleSharePct",
                    lambda row=row: 100 * float(row["eligible_share"]),
                    "{:.1f}",
                    ENVELOPE,
                )
                # The payment set against what the recipients actually lost:
                # above one is overcompensation on average, not merely in the
                # tail.
                emit(
                    f"gen{tag}{alias}GainPerEligibleOverLoss",
                    lambda row=row: (
                        float(row["mean_gain_gbp"])
                        / float(row["eligible_share"])
                        / float(jload(central)["mean_loss_gbp"])
                    ),
                    "{:.2f}",
                    ENVELOPE,
                )

    # --- the JRF block on both reference bases --------------------------
    #
    # JRF peg the discounted block to Ofgem typical consumption; the modelled
    # median domestic bill is materially below that basis, so a block set on it
    # is not the sponsor's block. Both are emitted to two decimals under the
    # names the prose uses, beside the rounded ones above.
    emit(
        "genJrfBlockOfgemBasisGbp",
        lambda: jrf_ref()["ofgem_block_gbp"],
        "{:,.2f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfBlockModelledGbp",
        lambda: jrf_ref()["modelled_median_block_gbp"],
        "{:,.2f}",
        "policy_diagnostics.json",
    )
    emit(
        "genJrfBlockBasisGapPct",
        lambda: 100 * (1 - jrf_ref()["modelled_over_ofgem"]),
        "{:.1f}",
        "policy_diagnostics.json",
    )

    # ------------------------------------------------------------------
    # remaining carried constant
    # ------------------------------------------------------------------
    for name, (value, fmt) in _CARRIED.items():
        emit(name, value, fmt, "carried constant — see _CARRIED")

    # ------------------------------------------------------------------
    # --- scenario grid (results/grid/grid.csv) ---------------------------
    if stale:
        note(STALE_SWEEP_WARNING)

    # The grid shows the paper's gradient is invariant to the gas/oil mix, and
    # locates the frontier at which motor fuel stops dominating the loss.
    def _grid() -> list[dict]:
        with open(R / "grid" / "grid.csv", newline="") as fh:
            return list(csv.DictReader(fh))

    emit(
        "genGridRatioMin",
        lambda: min(
            float(r["d1_d10_ratio"])
            for r in _grid()
            if r["d1_d10_ratio"] and float(r["d1_d10_ratio"]) > 0
        ),
        "{:.2f}",
        "grid/grid.csv",
    )
    emit(
        "genGridRatioMax",
        lambda: max(float(r["d1_d10_ratio"]) for r in _grid() if r["d1_d10_ratio"]),
        "{:.2f}",
        "grid/grid.csv",
    )

    def _frontier() -> float:
        """Mean gas/oil ratio at which motor fuel's share crosses 50 per cent.

        Interpolated within each oil row, then averaged over rows that cross.
        """
        by_oil: dict[float, list[dict]] = {}
        for r in _grid():
            oil = float(r["oil_pct"])
            if oil > 0:
                by_oil.setdefault(oil, []).append(r)
        slopes = []
        for oil, rows in by_oil.items():
            rows = sorted(rows, key=lambda r: float(r["gas_pct"]))
            rows = [r for r in rows if r["motor_fuel_share_pct"]]
            share = [float(r["motor_fuel_share_pct"]) for r in rows]
            gas = [float(r["gas_pct"]) for r in rows]
            for i in range(len(share) - 1):
                if share[i] >= 50 >= share[i + 1]:
                    t = (share[i] - 50) / (share[i] - share[i + 1])
                    slopes.append((gas[i] + t * (gas[i + 1] - gas[i])) / oil)
                    break
        if not slopes:
            raise ValueError("no 50 per cent crossing found in grid")
        return sum(slopes) / len(slopes)

    emit("genGridFrontierSlope", _frontier, "{:.2f}", "grid/grid.csv")

    # write
    # ------------------------------------------------------------------
    header = [
        "% values_generated.tex — machine-generated by analysis/emit_tex_values.py",
        f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"from canonical files under results/ (central scenario: {CENTRAL_SCENARIO}).",
        "% DO NOT EDIT BY HAND.",
        "% \\GENMISSING marks values whose canonical source was absent or is not "
        "computable from this dataset; it errors at LaTeX build time by design.",
        (
            "% DRAFT MODE: \\GENMISSING renders as a visible marker instead of "
            "erroring. Never commit a submitted draft built this way."
            if draft
            else "% \\GENMISSING errors at build time; run without --draft."
        ),
        (
            "\\newcommand{\\GENMISSING}{\\textcolor{red}{\\textbf{[?]}}}"
            if draft
            else "\\newcommand{\\GENMISSING}{\\errmessage{emit_tex_values: "
            "missing canonical result}}"
        ),
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + lines) + "\n")
    n_macros = sum(1 for line in lines if line.startswith("\\newcommand"))
    resolved = n_macros - len(missing)
    print(f"wrote {OUT} ({n_macros} macros, {resolved} resolved)")
    if stale:
        print(
            "WARNING: results/sensitivity/*.csv and results/grid/grid.csv were "
            "generated on the peak-fuel specification; the macros derived from "
            "them do not describe the damped main spec. Re-run "
            "analysis/run_sensitivity.py and analysis/run_grid.py."
        )
    if missing:
        print(f"WARNING: {len(missing)} macros emitted as \\GENMISSING:")
        for m in missing:
            print("  " + m)
    else:
        print("all macros resolved to real values")


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(ROOT))

    _parser = argparse.ArgumentParser(description="emit paper/values_generated.tex")
    _parser.add_argument(
        "--draft",
        action="store_true",
        help=(
            "Render missing values as a visible [?] marker instead of erroring, "
            "so an incomplete tree still compiles for reading. Never use for a "
            "submitted draft."
        ),
    )
    main(**vars(_parser.parse_args()))

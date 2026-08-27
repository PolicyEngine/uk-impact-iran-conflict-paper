"""Macro scenario definitions for the 2026 Iran-conflict energy shock.

This module is **pure scenario data plus deterministic arithmetic**. It imports
nothing from PolicyEngine and touches no microdata, so it is importable and
testable without a Hugging Face token. Everything that requires the
microsimulation lives downstream in :mod:`uk_iran_conflict.incidence`.

What lives here
---------------
Each :class:`Scenario` carries, for one macro state of the world:

* an :class:`OilPath` — Brent level and change versus the pre-war reference;
* a :class:`GasPath` — NBP wholesale level and change in p/therm;
* a :class:`PumpPricePath` — petrol and diesel retail changes, kept separate
  because the realised 2026 shock moved them very differently (+20% vs +36%);
* a :class:`PassThroughAssumptions` block — the wholesale-to-retail mapping,
  including the **explicitly named marginal-pricing parameter** that governs how
  much of a gas move reaches electricity;
* a derived quarterly :class:`CapStep` path for the Ofgem default tariff cap;
* separate ``gas`` and ``electricity`` retail shock factors.

Design rules that this module exists to enforce
-----------------------------------------------
1. **Gas and electricity are shocked asymmetrically.** They are separate
   NEED-calibrated variables in PolicyEngine UK, and GB electricity is priced at
   the margin by gas roughly 85% of the time (brief, "The shock"). A gas
   wholesale move therefore reaches electricity attenuated by both the
   marginal-pricing share and electricity's smaller wholesale cost share. The
   prior repo (``impact-iran-war-living-standards``) applied one common
   percentage to ``electricity_consumption + gas_consumption``; see
   ``docs/PRIOR_WORK_IRAN.md`` §2.
2. **The cap lags, through an averaging observation window.** Ofgem's default
   tariff cap is set from an observation window on the forward wholesale curve
   and reset quarterly. The window closes about seven weeks before the charge
   period and covers roughly the preceding quarter, putting midpoint-to-midpoint
   at four to five months — about **1.5 quarters**, which is
   :data:`CAP_LAG_QUARTERS`. The phase-in profile is *derived* from it by
   :func:`cap_phase_in_profile`, which now literally builds each quarter's
   observation window and averages :data:`GAS_PEAK_MONTHLY_PROFILE` over it,
   rather than sliding a linear ramp along a calendar.
3. **Wholesale is only part of a bill.** Wholesale energy costs are roughly
   40-50% of a dual-fuel direct-debit bill (network, policy, operating costs and
   margin make up the rest), so a +X% wholesale move enters as roughly +0.45X on
   the cap, and less than that on electricity alone.

Together (2)+(3) are the brief's Step 1: "a +X% wholesale move enters as ~0.45X
two-to-three quarters later, quantised into quarterly steps."

Sources
-------
Pre-war reference levels and the realised path
    Research brief, "The shock (context, as of August 2026)": UK gas peaked
    **+78p/therm** above pre-war (against +300p in 2022); Brent peaked
    **+$42/bbl (+57%)**, comparable to 2022's +$35; petrol **+20%**, diesel
    **+36%**. Underlying series: ICE Brent; NBP front-month/forward curve;
    DESNZ weekly road fuel prices.
NIESR scenarios
    NIESR *UK Economic Outlook*, Spring 2026 (Apr 2026): baseline GDP 0.9% in
    2026, CPI averaging 3.0% and peaking 4.1% in Jan 2027; **adverse $140/bbl**
    scenario giving CPI ~5% and a possible recession in H2-2026/H1-2027.
    NIESR *UK Economic Outlook*, Summer 2026 (Jul 2026): baseline GDP 1.1% in
    2026 and 2027, CPI averaging 3.1% and peaking 3.8% in Feb 2027; downside
    (oil **and gas +50% sustained**) giving GDP ~0.2pp lower and CPI peaking
    near 5%. The ~$103/bbl baseline oil assumption is the Spring/Summer 2026
    conditioning path.
Cap anchors
    Ofgem default tariff cap, 1 Jul - 30 Sep 2026: **£1,663/yr** for a typical
    dual-fuel direct-debit household on Ofgem's *revised* Typical Domestic
    Consumption Values (in force 1 Jul 2026); approximately £1,862 on the
    pre-July TDCV basis. The two bases are not comparable and this study uses
    the new basis throughout.
    Cornwall Insight, 19 Aug 2026: October 2026 cap forecast **£1,729, +4%**.
    Ofgem confirmed **£1,723** for October 2026 on 26 Aug 2026, superseding
    that forecast; the confirmed cap is the anchor
    (:data:`OFGEM_CAP_OCT_2026_GBP`), adjusted for the electricity VAT relief
    that holds it about £45 lower (:data:`OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP`).
Bill composition
    Ofgem cap breakdown by cost component; wholesale share of a dual-fuel bill
    in the 40-50% range across recent cap periods, lower for electricity than
    for gas because electricity carries proportionally more network and policy
    cost.
Marginal pricing
    Brief, "The shock": gas "sets the electricity price ~85% of the time".
    Underlying: National Grid ESO / Elexon half-hourly marginal-plant data.

Caveats a referee will raise, recorded here rather than hidden
--------------------------------------------------------------
* The pass-through coefficients are **calibrated, not estimated**. The brief
  identifies "no well-cited UK estimate of wholesale-to-retail pass-through in
  the price-cap era" as an open gap; this module's numbers are a transparent
  stand-in, and every one of them is a named module-level constant so a
  sensitivity analysis can sweep them.
* The pre-war reference levels are reconstructed from the brief's *changes*
  (a +57%/+$42 oil move implies a pre-war Brent of $42/0.57 = $73.7). They are
  stated explicitly as :data:`PREWAR_BRENT_USD_PER_BBL` and
  :data:`PREWAR_NBP_PENCE_PER_THERM` so they can be replaced with the actual
  series when it is loaded.
* Peak wholesale prices are not the prices the cap sees. The cap responds to an
  averaged forward window, so a peak-to-sustained damping factor is applied. It
  is not free: :data:`REALISED_SUSTAINED_FRACTION` is *solved*, jointly with the
  pre-war counterfactual cap, by :func:`solve_cap_calibration` so that the
  modelled cap path reproduces **both** observed caps — £1,663 for
  July-September 2026 and £1,723 (VAT-adjusted) for October-December 2026. It
  therefore moves automatically with the lag, the window and the bill-share
  assumptions instead of silently going stale when they change.
* **There is a pre-war counterfactual cap, and it is not an observed cap.** The
  round-3 referees found, independently, that the module was using the observed
  1 Jul - 30 Sep 2026 cap (£1,663) as the *un-shocked* denominator while
  simultaneously assigning 2026Q3 a phase-in of 1.0 — the same quarter serving
  as both the unshocked base and the fully-shocked numerator. That is a
  contradiction, not a calibration choice, and it made the whole
  conflict-attributable domestic move a Q4-versus-Q3 *step* of 6.31%. The
  counterfactual is now constructed explicitly as
  :data:`PREWAR_COUNTERFACTUAL_CAP_GBP`; the two observed caps are retained as
  what they are, observations used to *validate* the modelled path
  (:data:`CAP_VALIDATION`).
* **Both legs are annualised over one window** (:data:`MODELLED_WINDOW_LABEL`).
  Before this revision the domestic leg was averaged over 2026Q4-2027Q3 while
  the pump damping fraction was derived over calendar 2026, and the sum was
  labelled "2026". All three round-2 referees found it independently.
* There is no demand response anywhere in this module. Quantities are held
  fixed downstream (PolicyEngine UK issue #1114 — no consumption elasticity),
  making every result a Deaton first-order upper bound.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "PREWAR_BRENT_USD_PER_BBL",
    "PREWAR_NBP_PENCE_PER_THERM",
    "BASELINE_CAP_GBP",
    "OFGEM_CAP_JUL_2026_GBP",
    "PREWAR_COUNTERFACTUAL_CAP_GBP",
    "CAP_VALIDATION",
    "CapCalibration",
    "solve_cap_calibration",
    "REALISED_CAP_CALIBRATION",
    "bill_level_pass_through",
    "electricity_to_gas_pass_through_ratio",
    "BILL_LEVEL_PASS_THROUGH",
    "ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO",
    "GAS_PEAK_MONTHLY_PROFILE",
    "CAP_OBSERVATION_WINDOW_MONTHS",
    "observation_window_weights",
    "linear_ramp_phase_in_profile",
    "LINEAR_RAMP_PHASE_IN_PROFILE",
    "CAP_BASE_QUARTER",
    "CAP_BASE_PCT",
    "on_window",
    "gas_profile_variant",
    "WHOLESALE_SHARE_GAS_BILL",
    "WHOLESALE_SHARE_ELECTRICITY_BILL",
    "MARGINAL_PRICING_SHARE",
    "GAS_SHARE_OF_DUAL_FUEL_BILL",
    "CAP_LAG_QUARTERS",
    "CAP_PHASE_IN_PROFILE",
    "CAP_QUARTER_LABELS",
    "CAP_ANCHOR_QUARTER",
    "CAP_ANCHOR_PCT",
    "OFGEM_CAP_OCT_2026_GBP",
    "OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP",
    "SHOCK_ONSET_MONTH",
    "MODELLED_WINDOW_START",
    "MODELLED_WINDOW_END",
    "MODELLED_WINDOW_LABEL",
    "MODELLED_WINDOW_MONTHS",
    "CALENDAR_2026_MONTHS",
    "MONTHLY_CONSUMPTION_WEIGHTS_GAS",
    "MONTHLY_CONSUMPTION_WEIGHTS_ELECTRICITY",
    "PUMP_PEAK_MONTHLY_PROFILE",
    "REALISED_GAS_PCT_CHANGE",
    "REALISED_SUSTAINED_FRACTION",
    "REALISED_PUMP_SUSTAINED_FRACTION",
    "PUMP_SUSTAINED_FRACTION_CALENDAR_2026",
    "SYMMETRIC_SUSTAINED_FRACTION",
    "QUARTERLY_CONSUMPTION_WEIGHTS_GAS",
    "QUARTERLY_CONSUMPTION_WEIGHTS_ELECTRICITY",
    "months_between",
    "quarter_of",
    "quarters_of",
    "quarterly_consumption_weights",
    "cap_phase_in_profile",
    "pump_sustained_fraction",
    "sustained_fraction_for_cap_anchor",
    "OilPath",
    "GasPath",
    "PumpPricePath",
    "PassThroughAssumptions",
    "CapStep",
    "RetailEnergyShock",
    "Scenario",
    "SCENARIOS",
    "get_scenario",
    "scenario_keys",
]


# ---------------------------------------------------------------------------
# Reference levels
# ---------------------------------------------------------------------------

PREWAR_BRENT_USD_PER_BBL: float = 73.7
"""Pre-war (Feb 2026) Brent reference, US$/bbl.

Implied by the brief's realised move: oil peaked +$42/bbl and +57%, so the base
is 42 / 0.57 = $73.7/bbl. Replace with the ICE Brent Feb-2026 average when the
series is loaded.
"""

PREWAR_NBP_PENCE_PER_THERM: float = 90.0
"""Pre-war (Feb 2026) NBP wholesale gas reference, pence/therm.

The brief gives the realised move in levels (+78p/therm) but not the base. 90p
is the working assumption for the early-2026 NBP front-month, consistent with a
cap of £1,663 on the revised TDCV basis. **This is the single most replaceable
number in the module**: the realised scenario's percentage gas move (and hence
everything derived from it) scales inversely with it.
"""

OFGEM_CAP_JUL_2026_GBP: float = 1_663.0
"""**Observed** Ofgem default tariff cap, 1 Jul - 30 Sep 2026.

Typical dual-fuel direct debit on the revised (1 Jul 2026) Typical Domestic
Consumption Values; approximately £1,862 on the pre-July TDCV basis, and the two
are not comparable.

This is an **observation of a post-shock cap**, and it is used here only to
*validate* the modelled path (:data:`CAP_VALIDATION`). It is emphatically not
the un-shocked baseline: 1 Jul - 30 Sep 2026 is the fifth month of a war that
began on 28 February 2026, and the observation window that priced it closed in
mid-May 2026, well inside the conflict. Treating it as the counterfactual while
also assigning 2026Q3 a phase-in of 1.0 — which the module did before this
revision — makes the same quarter both the unshocked denominator and the fully
shocked numerator. See :data:`PREWAR_COUNTERFACTUAL_CAP_GBP`.
"""


# ---------------------------------------------------------------------------
# Pass-through structure (module-level defaults; overridable per scenario)
# ---------------------------------------------------------------------------

WHOLESALE_SHARE_GAS_BILL: float = 0.45
"""Wholesale energy cost as a share of a domestic **gas** bill under the cap.

Within the 40-50% range the brief cites for a dual-fuel bill; gas sits at the
upper half of that range because it carries less network and policy cost than
electricity.
"""

WHOLESALE_SHARE_ELECTRICITY_BILL: float = 0.35
"""Wholesale energy cost as a share of a domestic **electricity** bill.

Lower than for gas: electricity bills carry proportionally more network,
policy/social-obligation and capacity cost. This asymmetry is one of the two
reasons a gas shock reaches electricity attenuated.
"""

MARGINAL_PRICING_SHARE: float = 0.85
"""**The marginal-pricing pass-through parameter.**

Share of settlement periods in which gas-fired plant is the marginal
price-setting technology in GB, so that the wholesale electricity price moves
one-for-one (in energy-equivalent terms) with the gas price. Brief: gas "sets
the electricity price ~85% of the time".

This is deliberately a named, swept parameter rather than an implicit 1.0. It is
the second reason a gas shock reaches electricity attenuated, and setting it to
1.0 recovers the naive "shock gas and electricity together" assumption used in
the prior repo.
"""

GAS_SHARE_OF_DUAL_FUEL_BILL: float = 0.45
"""Gas as a share of the £ value of the typical dual-fuel cap.

Used only to aggregate the separate gas and electricity retail shocks back into
a single headline cap level. Note this is a *£ share of the bill*, distinct from
the brief's statistic that gas is 62% of UK final household energy consumption
*in kWh* (the highest in the G7) — gas kWh are much cheaper per unit.
"""

# ---------------------------------------------------------------------------
# The modelled year: ONE window, both legs
# ---------------------------------------------------------------------------
#
# Round-2 referees found, independently of one another, that the two legs of the
# shock were annualised over *different* windows and the sum labelled "2026":
# the domestic leg averaged the cap over 2026Q4-2027Q3 while the pump damping
# fraction was derived from a **calendar-2026** monthly profile whose first two
# months are pre-shock. Nothing downstream could be right while that held, so
# the window is now a single named object and **every** annualisation — the cap
# phase-in weights, the consumption weights and the pump damping fraction — is
# derived from it by the functions below rather than written down by hand.

SHOCK_ONSET_MONTH: str = "2026-03"
"""First month in which the conflict shock is present in prices.

The war starts on/around 28 February 2026, so March 2026 is the first month
carrying any of it. Both legs are zero before this month, by construction.
"""

MODELLED_WINDOW_START: str = "2026-03"
MODELLED_WINDOW_END: str = "2027-02"
MODELLED_WINDOW_LABEL: str = "March 2026 - February 2027 (twelve months from the shock)"
"""**The modelled year.** Everything the paper reports as an annual figure is an
average or a total over exactly these twelve months.

Why the shock-year and not calendar 2026
----------------------------------------
Three windows were on the table.

*Calendar 2026* is the window the Resolution Foundation £11bn comparator uses
and the one the pump fraction was already (implicitly) derived on. Its defect is
that two of its twelve months are pre-shock on *both* legs while the shock's
second half runs on into 2027: it reports twelve months of denominator against
ten months of shock, and it truncates the cap leg precisely where the cap's
institutional lag puts it. Under it the domestic leg is not small because the
cap moved little but because the window ends before the cap moves.

*2026Q4-2027Q3*, the window the domestic leg previously used, has the mirror
defect: it begins seven months after the shock and therefore misses the whole
of the pump-price peak.

*The twelve months from the shock* has neither. It contains one full annual
heating cycle (so the seasonal consumption weights sum to one exactly and no
renormalisation is needed), it contains the observed pump-price peak, plateau
and decay, and it contains the two Ofgem cap resets — October 2026 and January
2027 — that the shock actually reaches. It is the window over which "the cost of
the conflict to a household" is a well-posed question. The cost is that it is
**not** calendar 2026, so the RF £11bn is a different-period comparator and must
be labelled as one; :data:`CALENDAR_2026_MONTHS` is retained so both legs can
also be reported on the RF window as a robustness line.
"""


def months_between(start: str, end: str) -> tuple[str, ...]:
    """Inclusive ``"YYYY-MM"`` month labels from ``start`` to ``end``."""
    sy, sm = (int(part) for part in start.split("-"))
    ey, em = (int(part) for part in end.split("-"))
    first = sy * 12 + (sm - 1)
    last = ey * 12 + (em - 1)
    if last < first:
        raise ValueError(f"end {end!r} precedes start {start!r}")
    return tuple(f"{i // 12:04d}-{i % 12 + 1:02d}" for i in range(first, last + 1))


MODELLED_WINDOW_MONTHS: tuple[str, ...] = months_between(
    MODELLED_WINDOW_START, MODELLED_WINDOW_END
)
"""The twelve ``"YYYY-MM"`` labels of :data:`MODELLED_WINDOW_LABEL`."""

CALENDAR_2026_MONTHS: tuple[str, ...] = months_between("2026-01", "2026-12")
"""Calendar 2026, kept only so results can also be reported on the RF window."""


def quarter_of(month: str) -> str:
    """Ofgem cap quarter containing ``month`` (``"2026-05"`` -> ``"2026Q2"``)."""
    year, mon = (int(part) for part in month.split("-"))
    return f"{year}Q{(mon - 1) // 3 + 1}"


def quarters_of(months: tuple[str, ...]) -> tuple[str, ...]:
    """Cap quarters touched by ``months``, in calendar order, deduplicated."""
    out: list[str] = []
    for month in months:
        quarter = quarter_of(month)
        if quarter not in out:
            out.append(quarter)
    return tuple(out)


def _quarter_index(quarter: str) -> float:
    year, _, number = quarter.partition("Q")
    return int(year) * 4 + (int(number) - 1)


def _month_index(month: str) -> float:
    year, mon = (int(part) for part in month.split("-"))
    return year * 12 + (mon - 1)


# ---------------------------------------------------------------------------
# Seasonal consumption weights (monthly, so any window can be weighted)
# ---------------------------------------------------------------------------

MONTHLY_CONSUMPTION_WEIGHTS_GAS: tuple[float, ...] = (
    0.130,  # Jan
    0.115,  # Feb
    0.105,  # Mar
    0.090,  # Apr
    0.075,  # May
    0.055,  # Jun
    0.035,  # Jul
    0.030,  # Aug
    0.045,  # Sep
    0.090,  # Oct
    0.115,  # Nov
    0.115,  # Dec
)
"""Share of annual **domestic gas** consumption falling in each calendar month.

Indexed by month of the year (January first), so a window that is not a calendar
year can still be weighted correctly. Domestic gas is overwhelmingly space and
water heating, so it is strongly seasonal; these are the standard shape of the
DESNZ/Xoserve domestic gas demand year, and they aggregate to the quarterly
shape used before this revision (Q1 0.35, Q2 0.22, Q3 0.11, Q4 0.32) so the
change of window does not silently change the seasonality as well. Sums to one.
"""

MONTHLY_CONSUMPTION_WEIGHTS_ELECTRICITY: tuple[float, ...] = (
    0.105,  # Jan
    0.100,  # Feb
    0.095,  # Mar
    0.080,  # Apr
    0.077,  # May
    0.073,  # Jun
    0.062,  # Jul
    0.062,  # Aug
    0.066,  # Sep
    0.090,  # Oct
    0.093,  # Nov
    0.097,  # Dec
)
"""Share of annual **domestic electricity** consumption by calendar month.

Same construction as :data:`MONTHLY_CONSUMPTION_WEIGHTS_GAS` but materially
flatter — only part of domestic electricity is heating-related — and aggregating
to the previous quarterly shape (Q1 0.30, Q2 0.23, Q3 0.19, Q4 0.28). Sums to
one.
"""


def quarterly_consumption_weights(
    months: tuple[str, ...], monthly: tuple[float, ...]
) -> tuple[float, ...]:
    """Consumption weight of each quarter touched by ``months``.

    Returns one weight per element of ``quarters_of(months)``, normalised to sum
    to one. Because :data:`MODELLED_WINDOW_MONTHS` is a full twelve-month
    heating cycle the normalisation is the identity for the default window; it
    matters only for a partial window.
    """
    quarters = quarters_of(months)
    totals = dict.fromkeys(quarters, 0.0)
    for month in months:
        totals[quarter_of(month)] += monthly[int(month.split("-")[1]) - 1]
    grand = sum(totals.values())
    if grand <= 0:
        raise ValueError("consumption weights over the window sum to zero")
    return tuple(totals[q] / grand for q in quarters)


# ---------------------------------------------------------------------------
# The cap lag, and the phase-in profile derived from it
# ---------------------------------------------------------------------------

CAP_LAG_QUARTERS: float = 1.5
"""Quarters from a wholesale move to its full appearance in the Ofgem cap.

**Why 1.5 and not 3.** Ofgem's default-tariff-cap methodology sets each quarter's
wholesale allowance from an observation window on the forward curve that
*closes about seven weeks before the charge period begins* and covers roughly the
preceding quarter. The midpoint of that observation window therefore sits about
seven weeks plus half a quarter — call it four to five months — ahead of the
midpoint of the charge period it prices. Four to five months is **1.3 to 1.7
quarters**, not three.

The previous value of 3 came from reading the brief's "6-9 months" as a lag to
*full* pass-through rather than as the span over which a move is spread. It is
roughly double the institutional lag, and because a longer lag pushes the cap
move out of the modelled window it was doing real work: it is what made 2026
look "almost purely a pump-price event". Three referees flagged it. It is now
1.5 as the central case, and :func:`cap_phase_in_profile` derives the phase-in
profile *from* it, so the headline and the ``results/sensitivity/cap_lag.csv``
appendix can no longer disagree — they are the same function of the same
parameter. Lags of 1, 2, 3 and 4 are still swept, with 3 reported as the
previous specification.

The lag is a float, not an int: the institutional answer is not a whole number
of quarters and rounding it to one was part of the problem.
"""


CAP_QUARTER_LABELS: tuple[str, ...] = quarters_of(MODELLED_WINDOW_MONTHS)
"""The five Ofgem cap periods the modelled year touches.

``2026Q1`` (March only), ``2026Q2``, ``2026Q3`` and ``2026Q4`` — the two whose
observed caps calibrate the path — and ``2027Q1`` (January and February 2027).
Five rather than four because a twelve-month window that does not start on a
quarter boundary straddles five quarters; the consumption weights in
:data:`QUARTERLY_CONSUMPTION_WEIGHTS_GAS` carry only the months actually inside
the window, so nothing is double-counted.
"""


GAS_PEAK_MONTHLY_PROFILE: Mapping[str, float] = {
    # Every month before the shock is exactly zero by construction. Carried
    # back far enough that even a four-quarter lag's observation window for
    # 2026Q1 (which opens in January 2025) resolves without a lookup miss.
    **dict.fromkeys(months_between("2024-06", "2025-12"), 0.00),
    "2026-01": 0.00,
    "2026-02": 0.00,
    "2026-03": 0.55,
    "2026-04": 0.85,
    "2026-05": 1.00,
    "2026-06": 1.00,
    "2026-07": 0.90,
    "2026-08": 0.80,
    "2026-09": 0.70,
    "2026-10": 0.60,
    "2026-11": 0.55,
    "2026-12": 0.50,
    "2027-01": 0.45,
    "2027-02": 0.45,
    "2027-03": 0.45,
    "2027-04": 0.45,
    "2027-05": 0.45,
    "2027-06": 0.45,
}
"""Realised NBP wholesale gas uplift each month, as a fraction of the peak.

The gas analogue of :data:`PUMP_PEAK_MONTHLY_PROFILE`, and — like it — a
**calibration, not an estimate**. It is a piecewise-linear reading of the shape
the brief describes: pre-war through February 2026 (the war begins on 28
February, so February is effectively a pre-shock month and is held at exactly
zero, which also makes this profile identical to the ``shift=0, flatten=1``
cell of :func:`gas_profile_variant`), a fast ramp through the spring
as Hormuz risk is priced into the forward curve, a peak across May-June 2026,
then a slow decay through the second half that flattens out well above pre-war
rather than returning to it. 1.00 is the +78p/therm peak the brief quotes.

**Why the module needs this.** The cap does not respond to a price, it responds
to the *average of a forward curve over an observation window*. Without a
monthly wholesale path there is no way to say how much of the shock any given
cap period's window contains, and the module was previously forced to assert
that with a hand-parameterised linear ramp — which is what produced the
contradiction of a 2026Q3 phase-in of 1.0 sitting next to a 2026Q3 cap used as
the pre-shock base. With the path written down, each quarter's phase-in is
derived (:func:`cap_phase_in_profile`) instead of asserted.

**It is the most consequential replaceable number in the module after
:data:`PREWAR_NBP_PENCE_PER_THERM`.** The pre-war counterfactual cap is
identified off the *difference* between the 2026Q3 and 2026Q4 observation-window
averages, so a flatter profile (a smaller difference) implies a larger
conflict-attributable move and a lower counterfactual cap, and vice versa.
``analysis/run_variants.py`` sweeps it; the sweep, not this docstring, is the
honest statement of how much it matters.

Months outside the modelled year are carried so that observation windows which
open before it (2026Q1's and 2026Q2's, which open in 2025) and close after it
can be evaluated without a lookup miss.
"""


def gas_profile_variant(
    shift_months: int = 0,
    flatten: float = 1.0,
    profile: Mapping[str, float] | None = None,
    onset_month: str = SHOCK_ONSET_MONTH,
) -> dict[str, float]:
    """A perturbed :data:`GAS_PEAK_MONTHLY_PROFILE`, for the identification sweep.

    :func:`solve_cap_calibration` identifies the pre-war counterfactual cap off
    the *difference* between what the 2026Q3 and 2026Q4 observation windows
    price, and that difference is a property of this profile. The profile is a
    calibration, so the sweep is not optional: it is the honest statement of how
    much the headline rests on it.

    Two perturbations, each with a plain-English reading:

    ``shift_months``
        The whole path arrives ``shift_months`` later (negative for earlier).
        "The market priced Hormuz risk a month sooner than assumed."
    ``flatten``
        Post-onset values are pulled toward their own mean by this factor: 1.0
        is unchanged, 0 is a completely flat post-onset path, above 1 is a
        sharper spike. "The spike was less/more pronounced than assumed." A
        flatter path shrinks the window difference and so implies a *lower*
        counterfactual cap and a larger conflict-attributable move.

    Pre-onset months are held at exactly zero under both, because a pre-war
    month carrying shock is not a perturbation, it is an error.
    """
    table = dict(GAS_PEAK_MONTHLY_PROFILE if profile is None else profile)
    onset = _month_index(onset_month)
    if shift_months:
        shifted: dict[str, float] = {}
        for month, value in table.items():
            idx = int(_month_index(month)) + shift_months
            shifted[f"{idx // 12:04d}-{idx % 12 + 1:02d}"] = value
        # Keep the original key set; months shifted off either end read as the
        # nearest retained value so the windows still resolve.
        table = {
            m: shifted.get(m, 0.0 if _month_index(m) < onset else max(shifted.values()))
            for m in table
        }
    post = [v for m, v in table.items() if _month_index(m) >= onset]
    if flatten != 1.0 and post:
        mean = sum(post) / len(post)
        table = {
            m: (
                0.0
                if _month_index(m) < onset
                else max(0.0, mean + flatten * (v - mean))
            )
            for m, v in table.items()
        }
    else:
        table = {m: (0.0 if _month_index(m) < onset else v) for m, v in table.items()}
    return table


CAP_OBSERVATION_WINDOW_MONTHS: float = 3.0
"""Length of Ofgem's forward-curve observation window, in months.

"Covers roughly the preceding quarter" (Ofgem's default-tariff-cap methodology).
Together with the lead implied by :data:`CAP_LAG_QUARTERS` this fixes each cap
period's window exactly, so :func:`cap_phase_in_profile` can build it rather
than approximate it.
"""


def observation_window_weights(
    quarter: str,
    lag_quarters: float = CAP_LAG_QUARTERS,
    window_months: float = CAP_OBSERVATION_WINDOW_MONTHS,
) -> dict[str, float]:
    """Calendar-month weights of the observation window pricing ``quarter``.

    The window has length ``window_months`` and **closes** ``lead`` months
    before the charge period begins, where the lead is implied by the lag::

        lead = 3 * lag_quarters - window_months / 2 - 3 / 2

    That is the identity behind "1.5 quarters": the window's midpoint sits
    ``lead + window_months / 2`` months before the charge period starts, the
    charge period's midpoint sits 1.5 months after it starts, and the
    midpoint-to-midpoint distance is what :data:`CAP_LAG_QUARTERS` measures. At
    ``lag_quarters = 1.5`` and a three-month window the lead is 1.5 months —
    within a fortnight of Ofgem's stated "about seven weeks", which is the check
    that this parameterisation is the institutional one rather than a fit.

    Returns ``{"YYYY-MM": weight}`` with the weights summing to one; a window
    that straddles a month boundary splits that month proportionally, which is
    why the weights are not all equal.
    """
    if lag_quarters <= 0:
        raise ValueError(f"lag_quarters must be positive; got {lag_quarters!r}")
    if window_months <= 0:
        raise ValueError(f"window_months must be positive; got {window_months!r}")
    year, _, number = quarter.partition("Q")
    start = int(year) * 12 + (int(number) - 1) * 3  # month index of quarter start
    lead = 3.0 * lag_quarters - window_months / 2.0 - 1.5
    close = start - lead
    open_ = close - window_months
    weights: dict[str, float] = {}
    first = int(open_ // 1)
    last = int(-((-close) // 1))  # ceil
    for idx in range(first, last):
        overlap = min(close, idx + 1.0) - max(open_, float(idx))
        if overlap <= 0:
            continue
        label = f"{idx // 12:04d}-{idx % 12 + 1:02d}"
        weights[label] = weights.get(label, 0.0) + overlap / window_months
    return weights


def cap_phase_in_profile(
    lag_quarters: float = CAP_LAG_QUARTERS,
    quarters: tuple[str, ...] | None = None,
    onset_month: str = SHOCK_ONSET_MONTH,
    profile: Mapping[str, float] | None = None,
    window_months: float = CAP_OBSERVATION_WINDOW_MONTHS,
) -> tuple[float, ...]:
    """Fraction of the full wholesale peak that each cap quarter's window sees.

    **This is the round-3 fix.** The previous version was a linear ramp from zero
    at the shock onset to one at ``lag_quarters`` after it, evaluated at each cap
    quarter's midpoint, and it returned ``(0.0, 0.556, 1.0, 1.0, 1.0)`` — which
    says the 1 Jul - 30 Sep 2026 cap already carries *full* pass-through of the
    conflict. The module then used that same quarter's observed cap as the
    un-shocked baseline. One quarter cannot be both. Three round-3 referees found
    it independently, and it is why the conflict-attributable domestic move came
    out as a 6.31% Q4-versus-Q3 step and the domestic leg at £67 a year.

    The construction here is institutional rather than parametric. Each cap
    period is priced from a forward-curve observation window
    (:func:`observation_window_weights`); the fraction of the shock that period's
    cap can possibly contain is the weighted average of
    :data:`GAS_PEAK_MONTHLY_PROFILE` over that window. Nothing is asserted about
    "phase-in": it falls out.

    Three things change as a result, all of them for the better:

    * **2026Q3 is no longer 1.0.** Its window is mid-February to mid-May 2026, a
      window that is a quarter pre-war and catches only the ramp, so it prices
      about 0.63 of the peak. The July cap is a partly-shocked cap, which is what
      it is.
    * **The profile is no longer monotone.** 2027Q1's window (mid-August to
      mid-November 2026) sits on the decay, so it prices less of the peak than
      2026Q4's does. A cap that comes back down as the spike rolls out of the
      window is a property of the institution, and the old ramp — which was
      monotone by construction — could not represent it.
    * **The lag stays a single swept parameter.** ``lag_quarters`` now moves the
      window rather than the ramp, so ``results/sensitivity/cap_lag.csv`` is
      still a sweep of the same object the headline uses.

    ``onset_month`` is retained and used: any quarter whose window closes at or
    before the shock onset is forced to exactly zero, so a profile calibration
    that leaks a little pre-war value cannot manufacture a pre-war cap move.

    The linear ramp is kept, fixed, as :func:`linear_ramp_phase_in_profile` and
    reported as an alternative specification.
    """
    if quarters is None:
        quarters = CAP_QUARTER_LABELS
    table = GAS_PEAK_MONTHLY_PROFILE if profile is None else profile
    onset = _month_index(onset_month)
    out: list[float] = []
    for quarter in quarters:
        weights = observation_window_weights(quarter, lag_quarters, window_months)
        missing = [m for m in weights if m not in table]
        if missing:
            raise KeyError(
                f"gas profile has no entry for {missing} (observation window "
                f"for {quarter!r}); extend GAS_PEAK_MONTHLY_PROFILE"
            )
        value = sum(w * table[m] for m, w in weights.items())
        if all(_month_index(m) + 1 <= onset for m in weights):
            value = 0.0
        out.append(min(1.0, max(0.0, value)))
    return tuple(out)


def linear_ramp_phase_in_profile(
    lag_quarters: float = CAP_LAG_QUARTERS,
    quarters: tuple[str, ...] | None = None,
    onset_month: str = SHOCK_ONSET_MONTH,
    months: tuple[str, ...] | None = None,
) -> tuple[float, ...]:
    """The pre-round-3 linear ramp, with the partial-quarter bug fixed.

    ``phase_in(q) = clip((midpoint(q) - onset) / lag_quarters, 0, 1)``, retained
    as a labelled alternative specification so the effect of replacing it with
    the observation-window construction is visible rather than asserted.

    **The partial-quarter fix.** The ramp used to evaluate every quarter at the
    *full* quarter's midpoint while the consumption weights carried only the
    months actually inside the modelled window. 2026Q1 contributes March alone,
    but was evaluated at mid-February — before the shock — and so returned 0.0.
    Evaluated at the midpoint of its in-window months it returns 0.111, and the
    annual gas phase-in rises from 0.797 to 0.809. ``months`` supplies the window
    so each quarter is evaluated at the midpoint of the months of it that are
    actually charged; it defaults to :data:`MODELLED_WINDOW_MONTHS`.

    This function is *not* the default. It is a ramp in calendar time, and the
    cap does not respond to calendar time; it responds to an averaging window.
    """
    if lag_quarters <= 0:
        raise ValueError(f"lag_quarters must be positive; got {lag_quarters!r}")
    if quarters is None:
        quarters = CAP_QUARTER_LABELS
    if months is None:
        months = MODELLED_WINDOW_MONTHS
    onset = _month_index(onset_month) / 3.0
    in_window: dict[str, list[str]] = {}
    for month in months:
        in_window.setdefault(quarter_of(month), []).append(month)
    out: list[float] = []
    for quarter in quarters:
        charged = in_window.get(quarter)
        if charged:
            # Midpoint of the months of this quarter actually inside the window.
            midpoint = (
                sum(_month_index(m) + 0.5 for m in charged) / len(charged)
            ) / 3.0
        else:
            midpoint = _quarter_index(quarter) + 0.5
        out.append(min(1.0, max(0.0, (midpoint - onset) / lag_quarters)))
    return tuple(out)


CAP_PHASE_IN_PROFILE: tuple[float, ...] = cap_phase_in_profile()
"""Per-quarter share of the wholesale peak the cap's observation window prices.

``(0.000, 0.000, 0.633, 0.933, 0.658)`` at :data:`CAP_LAG_QUARTERS` = 1.5:
nothing in 2026Q1 (its window closed in November 2025), nothing in 2026Q2
(its window closes in mid-February 2026, before the shock), about two thirds
in 2026Q3, nearly all of it in 2026Q4, and back to two thirds
in 2027Q1 as the spike rolls out of the window.

The pre-round-3 value was ``(0.0, 0.556, 1.0, 1.0, 1.0)``. The difference
between those two tuples is the paper's headline.
"""

LINEAR_RAMP_PHASE_IN_PROFILE: tuple[float, ...] = linear_ramp_phase_in_profile()
"""The alternative-specification ramp, with the partial-quarter fix applied.

``(0.111, 0.556, 1.0, 1.0, 1.0)``. The leading 0.111 is the fix: 2026Q1 carries
March only, and March is post-onset.
"""

QUARTERLY_CONSUMPTION_WEIGHTS_GAS: tuple[float, ...] = quarterly_consumption_weights(
    MODELLED_WINDOW_MONTHS, MONTHLY_CONSUMPTION_WEIGHTS_GAS
)
"""Gas consumption weight of each element of :data:`CAP_QUARTER_LABELS`."""

QUARTERLY_CONSUMPTION_WEIGHTS_ELECTRICITY: tuple[float, ...] = (
    quarterly_consumption_weights(
        MODELLED_WINDOW_MONTHS, MONTHLY_CONSUMPTION_WEIGHTS_ELECTRICITY
    )
)
"""Electricity consumption weight of each element of :data:`CAP_QUARTER_LABELS`."""


# ---------------------------------------------------------------------------
# The pump leg, damped on the same window
# ---------------------------------------------------------------------------

PUMP_PEAK_MONTHLY_PROFILE: Mapping[str, float] = {
    "2026-01": 0.00,
    "2026-02": 0.00,
    "2026-03": 0.17,
    "2026-04": 0.50,
    "2026-05": 0.83,
    "2026-06": 1.00,
    "2026-07": 1.00,
    "2026-08": 1.00,
    "2026-09": 0.90,
    "2026-10": 0.75,
    "2026-11": 0.55,
    "2026-12": 0.40,
    "2027-01": 0.35,
    "2027-02": 0.35,
}
"""Realised pump-price uplift each month, as a fraction of the observed peak.

A **calibration, not an estimate**: there is no published annual-average 2026
pump series to fit. It is a transparent piecewise-linear reading of the DESNZ
weekly road-fuel path as described in the brief — pre-war levels through late
February 2026, a ramp to the peak by early summer, a short plateau, then a decay
over the second half that flattens out above pre-war rather than returning to
it. The 2027 months hold the tail flat at 0.35, which is the assumption a reader
is most likely to want to change; it is worth 0.06 on the window fraction.

This profile is now the *single* source of the pump damping fraction, and
:func:`pump_sustained_fraction` averages it over whatever window is asked for.
Previously the number 0.60 was written down in a docstring that derived it over
calendar 2026 while the domestic leg was averaged over 2026Q4-2027Q3.
"""


def pump_sustained_fraction(
    months: tuple[str, ...] = MODELLED_WINDOW_MONTHS,
    profile: Mapping[str, float] | None = None,
) -> float:
    """Mean of :data:`PUMP_PEAK_MONTHLY_PROFILE` over ``months``.

    This is the peak-to-window-average damping applied to the quoted pump peaks
    (petrol +20%, diesel +36%), and it is derived on **the same window** as the
    cap phase-in — which is the round-2 fix. Over
    :data:`MODELLED_WINDOW_MONTHS` it is 0.650; over calendar 2026 it is 0.592,
    the number the old constant rounded to 0.60.
    """
    table = PUMP_PEAK_MONTHLY_PROFILE if profile is None else profile
    missing = [m for m in months if m not in table]
    if missing:
        raise KeyError(f"pump profile has no entry for {missing}")
    return sum(table[m] for m in months) / len(months)


REALISED_PUMP_SUSTAINED_FRACTION: float = pump_sustained_fraction()
"""Peak-to-window-average damping for the realised 2026 **pump** path: 0.650.

The realised pump moves in the brief are *peaks*, exactly as the gas figure is a
peak. The stated justification for applying them undamped — road fuel passes
through in weeks, the cap in quarters — is a statement about **lag**, not about
**duration**: it licenses applying the peak sooner, not for twelve months.

A referee who disagrees changes :data:`PUMP_PEAK_MONTHLY_PROFILE`, not this
number, and ``realised_2026_peak_fuel`` (fraction 1.0) is reported alongside as
the explicit upper bound on the fuel channel.
"""

PUMP_SUSTAINED_FRACTION_CALENDAR_2026: float = pump_sustained_fraction(
    CALENDAR_2026_MONTHS
)
"""The same profile averaged over calendar 2026 (0.592).

Reported only so the effect of the window choice on the fuel leg is visible: it
is the number the pre-revision code used, and using it while averaging the cap
over 2026Q4-2027Q3 is the inconsistency this revision removes.
"""

SYMMETRIC_SUSTAINED_FRACTION: float = REALISED_PUMP_SUSTAINED_FRACTION
"""Common peak-to-window-average damping applied to **both** legs.

Motor fuel's share of the modelled loss depends only on the *ratio* of the two
damping fractions, so a symmetric-damping specification has to be a first-class
run rather than a footnote (``docs/FIXES.md`` A3). "Symmetric" means asserting
the pump leg's own monthly profile of the wholesale gas peak as well, so this is
by definition :data:`REALISED_PUMP_SUSTAINED_FRACTION`.

**What it costs.** Applying it to the gas leg abandons the Ofgem October-2026 cap
anchor that :data:`REALISED_SUSTAINED_FRACTION` is calibrated to. That is the
explicit trade-off: this specification buys internal symmetry between the legs at
the price of the external cap anchor, and neither can have both. Both are run.
"""


# ---------------------------------------------------------------------------
# The cap anchor, and the sustained fraction calibrated to it
# ---------------------------------------------------------------------------

OFGEM_CAP_OCT_2026_GBP: float = 1_723.0
"""**Observed** Ofgem default tariff cap, 1 Oct - 31 Dec 2026, confirmed 26 Aug 2026.

Typical dual-fuel direct debit on the revised (1 Jul 2026) TDCV basis, the same
basis as :data:`OFGEM_CAP_JUL_2026_GBP`. This supersedes the Cornwall Insight
forecast of £1,729 (19 Aug 2026): a confirmed cap is a better anchor than a
forecast, and the difference is worth £6.

Like the July cap this is an observation of a *shocked* cap. Together the two
observations identify the calibration (:func:`solve_cap_calibration`); neither
is the counterfactual.
"""

OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP: float = 45.0
"""Cap reduction from Ofgem's temporary removal of VAT on electricity, 1 Oct 2026.

The October 2026 cap is held roughly £45 lower by a VAT measure that has nothing
to do with the conflict, so the conflict-attributable October cap level is
``1723 + 45 = 1768``. Omitting it would attribute a tax cut to the war.

This also interacts directly with the scored VAT zero-rating instrument, since
part of that instrument may already be in the baseline; see
``docs/VALIDATION.md``.
"""

CAP_ANCHOR_QUARTER: str = "2026Q4"
"""The cap quarter :data:`CAP_ANCHOR_PCT` prices."""

CAP_BASE_QUARTER: str = "2026Q3"
"""The second observed cap quarter. **Not** the pre-war base — see below."""


@dataclass(frozen=True, slots=True)
class CapCalibration:
    """Pre-war counterfactual cap and sustained fraction, solved jointly.

    Attributes
    ----------
    prewar_cap_gbp:
        The counterfactual cap: what the typical dual-fuel cap would have been,
        on the revised TDCV basis, had the conflict not happened.
    sustained_fraction:
        Fraction of the quoted wholesale gas *peak* that is sustained long
        enough to be what the observation windows average to. Multiplies
        :data:`GAS_PEAK_MONTHLY_PROFILE`, which already carries the shape.
    steady_state_dual_fuel_pct:
        ``sustained_fraction x gas_pct_change x bill_level_pass_through`` — the
        proportional dual-fuel cap move at full pass-through.
    modelled_base_cap_gbp, modelled_anchor_cap_gbp:
        The modelled caps in the two observed quarters. Equal to the observed
        caps by construction; carried so the identity can be *asserted* in
        ``results/`` rather than believed.
    """

    prewar_cap_gbp: float
    sustained_fraction: float
    steady_state_dual_fuel_pct: float
    modelled_base_cap_gbp: float
    modelled_anchor_cap_gbp: float


def bill_level_pass_through(
    *,
    wholesale_share_gas_bill: float = WHOLESALE_SHARE_GAS_BILL,
    wholesale_share_electricity_bill: float = WHOLESALE_SHARE_ELECTRICITY_BILL,
    marginal_pricing_share: float = MARGINAL_PRICING_SHARE,
    gas_share_of_dual_fuel_bill: float = GAS_SHARE_OF_DUAL_FUEL_BILL,
) -> float:
    """The bill-level pass-through coefficient the code actually implements.

    ``w_g x ws_gas + (1 - w_g) x mps x ws_elec``. On the module defaults
    ``0.45 x 0.45 + 0.55 x 0.85 x 0.35 = 0.3661``: a +1% wholesale gas move
    raises the typical dual-fuel cap by 0.366%.

    Round-3 finding 4. Equation (1) of the paper states a pass-through of
    ``phi ~ 0.45``, which is :data:`WHOLESALE_SHARE_GAS_BILL` — the *gas-bill*
    coefficient, not the bill-level one. They differ by the electricity half of
    the bill being twice-attenuated. Naming the implemented coefficient here
    means the prose can quote it instead of quoting a component of it.
    """
    return (
        gas_share_of_dual_fuel_bill * wholesale_share_gas_bill
        + (1.0 - gas_share_of_dual_fuel_bill)
        * marginal_pricing_share
        * wholesale_share_electricity_bill
    )


def electricity_to_gas_pass_through_ratio(
    *,
    wholesale_share_gas_bill: float = WHOLESALE_SHARE_GAS_BILL,
    wholesale_share_electricity_bill: float = WHOLESALE_SHARE_ELECTRICITY_BILL,
    marginal_pricing_share: float = MARGINAL_PRICING_SHARE,
) -> float:
    """``phi_elec / phi_gas`` as implemented: ``mps x ws_elec / ws_gas``.

    Round-3 finding 4, second half. The paper writes the reduced form as
    ``phi_elec = psi . phi_gas``, i.e. the electricity retail shock is the gas
    retail shock attenuated by the marginal-pricing share alone. The code applies
    **two** independent attenuations — the marginal-pricing share *and*
    electricity's smaller wholesale cost share — so the implemented ratio is
    ``0.85 x 0.35 / 0.45 = 0.6611``, not ``psi = 0.85``. That is a deliberate and
    documented modelling choice (design rule 1); it is simply not the equation
    the paper prints.
    """
    if wholesale_share_gas_bill <= 0:
        raise ValueError("wholesale_share_gas_bill must be positive")
    return (
        marginal_pricing_share
        * wholesale_share_electricity_bill
        / wholesale_share_gas_bill
    )


BILL_LEVEL_PASS_THROUGH: float = bill_level_pass_through()
"""The implemented bill-level pass-through coefficient: 0.3661. See above."""

ELECTRICITY_TO_GAS_PASS_THROUGH_RATIO: float = electricity_to_gas_pass_through_ratio()
"""The implemented ``phi_elec / phi_gas``: 0.6611, not ``psi`` = 0.85. See above."""


def solve_cap_calibration(
    gas_pct_change: float,
    phase_in_at_base: float,
    phase_in_at_anchor: float,
    *,
    observed_base_cap_gbp: float = OFGEM_CAP_JUL_2026_GBP,
    observed_anchor_cap_gbp: float = OFGEM_CAP_OCT_2026_GBP,
    anchor_vat_relief_gbp: float = OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP,
    **shares: float,
) -> CapCalibration:
    """Solve jointly for the pre-war counterfactual cap and the sustained fraction.

    **The round-3 root-cause fix.** The module used to take the observed
    July-September 2026 cap as the un-shocked baseline and read the whole
    conflict-attributable domestic move off the October-versus-July *step*. Both
    caps are post-shock, so that step is the difference between two partly
    shocked quarters, not the shock. The counterfactual has to be constructed.

    It is constructed here, and it is exactly identified. Write ``C0`` for the
    pre-war counterfactual cap and ``y`` for the steady-state proportional
    dual-fuel cap move at full pass-through. Each observed cap is the
    counterfactual scaled by the share of that move its own observation window
    prices::

        1663        = C0 x (1 + w_base   x y)      # 1 Jul - 30 Sep 2026
        1723 + 45   = C0 x (1 + w_anchor x y)      # 1 Oct - 31 Dec 2026

    Two equations, two unknowns, and ``w_base`` and ``w_anchor`` come from
    :func:`cap_phase_in_profile` — the observation windows, not a free
    parameter. Dividing the second by the first eliminates ``C0``::

        y  = (k - 1) / (w_anchor - k x w_base),   k = (1723 + 45) / 1663
        C0 = 1663 / (1 + w_base x y)

    and the sustained fraction follows as ``y / (gas_pct x phi_bill)`` with
    ``phi_bill`` from :func:`bill_level_pass_through`.

    On the defaults this gives ``C0 = £1,441`` and a sustained fraction of
    ``0.765`` — against the pre-revision £1,663 and 0.199. A referee working the
    same correction by hand on the old profile got £1,564 and a doubling of the
    domestic leg; this construction, which also replaces the linear ramp,
    goes further in the same direction. Both say the same thing about the paper:
    the domestic leg was understated by a factor of two or more and the
    motor-fuel share correspondingly overstated.

    **What identifies it, and how fragile that is.** ``C0`` is identified off
    the *difference* ``w_anchor - w_base``, i.e. off how much more of the shock
    the October window prices than the July one. That difference is a property of
    :data:`GAS_PEAK_MONTHLY_PROFILE`, which is a calibration. A flatter monthly
    path shrinks the difference and pushes ``C0`` down; a steeper one pushes it
    up. This is a real identification weakness, it is not hidden, and
    ``analysis/run_variants.py`` sweeps the profile so the range is in
    ``results/`` rather than in a caveat.

    Raises
    ------
    ValueError
        If the windows cannot separate the two observations (``w_anchor`` too
        close to ``k x w_base``), or if the implied sustained fraction is outside
        ``(0, 1]``, or if the implied counterfactual is not below both observed
        caps. Every one of those is a statement that the calibration has failed,
        and the module refuses to return a number that would silently encode it.
    """
    phi = bill_level_pass_through(**shares)
    k = (observed_anchor_cap_gbp + anchor_vat_relief_gbp) / observed_base_cap_gbp
    denominator = phase_in_at_anchor - k * phase_in_at_base
    if abs(denominator) < 1e-6:
        raise ValueError(
            "the two cap observations are not separated by their observation "
            f"windows (w_anchor={phase_in_at_anchor!r}, w_base={phase_in_at_base!r}, "
            f"ratio={k!r}): the pre-war counterfactual cap is not identified. "
            "This is exactly the degeneracy the pre-round-3 profile hit by "
            "assigning both quarters a phase-in of 1.0."
        )
    y = (k - 1.0) / denominator
    if y <= 0:
        raise ValueError(
            "the pre-war counterfactual cap is not identified: the observation "
            f"windows imply a non-positive steady-state cap move ({y!r}). This "
            "happens when the later window prices no more of the shock than the "
            "earlier one — the degeneracy the pre-round-3 profile hit by giving "
            "2026Q3 and 2026Q4 the same phase-in of 1.0."
        )
    prewar = observed_base_cap_gbp / (1.0 + phase_in_at_base * y)
    if not 0 < prewar < observed_base_cap_gbp:
        raise ValueError(
            f"implied pre-war counterfactual cap £{prewar:.0f} is not strictly "
            f"below the observed July cap £{observed_base_cap_gbp:.0f}"
        )
    if gas_pct_change <= 0 or phi <= 0:
        raise ValueError("gas_pct_change and the bill pass-through must be positive")
    sustained = y / (gas_pct_change * phi)
    if not 0 < sustained <= 1.0:
        raise ValueError(
            f"implied sustained fraction {sustained!r} is outside (0, 1]; the "
            "cap observations cannot be reproduced by damping the quoted peak"
        )
    return CapCalibration(
        prewar_cap_gbp=prewar,
        sustained_fraction=sustained,
        steady_state_dual_fuel_pct=y,
        modelled_base_cap_gbp=prewar * (1.0 + phase_in_at_base * y),
        modelled_anchor_cap_gbp=prewar * (1.0 + phase_in_at_anchor * y),
    )


#: Realised 2026 wholesale gas peak, +78p/therm on the 90p pre-war reference.
REALISED_GAS_PCT_CHANGE: float = 78.0 / PREWAR_NBP_PENCE_PER_THERM

_BASE_INDEX = CAP_QUARTER_LABELS.index(CAP_BASE_QUARTER)
_ANCHOR_INDEX = CAP_QUARTER_LABELS.index(CAP_ANCHOR_QUARTER)

REALISED_CAP_CALIBRATION: CapCalibration = solve_cap_calibration(
    REALISED_GAS_PCT_CHANGE,
    CAP_PHASE_IN_PROFILE[_BASE_INDEX],
    CAP_PHASE_IN_PROFILE[_ANCHOR_INDEX],
)
"""The joint solution: pre-war counterfactual cap and sustained fraction."""

PREWAR_COUNTERFACTUAL_CAP_GBP: float = REALISED_CAP_CALIBRATION.prewar_cap_gbp
"""**The pre-war counterfactual cap, £1,441/yr.** The paper's domestic baseline.

What it is
----------
The typical dual-fuel default-tariff cap, on the revised (1 Jul 2026) TDCV
basis, that would have prevailed had the February-2026 conflict not happened —
a cap priced off a *pre-shock* forward curve. It is not observed, because the
counterfactual is not observed; it is constructed, and the construction is
:func:`solve_cap_calibration`, one division from two published caps and the two
observation windows that priced them.

Why it exists
-------------
Because the alternative was a contradiction. Before this revision the module
used the observed 1 Jul - 30 Sep 2026 cap (£1,663) as the un-shocked base while
:func:`cap_phase_in_profile` assigned that same quarter a phase-in of 1.0. The
conflict-attributable move was therefore a Q4-versus-Q3 step of +6.31%, the
domestic leg came out at **£67 per household per year** for a shock that took UK
wholesale gas 78p/therm above pre-war, and understating the domestic leg
mechanically inflated motor fuel's share of the total loss to 75.6%.

What it implies
---------------
The conflict-attributable October-2026 cap move is ``1768 / 1441 - 1``, about
**+22.7%**, not +6.31%; the July cap already carried +15.4% of it. The domestic
leg roughly trebles and motor fuel's share falls below a half.

What would move it
------------------
:data:`GAS_PEAK_MONTHLY_PROFILE`, mostly — see :func:`solve_cap_calibration` on
identification — and then :data:`CAP_LAG_QUARTERS` and the bill shares. All are
swept.
"""

BASELINE_CAP_GBP: float = PREWAR_COUNTERFACTUAL_CAP_GBP
"""Alias for :data:`PREWAR_COUNTERFACTUAL_CAP_GBP`, the cap levels are built from.

Retained under its old name because it is the field name on :class:`CapStep`.
Its *value* changed with this revision: it used to be the observed £1,663 July
cap, which is now :data:`OFGEM_CAP_JUL_2026_GBP`.
"""

REALISED_SUSTAINED_FRACTION: float = REALISED_CAP_CALIBRATION.sustained_fraction
"""Peak-to-sustained damping for the realised 2026 gas path: 0.765.

The brief's realised gas figure is a **peak**. This is the fraction of it that
is sustained long enough to be what the cap's observation windows average to,
solved jointly with :data:`PREWAR_COUNTERFACTUAL_CAP_GBP` so that the modelled
cap path reproduces *both* published caps.

**It is not comparable to the old 0.199.** That number was doing two jobs: it
carried the genuine peak-to-sustained damping *and* it absorbed the whole error
from treating a shocked quarter as the counterfactual, which is why it had to be
so small. Split properly, the shape of the damping lives in
:data:`GAS_PEAK_MONTHLY_PROFILE` and the level in this fraction, and the level
is 0.765 — close to the pump leg's 0.650 rather than a third of it. That the two
legs' damping fractions turn out to be *similar* once the baseline error is
removed is itself a result: the paper's motor-fuel majority depended on their
ratio, and most of that ratio was the bug.

Forward-looking scenarios specified as *sustained* levels (the NIESR pair) use
1.0 and are not calibrated to the cap.
"""

CAP_ANCHOR_PCT: float = (
    OFGEM_CAP_OCT_2026_GBP + OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP
) / PREWAR_COUNTERFACTUAL_CAP_GBP - 1.0
"""Conflict-attributable October-2026 cap move: **+22.7%** on the counterfactual.

Was +6.31%, measured against the observed July cap. That figure was a
Q4-versus-Q3 step between two shocked quarters being reported as the whole
conflict-attributable domestic move.
"""

CAP_BASE_PCT: float = OFGEM_CAP_JUL_2026_GBP / PREWAR_COUNTERFACTUAL_CAP_GBP - 1.0
"""Conflict-attributable July-2026 cap move: +15.4% on the counterfactual.

The part of the shock the old specification silently assigned to the baseline.
"""

CAP_VALIDATION: Mapping[str, float] = {
    "observed_jul_2026_gbp": OFGEM_CAP_JUL_2026_GBP,
    "modelled_jul_2026_gbp": REALISED_CAP_CALIBRATION.modelled_base_cap_gbp,
    "observed_oct_2026_vat_adjusted_gbp": (
        OFGEM_CAP_OCT_2026_GBP + OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP
    ),
    "modelled_oct_2026_gbp": REALISED_CAP_CALIBRATION.modelled_anchor_cap_gbp,
    "prewar_counterfactual_gbp": PREWAR_COUNTERFACTUAL_CAP_GBP,
    "sustained_fraction": REALISED_SUSTAINED_FRACTION,
    "steady_state_dual_fuel_pct": (REALISED_CAP_CALIBRATION.steady_state_dual_fuel_pct),
}
"""The observed caps and the caps the calibrated path produces, side by side.

The two pairs agree to floating-point, by construction — the calibration is
exactly identified, so this is an identity check rather than a fit. It is
persisted anyway: an identity that holds by construction is precisely the thing
that breaks silently when someone changes a share or a profile, and the tests
assert it.
"""


def sustained_fraction_for_cap_anchor(
    gas_pct_change: float,
    phase_in_at_anchor: float,
    target_cap_pct: float = CAP_ANCHOR_PCT,
    **shares: float,
) -> float:
    """Sustained fraction reproducing ``target_cap_pct`` at the anchor quarter.

    The single-observation solver, retained because the *symmetric* scenario and
    the appendix both need to ask "what fraction would reproduce this one cap
    move, holding the counterfactual fixed?". It is no longer how the central
    case is calibrated: with a pre-war counterfactual to identify, one
    observation is not enough, and :func:`solve_cap_calibration` uses both.

    ``target_cap_pct`` defaults to :data:`CAP_ANCHOR_PCT`, which is now measured
    against :data:`PREWAR_COUNTERFACTUAL_CAP_GBP` rather than against a shocked
    quarter, so this function and the joint solver agree by construction.
    """
    phi = bill_level_pass_through(**shares)
    denominator = phi * gas_pct_change * phase_in_at_anchor
    if denominator <= 0:
        raise ValueError(
            "cannot calibrate the sustained fraction: the anchor quarter carries "
            f"no pass-through (phase_in={phase_in_at_anchor!r}, "
            f"gas_pct_change={gas_pct_change!r})"
        )
    return target_cap_pct / denominator


# ---------------------------------------------------------------------------
# Component paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OilPath:
    """Brent crude price level and change versus the pre-war reference.

    Attributes
    ----------
    level_usd_per_bbl:
        Scenario Brent level, US$/bbl.
    prewar_usd_per_bbl:
        Reference level the change is measured from; defaults to
        :data:`PREWAR_BRENT_USD_PER_BBL`.
    """

    level_usd_per_bbl: float
    prewar_usd_per_bbl: float = PREWAR_BRENT_USD_PER_BBL

    @property
    def change_usd_per_bbl(self) -> float:
        """Absolute move versus pre-war, US$/bbl."""
        return self.level_usd_per_bbl - self.prewar_usd_per_bbl

    @property
    def pct_change(self) -> float:
        """Proportional move versus pre-war (0.57 == +57%)."""
        return self.change_usd_per_bbl / self.prewar_usd_per_bbl


@dataclass(frozen=True, slots=True)
class GasPath:
    """NBP wholesale gas level and change, in pence per therm.

    Levels are quoted in p/therm because that is the unit the brief and the UK
    market use (+78p/therm in 2026 against +300p/therm in 2022). Conversion to
    p/kWh is 1 therm = 29.3071 kWh and belongs downstream, not here.

    Attributes
    ----------
    change_pence_per_therm:
        Move versus pre-war, p/therm. This is the primary field: the brief gives
        the gas shock in levels, not percentages.
    prewar_pence_per_therm:
        Reference level; defaults to :data:`PREWAR_NBP_PENCE_PER_THERM`.
    """

    change_pence_per_therm: float
    prewar_pence_per_therm: float = PREWAR_NBP_PENCE_PER_THERM

    @property
    def level_pence_per_therm(self) -> float:
        """Scenario NBP level, p/therm."""
        return self.prewar_pence_per_therm + self.change_pence_per_therm

    @property
    def pct_change(self) -> float:
        """Proportional wholesale gas move versus pre-war (0.867 == +86.7%)."""
        return self.change_pence_per_therm / self.prewar_pence_per_therm


@dataclass(frozen=True, slots=True)
class PumpPricePath:
    """Retail road-fuel price changes, petrol and diesel kept separate.

    Petrol and diesel are separate variables in PolicyEngine UK
    (``parameters/household/consumption/fuel/prices/{petrol,diesel}.yaml``) and
    moved very differently in 2026 (+20% vs +36%): diesel is the distillate most
    exposed to Gulf refinery and Hormuz shipping disruption. Collapsing them into
    a single "fuel" percentage — as the prior repo did — discards a real and
    distributionally meaningful asymmetry, since diesel-vehicle ownership is not
    uniform across the income distribution or between urban and rural households.

    Attributes
    ----------
    petrol_pct_change, diesel_pct_change:
        Proportional retail pump-price changes versus pre-war (0.20 == +20%).
    """

    petrol_pct_change: float
    diesel_pct_change: float


@dataclass(frozen=True, slots=True)
class PassThroughAssumptions:
    """Wholesale-to-retail mapping for one scenario.

    Defaults are the module-level constants; a scenario overrides only what it
    needs to. Every field is a documented, sweepable parameter — the point of
    this class is that no pass-through number is buried in an expression.

    Attributes
    ----------
    wholesale_share_gas_bill:
        Wholesale cost share of a gas bill. See
        :data:`WHOLESALE_SHARE_GAS_BILL`.
    wholesale_share_electricity_bill:
        Wholesale cost share of an electricity bill. See
        :data:`WHOLESALE_SHARE_ELECTRICITY_BILL`.
    marginal_pricing_share:
        Share of time gas sets the GB electricity price. See
        :data:`MARGINAL_PRICING_SHARE`. Set to 1.0 to recover the naive
        symmetric-shock assumption.
    gas_share_of_dual_fuel_bill:
        £ weight of gas in the typical dual-fuel cap, used only for
        re-aggregation. See :data:`GAS_SHARE_OF_DUAL_FUEL_BILL`.
    lag_quarters:
        Quarters to full pass-through. See :data:`CAP_LAG_QUARTERS`.
    phase_in_profile:
        Per-quarter realised fraction of full pass-through. See
        :data:`CAP_PHASE_IN_PROFILE`.
    sustained_fraction:
        Fraction of the quoted wholesale move that is sustained long enough to
        enter the cap's observation window. 1.0 for scenarios specified as
        sustained levels; see :data:`REALISED_SUSTAINED_FRACTION` for the
        realised path.
    pump_sustained_fraction:
        Fraction of the quoted **pump-price** peak that is sustained across the
        modelled year. Defaults to 1.0, which reproduces the undamped-peak
        specification; see :data:`REALISED_PUMP_SUSTAINED_FRACTION`.
    consumption_weights_gas, consumption_weights_electricity:
        Share of annual consumption of each fuel falling in each modelled
        quarter, used to turn the quarterly phase-in profile into the single
        annual factor a household actually pays. See
        :data:`QUARTERLY_CONSUMPTION_WEIGHTS_GAS`.
    """

    wholesale_share_gas_bill: float = WHOLESALE_SHARE_GAS_BILL
    wholesale_share_electricity_bill: float = WHOLESALE_SHARE_ELECTRICITY_BILL
    marginal_pricing_share: float = MARGINAL_PRICING_SHARE
    gas_share_of_dual_fuel_bill: float = GAS_SHARE_OF_DUAL_FUEL_BILL
    lag_quarters: float = CAP_LAG_QUARTERS
    phase_in_profile: tuple[float, ...] = CAP_PHASE_IN_PROFILE
    sustained_fraction: float = 1.0
    pump_sustained_fraction: float = 1.0
    consumption_weights_gas: tuple[float, ...] = QUARTERLY_CONSUMPTION_WEIGHTS_GAS
    consumption_weights_electricity: tuple[float, ...] = (
        QUARTERLY_CONSUMPTION_WEIGHTS_ELECTRICITY
    )

    def __post_init__(self) -> None:
        if not self.phase_in_profile:
            raise ValueError("phase_in_profile must be non-empty")
        for weights_name in (
            "consumption_weights_gas",
            "consumption_weights_electricity",
        ):
            weights = getattr(self, weights_name)
            if len(weights) != len(self.phase_in_profile):
                raise ValueError(
                    f"{weights_name} ({len(weights)}) and phase_in_profile "
                    f"({len(self.phase_in_profile)}) must be the same length"
                )
            if any(w < 0 for w in weights):
                raise ValueError(f"{weights_name} must be non-negative")
            if sum(weights) <= 0:
                raise ValueError(f"{weights_name} must not sum to zero")
        for name in (
            "wholesale_share_gas_bill",
            "wholesale_share_electricity_bill",
            "marginal_pricing_share",
            "gas_share_of_dual_fuel_bill",
            "sustained_fraction",
            "pump_sustained_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")

    # -- steady-state retail shocks -------------------------------------

    def steady_state_gas_shock(self, gas: GasPath) -> float:
        """Full-pass-through proportional change in the **retail gas** unit price.

        ``sustained_fraction x wholesale_share_gas_bill x gas.pct_change``.
        """
        return self.sustained_fraction * self.wholesale_share_gas_bill * gas.pct_change

    def steady_state_electricity_shock(self, gas: GasPath) -> float:
        """Full-pass-through proportional change in the **retail electricity** price.

        A gas wholesale move reaches electricity twice-attenuated: once by
        :attr:`marginal_pricing_share` (gas only sets the power price ~85% of
        the time) and once by :attr:`wholesale_share_electricity_bill`
        (wholesale is a smaller slice of an electricity bill than of a gas
        bill). The result is materially smaller than the gas shock — that
        asymmetry is the modelling choice the paper turns on.
        """
        return (
            self.sustained_fraction
            * self.marginal_pricing_share
            * self.wholesale_share_electricity_bill
            * gas.pct_change
        )

    # -- annual (phase-in-averaged) retail shocks ------------------------

    def _annual_phase_in(self, weights: tuple[float, ...]) -> float:
        """Consumption-weighted average of the quarterly phase-in profile.

        This is the paper's Step 1 as written: "the annual price faced by a
        household is the consumption-weighted average of the quarterly cap
        levels prevailing over the modelled year, not the peak". Because each
        quarter's cap level is the steady-state shock scaled by that quarter's
        phase-in weight, and because the cap enters the bill linearly, the
        consumption-weighted average of the levels is the steady-state shock
        scaled by the consumption-weighted average of the profile — which is
        what this returns.
        """
        total = float(sum(weights))
        return float(
            sum(w * p for w, p in zip(weights, self.phase_in_profile, strict=True))
            / total
        )

    @property
    def annual_phase_in_gas(self) -> float:
        """Window-average gas phase-in factor: 0.7972 on the defaults.

        "Annual" means over :data:`MODELLED_WINDOW_LABEL` — the same twelve
        months the pump leg is damped over. Both legs are now on one window.
        """
        return self._annual_phase_in(self.consumption_weights_gas)

    @property
    def annual_phase_in_electricity(self) -> float:
        """Window-average electricity phase-in factor: 0.8028 on the defaults."""
        return self._annual_phase_in(self.consumption_weights_electricity)

    def annual_gas_shock(self, gas: GasPath) -> float:
        """Steady-state gas shock damped to its annual, phase-in-averaged level."""
        return self.annual_phase_in_gas * self.steady_state_gas_shock(gas)

    def annual_electricity_shock(self, gas: GasPath) -> float:
        """Steady-state electricity shock damped to its annual average level."""
        return self.annual_phase_in_electricity * self.steady_state_electricity_shock(
            gas
        )

    # -- pump prices ----------------------------------------------------

    def sustained_pump_changes(self, pump: PumpPricePath) -> tuple[float, float]:
        """Damped (petrol, diesel) proportional pump moves for the modelled year.

        The quoted moves are peaks; multiplying by
        :attr:`pump_sustained_fraction` converts them to the annual-average
        uplift actually borne by households. With the default 1.0 this is the
        identity, so existing scenarios are unchanged.
        """
        return (
            self.pump_sustained_fraction * pump.petrol_pct_change,
            self.pump_sustained_fraction * pump.diesel_pct_change,
        )


@dataclass(frozen=True, slots=True)
class CapStep:
    """One quarterly step of the implied Ofgem default tariff cap path.

    Attributes
    ----------
    quarter:
        Ofgem cap period label, e.g. ``"2027Q1"``.
    phase_in:
        Fraction of full steady-state pass-through realised in this quarter.
    gas_pct_change, electricity_pct_change:
        Retail unit-price changes for this quarter, versus pre-war.
    cap_gbp:
        Implied typical dual-fuel cap level, £/yr, on the revised TDCV basis.
    baseline_cap_gbp:
        Pre-shock cap the level is built from.
    """

    quarter: str
    phase_in: float
    gas_pct_change: float
    electricity_pct_change: float
    cap_gbp: float
    baseline_cap_gbp: float = BASELINE_CAP_GBP

    @property
    def cap_change_gbp(self) -> float:
        """Cap move versus the pre-shock baseline, £/yr."""
        return self.cap_gbp - self.baseline_cap_gbp

    @property
    def cap_pct_change(self) -> float:
        """Cap move versus the pre-shock baseline, proportional."""
        return self.cap_change_gbp / self.baseline_cap_gbp


@dataclass(frozen=True, slots=True)
class RetailEnergyShock:
    """Steady-state retail shock factors, gas and electricity kept apart.

    These are the multipliers applied downstream to PolicyEngine UK's separate
    ``gas_consumption`` and ``electricity_consumption`` variables. Their
    inequality is the point; never collapse them into ``domestic_energy_consumption``.
    """

    gas_pct_change: float
    electricity_pct_change: float

    @property
    def gas_factor(self) -> float:
        """Multiplicative factor on the retail gas bill (1.39 == +39%)."""
        return 1.0 + self.gas_pct_change

    @property
    def electricity_factor(self) -> float:
        """Multiplicative factor on the retail electricity bill."""
        return 1.0 + self.electricity_pct_change


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    """One macro state of the world, fully specified and self-documenting.

    Attributes
    ----------
    key:
        Stable machine identifier; also the key in :data:`SCENARIOS`.
    label:
        Short human-readable name for figures and tables.
    description:
        One-paragraph narrative of the state of the world.
    oil, gas, pump:
        The three price paths.
    pass_through:
        Wholesale-to-retail mapping used to derive the cap path and the retail
        shock factors.
    source:
        Citation string for every number in the scenario. Required, non-empty:
        no scenario enters the paper without a source.
    notes:
        Free text recording author judgement calls a referee should see —
        mappings that are inference rather than quotation.
    """

    key: str
    label: str
    description: str
    oil: OilPath
    gas: GasPath
    pump: PumpPricePath
    source: str
    notes: str = ""
    pass_through: PassThroughAssumptions = field(default_factory=PassThroughAssumptions)
    baseline_cap_gbp: float = BASELINE_CAP_GBP
    quarter_labels: tuple[str, ...] = CAP_QUARTER_LABELS

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(f"scenario {self.key!r} must carry a source citation")
        if len(self.quarter_labels) != len(self.pass_through.phase_in_profile):
            raise ValueError(
                f"scenario {self.key!r}: quarter_labels "
                f"({len(self.quarter_labels)}) and phase_in_profile "
                f"({len(self.pass_through.phase_in_profile)}) must be the same length"
            )

    # -- derived ---------------------------------------------------------

    @property
    def retail_shock(self) -> RetailEnergyShock:
        """Steady-state (full-pass-through) retail gas and electricity shocks."""
        return RetailEnergyShock(
            gas_pct_change=self.pass_through.steady_state_gas_shock(self.gas),
            electricity_pct_change=(
                self.pass_through.steady_state_electricity_shock(self.gas)
            ),
        )

    @property
    def annual_retail_shock(self) -> RetailEnergyShock:
        """**The domestic shock a household actually pays over the modelled year.**

        The steady-state shock scaled by the consumption-weighted average of the
        quarterly phase-in profile, separately for each fuel — the paper's Step 1
        (``docs/FIXES.md`` decision D2). :attr:`retail_shock` charges the
        steady-state (full-pass-through) level for twelve months, which is the
        peak the paper explicitly says it does not use; it is retained as the
        labelled steady-state alternative.

        Motor fuel is deliberately *not* touched by this: pump prices track spot
        crude within weeks rather than through a lagged cap window, so the fuel
        leg carries its own annual damping
        (:attr:`PassThroughAssumptions.pump_sustained_fraction`) instead.
        """
        return RetailEnergyShock(
            gas_pct_change=self.pass_through.annual_gas_shock(self.gas),
            electricity_pct_change=(
                self.pass_through.annual_electricity_shock(self.gas)
            ),
        )

    @property
    def sustained_pump_changes(self) -> tuple[float, float]:
        """(petrol, diesel) proportional moves after pump damping.

        This — not the raw :attr:`pump` peaks — is what the incidence
        calculation charges households for a full year.
        """
        return self.pass_through.sustained_pump_changes(self.pump)

    @property
    def cap_path(self) -> tuple[CapStep, ...]:
        """Implied quarterly Ofgem cap path.

        For each modelled quarter the steady-state gas and electricity retail
        shocks are scaled by that quarter's phase-in weight, then re-aggregated
        into a single dual-fuel cap level using
        :attr:`PassThroughAssumptions.gas_share_of_dual_fuel_bill`. This is the
        brief's Step 1 in full: lagged, quantised into quarterly steps, and
        asymmetric between the two fuels.
        """
        steady = self.retail_shock
        gas_weight = self.pass_through.gas_share_of_dual_fuel_bill
        steps: list[CapStep] = []
        for quarter, phase_in in zip(
            self.quarter_labels, self.pass_through.phase_in_profile, strict=True
        ):
            gas_pct = steady.gas_pct_change * phase_in
            elec_pct = steady.electricity_pct_change * phase_in
            dual_fuel_pct = gas_weight * gas_pct + (1.0 - gas_weight) * elec_pct
            steps.append(
                CapStep(
                    quarter=quarter,
                    phase_in=phase_in,
                    gas_pct_change=gas_pct,
                    electricity_pct_change=elec_pct,
                    cap_gbp=self.baseline_cap_gbp * (1.0 + dual_fuel_pct),
                    baseline_cap_gbp=self.baseline_cap_gbp,
                )
            )
        return tuple(steps)

    def cap_step(self, quarter: str) -> CapStep:
        """Return the :class:`CapStep` for ``quarter`` (e.g. ``"2027Q1"``)."""
        for step in self.cap_path:
            if step.quarter == quarter:
                return step
        raise KeyError(
            f"scenario {self.key!r} has no cap step for {quarter!r}; "
            f"available: {', '.join(self.quarter_labels)}"
        )

    @property
    def peak_cap_gbp(self) -> float:
        """Highest modelled cap level across the quarterly path, £/yr."""
        return max(step.cap_gbp for step in self.cap_path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_NIESR_SOURCE = (
    "NIESR, UK Economic Outlook, Spring 2026 (Apr 2026) and Summer 2026 "
    "(Jul 2026). Spring baseline: GDP 0.9% in 2026, CPI averaging 3.0% and "
    "peaking 4.1% in Jan 2027; adverse scenario conditioned on $140/bbl gives "
    "CPI ~5% and a possible recession in H2-2026/H1-2027. Summer baseline: GDP "
    "1.1% in 2026 and 2027, CPI averaging 3.1% and peaking 3.8% in Feb 2027; "
    "downside conditioned on oil and gas +50% sustained gives GDP ~0.2pp lower "
    "and CPI peaking near 5%."
)

_REALISED_SOURCE = (
    "Research brief, 'The shock (context, as of August 2026)', from ICE Brent, "
    "NBP forward curve and DESNZ weekly road fuel prices: UK gas peaked "
    "+78p/therm above pre-war (against +300p/therm in 2022); Brent peaked "
    "+$42/bbl (+57%), comparable to 2022's +$35/bbl; petrol +20%, diesel +36%. "
    "Observed caps: Ofgem default tariff cap £1,663/yr for 1 Jul-30 Sep 2026 "
    "and £1,723/yr for 1 Oct-31 Dec 2026 (confirmed 26 Aug 2026), both on the "
    "revised TDCV basis; the October cap is held about £45 lower by Ofgem's "
    "temporary removal of VAT on electricity from 1 Oct 2026, so its "
    "conflict-comparable level is £1,768. BOTH are post-shock observations and "
    "neither is the counterfactual: they jointly identify the pre-war "
    "counterfactual cap of £1,441 (PREWAR_COUNTERFACTUAL_CAP_GBP) and the "
    "sustained fraction, via solve_cap_calibration. The conflict-attributable "
    "October move is +22.7% against that counterfactual; the +6.31% this "
    "scenario previously used was the October-versus-July step between two "
    "already-shocked quarters."
)


SCENARIOS: Mapping[str, Scenario] = {
    "niesr_baseline": Scenario(
        key="niesr_baseline",
        label="NIESR baseline ($103/bbl)",
        description=(
            "NIESR's central conditioning path for the Spring and Summer 2026 "
            "Outlooks: the conflict persists but Hormuz stays open on average, "
            "Brent settles near $103/bbl, and CPI peaks at 3.8-4.1% around the "
            "turn of 2026-27 before returning toward target. The cap rises but "
            "does not approach 2022 territory."
        ),
        oil=OilPath(level_usd_per_bbl=103.0),
        gas=GasPath(change_pence_per_therm=20.0),
        pump=PumpPricePath(petrol_pct_change=0.14, diesel_pct_change=0.22),
        pass_through=PassThroughAssumptions(sustained_fraction=1.0),
        source=_NIESR_SOURCE,
        notes=(
            "NIESR publish the oil conditioning path ($103/bbl) but not a UK "
            "wholesale gas level, and not pump prices. Author mapping, stated "
            "so a referee can reject it: gas +20p/therm (roughly +22% on the "
            "90p pre-war reference) is set at about a quarter of the realised "
            "2026 peak, consistent with a sustained-but-unstressed market; "
            "petrol +14% and diesel +22% preserve the realised 2026 "
            "diesel-over-petrol ratio (36/20 = 1.8) scaled to this oil move "
            "(+40% versus the realised +57%). Because the path is specified as "
            "sustained, sustained_fraction is 1.0 — so a smaller headline gas "
            "move can still produce a larger cap effect than the realised "
            "scenario's damped peak."
        ),
    ),
    "niesr_adverse": Scenario(
        key="niesr_adverse",
        label="NIESR adverse ($140/bbl)",
        description=(
            "NIESR's adverse scenario: sustained Hormuz disruption takes Brent "
            "to $140/bbl, CPI peaks near 5%, and recession in H2-2026/H1-2027 "
            "becomes possible. Aligned with the Summer 2026 downside "
            "conditioning of oil and gas +50% sustained, this is the scenario "
            "in which the Autumn Budget 2026 support decision is binding."
        ),
        oil=OilPath(level_usd_per_bbl=140.0),
        gas=GasPath(change_pence_per_therm=45.0),
        pump=PumpPricePath(petrol_pct_change=0.32, diesel_pct_change=0.50),
        pass_through=PassThroughAssumptions(sustained_fraction=1.0),
        source=_NIESR_SOURCE,
        notes=(
            "Gas +45p/therm implements the Summer 2026 downside's explicit "
            "'oil and gas +50% sustained' on the 90p pre-war reference; note "
            "this is a sustained +50%, well short of the +86.7% realised peak "
            "but far more consequential for the cap. Pump prices scale the "
            "realised diesel/petrol ratio to a +90% oil move. $140/bbl is "
            "NIESR's Spring 2026 adverse figure; the Summer downside is "
            "expressed as a percentage rather than a level, and the two are "
            "treated here as the same state of the world."
        ),
    ),
    "realised_2026": Scenario(
        key="realised_2026",
        label="Realised 2026 shock",
        description=(
            "The shock as it actually happened between late February and "
            "August 2026: Brent peaking +$42/bbl (+57%), UK wholesale gas "
            "peaking +78p/therm, petrol +20% and diesel +36% at the pump. "
            "This is the paper's central estimate — the counterfactual is not "
            "hypothetical, it is the 2026 the UK has already lived through. "
            "Because every quoted figure is a *peak* rather than a sustained "
            "level, both channels are damped to the part of the peak that "
            "actually reaches households over the modelled year: the gas leg "
            "by a fraction solved jointly with the pre-war counterfactual cap "
            "so that the modelled path reproduces BOTH published caps (£1,663 "
            "for July-September and £1,768 VAT-adjusted for October-December), "
            "and the pump leg by a peak-to-window-average "
            "fraction derived over the same twelve months. The undamped-pump "
            "variant is "
            "reported separately as realised_2026_peak_fuel."
        ),
        oil=OilPath(level_usd_per_bbl=PREWAR_BRENT_USD_PER_BBL + 42.0),
        gas=GasPath(change_pence_per_therm=78.0),
        pump=PumpPricePath(petrol_pct_change=0.20, diesel_pct_change=0.36),
        pass_through=PassThroughAssumptions(
            sustained_fraction=REALISED_SUSTAINED_FRACTION,
            pump_sustained_fraction=REALISED_PUMP_SUSTAINED_FRACTION,
        ),
        source=_REALISED_SOURCE,
        notes=(
            "Oil level is constructed as pre-war + $42 so that the +57% in the "
            "brief holds by construction. The gas percentage (+86.7%) depends "
            "entirely on PREWAR_NBP_PENCE_PER_THERM = 90p, which is an "
            "assumption, not an observation. Both legs are annualised over ONE "
            "window, MODELLED_WINDOW_LABEL (March 2026 - February 2027): the "
            "cap phase-in weights, the seasonal consumption weights and the "
            "pump damping fraction are all derived from it. "
            "sustained_fraction = 0.765 is calibrated, not estimated: it is "
            "solved by solve_cap_calibration jointly with the pre-war "
            "counterfactual cap of £1,441, so that the modelled cap path "
            "reproduces both published 2026 caps rather than treating one of "
            "them as the un-shocked base. It replaces 0.199, which was small "
            "chiefly because it was absorbing that baseline error. "
            "pump_sustained_fraction = 0.650 is likewise a calibration, the mean "
            "of PUMP_PEAK_MONTHLY_PROFILE over the same twelve months. The two "
            "fractions are still different objects - one is a peak-to-sustained "
            "level feeding a forward-curve window, one a peak-to-average - but "
            "they are now measured over the same year AND are close to one "
            "another, where before they differed by a factor of three. The "
            "earlier "
            "specification applied the pump peaks undamped for a full year while "
            "damping gas; that is a claim about lag, not duration, and it is "
            "retained as realised_2026_peak_fuel, an explicit upper bound on the "
            "fuel channel."
        ),
    ),
    "realised_2026_peak_fuel": Scenario(
        key="realised_2026_peak_fuel",
        label="Realised 2026 shock, undamped pump peak (upper bound)",
        description=(
            "Identical to realised_2026 in every respect except that the "
            "observed peak pump moves (petrol +20%, diesel +36%) are applied "
            "undamped across the whole modelled year (March 2026 - February "
            "2027, MODELLED_WINDOW_LABEL - not a calendar year). This is not a "
            "central "
            "estimate: it is an explicit **upper bound** on the motor-fuel "
            "channel, and on motor fuel's share of the total loss. It is "
            "reported so the sensitivity of the paper's headline finding to "
            "the pump-damping assumption is visible rather than buried, and it "
            "is the specification the pre-audit results were produced on."
        ),
        oil=OilPath(level_usd_per_bbl=PREWAR_BRENT_USD_PER_BBL + 42.0),
        gas=GasPath(change_pence_per_therm=78.0),
        pump=PumpPricePath(petrol_pct_change=0.20, diesel_pct_change=0.36),
        pass_through=PassThroughAssumptions(
            sustained_fraction=REALISED_SUSTAINED_FRACTION,
            pump_sustained_fraction=1.0,
        ),
        source=_REALISED_SOURCE,
        notes=(
            "Charging households the peak pump price for twelve months while "
            "damping the gas peak to REALISED_SUSTAINED_FRACTION is internally "
            "inconsistent, and "
            "docs/VALIDATION.md Check 2b rejects it as a central case. It is "
            "kept only as a bound. Any number taken from this scenario must be "
            "reported with the undamped-peak assumption stated in the same "
            "sentence."
        ),
    ),
    "realised_2026_symmetric": Scenario(
        key="realised_2026_symmetric",
        label="Realised 2026 shock, symmetric peak damping",
        description=(
            "Identical to realised_2026 except that a single common "
            "peak-to-annual-average damping fraction "
            "(SYMMETRIC_SUSTAINED_FRACTION = 0.650) is applied to both the "
            "wholesale gas leg and the pump leg, instead of the solved 0.765 for "
            "gas and 0.650 for fuel. This is a first-class specification, not a "
            "footnote: motor fuel's share of the modelled loss depends only on "
            "the ratio of the two fractions, so the asymmetric split is exactly "
            "what generates the paper's motor-fuel majority. At any common "
            "fraction the steady-state fuel share falls to about 44.5%, and the "
            "published 55.8-67.8% range was one-sided in that both of its "
            "endpoints raised the share."
        ),
        oil=OilPath(level_usd_per_bbl=PREWAR_BRENT_USD_PER_BBL + 42.0),
        gas=GasPath(change_pence_per_therm=78.0),
        pump=PumpPricePath(petrol_pct_change=0.20, diesel_pct_change=0.36),
        pass_through=PassThroughAssumptions(
            sustained_fraction=SYMMETRIC_SUSTAINED_FRACTION,
            pump_sustained_fraction=SYMMETRIC_SUSTAINED_FRACTION,
        ),
        source=_REALISED_SOURCE,
        notes=(
            "The two fractions in realised_2026 are calibrated to different "
            "instruments — 0.765 to the two published 2026 caps jointly with the "
            "pre-war counterfactual, 0.650 to a monthly profile of the "
            "realised pump path — and the paper's central claim "
            "turns on their ratio rather than on either level. This scenario "
            "removes the ratio by deriving the gas fraction the same way as the "
            "pump one (the twelve-month profile arithmetic in "
            "SYMMETRIC_SUSTAINED_FRACTION), and it therefore **breaks the "
            "Ofgem cap calibration**: the implied 2026Q4 cap is "
            "no longer reproduces the published October cap. That is the honest "
            "trade-off and it is why both "
            "specifications are reported rather than one being chosen. Note the "
            "fuel share here is not literally 44.5%, because the main "
            "specification also applies the Step 1 consumption-weighted "
            "phase-in to the domestic leg, which damps the domestic channel "
            "further; 44.5% is the steady-state symmetric figure."
        ),
    ),
}
"""Registry of macro scenarios, keyed by :attr:`Scenario.key`.

``niesr_baseline`` and ``niesr_adverse`` are the brief's forward-looking pair;
``realised_2026`` is the observed path and is the paper's central case.
"""


def on_window(scenario: Scenario, months: tuple[str, ...]) -> Scenario:
    """Re-annualise ``scenario`` onto a different twelve-month window.

    Round-3 finding 10. ``realised_2026_peak_fuel`` — the explicit upper bound on
    the motor-fuel channel — was only ever run on
    :data:`MODELLED_WINDOW_LABEL`, while the Resolution Foundation's £11bn
    comparator is **calendar 2026**. Only the central case was re-annualised onto
    the RF window, so the bracket the paper states around £11bn mixed windows: a
    lower end on one twelve months and an upper end on another. Any scenario can
    now be moved onto any window, so the bracketing can be stated like-for-like.

    Everything that depends on the window is rebuilt from it and nothing is
    carried over: the cap quarters, the seasonal consumption weights of each
    fuel, the observation-window phase-in profile and the pump damping fraction.
    The calibration (``sustained_fraction``) is *not* re-solved — it is a
    property of the wholesale path and the cap's observation windows, not of the
    window a household's year is measured over — so the modelled cap path still
    reproduces both published caps.

    Note that a window which is not a whole heating cycle no longer has
    consumption weights summing to one before normalisation;
    :func:`quarterly_consumption_weights` normalises, which is the right
    treatment for an *average* price but means the resulting annual figure is a
    twelve-month-equivalent rather than a total over a partial window.
    """
    quarters = quarters_of(months)
    return dataclasses.replace(
        scenario,
        key=f"{scenario.key}_{months[0][:4]}",
        quarter_labels=quarters,
        pass_through=dataclasses.replace(
            scenario.pass_through,
            phase_in_profile=cap_phase_in_profile(
                scenario.pass_through.lag_quarters, quarters
            ),
            pump_sustained_fraction=(
                pump_sustained_fraction(months)
                if scenario.pass_through.pump_sustained_fraction < 1.0
                else 1.0
            ),
            consumption_weights_gas=quarterly_consumption_weights(
                months, MONTHLY_CONSUMPTION_WEIGHTS_GAS
            ),
            consumption_weights_electricity=quarterly_consumption_weights(
                months, MONTHLY_CONSUMPTION_WEIGHTS_ELECTRICITY
            ),
        ),
    )


def scenario_keys() -> tuple[str, ...]:
    """Return the registered scenario keys, in registry order."""
    return tuple(SCENARIOS)


def get_scenario(key: str) -> Scenario:
    """Look up a scenario by key, with a helpful error on a miss.

    Parameters
    ----------
    key:
        A key from :func:`scenario_keys`.

    Raises
    ------
    KeyError
        If ``key`` is not registered.
    """
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(
            f"unknown scenario {key!r}; available: {', '.join(SCENARIOS)}"
        ) from None

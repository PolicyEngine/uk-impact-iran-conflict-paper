"""Macro scenario definitions for the 2026 Iran-conflict energy shock.

This module is **pure scenario data plus deterministic arithmetic**. It imports
nothing from PolicyEngine and touches no microdata, so it is importable and
testable without a Hugging Face token. Everything that requires the
microsimulation lives downstream in ``uk_iran_conflict.shocks``.

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
2. **The cap lags.** Ofgem's default tariff cap is set from an observation
   window on the forward wholesale curve and reset quarterly, so a wholesale
   move enters retail bills with a lag. The observation window closes about
   seven weeks before the charge period and covers roughly the preceding
   quarter, putting midpoint-to-midpoint at four to five months — about **1.5
   quarters**, which is :data:`CAP_LAG_QUARTERS`, and half the value used before
   this revision. The phase-in profile is *derived* from it by
   :func:`cap_phase_in_profile`, never written down separately.
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
  is not free: :data:`REALISED_SUSTAINED_FRACTION` is *solved* by
  :func:`sustained_fraction_for_cap_anchor` so that the modelled October-2026
  cap reproduces Ofgem's confirmed £1,723, and it therefore moves automatically
  with the lag, the window and the bill-share assumptions instead of silently
  going stale when they change.
* **Both legs are annualised over one window** (:data:`MODELLED_WINDOW_LABEL`).
  Before this revision the domestic leg was averaged over 2026Q4-2027Q3 while
  the pump damping fraction was derived over calendar 2026, and the sum was
  labelled "2026". All three round-2 referees found it independently.
* There is no demand response anywhere in this module. Quantities are held
  fixed downstream (PolicyEngine UK issue #1114 — no consumption elasticity),
  making every result a Deaton first-order upper bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "PREWAR_BRENT_USD_PER_BBL",
    "PREWAR_NBP_PENCE_PER_THERM",
    "BASELINE_CAP_GBP",
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

BASELINE_CAP_GBP: float = 1_663.0
"""Ofgem default tariff cap, 1 Jul - 30 Sep 2026, typical dual-fuel direct debit,
on the revised (1 Jul 2026) Typical Domestic Consumption Values. Approximately
£1,862 on the pre-July TDCV basis; the two are not comparable.
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


def cap_phase_in_profile(
    lag_quarters: float = CAP_LAG_QUARTERS,
    quarters: tuple[str, ...] | None = None,
    onset_month: str = SHOCK_ONSET_MONTH,
) -> tuple[float, ...]:
    """Fraction of full steady-state cap pass-through realised in each quarter.

    A wholesale move enters the cap through an averaging observation window, so
    it arrives spread over successive resets rather than in one step. Modelled
    as a linear ramp from zero at the shock onset to one at ``lag_quarters``
    after it, evaluated at each **cap quarter's midpoint**:

        ``phase_in(q) = clip((midpoint(q) - onset) / lag_quarters, 0, 1)``

    This replaces the hand-written ``(0.35, 0.85, 1.00, 0.90)`` tuple, which
    carried no relation to :data:`CAP_LAG_QUARTERS` at all. That disconnection is
    exactly why the paper's headline (which used the tuple) and its cap-lag
    appendix (which used the lag) reported the same specification as £304 and
    £205. Here there is one parameter and one function, so they reconcile by
    construction.

    The ramp is monotone and does not unwind within the window: the old tuple's
    fourth entry of 0.90 modelled the peak rolling out of the observation window
    a year later, which is outside :data:`MODELLED_WINDOW_LABEL` and was in any
    case a different claim from the lag.
    """
    if lag_quarters <= 0:
        raise ValueError(f"lag_quarters must be positive; got {lag_quarters!r}")
    if quarters is None:
        quarters = CAP_QUARTER_LABELS
    onset = _month_index(onset_month) / 3.0
    out: list[float] = []
    for quarter in quarters:
        midpoint = _quarter_index(quarter) + 0.5
        out.append(min(1.0, max(0.0, (midpoint - onset) / lag_quarters)))
    return tuple(out)


CAP_QUARTER_LABELS: tuple[str, ...] = quarters_of(MODELLED_WINDOW_MONTHS)
"""The five Ofgem cap periods the modelled year touches.

``2026Q1`` (March only), ``2026Q2``, ``2026Q3``, ``2026Q4`` (the Ofgem-confirmed
£1,723 anchor) and ``2027Q1`` (January and February 2027). Five rather than four
because a twelve-month window that does not start on a quarter boundary
straddles five quarters; the consumption weights in
:data:`QUARTERLY_CONSUMPTION_WEIGHTS_GAS` carry only the months actually inside
the window, so nothing is double-counted.
"""

CAP_PHASE_IN_PROFILE: tuple[float, ...] = cap_phase_in_profile()
"""Per-quarter realised fraction of full pass-through, derived from the lag.

``(0.0, 0.556, 1.0, 1.0, 1.0)`` at :data:`CAP_LAG_QUARTERS` = 1.5: nothing in the
quarter the shock lands in (the observation window for 2026Q1 closed before the
war began), a little over half by the 2026Q2 reset, full from 2026Q3 on.
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
"""Ofgem default tariff cap for 1 Oct - 31 Dec 2026, **confirmed 26 Aug 2026**.

Typical dual-fuel direct debit on the revised (1 Jul 2026) TDCV basis, the same
basis as :data:`BASELINE_CAP_GBP`. This supersedes the Cornwall Insight forecast
of £1,729 (19 Aug 2026) that the scenario previously calibrated to
(``docs/FIXES.md`` C15): a confirmed cap is a better anchor than a forecast, and
the difference is worth £6.
"""

OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP: float = 45.0
"""Cap reduction from Ofgem's temporary removal of VAT on electricity, 1 Oct 2026.

The October 2026 cap is held roughly £45 lower by a VAT measure that has nothing
to do with the conflict, and that breaks comparability with the July-September
cap the anchor is measured against. The conflict-attributable move is therefore
``(1723 + 45) / 1663 - 1``, not ``1723 / 1663 - 1``. Omitting it would attribute
a tax cut to the war and understate the domestic leg by roughly two fifths.

This also interacts directly with the scored VAT zero-rating instrument, since
part of that instrument may already be in the baseline; see
``docs/VALIDATION.md`` and ``docs/FIXES.md`` C15.
"""

CAP_ANCHOR_QUARTER: str = "2026Q4"
"""The cap quarter :data:`CAP_ANCHOR_PCT` prices, and the anchor is imposed at."""

CAP_ANCHOR_PCT: float = (
    OFGEM_CAP_OCT_2026_GBP + OFGEM_OCT_2026_ELECTRICITY_VAT_RELIEF_GBP
) / BASELINE_CAP_GBP - 1.0
"""Conflict-attributable October-2026 cap move: +6.31% on the July-September cap."""


def sustained_fraction_for_cap_anchor(
    gas_pct_change: float,
    phase_in_at_anchor: float,
    target_cap_pct: float = CAP_ANCHOR_PCT,
    *,
    wholesale_share_gas_bill: float = WHOLESALE_SHARE_GAS_BILL,
    wholesale_share_electricity_bill: float = WHOLESALE_SHARE_ELECTRICITY_BILL,
    marginal_pricing_share: float = MARGINAL_PRICING_SHARE,
    gas_share_of_dual_fuel_bill: float = GAS_SHARE_OF_DUAL_FUEL_BILL,
) -> float:
    """Solve for the sustained fraction that reproduces the cap anchor exactly.

    The cap move in the anchor quarter is linear in the sustained fraction::

        cap_pct = s x phase_in x gas_pct x [ w_g x ws_gas
                                             + (1 - w_g) x mps x ws_elec ]

    so the fraction is a division, not a search. Making it a *function* rather
    than a literal is the point: ``docs/FIXES.md`` A4 records that the anchor
    identifies only the **product** of the sustained fraction and the phase-in
    weight at the anchor quarter, and writing 0.36 as a constant while the
    phase-in profile changed underneath it silently broke the anchor. Here the
    anchor holds by construction for any lag, any window and any bill-share
    assumption, and the sweep in ``analysis/run_variants.py`` sweeps the split
    with the product held.
    """
    bracket = (
        gas_share_of_dual_fuel_bill * wholesale_share_gas_bill
        + (1.0 - gas_share_of_dual_fuel_bill)
        * marginal_pricing_share
        * wholesale_share_electricity_bill
    )
    denominator = bracket * gas_pct_change * phase_in_at_anchor
    if denominator <= 0:
        raise ValueError(
            "cannot calibrate the sustained fraction: the anchor quarter carries "
            f"no pass-through (phase_in={phase_in_at_anchor!r}, "
            f"gas_pct_change={gas_pct_change!r})"
        )
    return target_cap_pct / denominator


#: Realised 2026 wholesale gas peak, +78p/therm on the 90p pre-war reference.
REALISED_GAS_PCT_CHANGE: float = 78.0 / PREWAR_NBP_PENCE_PER_THERM

REALISED_SUSTAINED_FRACTION: float = sustained_fraction_for_cap_anchor(
    REALISED_GAS_PCT_CHANGE,
    CAP_PHASE_IN_PROFILE[CAP_QUARTER_LABELS.index(CAP_ANCHOR_QUARTER)],
)
"""Peak-to-cap-relevant damping for the realised 2026 path: 0.199.

The brief's realised figures are **peaks**; the cap responds to an averaged
forward window, not a spike, so only part of a peak reaches retail. This is the
fraction that makes the modelled 2026Q4 cap step reproduce
:data:`CAP_ANCHOR_PCT` — Ofgem's confirmed October-2026 cap, adjusted for the
unrelated electricity VAT relief.

**It is not comparable to the old 0.36.** That number was calibrated against a
phase-in weight of 0.35 in the anchor quarter, so what the anchor pinned was the
product 0.126. Under :data:`CAP_LAG_QUARTERS` = 1.5 the cap has reached full
pass-through by October 2026, so the anchor quarter's phase-in weight is 1.0 and
the same anchor implies a fraction of 0.199 — a *larger* product (0.199 against
0.126), because the anchor target also moved from Cornwall's +4.0% to Ofgem's
VAT-adjusted +6.31%. Forward-looking scenarios specified as *sustained* levels
(the NIESR pair) use 1.0 and are not anchored.
"""


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
    "Cap anchors: Ofgem default tariff cap £1,663/yr for 1 Jul-30 Sep 2026 and "
    "£1,723/yr for 1 Oct-31 Dec 2026 (confirmed 26 Aug 2026), both on the "
    "revised TDCV basis; the October cap is held about £45 lower by Ofgem's "
    "temporary removal of VAT on electricity from 1 Oct 2026, so the "
    "conflict-attributable move is +6.31%, not +3.61%. This supersedes the "
    "Cornwall Insight forecast of £1,729 (19 Aug 2026) the scenario previously "
    "calibrated to."
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
            "by a fraction calibrated to reproduce Ofgem's confirmed £1,723 "
            "October 2026 cap (+6.31% once its unrelated electricity VAT relief "
            "is added back), and the pump leg by a peak-to-window-average "
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
            "sustained_fraction = 0.199 is calibrated, not estimated: it is "
            "solved by sustained_fraction_for_cap_anchor so the 2026Q4 cap step "
            "reproduces Ofgem's confirmed £1,723 October cap once the unrelated "
            "electricity VAT relief (about £45) is added back. "
            "pump_sustained_fraction = 0.650 is likewise a calibration, the mean "
            "of PUMP_PEAK_MONTHLY_PROFILE over the same twelve months. The two "
            "fractions are still different objects - one is a cap-window share, "
            "one a peak-to-average - but they are now measured over the same "
            "year, which the round-2 referees found they were not. The earlier "
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
            "undamped for the full calendar year. This is not a central "
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
            "damping the gas peak to 0.36 is internally inconsistent, and "
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
            "wholesale gas leg and the pump leg, instead of 0.199 for gas and "
            "0.650 for fuel. This is a first-class specification, not a "
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
            "instruments — 0.199 to Ofgem's confirmed October-2026 cap via the "
            "cap's forward-observation window, 0.650 to a monthly profile of the "
            "realised pump path — and the paper's central claim "
            "turns on their ratio rather than on either level. This scenario "
            "removes the ratio by deriving the gas fraction the same way as the "
            "pump one (the twelve-month profile arithmetic in "
            "SYMMETRIC_SUSTAINED_FRACTION), and it therefore **breaks the "
            "Ofgem October-2026 cap anchor**: the implied 2026Q4 cap step is "
            "far larger than the confirmed +6.31%. That is the honest "
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

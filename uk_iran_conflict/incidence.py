"""First-order incidence of an energy price shock, computed on the consumption side.

Why this module exists
----------------------
PolicyEngine UK has **no price channel**. Verified against the installed
release:

* ``gas_consumption``, ``electricity_consumption``, ``domestic_energy_consumption``
  and ``petrol_spending``/``diesel_spending`` are *input* variables — LCFS-imputed
  spend with no formula. No parameter reform moves them.
* ``gov.ofgem.energy_price_cap`` is read by exactly one variable,
  ``monthly_epg_consumption_level``, which is Energy Price Guarantee *subsidy*
  machinery, not a household price channel.
* ``gov.contrib.policyengine.economy.energy_bills`` is a dead parameter: no
  variable reads it.
* ``petrol_price`` *is* parameter-driven, but because ``petrol_spending`` is an
  input, raising the price parameter holds spend fixed and **cuts litres** —
  implicitly unit-elastic demand, and it lowers fuel duty. That is the opposite
  of a first-order price shock.
* There is no Warm Home Discount in the model at all — no parameter, no variable.

So the shock is applied here, on the consumption side, exactly as the paper's
Step 2 describes: :math:`\\Delta c = q \\times \\Delta p` with quantities held
fixed. This is the Deaton first-order approximation and an explicit **upper
bound** on the true welfare loss; the elasticity module supplies the robustness
check that shaves it.

Holding quantities fixed also fixes litres, so fuel duty (a specific, per-litre
duty) is unchanged by construction and only the VAT-inclusive spend rises.

What the tax-benefit system is used for
---------------------------------------
The shock itself never enters ``household_net_income`` — energy spend is not a
tax-benefit object. PolicyEngine supplies the *baseline*: net income, imputed
energy quantities, equivalisation, deciles, poverty and the household weights.
Policy responses that are genuine transfers are then scored against that
baseline (see :mod:`uk_iran_conflict.policies`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Variables pulled from the baseline simulation. Kept explicit so a rename in
#: policyengine-uk fails loudly here rather than silently yielding zeros.
BASELINE_VARIABLES: tuple[str, ...] = (
    "household_net_income",
    "household_weight",
    "household_count_people",
    "gas_consumption",
    "electricity_consumption",
    "petrol_spending",
    "diesel_spending",
    "household_income_decile",
    "equiv_hbai_household_net_income_ahc",
    # Round-3: the concept ``household_income_decile`` actually ranks on. Read
    # from the installed package rather than inferred — see
    # :func:`decile_concept_audit`. Without it the audit could not put the true
    # concept among its own candidates.
    "equiv_hbai_household_net_income",
    "household_equivalisation_ahc",
    "in_relative_poverty_ahc",
    "in_relative_poverty_bhc",
    "in_poverty_bhc",
    "in_poverty_ahc",
    "region",
    "country",
)

#: The income concept every "percentage of income" statistic divides by.
#:
#: ``"equivalised_ahc"`` is ``equiv_hbai_household_net_income_ahc``: HBAI
#: household net income **after housing costs**, divided by PolicyEngine UK's
#: ``household_equivalisation_ahc``. That scale is the HBAI/DWP AHC scale,
#: normalised so a childless couple equals 1.0:
#:
#: ===============================  =====
#: First adult                       0.58
#: Each additional adult             0.42
#: Each child aged 14 or over        0.42
#: Each child under 14               0.20
#: ===============================  =====
#:
#: (``policyengine_uk`` parameters ``household.demographic.equiv.ahc``.) This is
#: the concept the paper claims and the one the published distributional
#: literature uses, so it is the default. ``"unequivalised"`` is the raw
#: ``household_net_income`` the code used before ``docs/FIXES.md`` decision D1;
#: it is retained **only** to produce the robustness line that makes the change
#: visible.
INCOME_BASES: tuple[str, ...] = ("equivalised_ahc", "unequivalised")

DEFAULT_INCOME_BASIS: str = "equivalised_ahc"

#: How the domestic (gas and electricity) leg is annualised.
#:
#: ``"annual"`` is the paper's Step 1 and ``docs/FIXES.md`` decision D2: the
#: annual price faced by a household is the consumption-weighted average of the
#: quarterly cap levels prevailing over the modelled year, not the peak. See
#: :attr:`uk_iran_conflict.scenarios.Scenario.annual_retail_shock`.
#: ``"steady_state"`` charges the full-pass-through level for twelve months —
#: what the code did before D2. It is kept as a labelled alternative, never as
#: the headline. Motor fuel is unaffected by either: pump prices do not lag, so
#: the fuel leg carries its own annual damping.
DOMESTIC_BASES: tuple[str, ...] = ("annual", "steady_state")

DEFAULT_DOMESTIC_BASIS: str = "annual"

#: The reduced VAT rate on domestic fuel and power (HMRC). Domestic energy is
#: VAT-inclusive in LCFS spend, so zero-rating returns this fraction of the
#: VAT-inclusive bill: 1 - 1/1.05.
DOMESTIC_FUEL_VAT_RATE = 0.05


@dataclass(frozen=True)
class Baseline:
    """Baseline household arrays, all household-level and weight-aligned."""

    net_income: np.ndarray
    weight: np.ndarray
    people: np.ndarray
    gas: np.ndarray
    electricity: np.ndarray
    petrol: np.ndarray
    diesel: np.ndarray
    decile: np.ndarray
    equiv_income_ahc: np.ndarray
    in_poverty_bhc: np.ndarray
    in_poverty_ahc: np.ndarray
    region: np.ndarray
    country: np.ndarray
    #: The HBAI AHC equivalisation factor itself (couple = 1.0; see
    #: :data:`INCOME_BASES` for the scale). Optional so a synthetic baseline can
    #: be built without it; ``None`` is read as an unequivalised 1.0 everywhere.
    equivalisation_ahc: np.ndarray | None = None
    #: Equivalised HBAI net income **before** housing costs — the variable
    #: ``household_income_decile`` actually ranks on. Optional so a synthetic
    #: baseline can be built without it; :func:`decile_concept_audit` reports it
    #: as unavailable rather than substituting a proxy.
    equiv_income_bhc: np.ndarray | None = None
    #: Household-level means-tested benefit receipt indicator, from
    #: :func:`uk_iran_conflict.policies.means_tested_flag`. Optional: it needs a
    #: second pass over the microdata, so the run scripts attach it with
    #: ``dataclasses.replace`` rather than :func:`load_baseline` loading it.
    #: Required by the ``"mt_fuel_parity"`` calibration, which raises without it.
    means_tested: np.ndarray | None = None

    @property
    def equivalisation(self) -> np.ndarray:
        """The AHC equivalisation factor, defaulting to 1.0 when not loaded."""
        if self.equivalisation_ahc is None:
            return np.ones_like(self.net_income, dtype=float)
        return self.equivalisation_ahc

    @property
    def n(self) -> int:
        return len(self.net_income)

    @property
    def energy(self) -> np.ndarray:
        """Domestic energy spend (gas + electricity), £/yr."""
        return self.gas + self.electricity

    @property
    def motor_fuel(self) -> np.ndarray:
        """Motor fuel spend (petrol + diesel), £/yr."""
        return self.petrol + self.diesel


def load_baseline(dataset: str, period: int = 2026) -> Baseline:
    """Load the baseline arrays from a PolicyEngine UK microsimulation.

    ``dataset`` is a path to a populace/enhanced-FRS ``.h5``. The import is
    local so this module stays importable without microdata.
    """
    from policyengine_uk import Microsimulation  # noqa: PLC0415

    sim = Microsimulation(dataset=dataset)
    got: dict[str, np.ndarray] = {}
    for name in BASELINE_VARIABLES:
        got[name] = np.asarray(sim.calculate(name, period))
    return Baseline(
        net_income=got["household_net_income"],
        weight=got["household_weight"],
        people=got["household_count_people"],
        gas=got["gas_consumption"],
        electricity=got["electricity_consumption"],
        petrol=got["petrol_spending"],
        diesel=got["diesel_spending"],
        decile=got["household_income_decile"],
        equiv_income_ahc=got["equiv_hbai_household_net_income_ahc"],
        equiv_income_bhc=got["equiv_hbai_household_net_income"],
        in_poverty_bhc=got["in_relative_poverty_bhc"],
        in_poverty_ahc=got["in_relative_poverty_ahc"],
        region=got["region"],
        country=got["country"],
        equivalisation_ahc=got["household_equivalisation_ahc"],
    )


# --------------------------------------------------------------------------
# ONS-calibrated motor-fuel robustness variant
# --------------------------------------------------------------------------

#: ONS Family Spending FYE 2025 (LCFS, published 11 June 2026), motor-fuel
#: spend, £/yr, for the **bottom** and **top** gross-income deciles. Reported in
#: ``docs/VALIDATION.md`` Check 2d.
ONS_MOTOR_FUEL_D1_GBP: float = 318.0
ONS_MOTOR_FUEL_D10_GBP: float = 1_362.0

#: ONS Family Spending FYE 2025, Table A1: petrol, diesel and motor oils at
#: £18.40/week = £960/yr per household, all households. Used only as a
#: cross-check on the interpolated profile, never as the level we impose — see
#: :func:`rescale_motor_fuel_to_ons` for why the microdata total is preserved.
ONS_MOTOR_FUEL_MEAN_GBP: float = 960.0


def ons_motor_fuel_decile_targets() -> np.ndarray:
    """Target ONS motor-fuel spend by income decile, £/yr, deciles 1-10.

    ``docs/VALIDATION.md`` gives only the endpoints (D1 £318, D10 £1,362, a
    4.3x gradient) and the all-household mean (£960). The eight interior deciles
    are **interpolated, and that interpolation is an assumption of this
    robustness run, not a published statistic.**

    The interpolation is *log-linear* — a constant ratio between adjacent
    deciles, :math:`(1362/318)^{1/9} = 1.173` — rather than linear in levels.
    Two reasons, both stated so a referee can reject them:

    1. Motor-fuel spend is a roughly log-linear function of income across the
       distribution in every published UK budget-share table (it is driven by
       car access and mileage, both of which scale multiplicatively with
       income), so a constant-growth-factor profile is the natural shape.
    2. A linear-in-levels profile would put the implied all-household mean at
       £840, whereas the log-linear profile gives £801 — both below the ONS
       £960 mean, because ONS deciles are unequivalised *gross*-income deciles
       whose household sizes differ from ours. Neither reproduces the published
       mean exactly, which is precisely why this function is used to set decile
       **shares** and never the level.

    Returns
    -------
    np.ndarray
        Ten target means, £/yr, in decile order.
    """
    ratio = (ONS_MOTOR_FUEL_D10_GBP / ONS_MOTOR_FUEL_D1_GBP) ** (1.0 / 9.0)
    return ONS_MOTOR_FUEL_D1_GBP * ratio ** np.arange(10.0)


def ons_motor_fuel_scale_factors(base: Baseline) -> np.ndarray:
    """Per-decile rescaling factors that impose the ONS motor-fuel profile.

    Index ``d - 1`` holds the factor for decile ``d``.
    """
    targets = ons_motor_fuel_decile_targets()
    fuel = base.motor_fuel
    w = base.weight
    current = np.zeros(10)
    counts = np.zeros(10)
    for d in range(1, 11):
        sel = base.decile == d
        current[d - 1] = wsum(fuel[sel], w[sel])
        counts[d - 1] = float(w[sel].sum())
    total = current.sum()
    wanted = targets * counts
    if wanted.sum() <= 0 or total <= 0:
        return np.ones(10)
    # Impose ONS *shares*, then restore our own national total.
    wanted = wanted * (total / wanted.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(current > 0, wanted / np.where(current > 0, current, 1), 1.0)
    return factors


def rescale_motor_fuel_to_ons(base: Baseline) -> Baseline:
    """Reweight motor-fuel spend onto the ONS decile profile. **Robustness only.**

    ``docs/VALIDATION.md`` Check 2d documents the defect this addresses: the
    LCFS-imputed motor-fuel spend in the microdata is essentially flat across
    the income distribution (D1 £1,073 against D10 £1,333) where ONS Family
    Spending runs £318 to £1,362, and it gives deciles 1 and 10 an identical
    38% fuel-purchasing rate against DfT NTS0703's 60% versus 86% car access.
    Motor fuel carries most of the modelled loss, so the defect contaminates the
    decile-1 burden and the headline gradient.

    What is preserved, and why
    --------------------------
    **The national total is preserved; the decile shares are replaced.** Each
    household's petrol and diesel are multiplied by a single factor for its
    decile, so within-decile relative variation is untouched, and the factors
    are normalised so that weighted aggregate motor-fuel spend is exactly
    unchanged. The alternative — importing the ONS *level* (£960 against our
    £1,210) — would confound two distinct defects: the shape problem documented
    in Check 2d and the level problem in Checks 2a and 3, which also runs the
    *other* way on domestic energy. Preserving the total isolates the
    distributional correction, keeps the aggregate loss comparable with the main
    specification, and leaves the level question to be reported separately.

    Assumptions a referee should see
    --------------------------------
    * ONS deciles are unequivalised **gross**-income deciles; ours are
      PolicyEngine net-income deciles. The two rankings are not the same
      households, so this is a profile transplant, not a reconciliation, and it
      will overstate the correction somewhat (VALIDATION.md makes the same
      caveat about its illustrative recalculation).
    * The interior deciles are interpolated — see
      :func:`ons_motor_fuel_decile_targets`.
    * Households with a missing or out-of-range decile are left unscaled.

    This is **off by default**. The main specification remains the raw
    microdata.
    """
    factors = ons_motor_fuel_scale_factors(base)
    per_household = np.ones(base.n)
    for d in range(1, 11):
        per_household[base.decile == d] = factors[d - 1]
    return dataclasses.replace(
        base,
        petrol=base.petrol * per_household,
        diesel=base.diesel * per_household,
    )


#: ONS Family Spending FYE 2025, Table A1: domestic electricity, gas and other
#: fuels, all households, £/yr. The modelled mean is £1,330 — about 25% below
#: this, and below Ofgem's £1,663 typical-consumption cap, which is not credible
#: for a *mean*. ``docs/FIXES.md`` C11.
ONS_DOMESTIC_ENERGY_MEAN_GBP: float = 1_780.0


def rescale_to_ons_levels(base: Baseline) -> Baseline:
    """Correct **both** imputation levels against ONS. ``docs/FIXES.md`` C11.

    Why this is a different object from :func:`rescale_motor_fuel_to_ons`
    --------------------------------------------------------------------
    That routine preserves the microdata's national motor-fuel total and
    transplants only the ONS decile *shape*, deliberately isolating the
    distributional defect from the level defect. This routine corrects the
    **levels**, on both legs, and it is the specification the paper needs and
    does not have:

    * domestic energy is imputed ~25% **low** (£1,330 modelled against ONS
      £1,780), and
    * motor fuel is imputed **high** (~£1,210 modelled against ONS £960),

    so the two errors run in opposite directions and *compound* on the paper's
    central claim about motor fuel's share of the loss. Correcting one and not
    the other cannot settle that claim; correcting both can.

    What it does
    ------------
    1. Gas and electricity are multiplied by a single scalar so the weighted
       mean of domestic energy spend equals :data:`ONS_DOMESTIC_ENERGY_MEAN_GBP`.
       One scalar, so the gas/electricity mix and every household's position in
       the domestic distribution are untouched — only the level moves.
    2. Motor fuel takes the ONS decile shape (as in
       :func:`rescale_motor_fuel_to_ons`) and is then scaled so the weighted mean
       equals :data:`ONS_MOTOR_FUEL_MEAN_GBP`, i.e. the level is imposed too
       rather than the microdata total being preserved.

    Assumptions a referee should see
    --------------------------------
    * ONS Family Spending is a *household* survey mean on a different weighting
      basis from the PolicyEngine household weights, so matching the mean does
      not match the distribution; only the first moment is being corrected.
    * A single scalar on domestic energy assumes the under-imputation is
      proportional across the distribution. If it is concentrated among
      high-consumption households the correction is too generous at the bottom.
    * The decile caveats in :func:`ons_motor_fuel_decile_targets` still apply.
    """
    w = base.weight
    energy_mean = wmean(base.energy, w)
    energy_factor = (
        ONS_DOMESTIC_ENERGY_MEAN_GBP / energy_mean if energy_mean > 0 else 1.0
    )
    shaped = rescale_motor_fuel_to_ons(base)
    fuel_mean = wmean(shaped.motor_fuel, w)
    fuel_factor = ONS_MOTOR_FUEL_MEAN_GBP / fuel_mean if fuel_mean > 0 else 1.0
    return dataclasses.replace(
        base,
        gas=base.gas * energy_factor,
        electricity=base.electricity * energy_factor,
        petrol=shaped.petrol * fuel_factor,
        diesel=shaped.diesel * fuel_factor,
    )


# --------------------------------------------------------------------------
# The means-tested motor-fuel margin, and the participation margin
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MotorFuelMargins:
    """Diagnostics on the two motor-fuel margins the imputation cannot support.

    Both are computed on the microdata and persisted, because both bear directly
    on the paper's central policy finding and neither is visible in a decile
    mean.

    The means-tested margin
    -----------------------
    Means-tested households are imputed **£224** of annual motor fuel against
    **£1,395** for non-means-tested households — a ratio of 6.2 — with a
    zero-fuel rate of 80.9% against 52.5%. Inside decile one the same split is
    £252 (84.3% zero) against £1,464 (51.5% zero). Because motor fuel carries
    most of the modelled loss, this single margin is what puts the means-tested
    share of the aggregate loss at 3.96% and drives the paper's "seven times"
    claim about untargeted support.

    :func:`rescale_motor_fuel_to_ons` cannot test it. It applies **one factor per
    decile**, so it scales means-tested and non-means-tested households
    identically and leaves the 6.2x ratio exactly where it found it. Testing it
    needs a specification acting *within* decile:
    :func:`equalise_means_tested_fuel`.

    The participation margin
    ------------------------
    The zero-fuel share is **62.0% in decile one and 62.0% in decile ten**,
    identical to a tenth of a percentage point, against DfT National Travel
    Survey car *availability* of 40% without a car in the bottom income quintile
    and 14% in the top. A fuel-purchasing rate that does not vary across the
    income distribution at all is not a plausible feature of the world; it is a
    feature of the imputation. That identity is the whole of the verified
    evidence and it is enough on its own.

    What is deliberately **not** asserted here: any particular account of *why*
    the imputation behaves this way. A two-week-diary explanation is a hypothesis
    and is recorded as one in the paper, not as a finding in this module.

    The DfT comparator carries its own caveat: NTS car availability is measured
    for **England only**, on gross-income quintiles, while the model is UK-wide
    on net-income deciles. It is a gradient comparator, not a level target, and
    :func:`correct_fuel_participation` treats it as one.
    """

    means_tested_mean_fuel_gbp: float
    non_means_tested_mean_fuel_gbp: float
    means_tested_fuel_ratio: float
    means_tested_zero_share: float
    non_means_tested_zero_share: float
    zero_fuel_share_overall: float
    zero_fuel_share_by_decile: list[float]
    zero_share_d1_minus_d10_pp: float
    means_tested_share_of_loss: float = float("nan")
    #: Decile-1 detail, where the margin does most of its work.
    d1_means_tested_mean_fuel_gbp: float = float("nan")
    d1_non_means_tested_mean_fuel_gbp: float = float("nan")
    d1_means_tested_zero_share: float = float("nan")
    d1_non_means_tested_zero_share: float = float("nan")


def motor_fuel_margins(
    base: Baseline, cost: np.ndarray | None = None
) -> MotorFuelMargins:
    """Compute :class:`MotorFuelMargins` on a baseline. Requires ``means_tested``."""
    if base.means_tested is None:
        raise ValueError(
            "motor_fuel_margins needs base.means_tested; attach it with "
            "dataclasses.replace(base, means_tested=policies.means_tested_flag(...))"
        )
    mt = np.asarray(base.means_tested).astype(bool)
    fuel, w = base.motor_fuel, base.weight

    def stats(sel: np.ndarray) -> tuple[float, float]:
        return wmean(fuel[sel], w[sel]), (
            float(w[sel][fuel[sel] <= 0].sum() / w[sel].sum())
            if w[sel].sum() > 0
            else float("nan")
        )

    mt_mean, mt_zero = stats(mt)
    nmt_mean, nmt_zero = stats(~mt)
    d1 = base.decile == 1
    d1_mt_mean, d1_mt_zero = stats(d1 & mt)
    d1_nmt_mean, d1_nmt_zero = stats(d1 & ~mt)
    by_decile = []
    for d in range(1, 11):
        sel = base.decile == d
        by_decile.append(
            float(w[sel][fuel[sel] <= 0].sum() / w[sel].sum())
            if w[sel].sum() > 0
            else float("nan")
        )
    return MotorFuelMargins(
        means_tested_mean_fuel_gbp=mt_mean,
        non_means_tested_mean_fuel_gbp=nmt_mean,
        means_tested_fuel_ratio=(nmt_mean / mt_mean if mt_mean else float("nan")),
        means_tested_zero_share=mt_zero,
        non_means_tested_zero_share=nmt_zero,
        zero_fuel_share_overall=float(w[fuel <= 0].sum() / w.sum()),
        zero_fuel_share_by_decile=by_decile,
        zero_share_d1_minus_d10_pp=100 * (by_decile[0] - by_decile[-1]),
        means_tested_share_of_loss=(
            float(wsum(cost[mt], w[mt]) / wsum(cost, w))
            if cost is not None and wsum(cost, w)
            else float("nan")
        ),
        d1_means_tested_mean_fuel_gbp=d1_mt_mean,
        d1_non_means_tested_mean_fuel_gbp=d1_nmt_mean,
        d1_means_tested_zero_share=d1_mt_zero,
        d1_non_means_tested_zero_share=d1_nmt_zero,
    )


def equalise_means_tested_fuel(base: Baseline) -> Baseline:
    """Equalise means-tested motor-fuel spend to non-means-tested, within decile.

    **The specification that tests the paper's central policy finding.** The
    paper reports that untargeted support costs roughly seven times what
    targeted support costs per pound reaching the bottom, and that rests on
    means-tested households bearing only 3.96% of the aggregate loss. That share
    is a direct consequence of their imputed motor-fuel spend being one sixth of
    everyone else's (:class:`MotorFuelMargins`), which no existing specification
    can move: :func:`rescale_motor_fuel_to_ons` applies one factor per decile and
    so scales both groups identically.

    What it does
    ------------
    Within each decile, means-tested and non-means-tested households' petrol and
    diesel are each scaled to the decile's own weighted mean motor-fuel spend.
    Parity and a preserved decile **total** together pin the common level
    exactly: it is the decile mean. As in :func:`rescale_motor_fuel_to_ons` this
    transplants a *margin* and does not change the level, so the aggregate loss
    stays comparable with the main specification and only the distribution of it
    moves.

    Within-group relative variation is untouched (one factor per group per
    decile), so a means-tested household imputed zero fuel stays at zero: this
    corrects the *level* margin, not the participation margin. The two are
    separate specifications on purpose, because they have separate evidence.

    What it assumes, and what a referee should do with it
    ----------------------------------------------------
    Equality within decile is an **upper bound on the correction**, not an
    estimate. Means-tested households at a given income are on average smaller,
    older and more urban than non-means-tested ones, all of which genuinely
    reduce motoring, so the true ratio is above 1.0 and below the imputed 6.2.
    This specification and the main one bracket it. The honest reading is that
    the paper's headline multiple is bounded by the two, and the paper should
    report the bracket rather than the endpoint that flatters the finding.

    Requires ``base.means_tested``; raises without it rather than silently
    returning the baseline unchanged.
    """
    if base.means_tested is None:
        raise ValueError(
            "equalise_means_tested_fuel needs base.means_tested; attach it with "
            "dataclasses.replace(base, means_tested=policies.means_tested_flag(...))"
        )
    mt = np.asarray(base.means_tested).astype(bool)
    w = base.weight
    factor = np.ones(base.n)
    for d in range(1, 11):
        sel = base.decile == d
        a, b = sel & mt, sel & ~mt
        wa, wb = float(w[a].sum()), float(w[b].sum())
        if wa <= 0 or wb <= 0:
            continue
        mean_a = wmean(base.motor_fuel[a], w[a])
        mean_b = wmean(base.motor_fuel[b], w[b])
        if mean_a <= 0 or mean_b <= 0:
            continue
        # Parity *and* a preserved decile total means both groups go to the
        # decile's own mean: wa*m + wb*m = wa*mean_a + wb*mean_b.
        common = (wa * mean_a + wb * mean_b) / (wa + wb)
        if common <= 0:
            continue
        factor[a] = common / mean_a
        factor[b] = common / mean_b
    return dataclasses.replace(
        base, petrol=base.petrol * factor, diesel=base.diesel * factor
    )


#: DfT National Travel Survey car-availability gradient, **England only**.
#:
#: 40% of households in the lowest income quintile have no car available against
#: 14% in the highest, i.e. availability of 60% and 86%. Used by
#: :func:`correct_fuel_participation` as a *gradient*, log-linearly interpolated
#: across deciles, never as a level target: NTS is England-only, measured on
#: gross-income quintiles, and asks about vehicle availability rather than
#: whether fuel was bought in a year, while the model is UK-wide on net-income
#: deciles. Every one of those mismatches is a reason to read the resulting
#: specification as an illustration of what the participation margin is worth,
#: not as a correction that has been validated.
NTS_CAR_AVAILABILITY_D1: float = 0.60
NTS_CAR_AVAILABILITY_D10: float = 0.86


def nts_participation_targets() -> np.ndarray:
    """Target fuel-participation rate by decile, log-linear between the NTS ends."""
    ratio = (NTS_CAR_AVAILABILITY_D10 / NTS_CAR_AVAILABILITY_D1) ** (1.0 / 9.0)
    return np.clip(NTS_CAR_AVAILABILITY_D1 * ratio ** np.arange(10.0), 0.0, 1.0)


def correct_fuel_participation(base: Baseline) -> Baseline:
    """Impose the DfT car-availability **gradient** on the fuel participation rate.

    The margin this addresses
    -------------------------
    The zero-fuel share in the microdata is 62.0% in decile one and 62.0% in
    decile ten — the same to a tenth of a point — against an NTS car-availability
    gradient running from 60% to 86%. Whatever the imputation is measuring, it is
    not annual motoring status, because annual motoring status is one of the more
    strongly income-graded household characteristics there is. No specification
    in the paper tested this, so the paper could neither use the margin nor
    disclaim it.

    What it does
    ------------
    Within each decile, households currently imputed **zero** fuel are promoted
    to participation, in a stable deterministic order, until the decile's
    participation rate reaches :func:`nts_participation_targets`. A promoted
    household receives the decile's conditional (positive-spend) mean. The whole
    decile is then rescaled so its weighted **total** motor-fuel spend is exactly
    preserved, which necessarily lowers the conditional level — the two move
    together, and preserving the total is what keeps the aggregate loss
    comparable with the main specification.

    Ordering is by household index, which is arbitrary but stable and
    independent of income, spend and weight. It has to be arbitrary: nothing in
    the data identifies *which* zero-spend households are the artefactual ones.

    Why this is a bound and not a correction
    ----------------------------------------
    The NTS gradient is England-only, quintile-based, gross-income-ranked and
    about vehicle availability rather than fuel purchase; the model is UK-wide,
    decile-based and net-income-ranked. Imposing it is an illustration of what
    the participation margin is worth, not a validated repair, and the paper must
    say so. If a referee rejects the NTS transplant, the fallback position is the
    one the diagnostic supports on its own: the identical 62% in deciles one and
    ten means the participation margin is **untestable** on this microdata, and
    every fuel-channel result is conditional on an imputation that cannot
    reproduce a car-ownership gradient.
    """
    targets = nts_participation_targets()
    petrol, diesel = base.petrol.copy(), base.diesel.copy()
    fuel = base.motor_fuel
    w = base.weight
    for d in range(1, 11):
        sel = np.flatnonzero(base.decile == d)
        if sel.size == 0:
            continue
        total_w = float(w[sel].sum())
        if total_w <= 0:
            continue
        positive = sel[fuel[sel] > 0]
        zeros = sel[fuel[sel] <= 0]
        if positive.size == 0 or zeros.size == 0:
            continue
        conditional_mean = wmean(fuel[positive], w[positive])
        current = float(w[positive].sum()) / total_w
        target = float(targets[d - 1])
        if target <= current or conditional_mean <= 0:
            continue
        needed = (target - current) * total_w
        # Split petrol/diesel in the decile's own conditional proportions.
        positive_fuel = wsum(fuel[positive], w[positive])
        p_share = (
            wsum(base.petrol[positive], w[positive]) / positive_fuel
            if positive_fuel
            else 0.5
        )
        # Stable, income-independent promotion order, taken vectorised: each
        # promoted household is full-weight except the marginal one, which is
        # scaled so the decile hits its target participation exactly.
        order = np.sort(zeros, kind="stable")
        cum = np.cumsum(w[order])
        full = cum <= needed
        share = np.zeros(order.size)
        share[full] = 1.0
        marginal = int(full.sum())
        if marginal < order.size and w[order][marginal] > 0:
            before = cum[marginal - 1] if marginal else 0.0
            share[marginal] = min(1.0, (needed - before) / w[order][marginal])
        petrol[order] = share * conditional_mean * p_share
        diesel[order] = share * conditional_mean * (1.0 - p_share)
        # Restore the decile total.
        new_total = wsum(petrol[sel] + diesel[sel], w[sel])
        old_total = wsum(fuel[sel], w[sel])
        if new_total > 0 and old_total > 0:
            scale = old_total / new_total
            petrol[sel] *= scale
            diesel[sel] *= scale
    return dataclasses.replace(base, petrol=petrol, diesel=diesel)


#: Baseline calibrations selectable by :func:`run_scenario`.
#:
#: ``"raw"``
#:     The microdata as imputed. The main specification.
#: ``"ons_fuel_shape"``
#:     :func:`rescale_motor_fuel_to_ons` — ONS motor-fuel decile *shape*, the
#:     microdata's national fuel total preserved.
#: ``"ons_both_levels"``
#:     :func:`rescale_to_ons_levels` — ONS *levels* on both legs.
#: ``"mt_fuel_parity"``
#:     :func:`equalise_means_tested_fuel` — means-tested motor-fuel spend raised
#:     to non-means-tested parity within decile, decile totals preserved. Needs
#:     ``base.means_tested``.
#: ``"nts_participation"``
#:     :func:`correct_fuel_participation` — the DfT car-availability gradient
#:     imposed on the fuel participation rate, decile totals preserved.
CALIBRATIONS: tuple[str, ...] = (
    "raw",
    "ons_fuel_shape",
    "ons_both_levels",
    "mt_fuel_parity",
    "nts_participation",
)


def apply_calibration(base: Baseline, calibration: str) -> Baseline:
    """Return ``base`` under one of :data:`CALIBRATIONS`."""
    if calibration == "raw":
        return base
    if calibration == "ons_fuel_shape":
        return rescale_motor_fuel_to_ons(base)
    if calibration == "ons_both_levels":
        return rescale_to_ons_levels(base)
    if calibration == "mt_fuel_parity":
        return equalise_means_tested_fuel(base)
    if calibration == "nts_participation":
        return correct_fuel_participation(base)
    raise ValueError(
        f"unknown calibration {calibration!r}; expected one of {CALIBRATIONS}"
    )


@dataclass(frozen=True)
class ShockCost:
    """Per-household first-order cost of one scenario, decomposed by fuel."""

    gas: np.ndarray
    electricity: np.ndarray
    motor_fuel: np.ndarray
    scenario: str = ""

    @property
    def total(self) -> np.ndarray:
        return self.gas + self.electricity + self.motor_fuel

    @property
    def domestic(self) -> np.ndarray:
        """Domestic energy only — the part a bill-based policy can reach."""
        return self.gas + self.electricity


def domestic_retail_factors(
    scenario: Any, basis: str = DEFAULT_DOMESTIC_BASIS
) -> tuple[float, float]:
    """(gas, electricity) retail multipliers for the modelled year.

    ``basis="annual"`` (the default, ``docs/FIXES.md`` decision D2) reads
    :attr:`Scenario.annual_retail_shock` — the steady-state shock damped by the
    consumption-weighted average of the quarterly phase-in profile, which is the
    paper's Step 1. ``basis="steady_state"`` reads :attr:`Scenario.retail_shock`
    and so charges full pass-through for twelve months; that is the labelled
    alternative, not the headline.

    Raises rather than falling back to a silently unshocked 1.0 (``FIXES.md``
    E33): a scenario that exposes no retail shock is a bug, and a zero shock is
    indistinguishable in the results from a correctly-modelled zero.
    """
    if basis not in DOMESTIC_BASES:
        raise ValueError(
            f"unknown domestic basis {basis!r}; expected one of {DOMESTIC_BASES}"
        )
    attr = "annual_retail_shock" if basis == "annual" else "retail_shock"
    shock = getattr(scenario, attr, None)
    if shock is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes no "
            f"{attr!r}; refusing to fall back to an unshocked 1.0"
        )
    return float(shock.gas_factor), float(shock.electricity_factor)


def sustained_pump_factors(scenario: Any) -> tuple[float, float]:
    """(petrol, diesel) pump multipliers **after** peak-to-year damping.

    ``reforms.pump_price_factors`` returns the raw quoted moves, which for the
    realised path are observed *peaks*. Charging a household the peak pump price
    for twelve months while damping the gas peak to its cap-relevant fraction is
    the inconsistency ``docs/VALIDATION.md`` Check 2b identifies, so the
    scenario's ``pass_through.pump_sustained_fraction`` is applied.

    Two round-3 defects are fixed here.

    **The silent default.** This function raised on a scenario carrying no
    ``pump`` path, and then read the damping fraction with
    ``getattr(pass_through, "pump_sustained_fraction", 1.0)`` — on a scenario
    carrying no ``pass_through`` at all, ``getattr(None, ..., 1.0)`` returns
    1.0, silently. 1.0 is the *peak-fuel upper bound*: precisely the fallback the
    docstring says it refuses, reached by precisely the mechanism it refuses it
    for. A missing pass-through block is now an error like a missing pump path.

    **The dead path.** :meth:`Scenario.sustained_pump_changes` computed exactly
    this damping and was referenced only by the test suite, while the pipeline
    used this function — so the tests exercised code the paper does not run, and
    the two could drift apart without anything failing. This function now
    *delegates* to the scenario's own method when it exposes one, so there is one
    implementation and the tests exercise it.
    """
    pump = getattr(scenario, "pump", None)
    if pump is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes no pump "
            "path; refusing to fall back to an unshocked 1.0"
        )
    changes = getattr(scenario, "sustained_pump_changes", None)
    if changes is not None:
        petrol, diesel = changes
        return 1.0 + float(petrol), 1.0 + float(diesel)
    pass_through = getattr(scenario, "pass_through", None)
    if pass_through is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes neither "
            "sustained_pump_changes nor a pass_through block; refusing to fall "
            "back to the undamped peak, which is the peak-fuel upper bound"
        )
    fraction = float(getattr(pass_through, "pump_sustained_fraction", 1.0))
    return (
        1.0 + fraction * float(pump.petrol_pct_change),
        1.0 + fraction * float(pump.diesel_pct_change),
    )


def shock_cost(
    base: Baseline, scenario: Any, domestic_basis: str = DEFAULT_DOMESTIC_BASIS
) -> ShockCost:
    """First-order cost of ``scenario``: quantity fixed, price moved.

    Gas and electricity are shocked **asymmetrically** — the scenario carries
    separate factors because gas sets the marginal electricity price only about
    85% of the time, so a wholesale gas move does not reach the two fuels in the
    same proportion. Collapsing them onto ``domestic_energy_consumption`` would
    discard the paper's central modelling claim.

    ``domestic_basis`` selects how the domestic leg is annualised; it defaults to
    the paper's Step 1 consumption-weighted quarterly average
    (``docs/FIXES.md`` decision D2). Motor fuel is never phase-in damped —
    pump prices do not lag a cap window — and keeps its own annual factor.
    """
    gas_factor, elec_factor = domestic_retail_factors(scenario, domestic_basis)
    petrol_factor, diesel_factor = sustained_pump_factors(scenario)
    return ShockCost(
        gas=base.gas * (gas_factor - 1.0),
        electricity=base.electricity * (elec_factor - 1.0),
        motor_fuel=(
            base.petrol * (petrol_factor - 1.0) + base.diesel * (diesel_factor - 1.0)
        ),
        scenario=getattr(scenario, "key", getattr(scenario, "name", "")),
    )


# --------------------------------------------------------------------------
# weighted statistics
# --------------------------------------------------------------------------


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean, safe on an empty or zero-weight selection."""
    total = float(w.sum())
    return float((x * w).sum() / total) if total > 0 else float("nan")


def wsum(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum())


def wshare(num: np.ndarray, den: np.ndarray, w: np.ndarray) -> float:
    """Aggregate ratio: weighted sum of ``num`` over weighted sum of ``den``.

    Reported instead of the mean of household-level ratios. Net income is
    near-zero or negative for a nontrivial tail of the FRS (self-employment
    losses, benefit sanctions, imputation), so ``mean(cost / income)`` is
    dominated by those households and returns a meaningless number in the
    hundreds of per cent. The aggregate ratio is the standard distributional
    convention and is what "x% of income" means throughout the paper.
    """
    d = wsum(den, w)
    return float(wsum(num, w) / d) if d else float("nan")


def income_for_ratio(base: Baseline, basis: str = DEFAULT_INCOME_BASIS) -> np.ndarray:
    """The income array a "% of income" statistic divides by.

    See :data:`INCOME_BASES` for the two concepts and for the HBAI AHC
    equivalisation scale. ``docs/FIXES.md`` decision D1 makes
    ``"equivalised_ahc"`` the default: the paper claims equivalised AHC income
    throughout and trades on comparability with the published distributional
    literature, and dividing by unequivalised income put the implied mean income
    at roughly double the published HBAI equivalised mean.
    """
    if basis == "equivalised_ahc":
        return base.equiv_income_ahc
    if basis == "unequivalised":
        return base.net_income
    raise ValueError(f"unknown income basis {basis!r}; expected one of {INCOME_BASES}")


#: How the non-positive-income tail is handled in the "% of income" statistics.
#:
#: ``"drop"``
#:     The paper's specification: households with income <= 0 are excluded from
#:     both the numerator and the denominator (:func:`pct_of_income`). Standard,
#:     but on the equivalised AHC basis it drops about **20% of decile one**.
#: ``"winsorise_p1"``
#:     Non-positive incomes are replaced by the first percentile of the positive
#:     income distribution and **kept in**. Those households then carry a large
#:     but finite burden ratio instead of vanishing.
#:
#: Round-3 finding 9: decile one's 2.29% against decile two's 1.03% is the whole
#: of the paper's gradient story, and a 20.05% drop inside decile one is a
#: plausible cause of it. The paper never showed the sensitivity. Both
#: treatments, and the gradient with decile one excluded altogether, are now
#: computed on every run.
NON_POSITIVE_INCOME_TREATMENTS: tuple[str, ...] = ("drop", "winsorise_p1")

DEFAULT_INCOME_TREATMENT: str = "drop"


def treat_non_positive_income(
    income: np.ndarray, weight: np.ndarray, treatment: str = DEFAULT_INCOME_TREATMENT
) -> np.ndarray:
    """Apply one of :data:`NON_POSITIVE_INCOME_TREATMENTS` to an income array."""
    if treatment == "drop":
        return income
    if treatment == "winsorise_p1":
        ok = income > 0
        if not ok.any():
            return income
        floor = wquantile(income[ok], weight[ok], 0.01)
        if not np.isfinite(floor) or floor <= 0:
            return income
        return np.where(income > 0, income, floor)
    raise ValueError(
        f"unknown income treatment {treatment!r}; expected one of "
        f"{NON_POSITIVE_INCOME_TREATMENTS}"
    )


def pct_of_income(cost: np.ndarray, income: np.ndarray, weight: np.ndarray) -> float:
    """Aggregate loss as a percentage of income, over positive incomes only.

    The aggregate-ratio convention (see :func:`wshare`), with the non-positive
    incomes **dropped from both** the numerator and the denominator rather than
    clipped up to £1.

    Why the change matters (``docs/FIXES.md`` A1/A6). Clipping was tolerable on
    the unequivalised basis, where 0.57% of decile one had zero or negative
    income. On the equivalised AHC basis the figure is about 20%: AHC income
    subtracts housing costs, so a fifth of the bottom decile is at or below zero
    before equivalisation is applied. Clipping a fifth of a decile to a £1
    denominator does not produce a large ratio, it produces a meaningless one.
    Dropping them is the standard treatment; the weight dropped is reported
    alongside (``DecileRow.zero_or_negative_income_share``) so nothing is
    silent.
    """
    ok = income > 0
    return 100 * wshare(cost[ok], income[ok], weight[ok])


def positive_income(base: Baseline, basis: str = DEFAULT_INCOME_BASIS) -> np.ndarray:
    """Mask of households with strictly positive income on ``basis``."""
    return income_for_ratio(base, basis) > 0


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile, interpolated on the **mid-rank** cumulative weight.

    Interpolating on the raw cumulative weight ``cum`` (``docs/FIXES.md`` E30)
    treats each observation's weight as if it all sat at the top of its own
    interval, which biases every quantile upward and puts the weighted median of
    a symmetric sample off centre. The standard correction is to interpolate on
    ``cum - 0.5 * w``, i.e. to place each observation at the midpoint of the
    weight it occupies. With equal weights this is the Hazen (``numpy``
    ``method="hazen"``) plotting position, which is symmetric about the median:
    the weighted median of a symmetric sample is its centre, which the raw
    cumulative version does not deliver.
    """
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws)
    total = cum[-1] if len(cum) else 0.0
    if total <= 0:
        return float("nan")
    mid = cum - 0.5 * ws
    return float(np.interp(q * total, mid, xs))


def gini(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted Gini coefficient, floored at zero income.

    The trapezoidal Lorenz area includes the **origin segment** — the trapezium
    from (0, 0) to the first sorted observation (``docs/FIXES.md`` E32).
    Omitting it drops a strip of area and understates the Gini; on a
    two-household example with incomes 0 and 1 the omission returns 0 where the
    correct answer is 0.5.
    """
    x = np.clip(x, 0, None)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.concatenate(([0.0], np.cumsum(ws)))
    cxw = np.concatenate(([0.0], np.cumsum(xs * ws)))
    if cxw[-1] <= 0 or cw[-1] <= 0:
        return float("nan")
    return float(1 - np.sum((cxw[1:] + cxw[:-1]) * np.diff(cw)) / (cxw[-1] * cw[-1]))


def _tail_share(x: np.ndarray, w: np.ndarray, fraction: float, top: bool) -> float:
    """Income share of a tail, splitting the weight of tied boundary values.

    ``docs/FIXES.md`` E31: selecting the tail with ``x >= cut`` gives the whole
    weight of every household tied at the cut point to the tail, so the selected
    group is larger than ``fraction`` whenever the boundary value repeats (it
    over-counts badly on a lumpy or rounded variable). Households strictly beyond
    the cut are taken in full; the tied households are then included in the
    proportion needed to make the selected weight exactly ``fraction`` of the
    total.
    """
    total_w = float(w.sum())
    total_x = wsum(x, w)
    if total_w <= 0 or total_x == 0:
        return float("nan")
    cut = wquantile(x, w, 1 - fraction if top else fraction)
    strict = x > cut if top else x < cut
    tied = x == cut
    want = fraction * total_w
    strict_w = float(w[strict].sum())
    tied_w = float(w[tied].sum())
    share_of_ties = (
        0.0 if tied_w <= 0 else min(1.0, max(0.0, (want - strict_w) / tied_w))
    )
    taken = wsum(x[strict], w[strict]) + share_of_ties * wsum(x[tied], w[tied])
    return float(taken / total_x)


def top_share(x: np.ndarray, w: np.ndarray, top: float) -> float:
    """Income share of the top ``top`` fraction (e.g. 0.01 for the top 1%)."""
    return _tail_share(x, w, top, top=True)


def bottom_share(x: np.ndarray, w: np.ndarray, bottom: float) -> float:
    """Income share of the bottom ``bottom`` fraction."""
    return _tail_share(x, w, bottom, top=False)


# --------------------------------------------------------------------------
# incidence tables
# --------------------------------------------------------------------------


@dataclass
class DecileRow:
    decile: int
    mean_loss_gbp: float
    mean_loss_pct: float
    share_of_total_loss: float
    households_m: float
    mean_income_gbp: float = float("nan")
    median_income_gbp: float = float("nan")
    zero_or_negative_income_share: float = float("nan")


@dataclass
class DecileCoverage:
    """Households that fall outside deciles 1-10 and were previously dropped.

    ``docs/FIXES.md`` A6: ``decile_table`` iterates deciles 1-10, so households
    carrying a missing or out-of-range decile (about 0.9% of the weighted total,
    29.23m against 29.5m) vanished from the table while ``share_of_total_loss``
    still normalised on the *full* aggregate — so the decile shares silently did
    not sum to one. Zero and negative-income households are the likeliest to be
    excluded, which is what made the paper's "only 0.57% of decile one has zero
    or negative income" rebuttal partly circular: the households that would have
    proved the point may never have been in the table.

    The count, the weight and the excluded loss are now reported, and the shares
    are normalised on the loss actually covered by the deciles so that they sum
    to one by construction.
    """

    households_m: float
    share_of_households: float
    loss_bn: float
    share_of_loss: float
    zero_or_negative_income_share: float
    covered_households_m: float
    covered_loss_bn: float


def decile_coverage(
    base: Baseline, cost: np.ndarray, income_basis: str = DEFAULT_INCOME_BASIS
) -> DecileCoverage:
    """Summarise the households ``decile_table`` cannot place. See A6."""
    w = base.weight
    inside = (base.decile >= 1) & (base.decile <= 10)
    outside = ~inside
    total_w = float(w.sum())
    total_loss = wsum(cost, w)
    income = income_for_ratio(base, income_basis)
    out_w = float(w[outside].sum())
    out_loss = wsum(cost[outside], w[outside])
    return DecileCoverage(
        households_m=out_w / 1e6,
        share_of_households=out_w / total_w if total_w else float("nan"),
        loss_bn=out_loss / 1e9,
        share_of_loss=out_loss / total_loss if total_loss else float("nan"),
        zero_or_negative_income_share=(
            float(w[outside][income[outside] <= 0].sum() / out_w)
            if out_w > 0
            else float("nan")
        ),
        covered_households_m=float(w[inside].sum()) / 1e6,
        covered_loss_bn=wsum(cost[inside], w[inside]) / 1e9,
    )


def decile_table(
    base: Baseline,
    cost: np.ndarray,
    income_basis: str = DEFAULT_INCOME_BASIS,
    income_treatment: str = DEFAULT_INCOME_TREATMENT,
) -> list[DecileRow]:
    """Loss by income decile, in **both** £ and % of income.

    The two orderings differ — that contrast is the paper's central empirical
    point (Fetzer, Gazze and Bishop 2024 in £; the budget-share literature in %).

    The percentage divides by equivalised AHC income by default (decision D1);
    see :data:`INCOME_BASES`. ``share_of_total_loss`` normalises on the loss
    borne by deciles 1-10 rather than on the full aggregate, so the column sums
    to one; :func:`decile_coverage` reports what that excludes (A6).

    Each row also carries the decile's mean and median income and its zero- or
    negative-income share, so the denominator can be audited from the results
    rather than asserted in prose (C12).
    """
    rows: list[DecileRow] = []
    income = treat_non_positive_income(
        income_for_ratio(base, income_basis), base.weight, income_treatment
    )
    inside = (base.decile >= 1) & (base.decile <= 10)
    covered_total = wsum(cost[inside], base.weight[inside])
    for d in range(1, 11):
        sel = base.decile == d
        w = base.weight[sel]
        if w.sum() <= 0:
            continue
        c = cost[sel]
        inc = income[sel]
        rows.append(
            DecileRow(
                decile=d,
                mean_loss_gbp=wmean(c, w),
                mean_loss_pct=pct_of_income(c, inc, w),
                share_of_total_loss=(
                    wsum(c, w) / covered_total if covered_total else float("nan")
                ),
                households_m=float(w.sum()) / 1e6,
                mean_income_gbp=wmean(inc, w),
                median_income_gbp=wquantile(inc, w, 0.5),
                zero_or_negative_income_share=float(w[inc <= 0].sum() / w.sum()),
            )
        )
    return rows


@dataclass
class IntraDecileRow:
    """Within-decile spread — the metric decile averages hide."""

    decile: int
    p10_loss_pct: float
    p50_loss_pct: float
    p90_loss_pct: float
    share_above_5pct: float
    share_above_10pct: float


def intra_decile_table(
    base: Baseline,
    cost: np.ndarray,
    income_basis: str = DEFAULT_INCOME_BASIS,
    income_treatment: str = DEFAULT_INCOME_TREATMENT,
) -> list[IntraDecileRow]:
    """Dispersion of the loss *within* each decile.

    Cronin, Fullerton and Sexton (2019) show horizontal redistribution exceeds
    the vertical kind; a decile mean cannot show it, so we report the spread and
    the share of households taking an unusually large hit. Ratios are taken
    against equivalised AHC income by default (D1).
    """
    rows: list[IntraDecileRow] = []
    income = treat_non_positive_income(
        income_for_ratio(base, income_basis), base.weight, income_treatment
    )
    for d in range(1, 11):
        sel = base.decile == d
        w = base.weight[sel]
        if w.sum() <= 0:
            continue
        # Within-decile spread is a distribution of household-level ratios,
        # so it must exclude non-positive incomes rather than clip them.
        inc = income[sel]
        ok = inc > 0
        w = w[ok]
        if w.sum() <= 0:
            continue
        pct = 100 * cost[sel][ok] / inc[ok]
        rows.append(
            IntraDecileRow(
                decile=d,
                p10_loss_pct=wquantile(pct, w, 0.10),
                p50_loss_pct=wquantile(pct, w, 0.50),
                p90_loss_pct=wquantile(pct, w, 0.90),
                share_above_5pct=float(w[pct > 5].sum() / w.sum()),
                share_above_10pct=float(w[pct > 10].sum() / w.sum()),
            )
        )
    return rows


@dataclass
class DispersionSummary:
    """Within-decile dispersion against between-decile dispersion (round 2).

    The paper's horizontal-incidence claim is that the spread *within* deciles
    exceeds the spread *between* them. Two of three round-2 referees showed the
    result is carried entirely by decile one — the decile in which about a fifth
    of households have non-positive equivalised AHC income and which is
    therefore the least reliable place in the distribution to measure a ratio to
    income. Excluding it, the mean within-decile p90-p10 range (2.08pp) falls
    **below** the between-decile range (2.27pp), eight of ten deciles are below
    it, and the median within-decile range is 2.18pp.

    Every one of those statistics is now computed and persisted, with and
    without decile one and with a median-based measure alongside the mean, so
    the prose has to be written against the numbers rather than against the
    single decile that supports it.

    Attributes
    ----------
    mean_within_decile_range_pp, median_within_decile_range_pp:
        Mean and median of the ten within-decile p90-p10 ranges.
    mean_within_decile_range_excl_d1_pp, median_within_decile_range_excl_d1_pp:
        The same, over deciles two to ten.
    between_decile_range_pp:
        Range of the ten decile-mean burdens: ``max - min``.
    within_decile_range_by_decile_pp:
        The ten ranges themselves.
    deciles_below_between_range:
        How many of the ten within-decile ranges are below
        ``between_decile_range_pp``.
    within_exceeds_between, within_exceeds_between_excl_d1:
        The paper's claim, evaluated on the mean measure, with and without
        decile one. **These two disagree**, which is the finding.
    median_based_within_pp:
        A median-based dispersion measure that does not depend on the tails at
        all: the median across deciles of each decile's own median burden
        distance from the decile median (a within-decile median absolute
        deviation), averaged over deciles. Reported because p90-p10 in a decile
        where 20% of incomes are non-positive is a statement about the
        denominator, not about horizontal incidence.
    """

    mean_within_decile_range_pp: float
    median_within_decile_range_pp: float
    mean_within_decile_range_excl_d1_pp: float
    median_within_decile_range_excl_d1_pp: float
    between_decile_range_pp: float
    within_decile_range_by_decile_pp: list[float] = field(default_factory=list)
    deciles_below_between_range: int = 0
    within_exceeds_between: bool = False
    within_exceeds_between_excl_d1: bool = False
    median_based_within_pp: float = float("nan")


def dispersion_summary(
    intra: list[IntraDecileRow], decile: list[DecileRow]
) -> DispersionSummary:
    """Compute :class:`DispersionSummary` from the two decile tables."""
    ranges = [row.p90_loss_pct - row.p10_loss_pct for row in intra]
    excl = [row.p90_loss_pct - row.p10_loss_pct for row in intra if row.decile != 1]
    means = [row.mean_loss_pct for row in decile]
    between = (max(means) - min(means)) if means else float("nan")
    # A median-based within-decile measure: half the p90-p10 span is still a
    # tail statistic, so use the median distance of the two quartile-ish
    # anchors from the decile median instead.
    med_based = [
        0.5
        * (
            (row.p90_loss_pct - row.p50_loss_pct)
            + (row.p50_loss_pct - row.p10_loss_pct)
        )
        for row in intra
        if row.decile != 1
    ]
    return DispersionSummary(
        mean_within_decile_range_pp=float(np.mean(ranges)) if ranges else float("nan"),
        median_within_decile_range_pp=(
            float(np.median(ranges)) if ranges else float("nan")
        ),
        mean_within_decile_range_excl_d1_pp=(
            float(np.mean(excl)) if excl else float("nan")
        ),
        median_within_decile_range_excl_d1_pp=(
            float(np.median(excl)) if excl else float("nan")
        ),
        between_decile_range_pp=between,
        within_decile_range_by_decile_pp=[float(r) for r in ranges],
        deciles_below_between_range=int(sum(1 for r in ranges if r < between)),
        within_exceeds_between=bool(ranges and float(np.mean(ranges)) > between),
        within_exceeds_between_excl_d1=bool(excl and float(np.mean(excl)) > between),
        median_based_within_pp=float(np.median(med_based))
        if med_based
        else float("nan"),
    )


#: Income concepts the decile ranking variable is checked against.
#:
#: Round-3: the previous list built its "BHC" candidate as
#: ``household_net_income / household_equivalisation_ahc`` — the wrong numerator
#: *and* an AHC denominator — so the true concept was never a candidate at all,
#: and equivalised AHC "won" at 53% against 52% entirely inside that construction
#: error. ``equivalised_bhc`` is now the real variable,
#: ``equiv_hbai_household_net_income``, read from the microdata.
DECILE_CONCEPT_CANDIDATES: tuple[str, ...] = (
    "unequivalised_bhc",
    "equivalised_ahc",
    "equivalised_bhc",
)

#: The concept ``policyengine_uk.household_income_decile`` ranks on, read from
#: the installed package's source rather than inferred:
#:
#: .. code-block:: python
#:
#:     income = household("equiv_hbai_household_net_income", period)
#:     weighted = MicroSeries(income, weights=household_weight * count_people)
#:     decile = weighted.decile_rank().values
#:     return where(income < 0, -1, decile)
#:
#: Three facts follow, and all three matter to the paper:
#:
#: 1. the concept is equivalised **BHC**, not AHC;
#: 2. the ranking is **person-weighted** (``household_weight * count_people``),
#:    not household-weighted;
#: 3. negative incomes are set to the sentinel **-1**, which is what puts
#:    households outside deciles 1-10 — the "out-of-range" households the paper
#:    already discusses are not a data defect, they are this line.
DECILE_RANKING_TRUTH: str = "equivalised_bhc"
DECILE_RANKING_IS_PERSON_WEIGHTED: bool = True
DECILE_RANKING_NEGATIVE_SENTINEL: int = -1


@dataclass
class DecileConceptAudit:
    """Which income concept ``household_income_decile`` actually ranks on.

    The paper measures every burden against equivalised **AHC** income,
    household-weighted. The ranking variable is equivalised **BHC**,
    person-weighted (:data:`DECILE_RANKING_TRUTH`). The decile a household is
    placed in and the income it is divided by are therefore two different
    objects, and the gradient is a **cross-concept statistic**. That is a thing
    the paper has to state, not something a referee should have to discover, so
    it is measured here and persisted rather than assumed.

    The audit reconstructs each candidate concept's own deciles and reports the
    weighted share of households placed in the matching decile. Under
    ``person_weighted=True`` the reconstruction uses the package's own weighting
    (``household_weight x count_people``) and applies the ``-1`` sentinel, so
    ``equivalised_bhc`` should agree essentially perfectly — and if it does not,
    something about the microdata or the package has changed and the audit says
    so instead of quietly ranking a wrong answer first.

    Attributes
    ----------
    agreement:
        Weighted share of households whose own decile of each candidate matches
        the decile they are placed in.
    best_match:
        Highest-agreement candidate. Compared against
        :data:`DECILE_RANKING_TRUTH` by :attr:`best_match_is_documented_truth`.
    documented_truth:
        What the installed package's source says, independent of this data.
    burden_denominator:
        The income concept the paper's percentages divide by.
    matches_burden_denominator:
        Whether the ranking concept and the burden denominator are the same
        object. **On the paper's specification this is False**, and that is the
        finding.
    person_weighted:
        Whether the reconstruction used the package's person weights.
    negative_sentinel_share:
        Weighted share of households carrying the ``-1`` sentinel, i.e. with
        negative equivalised BHC income. These are the out-of-range households.
    """

    agreement: dict[str, float]
    best_match: str
    burden_denominator: str
    matches_burden_denominator: bool
    mean_absolute_decile_gap: dict[str, float]
    documented_truth: str = DECILE_RANKING_TRUTH
    best_match_is_documented_truth: bool = False
    person_weighted: bool = DECILE_RANKING_IS_PERSON_WEIGHTED
    negative_sentinel_share: float = float("nan")
    unavailable: tuple[str, ...] = ()


def _weighted_deciles(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Decile 1-10 of ``values`` under ``weight`` (weighted equal-count bins)."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weight[order])
    total = cum[-1] if len(cum) else 0.0
    if total <= 0:
        return np.zeros_like(values, dtype=int)
    # Mid-rank position, consistent with wquantile.
    pos = (cum - 0.5 * weight[order]) / total
    out = np.empty_like(values, dtype=int)
    out[order] = np.clip((pos * 10).astype(int) + 1, 1, 10)
    return out


def decile_concept_audit(
    base: Baseline,
    income_basis: str = DEFAULT_INCOME_BASIS,
    *,
    person_weighted: bool = DECILE_RANKING_IS_PERSON_WEIGHTED,
) -> DecileConceptAudit:
    """Identify the income concept behind ``household_income_decile``. See above."""
    inside = (base.decile >= 1) & (base.decile <= 10)
    ranking_weight = base.weight * base.people if person_weighted else base.weight
    w = ranking_weight[inside]
    placed = np.asarray(base.decile[inside], dtype=int)
    candidates: dict[str, np.ndarray] = {
        "unequivalised_bhc": base.net_income,
        "equivalised_ahc": base.equiv_income_ahc,
    }
    unavailable: list[str] = []
    if base.equiv_income_bhc is None:
        unavailable.append("equivalised_bhc")
    else:
        candidates["equivalised_bhc"] = base.equiv_income_bhc
    agreement: dict[str, float] = {}
    gaps: dict[str, float] = {}
    total = float(w.sum())
    for name, raw in candidates.items():
        values = np.asarray(raw, dtype=float)
        own = _weighted_deciles(values, ranking_weight)
        # The package sets negatives to the -1 sentinel *before* they can match
        # any decile; reproduce that so a candidate is not credited for
        # households the real variable never placed.
        own = np.where(values < 0, DECILE_RANKING_NEGATIVE_SENTINEL, own)[inside]
        agreement[name] = (
            float(w[own == placed].sum() / total) if total else float("nan")
        )
        gaps[name] = (
            float((np.abs(own - placed) * w).sum() / total) if total else float("nan")
        )
    best = max(agreement, key=lambda k: agreement[k])
    sentinel = base.decile == DECILE_RANKING_NEGATIVE_SENTINEL
    all_w = float(base.weight.sum())
    return DecileConceptAudit(
        agreement=agreement,
        best_match=best,
        burden_denominator=income_basis,
        matches_burden_denominator=(best == income_basis),
        mean_absolute_decile_gap=gaps,
        best_match_is_documented_truth=(best == DECILE_RANKING_TRUTH),
        person_weighted=person_weighted,
        negative_sentinel_share=(
            float(base.weight[sentinel].sum() / all_w) if all_w else float("nan")
        ),
        unavailable=tuple(unavailable),
    )


@dataclass
class GeographyRow:
    name: str
    mean_loss_gbp: float
    mean_loss_pct: float
    households_m: float


def geography_table(
    base: Baseline,
    cost: np.ndarray,
    level: str = "region",
    income_basis: str = DEFAULT_INCOME_BASIS,
) -> list[GeographyRow]:
    """Loss by region or country, as £ and as % of equivalised AHC income (D1).

    **Not constituency.** The dataset carries no constituency weight matrix and
    its ``local_authority`` column is a degenerate default (a single value for
    every household), so the paper's intended 650-seat figure is not producible
    from this release. Region (12) and country (4) are the real geography.
    """
    labels = base.region if level == "region" else base.country
    income = income_for_ratio(base, income_basis)
    rows: list[GeographyRow] = []
    for name in sorted(set(labels.tolist())):
        sel = labels == name
        w = base.weight[sel]
        if w.sum() <= 0:
            continue
        rows.append(
            GeographyRow(
                name=str(name),
                mean_loss_gbp=wmean(cost[sel], w),
                mean_loss_pct=pct_of_income(cost[sel], income[sel], w),
                households_m=float(w.sum()) / 1e6,
            )
        )
    return rows


@dataclass
class ScenarioResult:
    """Everything the paper reports for one scenario, pre-policy."""

    scenario: str
    aggregate_cost_bn: float
    mean_loss_gbp: float
    mean_loss_pct: float
    gas_share_of_loss: float
    electricity_share_of_loss: float
    motor_fuel_share_of_loss: float
    decile: list[DecileRow] = field(default_factory=list)
    intra_decile: list[IntraDecileRow] = field(default_factory=list)
    region: list[GeographyRow] = field(default_factory=list)
    gini_baseline: float = float("nan")
    gini_after: float = float("nan")
    poverty_bhc_baseline: float = float("nan")
    poverty_ahc_baseline: float = float("nan")
    income_basis: str = DEFAULT_INCOME_BASIS
    domestic_basis: str = DEFAULT_DOMESTIC_BASIS
    calibration: str = "raw"
    #: Share of weighted households dropped from the percentage statistics for
    #: having zero or negative income on this basis — about 20% of decile one on
    #: the equivalised AHC basis (A1/A6), against 0.57% unequivalised.
    zero_or_negative_income_share: float = float("nan")
    #: Mean of the income concept the percentages divide by. The sanity check
    #: that caught D1: on the unequivalised basis this was ~£60,825, about double
    #: the published HBAI equivalised AHC mean.
    mean_income_gbp: float = float("nan")
    median_income_gbp: float = float("nan")
    #: Mean loss as a percentage of income with the **numerator equivalised too**
    #: — cost divided by the same AHC scale before aggregation. The headline
    #: keeps the cash numerator over the equivalised denominator (the literal
    #: reading of D1, and the aggregate-ratio convention), which is mechanically
    #: larger by roughly the mean equivalisation factor; this field makes the
    #: difference auditable rather than invisible. It is NaN-free only when the
    #: baseline carries ``equivalisation_ahc``; a synthetic baseline without it
    #: reproduces ``mean_loss_pct``.
    mean_loss_pct_equivalised_both: float = float("nan")
    #: Annual domestic phase-in factors actually applied (D2).
    annual_phase_in_gas: float = float("nan")
    annual_phase_in_electricity: float = float("nan")
    coverage: DecileCoverage | None = None
    # --- round-2 disclosures -------------------------------------------
    #: The window both legs are annualised over. Round-2 finding 1: they used
    #: to be different windows, summed and labelled "2026".
    modelled_window: str = ""
    #: Peak-to-window damping actually applied to each leg. Recorded on EVERY
    #: result because Table 1 is not like-for-like without it: ``realised_2026``
    #: damps the pump leg to 0.65 while both NIESR scenarios use 1.0 (their
    #: paths are specified as sustained levels, not peaks, so 1.0 is right for
    #: them — but the reader has to be told, and the prose cannot disclose what
    #: the results do not carry).
    gas_sustained_fraction: float = float("nan")
    pump_sustained_fraction: float = float("nan")
    cap_lag_quarters: float = float("nan")
    #: Within- versus between-decile dispersion, with and without decile one.
    dispersion: DispersionSummary | None = None
    #: What income concept the decile ranking variable actually ranks on.
    decile_concept: DecileConceptAudit | None = None
    # --- the domestic-energy-only robustness anchor ---------------------
    #: The decile gradient computed on the **domestic-energy leg alone**.
    #:
    #: ``docs/VALIDATION.md`` ranks this the most informative check available to
    #: the paper and the decomposition was already being computed. Motor fuel is
    #: the channel with the broken imputation — ONS puts decile-one motor-fuel
    #: spend at £318 against our £1,073, and DfT NTS0703 puts 40% of the bottom
    #: income quintile without a car against our identical fuel-purchasing rate
    #: in deciles one and ten — so the all-channel gradient inherits that
    #: defect. The domestic leg is imputed from a variable 99% of households
    #: record, and it is the leg every bill-based instrument in the scorecard
    #: acts on. It is the gradient that can be defended.
    decile_domestic_only: list[DecileRow] = field(default_factory=list)
    domestic_only_d1_d10_ratio_pct: float = float("nan")
    domestic_only_d1_d10_ratio_gbp: float = float("nan")
    all_channel_d1_d10_ratio_pct: float = float("nan")
    # --- round-3 finding 9: the gradient without decile one, and without the
    # --- "drop the non-positive tail" convention that shapes decile one.
    #: The gradient measured from decile **two** instead of decile one. Decile
    #: one is where 20% of equivalised AHC incomes are non-positive and are
    #: dropped; decile two is the steepest point of the distribution that does
    #: not depend on that treatment.
    d2_d10_ratio_pct: float = float("nan")
    d2_d10_ratio_gbp: float = float("nan")
    #: The same statistics under ``"winsorise_p1"``: the non-positive tail kept
    #: in at the first percentile of positive income rather than dropped. If the
    #: gradient is a statement about incidence it should survive this; if it is a
    #: statement about the denominator convention, it will not.
    income_treatment: str = DEFAULT_INCOME_TREATMENT
    decile1_loss_pct_winsorised: float = float("nan")
    decile2_loss_pct_winsorised: float = float("nan")
    decile10_loss_pct_winsorised: float = float("nan")
    d1_d10_ratio_pct_winsorised: float = float("nan")
    d2_d10_ratio_pct_winsorised: float = float("nan")
    mean_loss_pct_winsorised: float = float("nan")
    #: Motor-fuel margin diagnostics, when the baseline carries ``means_tested``.
    motor_fuel_margins: MotorFuelMargins | None = None


def _ratio(rows: list[DecileRow], field_name: str, bottom: int = 1) -> float:
    """``bottom``-over-decile-ten value of ``field_name``, or NaN.

    ``bottom=2`` gives the gradient measured from decile two, which is the
    round-3 robustness ask: decile one is the decile in which a fifth of
    equivalised AHC incomes are non-positive and dropped from the denominator.
    """
    if not rows:
        return float("nan")
    top = next((r for r in rows if r.decile == 10), None)
    bottom_row = next((r for r in rows if r.decile == bottom), None)
    if top is None or bottom_row is None:
        return float("nan")
    denominator = getattr(top, field_name)
    return (
        float(getattr(bottom_row, field_name) / denominator)
        if denominator
        else float("nan")
    )


def _module_window_label(scenario: Any) -> str:
    """The modelled window the scenario module declares, if it declares one."""
    try:
        from uk_iran_conflict.scenarios import MODELLED_WINDOW_LABEL  # noqa: PLC0415
    except ImportError:  # pragma: no cover - synthetic scenarios in tests
        return ""
    return MODELLED_WINDOW_LABEL


def run_scenario(
    base: Baseline,
    scenario: Any,
    ons_fuel_calibration: bool = False,
    *,
    income_basis: str = DEFAULT_INCOME_BASIS,
    domestic_basis: str = DEFAULT_DOMESTIC_BASIS,
    calibration: str | None = None,
    income_treatment: str = DEFAULT_INCOME_TREATMENT,
) -> tuple[ScenarioResult, ShockCost]:
    """Score one scenario against the baseline.

    Parameters
    ----------
    ons_fuel_calibration:
        Back-compatible shorthand for ``calibration="ons_fuel_shape"``.
    income_basis:
        Denominator for every percentage-of-income statistic. Defaults to
        equivalised AHC income (``docs/FIXES.md`` decision D1);
        ``"unequivalised"`` reproduces the pre-D1 gradient as a robustness line.
    domestic_basis:
        ``"annual"`` (default, decision D2) applies the paper's Step 1
        consumption-weighted quarterly cap average to the domestic leg;
        ``"steady_state"`` charges full pass-through for twelve months.
    calibration:
        One of :data:`CALIBRATIONS`. ``"ons_both_levels"`` is the specification
        that corrects *both* imputation levels against ONS (C11);
        ``"mt_fuel_parity"`` and ``"nts_participation"`` are the two round-3
        margin specifications and both need ``base.means_tested`` / a decile.
    income_treatment:
        One of :data:`NON_POSITIVE_INCOME_TREATMENTS`, applied to the
        denominator. The ``"winsorise_p1"`` statistics are computed and reported
        on **every** run regardless, so the sensitivity of the gradient to this
        convention is always in the results (round-3 finding 9).

    Returns the result and the :class:`ShockCost`; note that under any
    non-``"raw"`` calibration the cost is computed on the recalibrated baseline,
    so downstream policy scoring must use the same baseline (see
    ``analysis/run_variants.py``).
    """
    if calibration is None:
        calibration = "ons_fuel_shape" if ons_fuel_calibration else "raw"
    elif ons_fuel_calibration:
        raise ValueError("pass either ons_fuel_calibration or calibration, not both")
    base = apply_calibration(base, calibration)
    cost = shock_cost(base, scenario, domestic_basis)
    total = cost.total
    w = base.weight
    agg = wsum(total, w)
    income = treat_non_positive_income(
        income_for_ratio(base, income_basis), base.weight, income_treatment
    )
    equivalisation = np.clip(base.equivalisation, 1e-9, None)
    after = np.clip(income - total, 0, None)
    pass_through = getattr(scenario, "pass_through", None)
    decile_rows = decile_table(base, total, income_basis, income_treatment)
    intra_rows = intra_decile_table(base, total, income_basis, income_treatment)
    domestic_rows = decile_table(base, cost.domestic, income_basis, income_treatment)

    # Round-3 finding 9. The gradient is recomputed two further ways on every
    # run: from decile two (so it does not rest on the decile where a fifth of
    # equivalised AHC incomes are non-positive) and with that tail winsorised
    # back in rather than dropped.
    winsorised = treat_non_positive_income(income, w, "winsorise_p1")

    def _decile_pct(values: np.ndarray, d: int) -> float:
        sel = base.decile == d
        return (
            pct_of_income(total[sel], values[sel], w[sel])
            if w[sel].sum() > 0
            else float("nan")
        )

    d1_win, d2_win, d10_win = (_decile_pct(winsorised, d) for d in (1, 2, 10))
    margins = motor_fuel_margins(base, total) if base.means_tested is not None else None
    return (
        ScenarioResult(
            scenario=cost.scenario,
            aggregate_cost_bn=agg / 1e9,
            mean_loss_gbp=wmean(total, w),
            mean_loss_pct=pct_of_income(total, income, w),
            gas_share_of_loss=wsum(cost.gas, w) / agg if agg else float("nan"),
            electricity_share_of_loss=(
                wsum(cost.electricity, w) / agg if agg else float("nan")
            ),
            motor_fuel_share_of_loss=(
                wsum(cost.motor_fuel, w) / agg if agg else float("nan")
            ),
            decile=decile_rows,
            intra_decile=intra_rows,
            region=geography_table(base, total, "region", income_basis),
            gini_baseline=gini(income, w),
            gini_after=gini(after, w),
            poverty_bhc_baseline=float(wmean(base.in_poverty_bhc, w)),
            poverty_ahc_baseline=float(wmean(base.in_poverty_ahc, w)),
            income_basis=income_basis,
            domestic_basis=domestic_basis,
            calibration=calibration,
            zero_or_negative_income_share=float(
                w[income <= 0].sum() / w.sum() if w.sum() else float("nan")
            ),
            mean_income_gbp=wmean(income, w),
            median_income_gbp=wquantile(income, w, 0.5),
            mean_loss_pct_equivalised_both=pct_of_income(
                total / equivalisation, income, w
            ),
            annual_phase_in_gas=float(
                getattr(pass_through, "annual_phase_in_gas", float("nan"))
            ),
            annual_phase_in_electricity=float(
                getattr(pass_through, "annual_phase_in_electricity", float("nan"))
            ),
            coverage=decile_coverage(base, total, income_basis),
            modelled_window=str(
                getattr(scenario, "modelled_window", "")
                or _module_window_label(scenario)
            ),
            gas_sustained_fraction=float(
                getattr(pass_through, "sustained_fraction", float("nan"))
            ),
            pump_sustained_fraction=float(
                getattr(pass_through, "pump_sustained_fraction", float("nan"))
            ),
            cap_lag_quarters=float(getattr(pass_through, "lag_quarters", float("nan"))),
            dispersion=dispersion_summary(intra_rows, decile_rows),
            decile_concept=decile_concept_audit(base, income_basis),
            decile_domestic_only=domestic_rows,
            domestic_only_d1_d10_ratio_pct=_ratio(domestic_rows, "mean_loss_pct"),
            domestic_only_d1_d10_ratio_gbp=_ratio(domestic_rows, "mean_loss_gbp"),
            all_channel_d1_d10_ratio_pct=_ratio(decile_rows, "mean_loss_pct"),
            d2_d10_ratio_pct=_ratio(decile_rows, "mean_loss_pct", bottom=2),
            d2_d10_ratio_gbp=_ratio(decile_rows, "mean_loss_gbp", bottom=2),
            income_treatment=income_treatment,
            decile1_loss_pct_winsorised=d1_win,
            decile2_loss_pct_winsorised=d2_win,
            decile10_loss_pct_winsorised=d10_win,
            d1_d10_ratio_pct_winsorised=(d1_win / d10_win if d10_win else float("nan")),
            d2_d10_ratio_pct_winsorised=(d2_win / d10_win if d10_win else float("nan")),
            mean_loss_pct_winsorised=pct_of_income(total, winsorised, w),
            motor_fuel_margins=margins,
        ),
        cost,
    )

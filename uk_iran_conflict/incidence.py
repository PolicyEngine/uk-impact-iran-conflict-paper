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


#: Baseline calibrations selectable by :func:`run_scenario`.
#:
#: ``"raw"``
#:     The microdata as imputed. The main specification.
#: ``"ons_fuel_shape"``
#:     :func:`rescale_motor_fuel_to_ons` — ONS motor-fuel decile *shape*, the
#:     microdata's national fuel total preserved.
#: ``"ons_both_levels"``
#:     :func:`rescale_to_ons_levels` — ONS *levels* on both legs.
CALIBRATIONS: tuple[str, ...] = ("raw", "ons_fuel_shape", "ons_both_levels")


def apply_calibration(base: Baseline, calibration: str) -> Baseline:
    """Return ``base`` under one of :data:`CALIBRATIONS`."""
    if calibration == "raw":
        return base
    if calibration == "ons_fuel_shape":
        return rescale_motor_fuel_to_ons(base)
    if calibration == "ons_both_levels":
        return rescale_to_ons_levels(base)
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
    scenario's ``pass_through.pump_sustained_fraction`` is applied here. It
    defaults to 1.0, so any scenario that does not set it is unchanged.

    Raises on a scenario carrying no pump path rather than returning a silently
    unshocked 1.0 (``docs/FIXES.md`` E33).
    """
    pump = getattr(scenario, "pump", None)
    if pump is None:
        raise AttributeError(
            f"scenario {getattr(scenario, 'key', scenario)!r} exposes no pump "
            "path; refusing to fall back to an unshocked 1.0"
        )
    pass_through = getattr(scenario, "pass_through", None)
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
    base: Baseline, cost: np.ndarray, income_basis: str = DEFAULT_INCOME_BASIS
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
    income = income_for_ratio(base, income_basis)
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
    base: Baseline, cost: np.ndarray, income_basis: str = DEFAULT_INCOME_BASIS
) -> list[IntraDecileRow]:
    """Dispersion of the loss *within* each decile.

    Cronin, Fullerton and Sexton (2019) show horizontal redistribution exceeds
    the vertical kind; a decile mean cannot show it, so we report the spread and
    the share of households taking an unusually large hit. Ratios are taken
    against equivalised AHC income by default (D1).
    """
    rows: list[IntraDecileRow] = []
    income = income_for_ratio(base, income_basis)
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


def run_scenario(
    base: Baseline,
    scenario: Any,
    ons_fuel_calibration: bool = False,
    *,
    income_basis: str = DEFAULT_INCOME_BASIS,
    domestic_basis: str = DEFAULT_DOMESTIC_BASIS,
    calibration: str | None = None,
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
        that corrects *both* imputation levels against ONS (C11).

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
    income = income_for_ratio(base, income_basis)
    equivalisation = np.clip(base.equivalisation, 1e-9, None)
    after = np.clip(income - total, 0, None)
    pass_through = getattr(scenario, "pass_through", None)
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
            decile=decile_table(base, total, income_basis),
            intra_decile=intra_decile_table(base, total, income_basis),
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
        ),
        cost,
    )

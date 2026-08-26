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
    "in_relative_poverty_ahc",
    "in_relative_poverty_bhc",
    "in_poverty_bhc",
    "in_poverty_ahc",
    "region",
    "country",
)

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


def sustained_pump_factors(scenario: Any) -> tuple[float, float]:
    """(petrol, diesel) pump multipliers **after** peak-to-year damping.

    ``reforms.pump_price_factors`` returns the raw quoted moves, which for the
    realised path are observed *peaks*. Charging a household the peak pump price
    for twelve months while damping the gas peak to its cap-relevant fraction is
    the inconsistency ``docs/VALIDATION.md`` Check 2b identifies, so the
    scenario's ``pass_through.pump_sustained_fraction`` is applied here. It
    defaults to 1.0, so any scenario that does not set it is unchanged.
    """
    from uk_iran_conflict import reforms  # noqa: PLC0415 — avoids a cycle

    petrol_factor, diesel_factor = reforms.pump_price_factors(scenario)
    pass_through = getattr(scenario, "pass_through", None)
    fraction = float(getattr(pass_through, "pump_sustained_fraction", 1.0))
    return (
        1.0 + fraction * (petrol_factor - 1.0),
        1.0 + fraction * (diesel_factor - 1.0),
    )


def shock_cost(base: Baseline, scenario: Any) -> ShockCost:
    """First-order cost of ``scenario``: quantity fixed, price moved.

    Gas and electricity are shocked **asymmetrically** — the scenario carries
    separate factors because gas sets the marginal electricity price only about
    85% of the time, so a wholesale gas move does not reach the two fuels in the
    same proportion. Collapsing them onto ``domestic_energy_consumption`` would
    discard the paper's central modelling claim.
    """
    from uk_iran_conflict import reforms  # noqa: PLC0415 — avoids a cycle

    gas_factor, elec_factor = reforms.retail_factors(scenario)
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


def positive_income(base: Baseline) -> np.ndarray:
    """Mask of households with strictly positive net income."""
    return base.net_income > 0


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile via the cumulative weight of sorted values."""
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws)
    if cum[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cum[-1], cum, xs))


def gini(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted Gini coefficient, floored at zero income."""
    x = np.clip(x, 0, None)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    cxw = np.cumsum(xs * ws)
    if cxw[-1] <= 0:
        return float("nan")
    # Trapezoidal Lorenz-curve area.
    return float(1 - np.sum((cxw[1:] + cxw[:-1]) * np.diff(cw)) / (cxw[-1] * cw[-1]))


def top_share(x: np.ndarray, w: np.ndarray, top: float) -> float:
    """Income share of the top ``top`` fraction (e.g. 0.01 for the top 1%)."""
    cut = wquantile(x, w, 1 - top)
    sel = x >= cut
    total = wsum(x, w)
    return float(wsum(x[sel], w[sel]) / total) if total else float("nan")


def bottom_share(x: np.ndarray, w: np.ndarray, bottom: float) -> float:
    cut = wquantile(x, w, bottom)
    sel = x <= cut
    total = wsum(x, w)
    return float(wsum(x[sel], w[sel]) / total) if total else float("nan")


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


def decile_table(base: Baseline, cost: np.ndarray) -> list[DecileRow]:
    """Loss by income decile, in **both** £ and % of net income.

    The two orderings differ — that contrast is the paper's central empirical
    point (Fetzer, Gazze and Bishop 2024 in £; the budget-share literature in %).
    """
    rows: list[DecileRow] = []
    total = wsum(cost, base.weight)
    for d in range(1, 11):
        sel = base.decile == d
        w = base.weight[sel]
        if w.sum() <= 0:
            continue
        c = cost[sel]
        income = np.clip(base.net_income[sel], 1, None)
        rows.append(
            DecileRow(
                decile=d,
                mean_loss_gbp=wmean(c, w),
                mean_loss_pct=100 * wshare(c, income, w),
                share_of_total_loss=(wsum(c, w) / total if total else float("nan")),
                households_m=float(w.sum()) / 1e6,
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


def intra_decile_table(base: Baseline, cost: np.ndarray) -> list[IntraDecileRow]:
    """Dispersion of the loss *within* each decile.

    Cronin, Fullerton and Sexton (2019) show horizontal redistribution exceeds
    the vertical kind; a decile mean cannot show it, so we report the spread and
    the share of households taking an unusually large hit.
    """
    rows: list[IntraDecileRow] = []
    for d in range(1, 11):
        sel = base.decile == d
        w = base.weight[sel]
        if w.sum() <= 0:
            continue
        # Within-decile spread is a distribution of household-level ratios,
        # so it must exclude non-positive incomes rather than clip them.
        inc = base.net_income[sel]
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
    base: Baseline, cost: np.ndarray, level: str = "region"
) -> list[GeographyRow]:
    """Loss by region or country.

    **Not constituency.** The dataset carries no constituency weight matrix and
    its ``local_authority`` column is a degenerate default (a single value for
    every household), so the paper's intended 650-seat figure is not producible
    from this release. Region (12) and country (4) are the real geography.
    """
    labels = base.region if level == "region" else base.country
    rows: list[GeographyRow] = []
    for name in sorted(set(labels.tolist())):
        sel = labels == name
        w = base.weight[sel]
        if w.sum() <= 0:
            continue
        income = np.clip(base.net_income[sel], 1, None)
        rows.append(
            GeographyRow(
                name=str(name),
                mean_loss_gbp=wmean(cost[sel], w),
                mean_loss_pct=100 * wshare(cost[sel], income, w),
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


def run_scenario(
    base: Baseline, scenario: Any, ons_fuel_calibration: bool = False
) -> tuple[ScenarioResult, ShockCost]:
    """Score one scenario against the baseline.

    Parameters
    ----------
    ons_fuel_calibration:
        If ``True``, motor-fuel spend is first reweighted onto the ONS Family
        Spending decile profile via :func:`rescale_motor_fuel_to_ons`. This is a
        **robustness variant** and is off by default: the main specification is
        the raw microdata.

    Returns the result and the :class:`ShockCost`; note that when
    ``ons_fuel_calibration`` is set, the cost is computed on the reweighted
    baseline, so any downstream policy scoring must use the same baseline (see
    ``analysis/run_variants.py``).
    """
    if ons_fuel_calibration:
        base = rescale_motor_fuel_to_ons(base)
    cost = shock_cost(base, scenario)
    total = cost.total
    w = base.weight
    agg = wsum(total, w)
    income = np.clip(base.net_income, 1, None)
    after = np.clip(base.net_income - total, 0, None)
    return (
        ScenarioResult(
            scenario=cost.scenario,
            aggregate_cost_bn=agg / 1e9,
            mean_loss_gbp=wmean(total, w),
            mean_loss_pct=100 * wshare(total, income, w),
            gas_share_of_loss=wsum(cost.gas, w) / agg if agg else float("nan"),
            electricity_share_of_loss=(
                wsum(cost.electricity, w) / agg if agg else float("nan")
            ),
            motor_fuel_share_of_loss=(
                wsum(cost.motor_fuel, w) / agg if agg else float("nan")
            ),
            decile=decile_table(base, total),
            intra_decile=intra_decile_table(base, total),
            region=geography_table(base, total, "region"),
            gini_baseline=gini(base.net_income, w),
            gini_after=gini(after, w),
            poverty_bhc_baseline=float(wmean(base.in_poverty_bhc, w)),
            poverty_ahc_baseline=float(wmean(base.in_poverty_ahc, w)),
        ),
        cost,
    )

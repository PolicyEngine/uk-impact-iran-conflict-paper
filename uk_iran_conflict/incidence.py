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
    petrol_factor, diesel_factor = reforms.pump_price_factors(scenario)
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


def run_scenario(base: Baseline, scenario: Any) -> tuple[ScenarioResult, ShockCost]:
    """Score one scenario against the baseline."""
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

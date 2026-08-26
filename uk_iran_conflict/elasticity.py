"""Consumption-response module for the 2026 UK energy shock.

=============================================================================
THE MAIN SPECIFICATION OF THIS PAPER USES ``ZERO_ELASTICITY``.
=============================================================================

Read this before using anything below.

The paper's headline incidence numbers are the **first-order Deaton
approximation with no substitution**:

    Δcost_h = q_h0 · Δp

i.e. the loss a household bears if it holds its physical consumption fixed at
the pre-shock bundle. Under a normal (downward-sloping, non-Giffen) demand
curve this is a strict **upper bound** on the compensating variation: any
substitution a household can make weakly reduces its true welfare loss. See
Deaton & Muellbauer (1980, AIDS) and Deaton (1989) for the first-order
argument; the second-order term is +(1/2)·ε·(Δp/p)·q0·Δp and is negative for
ε < 0.

So: *elasticity in this module is the ROBUSTNESS CHECK, not the headline.*
Every function here defaults to a form that reproduces the static result when
handed ``ZERO_ELASTICITY``, and :data:`ZERO_ELASTICITY` is the default
argument wherever an elasticity is optional. If you find yourself passing
``LABANDEIRA_SHORT_RUN`` or ``PRIESMANN_GAS_BY_DECILE`` into the pipeline that
produces a headline table, you are running a robustness variant and it must be
labelled as such in the output.

This inverts the choice made in the earlier PolicyEngine ``energy-price-shock``
repo, which applied income-varying elasticities in its main behavioural view.
That repo's provenance is recorded in ``docs/PRIOR_WORK_ENERGY.md`` and
carried forward in the docstrings below.

Why the upper bound is the right headline for *this* paper:

* Behavioural response in the winter of a price spike is partly involuntary
  rationing ("heat or eat"), not welfare-neutral substitution. Priesmann &
  Praktiknjo (2025) make exactly this point: low-income households may be
  "forced to impose health-threatening consumption restrictions on
  themselves". Treating their 26% consumption cut as a welfare saving is a
  category error.
* The elasticity estimates available are German, are short-run, and are
  identified off ±10-20% price variation. The 2026 shock is far larger.
* An explicit upper bound is defensible; a point estimate resting on
  transplanted foreign elasticities is not.

--------------------------------------------------------------------------
Upstream note
--------------------------------------------------------------------------

PolicyEngine UK has **no built-in consumption elasticity or price
pass-through** — open issue **#1114**, which records that UKMOD's TCO module
uses a value of **0.8**. This module is deliberately written with pure
functions, full type hints and **no PolicyEngine imports**, so that it stays
unit-testable in isolation and is a plausible candidate for upstream
contribution against #1114.

--------------------------------------------------------------------------
Functional form
--------------------------------------------------------------------------

Constant-elasticity (log-linear) demand:

    q1 / q0     = (p1 / p0) ** ε
    spend1/spend0 = (p1 / p0) ** (1 + ε)

The linear first-order alternative ``(1 + r)(1 + ε·r)`` is **not** used: at
ε = -0.64 and r = +1.61 it returns ``2.61 × (1 - 1.0304) = -0.079``, a negative
spending factor implying negative consumption. The constant-elasticity form
stays strictly positive for every ε ∈ (-1, 0] and r ≥ 0, and collapses to the
linear form as r → 0. (This form and this argument are carried over from
``energy_shock.sections._behavioural_factor_hh``.)

At ε = 0 the spend factor is exactly ``1 + r``, which is the static
first-order case — hence the invariant that ``ZERO_ELASTICITY`` reproduces the
main specification bit-for-bit.

--------------------------------------------------------------------------
Sources for every number in this module
--------------------------------------------------------------------------

**Labandeira, Labeaga & López-Otero (2017)**, "A meta-analysis on the price
elasticity of energy demand", *Energy Policy* 102:549-568. Meta-regression over
917 short-run / 959 long-run estimates (selected sample, 5% trimmed).

  Table 6, average price elasticity by energy product:

  ==============  ==========  =========
  Product         Short run   Long run
  ==============  ==========  =========
  Electricity     -0.126*     -0.365*
  Natural gas     -0.180***   -0.684*
  Gasoline        -0.293***   -0.773***
  Diesel          -0.153**    -0.443***
  Heating oil     -0.017 n.s. -0.185
  ==============  ==========  =========

  (*** 1%, ** 5%, * 10% significance.)

  Table 5, aggregate energy: short run -0.221 (GLS) / -0.207 (fixed effects);
  long run -0.584 / -0.608 — rounded in the paper's conclusion to -0.21 and
  -0.61. Table 4, residential consumers only: -0.215 (SR) / -0.617 (LR).

  NOTE on prior work: the earlier ``energy-price-shock`` README attributed a
  value of "-0.15" to this meta-analysis. **No such figure appears in the
  paper.** Do not propagate it.

**Priesmann & Praktiknjo (2025)**, "Estimating short- and long-run price and
income elasticities of final energy demand as a function of household income",
*Energy Policy* 207:114850. German SOEP (electricity, gas) and MOP (car fuels);
bias-corrected method-of-moments dynamic panel; elasticities as continuous
functions of household income.

  ============  ===========================  ===========================
  Carrier       Short run, low → high income  Long run, low → high income
  ============  ===========================  ===========================
  Electricity   -0.27 → -0.44                -0.22 → -0.64
  Gas/heating   -0.64 → -0.11                -0.58 → -0.15
  Car fuels     -0.47 → -0.14 (not S/L sep.) (same)
  ============  ===========================  ===========================

  **The electricity gradient runs the OPPOSITE WAY to the gas gradient**:
  richer households are *more* electricity-price-elastic, poorer households
  are *more* gas-price-elastic. The prior repo applied the gas gradient to
  combined electricity+gas consumption; this module keeps the carriers
  separate so that mistake cannot recur.

  Table 12 reports the own-estimate car-fuel span as -0.19 to -0.47; the
  abstract and conclusion give -0.47 (low) to -0.14 (high). We use the
  abstract's endpoints and flag the discrepancy.

  Gas elasticity becomes statistically insignificant for high-income
  households above a threshold, so -0.11 at the top decile should be read as
  "approximately no response", not as a precise estimate.

**Transferability caveats** (inherited, with attribution, from the prior repo's
README): these are German estimates. Applying them to UK households assumes the
UK income gradient in responsiveness mirrors Germany's. Interior decile values
are linear interpolation between published endpoints and are a modelling
choice, not a result of either paper. Extrapolating a constant-elasticity form
to shocks far outside the ±10-20% identifying variation is illustrative, not
predictive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "Carrier",
    "ElasticitySpec",
    "ZERO_ELASTICITY",
    "LABANDEIRA_2017_SHORT_RUN",
    "LABANDEIRA_2017_LONG_RUN",
    "LABANDEIRA_2017_RESIDENTIAL_SHORT_RUN",
    "PRIESMANN_2025_SHORT_RUN_ENDPOINTS",
    "PRIESMANN_2025_LONG_RUN_ENDPOINTS",
    "PRIOR_REPO_ELASTICITY_BY_DECILE",
    "interpolate_by_decile",
    "priesmann_by_decile",
    "elasticity_for",
    "quantity_factor",
    "spend_factor",
    "spend_change",
    "static_spend_change",
    "consumption_reduction",
    "deadweight_share",
    "laspeyres_cv",
    "paasche_cv",
    "cv_bounds",
    "welfare_shaved_share",
    "resolve_elasticity_spec",
]

# --------------------------------------------------------------------------
# Carriers
# --------------------------------------------------------------------------

Carrier = Literal["gas", "electricity", "motor_fuel"]

CARRIERS: Final[tuple[Carrier, ...]] = ("gas", "electricity", "motor_fuel")

N_DECILES: Final[int] = 10

# --------------------------------------------------------------------------
# THE MAIN SPECIFICATION
# --------------------------------------------------------------------------

ZERO_ELASTICITY: Final[dict[Carrier, float]] = {
    "gas": 0.0,
    "electricity": 0.0,
    "motor_fuel": 0.0,
}
"""No-substitution elasticities — **the paper's main specification**.

With ε = 0 the spend factor is exactly ``1 + r``, reproducing the first-order
Deaton approximation Δcost = q0 · Δp. This is an explicit upper bound on the
welfare loss, not a behavioural prediction. Anything else in this module is a
robustness variant.
"""

# --------------------------------------------------------------------------
# Labandeira et al. (2017) — flat (income-invariant) elasticities
# --------------------------------------------------------------------------

LABANDEIRA_2017_SHORT_RUN: Final[dict[Carrier, float]] = {
    "gas": -0.180,  # Table 6, "Natural Gas", significant at 1%
    "electricity": -0.126,  # Table 6, "Electricity", significant at 10%
    "motor_fuel": -0.293,  # Table 6, "Gasoline", significant at 1%
}
"""Labandeira et al. (2017) Table 6 short-run means.

Motor fuel uses the gasoline figure (-0.293). The diesel figure is -0.153; for
a diesel-weighted basket, blend the two by litres. Heating oil is -0.017
(not significant) — relevant to the off-gas-grid robustness cut.
"""

LABANDEIRA_2017_LONG_RUN: Final[dict[Carrier, float]] = {
    "gas": -0.684,  # Table 6
    "electricity": -0.365,  # Table 6
    "motor_fuel": -0.773,  # Table 6, gasoline
}
"""Labandeira et al. (2017) Table 6 long-run means.

Long-run response is **not** the right horizon for a within-year price spike —
it embeds appliance and vehicle stock turnover. Included only for a
sensitivity band.
"""

LABANDEIRA_2017_DIESEL_SHORT_RUN: Final[float] = -0.153
LABANDEIRA_2017_HEATING_OIL_SHORT_RUN: Final[float] = -0.017

LABANDEIRA_2017_RESIDENTIAL_SHORT_RUN: Final[float] = -0.215
"""Labandeira et al. (2017) Table 4, residential consumers, all products.

The single-number aggregate alternative to the per-carrier Table 6 values.
Table 5's aggregate-energy equivalents are -0.221 (GLS) / -0.207 (fixed
effects), rounded to -0.21 in the paper's conclusion.
"""

# --------------------------------------------------------------------------
# Priesmann & Praktiknjo (2025) — income-varying elasticities
# --------------------------------------------------------------------------

PRIESMANN_2025_SHORT_RUN_ENDPOINTS: Final[dict[Carrier, tuple[float, float]]] = {
    # (lowest-income endpoint, highest-income endpoint)
    "gas": (-0.64, -0.11),
    "electricity": (-0.27, -0.44),
    "motor_fuel": (-0.47, -0.14),
}
"""Priesmann & Praktiknjo (2025) short-run endpoints, (low income, high income).

Gas falls in magnitude with income (-0.64 → -0.11); **electricity rises**
(-0.27 → -0.44); car fuels fall (-0.47 → -0.14, no short/long separation).
The electricity sign reversal is the substantive reason to keep carriers apart.
"""

PRIESMANN_2025_LONG_RUN_ENDPOINTS: Final[dict[Carrier, tuple[float, float]]] = {
    "gas": (-0.58, -0.15),
    "electricity": (-0.22, -0.64),
    "motor_fuel": (-0.47, -0.14),  # not separable short/long in the source
}
"""Priesmann & Praktiknjo (2025) long-run endpoints, (low income, high income)."""

PRIESMANN_2025_INCOME_ELASTICITY_ENDPOINTS: Final[
    dict[Carrier, tuple[float, float]]
] = {
    "gas": (0.079, 0.0),  # declines with income; ~insignificant at the top
    "electricity": (0.048, 0.0),  # 0.048 (low) → insignificant (high)
    "motor_fuel": (0.060, 0.443),  # the one carrier that RISES with income
}
"""Priesmann & Praktiknjo (2025) short-run *income* elasticities.

Not used by the price-response functions here; recorded because the motor-fuel
income gradient (0.060 → 0.443) matters for any real-income-loss second round.
"""

PRIOR_REPO_ELASTICITY_BY_DECILE: Final[dict[int, float]] = {
    d: -0.64 + (d - 1) * (-0.11 - -0.64) / 9 for d in range(1, N_DECILES + 1)
}
"""Exactly what ``energy-price-shock/energy_shock/config.py`` used.

D1 = -0.640 … D10 = -0.110, linearly interpolated, applied to **combined**
electricity + gas consumption. Reproduced verbatim so the new paper can
replicate the prior repo's headline numbers (weighted mean -0.382; D1
consumption cut 26.0% at +60%).

It is **not** recommended for new work: it applies the *gas* gradient to
electricity, whose gradient in the source paper runs the other way. Use
:func:`priesmann_by_decile` per carrier instead.
"""


# --------------------------------------------------------------------------
# Income-varying elasticity construction
# --------------------------------------------------------------------------


def interpolate_by_decile(
    low: float,
    high: float,
    n_deciles: int = N_DECILES,
) -> dict[int, float]:
    """Linearly interpolate an elasticity across income deciles.

    ``low`` is assigned to decile 1 and ``high`` to decile ``n_deciles``.

    The published sources give **endpoints only** (a low-income and a
    high-income value). The interior deciles produced here are a modelling
    convenience, not an estimate — the same choice the prior repo documented:
    "the interior values (D2-D9) are our own linear interpolation between
    those endpoints and are not a result of that paper."

    >>> interpolate_by_decile(-0.64, -0.11)[1]
    -0.64
    >>> round(interpolate_by_decile(-0.64, -0.11)[10], 3)
    -0.11
    """
    if n_deciles < 2:
        raise ValueError("n_deciles must be at least 2")
    step = (high - low) / (n_deciles - 1)
    return {d: low + (d - 1) * step for d in range(1, n_deciles + 1)}


def priesmann_by_decile(
    carrier: Carrier,
    horizon: Literal["short_run", "long_run"] = "short_run",
    n_deciles: int = N_DECILES,
) -> dict[int, float]:
    """Income-varying elasticities for one carrier, by income decile.

    Endpoints from Priesmann & Praktiknjo (2025); interior deciles linearly
    interpolated (see :func:`interpolate_by_decile`).

    >>> round(priesmann_by_decile("gas")[1], 3)
    -0.64
    >>> round(priesmann_by_decile("electricity")[1], 3)
    -0.27
    >>> round(priesmann_by_decile("electricity")[10], 3)
    -0.44
    """
    table = (
        PRIESMANN_2025_SHORT_RUN_ENDPOINTS
        if horizon == "short_run"
        else PRIESMANN_2025_LONG_RUN_ENDPOINTS
    )
    if carrier not in table:
        raise KeyError(f"unknown carrier {carrier!r}; expected one of {CARRIERS}")
    low, high = table[carrier]
    return interpolate_by_decile(low, high, n_deciles=n_deciles)


@dataclass(frozen=True, slots=True)
class ElasticitySpec:
    """A complete, self-describing elasticity specification for one run.

    Either a flat per-carrier elasticity or an income-varying one, never both.

    :param name: identifier that must appear in any output table produced with
        this spec, so a robustness run can never be mistaken for the headline.
    :param flat: per-carrier income-invariant elasticities.
    :param by_decile: per-carrier ``{decile: elasticity}`` mappings.
    :param source: citation string.
    :param is_main_specification: True only for the zero-elasticity spec.
    """

    name: str
    source: str
    flat: Mapping[Carrier, float] | None = None
    by_decile: Mapping[Carrier, Mapping[int, float]] | None = None
    is_main_specification: bool = False

    def __post_init__(self) -> None:
        if (self.flat is None) == (self.by_decile is None):
            raise ValueError("supply exactly one of `flat` or `by_decile`")

    def epsilon(self, carrier: Carrier, decile: int | None = None) -> float:
        """Elasticity for a carrier, optionally at a given income decile."""
        return elasticity_for(self, carrier, decile)

    @classmethod
    def main(cls) -> ElasticitySpec:
        """The paper's main specification: no substitution.

        >>> ElasticitySpec.main().is_main_specification
        True
        >>> ElasticitySpec.main().epsilon("gas")
        0.0
        """
        return cls(
            name="zero_elasticity_main_spec",
            source=(
                "First-order Deaton approximation, no substitution. "
                "Deaton & Muellbauer (1980). Explicit upper bound on welfare loss."
            ),
            flat=ZERO_ELASTICITY,
            is_main_specification=True,
        )

    @classmethod
    def labandeira_flat(
        cls, horizon: Literal["short_run", "long_run"] = "short_run"
    ) -> ElasticitySpec:
        """Robustness variant: flat per-carrier meta-analytic elasticities."""
        return cls(
            name=f"labandeira_2017_{horizon}",
            source=(
                "Labandeira, Labeaga & Lopez-Otero (2017), Energy Policy "
                "102:549-568, Table 6."
            ),
            flat=(
                LABANDEIRA_2017_SHORT_RUN
                if horizon == "short_run"
                else LABANDEIRA_2017_LONG_RUN
            ),
        )

    @classmethod
    def priesmann_income_varying(
        cls,
        horizon: Literal["short_run", "long_run"] = "short_run",
        n_deciles: int = N_DECILES,
    ) -> ElasticitySpec:
        """Robustness variant: income-varying elasticities, per carrier.

        This is the specification Priesmann & Praktiknjo exists to support:
        gas responsiveness falling with income, electricity responsiveness
        rising with it.
        """
        return cls(
            name=f"priesmann_2025_{horizon}_by_decile",
            source=(
                "Priesmann & Praktiknjo (2025), Energy Policy 207:114850; "
                "published endpoints, interior deciles interpolated."
            ),
            by_decile={
                c: priesmann_by_decile(c, horizon=horizon, n_deciles=n_deciles)
                for c in CARRIERS
            },
        )

    @classmethod
    def prior_repo_replication(cls) -> ElasticitySpec:
        """Replicates ``energy-price-shock``'s combined-carrier gas gradient.

        Provided so the earlier repo's published numbers can be reproduced.
        Not recommended for new estimates — see
        :data:`PRIOR_REPO_ELASTICITY_BY_DECILE`.
        """
        table = dict(PRIOR_REPO_ELASTICITY_BY_DECILE)
        return cls(
            name="prior_repo_gas_gradient_all_carriers",
            source=(
                "PolicyEngine/energy-price-shock config.ELASTICITY_BY_DECILE; "
                "Priesmann gas endpoints applied to combined elec+gas."
            ),
            by_decile={c: table for c in CARRIERS},
        )


def elasticity_for(
    spec: ElasticitySpec,
    carrier: Carrier,
    decile: int | None = None,
    missing_decile_fallback: float | None = None,
) -> float:
    """Resolve a single elasticity from a spec.

    For an income-varying spec, ``decile`` is required. Deciles outside
    ``1..N`` (top-coded or missing in the microdata) resolve to
    ``missing_decile_fallback`` if given, else to the unweighted mean of the
    carrier's decile table — never to 0.0, which would silently eliminate the
    household's response and quietly turn a robustness run into the main spec.

    >>> elasticity_for(ElasticitySpec.main(), "gas")
    0.0
    >>> round(elasticity_for(ElasticitySpec.priesmann_income_varying(), "gas", 1), 3)
    -0.64
    """
    if spec.flat is not None:
        try:
            return float(spec.flat[carrier])
        except KeyError as exc:
            raise KeyError(f"spec {spec.name!r} has no carrier {carrier!r}") from exc

    assert spec.by_decile is not None
    try:
        table = spec.by_decile[carrier]
    except KeyError as exc:
        raise KeyError(f"spec {spec.name!r} has no carrier {carrier!r}") from exc

    if decile is None:
        raise ValueError(
            f"spec {spec.name!r} is income-varying; a decile must be supplied"
        )
    if decile in table:
        return float(table[decile])
    if missing_decile_fallback is not None:
        return float(missing_decile_fallback)
    return float(sum(table.values()) / len(table))


# --------------------------------------------------------------------------
# Demand response
# --------------------------------------------------------------------------


def _check_price_ratio(price_ratio: float) -> None:
    if price_ratio <= 0.0:
        raise ValueError(f"price_ratio must be positive, got {price_ratio}")


def _check_epsilon(epsilon: float) -> None:
    if epsilon > 0.0:
        raise ValueError(
            f"epsilon must be <= 0 for a normal good, got {epsilon}; "
            "a positive own-price elasticity implies a Giffen good"
        )
    if epsilon <= -1.0:
        raise ValueError(
            f"epsilon must be > -1, got {epsilon}; at or below -1 a price rise "
            "weakly reduces total spend, which no published residential "
            "estimate supports (Labandeira 2017 short-run range is "
            "[-0.803, 0.066])"
        )


def quantity_factor(price_ratio: float, epsilon: float) -> float:
    """Ratio of post-shock to pre-shock **quantity**: ``(p1/p0) ** ε``.

    ``price_ratio`` is ``p1 / p0`` (so 1.60 for a +60% shock).

    >>> quantity_factor(1.6, 0.0)
    1.0
    >>> round(quantity_factor(1.6, -0.64), 4)
    0.7402
    """
    _check_price_ratio(price_ratio)
    _check_epsilon(epsilon)
    return price_ratio**epsilon


def spend_factor(price_ratio: float, epsilon: float) -> float:
    """Ratio of post-shock to pre-shock **spend**: ``(p1/p0) ** (1 + ε)``.

    At ε = 0 this is exactly ``price_ratio`` — the main specification.

    Never uses the linear form ``(1 + r)(1 + ε·r)``, which returns a negative
    (physically impossible) factor at e.g. ε = -0.64, r = +1.61.

    >>> round(spend_factor(1.6, 0.0), 6)
    1.6
    >>> round(spend_factor(2.61, -0.64), 4)   # +161%, D1 gas elasticity
    1.4125
    >>> spend_factor(2.61, -0.64) > 0         # linear form would give -0.079
    True
    """
    _check_price_ratio(price_ratio)
    _check_epsilon(epsilon)
    return price_ratio ** (1.0 + epsilon)


def spend_change(
    baseline_spend: float,
    price_ratio: float,
    epsilon: float = 0.0,
) -> float:
    """Change in annual spend on one carrier, in the same units as input.

    ``epsilon`` defaults to 0.0 — the main specification. Passing a non-zero
    value makes this a robustness calculation.

    >>> round(spend_change(1000.0, 1.6), 6)   # main spec: pure first-order
    600.0
    >>> round(spend_change(1000.0, 1.6, -0.64), 1)
    184.4
    """
    if baseline_spend < 0.0:
        raise ValueError("baseline_spend must be non-negative")
    return baseline_spend * (spend_factor(price_ratio, epsilon) - 1.0)


def static_spend_change(baseline_spend: float, price_ratio: float) -> float:
    """The main specification, named explicitly: ``q0 · Δp``.

    Identical to ``spend_change(baseline_spend, price_ratio, 0.0)``. Use this
    in headline code paths so the intent is legible at the call site.

    >>> round(static_spend_change(1000.0, 1.6), 6)
    600.0
    """
    return spend_change(baseline_spend, price_ratio, 0.0)


def consumption_reduction(price_ratio: float, epsilon: float) -> float:
    """Fraction by which physical consumption falls, as a positive number.

    ``1 - (p1/p0) ** ε``. Reported as a positive percentage in output tables.
    The prior repo's linear ``ε·r`` version exceeded 1.0 (negative
    consumption) for low deciles at the +161% scenario.

    >>> consumption_reduction(1.6, 0.0)
    0.0
    >>> round(consumption_reduction(1.6, -0.64) * 100, 1)   # matches prior repo
    26.0
    """
    return 1.0 - quantity_factor(price_ratio, epsilon)


def deadweight_share(price_ratio: float, epsilon: float) -> float:
    """Share of the static (upper-bound) loss removed by substitution.

    ``1 - spend_change(ε) / spend_change(0)``. This is the quantity to report
    when justifying the zero-elasticity headline: it is exactly how much of
    the reported loss the elasticity robustness check would shave off.

    Returns 0.0 for the main specification.

    >>> deadweight_share(1.6, 0.0)
    0.0
    >>> round(deadweight_share(1.6, -0.64), 3)
    0.693
    """
    static = spend_factor(price_ratio, 0.0) - 1.0
    if static == 0.0:
        return 0.0
    return 1.0 - (spend_factor(price_ratio, epsilon) - 1.0) / static


# --------------------------------------------------------------------------
# Money-metric welfare: bounds on the compensating variation
# --------------------------------------------------------------------------
#
# ``spend_change`` measures the change in EXPENDITURE, not in welfare. At
# eps = -0.8 it reports a household that stops heating its home as barely worse
# off, because it stopped spending the money -- the "heat or eat" fallacy this
# module's own preamble criticises. Expenditure change is the wrong object for
# a distributional welfare statement and must never be reported as the cost of
# the shock under a non-zero elasticity.
#
# The right object is the compensating variation (CV): the transfer that would
# restore pre-shock utility at post-shock prices. It is not identified without
# the full demand system, but it is tightly bracketed by two index numbers that
# need nothing beyond q0, q1 and dp:
#
#     q1 . dp  <=  CV  <=  q0 . dp
#     (Paasche)          (Laspeyres)
#
# The upper bound is exactly the paper's zero-elasticity headline. The lower
# bound values the POST-adjustment bundle at the price change, i.e. it credits
# the household only with the money it saves on units it no longer buys, while
# still charging it for everything it does buy. Both are money-metric; the
# spend change is not. See Deaton & Muellbauer (1980), ch. 7, on the
# Laspeyres/Paasche bracket for the cost-of-living index.
#
# The width of the bracket is ``1 - (p1/p0)**eps``, which is exactly
# :func:`consumption_reduction`: the maximum share of the static loss that
# demand response can remove is the share of consumption it removes. At a
# +14% price move and eps = -0.8 that is 9.9%, not the 81% that the spend
# change implies. Demand response is therefore a SMALL source of uncertainty
# once measured in welfare rather than in spending.


def laspeyres_cv(baseline_spend: float, price_ratio: float) -> float:
    """Upper bound on the compensating variation: ``q0 . dp``.

    Independent of the elasticity by construction -- this is the paper's
    zero-elasticity headline, restated as a welfare bound.

    >>> round(laspeyres_cv(1000.0, 1.14), 4)
    140.0
    """
    if baseline_spend < 0.0:
        raise ValueError("baseline_spend must be non-negative")
    _check_price_ratio(price_ratio)
    return baseline_spend * (price_ratio - 1.0)


def paasche_cv(
    baseline_spend: float, price_ratio: float, epsilon: float = 0.0
) -> float:
    """Lower bound on the compensating variation: ``q1 . dp``.

    ``q1 = q0 * (p1/p0)**eps`` is the post-adjustment quantity, so this is
    ``baseline_spend * (p1/p0)**eps * (p1/p0 - 1)``.

    At ``epsilon = 0`` it coincides with :func:`laspeyres_cv`.

    >>> round(paasche_cv(1000.0, 1.14), 4)
    140.0
    >>> round(paasche_cv(1000.0, 1.14, -0.8), 2)   # ~10% below the upper bound
    126.07
    """
    if baseline_spend < 0.0:
        raise ValueError("baseline_spend must be non-negative")
    return baseline_spend * quantity_factor(price_ratio, epsilon) * (price_ratio - 1.0)


def cv_bounds(
    baseline_spend: float, price_ratio: float, epsilon: float = 0.0
) -> tuple[float, float]:
    """``(lower, upper)`` money-metric bounds on the compensating variation.

    Use this, never :func:`spend_change`, whenever a number is going to be
    described as a cost, a loss or a burden under a non-zero elasticity.

    >>> lo, hi = cv_bounds(1000.0, 1.14, -0.8)
    >>> round(lo, 2), round(hi, 2)
    (126.07, 140.0)
    >>> lo <= hi
    True
    """
    return (
        paasche_cv(baseline_spend, price_ratio, epsilon),
        laspeyres_cv(baseline_spend, price_ratio),
    )


def welfare_shaved_share(price_ratio: float, epsilon: float) -> float:
    """Maximum share of the static loss that demand response can remove.

    ``1 - (p1/p0)**eps`` -- identically :func:`consumption_reduction`, and the
    welfare analogue of :func:`deadweight_share`. The two differ by an order of
    magnitude at plausible elasticities, and the difference is the whole of
    fix A5: ``deadweight_share`` is a statement about spending, this is a
    statement about welfare.

    >>> round(welfare_shaved_share(1.14, -0.8), 4)
    0.0995
    >>> round(deadweight_share(1.14, -0.8), 4)     # the spending measure
    0.8103
    """
    return consumption_reduction(price_ratio, epsilon)


def resolve_elasticity_spec(spec: object = "main") -> ElasticitySpec:
    """Resolve an :class:`ElasticitySpec` from a name, or pass one through.

    Names: ``"main"`` / ``"zero"`` (the paper's main specification: no
    substitution, the Deaton first-order approximation and an explicit upper
    bound on the loss), ``"labandeira_short_run"``, ``"labandeira_long_run"``,
    ``"priesmann_short_run"``, ``"priesmann_long_run"``, ``"prior_repo"``.

    Lives here rather than in a runner module so that resolving a spec never
    requires PolicyEngine or microdata. (Moved from the deleted
    ``uk_iran_conflict.runner``; see docs/FIXES.md C14.)

    >>> resolve_elasticity_spec("zero").is_main_specification
    True
    """
    if isinstance(spec, ElasticitySpec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"expected a name or an ElasticitySpec, got {type(spec)!r}")
    builders = {
        "main": ElasticitySpec.main,
        "zero": ElasticitySpec.main,
        "labandeira_short_run": lambda: ElasticitySpec.labandeira_flat("short_run"),
        "labandeira_long_run": lambda: ElasticitySpec.labandeira_flat("long_run"),
        "priesmann_short_run": lambda: ElasticitySpec.priesmann_income_varying(
            "short_run"
        ),
        "priesmann_long_run": lambda: ElasticitySpec.priesmann_income_varying(
            "long_run"
        ),
        "prior_repo": ElasticitySpec.prior_repo_replication,
    }
    if spec not in builders:
        raise KeyError(f"unknown elasticity spec {spec!r}; known: {sorted(builders)}")
    return builders[spec]()


def basket_spend_change(
    baseline_spend: Mapping[Carrier, float],
    price_ratio: Mapping[Carrier, float],
    spec: ElasticitySpec | None = None,
    decile: int | None = None,
) -> dict[Carrier, float]:
    """Per-carrier spend change for a household facing carrier-specific shocks.

    ``spec`` defaults to :meth:`ElasticitySpec.main` — the zero-elasticity
    main specification. Carrier-specific price ratios are the point: gas,
    electricity and motor fuel move by different amounts in the 2026 shock,
    and (under a non-zero spec) respond at different, oppositely-sloped
    income gradients.

    >>> {k: round(v, 6) for k, v in
    ...     basket_spend_change({"gas": 800.0}, {"gas": 1.3}).items()}
    {'gas': 240.0}
    """
    spec = spec or ElasticitySpec.main()
    out: dict[Carrier, float] = {}
    for carrier, spend in baseline_spend.items():
        ratio = price_ratio.get(carrier)
        if ratio is None:
            raise KeyError(f"no price ratio supplied for carrier {carrier!r}")
        eps = elasticity_for(spec, carrier, decile)
        out[carrier] = spend_change(spend, ratio, eps)
    return out


def apply_to_deciles(
    baseline_spend_by_decile: Sequence[float] | Mapping[int, float],
    price_ratio: float,
    spec: ElasticitySpec | None = None,
    carrier: Carrier = "gas",
) -> dict[int, float]:
    """Spend change by income decile for one carrier.

    ``spec`` defaults to the main (zero-elasticity) specification, in which
    case every decile's change is simply ``spend · (price_ratio - 1)`` and the
    only cross-decile variation comes from baseline spend.

    >>> apply_to_deciles({1: 1000.0, 2: 2000.0}, 1.5)
    {1: 500.0, 2: 1000.0}
    """
    spec = spec or ElasticitySpec.main()
    if isinstance(baseline_spend_by_decile, Mapping):
        items = dict(baseline_spend_by_decile)
    else:
        items = {i + 1: v for i, v in enumerate(baseline_spend_by_decile)}
    return {
        d: spend_change(v, price_ratio, elasticity_for(spec, carrier, d))
        for d, v in items.items()
    }

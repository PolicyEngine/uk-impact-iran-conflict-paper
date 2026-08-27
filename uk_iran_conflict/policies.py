"""The Autumn Budget 2026 policy responses, scored against the shock.

Three of the five options in the live debate **do not exist in PolicyEngine UK**
(verified against the installed release): there is no Warm Home Discount
parameter or variable at all, and the Energy Price Guarantee machinery
(``gov.ofgem.energy_price_guarantee`` feeding ``monthly_epg_consumption_level``)
is a flat cap-ratio scaler with no means test and no consumption-block
structure, so it cannot express either a social tariff or the JRF two-tier
block. They are therefore computed here, transparently, from PolicyEngine's own
eligibility and household variables rather than faked as parameter reforms.

Each policy returns a per-household **gain** in £/yr, to be set against the
per-household loss from :mod:`uk_iran_conflict.incidence`. The paper's scoring
metrics follow: cost per £ of bottom-decile gain, share of spend reaching
deciles 1-3, and the share of losers left uncompensated within each decile.

Costings quoted in each docstring are the sponsors' own published figures, so
the simulated cost can be compared against them as a validation check.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from uk_iran_conflict.incidence import (
    DOMESTIC_FUEL_VAT_RATE,
    Baseline,
    ShockCost,
    wmean,
    wquantile,
    wsum,
)

#: Households on a means-tested benefit are the social tariff's target
#: population. The anchor fact in the policy debate is that roughly 40% of
#: households struggling to heat their home are *not* in this group (JRF,
#: 9 Apr 2026), which is what the uncompensated-loser metric quantifies.
#:
#: Split into two tiers because the previous single tuple was resolved inside
#: a bare ``except Exception: continue`` (docs/FIXES.md C13). A variable
#: renamed upstream therefore *silently shrank the modelled means-tested
#: population* — which is exactly the symptom the paper reports as a finding
#: (15.7% of households against a 7.2m UC caseload, docs/VALIDATION.md Check
#: 4). A defect and its own headline finding must not be indistinguishable.
MEANS_TESTED_REQUIRED: tuple[str, ...] = (
    "universal_credit",
    "pension_credit",
    "housing_benefit",
    "esa_income",
)
"""Benefits that **must** resolve. Absence is a hard error, never a shrug."""

MEANS_TESTED_LEGACY: tuple[str, ...] = (
    "child_tax_credit",
    "working_tax_credit",
    "income_support",
    "jsa_income",
)
"""Benefits genuinely being wound down (Income Support's final DWP release was
Nov 2025; tax credits closed to new claims). These may legitimately be absent
from a given policyengine-uk release, so absence is **recorded in the audit**
rather than tolerated in silence."""

#: Backwards-compatible union, in the original order.
MEANS_TESTED_VARIABLES: tuple[str, ...] = (
    "universal_credit",
    "pension_credit",
    "child_tax_credit",
    "working_tax_credit",
    "housing_benefit",
    "income_support",
    "jsa_income",
    "esa_income",
)

#: PolicyEngine person-level indicator summed to the household to get a real
#: child count (docs/FIXES.md B8). Tried in order; the first that resolves
#: wins, so a rename upstream degrades to the next candidate rather than to a
#: wrong answer.
CHILD_COUNT_CANDIDATES: tuple[str, ...] = (
    "household_count_children",
    "num_children",
    "is_child",
)

#: Populated as a side effect of :func:`means_tested_flag`, which already opens
#: the microsimulation the count has to come from. Keyed by household count so
#: a stale array can never be silently applied to a different baseline.
_CHILD_COUNTS: dict[int, np.ndarray] = {}


def _resolve(sim: Any, name: str, period: int) -> np.ndarray | None:
    """Household-mapped values for ``name``, or ``None`` if it does not exist.

    Existence is checked against the tax-benefit system's variable registry
    rather than inferred from a caught exception, so a genuine failure inside
    ``calculate`` (a broken formula, a missing input) propagates instead of
    being mistaken for a missing variable.
    """
    variables = getattr(getattr(sim, "tax_benefit_system", None), "variables", None)
    if variables is not None and name not in variables:
        return None
    return np.asarray(sim.calculate(name, period, map_to="household"))


def load_household_children(dataset: str, period: int = 2026) -> np.ndarray:
    """Real child count per household, from PolicyEngine UK.

    ``np.clip(base.people - 2, 0, None)`` — the expression this replaces — is
    household size minus two. It gives a lone parent with one child nothing and
    a three-adult household a child allowance (docs/FIXES.md B8).
    """
    from policyengine_uk import Microsimulation  # noqa: PLC0415

    sim = Microsimulation(dataset=dataset)
    return _household_children(sim, period)


def _household_children(sim: Any, period: int) -> np.ndarray:
    tried: list[str] = []
    for name in CHILD_COUNT_CANDIDATES:
        tried.append(name)
        got = _resolve(sim, name, period)
        if got is None:
            continue
        children = np.asarray(got, dtype=float)
        _CHILD_COUNTS[len(children)] = children
        return children
    raise RuntimeError(
        "no child-count variable resolved in policyengine-uk; tried "
        f"{tried}. Refusing to fall back on household size, which is the "
        "defect docs/FIXES.md B8 exists to remove."
    )


def child_counts(base: Baseline) -> np.ndarray:
    """Child count aligned to ``base``, from the microdata — never from size.

    Resolution order:

    1. a ``children`` (or ``count_children``) field on the ``Baseline``, if the
       sibling incidence module has added one;
    2. the array cached by :func:`means_tested_flag` /
       :func:`load_household_children` for a baseline of this length.

    Raises if neither is available, rather than reinstating the size proxy.
    """
    for attr in ("children", "count_children", "child_count"):
        got = getattr(base, attr, None)
        if got is not None:
            arr = np.asarray(got, dtype=float)
            if len(arr) == base.n:
                return arr
    cached = _CHILD_COUNTS.get(base.n)
    if cached is not None:
        return np.asarray(cached, dtype=float)
    raise RuntimeError(
        "no child count available for this baseline. Call "
        "policies.means_tested_flag(dataset, period) or "
        "policies.load_household_children(dataset, period) first — both cache "
        "the count off the same microsimulation. Household size minus two is "
        "not an acceptable substitute (docs/FIXES.md B8)."
    )


def means_tested_audit(dataset: str, period: int = 2026) -> dict[str, Any]:
    """Resolve the means-tested variable set and count who each one reaches.

    Written to ``results/`` so the paper's 15.7% coverage claim is checkable
    against the DWP caseloads in ``docs/VALIDATION.md`` Check 4 *variable by
    variable*, instead of resting on a set the reader cannot see.
    """
    from policyengine_uk import Microsimulation  # noqa: PLC0415

    sim = Microsimulation(dataset=dataset)
    weight = np.asarray(sim.calculate("household_weight", period))

    resolved: dict[str, dict[str, float]] = {}
    missing_required: list[str] = []
    missing_legacy: list[str] = []
    flag: np.ndarray | None = None

    for tier, names in (
        ("required", MEANS_TESTED_REQUIRED),
        ("legacy", MEANS_TESTED_LEGACY),
    ):
        for name in names:
            got = _resolve(sim, name, period)
            if got is None:
                (missing_required if tier == "required" else missing_legacy).append(
                    name
                )
                continue
            receiving = np.asarray(got) > 0
            resolved[name] = {
                "tier": tier,
                "households_m": float(weight[receiving].sum() / 1e6),
                "share_of_households": float(weight[receiving].sum() / weight.sum()),
                "mean_amount_gbp_if_receiving": float(
                    np.asarray(got)[receiving].dot(weight[receiving])
                    / weight[receiving].sum()
                )
                if weight[receiving].sum() > 0
                else float("nan"),
            }
            flag = receiving if flag is None else (flag | receiving)

    if missing_required:
        raise RuntimeError(
            "means-tested benefit variables missing from policyengine-uk: "
            f"{missing_required}. These are required; a silently smaller "
            "means-tested population would be indistinguishable from the "
            "paper's own coverage finding (docs/FIXES.md C13)."
        )
    assert flag is not None

    _household_children(sim, period)  # populate the child-count cache

    return {
        "period": period,
        "resolved_variables": sorted(resolved),
        "missing_required": missing_required,
        "missing_legacy": missing_legacy,
        "by_variable": resolved,
        "any_means_tested_households_m": float(weight[flag].sum() / 1e6),
        "any_means_tested_share": float(weight[flag].sum() / weight.sum()),
        "total_households_m": float(weight.sum() / 1e6),
        "note": (
            "Union over the resolved variables, annual receipt, household "
            "level. Compare against DWP administrative caseloads "
            "(docs/VALIDATION.md Check 4): UC alone is 7.2m GB households at "
            "May 2026. A large shortfall here is a PolicyEngine UK "
            "data-quality finding, not a modelling choice — but it is only a "
            "finding if the variable set actually resolved, which is what "
            "this file records."
        ),
        "flag": flag,
    }


def write_means_tested_audit(
    dataset: str, period: int = 2026, path: str | Path | None = None
) -> Path:
    """Run :func:`means_tested_audit` and persist it as JSON."""
    audit = means_tested_audit(dataset, period)
    audit = {k: v for k, v in audit.items() if k != "flag"}
    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "results" / "means_tested_audit.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2))
    return path


def means_tested_flag(
    dataset: str, period: int = 2026, audit_path: str | Path | None = None
) -> np.ndarray:
    """Household-level indicator of any means-tested benefit receipt.

    **Fails loudly** if any of :data:`MEANS_TESTED_REQUIRED` is missing. Also
    caches the household child count (:func:`child_counts`) and, when
    ``audit_path`` is given, writes the resolved variable set and each
    variable's household count for the reader to check.
    """
    audit = means_tested_audit(dataset, period)
    flag = audit.pop("flag")
    if audit_path is not None:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2))
    return np.asarray(flag)


@dataclass(frozen=True)
class Policy:
    """A costed policy response, with the parameter space it actually lives in.

    Every instrument here is homogeneous of degree one in a single generosity
    parameter, so scaling its gain array by a scalar is exactly equivalent to
    scaling that parameter. What the pre-revision code did **not** record is
    that the parameter has a range: a social tariff cannot discount more than
    100% of a bill, and zero-rating cannot take VAT below zero. Scoring all five
    instruments at a common £5bn envelope by scalar multiplication therefore
    implied a 138% social-tariff discount, a £1,083 Warm Home Discount and a
    negative VAT rate, and the "saturation" finding that followed was mechanical
    — households were being paid more than they lost. All three round-2 referees
    found it.

    Attributes
    ----------
    parameter, parameter_units:
        Name and units of the generosity parameter the gain is linear in, so
        every scaled row can report the parameter it implies.
    stated_parameter:
        The sponsor's own value of it.
    feasible_max:
        Largest value of the parameter that is *inside the instrument's own
        parameter space*. Either a float, or a callable taking
        ``(base, cost, mt)`` for a ceiling that has to be read off the data.
    generic_template:
        How to name a row whose implied parameter is outside that space. Such a
        row is not the sponsor's instrument any more and must not carry its
        name; it is described by what it does (``"proportional domestic-bill
        subsidy at 12.4 per cent"``). Formatted with ``p`` = implied parameter.
    means_tested:
        Whether eligibility is the binding margin. For these instruments the
        envelope can also be absorbed by widening *who* is eligible at the
        sponsor's own generosity, which is the margin a real policymaker would
        use and the one the paper's own coverage finding says matters — see
        :func:`score_policy_by_eligibility`.
    """

    key: str
    label: str
    source: str
    stated_cost_bn: float
    gain: Callable[[Baseline, ShockCost, np.ndarray], np.ndarray]
    parameter: str = "generosity"
    parameter_units: str = ""
    stated_parameter: float = 1.0
    feasible_max: float | Callable[[Baseline, ShockCost, np.ndarray], float] = float(
        "inf"
    )
    generic_template: str = "generic instrument scaled to {p:.4g}"
    means_tested: bool = False

    def feasible_max_parameter(
        self, base: Baseline, cost: ShockCost, mt: np.ndarray
    ) -> float:
        """The instrument's parameter ceiling, resolved against the data."""
        if callable(self.feasible_max):
            return float(self.feasible_max(base, cost, mt))
        return float(self.feasible_max)

    def describe(self, implied_parameter: float, feasible: bool) -> str:
        """Label a row: the real policy name only while it stays feasible."""
        if feasible:
            return self.label
        return self.generic_template.format(p=implied_parameter)


def _social_tariff(
    base: Baseline, cost: ShockCost, mt: np.ndarray, discount: float = 0.35
) -> np.ndarray:
    """Means-tested discount on the domestic energy bill.

    The option Reeves has signalled: income-based support rather than universal.
    Modelled as a ``discount`` proportion off the *shocked* domestic bill for
    households on a means-tested benefit. Its weakness is definitional, not
    computational — it reaches nobody outside the means-tested population.
    """
    shocked_bill = base.energy + cost.domestic
    return np.where(mt, discount * shocked_bill, 0.0)


def _jrf_block(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    block_share: float = 0.50,
    discount: float = 0.50,
    per_child: float = 60.0,
    revalue_at_post_shock_prices: bool = False,
) -> np.ndarray:
    """JRF universal discounted block: a cheaper first tranche for everyone.

    JRF (9 Apr 2026): a discounted rate on the first 50% of typical
    consumption plus a per-child allowance, ~£5bn, fully offsetting the cost
    rise for deciles 1-3. Universal by design — that is the point, since it
    reaches the ~40% of struggling households outside the means-tested system.

    **Specification (docs/FIXES.md B7).** JRF propose a discounted *rate* on
    the block: the household pays ``(1 - discount)`` times the going rate for
    every unit inside the block. The subsidy is therefore a discount on the
    **level** of the covered spend, exactly as :func:`_social_tariff` is a
    discount on the level of the whole bill —

        gain = discount x (covered block, valued at post-shock prices)

    The previous implementation discounted only the *shock component*
    (``discount * covered * shock_rate``), i.e. a rebate of part of the price
    *increase* on the block. That is a different and far cheaper instrument:
    at a +11% domestic shock it costs roughly a ninth of the level subsidy,
    which is why the modelled cost came out at £1.9bn against JRF's £5bn. The
    paper attributed that gap to JRF being more generous. It was a
    mis-specification, not a calibration difference.

    Typical consumption is the weighted median domestic bill, so the block is a
    fixed quantity rather than a proportion of each household's own bill (a
    proportional discount would mechanically pay most to the biggest users).
    Households whose whole bill is smaller than the block get their whole bill
    discounted and no more.

    ``per_child`` uses the microdata child count (:func:`child_counts`), not
    household size minus two (docs/FIXES.md B8).

    ``revalue_at_post_shock_prices`` (round-2 referee 3, and it defaults to
    **False**)
    ---------------------------------------------------------------------
    The previous implementation multiplied the covered block by
    ``1 + cost.domestic / base.energy`` — an undocumented revaluation of the
    block at *post-shock* prices — and that revaluation is the whole of the
    difference between the modelled cost and the "more generous than JRF's £5bn"
    correction the paper draws from it. It is wrong twice over:

    * JRF's £5bn is benchmarked to the **April 2026 price cap**, i.e. to
      pre-shock levels. Valuing our block at post-shock levels compares a
      shocked costing with an unshocked one and reads the difference as
      generosity.
    * The instrument is a discounted rate on a *fixed quantity* block. The
      quantity does not move with the price, so the subsidy per unit is a
      discount on the going rate — but the "going rate" the sponsor pegs to is
      the cap they costed against, not the cap after a shock that the
      instrument exists to offset. Revaluing makes the instrument mechanically
      more generous the worse the shock is, which is a policy design nobody
      proposed.

    So the block is valued at baseline prices by default. The flag is retained,
    documented, so the previous number can be reproduced and the £-difference
    reported rather than merely asserted.
    """
    from uk_iran_conflict.incidence import wquantile  # noqa: PLC0415

    typical = wquantile(base.energy, base.weight, 0.5)
    block_value = block_share * typical
    covered = np.minimum(base.energy, block_value)
    if revalue_at_post_shock_prices:
        covered = covered * (1.0 + _shock_rate(base, cost))
    return discount * covered + per_child * child_counts(base)


def _shock_rate(base: Baseline, cost: ShockCost) -> np.ndarray:
    """Proportional domestic price rise implied by the shock, per household."""
    return np.divide(
        cost.domestic,
        np.clip(base.energy, 1, None),
        out=np.zeros_like(cost.domestic),
        where=base.energy > 0,
    )


def _whd_expansion(
    base: Baseline, cost: ShockCost, mt: np.ndarray, amount: float = 150.0
) -> np.ndarray:
    """Warm Home Discount at £150, automatic for the means-tested population.

    Qualifying date 23 Aug 2026; property-cost test scrapped, +2.7m households.
    **Not in PolicyEngine UK** — no parameter, no variable — so it is applied
    here as a flat payment to the means-tested population.
    """
    return np.where(mt, amount, 0.0)


def _vat_zero(base: Baseline, cost: ShockCost, mt: np.ndarray) -> np.ndarray:
    """Zero-rate VAT on domestic fuel and power (5% -> 0%).

    LCFS domestic energy spend is VAT-inclusive, so the gain is the VAT
    component of the *shocked* bill: ``bill * (1 - 1/1.05)``. Strictly
    regressive in cash terms — it pays most to the largest bills — which is the
    point of scoring it.
    """
    shocked_bill = base.energy + cost.domestic
    return shocked_bill * (1 - 1 / (1 + DOMESTIC_FUEL_VAT_RATE))


def _ippr_rebate(
    base: Baseline, cost: ShockCost, mt: np.ndarray, amount: float = 183.0
) -> np.ndarray:
    """IPPR: claw back network-company windfalls into a flat £183 rebate."""
    return np.full_like(base.energy, amount, dtype=float)


def _mean_eligible_domestic_bill(
    base: Baseline, cost: ShockCost, mt: np.ndarray
) -> float:
    """Mean shocked domestic bill of the means-tested population, £/yr.

    The ceiling used for the two flat-payment instruments. A payment larger than
    the recipient's own energy bill is no longer energy-bill support of any
    kind: it is unconditional cash, scored against a loss it exceeds. There is
    no statutory maximum Warm Home Discount to appeal to, so this is the
    instrument-internal ceiling — the point beyond which the thing being scored
    stops being the thing named — and it is read off the data rather than
    invented. It is stated in the results as ``feasible_max_parameter`` so a
    reader who prefers a different ceiling can see exactly what was imposed.
    """
    shocked = base.energy + cost.domestic
    sel = np.asarray(mt, dtype=bool)
    w = base.weight[sel]
    return float(wmean(shocked[sel], w)) if w.sum() > 0 else float("nan")


POLICIES: dict[str, Policy] = {
    "social_tariff": Policy(
        "social_tariff",
        "Means-tested social tariff (35% bill discount)",
        "HM Treasury signal, 2026; modelled here, as no such "
        "instrument exists in PolicyEngine UK",
        stated_cost_bn=float("nan"),
        gain=_social_tariff,
        parameter="bill discount",
        parameter_units="per cent of the shocked domestic bill",
        stated_parameter=35.0,
        # A discount cannot exceed the bill: 100% is free energy, and anything
        # above it is a payment for consuming, not a discount.
        feasible_max=100.0,
        generic_template=(
            "proportional domestic-bill subsidy at {p:.1f} per cent, "
            "means-tested population"
        ),
        means_tested=True,
    ),
    "jrf_block": Policy(
        "jrf_block",
        "JRF universal discounted block (50% of typical use, + per-child)",
        "Joseph Rowntree Foundation, 9 Apr 2026",
        stated_cost_bn=5.0,
        gain=_jrf_block,
        parameter="block discount",
        parameter_units="per cent off the first 50% of typical use",
        stated_parameter=50.0,
        # The block can at most be free; the per-child allowance scales with it.
        feasible_max=100.0,
        generic_template=(
            "discounted first block at {p:.1f} per cent off 50 per cent of "
            "typical use, universal"
        ),
    ),
    "whd_expansion": Policy(
        "whd_expansion",
        "Warm Home Discount expansion (£150)",
        "DESNZ, qualifying date 23 Aug 2026",
        stated_cost_bn=float("nan"),
        gain=_whd_expansion,
        parameter="payment",
        parameter_units="£ per eligible household per year",
        stated_parameter=150.0,
        feasible_max=_mean_eligible_domestic_bill,
        generic_template=(
            "flat payment of £{p:.0f} a year to the means-tested population"
        ),
        means_tested=True,
    ),
    "vat_zero": Policy(
        "vat_zero",
        "Zero-rate VAT on domestic fuel (5% -> 0%)",
        "HMRC reduced rate; long-standing proposal",
        stated_cost_bn=float("nan"),
        gain=_vat_zero,
        parameter="VAT points removed",
        parameter_units="percentage points of VAT on domestic fuel and power",
        stated_parameter=5.0,
        # **An arithmetic ceiling, not a judgement.** The reduced rate is 5%;
        # removing more than five points is a negative VAT rate. The paper
        # already says this in prose while its own scorecard crowned the
        # instrument at x2.47 of it.
        feasible_max=5.0,
        generic_template=(
            "proportional domestic-bill subsidy at {p:.1f} per cent, universal"
        ),
    ),
    "ippr_rebate": Policy(
        "ippr_rebate",
        "IPPR network-windfall rebate (£183 flat)",
        "IPPR, Apr 2026",
        stated_cost_bn=float("nan"),
        gain=_ippr_rebate,
        parameter="rebate",
        parameter_units="£ per household per year",
        stated_parameter=183.0,
        # Funded by clawing back network-company windfalls, so the ceiling is
        # the windfall. IPPR's own £183 x 29.5m households is about £5.4bn, so
        # the instrument already spends its funding source; there is no headroom
        # above it and the common envelope scales it *down*, never up.
        feasible_max=183.0,
        generic_template="flat universal payment of £{p:.0f} a year",
    ),
}


#: Common exchequer envelope, £bn, at which every instrument is *also*
#: scored (docs/FIXES.md B7). The sponsors' own costings differ by a factor of
#: four, so scoring only at stated cost compares a £1.3bn instrument with a
#: £5.4bn one and then draws a conclusion about *targeting* from the
#: difference. Targeting is a statement about the shape of the spend, and the
#: shape is only comparable at a common size.
#:
#: £5bn is JRF's published costing — the only sponsor figure among the five
#: that is both stated and independently documented — so the common-envelope
#: column is anchored on a real number rather than a round invention.
COMMON_ENVELOPE_BN: float = 5.0


@dataclass
class PolicyScore:
    """The paper's scorecard for one policy against one scenario."""

    policy: str
    label: str
    cost_bn: float
    stated_cost_bn: float
    share_to_bottom_three: float
    cost_per_pound_decile_one: float
    mean_gain_gbp: float
    uncompensated_share_overall: float
    uncompensated_by_decile: dict[int, float]
    net_loss_after_policy_gbp: float
    fully_compensated_share: float
    # --- continuous compensation measures (docs/FIXES.md B9) --------------
    # "Share of losers uncompensated" is knife-edge: a household short by £1
    # counts identically to one short by £900, and it is applied to a loss
    # that is itself an upper bound. 84% against 88% therefore carries almost
    # no information, and it produced the claim that VAT zero-rating
    # "compensates nobody" while that instrument delivers the second-highest
    # mean gain and a LOWER mean residual loss than the social tariff. These
    # measures are continuous in the shortfall, so they cannot do that.
    share_of_aggregate_loss_offset: float = float("nan")
    mean_residual_loss_gbp: float = float("nan")
    median_residual_loss_gbp: float = float("nan")
    mean_residual_loss_by_decile: dict[int, float] = field(default_factory=dict)
    median_residual_loss_by_decile: dict[int, float] = field(default_factory=dict)
    share_of_loss_offset_by_decile: dict[int, float] = field(default_factory=dict)
    mean_gain_by_decile: dict[int, float] = field(default_factory=dict)
    #: Envelope this score was computed at: "stated" (the sponsor's own
    #: parameters) or "common" (rescaled to :data:`COMMON_ENVELOPE_BN`).
    envelope: str = "stated"
    envelope_bn: float = float("nan")
    #: Factor the sponsor's own generosity was multiplied by to hit the common
    #: envelope. 1.0 for a stated-cost score.
    envelope_scale: float = 1.0
    # --- feasible parameter space (round-2 finding 2) --------------------
    #: Name and units of the generosity parameter the scaling acts on.
    parameter: str = ""
    parameter_units: str = ""
    #: The sponsor's own value of that parameter.
    stated_parameter: float = float("nan")
    #: The value this row's scaling implies. Recorded for EVERY scaled row, so
    #: a 138% bill discount or a negative VAT rate is visible in the results
    #: instead of hiding inside a scale factor.
    implied_parameter: float = float("nan")
    #: Largest value of the parameter inside the instrument's own parameter
    #: space; see :meth:`Policy.feasible_max_parameter`.
    feasible_max_parameter: float = float("nan")
    #: Whether ``implied_parameter`` is inside that space.
    is_feasible: bool = True
    #: Exchequer cost this instrument can actually absorb at
    #: ``feasible_max_parameter``. Below the common envelope for an instrument
    #: that saturates, which is the honest form of the saturation finding.
    absorbable_envelope_bn: float = float("nan")
    #: What this row should be called. The real policy name only while the row
    #: is feasible; otherwise a description of what it actually does.
    label_used: str = ""
    #: Share of households eligible, for the eligibility-widening rows.
    eligible_share: float = float("nan")
    #: Units string for :attr:`cost_per_pound_decile_one`.
    cost_per_pound_decile_one_units: str = ""


def _residual_loss(loss: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Loss still borne after the policy, floored at zero.

    Floored because a household paid more than it lost has not made a welfare
    *gain* from the shock; carrying the negative through would let
    over-compensation at the top net off under-compensation at the bottom and
    flatter every universal instrument.
    """
    return np.clip(loss - gain, 0.0, None)


def score_policy(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    gain: np.ndarray | None = None,
    envelope: str = "stated",
    envelope_bn: float = float("nan"),
    envelope_scale: float = 1.0,
    *,
    implied_parameter: float | None = None,
    feasible_max_parameter: float = float("nan"),
    is_feasible: bool = True,
    absorbable_envelope_bn: float = float("nan"),
    label_used: str | None = None,
    eligible_share: float = float("nan"),
) -> tuple[PolicyScore, np.ndarray]:
    """Score ``policy`` against the shock, returning the gain array too.

    ``uncompensated`` counts households still worse off after the policy — the
    metric decile averages hide and the one the targeting argument turns on.
    It is reported alongside, never instead of, the continuous measures on
    :class:`PolicyScore` (docs/FIXES.md B9).

    ``gain`` may be supplied pre-computed (used by
    :func:`score_policy_at_envelope`); otherwise the policy computes its own.
    """
    if gain is None:
        gain = policy.gain(base, cost, mt)
    if implied_parameter is None:
        implied_parameter = policy.stated_parameter * envelope_scale
    if label_used is None:
        label_used = policy.describe(implied_parameter, is_feasible)
    w = base.weight
    loss = cost.total
    net = loss - gain
    residual = _residual_loss(loss, gain)
    losers = loss > 0

    total_gain = wsum(gain, w)
    total_loss = wsum(loss, w)
    # Round-2 referee 3: ``decile <= 3`` has no lower guard, so it swept in the
    # ~0.24m households carrying an out-of-range decile — the ones that are
    # 100% non-positive income and that ``incidence.decile_table`` correctly
    # excludes — and credited their gain to the bottom three deciles.
    bottom3 = (base.decile >= 1) & (base.decile <= 3)
    d1 = base.decile == 1
    d1_gain_bn = wsum(gain[d1], w[d1]) / 1e9

    by_decile: dict[int, float] = {}
    mean_residual_d: dict[int, float] = {}
    median_residual_d: dict[int, float] = {}
    offset_d: dict[int, float] = {}
    mean_gain_d: dict[int, float] = {}
    for d in range(1, 11):
        sel = base.decile == d
        wd_all = w[sel]
        if wd_all.sum() <= 0:
            continue
        mean_residual_d[d] = wmean(residual[sel], wd_all)
        median_residual_d[d] = wquantile(residual[sel], wd_all, 0.5)
        mean_gain_d[d] = wmean(gain[sel], wd_all)
        loss_d = wsum(loss[sel], wd_all)
        # Round-2 referees: this used to cap the gain at the decile AGGREGATE
        # loss, while the headline ``share_of_aggregate_loss_offset`` caps at
        # each HOUSEHOLD's own loss. The two are different statistics, and the
        # decile version reported deciles as 100% offset next to a £225 mean
        # residual loss in the very same row — because within the decile the
        # over-compensated households were paying for the under-compensated
        # ones. Unified on the household-level definition, which is the one the
        # headline uses and the only one consistent with a positive residual.
        offset_d[d] = (
            float(wsum(np.minimum(gain[sel], loss[sel]), wd_all) / loss_d)
            if loss_d > 0
            else float("nan")
        )
        sel_losers = sel & losers
        wd = w[sel_losers]
        if wd.sum() > 0:
            by_decile[d] = float(wd[net[sel_losers] > 0].sum() / wd.sum())

    wl = w[losers]
    return (
        PolicyScore(
            policy=policy.key,
            label=policy.label,
            cost_bn=total_gain / 1e9,
            stated_cost_bn=policy.stated_cost_bn,
            share_to_bottom_three=(
                wsum(gain[bottom3], w[bottom3]) / total_gain
                if total_gain
                else float("nan")
            ),
            # Round-2 referee 1: this was £bn of total cost divided by the
            # MEAN £/household gain in decile one, and printed as pounds. £bn
            # over £ is not pounds and is not anything else either; the ranking
            # survived because the denominator is monotone in the same thing,
            # but every quoted level was meaningless. It is now total exchequer
            # cost per £1 actually delivered to decile one — both sides in £bn,
            # so the ratio is dimensionless and never below 1.
            cost_per_pound_decile_one=(
                (total_gain / 1e9) / d1_gain_bn if d1_gain_bn > 0 else float("nan")
            ),
            cost_per_pound_decile_one_units=(
                "£ of total exchequer cost per £1 of gain reaching decile one "
                "(dimensionless, >= 1)"
            ),
            mean_gain_gbp=wmean(gain, w),
            uncompensated_share_overall=(
                float(wl[net[losers] > 0].sum() / wl.sum())
                if wl.sum() > 0
                else float("nan")
            ),
            uncompensated_by_decile=by_decile,
            net_loss_after_policy_gbp=wmean(net, w),
            fully_compensated_share=(
                float(wl[net[losers] <= 0].sum() / wl.sum())
                if wl.sum() > 0
                else float("nan")
            ),
            # The aggregate loss actually offset: gain is credited only up to
            # each household's own loss, so a pound paid to a household with
            # nothing left to lose does not count as compensation.
            share_of_aggregate_loss_offset=(
                float(wsum(np.minimum(gain, loss), w) / total_loss)
                if total_loss > 0
                else float("nan")
            ),
            mean_residual_loss_gbp=wmean(residual, w),
            median_residual_loss_gbp=wquantile(residual, w, 0.5),
            mean_residual_loss_by_decile=mean_residual_d,
            median_residual_loss_by_decile=median_residual_d,
            share_of_loss_offset_by_decile=offset_d,
            mean_gain_by_decile=mean_gain_d,
            envelope=envelope,
            envelope_bn=envelope_bn,
            envelope_scale=envelope_scale,
            parameter=policy.parameter,
            parameter_units=policy.parameter_units,
            stated_parameter=policy.stated_parameter,
            implied_parameter=implied_parameter,
            feasible_max_parameter=feasible_max_parameter,
            is_feasible=is_feasible,
            absorbable_envelope_bn=absorbable_envelope_bn,
            label_used=label_used,
            eligible_share=eligible_share,
        ),
        gain,
    )


def _envelope_scale(
    base: Baseline, gain: np.ndarray, envelope_bn: float, key: str
) -> float:
    total = wsum(gain, base.weight)
    if total <= 0:
        raise ValueError(f"policy {key!r} has non-positive cost; cannot rescale")
    return envelope_bn * 1e9 / total


def score_policy_at_envelope(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    *,
    cap_at_feasible_max: bool = True,
) -> tuple[PolicyScore, np.ndarray]:
    """Score ``policy`` at a common exchequer envelope, inside its own parameter space.

    Two rows come out of this function depending on ``cap_at_feasible_max``, and
    ``scorecard`` emits both.

    ``cap_at_feasible_max=True`` (the ``"common_capped"`` row)
        The generosity parameter is scaled toward the envelope but **stopped at
        the instrument's feasible maximum**. If the instrument saturates before
        the envelope is spent, the row reports what it *can* absorb
        (:attr:`PolicyScore.absorbable_envelope_bn`) rather than pretending to
        spend money it has no way to spend. This is the honest form of the
        paper's saturation finding: some instruments cannot reach £5bn at all,
        which is a far stronger statement than that they stop helping once they
        do.

    ``cap_at_feasible_max=False`` (the ``"common_scaled"`` row)
        The previous behaviour — pure scalar multiplication — kept so the
        published numbers remain auditable, but with the implied parameter
        recorded and the row **renamed**. At a £5bn envelope the scaling
        implied a 138% social-tariff discount (x3.95), a £1,083 Warm Home
        Discount (x7.22) and a negative VAT rate (x2.47). A row like that is not
        the Warm Home Discount and must not be called it; the paper crowning
        "VAT zero-rating" the winner at x2.47 while separately and correctly
        stating that the 5% rate is an arithmetic ceiling is precisely the
        contradiction three referees caught. Infeasible rows now carry
        :attr:`Policy.generic_template` — "proportional domestic-bill subsidy at
        12.4 per cent" — instead of a real policy's name.

    Scaling the gain array is exactly equivalent to scaling the parameter,
    because every instrument here is homogeneous of degree one in it; what was
    missing was never the algebra but the parameter's range.
    """
    gain = policy.gain(base, cost, mt)
    scale = _envelope_scale(base, gain, envelope_bn, policy.key)
    ceiling = policy.feasible_max_parameter(base, cost, mt)
    stated = policy.stated_parameter
    max_scale = ceiling / stated if stated > 0 else float("inf")
    absorbable_bn = envelope_bn * (min(1.0, max_scale / scale) if scale > 0 else 1.0)

    if cap_at_feasible_max:
        scale = min(scale, max_scale)
    implied = stated * scale
    feasible = implied <= ceiling * (1 + 1e-12)

    return score_policy(
        base,
        cost,
        mt,
        policy,
        gain=gain * scale,
        envelope="common_capped" if cap_at_feasible_max else "common_scaled",
        envelope_bn=envelope_bn,
        envelope_scale=scale,
        implied_parameter=implied,
        feasible_max_parameter=ceiling,
        is_feasible=feasible,
        absorbable_envelope_bn=absorbable_bn,
    )


def widen_eligibility(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float,
) -> np.ndarray:
    """Extend eligibility down the income ranking until the envelope is spent.

    The margin a real policymaker uses. A social tariff that has to spend more
    money does not discount 138% of the bill; it lets more households in — and
    the paper's own coverage finding (PolicyEngine UK captures 3.0m of a 7.2m
    Universal Credit caseload, so the modelled means-tested population is 15.7%
    of households against a plausible 30%+) says that eligibility, not
    generosity, is where the action is. Widening at the sponsor's own parameter
    is therefore the *feasible* way to reach a common envelope for a
    means-tested instrument, and it is the comparison the scaled row was
    standing in for.

    Households outside the modelled means-tested population are added in
    ascending order of equivalised AHC income until the next household would
    take the total past ``envelope_bn``. Returns a new eligibility mask.
    """
    already = np.asarray(mt, dtype=bool)
    everyone = np.ones_like(already, dtype=bool)
    gain_if_eligible = policy.gain(base, cost, everyone)
    w = base.weight
    spent = wsum(np.where(already, gain_if_eligible, 0.0), w)
    remaining = envelope_bn * 1e9 - spent
    out = already.copy()
    if remaining <= 0:
        return out
    candidates = np.flatnonzero(~already)
    order = candidates[np.argsort(base.equiv_income_ahc[candidates], kind="stable")]
    running = np.cumsum(gain_if_eligible[order] * w[order])
    take = order[running <= remaining]
    out[take] = True
    return out


def score_policy_by_eligibility(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float = COMMON_ENVELOPE_BN,
) -> tuple[PolicyScore, np.ndarray]:
    """Reach the common envelope by widening eligibility, not by scaling generosity.

    Only defined for :attr:`Policy.means_tested` instruments; the universal ones
    have no eligibility margin to widen. The generosity parameter is left at the
    sponsor's own value, so the row is always feasible and keeps the policy's
    real name.
    """
    if not policy.means_tested:
        raise ValueError(
            f"policy {policy.key!r} is universal; it has no eligibility margin"
        )
    widened = widen_eligibility(base, cost, mt, policy, envelope_bn)
    gain = policy.gain(base, cost, widened)
    w = base.weight
    return score_policy(
        base,
        cost,
        mt,
        policy,
        gain=gain,
        envelope="common_eligibility",
        envelope_bn=envelope_bn,
        envelope_scale=1.0,
        implied_parameter=policy.stated_parameter,
        feasible_max_parameter=policy.feasible_max_parameter(base, cost, mt),
        is_feasible=True,
        absorbable_envelope_bn=wsum(gain, w) / 1e9,
        eligible_share=float(w[widened].sum() / w.sum()),
    )


def scorecard(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    policies: dict[str, Policy] | None = None,
) -> list[PolicyScore]:
    """Every instrument at its stated cost and at a common envelope, three ways.

    Per policy, in order:

    ``stated``
        The sponsor's own parameters.
    ``common_capped``
        Scaled toward ``envelope_bn`` but stopped at the instrument's feasible
        maximum, reporting the envelope it can actually absorb.
    ``common_scaled``
        The pure scalar rescaling, retained for auditability, with the implied
        parameter recorded and the row renamed generically when that parameter
        is outside the instrument's parameter space.
    ``common_eligibility`` (means-tested instruments only)
        The envelope reached by widening *who* is eligible at the sponsor's own
        generosity.

    Consumers should read ``envelope``, ``is_feasible`` and ``label_used``
    together: a conclusion about targeting drawn from a ``common_scaled`` row
    with ``is_feasible == False`` is a conclusion about a policy that cannot
    exist.
    """
    chosen = POLICIES if policies is None else policies
    out: list[PolicyScore] = []
    for policy in chosen.values():
        ceiling = policy.feasible_max_parameter(base, cost, mt)
        out.append(
            score_policy(
                base,
                cost,
                mt,
                policy,
                implied_parameter=policy.stated_parameter,
                feasible_max_parameter=ceiling,
                is_feasible=policy.stated_parameter <= ceiling * (1 + 1e-12),
            )[0]
        )
        out.append(
            score_policy_at_envelope(
                base, cost, mt, policy, envelope_bn, cap_at_feasible_max=True
            )[0]
        )
        out.append(
            score_policy_at_envelope(
                base, cost, mt, policy, envelope_bn, cap_at_feasible_max=False
            )[0]
        )
        if policy.means_tested:
            out.append(
                score_policy_by_eligibility(base, cost, mt, policy, envelope_bn)[0]
            )
    return out

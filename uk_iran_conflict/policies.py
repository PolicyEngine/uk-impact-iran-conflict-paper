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
    """A costed policy response."""

    key: str
    label: str
    source: str
    stated_cost_bn: float
    gain: Callable[[Baseline, ShockCost, np.ndarray], np.ndarray]


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
    """
    from uk_iran_conflict.incidence import wquantile  # noqa: PLC0415

    typical = wquantile(base.energy, base.weight, 0.5)
    block_value = block_share * typical
    covered = np.minimum(base.energy, block_value)
    # Value the covered block at post-shock prices: the discount is a rate the
    # household pays, so it applies to the shocked level, not the baseline one.
    shocked_covered = covered * (1.0 + _shock_rate(base, cost))
    return discount * shocked_covered + per_child * child_counts(base)


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


POLICIES: dict[str, Policy] = {
    "social_tariff": Policy(
        "social_tariff",
        "Means-tested social tariff (35% bill discount)",
        "HM Treasury signal, 2026; modelled here, as no such "
        "instrument exists in PolicyEngine UK",
        stated_cost_bn=float("nan"),
        gain=_social_tariff,
    ),
    "jrf_block": Policy(
        "jrf_block",
        "JRF universal discounted block (50% of typical use, + per-child)",
        "Joseph Rowntree Foundation, 9 Apr 2026",
        stated_cost_bn=5.0,
        gain=_jrf_block,
    ),
    "whd_expansion": Policy(
        "whd_expansion",
        "Warm Home Discount expansion (£150)",
        "DESNZ, qualifying date 23 Aug 2026",
        stated_cost_bn=float("nan"),
        gain=_whd_expansion,
    ),
    "vat_zero": Policy(
        "vat_zero",
        "Zero-rate VAT on domestic fuel (5% -> 0%)",
        "HMRC reduced rate; long-standing proposal",
        stated_cost_bn=float("nan"),
        gain=_vat_zero,
    ),
    "ippr_rebate": Policy(
        "ippr_rebate",
        "IPPR network-windfall rebate (£183 flat)",
        "IPPR, Apr 2026",
        stated_cost_bn=float("nan"),
        gain=_ippr_rebate,
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
    w = base.weight
    loss = cost.total
    net = loss - gain
    residual = _residual_loss(loss, gain)
    losers = loss > 0

    total_gain = wsum(gain, w)
    total_loss = wsum(loss, w)
    bottom3 = base.decile <= 3
    d1 = base.decile == 1
    d1_gain = wmean(gain[d1], w[d1]) if w[d1].sum() > 0 else float("nan")

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
        offset_d[d] = (
            float(min(wsum(gain[sel], wd_all), loss_d) / loss_d)
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
            cost_per_pound_decile_one=(
                (total_gain / 1e9) / d1_gain if d1_gain else float("nan")
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
        ),
        gain,
    )


def score_policy_at_envelope(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float = COMMON_ENVELOPE_BN,
) -> tuple[PolicyScore, np.ndarray]:
    """Score ``policy`` rescaled to a common exchequer envelope.

    Every instrument here is homogeneous of degree one in its own generosity
    parameter — the social tariff's ``discount``, the WHD amount, the IPPR
    rebate amount, the JRF block's ``discount`` and ``per_child``, and the VAT
    rate — so multiplying the gain array by a scalar is *exactly* equivalent to
    scaling those parameters, and needs no re-parameterisation of each
    instrument. (VAT zero-rating cannot in reality exceed the 5% rate, so a
    scale factor above one there describes a hypothetical deeper rate cut and
    is reported as such rather than clipped.)

    What this fixes: at stated cost the scorecard compares instruments of
    wildly different size and then reads the differences as *targeting*
    (docs/FIXES.md B7).
    """
    gain = policy.gain(base, cost, mt)
    total = wsum(gain, base.weight)
    if total <= 0:
        raise ValueError(f"policy {policy.key!r} has non-positive cost; cannot rescale")
    scale = envelope_bn * 1e9 / total
    return score_policy(
        base,
        cost,
        mt,
        policy,
        gain=gain * scale,
        envelope="common",
        envelope_bn=envelope_bn,
        envelope_scale=scale,
    )


def scorecard(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    policies: dict[str, Policy] | None = None,
) -> list[PolicyScore]:
    """Every instrument at its own stated cost **and** at a common envelope.

    Returns both sets, tagged by :attr:`PolicyScore.envelope`, so a table can
    put them side by side and the reader can see which conclusions survive
    holding the exchequer cost fixed.
    """
    chosen = POLICIES if policies is None else policies
    out: list[PolicyScore] = []
    for policy in chosen.values():
        out.append(score_policy(base, cost, mt, policy)[0])
        out.append(score_policy_at_envelope(base, cost, mt, policy, envelope_bn)[0])
    return out

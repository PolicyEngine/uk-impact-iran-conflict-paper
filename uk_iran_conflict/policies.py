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

    contributing = sorted(k for k, v in resolved.items() if v["households_m"] > 0)
    empty = sorted(k for k, v in resolved.items() if v["households_m"] <= 0)

    return {
        "period": period,
        "resolved_variables": sorted(resolved),
        # Round-3 referees: resolving and contributing are different things.
        # Four of the eight listed variables resolve against the tax-benefit
        # system and then return **exactly zero households** (child tax credit,
        # working tax credit, income support, JSA-income are wound down in the
        # modelled year). "All eight resolve correctly" is true and says nothing
        # about which of them actually put a household into the means-tested
        # population. The two sets are therefore persisted separately, and the
        # means test is carried by ``contributing_variables`` alone.
        "contributing_variables": contributing,
        "empty_variables": empty,
        "n_resolved": len(resolved),
        "n_contributing": len(contributing),
        "n_empty": len(empty),
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
            "this file records. `resolved_variables` is what the release "
            "defines; `contributing_variables` is what reaches a household. "
            "Only the second bears on the modelled means-tested population."
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
    #: What the ceiling in :attr:`feasible_max` **is**, in one sentence, so the
    #: prose can describe the rule rather than guess at it. Round-4 finding 2:
    #: the Warm Home Discount's ceiling was described in the paper as "the
    #: payment that exhausts the eligible population's entire loss" when the
    #: code used the mean domestic *bill* of that population — about six and a
    #: half times the mean loss. A ceiling rule that is not written down gets
    #: described wrongly.
    feasible_max_rule: str = ""
    #: Reference quantity the instrument is defined against, where it has one
    #: (the JRF block's typical consumption). Persisted on every row so the
    #: quantity being subsidised is never implicit.
    reference_basis: str = ""
    reference_quantity: float | Callable[[Baseline], float] | None = None

    def feasible_max_parameter(
        self, base: Baseline, cost: ShockCost, mt: np.ndarray
    ) -> float:
        """The instrument's parameter ceiling, resolved against the data."""
        if callable(self.feasible_max):
            return float(self.feasible_max(base, cost, mt))
        return float(self.feasible_max)

    def reference_quantity_gbp(self, base: Baseline) -> float:
        """The instrument's reference quantity, resolved against the data."""
        if self.reference_quantity is None:
            return float("nan")
        if callable(self.reference_quantity):
            return float(self.reference_quantity(base))
        return float(self.reference_quantity)

    def stated_cost_simulated_bn(
        self, base: Baseline, cost: ShockCost, mt: np.ndarray
    ) -> float:
        """Simulated exchequer cost at the sponsor's own parameter, £bn."""
        return wsum(self.gain(base, cost, mt), base.weight) / 1e9

    def max_scale(self, base: Baseline, cost: ShockCost, mt: np.ndarray) -> float:
        """Ratio of the feasible maximum parameter to the stated one."""
        ceiling = self.feasible_max_parameter(base, cost, mt)
        return (
            ceiling / self.stated_parameter
            if self.stated_parameter > 0
            else float("inf")
        )

    def feasible_max_cost_bn(
        self, base: Baseline, cost: ShockCost, mt: np.ndarray
    ) -> float:
        """Cost of running this instrument **at its feasible maximum**, £bn.

        The true feasible maximum: the instrument is homogeneous of degree one
        in its parameter, so this is the stated-parameter cost times
        :meth:`max_scale`. It is *not* capped at any envelope — an envelope is a
        budget constraint, not a property of the instrument — which is exactly
        the conflation the round-3 referees found.
        """
        scale = self.max_scale(base, cost, mt)
        if not np.isfinite(scale):
            return float("inf")
        return self.stated_cost_simulated_bn(base, cost, mt) * scale

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


#: Ofgem's typical-consumption annual dual-fuel bill on the cap basis JRF
#: costed their proposal against (£/yr, April 2026 cap, direct debit, 11,500
#: kWh gas + 2,700 kWh electricity). **The reference quantity for the JRF
#: block's "typical consumption", and the default.**
#:
#: Round-3 referees: the block was previously pegged to the weighted median of
#: our own *modelled* domestic bill. That is not JRF's typical consumption and
#: not anybody's — the modelled domestic imputation is roughly a quarter low
#: (docs/VALIDATION.md), so the block quantity was understated on a
#: non-comparable base while the paper read the resulting cost against JRF's
#: own £5bn and called the difference generosity. The basis is now an explicit
#: parameter and both bases are reported side by side
#: (:func:`jrf_reference_quantities`).
OFGEM_TYPICAL_ANNUAL_BILL_GBP: float = 1_723.0

#: The two admissible reference bases for the JRF block's typical consumption.
JRF_REFERENCE_BASES: tuple[str, ...] = ("ofgem_typical_consumption", "modelled_median")

JRF_DEFAULT_REFERENCE_BASIS: str = "ofgem_typical_consumption"


def jrf_reference_quantity(
    base: Baseline,
    basis: str = JRF_DEFAULT_REFERENCE_BASIS,
    typical_consumption_gbp: float | None = None,
) -> float:
    """Resolve the block's reference annual bill, £/yr.

    ``typical_consumption_gbp`` overrides everything (so a sibling module that
    recalibrates the baseline can pass its own figure and nothing here has to
    be edited). Otherwise ``basis`` selects between Ofgem's published typical
    consumption — JRF's own peg, and the default — and the weighted median of
    the modelled domestic bill, retained so the previous number is reproducible.
    """
    if typical_consumption_gbp is not None:
        return float(typical_consumption_gbp)
    if basis == "ofgem_typical_consumption":
        return float(OFGEM_TYPICAL_ANNUAL_BILL_GBP)
    if basis == "modelled_median":
        return float(wquantile(base.energy, base.weight, 0.5))
    raise ValueError(f"unknown JRF reference basis {basis!r}; {JRF_REFERENCE_BASES}")


def jrf_reference_quantities(
    base: Baseline, block_share: float = 0.50
) -> dict[str, float]:
    """Both reference bases and the block each implies — reported, not chosen.

    The paper must be able to say what quantity it subsidised and what the
    alternative basis would have given, without a reader having to rerun
    anything.
    """
    ofgem = jrf_reference_quantity(base, "ofgem_typical_consumption")
    modelled = jrf_reference_quantity(base, "modelled_median")
    return {
        "basis_used": JRF_DEFAULT_REFERENCE_BASIS,
        "ofgem_typical_consumption_gbp": ofgem,
        "ofgem_block_gbp": block_share * ofgem,
        "modelled_median_domestic_bill_gbp": modelled,
        "modelled_median_block_gbp": block_share * modelled,
        "modelled_over_ofgem": (modelled / ofgem) if ofgem > 0 else float("nan"),
        "block_share": block_share,
        "note": (
            "JRF peg the discounted block to Ofgem typical consumption. The "
            "modelled median is the weighted median of this model's own "
            "domestic bill, which the validation notes is materially below the "
            "Ofgem basis; a block set on it is not the sponsor's block and its "
            "cost is not comparable with the sponsor's costing."
        ),
    }


#: The JRF block's three parameters, and which of them the sponsor actually
#: published. **Round-4 finding 6.** JRF state the block SIZE ("50% of typical
#: consumption"), the per-child allowance's existence, and the total (~£5bn).
#: They do **not** publish the discount rate — their proposal is "a discounted
#: rate on the first 50% of typical consumption", with the rate unstated (see
#: docs/RESEARCH_BRIEF.md, Moore & Cook, JRF, 9 Apr 2026). The 50% discount and
#: the £60 per child modelled here are *this paper's* assumptions, and they are
#: the whole of the costing gap: see :func:`jrf_costing_gap`.
JRF_BLOCK_SHARE: float = 0.50
JRF_BLOCK_DISCOUNT: float = 0.50
JRF_PER_CHILD_GBP: float = 60.0
JRF_STATED_COST_BN: float = 5.0

#: Which JRF block parameters are the sponsor's and which are ours.
JRF_PARAMETER_PROVENANCE: dict[str, str] = {
    "block_share": "sponsor: 50 per cent of typical consumption",
    "reference_quantity": (
        "sponsor: Ofgem typical consumption (the default reference basis, so "
        "the modelled block and JRF's are on the SAME basis and their costings "
        "are directly comparable)"
    ),
    "discount": (
        "OURS. JRF say 'a discounted rate' and do not publish the rate. 50 per "
        "cent is this paper's assumption, chosen by analogy with the social "
        "tariff, and it is not attributable to JRF"
    ),
    "per_child_gbp": (
        "OURS. JRF propose a per-child allowance without publishing its value; "
        "£60 is this paper's assumption"
    ),
    "eligibility": (
        "sponsor: universal. JRF describe the block as universal by design, "
        "which is the feature they contrast with a social tariff"
    ),
    "total_cost_bn": "sponsor: ~£5bn",
}


def jrf_costing_gap(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    block_share: float = JRF_BLOCK_SHARE,
    discount: float = JRF_BLOCK_DISCOUNT,
    per_child: float = JRF_PER_CHILD_GBP,
    sponsor_cost_bn: float = JRF_STATED_COST_BN,
    reference_basis: str = JRF_DEFAULT_REFERENCE_BASIS,
) -> dict[str, Any]:
    """Decompose the gap between the modelled JRF block and JRF's own £5bn.

    **Round-4 finding 6, resolved rather than re-worded.** The block defaults to
    JRF's own reference basis (:data:`OFGEM_TYPICAL_ANNUAL_BILL_GBP`), so the
    two costings *are* directly comparable and the paper may not decline the
    comparison on the grounds that the bases differ. The modelled cost is about
    £10.9bn against JRF's ~£5bn — a factor of 2.19 — and this function says
    where it comes from.

    The decomposition
    -----------------
    Write ``W`` for weighted households, ``B = block_share x typical`` for the
    block's cash value and ``bill_i`` for each household's modelled domestic
    bill. Then

    ``universal_ceiling`` = ``discount x B x W`` + the per-child allowance
        what the instrument costs if **every** household's bill covers the whole
        block. This depends on nothing in the microdata except the household
        count.
    ``block_truncation`` = the reduction from households whose whole bill is
        smaller than the block and so cannot use all of it — the only channel
        through which the modelled bill distribution can move the cost at all.
    ``modelled_cost`` = ``universal_ceiling - block_truncation``.

    The result
    ----------
    The truncation term is genuinely present (the modelled domestic bill is
    about a quarter low, docs/VALIDATION.md) but it works in the **wrong
    direction to explain the gap**: it makes the modelled block *cheaper* than
    the mechanical ceiling, and it is only about a fifth of the difference. The
    residual is not a microdata artefact at all — it is visible with no
    microdata whatsoever, because JRF's own £5bn over ~29.5m households is about
    £170 a household, while a 50 per cent discount on 50 per cent of a £1,723
    typical bill is £431 a household before any allowance. **£5bn cannot buy the
    instrument as modelled, on JRF's own basis, for any distribution of bills.**

    So the gap is in the parameters, and exactly one of them is unpublished:
    the discount rate (:data:`JRF_PARAMETER_PROVENANCE`). Holding the per-child
    allowance fixed, JRF's own total pins the discount at about 20 per cent, not
    the 50 per cent modelled here. ``implied_discount`` is that number. The two
    other single-parameter reconciliations are reported beside it —
    ``implied_block_share`` and ``implied_eligible_share`` — so a reader can see
    that any one of the three closes the gap alone and that JRF's publication
    does not say which.

    ``netting_off_existing_support`` is ruled out arithmetically: the only
    existing instrument of the five that actually exists in UK policy is the
    Warm Home Discount, whose modelled cost at £150 is an order of magnitude
    below the residual.
    """
    typical = jrf_reference_quantity(base, reference_basis)
    block_value = block_share * typical
    w = base.weight
    households = float(w.sum())
    children_bn = per_child * wsum(child_counts(base), w) / 1e9
    ceiling_block_bn = discount * block_value * households / 1e9
    covered = np.minimum(base.energy, block_value)
    modelled_block_bn = discount * wsum(covered, w) / 1e9
    truncation_bn = ceiling_block_bn - modelled_block_bn
    modelled_bn = modelled_block_bn + children_bn
    gap_bn = modelled_bn - sponsor_cost_bn
    # The truncation term works in the OPPOSITE direction to the gap: it makes
    # the modelled block cheaper. None of the gap is explained by it, so the
    # residual the microdata cannot account for is the whole gap.
    residual_bn = gap_bn

    implied_discount = (
        (sponsor_cost_bn * 1e9 - children_bn * 1e9) / wsum(covered, w)
        if wsum(covered, w) > 0
        else float("nan")
    )
    # Block share that would cost the sponsor's total at the modelled discount.
    # Only defined if the modelled instrument costs MORE than the sponsor's
    # total at the modelled block share; on a small synthetic baseline it may
    # not, and a bisection that cannot bracket the target must say so rather
    # than return its own lower bound.
    reachable = modelled_bn >= sponsor_cost_bn >= children_bn
    lo, hi = 0.0, block_share
    for _ in range(200 if reachable else 0):
        mid = 0.5 * (lo + hi)
        trial = (
            discount * wsum(np.minimum(base.energy, mid * typical), w) / 1e9
            + children_bn
        )
        if trial > sponsor_cost_bn:
            hi = mid
        else:
            lo = mid
    implied_block_share = 0.5 * (lo + hi) if reachable else float("nan")
    # Eligible share, admitting poorest-first at the modelled parameters.
    gain = discount * covered + per_child * child_counts(base)
    order = np.argsort(base.equiv_income_ahc, kind="stable")
    running = np.cumsum(gain[order] * w[order])
    cum_w = np.cumsum(w[order])
    idx = int(np.searchsorted(running, sponsor_cost_bn * 1e9))
    implied_eligible_share = (
        float(cum_w[min(idx, len(cum_w) - 1)] / households)
        if households
        else float("nan")
    )
    whd_bn = POLICIES["whd_expansion"].stated_cost_simulated_bn(base, cost, mt)

    return {
        "comparable": True,
        "why_comparable": (
            "The modelled block is pegged to JRF's own reference quantity "
            f"({reference_basis}, £{typical:,.0f}), which is what "
            "JRF_DEFAULT_REFERENCE_BASIS selects. The two costings are on the "
            "same basis and the comparison may not be declined."
        ),
        "sponsor_cost_bn": sponsor_cost_bn,
        "modelled_cost_bn": modelled_bn,
        "gap_bn": gap_bn,
        "ratio": (modelled_bn / sponsor_cost_bn if sponsor_cost_bn else float("nan")),
        "decomposition": {
            "universal_ceiling_block_bn": ceiling_block_bn,
            "per_child_allowance_bn": children_bn,
            "universal_ceiling_total_bn": ceiling_block_bn + children_bn,
            "block_truncation_bn": truncation_bn,
            "block_truncation_share_of_ceiling": (
                truncation_bn / ceiling_block_bn if ceiling_block_bn else float("nan")
            ),
            "modelled_block_bn": modelled_block_bn,
            "residual_unexplained_by_microdata_bn": residual_bn,
            "note": (
                "The modelled bill distribution enters through ONE channel: "
                "households whose whole bill is below the block cannot use all "
                "of it. That channel makes the modelled block CHEAPER, so it "
                "cannot explain a modelled cost above the sponsor's. The whole "
                "of the gap survives it."
            ),
        },
        "per_household": {
            "households_m": households / 1e6,
            "sponsor_implied_gbp": sponsor_cost_bn * 1e9 / households,
            "modelled_gbp": modelled_bn * 1e9 / households,
            "mechanical_full_block_gbp": discount * block_value,
            "microdata_free": (
                "These four numbers use no microdata beyond the household "
                "count. A 50 per cent discount on 50 per cent of a "
                f"£{typical:,.0f} typical bill is £{discount * block_value:,.0f} "
                "a household; JRF's own total is "
                f"£{sponsor_cost_bn * 1e9 / households:,.0f}. The gap is "
                "arithmetic in the sponsor's published figures, not an artefact "
                "of this model."
            ),
        },
        "single_parameter_reconciliations": {
            "implied_discount": implied_discount,
            "modelled_discount": discount,
            "implied_block_share": implied_block_share,
            "implied_block_share_is_defined": bool(reachable),
            "modelled_block_share": block_share,
            "implied_eligible_share_poorest_first": implied_eligible_share,
            "modelled_eligible_share": 1.0,
            "note": (
                "Each of these three, alone, reproduces the sponsor's total. "
                "Only one of the three is unpublished by the sponsor."
            ),
        },
        "parameter_provenance": JRF_PARAMETER_PROVENANCE,
        "netting_off_existing_support": {
            "warm_home_discount_modelled_bn": whd_bn,
            "share_of_gap_it_could_close": (
                whd_bn / gap_bn if gap_bn else float("nan")
            ),
            "can_close_gap": bool(whd_bn >= gap_bn),
            "note": (
                "Netting off existing support is the third candidate "
                "explanation and it does not survive: the Warm Home Discount is "
                "the only one of the five instruments that presently exists, "
                "and its modelled cost is far below the gap."
            ),
        },
        "resolution": (
            "RESOLVED. The gap is not a difference of basis, not a difference "
            "of eligible set that JRF state, and not netting off. It is the "
            "discount rate: JRF publish the block SIZE and the total but not "
            "the RATE, and the 50 per cent modelled here is this paper's own "
            "assumption. JRF's own total implies a discount of about "
            f"{100 * implied_discount:.0f} per cent on the same block. The "
            "paper should report the modelled block as a 50 per cent discount "
            "costing £{:.1f}bn, state that JRF's £5bn corresponds to a "
            "{:.0f} per cent discount on the same block, and stop describing "
            "the difference as generosity, as incomparable bases, or as "
            "unexplained."
        ).format(modelled_bn, 100 * implied_discount),
    }


def _jrf_block(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    block_share: float = 0.50,
    discount: float = 0.50,
    per_child: float = 60.0,
    revalue_at_post_shock_prices: bool = False,
    reference_basis: str = JRF_DEFAULT_REFERENCE_BASIS,
    typical_consumption_gbp: float | None = None,
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

    ``reference_basis`` / ``typical_consumption_gbp`` (round-3 referees)
    ------------------------------------------------------------------
    Typical consumption is a **fixed quantity**, not a proportion of each
    household's own bill (a proportional discount would mechanically pay most
    to the biggest users). Which fixed quantity is now explicit and defaults to
    JRF's own peg, Ofgem typical consumption
    (:data:`OFGEM_TYPICAL_ANNUAL_BILL_GBP`), rather than to the weighted median
    of this model's domestic bill — an imputation the validation puts about a
    quarter low, on which the block was understated and its cost not comparable
    with the sponsor's £5bn. Both bases are reported by
    :func:`jrf_reference_quantities`.
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
    typical = jrf_reference_quantity(base, reference_basis, typical_consumption_gbp)
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


#: Named ceiling rules for a flat per-household payment, so a rule is never
#: implicit. Round-4 finding 2.
FLAT_PAYMENT_CEILING_RULES: tuple[str, ...] = (
    "mean_eligible_domestic_bill",
    "mean_eligible_loss",
)

#: Which rule the Warm Home Discount expansion is scored on. The **bill** rule,
#: deliberately, because the instrument-internal question is "at what payment
#: does this stop being energy-bill support?", not "at what payment is nobody
#: left short?". Both are computed and persisted by
#: :func:`flat_payment_ceilings`; the paper must describe the one it uses.
WHD_CEILING_RULE: str = "mean_eligible_domestic_bill"


def _mean_eligible_loss(base: Baseline, cost: ShockCost, mt: np.ndarray) -> float:
    """Mean **total shock loss** of the means-tested population, £/yr.

    The loss-exhausting flat payment: paid to every eligible household it
    spends exactly the eligible population's aggregate loss, so in aggregate
    the group is made whole (individually it is not — a flat payment
    overcompensates the below-mean losers and undercompensates the rest, which
    is what the overcompensation fields measure).

    This is the ceiling the paper's prose describes. It is **not** the ceiling
    the scorecard uses: see :func:`_mean_eligible_domestic_bill` and
    :data:`WHD_CEILING_RULE`.
    """
    sel = np.asarray(mt, dtype=bool)
    w = base.weight[sel]
    return float(wmean(cost.total[sel], w)) if w.sum() > 0 else float("nan")


def flat_payment_ceilings(
    base: Baseline, cost: ShockCost, mt: np.ndarray
) -> dict[str, Any]:
    """Both admissible ceilings for a flat means-tested payment, side by side.

    **Round-4 finding 2.** The scorecard's Warm Home Discount ceiling is
    :func:`_mean_eligible_domestic_bill` — the mean shocked domestic *bill* of
    the eligible population. The paper described that number as "the payment
    that exhausts the eligible population's entire loss", which is a different
    quantity entirely: the mean loss among eligibles is roughly a sixth of
    their mean bill, because most of the modelled loss sits in motor fuel and
    the means-tested population is imputed almost none of it. That is why the
    same row discloses a £835 payment against a £125 mean loss and reads as a
    contradiction — it is not a contradiction, it is two different rules, one
    of which was never written down.

    Both are returned, with the ratio, so the prose can say which rule it
    imposed and what the other one would have given.
    """
    bill = _mean_eligible_domestic_bill(base, cost, mt)
    loss = _mean_eligible_loss(base, cost, mt)
    sel = np.asarray(mt, dtype=bool)
    w = base.weight
    eligible_w = float(w[sel].sum())
    return {
        "rule_used": WHD_CEILING_RULE,
        "rules": list(FLAT_PAYMENT_CEILING_RULES),
        "mean_eligible_domestic_bill_gbp": bill,
        "mean_eligible_loss_gbp": loss,
        "bill_over_loss": (bill / loss) if loss else float("nan"),
        "cost_at_bill_rule_bn": bill * eligible_w / 1e9,
        "cost_at_loss_rule_bn": loss * eligible_w / 1e9,
        "eligible_households_m": eligible_w / 1e6,
        "what_each_rule_is": {
            "mean_eligible_domestic_bill": (
                "the mean shocked domestic energy bill of the eligible "
                "population: the point beyond which a payment is no longer "
                "energy-bill support of any kind, because it exceeds the bill "
                "it is meant to help pay. This is the rule the scorecard uses."
            ),
            "mean_eligible_loss": (
                "the mean TOTAL shock loss of the eligible population: the flat "
                "payment that, paid to all of them, spends exactly their "
                "aggregate loss. This is the rule the paper's prose described. "
                "It is far smaller, because the means-tested population is "
                "imputed very little motor fuel and motor fuel carries most of "
                "the loss."
            ),
        },
        "note": (
            "The paper may describe the ceiling as bill-exhausting or as "
            "loss-exhausting, but not as both, and the number it quotes must "
            "match the rule it names."
        ),
    }


def feasible_max_identity(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policies: dict[str, Policy] | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Which instruments have the **same** feasible maximum by construction.

    **Round-4 finding 1.** The social tariff's ceiling is a 100% discount on the
    means-tested population's shocked domestic bill. The Warm Home Discount's
    ceiling is :func:`_mean_eligible_domestic_bill` — the *mean* shocked
    domestic bill of the *same* population. Aggregated over that population the
    two are the same sum:

        sum_i w_i x 1.00 x bill_i
          == (sum_i w_i) x mean_w(bill)
          == sum_i w_i x bill_i

    so their feasible-maximum costs are equal to floating point (£3.76bn each),
    not by coincidence and not as independent confirmation of anything. The
    paper presented them as two instruments independently establishing that
    means-tested support saturates below the envelope, which doubles the
    apparent evidence for a single arithmetic fact.

    This function detects the coincidence from the numbers themselves rather
    than asserting it, groups the instruments that share a feasible maximum,
    and sets ``report_as_one_result`` so a table builder or a prose writer
    cannot treat a group of size two as two results.
    """
    chosen = POLICIES if policies is None else policies
    costs = {
        key: policy.feasible_max_cost_bn(base, cost, mt)
        for key, policy in chosen.items()
    }
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, value in costs.items():
        if key in seen or not np.isfinite(value):
            continue
        members = [
            other
            for other, other_value in costs.items()
            if np.isfinite(other_value)
            and abs(other_value - value) <= tolerance * max(abs(value), 1.0)
        ]
        seen.update(members)
        if len(members) > 1:
            values = [costs[m] for m in members]
            groups.append(
                {
                    "policies": members,
                    "feasible_max_cost_bn": value,
                    "max_absolute_difference_bn": max(values) - min(values),
                    "distinct_results": 1,
                    "reported_as_distinct_results": len(members),
                }
            )
    identical: dict[str, list[str]] = {}
    for group in groups:
        for member in group["policies"]:
            identical[member] = [m for m in group["policies"] if m != member]
    return {
        "tolerance": tolerance,
        "feasible_max_cost_bn": costs,
        "identical_groups": groups,
        "identical_to": identical,
        "any_identical": bool(groups),
        "report_as_one_result": bool(groups),
        "distinct_feasible_maxima": len(costs)
        - sum(len(g["policies"]) - 1 for g in groups),
        "proof": (
            "A 100% discount on every eligible household's shocked domestic "
            "bill and a flat payment equal to the WEIGHTED MEAN of those same "
            "bills, paid to the same population, are the same aggregate sum by "
            "the definition of a weighted mean. The equality is an identity, "
            "not a finding, and the two rows are one result."
        ),
        "note": (
            "Round-4 finding 1. Any group listed here must be reported as ONE "
            "result. Two instruments whose ceilings are the same number by "
            "construction do not independently confirm saturation."
        ),
    }


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
        feasible_max_rule=(
            "a 100 per cent discount: the whole shocked domestic bill of the "
            "means-tested population, i.e. free energy. Aggregated this is the "
            "SAME sum as the Warm Home Discount's bill-based ceiling — see "
            "feasible_max_identity"
        ),
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
        feasible_max_rule=(
            "a 100 per cent discount on the block: the first 50 per cent of "
            "typical consumption supplied free, plus the per-child allowance"
        ),
        generic_template=(
            "discounted first block at {p:.1f} per cent off 50 per cent of "
            "typical use, universal"
        ),
        reference_basis=JRF_DEFAULT_REFERENCE_BASIS,
        reference_quantity=(lambda base: jrf_reference_quantity(base)),
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
        feasible_max_rule=(
            "the MEAN SHOCKED DOMESTIC BILL of the eligible population "
            "(_mean_eligible_domestic_bill), NOT the payment that exhausts "
            "their loss: the mean loss among eligibles is several times "
            "smaller. Both rules are computed by flat_payment_ceilings"
        ),
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
        feasible_max_rule=(
            "the reduced rate itself: removing more than five percentage "
            "points is a negative VAT rate"
        ),
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
        feasible_max_rule=(
            "the sponsor's own parameter: the instrument is funded by the "
            "network-company windfall and £183 already spends it, so there is "
            "no headroom above the stated value"
        ),
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
    #: Share of households eligible. Persisted on **every** row, not only the
    #: eligibility-widening ones: a means-tested instrument whose widened
    #: eligibility has reached 1.0 has stopped being means-tested, and that fact
    #: must be visible in the row rather than inferable from its cost.
    eligible_share: float = float("nan")
    #: Eligible households, millions, on the same definition.
    eligible_households_m: float = float("nan")
    #: True when eligibility has reached (within tolerance) every household.
    eligibility_is_universal: bool = False
    #: Plain-language warning attached to a row whose means test has vanished.
    eligibility_note: str = ""
    #: Units string for :attr:`cost_per_pound_decile_one`.
    cost_per_pound_decile_one_units: str = ""
    # --- what kind of row this is (round-3 finding 1) ---------------------
    #: Machine-readable row type; equal to :attr:`envelope`. Kept beside
    #: :attr:`row_semantics` so a table builder never has to guess what a
    #: column means.
    row_kind: str = ""
    #: One sentence saying what operation produced this row. The "feasible
    #: maximum" column previously mixed three of them — raising an instrument
    #: to its ceiling, leaving it alone, and scaling it *down* to fit an
    #: envelope — under one label.
    row_semantics: str = ""
    #: Cost of the instrument at its own feasible maximum, £bn, **uncapped by
    #: any envelope**. The same value on every row of a given policy, because
    #: it is a property of the instrument.
    feasible_max_cost_bn: float = float("nan")
    #: Envelope minus what this row actually spends. The two arms of the
    #: comparison do not spend the same money, and the prose may not claim they
    #: do; this is the number that settles it.
    envelope_shortfall_bn: float = float("nan")
    #: Whether this row spends the full envelope (within 0.5%).
    spends_full_envelope: bool = False
    # --- overcompensation (round-3 finding 2) ----------------------------
    #: Households paid **more than they lost**, millions and as shares. The
    #: claim that no household is overcompensated is testable on our own
    #: numbers, and at the feasible maximum it is false; these fields let the
    #: paper state it instead of denying it.
    overcompensated_households_m: float = float("nan")
    overcompensated_share_of_households: float = float("nan")
    overcompensated_share_of_recipients: float = float("nan")
    #: Total spend going to those households, £bn, and the part of it above
    #: their own loss (the pounds that compensate nothing).
    overcompensated_spend_bn: float = float("nan")
    overcompensated_excess_bn: float = float("nan")
    overcompensated_share_of_spend: float = float("nan")
    #: Mean gain and mean loss among recipients, £/yr — the pair the referees
    #: compared (£760 paid against a £275 mean loss).
    mean_gain_if_recipient_gbp: float = float("nan")
    mean_loss_if_recipient_gbp: float = float("nan")
    mean_loss_gbp: float = float("nan")
    #: Ratio of the two. Above 1 means the average recipient is paid more than
    #: it lost.
    gain_to_loss_ratio_recipients: float = float("nan")
    # --- reference quantity (round-3 finding 5) ---------------------------
    reference_basis: str = ""
    reference_quantity_gbp: float = float("nan")
    # --- admission rule (round-4 finding 3) -------------------------------
    #: For a ``common_eligibility`` row: which observable the state ranked
    #: non-claimants on to admit them (:data:`ADMISSION_RULES`). Empty for
    #: every other row type.
    admission_rule: str = ""
    #: Whether that rule is the perfect-observability rule: the state ranking
    #: every non-claimant by equivalised AHC income without error. A row for
    #: which this is true is the eligibility arm's BEST case on targeting and
    #: the prose may not report it as *the* eligibility-arm result. It is an
    #: upper bound on what the state can observe, not on every metric — see
    #: :func:`eligibility_admission_range`.
    admission_rule_is_upper_bound: bool = False


#: What each row type actually is, in one sentence. Emitted with every row.
#:
#: Round-3 referees: the pre-revision scorecard had a column labelled "feasible
#: maximum" that was ``min(envelope_scale, max_scale)`` — three different
#: operations depending on the instrument. It raised the social tariff (35 ->
#: 100%) and the Warm Home Discount (£150 -> £758) to their ceilings, left VAT
#: alone, and scaled the JRF block (50 -> 35.8%) and the flat rebate (£183 ->
#: £170) *down* to fit £5bn. Reading down that column compared a ceiling with a
#: budget constraint. The two questions are now two row types:
#: ``feasible_max`` answers "how far can this instrument go?" and
#: ``common_capped`` answers "how much of the envelope can it absorb?".
ROW_SEMANTICS: dict[str, str] = {
    "stated": ("the sponsor's own parameter, at whatever it costs"),
    "feasible_max": (
        "the instrument at its OWN feasible maximum parameter, uncapped by any "
        "envelope: the true feasible maximum, and the only row down which "
        "feasible maxima are comparable"
    ),
    "common_capped": (
        "what the instrument can absorb WITHIN the envelope: scaled toward it "
        "and stopped at the feasible maximum, so an instrument that saturates "
        "reports the smaller sum it can actually spend. Not a feasible maximum "
        "— for an instrument that already costs more than the envelope this "
        "scales generosity DOWN"
    ),
    "common_scaled": (
        "pure scalar rescaling to the envelope, retained only for "
        "auditability; the implied parameter may be outside the instrument's "
        "parameter space, in which case the row is renamed generically"
    ),
    "common_eligibility": (
        "the sponsor's own generosity, with eligibility widened down the income "
        "ranking until the envelope is spent or everyone is in"
    ),
}

#: A row is treated as universal — no longer means-tested in anything but name
#: — at or above this eligible share.
UNIVERSAL_ELIGIBILITY_TOLERANCE: float = 0.999


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
    eligible_share: float | None = None,
    feasible_max_cost_bn: float = float("nan"),
    admission_rule: str = "",
    admission_rule_is_upper_bound: bool = False,
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

    # --- who is eligible, on every row -----------------------------------
    recipients = gain > 0
    w_total = w.sum()
    if eligible_share is None:
        eligible_share = (
            float(w[recipients].sum() / w_total) if w_total > 0 else (float("nan"))
        )
    eligible_households_m = (
        float(eligible_share * w_total / 1e6)
        if np.isfinite(eligible_share)
        else (float("nan"))
    )
    is_universal = bool(
        np.isfinite(eligible_share)
        and eligible_share >= UNIVERSAL_ELIGIBILITY_TOLERANCE
    )
    eligibility_note = ""
    if is_universal and policy.means_tested:
        eligibility_note = (
            f"eligibility has reached {eligible_share:.4f} of households: this "
            "row is universal, not means-tested, and is arithmetically a flat "
            "payment to everyone. Its targeting statistics are those of a "
            "universal rebate and must not be read as a means test."
        )
    elif is_universal:
        eligibility_note = "universal by design; every household is eligible."

    # --- overcompensation --------------------------------------------------
    over = recipients & (gain > loss)
    w_over = w[over].sum()
    over_spend = wsum(gain[over], w[over])
    over_excess = wsum((gain - loss)[over], w[over])
    w_recipients = w[recipients].sum()

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
            eligible_households_m=eligible_households_m,
            eligibility_is_universal=is_universal,
            eligibility_note=eligibility_note,
            row_kind=envelope,
            row_semantics=ROW_SEMANTICS.get(envelope, ""),
            feasible_max_cost_bn=feasible_max_cost_bn,
            envelope_shortfall_bn=(
                envelope_bn - total_gain / 1e9
                if np.isfinite(envelope_bn)
                else float("nan")
            ),
            spends_full_envelope=bool(
                np.isfinite(envelope_bn)
                and envelope_bn > 0
                and abs(total_gain / 1e9 - envelope_bn) <= 0.005 * envelope_bn
            ),
            overcompensated_households_m=float(w_over / 1e6),
            overcompensated_share_of_households=(
                float(w_over / w_total) if w_total > 0 else float("nan")
            ),
            overcompensated_share_of_recipients=(
                float(w_over / w_recipients) if w_recipients > 0 else float("nan")
            ),
            overcompensated_spend_bn=float(over_spend / 1e9),
            overcompensated_excess_bn=float(over_excess / 1e9),
            overcompensated_share_of_spend=(
                float(over_spend / total_gain) if total_gain > 0 else float("nan")
            ),
            mean_gain_if_recipient_gbp=(
                wmean(gain[recipients], w[recipients])
                if w_recipients > 0
                else float("nan")
            ),
            mean_loss_if_recipient_gbp=(
                wmean(loss[recipients], w[recipients])
                if w_recipients > 0
                else float("nan")
            ),
            mean_loss_gbp=wmean(loss, w),
            gain_to_loss_ratio_recipients=(
                float(
                    wmean(gain[recipients], w[recipients])
                    / wmean(loss[recipients], w[recipients])
                )
                if w_recipients > 0 and wmean(loss[recipients], w[recipients]) > 0
                else float("nan")
            ),
            reference_basis=policy.reference_basis,
            reference_quantity_gbp=policy.reference_quantity_gbp(base),
            admission_rule=admission_rule,
            admission_rule_is_upper_bound=admission_rule_is_upper_bound,
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
    stated_cost_bn = wsum(gain, base.weight) / 1e9
    feasible_cost_bn = (
        stated_cost_bn * max_scale if np.isfinite(max_scale) else float("inf")
    )
    # What the instrument can absorb inside the envelope: the smaller of the
    # envelope and its own feasible-maximum cost. NOT its feasible maximum.
    absorbable_bn = min(envelope_bn, feasible_cost_bn)

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
        feasible_max_cost_bn=feasible_cost_bn,
    )


def score_policy_at_feasible_max(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float = COMMON_ENVELOPE_BN,
) -> tuple[PolicyScore, np.ndarray]:
    """Score the instrument at its **own** feasible maximum parameter.

    The row the pre-revision "feasible maximum" column claimed to be and was
    not. No envelope enters: the parameter goes to
    :meth:`Policy.feasible_max_parameter` and the cost is whatever that costs —
    a 100% JRF block is roughly £14bn, and saying so is the point. ``envelope_bn``
    is carried only so the row can report how far past (or short of) the common
    envelope the instrument's own ceiling lies.

    Read this row for "how far can this instrument go?" and the
    ``common_capped`` row for "how much of £5bn can it absorb?". They are
    different questions and were previously the same column.
    """
    gain = policy.gain(base, cost, mt)
    ceiling = policy.feasible_max_parameter(base, cost, mt)
    stated = policy.stated_parameter
    scale = ceiling / stated if stated > 0 else float("inf")
    if not np.isfinite(scale):
        raise ValueError(
            f"policy {policy.key!r} has no finite feasible maximum; it cannot "
            "be scored at one"
        )
    stated_cost_bn = wsum(gain, base.weight) / 1e9
    feasible_cost_bn = stated_cost_bn * scale
    return score_policy(
        base,
        cost,
        mt,
        policy,
        gain=gain * scale,
        envelope="feasible_max",
        envelope_bn=envelope_bn,
        envelope_scale=scale,
        implied_parameter=ceiling,
        feasible_max_parameter=ceiling,
        is_feasible=True,
        absorbable_envelope_bn=min(envelope_bn, feasible_cost_bn),
        feasible_max_cost_bn=feasible_cost_bn,
    )


#: How the state decides **who** to admit when it widens eligibility, ordered
#: from perfect income observability to none.
#:
#: **Round-4 finding 3.** The original rule — and still the default, because it
#: is the right upper bound — admits non-claimant households in ascending order
#: of equivalised after-housing-costs income. That assumes the state can rank
#: every household *outside* the benefit system by AHC-equivalised income,
#: poorest first, with no error. It cannot: AHC income requires housing costs
#: and household composition it does not hold for non-claimants, which is
#: precisely the administrative capability this paper argues does not exist and
#: the reason a social tariff is hard to deliver at all. Running only that rule
#: reports the eligibility arm at its **best case** and compares it against a
#: generosity arm with no corresponding idealisation.
#:
#: ``"equivalised_ahc_income"``
#:     Perfect observability, poorest first. **Upper bound.**
#: ``"unequivalised_net_income"``
#:     Household net income, poorest first, with no AHC deduction and no
#:     equivalisation — closer to what an income-based screen could actually run
#:     off administrative income data, and wrong in a way that is correlated
#:     with household size and housing costs.
#: ``"highest_domestic_bill"``
#:     Largest shocked domestic bill first. Not an income test at all: it is the
#:     observable the *supplier* holds, and it is the only ranking a scheme
#:     administered through energy accounts could apply without new data.
#: ``"random"``
#:     No observability whatsoever. Admission by lottery among non-claimants.
#:     **Lower bound**, and the right benchmark for what "widening eligibility"
#:     achieves when the state cannot see who is poor.
ADMISSION_RULES: tuple[str, ...] = (
    "equivalised_ahc_income",
    "unequivalised_net_income",
    "highest_domestic_bill",
    "random",
)

#: What the state has to be able to see, per rule. Round-4 finding 3.
ADMISSION_RULE_REQUIREMENTS: dict[str, str] = {
    "equivalised_ahc_income": (
        "after-housing-costs income and household composition for every "
        "household OUTSIDE the benefit system, ranked without error — the "
        "capability this paper argues does not exist"
    ),
    "unequivalised_net_income": (
        "household net income only: no housing costs, no equivalisation. "
        "Closer to administrative income data, and mis-ranks large households "
        "and high-rent households"
    ),
    "highest_domestic_bill": (
        "nothing the state does not already have: the energy account's own "
        "bill. Not an income test — it admits big users, who are not the poor"
    ),
    "random": "nothing at all: admission by lottery among non-claimants",
}

#: The rule the headline eligibility arm uses. Deliberately the upper bound, so
#: the paper's comparison is stated as the best case and labelled as one.
DEFAULT_ADMISSION_RULE: str = "equivalised_ahc_income"

#: Seed for ``"random"`` admission, so the row is reproducible.
ADMISSION_RANDOM_SEED: int = 20260826


def admission_order(
    base: Baseline,
    cost: ShockCost,
    candidates: np.ndarray,
    rule: str = DEFAULT_ADMISSION_RULE,
    seed: int | None = None,
) -> np.ndarray:
    """Order ``candidates`` (household indices) for admission under ``rule``.

    Separated from :func:`widen_eligibility` so a new rule is a change in one
    place and so the ordering itself is testable without scoring anything.
    """
    if rule == "equivalised_ahc_income":
        key = np.asarray(base.equiv_income_ahc, dtype=float)[candidates]
    elif rule == "unequivalised_net_income":
        key = np.asarray(base.net_income, dtype=float)[candidates]
    elif rule == "highest_domestic_bill":
        key = -(base.energy + cost.domestic)[candidates]
    elif rule == "random":
        rng = np.random.default_rng(ADMISSION_RANDOM_SEED if seed is None else seed)
        key = rng.random(len(candidates))
    else:
        raise ValueError(f"unknown admission rule {rule!r}; expected {ADMISSION_RULES}")
    return candidates[np.argsort(key, kind="stable")]


def widen_eligibility(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float,
    max_eligible_share: float = 1.0,
    admission_rule: str = DEFAULT_ADMISSION_RULE,
    seed: int | None = None,
) -> np.ndarray:
    """Extend eligibility to non-claimants until the envelope is spent.

    The margin a real policymaker uses. A social tariff that has to spend more
    money does not discount 138% of the bill; it lets more households in — and
    the paper's own coverage finding (PolicyEngine UK captures 3.0m of a 7.2m
    Universal Credit caseload, so the modelled means-tested population is 15.7%
    of households against a plausible 30%+) says that eligibility, not
    generosity, is where the action is. Widening at the sponsor's own parameter
    is therefore the *feasible* way to reach a common envelope for a
    means-tested instrument, and it is the comparison the scaled row was
    standing in for.

    ``admission_rule`` (round-4 finding 3)
    --------------------------------------
    **Who gets let in, and what the state has to be able to see to let them
    in.** The default ranks non-claimants by equivalised AHC income, poorest
    first — perfect income observability outside the benefit system, which is
    the capability this paper argues does not exist. It is retained as the
    default because it is the correct **upper bound** on what widening can
    achieve, and it is labelled as one; :data:`ADMISSION_RULES` documents the
    weaker rules, and :func:`eligibility_admission_range` runs all of them so
    the paper reports a range rather than the best case.

    ``max_eligible_share`` caps how far the widening may go, as a share of
    weighted households. It defaults to 1.0 — no cap — because the finding is
    that a £150 payment run to a £5bn envelope reaches *everyone*, and
    suppressing that would hide it. Callers that want a means test to remain a
    means test can pass e.g. 0.5; either way the resulting share is persisted on
    the row and flagged when it reaches universality (round-3 finding 3).
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
    order = admission_order(base, cost, candidates, admission_rule, seed)
    running = np.cumsum(gain_if_eligible[order] * w[order])
    take = order[running <= remaining]
    if max_eligible_share < 1.0:
        w_total = w.sum()
        allowed = max_eligible_share * w_total - w[already].sum()
        cum_w = np.cumsum(w[take])
        take = take[cum_w <= max(allowed, 0.0)]
    out[take] = True
    return out


def score_policy_by_eligibility(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    max_eligible_share: float = 1.0,
    admission_rule: str = DEFAULT_ADMISSION_RULE,
    seed: int | None = None,
) -> tuple[PolicyScore, np.ndarray]:
    """Reach the common envelope by widening eligibility, not by scaling generosity.

    Only defined for :attr:`Policy.means_tested` instruments; the universal ones
    have no eligibility margin to widen. The generosity parameter is left at the
    sponsor's own value, so the row is always feasible and keeps the policy's
    real name.

    ``admission_rule`` says what the state has to be able to observe to run the
    widening (:data:`ADMISSION_RULES`). The default is the perfect-observability
    upper bound and the row is labelled as such; use
    :func:`eligibility_admission_range` for the whole range.
    """
    if not policy.means_tested:
        raise ValueError(
            f"policy {policy.key!r} is universal; it has no eligibility margin"
        )
    widened = widen_eligibility(
        base,
        cost,
        mt,
        policy,
        envelope_bn,
        max_eligible_share=max_eligible_share,
        admission_rule=admission_rule,
        seed=seed,
    )
    gain = policy.gain(base, cost, widened)
    w = base.weight
    share = float(w[widened].sum() / w.sum())
    # Round-3 finding 3: widening a £150 flat payment to a £5bn envelope puts
    # every household in. At that point the row is not the Warm Home Discount
    # and not a means-tested instrument; it is arithmetically the flat rebate,
    # with the same share-to-D1-3 and the same cost per pound. It is renamed
    # accordingly rather than left carrying a means-tested policy's name.
    label_used = policy.label
    if share >= UNIVERSAL_ELIGIBILITY_TOLERANCE:
        label_used = (
            f"{policy.label}, widened to universal eligibility: "
            f"{policy.parameter} held at {policy.stated_parameter:g} "
            f"({policy.parameter_units}) and paid to every household — "
            "no means test remains at this envelope"
        )
    # Round-4 finding 3: the admission rule is part of what the row IS, so it
    # travels with the label. A row admitted poorest-first on equivalised AHC
    # income is an upper bound and must say so wherever it is read.
    if admission_rule == DEFAULT_ADMISSION_RULE:
        label_used += (
            " [admission: poorest-first on equivalised AHC income — assumes "
            "perfect income observability outside the benefit system; UPPER "
            "BOUND]"
        )
    else:
        label_used += f" [admission: {admission_rule}]"
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
        eligible_share=share,
        label_used=label_used,
        feasible_max_cost_bn=policy.feasible_max_cost_bn(base, cost, mt),
        admission_rule=admission_rule,
        admission_rule_is_upper_bound=(admission_rule == DEFAULT_ADMISSION_RULE),
    )


def eligibility_admission_range(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    policy: Policy,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    rules: tuple[str, ...] = ADMISSION_RULES,
    random_seeds: tuple[int, ...] = (20260826, 7, 101, 2718, 31415),
) -> dict[str, Any]:
    """Score the eligibility arm under **every** admission rule, not the best one.

    **Round-4 finding 3.** The paper reported the eligibility arm offsetting
    26.6% of aggregate loss against the generosity arm's 5.2%, and that number
    is produced by admitting non-claimants in ascending order of equivalised
    AHC income — the state ranking every household outside the benefit system,
    poorest first, without error. The paper's own argument is that this
    capability does not exist. Reporting the arm at that rule alone states the
    upper bound as the result.

    Every rule in :data:`ADMISSION_RULES` is run here at the same envelope and
    the same generosity, and the spread is reported. ``"random"`` is additionally
    run over several seeds and its mean and range recorded, so the lower bound
    is not one lucky draw.

    Returns the per-rule statistics plus ``range`` — the min and max across
    rules of each headline measure — so the paper can write "between x and y,
    depending on what the state can observe" instead of "y".
    """
    per_rule: dict[str, Any] = {}
    for rule in rules:
        score = score_policy_by_eligibility(
            base, cost, mt, policy, envelope_bn, admission_rule=rule
        )[0]
        entry: dict[str, Any] = {
            "admission_rule": rule,
            "is_upper_bound": rule == DEFAULT_ADMISSION_RULE,
            "what_the_state_must_observe": ADMISSION_RULE_REQUIREMENTS[rule],
            "cost_bn": score.cost_bn,
            "eligible_share": score.eligible_share,
            "share_of_aggregate_loss_offset": score.share_of_aggregate_loss_offset,
            "share_to_bottom_three": score.share_to_bottom_three,
            "uncompensated_share_overall": score.uncompensated_share_overall,
            "mean_residual_loss_gbp": score.mean_residual_loss_gbp,
            "cost_per_pound_decile_one": score.cost_per_pound_decile_one,
        }
        if rule == "random":
            draws = [
                score_policy_by_eligibility(
                    base, cost, mt, policy, envelope_bn, admission_rule=rule, seed=seed
                )[0].share_of_aggregate_loss_offset
                for seed in random_seeds
            ]
            entry["random_draws"] = {
                "seeds": list(random_seeds),
                "share_of_aggregate_loss_offset": draws,
                "mean": float(np.mean(draws)),
                "min": float(np.min(draws)),
                "max": float(np.max(draws)),
                "std": float(np.std(draws)),
            }
        per_rule[rule] = entry

    measures = (
        "share_of_aggregate_loss_offset",
        "share_to_bottom_three",
        "uncompensated_share_overall",
        "eligible_share",
        "cost_bn",
    )
    ranges = {
        measure: {
            "min": min(v[measure] for v in per_rule.values()),
            "max": max(v[measure] for v in per_rule.values()),
            "upper_bound_rule_value": per_rule[DEFAULT_ADMISSION_RULE][measure],
            "argmin": min(per_rule, key=lambda k: per_rule[k][measure]),
            "argmax": max(per_rule, key=lambda k: per_rule[k][measure]),
        }
        for measure in measures
    }
    # Round-4: the default rule is the upper bound on OBSERVABILITY, and hence
    # on targeting — it admits the poorest first. It is NOT automatically the
    # best rule on every measure, and the paper should say so: most of the
    # modelled loss is motor fuel, which is not what a domestic-bill instrument
    # pays out on, so a rule that admits the biggest BILLS can offset more
    # aggregate loss than a rule that admits the lowest incomes. Which measures
    # the default actually wins is computed, not assumed.
    best_on = [
        measure
        for measure in ("share_of_aggregate_loss_offset", "share_to_bottom_three")
        if ranges[measure]["argmax"] == DEFAULT_ADMISSION_RULE
    ]
    return {
        "policy": policy.key,
        "envelope_bn": envelope_bn,
        "default_rule": DEFAULT_ADMISSION_RULE,
        "default_rule_maximises": best_on,
        "default_rule_does_not_maximise": [
            measure
            for measure in ("share_of_aggregate_loss_offset", "share_to_bottom_three")
            if measure not in best_on
        ],
        "upper_bound_is_on_observability_not_on_every_measure": (
            "The default rule assumes perfect income observability and so "
            "bounds what widening can achieve as TARGETING. It does not "
            "dominate every metric: most of the modelled loss is motor fuel, "
            "which a domestic-bill instrument does not pay out on, so a rule "
            "that admits the largest bills can offset more aggregate loss "
            "while reaching fewer poor households. Report the range, and say "
            "which measure is being ranked."
        ),
        "rules": list(rules),
        "by_rule": per_rule,
        "range": ranges,
        "note": (
            "Round-4 finding 3. The headline eligibility arm uses "
            f"{DEFAULT_ADMISSION_RULE!r}, which assumes the state can rank "
            "every non-claimant household by equivalised after-housing-costs "
            "income. That is the capability this paper argues does not exist, "
            "so that row is an UPPER BOUND. The paper must report the range "
            "across admission rules, with the default labelled as the best "
            "case and 'random' as the no-observability benchmark."
        ),
    }


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
    ``feasible_max``
        The instrument at its **own** feasible maximum, uncapped by any
        envelope: the true feasible maximum, and the only row down which
        feasible maxima are comparable.
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

    ``feasible_max`` and ``common_capped`` answer different questions and the
    pre-revision scorecard ran them together in a single column labelled
    "feasible maximum" (:data:`ROW_SEMANTICS`). Every row carries
    ``row_semantics`` saying which it is.

    Consumers should read ``envelope``, ``is_feasible`` and ``label_used``
    together: a conclusion about targeting drawn from a ``common_scaled`` row
    with ``is_feasible == False`` is a conclusion about a policy that cannot
    exist.
    """
    chosen = POLICIES if policies is None else policies
    out: list[PolicyScore] = []
    for policy in chosen.values():
        ceiling = policy.feasible_max_parameter(base, cost, mt)
        feasible_cost_bn = policy.feasible_max_cost_bn(base, cost, mt)
        out.append(
            score_policy(
                base,
                cost,
                mt,
                policy,
                implied_parameter=policy.stated_parameter,
                feasible_max_parameter=ceiling,
                is_feasible=policy.stated_parameter <= ceiling * (1 + 1e-12),
                feasible_max_cost_bn=feasible_cost_bn,
            )[0]
        )
        out.append(score_policy_at_feasible_max(base, cost, mt, policy, envelope_bn)[0])
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
            # One row, on the default admission rule, so the scorecard's shape
            # is unchanged. That rule is the perfect-observability UPPER BOUND
            # (round-4 finding 3): the row says so in ``label_used`` and in
            # ``admission_rule_is_upper_bound``, and the weaker rules are
            # scored and persisted by :func:`eligibility_admission_range`, which
            # ``policy_diagnostics`` writes out. A range belongs in a
            # diagnostics file, not in a table with one row per instrument.
            out.append(
                score_policy_by_eligibility(base, cost, mt, policy, envelope_bn)[0]
            )
    return out


# --------------------------------------------------------------------------
# Diagnostics the emitter needs and previously carried as literals
# --------------------------------------------------------------------------


def large_loser_outside_means_test(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    threshold_pct: float = 5.0,
) -> dict[str, float]:
    """Share of *large losers* sitting outside the modelled means-tested system.

    ``genLargeLoserOutsideMeansTest`` in the emitter was the last hardcoded
    literal in a paper whose reproduction appendix promises that every number is
    emitted mechanically from ``results/``. It is computed here, off the same
    arrays the scorecard uses, so it can be persisted like everything else.

    A large loser is a household with positive net income losing more than
    ``threshold_pct`` per cent of it to the shock. The statistic is the weighted
    share of those households not on any *modelled* means-tested benefit.

    **Its ceiling is reported beside it.** A referee's objection is that with a
    modelled means-tested population of only ~15.7% of households the statistic
    can be no lower than ``100 - 15.7`` even if the means test hit every single
    large loser, so a value near that ceiling carries little information beyond
    the coverage shortfall the paper already reports as a finding. That is a
    fair reading, and ``ceiling_pct`` / ``headroom_pct`` make it checkable
    rather than something a reader has to reconstruct: ``headroom_pct`` is the
    only part of the statistic that is about targeting rather than about
    coverage.
    """
    w = base.weight
    income = np.asarray(base.net_income, dtype=float)
    loss = np.asarray(cost.total, dtype=float)
    mt = np.asarray(mt, dtype=bool)
    pos = income > 0
    burden = np.zeros_like(loss)
    np.divide(100.0 * loss, income, out=burden, where=pos)
    heavy = pos & (burden > threshold_pct)
    w_heavy = w[heavy].sum()
    w_total = w.sum()
    mt_share = float(w[mt].sum() / w_total) if w_total > 0 else float("nan")
    share_out = float(w[heavy & ~mt].sum() / w_heavy) if w_heavy > 0 else float("nan")
    ceiling = 100.0 * (1.0 - mt_share)
    return {
        "threshold_pct": float(threshold_pct),
        "large_losers_m": float(w_heavy / 1e6),
        "large_loser_share_of_households": (
            float(w_heavy / w_total) if w_total > 0 else float("nan")
        ),
        "share_outside_means_test_pct": 100.0 * share_out,
        "means_tested_share_of_households_pct": 100.0 * mt_share,
        "ceiling_pct": ceiling,
        "headroom_pct": 100.0 * share_out - ceiling,
        "note": (
            "Modelled means-tested population only. The statistic cannot fall "
            "below its ceiling of 100 minus the means-tested share of "
            "households, so it is near-tautological when it sits close to that "
            "ceiling; `headroom_pct` is the part that is about targeting. "
            "Persisted so the emitter carries no literal, but a referee argues "
            "it would be better dropped from the paper than qualified."
        ),
    }


def policy_diagnostics(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    policies: dict[str, Policy] | None = None,
) -> dict[str, Any]:
    """Everything about the policy block that is not a per-row scorecard field.

    Written to ``results/`` so the paper can state, mechanically:

    * the true feasible maximum cost of each instrument, beside what it can
      absorb inside the envelope (round-3 finding 1);
    * what each arm of the common-envelope comparison **actually spends**, so
      the prose can stop claiming the arms spend the same £5bn (finding 4);
    * the JRF block's reference quantity on both bases (finding 5);
    * the large-loser statistic and its ceiling (finding 8).

    No headline number is embedded: every figure is computed from the baseline
    and shock passed in.
    """
    chosen = POLICIES if policies is None else policies
    per_policy: dict[str, Any] = {}
    for key, policy in chosen.items():
        stated_bn = policy.stated_cost_simulated_bn(base, cost, mt)
        feasible_bn = policy.feasible_max_cost_bn(base, cost, mt)
        entry: dict[str, Any] = {
            "label": policy.label,
            "parameter": policy.parameter,
            "parameter_units": policy.parameter_units,
            "stated_parameter": policy.stated_parameter,
            "feasible_max_parameter": policy.feasible_max_parameter(base, cost, mt),
            "stated_cost_simulated_bn": stated_bn,
            "feasible_max_cost_bn": feasible_bn,
            "absorbable_within_envelope_bn": min(envelope_bn, feasible_bn),
            "saturates_below_envelope": bool(feasible_bn < envelope_bn),
            "means_tested": policy.means_tested,
        }
        entry["feasible_max_rule"] = policy.feasible_max_rule
        arms: dict[str, float] = {
            "generosity_arm_spend_bn": min(envelope_bn, feasible_bn),
        }
        if policy.means_tested:
            elig = score_policy_by_eligibility(base, cost, mt, policy, envelope_bn)[0]
            arms["eligibility_arm_spend_bn"] = elig.cost_bn
            arms["eligibility_arm_eligible_share"] = elig.eligible_share
            arms["eligibility_arm_is_universal"] = elig.eligibility_is_universal
            arms["eligibility_arm_admission_rule"] = elig.admission_rule
            arms["eligibility_arm_admission_rule_is_upper_bound"] = (
                elig.admission_rule_is_upper_bound
            )
            entry["admission_rules"] = eligibility_admission_range(
                base, cost, mt, policy, envelope_bn
            )
        arms["envelope_bn"] = envelope_bn
        arms["arms_spend_the_same"] = bool(
            all(
                abs(v - envelope_bn) <= 0.005 * envelope_bn
                for k, v in arms.items()
                if k.endswith("_spend_bn")
            )
        )
        entry["envelope_arms"] = arms
        per_policy[key] = entry

    return {
        "envelope_bn": envelope_bn,
        "row_semantics": ROW_SEMANTICS,
        "by_policy": per_policy,
        "jrf_reference_quantities": jrf_reference_quantities(base),
        # Round-4 findings 1, 2 and 6.
        "feasible_max_identity": feasible_max_identity(base, cost, mt, chosen),
        "flat_payment_ceilings": flat_payment_ceilings(base, cost, mt),
        "jrf_costing_gap": jrf_costing_gap(base, cost, mt),
        "large_loser_outside_means_test": large_loser_outside_means_test(
            base, cost, mt
        ),
        "note": (
            "`feasible_max_cost_bn` is a property of the instrument and ignores "
            "the envelope; `absorbable_within_envelope_bn` is the budget "
            "constraint. A column that mixes them is not a feasible-maximum "
            "column."
        ),
    }


def write_policy_diagnostics(
    base: Baseline,
    cost: ShockCost,
    mt: np.ndarray,
    envelope_bn: float = COMMON_ENVELOPE_BN,
    path: str | Path | None = None,
    policies: dict[str, Policy] | None = None,
) -> Path:
    """Persist :func:`policy_diagnostics` as JSON."""
    payload = policy_diagnostics(base, cost, mt, envelope_bn, policies)
    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "results" / "policy_diagnostics.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path

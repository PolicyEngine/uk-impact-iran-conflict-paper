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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from uk_iran_conflict.incidence import (
    DOMESTIC_FUEL_VAT_RATE,
    Baseline,
    ShockCost,
    wmean,
    wsum,
)

#: Households on a means-tested benefit are the social tariff's target
#: population. The anchor fact in the policy debate is that roughly 40% of
#: households struggling to heat their home are *not* in this group (JRF,
#: 9 Apr 2026), which is what the uncompensated-loser metric quantifies.
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


def means_tested_flag(dataset: str, period: int = 2026) -> np.ndarray:
    """Household-level indicator of any means-tested benefit receipt."""
    from policyengine_uk import Microsimulation  # noqa: PLC0415

    sim = Microsimulation(dataset=dataset)
    flag: np.ndarray | None = None
    for name in MEANS_TESTED_VARIABLES:
        try:
            got = np.asarray(sim.calculate(name, period, map_to="household"))
        except Exception:  # noqa: BLE001 — a missing benefit is not fatal
            continue
        flag = got > 0 if flag is None else (flag | (got > 0))
    if flag is None:
        raise RuntimeError("no means-tested benefit variables resolved")
    return flag


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

    JRF (9 Apr 2026): a discounted rate on the first 50% of typical consumption
    plus a per-child allowance, ~£5bn, fully offsetting the cost rise for
    deciles 1-3. Universal by design — that is the point, since it reaches the
    ~40% of struggling households outside the means-tested system.

    Typical consumption is the weighted median domestic bill, so the block is a
    fixed quantity rather than a proportion of each household's own bill (a
    proportional discount would mechanically pay most to the biggest users).
    """
    from uk_iran_conflict.incidence import wquantile  # noqa: PLC0415

    typical = wquantile(base.energy, base.weight, 0.5)
    block_value = block_share * typical
    covered = np.minimum(base.energy, block_value)
    children = np.clip(base.people - 2, 0, None)
    return discount * covered * _shock_rate(base, cost) + per_child * children


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


def score_policy(
    base: Baseline, cost: ShockCost, mt: np.ndarray, policy: Policy
) -> tuple[PolicyScore, np.ndarray]:
    """Score ``policy`` against the shock, returning the gain array too.

    ``uncompensated`` counts households still worse off after the policy — the
    metric decile averages hide and the one the targeting argument turns on.
    """
    gain = policy.gain(base, cost, mt)
    w = base.weight
    loss = cost.total
    net = loss - gain
    losers = loss > 0

    total_gain = wsum(gain, w)
    bottom3 = base.decile <= 3
    d1 = base.decile == 1
    d1_gain = wmean(gain[d1], w[d1]) if w[d1].sum() > 0 else float("nan")

    by_decile: dict[int, float] = {}
    for d in range(1, 11):
        sel = (base.decile == d) & losers
        wd = w[sel]
        if wd.sum() <= 0:
            continue
        by_decile[d] = float(wd[net[sel] > 0].sum() / wd.sum())

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
        ),
        gain,
    )

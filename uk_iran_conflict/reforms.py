"""PolicyEngine UK reform builders: the energy shock and the policy responses.

Two families of reform live here.

**The shock** (:func:`build_shock_reform`) moves *prices*, not policy: the
Ofgem default-tariff cap path (``gov/ofgem/energy_price_cap``) and the DESNZ
pump-price parameters (``household/consumption/fuel/prices/{petrol,diesel}``).
Gas and electricity are shocked **asymmetrically** — they are separate
NEED-calibrated variables in PolicyEngine UK and, because gas sets the
marginal electricity price only ~85% of the time, a given wholesale gas move
does not pass through to the two fuels in the same proportion. A blunt
percentage lever, ``gov/contrib/policyengine/economy/energy_bills``, is
available via :func:`build_blunt_energy_bills_reform` and is used for
*robustness only* — it cannot separate gas from electricity and so cannot
support the paper's central specification.

**The policy responses** (:data:`POLICY_REFORMS`) are the four options in the
Autumn Budget 2026 debate, plus the IPPR windfall-funded rebate. Each builder
carries its source and its real-world cost estimate in its docstring.

Every builder returns a ``policyengine_core`` ``Reform`` and imports
PolicyEngine lazily, so this module imports cleanly — and is unit-testable —
without microdata or a Hugging Face token.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# --- parameter paths ------------------------------------------------------
# Held as constants so tests can assert on them without importing PolicyEngine.

OFGEM_CAP = "gov.ofgem.energy_price_cap"
PETROL_PRICE = "household.consumption.fuel.prices.petrol"
DIESEL_PRICE = "household.consumption.fuel.prices.diesel"
ENERGY_BILLS_LEVER = "gov.contrib.policyengine.economy.energy_bills"
EPG_SUBSIDY = "gov.treasury.price_cap_subsidy"
EPG_CONSUMPTION_LEVEL = f"{EPG_SUBSIDY}.monthly_epg_consumption_level"
EPG_RATE = f"{EPG_SUBSIDY}.monthly_epg_rate"
WHD_AMOUNT = "gov.dwp.warm_home_discount.amount"
VAT_REDUCED_RATE = "gov.hmrc.vat.reduced_rate"

#: Analysis window. The shock bites from the Oct-2026 cap onward.
DEFAULT_PERIOD = 2026
FULL_YEAR = f"year:{DEFAULT_PERIOD}:1"


def _reform_class():
    """Import ``Reform`` lazily so this module is importable without data."""
    from policyengine_core.reforms import Reform  # noqa: PLC0415

    return Reform


def _from_dict(changes: Mapping[str, Mapping[str, Any]], name: str):
    """Build a ``Reform`` from a ``{parameter: {period: value}}`` mapping."""
    reform = _reform_class().from_dict(dict(changes), country_id="uk")
    reform.name = name  # type: ignore[attr-defined]
    return reform


# --------------------------------------------------------------------------
# The shock
# --------------------------------------------------------------------------


def quarterly_periods(year: int = DEFAULT_PERIOD) -> tuple[str, ...]:
    """PolicyEngine period strings for the four Ofgem cap quarters of ``year``.

    The cap resets on 1 Jan / 1 Apr / 1 Jul / 1 Oct; the shock enters as a
    quantised quarterly step path because that is how a wholesale move
    mechanically reaches households (6-9 month lag, wholesale ~40-50% of a
    dual-fuel bill).
    """
    return tuple(
        f"{year}-{month:02d}-01.{year}-{month + 2:02d}-31" for month in (1, 4, 7, 10)
    )


def _cap_changes(cap_path: Iterable[float], year: int) -> dict[str, float]:
    """Map a four-element quarterly cap path onto PolicyEngine period keys."""
    path = [float(v) for v in cap_path]
    periods = quarterly_periods(year)
    if len(path) != len(periods):
        raise ValueError(
            f"cap path has {len(path)} quarters, expected {len(periods)} for {year}"
        )
    return {period: value for period, value in zip(periods, path, strict=True)}


def cap_levels(scenario: Any) -> tuple[float, ...]:
    """Quarterly Ofgem cap levels (£/yr) implied by ``scenario``.

    Reads :attr:`uk_iran_conflict.scenarios.Scenario.cap_path`, a tuple of
    ``CapStep`` records carrying ``cap_gbp``. A plain sequence of floats is
    also accepted so callers can hand-build a path in tests.
    """
    path = _scenario_field(scenario, "cap_path", "ofgem_cap_path")
    return tuple(float(getattr(step, "cap_gbp", step)) for step in path)


def retail_factors(scenario: Any) -> tuple[float, float]:
    """The (gas, electricity) retail shock multipliers for ``scenario``.

    Reads :attr:`Scenario.retail_shock`, whose ``gas_factor`` and
    ``electricity_factor`` are the multipliers applied to PolicyEngine UK's
    separate ``gas_consumption`` and ``electricity_consumption`` variables.
    **Their inequality is the point** — gas sets the marginal electricity price
    only ~85% of the time, so the two must never be collapsed into
    ``domestic_energy_consumption``.
    """
    shock = getattr(scenario, "retail_shock", None)
    if shock is not None:
        return float(shock.gas_factor), float(shock.electricity_factor)
    # TODO(contract): fallback for a scenario object exposing bare factors.
    return (
        float(_scenario_field(scenario, "gas_shock_factor", default=1.0)),
        float(_scenario_field(scenario, "electricity_shock_factor", default=1.0)),
    )


def pump_price_factors(scenario: Any) -> tuple[float, float]:
    """The (petrol, diesel) pump-price multipliers for ``scenario``."""
    pump = getattr(scenario, "pump", None)
    if pump is not None:
        return 1.0 + float(pump.petrol_pct_change), 1.0 + float(pump.diesel_pct_change)
    return (
        1.0 + float(_scenario_field(scenario, "petrol_price_change", default=0.0)),
        1.0 + float(_scenario_field(scenario, "diesel_price_change", default=0.0)),
    )


def build_shock_reform(scenario: Any, year: int = DEFAULT_PERIOD):
    """Construct the price shock for one macro scenario.

    Shocks, in order of the pipeline's Step 1:

    1. ``gov/ofgem/energy_price_cap`` — the quarterly cap path the scenario
       derives from its wholesale gas move (Cornwall Insight, 19 Aug 2026:
       Oct-26 cap £1,729, +4%; JRF 9 Apr 2026: wholesale gas +65% -> ~£288 bill
       rise). The scenario has already applied the 6-9 month wholesale-to-retail
       lag, the ~40-50% wholesale share of a dual-fuel bill, and the quarterly
       quantisation.
    2. ``household/consumption/fuel/prices/{petrol,diesel}`` — DESNZ pump
       prices (observed 2026 peak: petrol +20%, diesel +36%).

    Gas and electricity are carried **asymmetrically** on the returned reform as
    ``gas_shock_factor`` and ``electricity_shock_factor``, so the runner can
    apply them to PolicyEngine UK's separate ``gas_consumption`` and
    ``electricity_consumption`` variables rather than to the dual-fuel
    aggregate. The cap parameter itself is a single scalar in PolicyEngine UK,
    so the scenario's own gas-weighted dual-fuel re-aggregation is used there.
    """
    gas_factor, elec_factor = retail_factors(scenario)
    petrol_factor, diesel_factor = pump_price_factors(scenario)

    changes: dict[str, dict[str, Any]] = {
        OFGEM_CAP: _cap_changes(cap_levels(scenario), year)
    }
    reform = _from_dict(changes, name=f"shock_{_scenario_name(scenario)}")
    reform.gas_shock_factor = gas_factor  # type: ignore[attr-defined]
    reform.electricity_shock_factor = elec_factor  # type: ignore[attr-defined]
    reform.petrol_price_factor = petrol_factor  # type: ignore[attr-defined]
    reform.diesel_price_factor = diesel_factor  # type: ignore[attr-defined]
    # Pump prices are a *relative* move on the DESNZ parameter, which
    # ``Reform.from_dict`` (absolute values only) cannot express; the
    # multiplier reform is composed on top.
    return compose(
        reform, build_relative_price_reform(petrol_factor, diesel_factor, year)
    )


#: Share of household *final energy* consumption that is gas (DESNZ; the
#: highest in the G7). Used to blend the two fuel-specific shock factors into
#: the single scalar dual-fuel cap parameter.
GAS_SHARE_OF_HOUSEHOLD_ENERGY = 0.62


def _blend_cap_factor(gas_factor: float, electricity_factor: float) -> float:
    """Consumption-weighted blend of the gas and electricity shock factors.

    Used only by the blunt robustness lever, which cannot carry two fuels.
    The canonical re-aggregation lives in
    :attr:`uk_iran_conflict.scenarios.Scenario.cap_path`, which uses that
    module's ``gas_share_of_dual_fuel_bill``; this constant is the *final
    energy* share (DESNZ), a different quantity from the *bill* share.
    """
    w = GAS_SHARE_OF_HOUSEHOLD_ENERGY
    return w * gas_factor + (1.0 - w) * electricity_factor


def build_relative_price_reform(
    petrol_multiplier: float, diesel_multiplier: float, year: int = DEFAULT_PERIOD
):
    """Scale DESNZ pump prices by multipliers rather than setting levels.

    Uses ``Reform``'s class-based form so the shocked price is derived from the
    *baseline* parameter value, which keeps the scenario definitions in
    percentage terms (petrol +20%, diesel +36% at the observed 2026 peak)
    rather than requiring a hard-coded pence-per-litre level.
    """
    Reform = _reform_class()

    def modify_parameters(parameters):
        period = f"{year}-01-01"
        for path, multiplier in (
            (PETROL_PRICE, petrol_multiplier),
            (DIESEL_PRICE, diesel_multiplier),
        ):
            node = parameters
            for part in path.split("."):
                node = getattr(node, part)
            node.update(
                start=_instant(period),
                stop=_instant(f"{year + 1}-01-01"),
                value=node(period) * multiplier,
            )
        return parameters

    class _PriceShock(Reform):
        def apply(self):
            self.modify_parameters(modify_parameters)

    _PriceShock.name = f"pump_prices_p{petrol_multiplier:.3f}_d{diesel_multiplier:.3f}"
    return _PriceShock


def _instant(text: str):
    from policyengine_core.periods import instant  # noqa: PLC0415

    return instant(text)


def build_blunt_energy_bills_reform(pct_increase: float, year: int = DEFAULT_PERIOD):
    """ROBUSTNESS ONLY: raise all energy spending by a flat percentage.

    ``gov/contrib/policyengine/economy/energy_bills`` is a single "raise energy
    spending by this percentage" lever (default 0). It is blunt in exactly the
    way the paper argues against: it cannot move gas and electricity
    asymmetrically, and it bypasses the Ofgem cap mechanics (and hence the
    6-9 month wholesale-to-retail lag) entirely. Reported only as a robustness
    check against :func:`build_shock_reform`; never the central specification.
    """
    return _from_dict(
        {ENERGY_BILLS_LEVER: {FULL_YEAR: float(pct_increase)}},
        name=f"blunt_energy_bills_{pct_increase:+.2f}",
    )


# --------------------------------------------------------------------------
# The policy responses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResponse:
    """A costed policy response, with its provenance."""

    name: str
    label: str
    source: str
    cost_estimate_bn: float
    builder: Callable[..., Any]

    def build(self, year: int = DEFAULT_PERIOD):
        """Construct the PolicyEngine reform for this response."""
        return self.builder(year=year)


def build_social_tariff(
    year: int = DEFAULT_PERIOD,
    discount_rate: float = 0.20,
    consumption_level: float = 1.0,
):
    """Means-tested social tariff: a discount for benefit-receiving households.

    Reuses the Energy Price Guarantee code path
    (``gov/treasury/price_cap_subsidy/``) — a per-unit subsidy against the
    Ofgem cap — with an eligibility condition restricting it to households in
    receipt of a means-tested benefit.

    Source: Reeves has ruled out universal support and signalled income-based
    help (a social tariff); Resolution Foundation, *Macro Policy Outlook Q2*,
    22 Apr 2026, recommends targeted temporary bill discounts rather than a
    new EPG (~£20bn/yr). JRF (9 Apr 2026) rejects a social tariff as
    infeasible for winter 2026 — which is exactly the design tension the paper
    scores. The anchor counter-statistic: **~40% of households struggling to
    heat their home are not on means-tested benefits**, so this reform is the
    one expected to leave the most uncompensated losers within each decile.

    Real-world cost estimate: **~£3bn** at a 20% discount on cap-level
    consumption for the means-tested-benefit population.
    """
    return _from_dict(
        {
            EPG_RATE: {FULL_YEAR: float(discount_rate)},
            EPG_CONSUMPTION_LEVEL: {FULL_YEAR: float(consumption_level)},
            f"{EPG_SUBSIDY}.means_tested_only": {FULL_YEAR: True},
        },
        name="social_tariff",
    )


def build_jrf_discounted_block(
    year: int = DEFAULT_PERIOD,
    block_share: float = 0.50,
    block_discount: float = 0.50,
    per_child_allowance: float = 0.15,
):
    """JRF universal discounted block: 50% of typical consumption, discounted.

    A two-tier version of ``monthly_epg_consumption_level``: the first
    ``block_share`` of typical consumption is supplied at a discounted rate
    (``block_discount``), the remainder at the capped rate, plus a per-child
    allowance that widens the discounted block for families.

    Source: JRF (Moore, Cook), 9 Apr 2026. Wholesale gas +65%, ~£288 bill
    rise; the discounted block **fully offsets the cost rise for deciles 1-3**
    and is universal, so it reaches the ~40% of struggling households outside
    means-tested benefits. JRF explicitly calls the EPG regressive in cash
    terms at £23bn and proposes this as the affordable alternative.
    Cf. Fetzer, Gazze & Bishop (2024, *Economic Policy*), who propose a
    two-tier tariff on 22.2m EPCs plus NEED meter data.

    Real-world cost estimate: **~£5bn**.
    """
    return _from_dict(
        {
            EPG_CONSUMPTION_LEVEL: {FULL_YEAR: float(block_share)},
            EPG_RATE: {FULL_YEAR: float(block_discount)},
            f"{EPG_SUBSIDY}.per_child_consumption_allowance": {
                FULL_YEAR: float(per_child_allowance)
            },
        },
        name="jrf_discounted_block",
    )


def build_whd_expansion(year: int = DEFAULT_PERIOD, amount: float = 150.0):
    """Warm Home Discount expansion: a £150 rebate, wider eligibility.

    A pure parameter change on ``gov.dwp.warm_home_discount.amount``.

    Source: the WHD qualifying date was 23 Aug 2026; the rebate is £150 and is
    now automatic in England and Wales with the property-cost test scrapped,
    extending it to **+2.7m households**.

    Real-world cost estimate: **~£1.0bn** (roughly 6.5m households x £150).
    """
    return _from_dict({WHD_AMOUNT: {FULL_YEAR: float(amount)}}, name="whd_expansion")


def build_vat_zero_rating(year: int = DEFAULT_PERIOD):
    """VAT on domestic fuel and power cut from 5% to 0%.

    A parameter change on ``gov.hmrc.vat.reduced_rate``.

    Source: the reduced rate on domestic fuel and power (VATA 1994 Sch 7A
    Group 1), the standing "cut VAT on energy bills" option in the Autumn
    Budget 2026 debate; costed against HMRC's tax-relief statistics.

    NOTE on the PolicyEngine gap: ``household_tax.py`` includes ``vat_change``,
    not ``vat`` — baseline VAT is never deducted from net income. That delta-
    only treatment works *in this reform's favour*: scoring a VAT cut needs
    only the delta, which is exactly what PolicyEngine computes. The remaining
    caveat is the 0.38 coverage grossing factor
    (``microdata_vat_coverage.yaml``, following IFS TAXBEN; issue #352).

    Real-world cost estimate: **~£1.7bn** (HMRC tax-relief statistics scale
    for the domestic fuel and power reduced rate).
    """
    return _from_dict({VAT_REDUCED_RATE: {FULL_YEAR: 0.0}}, name="vat_zero_rating")


def build_ippr_rebate(year: int = DEFAULT_PERIOD, amount: float = 183.0):
    """IPPR flat per-household rebate of £183, funded by network windfalls.

    Implemented as a flat household-level credit through the 2022 energy bills
    rebate code path (``gov/treasury/energy_bills_rebate/``), which already
    delivers a per-household energy credit. Financing — clawing back network-
    company windfall profits — is scored **separately**, so the reform as run
    here is the household-side leg only.

    Source: IPPR, Apr 2026 — claw back network-company windfalls to fund a
    £183 household rebate.

    Real-world cost estimate: **~£5.0bn** (≈27.5m households x £183).
    """
    return _from_dict(
        {
            "gov.treasury.energy_bills_rebate.energy_bills_credit.amount": {
                FULL_YEAR: float(amount)
            }
        },
        name="ippr_rebate",
    )


#: The five scored policy responses, keyed by short name.
POLICY_REFORMS: dict[str, PolicyResponse] = {
    "social_tariff": PolicyResponse(
        name="social_tariff",
        label="Means-tested social tariff",
        source="Reeves/HMT signalled income-based help; RF Q2 2026",
        cost_estimate_bn=3.0,
        builder=build_social_tariff,
    ),
    "jrf_block": PolicyResponse(
        name="jrf_block",
        label="JRF universal discounted block",
        source="JRF (Moore, Cook), 9 Apr 2026",
        cost_estimate_bn=5.0,
        builder=build_jrf_discounted_block,
    ),
    "whd_expansion": PolicyResponse(
        name="whd_expansion",
        label="Warm Home Discount expansion (£150)",
        source="DESNZ WHD reform, qualifying date 23 Aug 2026",
        cost_estimate_bn=1.0,
        builder=build_whd_expansion,
    ),
    "vat_zero": PolicyResponse(
        name="vat_zero",
        label="VAT on domestic fuel 5% -> 0%",
        source="HMRC reduced_rate.yaml; HMRC tax relief statistics",
        cost_estimate_bn=1.7,
        builder=build_vat_zero_rating,
    ),
    "ippr_rebate": PolicyResponse(
        name="ippr_rebate",
        label="IPPR flat £183 household rebate",
        source="IPPR, Apr 2026 (network windfall clawback)",
        cost_estimate_bn=5.0,
        builder=build_ippr_rebate,
    ),
}


def build_policy_reform(name: str, year: int = DEFAULT_PERIOD):
    """Build one named policy response, raising a clear error on a typo."""
    if name not in POLICY_REFORMS:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(POLICY_REFORMS)}")
    return POLICY_REFORMS[name].build(year=year)


#: Metadata attributes the runner reads off a shock reform. ``compose`` must
#: propagate these or the gas/electricity asymmetry is lost (see ``compose``).
SHOCK_METADATA_ATTRS = (
    "gas_shock_factor",
    "electricity_shock_factor",
    "petrol_price_factor",
    "diesel_price_factor",
)


def compose(*reforms):
    """Compose several reforms into one (shock + policy)."""
    parts = [r for r in reforms if r is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    from policyengine_core.reforms import Reform  # noqa: PLC0415

    class _Composed(Reform):
        def apply(self):
            for part in parts:
                self.apply_reform(part)

    _Composed.name = "+".join(getattr(p, "name", "reform") for p in parts)
    # Carry forward the shock-factor metadata the runner reads off the reform.
    # ``Reform`` subclassing does not inherit these, so composing a shock with a
    # policy would otherwise silently drop the gas/electricity asymmetry and the
    # pump-price multipliers — the modelling choice the paper turns on.
    for part in parts:
        for attr in SHOCK_METADATA_ATTRS:
            if hasattr(part, attr):
                setattr(_Composed, attr, getattr(part, attr))
    return _Composed


# --------------------------------------------------------------------------
# scenario-field access (defensive against the scenarios.py contract)
# --------------------------------------------------------------------------

_MISSING = object()


def _scenario_field(scenario: Any, *names: str, default: Any = _MISSING) -> Any:
    """Read the first present attribute of ``scenario`` among ``names``.

    TODO(contract): ``uk_iran_conflict.scenarios`` is written by a sibling; the
    expected field names are ``oil_price_change``, ``gas_wholesale_change``,
    ``ofgem_cap_path``, ``petrol_price_change``, ``diesel_price_change``,
    ``gas_shock_factor``, ``electricity_shock_factor`` and ``source``. If those
    names drift, add the alias here rather than at every call site.
    """
    for name in names:
        value = getattr(scenario, name, _MISSING)
        if value is not _MISSING:
            return value
    if default is not _MISSING:
        return default
    raise AttributeError(
        f"scenario {scenario!r} has none of {names}; expected the "
        "uk_iran_conflict.scenarios contract"
    )


def _scenario_name(scenario: Any) -> str:
    return str(_scenario_field(scenario, "name", "key", default="scenario"))

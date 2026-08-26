"""Run baseline / shocked / shocked+policy simulations and summarise deltas.

Three simulations per (scenario, policy) cell:

* **baseline** — PolicyEngine UK on FRS 2023-24, uprated on OBR EFO Mar 2026;
* **shocked** — baseline plus the price shock from
  :func:`uk_iran_conflict.reforms.build_shock_reform`, optionally with a
  consumption response from :mod:`uk_iran_conflict.elasticity` (the main
  specification is zero elasticity — the Deaton first-order approximation,
  and an explicit upper bound on the loss);
* **shocked+policy** — the above plus one of the five scored responses.

The summary deliberately mirrors PolicyEngine's own output set: budgetary,
income-decile average and relative impacts, **intra-decile winners and losers**
(the paper's distinctive uncompensated-losers metric — Cronin, Fullerton &
Sexton 2019; Sallee 2019; Douenne 2020), poverty across all four UK measures
(relative/absolute x BHC/AHC), Gini and top-1%/top-10%/bottom-50% shares, and
constituency aggregates off the 650-seat weight sets.

PolicyEngine imports are lazy throughout: this module imports and its dataclass
is constructible without microdata or a Hugging Face token.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from uk_iran_conflict.reforms import build_policy_reform, build_shock_reform, compose

# --- the four UK poverty measures ----------------------------------------
POVERTY_VARIABLES: dict[str, str] = {
    "relative_bhc": "in_relative_poverty_bhc",
    "relative_ahc": "in_relative_poverty_ahc",
    "absolute_bhc": "in_absolute_poverty_bhc",
    "absolute_ahc": "in_absolute_poverty_ahc",
}

#: Income concept used for ranking, deciles and inequality: HBAI cash
#: disposable income, equivalised, matching the ``in_*_poverty_*`` concept.
INCOME_VARIABLE = "household_net_income"
EQUIV_INCOME_VARIABLE = "equiv_hbai_household_net_income"

#: A change smaller than this share of baseline net income counts as "no
#: material change" in the intra-decile winners/losers split. PolicyEngine's
#: ``IntraDecileImpact`` uses the same 5% band.
INTRA_DECILE_BAND = 0.05


@dataclass(frozen=True)
class ScenarioResult:
    """Structured summary of one (scenario, policy) cell."""

    scenario: str
    policy: str | None
    period: int
    elasticity: str

    # budgetary
    exchequer_cost: float
    household_energy_spend_change: float

    # distribution (decile -> value); deciles of baseline equivalised income
    decile_average_change: dict = field(default_factory=dict)
    decile_relative_change: dict = field(default_factory=dict)
    #: decile -> {"gain_more_5", "gain_less_5", "no_change", "lose_less_5",
    #: "lose_more_5"} population shares — the uncompensated-losers metric
    intra_decile: dict = field(default_factory=dict)

    # poverty: measure -> change in headcount rate
    poverty_change: dict = field(default_factory=dict)
    deep_poverty_change: dict = field(default_factory=dict)

    # inequality
    gini_baseline: float = 0.0
    gini_reform: float = 0.0
    top_one_percent_share_baseline: float = 0.0
    top_one_percent_share_reform: float = 0.0
    top_ten_percent_share_baseline: float = 0.0
    top_ten_percent_share_reform: float = 0.0
    bottom_fifty_percent_share_baseline: float = 0.0
    bottom_fifty_percent_share_reform: float = 0.0

    # geography: GSS code -> aggregates over the 650-seat weight sets
    constituency_average_change: dict = field(default_factory=dict)
    constituency_relative_change: dict = field(default_factory=dict)

    # provenance
    source: str = ""


# --------------------------------------------------------------------------
# weighted statistics (pure numpy — unit-testable without PolicyEngine)
# --------------------------------------------------------------------------


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini coefficient, bottom-coded at zero.

    Negative incomes push the Gini above 1, so values are clipped at zero
    first — the same convention as uk-ai-study.
    """
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cv = np.cumsum(v * w)
    if cv[-1] == 0:
        return 0.0
    return float(1 - 2 * np.sum((cv - v * w / 2) * w) / (cv[-1] * cw[-1]))


def weighted_quantile_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Population share at or below each unit, in [0, 1]."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = cum / cum[-1]
    return ranks


def deciles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted decile (1-10) of ``values``."""
    ranks = weighted_quantile_ranks(values, weights)
    return np.clip(np.ceil(ranks * 10).astype(int), 1, 10)


def top_share(values: np.ndarray, weights: np.ndarray, share: float) -> float:
    """Income share of the top ``share`` (e.g. 0.01) of the distribution."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    weights = np.asarray(weights, dtype=float)
    ranks = weighted_quantile_ranks(values, weights)
    mask = ranks > (1.0 - share)
    total = float((values * weights).sum())
    if total == 0:
        return 0.0
    return float((values[mask] * weights[mask]).sum() / total)


def bottom_share(values: np.ndarray, weights: np.ndarray, share: float) -> float:
    """Income share of the bottom ``share`` of the distribution."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    weights = np.asarray(weights, dtype=float)
    ranks = weighted_quantile_ranks(values, weights)
    mask = ranks <= share
    total = float((values * weights).sum())
    if total == 0:
        return 0.0
    return float((values[mask] * weights[mask]).sum() / total)


def intra_decile_breakdown(
    baseline_income: np.ndarray,
    reform_income: np.ndarray,
    weights: np.ndarray,
    decile: np.ndarray,
    band: float = INTRA_DECILE_BAND,
) -> dict[int, dict[str, float]]:
    """Population shares in each gain/loss band within each decile.

    This is the paper's headline horizontal-incidence statistic: the share of
    each decile that *loses* despite the decile-average effect. Bands follow
    PolicyEngine's ``IntraDecileImpact``: gain/lose more than ``band`` of
    baseline net income, gain/lose less than ``band``, or no change.
    """
    baseline_income = np.asarray(baseline_income, dtype=float)
    reform_income = np.asarray(reform_income, dtype=float)
    weights = np.asarray(weights, dtype=float)
    decile = np.asarray(decile, dtype=int)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(
            baseline_income > 0,
            (reform_income - baseline_income) / baseline_income,
            0.0,
        )
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)

    out: dict[int, dict[str, float]] = {}
    for d in range(1, 11):
        in_decile = decile == d
        w = weights[in_decile]
        r = rel[in_decile]
        total = float(w.sum())
        if total == 0:
            out[d] = {k: 0.0 for k in _BANDS}
            continue
        out[d] = {
            "gain_more_5": float(w[r > band].sum() / total),
            "gain_less_5": float(w[(r > 0) & (r <= band)].sum() / total),
            "no_change": float(w[r == 0].sum() / total),
            "lose_less_5": float(w[(r < 0) & (r >= -band)].sum() / total),
            "lose_more_5": float(w[r < -band].sum() / total),
        }
    return out


_BANDS = ("gain_more_5", "gain_less_5", "no_change", "lose_less_5", "lose_more_5")


def uncompensated_loser_share(
    intra_decile: Mapping[int, Mapping[str, float]],
) -> dict[int, float]:
    """Share of each decile left worse off — losers of any size.

    The metric everyone misses (Cronin, Fullerton & Sexton 2019: within-decile
    horizontal redistribution exceeds between-decile vertical redistribution;
    Sallee 2019: with observable-based transfers, compensating all losers is
    infeasible).
    """
    return {
        int(d): float(bands.get("lose_less_5", 0.0) + bands.get("lose_more_5", 0.0))
        for d, bands in intra_decile.items()
    }


# --------------------------------------------------------------------------
# simulation plumbing
# --------------------------------------------------------------------------


def resolve_elasticity_spec(spec: Any = "main"):
    """Resolve an :class:`~uk_iran_conflict.elasticity.ElasticitySpec`.

    Accepts a spec object, or one of the names ``"main"`` /
    ``"zero"`` (the paper's main specification: no substitution, the Deaton
    first-order approximation and an explicit upper bound on the loss),
    ``"labandeira_short_run"``, ``"labandeira_long_run"``,
    ``"priesmann_short_run"``, ``"priesmann_long_run"``,
    ``"prior_repo"``.

    PolicyEngine UK has no consumption elasticity or price pass-through of its
    own (open issue #1114, which notes UKMOD's TCO uses 0.8), so every non-zero
    spec is reported strictly as a robustness variant.
    """
    from uk_iran_conflict.elasticity import ElasticitySpec  # noqa: PLC0415

    if not isinstance(spec, str):
        return spec
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


def _load_dataset(dataset_path: str | Path):
    from policyengine_uk.data import UKSingleYearDataset  # noqa: PLC0415

    return UKSingleYearDataset(file_path=str(dataset_path))


def _simulation(dataset, reform=None):
    from policyengine_uk import Microsimulation  # noqa: PLC0415

    return Microsimulation(dataset=dataset, reform=reform)


def _array(sim, variable: str, period: int, map_to: str = "household") -> np.ndarray:
    return np.asarray(
        sim.calculate(variable, period=period, map_to=map_to).values, dtype=float
    )


def _household_metrics(sim, period: int) -> dict[str, Any]:
    """Everything the summary needs from one simulation, in one pass."""
    hh_weight = _array(sim, "household_weight", period)
    people = _array(sim, "household_count_people", period)
    person_weight = _array(sim, "person_weight", period, map_to="person")
    equiv = _array(sim, EQUIV_INCOME_VARIABLE, period)
    metrics: dict[str, Any] = {
        "household_weight": hh_weight,
        "household_count_people": people,
        "person_weight": person_weight,
        "net_income": _array(sim, INCOME_VARIABLE, period),
        "equiv_net_income": equiv,
        "gov_balance": float((_array(sim, "gov_balance", period) * hh_weight).sum()),
    }
    try:
        energy = _array(sim, "domestic_energy_consumption", period)
    except Exception:  # noqa: BLE001 — variable name drift must not abort a run
        energy = np.full_like(hh_weight, np.nan)
    metrics["energy_spend_household"] = energy
    metrics["energy_spend"] = float(np.nansum(energy * hh_weight))

    for label, variable in POVERTY_VARIABLES.items():
        metrics[f"poverty_{label}"] = float(
            np.average(
                _array(sim, variable, period, map_to="person"), weights=person_weight
            )
        )
        deep = variable.replace("in_", "in_deep_")
        try:
            metrics[f"deep_poverty_{label}"] = float(
                np.average(
                    _array(sim, deep, period, map_to="person"), weights=person_weight
                )
            )
        except Exception:  # noqa: BLE001 — deep variants are not defined for
            # every measure in every policyengine-uk release
            metrics[f"deep_poverty_{label}"] = float("nan")

    person_equiv_weight = hh_weight * people
    metrics["gini"] = gini(equiv, person_equiv_weight)
    metrics["top1"] = top_share(equiv, person_equiv_weight, 0.01)
    metrics["top10"] = top_share(equiv, person_equiv_weight, 0.10)
    metrics["bottom50"] = bottom_share(equiv, person_equiv_weight, 0.50)
    return metrics


def constituency_aggregates(
    sim,
    period: int,
    change: np.ndarray,
    baseline_income: np.ndarray,
    weight_matrix: np.ndarray | None = None,
    codes: Sequence[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate the household-level change onto the 650 Westminster seats.

    PolicyEngine UK ships a 650-constituency weight set: a (650, n_households)
    matrix of grossing weights, synthetic re-optimisations of ~20k households.
    Aggregating the same household change with two different denominators —
    per household (£) and as a share of income (%) — produces the paper's
    Step 3 result: **the same shock with opposite geography** (Fetzer, Gazze &
    Bishop 2024: affluent areas lose more in £; budget-share literature: poor
    households lose more in %).
    """
    if weight_matrix is None:
        weight_matrix, codes = load_constituency_weights(sim, period)
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    if codes is None:
        codes = [f"seat_{i:03d}" for i in range(weight_matrix.shape[0])]

    change = np.asarray(change, dtype=float)
    baseline_income = np.asarray(baseline_income, dtype=float)

    households = weight_matrix.sum(axis=1)
    total_change = weight_matrix @ change
    total_income = weight_matrix @ baseline_income

    with np.errstate(divide="ignore", invalid="ignore"):
        average = np.where(households > 0, total_change / households, 0.0)
        relative = np.where(total_income != 0, total_change / total_income, 0.0)
    return (
        {str(c): float(v) for c, v in zip(codes, average, strict=True)},
        {str(c): float(v) for c, v in zip(codes, relative, strict=True)},
    )


def load_constituency_weights(sim, period: int) -> tuple[np.ndarray, list[str]]:
    """Load the 650-seat weight matrix and its GSS codes from the simulation.

    Wrapped in its own function because the accessor has moved between
    policyengine-uk releases; a single failure point is easier to repin.
    """
    from policyengine_uk.utils.constituencies import (  # noqa: PLC0415
        get_constituency_weights,
    )

    matrix, codes = get_constituency_weights(sim, period)
    return np.asarray(matrix, dtype=float), [str(c) for c in codes]


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def run_scenario(
    dataset_path: str | Path,
    scenario: Any,
    policy: str | None = None,
    period: int = 2026,
    elasticity: Any = "main",
    include_constituencies: bool = True,
) -> ScenarioResult:
    """Run one (scenario, policy) cell and return its summary.

    ``policy is None`` scores the bare shock — the counterfactual against
    which every response is measured. Otherwise the shock and the policy are
    composed into a single reform so the policy is scored *in the shocked
    world*, not against an unshocked baseline.
    """
    # Resolve the spec up front so a bad name fails before the (expensive)
    # dataset load.
    spec = resolve_elasticity_spec(elasticity)

    dataset = _load_dataset(dataset_path)
    baseline = _simulation(dataset)

    shock = build_shock_reform(scenario, year=period)
    reform = (
        compose(shock, build_policy_reform(policy, year=period)) if policy else shock
    )
    reformed = _simulation(dataset, reform=reform)

    base = _household_metrics(baseline, period)
    ref = _household_metrics(reformed, period)

    weight = base["household_weight"]
    people = base["household_count_people"]
    decile = deciles(base["equiv_net_income"], weight * people)

    if not getattr(spec, "is_main_specification", False):
        # The main specification is zero elasticity. Robustness specs rescale
        # the shocked energy spend before the income change is formed.
        ref = _apply_consumption_response(ref, base, spec, decile)

    change = ref["net_income"] - base["net_income"]

    decile_average: dict[int, float] = {}
    decile_relative: dict[int, float] = {}
    for d in range(1, 11):
        mask = decile == d
        w = weight[mask]
        if w.sum() == 0:
            decile_average[d] = decile_relative[d] = 0.0
            continue
        decile_average[d] = float(np.average(change[mask], weights=w))
        income = float((base["net_income"][mask] * w).sum())
        decile_relative[d] = float((change[mask] * w).sum() / income) if income else 0.0

    intra = intra_decile_breakdown(
        base["net_income"], ref["net_income"], weight, decile
    )

    constituency_average: dict[str, float] = {}
    constituency_relative: dict[str, float] = {}
    if include_constituencies:
        try:
            constituency_average, constituency_relative = constituency_aggregates(
                baseline, period, change, base["net_income"]
            )
        except Exception as exc:  # noqa: BLE001 — geography is optional output
            print(f"warning: constituency aggregates unavailable ({exc})")

    return ScenarioResult(
        scenario=str(getattr(scenario, "name", scenario)),
        policy=policy,
        period=period,
        elasticity=str(getattr(spec, "name", elasticity)),
        exchequer_cost=base["gov_balance"] - ref["gov_balance"],
        household_energy_spend_change=ref["energy_spend"] - base["energy_spend"],
        decile_average_change={int(k): v for k, v in decile_average.items()},
        decile_relative_change={int(k): v for k, v in decile_relative.items()},
        intra_decile={int(k): v for k, v in intra.items()},
        poverty_change={
            label: ref[f"poverty_{label}"] - base[f"poverty_{label}"]
            for label in POVERTY_VARIABLES
        },
        deep_poverty_change={
            label: ref[f"deep_poverty_{label}"] - base[f"deep_poverty_{label}"]
            for label in POVERTY_VARIABLES
        },
        gini_baseline=base["gini"],
        gini_reform=ref["gini"],
        top_one_percent_share_baseline=base["top1"],
        top_one_percent_share_reform=ref["top1"],
        top_ten_percent_share_baseline=base["top10"],
        top_ten_percent_share_reform=ref["top10"],
        bottom_fifty_percent_share_baseline=base["bottom50"],
        bottom_fifty_percent_share_reform=ref["bottom50"],
        constituency_average_change=constituency_average,
        constituency_relative_change=constituency_relative,
        source=str(getattr(scenario, "source", "")),
    )


def _apply_consumption_response(
    ref: dict[str, Any],
    base: dict[str, Any],
    spec: Any,
    decile: np.ndarray,
    carrier: str = "gas",
) -> dict[str, Any]:
    """Apply an elasticity spec to the shocked world's energy spend.

    Under the main (zero-elasticity) spec the spend factor is exactly the price
    ratio, so this is the identity and the measured loss is the first-order
    upper bound. Under a robustness spec, households cut consumption; by the
    envelope theorem the marginal unit forgone is worth what it cost, so the
    reduction in spending returns to net income as a first-order money-metric
    saving.

    ``spec`` may be income-varying (Priesmann), in which case the household's
    baseline income decile selects the elasticity.
    """
    from uk_iran_conflict.elasticity import (  # noqa: PLC0415
        elasticity_for,
        spend_factor,
    )

    adjusted = dict(ref)
    base_q = np.asarray(base.get("energy_spend_household"), dtype=float)
    ref_q = np.asarray(ref.get("energy_spend_household"), dtype=float)
    if base_q.shape != ref_q.shape or not np.any(np.isfinite(base_q)):
        return adjusted

    with np.errstate(divide="ignore", invalid="ignore"):
        price_ratio = np.where(base_q > 0, ref_q / base_q, 1.0)
    price_ratio = np.nan_to_num(price_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    price_ratio = np.clip(price_ratio, 1e-6, None)

    epsilons = np.array(
        [elasticity_for(spec, carrier, int(d)) for d in np.asarray(decile, int)]
    )
    factors = np.array(
        [
            spend_factor(float(r), float(e))
            for r, e in zip(price_ratio, epsilons, strict=True)
        ]
    )
    new_q = base_q * factors
    saving = np.nan_to_num(ref_q - new_q, nan=0.0)

    adjusted["energy_spend_household"] = new_q
    adjusted["energy_spend"] = float(np.nansum(new_q * ref["household_weight"]))
    adjusted["net_income"] = ref["net_income"] + saving
    return adjusted


def write_result(result: ScenarioResult, path: str | Path) -> None:
    """Serialise one result to JSON (mirrors uk-ai-study's ``write_result``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))


def read_result(path: str | Path) -> ScenarioResult:
    """Read back a serialised result (JSON dict keys return as strings)."""
    data = json.loads(Path(path).read_text())
    return ScenarioResult(**data)

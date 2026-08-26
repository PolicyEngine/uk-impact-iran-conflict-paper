# Prior work inventory: `PolicyEngine/energy-price-shock`

What already exists in the earlier PolicyEngine energy-shock repo, what is directly
reusable for *From Hormuz to the Household*, and what has to be built from scratch.
Everything below was read from the repo itself; values are quoted verbatim.

Repo layout: `energy_shock/` (Python package: `config.py`, `baseline.py`,
`sections.py` — 977 lines, `generate.py` — 167 lines, `__main__.py`), a single test
module `tests/test_elasticity.py` (130 lines), a Next.js 16 / React 19 dashboard in
`src/`, committed result JSON in `src/data/`, a 7.7 MB GeoJSON in `public/data/`, and
two reference PDFs in `papers/`.

---

## 1. Elasticity values and their sources

### 1.1 What the prior repo actually uses

`energy_shock/config.py` defines exactly one elasticity object:

```python
ELASTICITY_BY_DECILE = {d: -0.64 + (d - 1) * (-0.11 - -0.64) / 9 for d in range(1, 11)}
```

which evaluates to (as committed in every `results*.json` under `config.elasticity_by_decile`):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ε | −0.640 | −0.581 | −0.522 | −0.463 | −0.404 | −0.346 | −0.287 | −0.228 | −0.169 | −0.110 |

Weighted population mean under the enhanced FRS 2023-24 weights: **−0.382**
(`results.json` → `behavioural[*].mean_elasticity`).

**Only the D1 and D10 endpoints are sourced.** The config comment is explicit:

> The D1 (−0.64) and D10 (−0.11) endpoints are reported by Priesmann & Praktiknjo
> (2025) for German gas; the interior values (D2-D9) are our own linear interpolation
> between those endpoints and are not a result of that paper.

### 1.2 Priesmann & Praktiknjo (2025), *Energy Policy* 207:114850 — what the paper says

RWTH Aachen; German SOEP (electricity, gas) + MOP (car fuels), 2018/2021 waves;
bias-corrected method-of-moments dynamic panel; income-continuous elasticity functions.
Verbatim from the abstract and Table 12 ("Own estimates"):

| Carrier | Short run (low income → high income) | Long run |
|---|---|---|
| **Electricity** | −0.27 → **−0.44** (elasticity **rises** with income) | −0.22 → −0.64 |
| **Gas / heating** | **−0.64 → −0.11** (elasticity **falls** with income) | −0.58 → −0.15 |
| **Car fuels** | −0.47 (low) → −0.14 (high); not separable S/L. Table 12 states the own-estimate span as −0.19 to −0.47 | (same) |

Income elasticities: electricity 0.048 (low) → insignificant (high); gas 0.079 →
declining; **car fuel income elasticity rises with income, 0.060 → 0.443**.

**Two errors in the prior repo to avoid repeating:**

1. It applies the **gas** gradient to **electricity + gas combined**. Priesmann's own
   electricity result has the *opposite sign of gradient* (−0.27 for low income,
   −0.44 for high). Combining them under the gas gradient overstates low-decile
   responsiveness on the electricity half of the bill. The README concedes the
   assumption ("electricity responds at the same elasticity as gas … best read as an
   upper bound") but the code does not separate the carriers.
2. `README.md` attributes "**−0.15**" to Labandeira et al. (2017) as the meta-analytic
   population mean. **No such number appears in that paper.** See below.

### 1.3 Labandeira, Labeaga & López-Otero (2017), *Energy Policy* 102:549–568 — actual values

Meta-regression over 917 short-run / 959 long-run estimates (selected sample after 5%
trim). Headline figures:

- Table 5, average energy elasticity: **short run −0.221 (GLS) / −0.207 (fixed-effects
  panel); long run −0.584 / −0.608.** The conclusion rounds these to **−0.21 (ST) and
  −0.61 (LT)**.
- Table 6, by product (the values the new paper should use):

| Product | Short run | Long run |
|---|---|---|
| Electricity | **−0.126** (10% sig.) | −0.365 |
| Natural gas | **−0.180** (1% sig.) | −0.684 |
| Gasoline | **−0.293** (1% sig.) | −0.773 |
| Diesel | **−0.153** (5% sig.) | −0.443 |
| Heating oil | −0.017 (n.s.) | −0.185 |

- Table 4, by consumer type: residential **−0.215 (ST) / −0.617 (LT)**.
- Table 3, selected-sample descriptives: ST mean −0.186, median −0.140, SD 0.168,
  range [−0.803, 0.066]; LT mean −0.524, median −0.429, SD 0.390.

The repo's "−0.15" matches none of these. Use −0.126 / −0.180 / −0.293 / −0.153 by
carrier, or −0.215 for a residential aggregate.

### 1.4 Functional form (this part is good and worth keeping)

`sections._behavioural_factor_hh`:

```python
return (1.0 + price_pct) ** (1.0 + epsilon_hh)
```

i.e. `q1/q0 = (p1/p0)**ε`, so `spend1/spend0 = (p1/p0)**(1+ε)`. The docstring and
`tests/test_elasticity.py::test_behavioural_factor_physically_admissible_at_extreme_shock`
document why: the linear first-order form `(1+p)(1+εp)` gives
`2.61 × (1 − 0.64×1.61) ≈ −0.079` — a *negative* spending factor — at ε = −0.64,
p = +1.61. The constant-elasticity form stays admissible for all ε ∈ (−1, 0], p ≥ 0.
**Reuse this form and this regression test.**

Missing-decile handling (`_epsilon_per_household`): households with `decile <= 0` get
the weight-weighted mean ε of observed deciles rather than ε = 0.

### 1.5 Validity band caveat, quoted

> the constant-elasticity form is applied out to +161% (Q1 2023 peak scenario), well
> outside the ±10-20% band over which the underlying elasticity studies are validated.
> Treat the extreme-shock scenarios as illustrative rather than predictive.

---

## 2. How the shock was applied to PolicyEngine

**Short answer: it mostly wasn't.** The shock is applied *outside* PolicyEngine, as a
scalar multiplier on two extracted arrays. PolicyEngine is called only for the
baseline and for two *policy* reforms.

### 2.1 Baseline extraction (`baseline.run_baseline`)

`policyengine_uk.Microsimulation(dataset=DATASET_URL, reform=None)`, year `YEAR = 2026`,
`DATASET_URL = "hf://policyengine/policyengine-uk-data/enhanced_frs_2023_24.h5"`
(private HF repo; `HUGGING_FACE_TOKEN` required). Household-level variables pulled:

`electricity_consumption`, `gas_consumption` (summed into `energy`),
`household_net_income`, `household_weight` (via `unweighted=True`),
`household_income_decile`, `region`, `tenure_type`, `accommodation_type`,
`household_id`; plus benunit `family_type` and person `is_SP_age`, groupby-aggregated
into a 7-way `hh_type` (`SINGLE_PENSIONER`, `COUPLE_PENSIONER`, `SINGLE_WORKING_AGE`,
`COUPLE_NO_CHILDREN`, `COUPLE_WITH_CHILDREN`, `LONE_PARENT`, `OTHER`).

Note `energy = electricity_consumption + gas_consumption` **everywhere**; motor fuel is
never touched, and `domestic_energy_consumption` is not used.

### 2.2 The shock itself

`config.PRICE_SCENARIOS`, anchored on `CURRENT_CAP = 1_641` (Ofgem Q2 2026):

| Scenario | New cap | pct |
|---|---|---|
| +10% | 1,805 | 0.10 |
| +20% | 1,969 | 0.20 |
| +30% | 2,133 | 0.30 |
| +60% | 2,625 | 0.60 (`SHOCK_CAP`) |
| Q1 2023 peak | 4,279 | +161% |

`pct = (new_cap - CURRENT_CAP) / CURRENT_CAP`, applied as `energy * pct` (static) or
`energy * ((1+pct)**(1+ε) - 1)` (behavioural). **Uniform across gas and electricity** —
no asymmetric shock, and no use of the
`gov.contrib.policyengine.economy.energy_bills` lever named in the research brief,
nor of `gov.ofgem.energy_price_cap`. `baseline.build_reform_simulation`'s docstring
shows `{"gov.ofgem.energy_price_cap": {"2026-01-01": 2625}}` as an *example* but no
code path uses it.

README states the known distortion: the £1,641 cap "bundles roughly £290/yr of fixed
standing charges", so a uniform percentage shock implicitly rescales standing charges.

### 2.3 The two reforms that *do* go through PolicyEngine

```python
{
    "gov.treasury.energy_bills_rebate.energy_bills_credit": {"2026-01-01": 400}
}  # → ebr_energy_bills_credit
{
    "gov.treasury.energy_bills_rebate.council_tax_rebate.amount": {"2026-01-01": 300}
}  # → ebr_council_tax_rebate
```

Both from the 2022 `energy_bills_rebate` module. A repo-level fix worth carrying over:
`ebr_council_tax_rebate` in policyengine-uk keys off `council_tax_band` alone and so
pays *any* A–D household including Scotland/Wales/NI; `sections._england_mask_unfiltered`
zeroes non-English rows to restore the real policy's scope.

The other three policies are pure numpy, no PolicyEngine: `bn_transfer` (flat payment
= population-mean shock), `bn_epg` (`payment = energy * pct`, full cap-freeze), and
`neg` — the NEF National Energy Guarantee, `ELEC_RATE = 24.70/100 £/kWh`,
`GAS_RATE = 5.70/100`, `NEG_ELEC_KWH = 2_900` → `NEG_ELEC_SPEND = £716`, subsidy
`min(elec, threshold)` indexed to **static pre-shock** consumption
(`"subsidy_indexed_to": "static_baseline_consumption"`).

**The EPG / `price_cap_subsidy` code path described in the research brief is not used
at all.** Nor is fuel duty, VAT, or the carbon-tax contrib module.

---

## 3. Breakdowns: nation-level yes, constituency-level **no**

### 3.1 Nation level — how it works

`baseline.filter_by_country(data, country)` maps `region` → country through
`config.REGION_TO_COUNTRY` (the nine English regions → `ENGLAND`; `SCOTLAND`, `WALES`,
`NORTHERN_IRELAND` map to themselves), builds a boolean mask, and slices every
household array. `generate.run_all_countries()` runs the baseline microsim **once**
and re-runs every section against each mask, writing `results_{england,scotland,wales,
northern_ireland}.json` and the matching `results_breakdowns_*.json`. There is also a
within-UK `by_country` breakdown on every scenario.

This is a straightforward mask-and-re-aggregate pattern with no re-weighting: the
national subsets simply reuse the UK household weights of the households that fall in
each region. Committed nation results (`results_breakdowns.json → country`):

| Country | Elec | Gas | Total energy | Net income | Burden | Households |
|---|---|---|---|---|---|---|
| Northern Ireland | £1,141 | £838 | £1,979 | £54,238 | 3.65% | 0.7m |
| Scotland | £1,086 | £753 | £1,839 | £54,689 | 3.36% | 2.7m |
| England | £971 | £774 | £1,745 | £54,029 | 3.23% | 27.0m |
| Wales | £788 | £613 | £1,401 | £48,325 | 2.90% | 1.6m |

### 3.2 Constituency level — **does not exist**

This is the single most important finding for the new paper, whose headline figure is
constituency-level.

- `public/data/uk_constituencies_2024.geojson` is a **650-feature FeatureCollection**
  (539 `Polygon`, 111 `MultiPolygon`), 7.7 MB, real boundary geometry. Properties per
  feature: `fid`, `Name`, `AltName`, `GSScode` (e.g. `E14001063`), `3CODE` (e.g.
  `AHT`), `Type` (`borough`/`county`), `CTR_REG` (e.g. `South East`), `CRCODE`
  (`SE`), `Country`, `Electorate`, `sqkm`. First feature: Aldershot, E14001063,
  electorate 76,765, 57.1 km².
- **No code reads it.** `grep` over `src/components/Dashboard.jsx` (1,561 lines) finds
  no constituency, geojson, or map reference — the file was committed and never wired
  up. There is no `parliamentary_constituency` variable pull, no constituency weight
  set, no per-seat aggregation, and no constituency key anywhere in any results JSON.

So the entire constituency pipeline — loading PolicyEngine UK's 650-seat weight sets,
aggregating £-loss and %-of-income loss per seat, and rendering two side-by-side maps —
**must be built from scratch.** The GeoJSON is the one genuinely reusable asset, and
`GSScode` is the join key to PolicyEngine's constituency identifiers.

### 3.3 Other breakdowns that do exist

By income decile, by `tenure_type`, by `hh_type`, by country — for baseline burden,
each shock scenario (static *and* behavioural), and each of the five policies
post-shock. `_grouped_post_policy` reports both a **signed** `net_change` (negative =
over-compensated) and an **underwater-only** `extra_cost = mean(max(shock − payment, 0))`.
That underwater-only floor is the closest thing here to the brief's
"uncompensated losers within decile" statistic, but it is a *mean of the truncated
residual*, not a **count/share of losers** — the `IntraDecileImpact` winner/loser
share the brief asks for is not computed.

---

## 4. Results that already exist

`src/data/`, ten committed JSON files (~1.4 MB total): `results{,_england,_scotland,_wales,_northern_ireland}.json`
and `results_breakdowns{,_england,...}.json`.

`results.json` keys: `baseline`, `shock_scenarios`, `behavioural`, `policies`,
`policy_post_shock` (keyed `flat_transfer`, `ct_rebate`, `bn_transfer`, `bn_epg`,
`neg`), `config`.

UK baseline: **32.0m households**, mean energy spend **£1,742** (elec £976 / gas £766,
56.0% electricity), total **£55.7bn**, mean net income **£53,812**.

Energy burden by decile (`baseline.deciles`) — note the non-monotonicity, which is
itself worth checking before reuse:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Energy £ | 983 | 1,502 | 1,256 | 1,977 | 1,664 | 1,771 | 1,542 | 1,784 | 2,640 | 2,470 |
| Net income £ | 20,343 | 29,274 | 34,881 | 37,536 | 40,657 | 52,796 | 53,110 | 66,896 | 75,233 | 141,526 |
| Burden % | 4.83 | 5.13 | 3.60 | 5.27 | 4.09 | 3.36 | 2.90 | 2.67 | 3.51 | 1.75 |

D1 < D2 and D3 < D2 in £ terms, and burden is *not* monotone. Consistent with
Fetzer–Gazze–Bishop's "affluent areas lose more in £", but the D3/D5/D7 dips look like
imputation noise in the LCFS→FRS fusion and should be diagnosed, not inherited.

+60% scenario (the paper-relevant one): static mean hit **£1,045/yr**, behavioural mean
hit **£637**, bill saving **£408**. D1: static £589 → behavioural £181 (2.90% → 0.89%
of income), consumption cut **26.0%**. That 26% cut for the poorest decile at +60% is
the number a zero-elasticity main specification deliberately refuses to assume away.

+10% scenario: mean hit £174/yr (£15/month), £5.6bn aggregate; NI £198, Scotland £184,
England £174, Wales £140.

Policy costs: flat £400 transfer → **£12.8bn**, offsets 68% of the D1 shock and 27% of
D10's; England-only CT rebate → **£6.5bn**, mean £204/household; NEG baseline cost
**£16.0bn**, mean benefit £500.

---

## 5. PolicyEngine version pinned

`pyproject.toml`:

```toml
requires-python = ">=3.13"
dependencies = [
    "policyengine-uk>=2.88.0",
    "microdf-python>=1.2.0",
    "pandas>=2.0",
    "numpy>=1.26",
]
```

**This is a floor, not a pin** — `>=2.88.0` is unreproducible and the new repo should
pin an exact version (the brief calls for "pinned `policyengine-uk`"). There is no
lockfile for the Python side (`bun.lock` covers only the JS dashboard).

The repo also records a live blocker worth knowing about (`baseline.py` module
docstring, echoed in `pyproject.toml`):

> A migration to the unified `policyengine.py` API was trialled but blocked by a
> bootstrap mismatch in `policyengine.py` 4.1.x's bundled UK release manifest (pinned
> to a data-package version without a published `release_manifest.json` on Hugging
> Face).

So: use `policyengine_uk.Microsimulation` directly, not the 4.x unified API.

---

## 6. Directly reusable vs must be rebuilt

### Reusable close to as-is

- **`_behavioural_factor_hh`** — the constant-elasticity spend factor `(1+p)**(1+ε)`
  and the argument for it over the linear form. Ported into
  `uk_iran_conflict/elasticity.py`.
- **`tests/test_elasticity.py`** — especially the regression test that the linear form
  is *not* being used. Ported.
- **Baseline extraction pattern** — `_hh_array` / `_person_array` / `_benunit_array` /
  `_weights(unweighted=True)`, and the benunit+person → household `hh_type` groupby.
- **`weighted_mean`, `decile_means`, `filter_by_country`, `REGION_TO_COUNTRY`.**
- **The England-only mask for `ebr_council_tax_rebate`** — a real bug fix in the
  underlying model's policy scope.
- **The two `energy_bills_rebate` reform dicts** — a working template for the flat
  rebate and the IPPR-style per-household credit in Step 4.
- **`uk_constituencies_2024.geojson`** — 650 features with `GSScode`, ready for the
  headline map.
- **The caveat text** in `README.md` and `config.py` (Priesmann transferability, the
  standing-charge distortion, the out-of-band extrapolation) — well drafted, reusable
  as paper prose with attribution to the earlier repo.

### Must be rebuilt

1. **The constituency pipeline, entirely.** 650-seat weight sets → per-seat £-loss and
   %-of-income loss → two hex/choropleth maps. Nothing exists beyond the boundary file.
2. **Asymmetric gas vs electricity shock.** The prior repo shocks a single combined
   cap uniformly. The brief's Step 1 (gas 62% of household final energy, gas sets the
   electricity price ~85% of the time, cap lags wholesale 6–9 months, wholesale ≈
   40–50% of a dual-fuel bill, so +X% wholesale → ~0.45X two-to-three quarters later,
   quantised quarterly) has no counterpart here at all.
3. **Motor fuel.** Petrol +20% / diesel +36% is central to the 2026 shock and entirely
   absent. Needs the `household/consumption/fuel/prices/{petrol,diesel}.yaml`
   DESNZ path and household litres = spend ÷ pump price.
4. **Zero-elasticity main specification.** The prior repo has no no-response option:
   `ELASTICITY_BY_DECILE` is always applied when behavioural output is requested. The
   new paper inverts this — static Deaton first-order is the main spec, elasticity the
   robustness check. Handled by `ZERO_ELASTICITY` in `uk_iran_conflict/elasticity.py`.
5. **Carrier-specific elasticities.** Gas gradient applied to electricity must be
   replaced by Priesmann's *own* electricity gradient (which runs the other way) plus
   Labandeira Table 6 point estimates for a flat-elasticity variant.
6. **Uncompensated-loser shares within decile.** Only the truncated-mean residual
   exists; the actual within-decile loser *count/share*
   (Cronin–Fullerton–Sexton / Sallee / Douenne) is not computed.
7. **The scenario set.** `CURRENT_CAP = 1_641`, "Ofgem Q2 2026" — superseded by the
   Cornwall Insight Oct-26 cap of **£1,729** (19 Aug 2026). The +10/20/30/60% and Q1-2023
   ladder must be replaced by NIESR baseline ($103/bbl) vs adverse ($140/bbl) mapped
   into a cap path.
8. **The policy set.** Only flat transfer, CT rebate, notional cap-freeze and NEG
   exist. Social tariff, the JRF universal discounted block (two-tier
   `monthly_epg_consumption_level`), WHD expansion, and the domestic-fuel VAT cut all
   need building.
9. **Reproducibility scaffolding.** No `analysis/run_all.py`, no `results/` CSV, no
   `paper/values_generated.tex` emitter, no exact version pin, no lockfile. Only one
   test module, covering only elasticity arithmetic — nothing touches the microsim path.
10. **The "−0.15 Labandeira" attribution** must be dropped or corrected wherever it is
    carried forward.

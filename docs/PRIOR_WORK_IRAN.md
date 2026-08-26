# Prior work inventory — `impact-iran-war-living-standards`

Purpose: a precise record of what the earlier PolicyEngine Iran-shock repo actually
did, so this paper does not redo it and does not inherit its weaknesses unexamined.

Source snapshot read: `impact-iran-war-living-standards` (package
`energy-price-shock-impact` v0.1.0, module `src/iran_impact/`, Next.js dashboard,
results JSON committed at `dashboard/public/data/iran_impact_results.json`).
Live dashboard: `https://uk-energy-shock-impact.vercel.app`.

---

## 1. Environment and pinning

`pyproject.toml`:

- `requires-python = ">=3.13"`, build backend `hatchling`, package root `src/iran_impact`.
- Runtime deps: `numpy>=1.26`, `pandas>=2.2`, `microdf-python`.
- **Pinned `policyengine==5.0.1`** (the `policyengine.py` orchestration wrapper — *not*
  `policyengine-uk` directly), declared identically under two optional-dependency
  extras `uk` and `simulation`. There is no pin on `policyengine-uk` itself, so the
  underlying country-model version was only transitively determined. **The new repo
  should pin `policyengine-uk` explicitly**, per the brief's conventions.
- `[tool.pytest.ini_options] testpaths = ["tests"]` — but **there is no `tests/`
  directory in the repo**. Zero tests exist. Nothing in the prior work is verified.

Simulation entry point (`pipeline.run_baseline`):

```python
from policyengine.tax_benefit_models.uk import managed_microsimulation

sim = managed_microsimulation()
```

i.e. the managed Enhanced FRS 2023-24 microdata via the `policyengine` package
(private Hugging Face microdata behind a token, per the brief).

Simulated year: `YEAR = 2027` (the 2027-28 tax year, chosen as the year Autumn
Budget 2026 decisions bite). Weighted household count in the results JSON:
**29.6m households** — note this is below the ~32m the README claims, worth
reconciling.

---

## 2. Scenarios as defined (`config.py`)

Three scenarios, each a flat dict of four percentage-point/percent numbers. Verbatim:

```python
SCENARIOS = {
    "low_shock": {
        "cap_increase_pct": 15,
        "cpi_increase_pp": 1.0,
        "fuel_pct": 20,
        "food_increase_pct": 2.0,
    },
    "central_shock": {
        "cap_increase_pct": 45,
        "cpi_increase_pp": 2.5,
        "fuel_pct": 45,
        "food_increase_pct": 4.0,
    },
    "severe_shock": {
        "cap_increase_pct": 90,
        "cpi_increase_pp": 4.5,
        "fuel_pct": 80,
        "food_increase_pct": 6.5,
    },
}
```

Anchors, as documented in the config docstring and `dashboard/src/lib/scenarioContent.js`:

| Key | Narrative anchor | Cited source |
|---|---|---|
| `low_shock` | De-escalation from the Aug-2026 position; Brent ~$85/bbl (4 Aug 2026 spot); pump ~157p petrol / ~187p diesel (~+20% on Autumn Budget 2025); cap in line with the observed +13% Jul-2026 rise plus Cornwall Insight Q4 (~£1,700) | Cornwall Insight; observed spot |
| `central_shock` | Sustained Hormuz constraint; Goldman Sachs extended-closure Brent >$100 through 2026 ($120 Q3 / $115 Q4); oil→retail pass-through per Commons Library CBP-10601; CPI adder consistent with BoE Jun-2026 ~3%→4%+ | oilprice.com Goldman note; commonslibrary CBP-10601 |
| `severe_shock` | Extended full closure / prolonged war; Goldman extreme-adverse (Brent >$115–120); Oxford Economics prolonged-war (world CPI 7.7%, global recession) | Goldman; Oxford Economics |

Baseline cap constant:

```python
CURRENT_ENERGY_CAP = 1_663  # Ofgem default tariff cap, 1 Jul–30 Sep 2026,
# NEW TDCV basis (≈£1,862 on the pre-Jul-2026 basis)
```

**Critical gaps in the scenario design relative to this paper's brief:**

1. **No oil price level anywhere in code.** $85 / $100 / $115–120 appear only in
   comments and dashboard prose. Nothing maps a `$/bbl` level to the modelled shock.
2. **No wholesale gas variable at all.** There is no p/therm, no NBP/TTF, nothing.
   The cap increase is asserted directly as a percentage.
3. **No lag structure.** `cap_increase_pct` is applied instantaneously and uniformly
   for a full year. The 6–9 month forward-wholesale lag and the ~40–50% wholesale
   share of a dual-fuel bill are nowhere represented; there is no quarterly cap path.
4. **Gas and electricity are not shocked separately.** `run_baseline` computes
   `energy = electricity_consumption + gas_consumption` and then applies one common
   `cap_increase_pct` to the sum. `electricity` is retained separately *only* to score
   the electricity VAT cut. The marginal-pricing/gas-sets-power ~85% fact is not
   modelled — this is exactly the choice the brief says makes it a UK paper, and the
   prior work does not make it.
5. **Petrol and diesel are not distinguished.** A single `fuel_pct` covers both.
6. Scenario labels are not the brief's NIESR baseline/adverse frame at all
   (no $103/$140), and no scenario represents the *realised* 2026 path
   (oil +57%/+$42, gas +78p/therm, petrol +20%, diesel +36%).

---

## 3. How the shock was actually applied to PolicyEngine

**It was not a PolicyEngine reform.** No `Reform`, no parameter override, no
`policyengine.contrib` lever was constructed anywhere in the codebase. Confirmed:
`energy_bills.yaml`, `energy_price_cap.yaml`, `price_cap_subsidy`,
`fuel/prices/*.yaml` are never referenced.

The pattern is: **PolicyEngine is used once, read-only, to extract baseline
household arrays; all shocks and all policies are then arithmetic in numpy on those
arrays.** `run_baseline` pulls, at household level for 2027:

`electricity_consumption`, `gas_consumption`, `household_net_income`,
`equiv_household_net_income`, `household_count_people`, `household_weight`
(unweighted=True), `household_income_decile` (clipped to 1–10 to keep
negative-income households), `region`, `tenure_type`, `council_tax_band`, plus
benunit/person aggregations of `universal_credit`, `pension_credit`,
`income_support`, `housing_benefit`, `child_benefit`, `esa_income`, `jsa_income`,
`pip`, `dla`, `attendance_allowance`, `carers_allowance`, `family_type`, `is_SP_age`.

The four transmission channels (`compute_scenario`), each a single multiplication:

```python
energy_shock = energy * cap_increase_pct  # energy = elec + gas
fuel_shock = fuel_cost * fuel_pct  # fuel_cost is SYNTHETIC (see below)
food_shock = food_cost * food_increase_pct  # food_cost is SYNTHETIC
benefit_uprating_lag = benefit_income * cpi_increase_pp * UPRATING_LAG_FACTOR
net_impact = energy_shock + fuel_shock + food_shock + benefit_uprating_lag
```

**Fuel and food spending are not microdata.** They are imputed from a scalar base
times a decile step function — *not* from LCFS, *not* from PolicyEngine's fuel-duty
litres or COICOP categories, despite both existing:

```python
BASE_FUEL_SPEND = 1_300  # ONS LCF FYE2024 Family Spending A6, ~£25/wk
BASE_FOOD_SPEND = 5_000  # ~£95/wk
FUEL_DECILE_FACTORS = {
    1: 0.70,
    2: 0.70,
    3: 0.90,
    4: 0.90,
    5: 1.00,
    6: 1.00,
    7: 1.15,
    8: 1.15,
    9: 1.25,
    10: 1.25,
}
FOOD_DECILE_FACTORS = {
    1: 0.65,
    2: 0.65,
    3: 0.80,
    4: 0.80,
    5: 1.00,
    6: 1.00,
    7: 1.20,
    8: 1.20,
    9: 1.45,
    10: 1.45,
}
```

Every household in a decile therefore has *identical* fuel and food spend. Within-decile
horizontal variation in the fuel and food channels is **exactly zero by construction** —
which destroys the Cronin–Fullerton–Sexton / Douenne horizontal-loser result the new
paper is built on. The config comment concedes this ("A future improvement is direct
LCF-based imputation onto the FRS microdata").

Uprating lag:

```python
UPRATING_LAG_FACTOR = 0.5  # expected-value erosion over the year, not the 12-month max
```
applied to CPI-linked benefits with the state pension deliberately excluded (triple lock).

---

## 4. Policies scored (`compute_policies`) — all stylised, none as reforms

| Key | Construction | Constant |
|---|---|---|
| `energy_price_guarantee` | `max(0, energy*cap_pct − energy*0.10)` | `EPG_CAP_PCT = 0.10` |
| `flat_rebate` | flat to every household | `FLAT_REBATE = 400` |
| `ct_rebate` | CT bands A–D | `CT_REBATE = 300` (stylised UK-wide; the 2022 scheme was England-only £150) |
| `uc_uplift` | £20/wk × 52 to UC recipients | `UC_UPLIFT_WEEKLY = 20` |
| `fuel_duty_cut` | `5p × 1200 litres`, scaled by `fuel_cost/BASE_FUEL_SPEND` | `FUEL_DUTY_CUT_PENCE = 5`, `MEAN_ANNUAL_LITRES = 1_200` |
| `means_tested_payment` | to any means-tested-benefit household | `MEANS_TEST_AMOUNT = 650` |
| `elec_vat_cut` | `electricity × (1+cap_pct) × 5/105` | `ELEC_VAT_SAVING_RATE = 5/105` |
| `accelerated_uprating` | refunds the whole uprating-lag channel | — |
| `social_tariff` | 50% of the energy shock, if UC **or** household income < £20k | `SOCIAL_TARIFF_INCOME_THRESHOLD = 20_000`, `SOCIAL_TARIFF_DISCOUNT = 0.50` |
| `combined` | sum of the eight `COMBINED_KEYS` (social tariff excluded), clipped to `net_impact`; unclipped sum retained as `_combined_outlay` for fiscal cost | — |

The clipped-benefit / unclipped-outlay split is a genuinely good idea and worth keeping.

Notably **absent**: the JRF universal discounted block (50% of typical consumption at
a discounted rate + per-child allowance), the WHD expansion, and the VAT cut on
*domestic fuel* generally (only electricity). Three of the brief's five Step-4
policies are unbuilt.

---

## 5. Outcome measures

- Fuel poverty: `energy / net_income > 0.10` (`FUEL_POVERTY_THRESHOLD = 0.10`).
  Explicitly flagged in code and metadata as **not** England's official LILEE metric
  and not comparable with official statistics.
- Poverty: people below `0.6 ×` weighted-median equivalised net income (BHC),
  person-weighted (`weights × household_count_people`). Post-shock equivalised income
  approximated by `equiv_income × clip(1 − cost/income, 0, None)`.
- Winners/losers: `±£1` threshold (`WINNERS_LOSERS_THRESHOLD = 1`), reported by
  quintile as pct_winners / pct_unchanged / pct_losers.
- Policy scoring: `fiscal_cost_bn`, `avg_benefit_per_hh`, `targeting_bottom40`,
  fuel-poverty rate before/after, `n_lifted_from_poverty`, plus per-quintile
  `mean_benefit`, `mean_residual_impact`, `mean_benefit_pct_income`, `benefit_share_pct`.

---

## 6. Outputs produced

`run_pipeline.py` / `iran-impact-build` writes one JSON to `data/` and
`dashboard/public/data/iran_impact_results.json`. Structure:

- `year`, `current_energy_cap`, `metadata` (cap basis, fuel-poverty and poverty definitions)
- `baseline`: `n_households_m 29.6`, `mean_energy_spend £1,331`,
  `mean_net_income £61,924`, `total_energy_spend_bn 39.4`,
  `fuel_poverty_rate_pct 5.8`, `fuel_poor_households 1,710,790`, plus by-quintile
  energy spend / income / energy share / FP rate (Q1: £1,059 spend, £23,517 income,
  **11.4% energy share**, 14.7% FP).
- `scenarios.<key>`: `params`, `summary`, `by_quintile`, `by_region` (12 rows),
  `by_tenure` (5), `by_country` (4), `by_hh_type` (6), `fp_by_tenure`,
  `channel_decomposition`.
- `policy_responses.<key>.<policy>`: as in §5.

Headline results, for reference/sanity-checking the new build:

| Scenario | Mean net impact | % of income | Total £bn | FP rate 5.8% → | Pushed into poverty |
|---|---|---|---|---|---|
| low (+15/+20) | £578 | 1.7% | 17.1 | 6.9% | 423,751 |
| central (+45/+45) | £1,429 | 4.3% | 42.3 | 10.3% | 986,615 |
| severe (+90/+80) | £2,643 | 8.1% | 78.2 | 15.3% | 2,083,810 |

Central-scenario channel decomposition (mean £/household): energy 599, fuel 585,
food 205, uprating lag 40. **The synthetic fuel channel is nearly as large as the
real microdata energy channel** — the largest single credibility exposure in the
prior work.

Q1 mean impact under `central_shock` is £1,067 = **10.1% of income**, against 4.3%
at the mean: the regressivity result is there, but it is partly mechanical from the
decile step functions.

Granularity ceiling: **region (12) is the finest geography**. No constituency, no LA,
no hex maps. The 650-seat weight sets — the paper's actual contribution — are untouched.

---

## 7. Directly reusable vs must-rebuild

**Reusable as-is (lift and adapt):**

- The `managed_microsimulation()` → household-array extraction pattern, and the exact
  variable list in `run_baseline`.
- `_build_household_type`, `_build_uc_recipients`, `_build_means_tested_receipt`,
  `_build_benefit_income` (with its state-pension exclusion rationale). These are the
  fiddly bits and they are sound.
- microdf helpers `weighted_mean` / `weighted_sum` / `_weighted_median`, `_safe_div`,
  and the decile clip-to-1–10 fix for negative-income households.
- The clipped-benefit vs unclipped-fiscal-outlay accounting in `compute_policies`.
- Person-weighted poverty line and `_equiv_after_cost`.
- The winners/losers-by-quintile scaffolding — extend it to the within-decile
  uncompensated-loser count the brief needs.
- Policy constants and their provenance comments (£400 EBSS, £650 CoL payment, 5p fuel
  duty to 31 Dec 2026, elec VAT 5%→0% Oct-26–Mar-27 at ~£45/hh / ~£850m).
- The results-JSON schema shape and the whole dashboard, if a dashboard is wanted.

**Must be rebuilt from scratch:**

- **The entire scenario layer.** Oil-price levels, wholesale gas in p/therm, the
  quarterly Ofgem cap path with the 6–9 month lag and ~0.45 wholesale weight, separate
  gas/electricity shock factors with an explicit marginal-pricing pass-through
  parameter, separate petrol/diesel. None of this exists.
- **Shock application as PolicyEngine reforms** rather than numpy arithmetic — the
  brief's Step 1/Step 4 both require it (`energy_bills.yaml`, `price_cap_subsidy`,
  `energy_price_cap.yaml`, `fuel/prices/{petrol,diesel}.yaml`, `reduced_rate.yaml`).
- **Fuel and food incidence from real microdata** (LCFS-fused COICOP, PolicyEngine's
  fuel-duty litres) instead of the decile step functions.
- **Constituency/LA geography** and the two-hex-map figure.
- The JRF discounted block, WHD expansion, domestic-fuel VAT cut, IPPR rebate.
- Tests. There are none.
- `paper/values_generated.tex` emission — no LaTeX pipeline existed.

---

## 8. What a referee would question

1. **Synthetic fuel and food spend.** Two of four channels are a scalar × a
   ten-step decile function. Zero within-decile variance; the decile gradient is
   assumed, and it *is* the distributional result for those channels. Referees will
   see the mean fuel shock (£585) exceeding real-microdata energy (£599) at central
   and stop reading. Non-negotiable rebuild.
2. **No pass-through, no lag, no elasticity.** A wholesale/oil move is asserted
   directly as a retail percentage for a full year, with no cap lag and no wholesale
   share. Combined with zero demand response (PolicyEngine issue #1114), the estimate
   is a pure Deaton first-order upper bound — defensible only if stated as such,
   which the prior repo does not do.
3. **Gas and electricity shocked identically.** Contradicts UK marginal pricing and
   wastes the fact that they are separate NEED-calibrated variables.
4. **10%-of-income fuel poverty.** Flagged in code, but it is a pre-2013 metric; the
   baseline 5.8% is far below official England LILEE (~13%) precisely because it uses
   *net income* and PolicyEngine's modelled bills. Any headline "extra fuel-poor
   households" number is not comparable to anything official.
5. **Mean net income £61,924 at household level** with Q1 at £23,517 — plausible for
   gross-of-VAT household net income, but the brief's Gap 1 (baseline VAT never
   deducted from net income; `household_tax.py` carries `vat_change`, not `vat`) means
   the denominator of every "% of income" figure is overstated. The prior repo never
   mentions this. State it.
6. **No general equilibrium.** Per Känzig (2023), income/employment channels dominate
   for poorer households and GE/indirect effects are ~2/3 of the response; per Goulder
   et al. (2019) microsimulation is use-side only. The prior repo has no caveat section
   at all — it presents use-side losses as *the* living-standards impact.
7. **Food channel double-counting risk.** `food_shock` (energy→food pass-through) and
   `benefit_uprating_lag` (CPI adder × benefit income) are both driven by the same
   inflation, and the CPI adder is itself partly the food and energy rises. The
   channels are summed without adjustment.
8. **Uprating-lag factor of 0.5** is asserted, not derived; and `cpi_increase_pp` is
   an independent free parameter rather than something implied by the price shocks.
9. **`combined` package** sums eight policies including both the EPG and the flat
   rebate — a package no government proposed — and its £bn cost is the unclipped sum.
   Read it as an upper bound, not a proposal.
10. **Scenario probabilities absent.** Three points on a line with no weight, and
    `severe_shock` (+90% cap) sits near the 2022 crisis when the brief records the
    realised 2026 gas move as +78p/therm against +300p in 2022 — the severe scenario
    is arguably mis-anchored an order of magnitude high on gas.
11. **No tests, no version pin on `policyengine-uk`, no reproducibility record**
    (no seed, no data vintage hash) for a committed results JSON.

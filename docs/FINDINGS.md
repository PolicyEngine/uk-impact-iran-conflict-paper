# Findings from the first full run — what the prose must be corrected to

> **HISTORICAL RECORD — SUPERSEDED. Do not cite these numbers.** This file
> records the **first** full run of the pipeline, before three rounds of referee
> review. It is kept unedited as an audit trail of what was found and when; it
> is deliberately **not** rewritten to match later results, because the
> corrections it prompted are themselves part of the record.
>
> Several headline figures below have since been superseded. In particular:
>
> - the **£13.9bn** aggregate and **£471** mean loss predate decision D2 in
>   `docs/FIXES.md` (the consumption-weighted quarterly phase-in, which the
>   paper's Step 1 always claimed and the code did not implement). On the main
>   specification the aggregate is materially lower.
> - the **67.8%** motor-fuel share is a single point estimate on one damping
>   calibration. `docs/VALIDATION.md` Check 2 shows it is inflated by two
>   compounding imputation errors, and the paper now reports the share as a
>   **range across specifications**, not as a point.
> - percentage-of-income statistics here are on the **unequivalised**
>   denominator; decision D1 moved everything to equivalised AHC income.
>
> Where this file and the documents below disagree, **they win**:
>
> - `docs/VALIDATION.md` — external validation against ONS, DfT, DWP and the literature
> - `docs/REFEREE_REPORTS.md`, `docs/REFEREE_ROUND2.md` — the referee reports
> - `docs/FIXES.md` — the resulting work list and the two decisions taken
> - `results/` and `paper/values_generated.tex` — the current numbers, always
>
> What in this file **is** still current: §1, the structural finding that
> PolicyEngine UK has no price channel, that `energy_bills` is a dead parameter
> and that there is no Warm Home Discount. That finding held through every
> round and is a contribution of the paper.

Generated from the real pipeline (`analysis/run_incidence.py`, `run_sensitivity.py`)
against PolicyEngine UK microdata (`populace_uk_2023.h5`, 535,080 households,
29.5m weighted, period 2026). **Every claim below is verified.** The drafted
prose predates these numbers and contradicts several of them.

## 1. The model has no price channel (structural — already handled in code)

`gas_consumption`, `electricity_consumption`, `domestic_energy_consumption`,
`petrol_spending` and `diesel_spending` are **input** variables with no formula.
`gov.ofgem.energy_price_cap` is read by exactly one variable
(`monthly_epg_consumption_level`, EPG subsidy machinery).
`gov.contrib.policyengine.economy.energy_bills` is a **dead parameter** — no
variable reads it. There is **no Warm Home Discount** at all: no parameter, no
variable. `petrol_price` is parameter-driven but, because `petrol_spending` is an
input, raising it *cuts litres* — implicitly unit-elastic, the opposite of a
first-order shock.

Consequence: the shock is applied on the consumption side in
`uk_iran_conflict/incidence.py`, and three of the five policies are computed
directly in `uk_iran_conflict/policies.py`. The methodology section's account of
shocking parameters via reforms is **wrong** and must be rewritten to describe
what the code actually does. This is a contribution, not an embarrassment: it is
a concrete, citable finding about what PolicyEngine UK can and cannot do, and it
motivates the upstream PRs.

## 2. THE BIG ONE: motor fuel is two-thirds of the loss

| Channel | Share of aggregate loss (realised 2026) |
|---|---|
| Motor fuel | **67.8%** |
| Gas | 17.6% |
| Electricity | 14.6% |

(NIESR adverse: 66.1% / 18.5% / 15.4%.)

This reframes the paper. It is **not** mainly a domestic-energy-bill shock; it is
mainly a **pump-price** shock. Every domestic-bill instrument in the policy
debate — social tariff, JRF block, WHD, VAT zero-rating — can therefore reach at
most about a third of the loss **by construction**. That, not targeting design,
is the deepest reason the compensation options underperform, and it is a genuinely
novel and policy-relevant point that the current draft does not make at all.

It also explains the non-monotonic cash gradient (vehicle ownership, not energy
efficiency, drives the £ profile).

## 3. Headline numbers (realised 2026, the central scenario)

- Aggregate cost **£13.9bn**; mean household loss **£471** (**0.77%** of income
  on the aggregate ratio; **0.32%** for the median household)
- Decile 1: **2.07%** of income, **£395**. Decile 10: **0.36%**, **£539**
- Validation: mean £471 against Resolution Foundation's independently published
  **£480** for a typical working-age household — close, and worth stating
- NIESR baseline **£9.3bn**; NIESR adverse **£21.1bn** (decile 1 at 3.17%)

## 4. The distributional gradient is robust; the cash gradient is not monotonic

Burden by decile (% of income), aggregate ratio:
2.07, 1.03, 0.92, 0.92, 0.99, 0.75, 0.76, 1.00, 0.75, 0.36

Loss by decile (£):
395, 296, 344, 419, 466, 403, 475, **754**, 631, 539

**Correction required.** The results draft says the loss "rises across deciles,
from £395 in decile one to £539 in decile ten". True at the endpoints, false in
between: decile 8 loses most in cash (£754), and decile 1 loses more than deciles
2 and 3. Say "broadly rises" or, better, report the endpoints and note the peak
at decile 8 with the motor-fuel explanation.

**The %-gradient IS robust** — it survives on medians, so it is not an artefact
of the low-income tail:

| | Decile 1 | Decile 10 | Ratio |
|---|---|---|---|
| Aggregate ratio | 2.07% | 0.36% | 5.75x |
| Median household | 0.67% | 0.11% | 6.1x |

Decile 1's median income is £16,000 and only 0.57% of it has zero or negative
income, so the gradient is not a division-by-small-income artefact. State this
explicitly — it is the obvious referee attack and it does not land.

Within-decile spread exceeds between-decile spread (Cronin, Fullerton and Sexton
2019 holds): mean within-decile p90-p10 range **2.75pp** vs between-decile range
**1.71pp**. Decile 1 is much the widest (p50 0.67%, p90 8.59%).

## 5. Policy scorecard (realised 2026)

| Policy | Cost £bn | % to D1-3 | % of losers uncompensated |
|---|---|---|---|
| Social tariff (35% discount) | 1.3 | 57.5 | 84.8 |
| JRF universal block | 1.9 | 31.8 | 84.8 |
| WHD expansion (£150) | 0.7 | 60.0 | 87.7 |
| VAT zero-rating | 2.1 | 25.3 | **100.0** |
| IPPR flat rebate (£183) | 5.4 | 30.8 | **44.1** |

**Corrections required.**

- The intro contrasts the social tariff and the JRF block on uncompensated share.
  That contrast **does not survive**: both are 84.8%. The JRF block's advantage is
  in *who* it reaches, not the headline count. Rewrite accordingly.
- VAT zero-rating compensates **nobody** fully — a 4.8% VAT cut cannot offset a
  ~14% domestic price rise, and it reaches none of the 68% motor-fuel channel.
- The IPPR flat rebate is the only option compensating a majority (55.9% fully
  compensated). Universality beats targeting here, which speaks directly to the
  Levell-O'Connell-Smith result and against the "always target" consensus.
- JRF's own costing is ~£5bn; our block costs £1.9bn, so our parameterisation is
  less generous than theirs. Say so rather than implying agreement.

## 6. Sensitivities (`results/sensitivity/`)

**Elasticity dominates.** Aggregate £13.89bn (eps=0) to £2.55bn (eps=-0.8),
near-linear. Credible short-run band is much tighter: Labandeira short-run shaves
**27%**, Priesmann short-run **32%**. The main specification stays zero-elasticity
(first-order, explicit upper bound). Income-varying elasticities flatten the
D1/D10 ratio from 5.7 to 3.6 (Priesmann) — the "heat or eat" effect — but the
qualitative ordering survives every row.

**Cap lag is a windowing artefact.** Cumulative loss is *exactly invariant* at
£437/household (£12.88bn) across lags 1-4. Only calendar-year attribution moves
(annualised 2026 £403 to £320). Report cumulative as the headline and treat the
annual figure as attribution.

**The asymmetry barely matters for headline incidence — say so plainly.**
Sweeping the marginal-pricing share 0.70 to 1.00 moves mean loss £459 to £483
(5.2%) and leaves the D1/D10 ratio frozen at ~5.72. What it moves is
*composition*: gas share of the domestic loss 59.3% to 50.5%.

**Correction required.** The methodology calls the asymmetry "the modelling
choice that makes this a UK paper". Overstated. It is a compositional refinement
that matters for fuel-level policy design, not for headline incidence. Keep it,
justify it, and report honestly that it does not move the distributional result.

Uncertainty ranking to state in the appendix:
**demand response >> calendar-year lag >> marginal-pricing share.**

## 7. What CANNOT be done with this dataset

**Constituency analysis is impossible.** There is no constituency weight matrix in
`populace_uk_2023.h5` (the calibration file has 149 targets, not the 1,512 the
brief claimed), and `local_authority` is a degenerate default — **every household
is 'MAIDSTONE'**. Region (12) and country (4) are the only real geography.

This kills the paper's stated headline contribution. The abstract, intro,
results and appendix all currently promise a 650-seat hex-map result. They must
be rewritten. The replacement headline is the **£-versus-% decile contrast**
(`fig2_decile_dual.png`), with region as the geographic cut
(`fig4_region.png`).

Affected macros, all emitting `\GENMISSING`: `\genConstituencyRankCorr` (abstract,
intro, results, appendix), `\genTailOverlapCount`, `\genElasticityRankCorrHigh`.

**No heating-fuel variable exists**, so there is no on/off-gas-grid split:
`\genOffGridLossGbp`, `\genOnGridLossGbp` and Table `tab:heating` have no backing
data. Cut the table and move the off-gas-grid question to further work (where
Douenne 2020 already sits as the no-UK-counterpart gap).

## 8. Means-tested coverage — a limitation to declare

PolicyEngine's modelled means-tested population is **15.7% of households (4.6m)**:
UC 3.03m, Pension Credit 1.22m, Housing Benefit 0.73m, income-based ESA 0.34m.
All eight variables resolve correctly — this is not a code failure — but it sits
**below administrative UC caseloads (~7m households)**, reflecting modelled
take-up. Consequence: the social tariff's and WHD's reach is probably
**understated** and the uncompensated shares correspondingly **overstated**.
State this in Limitations and treat those two rows as an upper bound on the
uncompensated count.

Related: of households losing more than 5% of net income, **98%** are outside the
means-tested system on these numbers. That is our figure; the brief's ~40% anchor
is JRF's, measured differently (households "struggling to heat their home"). Do
not conflate them — cite ours as ours and JRF's as theirs.

## 9. Figures now available (`results/figures/`)

`fig1_price_path`, `fig2_decile_dual` (**new headline**), `fig3_within_decile`,
`fig4_region` (replaces the constituency maps), `fig5_fuel_decomposition`
(replaces the constituency scatter), `fig6_uncompensated`, `fig7_policy_deciles`,
`fig8_benefit_status`.

Tables in `paper/tables/`: `tab_decile`, `tab_intra_decile`, `tab_scenario`,
`tab_scorecard`, `tab_region`.

---

## 10. Post-hoc verifications (added after the first prose pass)

### The decile-8 cash peak IS a motor-fuel effect — verified

Mean motor-fuel spend and the share of households recording any, by decile:

| Decile | Motor fuel £ | Domestic energy £ | % with fuel > 0 |
|---|---|---|---|
| 1 | 1,073 | 1,080 | 38.0 |
| 4 | 1,064 | 1,321 | 46.6 |
| 7 | 1,235 | 1,292 | 44.8 |
| **8** | **2,014** | 1,736 | **54.3** |
| 9 | 1,618 | 1,586 | 47.3 |
| 10 | 1,333 | 1,542 | 38.0 |

Decile 8 has both the highest motor-fuel spend and the highest participation, so
the vehicle/mileage explanation for the £-gradient peak is supported, not
speculation.

**But flag the artefact.** Decile 10 records the *same* 38.0% fuel-purchasing
share as decile 1, which is not credible: the richest decile should not look like
the poorest on vehicle use. This is very likely an LCFS-to-FRS imputation
artefact in the fused consumption data, and since motor fuel is 68% of the loss
it is the single largest threat to the £-profile result. Treat the decile-level
*cash* profile as indicative rather than precise, state the artefact in
Limitations, and note that the **percentage** gradient — which is what the paper
concludes from — is robust on medians and does not depend on this.

### The sensitivity CSVs exist; the emitter is stale

`results/sensitivity/{elasticity,cap_lag,asymmetry}.csv` are all present with the
required columns. `analysis/emit_tex_values.py` reports them "not produced"
because it was written before they existed and hardcodes that message. Fix the
emitter; the data is there. `\genElasticityRankCorrHigh` remains genuinely
impossible (constituency-based).

# VALIDATION — our numbers against the published record

> **STABLE REFERENCE — DO NOT RENUMBER.** Docstrings throughout
> `uk_iran_conflict/` and `analysis/` cite this file by check number
> (`docs/VALIDATION.md Check 2b`, `Check 2d`, `Check 4`, ...) to explain why a
> piece of code does what it does. Those labels are part of the replication
> package's interface. Rules:
>
> - **Never renumber a check or a sub-check.** Checks are numbered 1–7; the
>   sub-checks under Check 2 are lettered 2a–2d.
> - **Never delete a check**, and never change what one asserts. Append a dated
>   note instead; the code that cites it was written against the original
>   wording.
> - New checks append at the end with the next free number.
>
> This file is a normative reference for the code, so it ships **inside** the
> replication package.
>
> **Scope note.** The checks below were run against the **first** results tree
> (`docs/FINDINGS.md`, now superseded). They are retained in that form because
> the code cites them and because they are the record of how each defect was
> found. Their *verdicts* — which numbers survive contact with the published
> record and which do not — are what the code depends on, and those still hold.
> The specific "ours" figures quoted in the comparison tables are first-run
> values; the current figures are in `results/` and
> `paper/values_generated.tex`.

Adversarial check of the realised-2026 results (`results/realised_2026/`,
`docs/FINDINGS.md`) against published institutional and academic estimates.
Written as a referee would: the question is not "is our number defensible" but
"what would a hostile reader do with it".

**Verdict up front.** Three results survive contact with the literature (the
aggregate, the elasticity citations, the direction of the decile gradient). Two
do not survive in their current form (the £480 "validation", the 67.8% motor-fuel
share). Two need restating rather than rejecting (the baseline energy level, the
means-tested coverage). One — the decile-level cash and percentage profile — is
materially at risk because it inherits the motor-fuel imputation problem.

Everything below was checked against primary sources in August 2026. Where a
source could not be retrieved this is stated rather than papered over.

---

## Check 1 — the Resolution Foundation £480. **FAILS. Stop citing it as validation.**

| | Value | What it measures |
|---|---|---|
| Ours | £471 | Mean annual household cost of the gas + electricity + motor-fuel price shock, calendar 2026, all 29.5m households, first-order (quantities fixed) |
| RF | £480 | Cash value of a **1.5pp downgrade to real disposable income growth** (+0.9% → −0.6%) for the **median working-age household** over **FY2026-27** |

Source: Mike Brewer, "Higher energy prices could leave typical British
households £480 worse off this year", Resolution Foundation, 15 April 2026 —
https://www.resolutionfoundation.org/comment/higher-energy-prices-could-leave-typical-british-households-480-worse-off-this-year/

Verbatim: *"the typical working-age household currently looks set to be £480
worse off this year than they would have been without the conflict"* and *"the
middling working-age household, previously on track for 0.9 per cent growth, is
now set to see its income dip by 0.6 per cent — a difference of £480."*

**The £480 is not an energy number.** It is a whole-income counterfactual running
through the entire inflation basket — energy, food, second-round effects, nominal
earnings, benefit uprating — for a *median* household of a *restricted*
(working-age) population over a *fiscal* year. Ours is a *population mean* over
*all* households (including pensioners), over a *calendar* year, restricted to
*three energy channels*, with no income-side or second-round effects at all.

Four separate mismatches (concept, statistic, population, period), each worth tens
of pounds and none of them signed the same way. That the two land within £9 of
each other is a coincidence. It is not evidence about our pipeline, and no
referee who reads the RF piece will accept it as such.

There is also an internal-consistency problem if we keep the comparison: our
*median*-household burden is 0.32% of income, and our mean loss of £471 sits
against a median-household figure that is much smaller. Comparing our mean to
RF's median compounds the error in the direction that flatters us.

**Action.** Delete the validation claim from `intro.tex` (line 12) and the
softened version in `results.tex` (line 32), and from FINDINGS §3. If the RF
piece is cited at all it should be as context for the *scale of the macro hit*,
explicitly labelled as a different object. The honest position is that the mean
household loss has **no external validation** — which is a limitation to declare,
not a gap to fill with a coincidence. The genuine external check available to us
is Check 5 (RF's £11bn aggregate), which at least measures the same thing.

---

## Check 2 — motor fuel at 67.8% of the loss. **OVERSTATED. Two compounding errors, one of them ours by construction.**

Our result: motor fuel 67.8% of aggregate loss, gas 17.6%, electricity 14.6%.

### 2a. The spending base does not match ONS

From `results/realised_2026/aggregates.json`: aggregate domestic energy
£39.23bn, aggregate motor fuel £35.71bn — i.e. motor fuel is **47.6%** of
combined household energy+fuel outlay in our microdata.

ONS Family Spending FYE 2025 (LCFS, published 11 June 2026, Table A1) —
https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/bulletins/familyspendingintheuk/april2024tomarch2025

| | ONS £/week | ONS annualised | Ours (per household) |
|---|---|---|---|
| Petrol, diesel and motor oils (7.2.2) | 18.40 | **£960** | **£1,210** (+26%) |
| Electricity, gas and other fuels (4.4) | 35.90 | £1,873 | — |
| — gas + electricity only (4.4.1+4.4.2) | 34.10 | **£1,780** | **£1,330** (−25%) |
| **Motor fuel share of combined** | | **33.9%** | **47.6%** |

Both legs are wrong, and they are wrong in opposite directions, so the error in
the *share* compounds. Our motor-fuel base is a quarter too high and our domestic
base a quarter too low, relative to the survey our consumption data is imputed
from.

### 2b. The shock itself is applied asymmetrically in a way that is not defensible

Backing the implied price uplifts out of the results:

- domestic energy: £4.47bn loss on a £39.23bn base = **+11.4%**
- motor fuel: £9.42bn loss on a £35.71bn base = **+26.4%**

That is by design. `scenarios.py` damps the wholesale gas move by
`REALISED_SUSTAINED_FRACTION = 0.36` so the cap path reproduces Cornwall
Insight's October step, but applies the **peak** pump-price moves (+20% petrol,
+36% diesel) **undamped for a full year** (`notes`: *"Pump-price changes are
observed peaks and are applied directly, without damping"*).

The stated justification — road fuel passes through in weeks, the cap in
quarters — is a claim about *lag*, not about *duration*. It licenses applying the
peak sooner; it does not license applying it for twelve months. Charging
households a 26% annual average pump-price increase off a peak that lasted part
of a year is not a first-order upper bound, it is a different (and much larger)
shock. The domestic side got a sustained-fraction haircut for exactly this
reason. The fuel side did not.

### 2c. What the share would be if both were fixed

Applying our own uplifts (26.4% fuel, 11.4% domestic) to the **ONS** spending
split gives a motor-fuel loss share of **54.3%**, not 67.8%. Applying a damped
pump path on top of that would take it below half.

**Verdict.** The qualitative headline — *this is substantially a pump-price shock
and domestic-bill instruments cannot reach most of it* — survives; motor fuel is
plausibly around half the loss and that is still a novel and policy-relevant
point. The specific figure **67.8% does not survive**, and neither does the
"at most about a third" claim about the reach of bill-based policy, which becomes
"at most about half". FINDINGS §2's framing ("**THE BIG ONE**") needs rewriting
around a range, not a point estimate.

### 2d. The fuel-purchasing incidence artefact is confirmed, and it is worse than suspected

FINDINGS §10 already flags decile 10 recording the same 38.0% fuel-purchasing
share as decile 1. The published data confirms this is an artefact:

- **LCFS recording incidence** (ONS Family Spending FYE 2025, Table A1, "recording
  households in sample"): **2,830 of 5,000 = 56.6%** of households record any
  motor-fuel spend, versus **99.0%** for domestic energy. So our decile figures
  (38–54%) are low against the survey aggregate, and our variation across deciles
  is the wrong shape.
- **Car access by income** (DfT National Travel Survey NTS0703, 2024 data,
  published 27 Aug 2025, England) —
  https://www.gov.uk/government/statistical-data-sets/nts07-car-ownership-and-access
  — **40% of the lowest income quintile has no car** versus **14% of the highest**.
  Car access therefore runs 60% (bottom) to 86% (top). Our imputation gives the
  top and bottom deciles *identical* fuel incidence. It is flatly contradicted.

The consequence is bigger than a footnote. ONS motor-fuel spend by gross income
decile runs **£318 (D1) to £1,362 (D10)** — a 4.3x gradient. Source: ONS Family
Spending in the UK, FYE 2025 (published 11 June 2026), **Table A6, in Workbook 1
"Detailed expenditure and trends"** —
https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/expenditure/bulletins/familyspendingintheuk/april2024tomarch2025
Note that Workbook 2 ("Expenditure by income") contains Tables A4, A5, A7 and
A8 and **does not contain A6**; and the standalone
`ons.gov.uk/...datasets/...tablea6` landing page is **frozen at FYE 2018** and
must not be cited for current figures. Take A6 from Workbook 1 of the current
release. Our imputed figures
run **£1,073 (D1) to £1,333 (D10)** — essentially flat, and **3.4x too high in
decile 1** while roughly correct in decile 10. The imputation is loading motor
fuel onto poor households that do not own cars.

This does not merely blur the cash profile. It inflates the *percentage* gradient
that the paper's central conclusion rests on — see Check 6.

---

## Check 3 — mean domestic energy spend. **TOO LOW, and our own docs disagree with each other.**

First, an internal inconsistency to resolve before anything else:
FINDINGS §3/§10 states mean domestic energy spend **£1,490** (gas £654 +
electricity £836), but `results/realised_2026/aggregates.json` gives
`aggregate_energy_spend_bn = 39.233`, which over 29.5m weighted households is
**£1,330**. These cannot both be right. Whichever is correct, one of them is in
the paper's supporting record and should not be.

Against the published benchmarks:

| Benchmark | Value | Source |
|---|---|---|
| Ofgem price cap, Oct–Dec 2026, typical dual fuel DD | **£1,723** (new TDCV) / £1,935 (old TDCV) | Ofgem, announced 26 Aug 2026 |
| Ofgem cap, Jul–Sep 2026 | £1,663 (new TDCV) / £1,862 (old) | Ofgem |
| DESNZ average annual bill 2025/26 (fixed 3,400 kWh elec / 11,200 kWh gas) | elec £1,077 + gas £821 = **£1,899** | DESNZ *Quarterly Energy Prices*, 30 June 2026 |
| ONS LCFS actual outlay, gas + electricity, FYE 2025 | **£1,780** | ONS Family Spending FYE 2025 |
| **Ours** | **£1,330** (or £1,490) | `aggregates.json` / FINDINGS |

Our figure is **25–30% below the ONS survey measure of actual household outlay** —
and the ONS figure is the right comparator, because it is the same object
(realised spend, all households, including those who use no gas) from the same
survey our consumption data is imputed from. It is also FYE 2025 and therefore
*before* the 2026 shock, so uprating to 2026 widens the gap further.

Is the gap explicable? Partly, and only partly:

- Comparison against the **cap** is legitimately loose. The cap is a
  typical-consumption construct, not a mean bill, and Ofgem cut the TDCV on
  1 July 2026 (electricity 2,700→2,500 kWh, gas 11,500→9,500 kWh). Any 2026 cap
  figure must be labelled old- or new-basis; the same October cap is £1,723 or
  £1,935. **Note this affects our scenario calibration:** `scenarios.py` targets
  Cornwall Insight's £1,729, which is (a) a superseded forecast — Ofgem confirmed
  £1,723 on 26 Aug 2026 — and (b) on the new TDCV basis. If the calibration
  compares a new-TDCV October cap to an old-TDCV reference, the implied step is
  wrong. This should be checked in code.
- Comparison against **ONS LCFS outlay** is *not* loose, and that is where the gap
  is largest. Smaller households, prepayment, and non-gas households are all
  already inside the ONS mean.

**Verdict.** Not fully explicable as a definitional artefact. This looks like a
genuine calibration shortfall in the PolicyEngine energy imputation — plausibly
the NEED kWh calibration combined with Q2-2026 unit rates missing the standing
charges and the actual outlay distribution. It biases the domestic loss **down**,
which is the second half of the compounding error in Check 2, and it makes every
domestic-bill policy in the scorecard (social tariff, JRF block, WHD,
VAT zero-rating) look **cheaper and less effective than it would be on ONS-consistent
spend**. The £13.9bn aggregate is understated on the domestic leg and overstated
on the fuel leg.

---

## Check 4 — means-tested coverage 15.7% (4.6m households). **A large and quantifiable shortfall.**

Ours: 4.6m households (15.7%) — UC 3.03m, Pension Credit 1.22m, Housing Benefit
0.73m, income-based ESA 0.34m.

Administrative caseloads (DWP, Great Britain):

| Benefit | Administrative | Ours | Gap |
|---|---|---|---|
| Universal Credit (May 2026) | **7.2m households** (8.4m people) | 3.03m | **−4.2m, we capture 42%** |
| Pension Credit (Aug 2025) | 1.4m recipients | 1.22m | −0.2m |
| Housing Benefit (Nov 2025) | 1.5m claims (1.1m pension-age) | 0.73m | −0.8m |
| Income-related ESA (Aug 2025) | 0.18m | 0.34m | **+0.16m, we over-count** |
| Income Support | ~0 (discontinued, final release Nov 2025) | — | — |

Sources: DWP Universal Credit quarterly statistics to 14 May 2026 —
https://www.gov.uk/government/statistics/universal-credit-quarterly-statistics-29-april-2013-to-14-may-2026/universal-credit-quarterly-statistics-29-april-2013-to-14-may-2026
; DWP benefit statistics February 2026 —
https://www.gov.uk/government/statistics/dwp-benefit-statistics-february-2026/dwp-benefit-statistics-february-2026

The UC gap is the one that matters. **UC alone is 7.2m GB households against
29.0m UK households (ONS, *Families and households in the UK: 2025*) — about
25%.** Our figure for the *entire* means-tested population, 15.7%, is well below
the caseload of the single largest means-tested benefit. Allowing for overlap,
the true means-tested household share is plausibly 30%+ (own calculation — no
official total exists; DWP does not publish one and the counting units differ
across benefits, so this should be labelled as our arithmetic, not cited as a
statistic).

Two caveats that do not rescue the number: DWP is GB and ONS households is UK
(a ~3% denominator effect, not a factor-of-two effect); and UC is a caseload at a
point in time whereas ours is annual receipt, which if anything should push *our*
number **up**, not down.

**Verdict.** FINDINGS §8's framing — "reflecting modelled take-up", a limitation
to declare — understates it. Under-covering UC by 4.2m households is not a
take-up nuance; it is a factor-of-2.4 miss on the largest means-tested caseload
in the country, and it should be raised as a **data-quality finding about
PolicyEngine UK** (in the same spirit as the missing price channel) rather than
buried in Limitations.

Two results depend on it directly and should be labelled as bounds, not
estimates: the social tariff's and WHD's uncompensated shares (84.8% and 87.7%)
are **upper bounds**, and the "98% of households losing more than 5% of income are
outside the means-tested system" claim is the most fragile number in the paper —
it is an upper bound computed from a population we know is missing well over half
of UC. FINDINGS §8 is right that it must not be conflated with JRF's ~40%; it
should also not be stated as 98% without the bound language attached.

---

## Check 5 — aggregate £13.9bn. **Broadly reconciles with RF's £11bn — this is our real external check.**

RF: *"a sustained return to these peaks would see British households spending an
extra £11 billion on fuel and energy in 2026 than if prices had remained at
early-2026 levels"* — Pittaway, Smith & Thwaites, *The Macroeconomic Policy
Outlook Q2 2026*, 22 April 2026 —
https://www.resolutionfoundation.org/publications/the-macroeconomic-policy-outlook-q2-2026/

This is the right comparator, and much better than the £480:

- same object (extra household spending on energy **and road fuel**)
- same period (calendar 2026)
- same population (all British households)
- same counterfactual family (versus pre-shock prices)
- RF gives it as ~0.5% of aggregate household income; ours is 0.77%

Ours is **26% higher** than RF's, on a scenario RF describes as a *sustained
return to peaks* — i.e. RF's £11bn is the aggressive case, and we are above it.
That is consistent with Check 2b: we apply peak pump prices for a full year. The
gap is roughly the size of the undamped fuel leg.

JRF's £288 — *"Current predictions suggest a possible £288 annual increase"*,
Moore & Cook, *Addressing the 2026 energy price crisis*, JRF, 9 April 2026,
https://www.jrf.org.uk/cost-of-living/addressing-the-2026-energy-price-crisis —
is a **typical dual-fuel domestic bill** increase, not a household total. Compare
it to our *domestic-only* loss: £4.47bn / 29.5m = **£151 per household**, or
about half JRF's £288. Consistent with our domestic base being ~25% low
(Check 3) and our cap uplift being modest at +11.4%. Do **not** compare £288 to
our £471 — different objects, and the comparison would look like agreement by
accident, the same mistake as Check 1.

JRF's £5bn universal block costing versus our £1.9bn: FINDINGS §5 already handles
this correctly (our parameterisation is less generous). Worth adding that JRF's
£5bn is benchmarked to an EPG-style cap at the April 2026 price cap
(£3.7bn gas / £1.3bn electricity), so the objects differ in generosity *and* in
what they are pegged to.

**Verdict.** Agrees within the range one would expect given the method
differences, and the direction of the discrepancy is explained by a known feature
of our own specification. **This should replace the £480 as the paper's external
check**, stated with the 26% gap and its cause, not as "close agreement".

---

## Check 6 — the decile gradient (5.75x aggregate / 6.1x median). **Defensible on the income basis, but at risk from the Check 2d artefact.**

Published UK comparators:

| Source | Measure | D1 | D10 | Ratio |
|---|---|---|---|---|
| NEF, Feb 2022 | *Increase* in energy costs, % of disposable income (Apr-22 cap) | 6% (£724) | 0.75% | **7.5x** |
| ONS, Feb 2022 | Gas+elec spend, % of disposable income (level) | 7% | 2% | 3.5x |
| IFS R85 (2013) | Energy budget **share** of expenditure | 15.8% | 3.3% | 4.8x |
| IFS R85 (2013) | Cost-of-living effect of a 5% energy price rise | 0.8% | <0.2% | >4x |
| IFS, Aug 2022 | Gas+elec share of budget | 11% | 4% | 2.75x |
| RF, Aug 2022 (*A chilling crisis*) | Energy share of expenditure, **net of £400 EBSS** | 14% | 8% | 1.75x |
| **Ours** | Loss as % of net income, aggregate ratio | 2.07% | 0.36% | **5.75x** |

Sources: NEF https://neweconomics.org/2022/02/poorest-10-of-families-will-see-energy-costs-increase-by-724-a-7-5-times-larger-rise-than-the-richest-10-of-families
; ONS https://www.ons.gov.uk/economy/inflationandpriceindices/articles/energypricesandtheireffectonhouseholds/2022-02-01
; IFS R85 https://ifs.org.uk/sites/default/files/output_url_files/r85.pdf
; RF https://www.resolutionfoundation.org/app/uploads/2022/08/A-chilling-crisis.pdf

**On the income basis our 5.75x sits inside the published bracket** — above ONS's
3.5x level ratio, below NEF's 7.5x, and NEF is the closest comparator (a 2022
*shock*, % of disposable income, D1 vs D10). On an **expenditure** basis 5.75x
would exceed every published UK figure. The paper must state its denominator
loudly and repeatedly; it is net income, and that is why the number is at the top
of the range. Referees will also ask whether the figure is gross or net of policy
support — RF's 1.75x is net of the £400 EBSS, which flattens it enormously — so
say explicitly that ours is the **pre-policy** gradient.

**The threat is not the denominator, it is the numerator.** FINDINGS §4 defends
the gradient against the "division by small income" attack, and that defence
holds (median-based ratio 6.1x, D1 median income £16,000, only 0.57% at zero or
negative). But it defends against the wrong attack. The live problem is Check 2d:
motor fuel is 68% of our loss and our imputation gives decile 1 **£1,073** of
motor-fuel spend where ONS gives **£318**. Decile 1's loss is inflated by fuel it
probably does not buy.

Rough magnitude, applying our own price uplifts to ONS decile spending and our
implied decile mean incomes: D1 burden falls from 2.07% to about **1.2%** and the
D1:D10 ratio falls from 5.75x to roughly **2.8x**. That is an illustrative
recalculation, not a re-run — ONS deciles are unequivalised *gross* income
deciles and ours are net-income deciles, so the two rankings are not the same
households, and the true correction will be smaller than this. But the direction
is unambiguous and the magnitude is large enough that "the gradient is robust"
cannot be asserted on the median check alone.

**Verdict.** The *sign* and the *existence* of a steep gradient are safe — every
source in the table agrees energy shocks are regressive in income terms, and the
domestic-energy leg alone (which is our better-behaved channel) delivers it. The
*magnitude* of 5.75x is not safe, because most of it is carried by the channel
with the broken imputation. This should be reported with an explicit sensitivity
showing the gradient computed on the domestic-only loss.

Finally, on Fetzer, Gazze & Bishop (2024), *Economic Policy* 39(120):711 —
https://academic.oup.com/economicpolicy/article/39/120/711/7709888 — the paper
confirms affluent areas are more exposed **in absolute terms**, and the authors
state explicitly (p.4) that *"we can only conduct our distributional analysis in
absolute terms, as we cannot compute energy expenditures as a share of income."*
So FGB does **not** contradict our percentage gradient and cannot be used to
support or undercut it. The paper should say so rather than presenting the two as
"crossed" incidence facts — the crossing is real, but FGB supplies only one arm of
it and disclaims the other.

---

## Check 7 — elasticity citations. **PASS. Labandeira verified verbatim; one Priesmann label needs care.**

**Labandeira, Labeaga & López-Otero (2017)**, *Energy Policy* 102:549–568,
"A meta-analysis on the price elasticity of energy demand" —
https://www.sciencedirect.com/science/article/abs/pii/S0301421517300022

Table 6 confirmed from the full text:

| Product | Short run | Long run |
|---|---|---|
| Electricity | −0.126* | −0.365* |
| Natural gas | −0.180*** | −0.684* |
| Gasoline | −0.293*** | −0.773*** |
| Diesel | −0.153** | −0.443*** |
| Heating oil | −0.017 (n.s.) | −0.185 (n.s.) |

`uk_iran_conflict/elasticity.py` states all of these correctly, including the
diesel value and the significance levels, and correctly records that the prior
repo's "−0.15" attribution is not in the paper. **The prior repo's error is not
repeated here.** For completeness: the paper's actual headline is **−0.21 short
run / −0.61 long run** (Table 5 aggregate energy, and repeated in the
conclusion), and "−0.15" appears only as the Table 4 raw average for "Energy"
(−0.149) and in the discussion of *diesel*. If the paper cites a single
Labandeira number anywhere in prose, it must be −0.21/−0.61, not −0.15.

**Priesmann & Praktiknjo (2025)**, *Energy Policy* 207:114850, "Estimating short-
and long-run price and income elasticities of final energy demand as a function
of household income" — https://www.sciencedirect.com/science/article/pii/S030142152500357X
— exists as cited. The endpoints in `elasticity.py` match the published
abstract exactly:

- electricity SR −0.27 (low income) → −0.44 (high income); LR −0.22 → −0.64
- gas SR −0.64 → −0.11; LR −0.58 → −0.15
- car fuels −0.47 → −0.14, not separated short/long

**One caution.** `results/sensitivity/elasticity.csv` carries
`epsilon_mean = −0.3205` (Priesmann short run) and `−0.3450` (long run) in rows
whose `source` column reads "Priesmann & Praktiknjo (2025)". Those means are
**ours** — spend-weighted averages of our own linear interpolation across
deciles — and do not appear in the paper. The CSV already says "published
endpoints, interior deciles interpolated", which is honest, but if either number
reaches the paper's prose or a table it must be labelled a derived quantity. A
referee checking Priesmann for "−0.32" will not find it.

**Two other citation errors found while checking, both currently in
`RESEARCH_BRIEF.md` and therefore likely in `references.bib`:**

1. **"Advani, Bassi, Levell & Rasul (IFS R85, 2013)" is wrong twice over.**
   R85 is *Household Energy Use in Britain: A Distributional Analysis* by
   **Advani, Johnson, Leicester & Stoye**. The title in the brief ("Energy use
   policies and carbon pricing in the UK") belongs to **R84**, by Advani, Bassi,
   Bowen, Fankhauser, Johnson, Leicester & Stoye. **Levell and Rasul are not
   authors of either.** R85 is the distributional report and the one we want.
   https://ifs.org.uk/sites/default/files/output_url_files/r85.pdf
2. **The "7.7pp" figure is not in IFS W25/21.** W25/21 (Chen, Levell &
   O'Connell, May 2025, *Measuring cost of living inequality during an inflation
   surge*) reports **5.5pp** between the 1st and 4th expenditure quartiles. The
   7.7pp decile figure is in the earlier **W24/36, *Cheapflation and the rise of
   inflation inequality*** (Aug 2024). Both caveats matter: it is **cumulative**
   over nine quarters (2021Q3–2023Q3), and it is **grocery scanner data only** —
   it is not an energy result and should not be used as if it were.
   https://ifs.org.uk/sites/default/files/2024-08/Cheapflation-and-the-rise-of-inflation-inequality_1.pdf

---

## Ranked list: what is most at risk

**1. The 67.8% motor-fuel share, and everything built on it.**
Two compounding errors: a spending base that is 40% too fuel-heavy relative to
ONS, and a shock that applies peak pump prices undamped for a full year while
damping the domestic side by 0.36. ONS-consistent spending shares with our own
uplifts give ~54%. *Change required:* re-run with the pump path damped on the
same logic as the cap (or report both), report the fuel share as a range, and
replace "at most about a third" with "at most about half" wherever bill-based
policy reach is discussed. If the re-run is not feasible, the share must be
presented as an upper bound with the undamped-peak assumption stated in the
sentence that reports it.

**2. The decile-1 burden of 2.07% and the 5.75x gradient.**
Contaminated by the same imputation. ONS puts D1 motor-fuel spend at £318; we
impute £1,073. DfT NTS0703 puts 40% of the bottom quintile without a car against
14% of the top, while our imputation gives both deciles an identical 38%
fuel-purchasing rate. *Change required:* report the gradient computed on the
domestic-energy loss alone as the robustness anchor; present the all-channel
gradient as an upper bound; move the imputation artefact from a Limitations
footnote into the results discussion.

**3. The £480 "external validation".**
Measures a different thing in four separate ways. *Change required:* delete from
`intro.tex` and `results.tex`, replace with the RF £11bn comparison and its
honest 26% gap.

**4. The £1,330 / £1,490 baseline domestic energy spend.**
25–30% below the ONS survey measure of actual outlay, and our two internal
records disagree with each other. Biases the domestic loss down and makes every
domestic-bill policy look cheaper than it is. *Change required:* resolve the
internal inconsistency; benchmark the imputation against ONS Family Spending FYE
2025 Table A6 (Workbook 1, "Detailed expenditure and trends" — not Workbook 2,
and not the standalone `tablea6` dataset page, which is frozen at FYE 2018) and
report the ratio in the appendix; if the shortfall is real,
state that domestic-leg results are conservative by roughly a quarter.

**5. The 15.7% means-tested coverage and the 98% uncompensated claim.**
UC alone is 7.2m households; we model 3.03m. *Change required:* promote from
Limitations to a stated data finding; label the social tariff and WHD
uncompensated shares as upper bounds; attach bound language to the 98%.

**6. The scenario calibration against £1,729.**
A superseded Cornwall forecast (Ofgem confirmed £1,723 on 26 Aug 2026), and on
the new TDCV basis introduced 1 July 2026 (elec 2,700→2,500 kWh, gas
11,500→9,500 kWh). *Change required:* verify the calibration compares like with
like on TDCV basis, and update the citation to the confirmed cap. Also note
Ofgem's temporary removal of VAT on electricity from 1 Oct 2026, which holds the
October cap ~£45 lower and breaks comparability with earlier quarters — this
interacts directly with our VAT zero-rating policy scenario and may already be
partly in the baseline.

**7. Two bibliography errors** (IFS R85 authorship/title, and 7.7pp attributed to
W25/21 rather than W24/36, cumulative and grocery-only). Cheap to fix, embarrassing
if a referee finds them.

**Not at risk:** the elasticity module's Labandeira and Priesmann values (Check 7
— verified, and the prior repo's error is not repeated); the aggregate £13.9bn as
a reconcilable figure against RF's £11bn (Check 5); the *direction* of the decile
gradient (Check 6); the cap-lag invariance result and the finding that the
gas/electricity asymmetry does not move headline incidence, both of which are
internal and do not depend on any external benchmark.

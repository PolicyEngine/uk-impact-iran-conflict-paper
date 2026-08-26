# Sensitivity and robustness results

Produced by `analysis/run_sensitivity.py` from the private PolicyEngine UK
microdata (535,080 households, 29.5m weighted). Every sweep uses the
**`realised_2026`** scenario — retail gas ×1.1404, retail electricity ×1.0928 —
and the same baseline, loaded once. Percentages of income are always the
**aggregate ratio** (weighted loss ÷ weighted income), never a mean of
household-level ratios.

Headline for reference: under the paper's main specification the realised 2026
shock costs a mean of **£471 per household (0.77% of net income)**, **£13.9bn**
in aggregate, and runs from **2.07% of income in decile 1** to **0.36% in
decile 10** — a 5.7:1 ratio in percentage terms, while the £ loss runs the
other way (£395 in D1 against £539 in D10). That £-versus-% reversal is the
Fetzer–Gazze–Bishop point, and it survives every sweep below.

---

## 1. `elasticity.csv` — how much of the upper bound substitution shaves off

The main specification is zero elasticity: Δcost = q₀·Δp, the first-order
Deaton approximation and an explicit **upper bound** on the compensating
variation. The sweep applies the constant-elasticity spend factor
(p₁/p₀)^(1+ε) — never the linear form — over a flat grid from 0 to −0.8, plus
the five named specs in `uk_iran_conflict.elasticity`.

| spec | mean ε | mean loss £ | % of income | D1 £ / % | D10 £ / % | £bn | upper bound shaved |
|---|---|---|---|---|---|---|---|
| **zero (main)** | 0.00 | **471** | **0.77** | 395 / 2.07 | 539 / 0.36 | **13.89** | — |
| flat −0.1 | −0.10 | 420 | 0.69 | 352 / 1.84 | 480 / 0.32 | 12.36 | 11% |
| flat −0.2 | −0.20 | 369 | 0.61 | 309 / 1.62 | 422 / 0.28 | 10.87 | 22% |
| flat −0.3 | −0.30 | 319 | 0.53 | 268 / 1.40 | 365 / 0.24 | 9.42 | 32% |
| flat −0.4 | −0.40 | 271 | 0.45 | 227 / 1.19 | 310 / 0.21 | 7.99 | 43% |
| flat −0.5 | −0.50 | 223 | 0.37 | 187 / 0.98 | 255 / 0.17 | 6.59 | 53% |
| flat −0.6 | −0.60 | 177 | 0.29 | 148 / 0.78 | 202 / 0.14 | 5.21 | 62% |
| flat −0.7 | −0.70 | 131 | 0.22 | 110 / 0.58 | 150 / 0.10 | 3.87 | 72% |
| flat −0.8 | −0.80 | 87 | 0.14 | 73 / 0.38 | 99 / 0.07 | 2.55 | 82% |
| Labandeira short run | −0.22 | 344 | 0.57 | 288 / 1.51 | 394 / 0.26 | 10.15 | 27% |
| Labandeira long run | −0.63 | 134 | 0.22 | 111 / 0.58 | 153 / 0.10 | 3.94 | 72% |
| Priesmann short run | −0.32 | 320 | 0.53 | 200 / 1.05 | 435 / 0.29 | 9.42 | 32% |
| Priesmann long run | −0.34 | 314 | 0.52 | 206 / 1.08 | 415 / 0.28 | 9.25 | 33% |
| prior-repo replication | −0.34 | 304 | 0.50 | 133 / 0.70 | 474 / 0.32 | 8.95 | 36% |

(ε is the spend-weighted mean elasticity implied by the spec; for the
income-varying specs it is a summary, not a parameter.)

**What it shows.** This is the sweep that moves the answer most, and it moves
it a long way: the aggregate loss runs from £13.9bn at ε = 0 to £2.6bn at
ε = −0.8. The relationship is close to linear in ε — each 0.1 of elasticity
removes roughly a tenth of the upper bound (11%, 22%, 32%, 43%, …). The
credible short-run band, the one that matters for a within-year price spike,
is narrower: Labandeira's short-run per-carrier means (−0.18 gas, −0.13
electricity, −0.29 gasoline) shave **27%**, and Priesmann's income-varying
short run shaves **32%**. So a reader who rejects the zero-elasticity headline
outright should still land near **£10bn and a mean loss around £320–£345**, not
somewhere qualitatively different. Labandeira's *long*-run figures shave 72%,
but they embed appliance and vehicle stock turnover and are the wrong horizon
for a 2026 spike; they are a band, not a candidate headline.

**What it does not change.** Nothing about the distributional story. The flat
grid rescales every decile by the same factor, so the D1/D10 percentage ratio
stays at 5.7:1 throughout. The income-varying specs *do* flatten the gradient —
Priesmann takes D1 from 2.07% to 1.05% against D10's 0.29%, a ratio of 3.6:1,
and the prior repo's replication (which wrongly applies the gas gradient to
electricity) flattens it further to 2.2:1 — because poor households are modelled
as the ones with room to cut back. That is exactly the "heat or eat" objection:
the flattening is an artefact of treating involuntary rationing as
welfare-neutral substitution, which is the substantive reason the paper keeps
the upper bound as its headline. Either way the ordering is untouched: the poor
lose more in percentage terms and less in cash in every single row.

---

## 2. `cap_lag.csv` — genuine effect versus windowing artefact

`CAP_LAG_QUARTERS = 3` places the phase-in profile (0.35, 0.85, 1.00, 0.90) on
the calendar starting at 2026Q1 + lag. The domestic-energy (cap) channel is
lagged; motor fuel is not, because pump prices pass through in weeks — a
£9.4bn-a-year channel that arrives immediately regardless of the lag.

| lag (qtrs) | first cap quarter | phase-in qtrs inside 2026 | annualised mean £ | annualised £bn | cumulative mean £ | cumulative £bn | annualised share of a full-pass-through year | cumulative share |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026Q2 | 3 | 403 | 11.88 | 437 | 12.88 | 55% | 78% |
| 2 | 2026Q3 | 2 | 365 | 10.76 | 437 | 12.88 | 30% | 78% |
| **3 (paper)** | **2026Q4** | **1** | **333** | **9.81** | **437** | **12.88** | **9%** | **78%** |
| 4 | 2027Q1 | 0 | 320 | 9.42 | 437 | 12.88 | 0% | 78% |

**What it shows.** The two columns tell opposite stories, and that is the whole
point of reporting both. The **cumulative** loss is *exactly invariant* to the
lag — £437 per household, £12.9bn, at every value. A lag shifts the path in
time; it does not change its size. The **annualised** 2026 figure falls from
£403 to £320 as the lag lengthens, and the domestic channel inside 2026 collapses
from 55% of a full-pass-through year to 0%. That entire movement is a windowing
artefact: at the paper's lag of 3, only 2026Q4 (phase-in 0.35, one quarter of a
year) falls inside 2026, so just **9%** of the eventual domestic pass-through is
booked to the calendar year — the shock is overwhelmingly a 2027 event that the
2026 window happens to clip.

**How to read the paper's numbers.** The lag choice is contestable but it is
almost entirely a question of *which year the bill lands in*, not how big it is.
Any statement of the form "the 2026 loss is X" is lag-sensitive by roughly ±£40
per household across the plausible 1–4 quarter range; any statement about the
total burden of the shock is not sensitive at all. The annualised figures are
also dominated by motor fuel (£9.4bn of the £9.8bn at lag 3), which is precisely
because the fast pump channel and the slow cap channel arrive in different years
— an asymmetry worth stating as a result rather than hiding in a footnote. The
appendix should quote the cumulative figure as the burden of the shock and the
annualised figure only as a calendar-year accounting statement.

---

## 3. `asymmetry.csv` — how much the gas/electricity split actually matters

`MARGINAL_PRICING_SHARE = 0.85` is the share of the time gas sets the GB
marginal electricity price; setting it to 1.0 recovers the naive symmetric
assumption that a wholesale gas move reaches electricity undamped.

| share | gas ×  | elec × | mean £ | % income | D1 % | D10 % | gradient (pp) | D1/D10 ratio | gas share of domestic loss | elec share | £bn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.70 | 1.1404 | 1.0764 | 459 | 0.75 | 2.02 | 0.35 | 1.66 | 5.73 | 59.3% | 40.7% | 13.53 |
| **0.85 (paper)** | 1.1404 | 1.0928 | **471** | **0.77** | **2.07** | **0.36** | **1.71** | **5.72** | **54.6%** | **45.4%** | **13.89** |
| 1.00 | 1.1404 | 1.1092 | 483 | 0.79 | 2.12 | 0.37 | 1.75 | 5.71 | 50.5% | 49.5% | 14.24 |

**What it shows — stated plainly: it barely moves the answer.** Going all the
way from 0.70 to the naive 1.00 changes the mean loss by **£24 (5.2%)**, the
aggregate by £0.7bn, and the loss as a share of income by 0.04pp. The decile
gradient moves from 1.66pp to 1.75pp and the D1/D10 percentage ratio is
essentially frozen at 5.72 across the whole range. Nothing in the paper's
distributional conclusions turns on this parameter.

**Where it does bite** is the composition of the loss, which is the one column
that moves materially: the gas share of the domestic-energy loss falls from
59.3% to 50.5% and electricity rises from 40.7% to 49.5% as the parameter goes
from 0.70 to 1.00. At the naive symmetric assumption the shock looks like an
essentially even gas/electricity event; at the paper's 0.85 it is meaningfully
gas-weighted. That matters for **policy design, not for headline incidence** —
a gas-only social tariff, an electricity-side levy rebate, or the JRF block's
choice of which unit rate to discount all land differently depending on which
fuel carries the loss, whereas the aggregate cost and the decile profile do not.

**The honest summary for the appendix.** The asymmetry is defensible on its
merits and it is the right modelling choice, but it should be presented as a
*compositional* refinement rather than as load-bearing for the paper's headline
result. A referee who insists on the symmetric assumption gets a 5% larger mean
loss and the same distributional conclusions. The claim to defend is not "the
asymmetry changes the answer" — it does not — but "the asymmetry is what makes
the fuel-level policy analysis meaningful." Note also that the asymmetry sweep
moves the answer far less than the elasticity sweep (5% versus up to 82%): if
the appendix ranks its uncertainties, demand response comes first, the
calendar-year lag second, and the marginal-pricing share a distant third.

---

## Files and schema notes

- `elasticity.csv` — 14 rows, named specs first then the flat grid in
  increasing magnitude, so the last row is the strongest-response variant.
  Carries `exchequer_cost_bn` and `mean_household_change` aliases for
  `analysis/emit_tex_values.py`. It does **not** carry
  `constituency_rank_corr`: the dataset has no constituency weight matrix and
  its `local_authority` column is degenerate, so no constituency statistic is
  producible from this release and none has been invented. That macro will
  emit `\GENMISSING` until the emitter is pointed elsewhere.
- `cap_lag.csv` — 4 rows; `annualised_share` and `cumulative_share` are shares
  of a full-pass-through year of the *domestic* channel, the quantity the lag
  acts on.
- `asymmetry.csv` — 3 rows; `gas_share_of_domestic_loss` excludes motor fuel so
  the fast pump channel does not dilute the fuel-split comparison.

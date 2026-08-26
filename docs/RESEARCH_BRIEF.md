# Research brief — shared context for all contributors

> **SUPERSEDED IN PART — read this first.** This is the *original* scoping brief,
> kept for provenance. Several of its claims were disproved once the pipeline was
> built against the real model and data. Where this file and the documents below
> disagree, **they win**:
>
> - `docs/FINDINGS.md` — what the first full run actually showed
> - `docs/VALIDATION.md` — external validation against ONS, DfT, DWP and the literature
> - `docs/REFEREE_REPORTS.md` — three referee reports, all major revision
> - `docs/FIXES.md` — the resulting work list and the decisions taken
>
> Specifically, and contrary to this brief: PolicyEngine UK has **no price channel**
> (energy and fuel spend are inputs with no formulas); the `energy_bills` parameter
> is **dead** — no variable reads it; there is **no Warm Home Discount**; there are
> **no constituency weights** in this data release and `local_authority` is
> degenerate, so the seat-level contribution below was withdrawn; and the
> distributional headline is now reported across seven specifications rather than
> one. The one-sentence contribution stated below is **not** the paper's
> contribution any more.

## The paper

**Title:** From Hormuz to the Household: Macro-to-Micro Incidence of the 2026 Energy Shock in the UK

**Author:** Vahid Ahmadi (Research Associate, PolicyEngine), vahid@policyengine.org

**One-sentence contribution:** Nobody has built a macro→microsimulation link for the
UK. PolicyEngine's 650-constituency weight sets can produce a distributional output —
£-loss versus %-of-income-loss at parliamentary-seat level — that no existing UK
analysis can, alongside within-decile uncompensated-loser counts.

## The shock (context, as of August 2026)

US/Israel–Iran war from ~28 Feb 2026; Strait of Hormuz effectively closed at points.

- UK gas peaked **+78p/therm** above pre-war (vs +300p in 2022)
- Oil peaked **+$42/bbl (+57%)**, comparable to 2022's +$35
- Petrol **+20%**, diesel **+36%**
- IMF and OECD cut UK 2026 growth by **0.5pp — the largest G7 markdown**
- Gas is **62% of UK final household energy consumption (highest in G7)** and sets
  the electricity price **~85% of the time**

## Institutional positions

| Source | Date | Finding |
|---|---|---|
| Resolution Foundation (Brewer) | 15 Apr 2026 | Typical working-age household **£480 worse off** in 2026-27. Poorest fifth income growth cut 2.8%→1.2%; median +0.9%→−0.6%. |
| RF Macro Policy Outlook Q2 | 22 Apr 2026 | £11bn extra household energy/fuel spend at sustained peaks. Severe scenario: GDP −0.9% over 3yrs, borrowing +£16bn in 2029-30. Recommends targeted temporary bill discounts, not a new EPG (~£20bn/yr). |
| JRF (Moore, Cook) | 9 Apr 2026 | Wholesale gas +65%; ~£288 bill rise. Proposes **universal discounted block** = 50% of typical consumption at a discounted rate + per-child allowance, **~£5bn**, fully offsets cost rise for deciles 1–3. Rejects social tariff as infeasible for winter 2026. Calls EPG regressive in cash terms at £23bn. |
| NIESR Spring 2026 | Apr 2026 | GDP 0.9% (2026), CPI avg 3.0% peaking **4.1% Jan 2027**; adverse $140/bbl scenario → CPI ~5%, possible recession H2-26/H1-27. |
| NIESR Summer 2026 | Jul 2026 | Baseline GDP 1.1% 2026 and 2027, CPI avg 3.1% peaking 3.8% Feb 2027. Downside (oil and gas +50% sustained): GDP ~0.2pp lower, CPI peaking near 5%. |
| IPPR | Apr 2026 | Claw back network-company windfalls → **£183 household rebate**. |
| Cornwall Insight | 19 Aug 2026 | Oct-26 cap **£1,729**, +4%. |

**Key political fact:** Reeves has ruled out universal support and signalled
income-based help (a social tariff). JRF/TUC counter with the universal discounted
block. The anchor statistic in every submission: **~40% of households struggling to
heat their home are not on means-tested benefits.** Warm Home Discount qualifying
date was 23 Aug 2026; £150, now automatic in England and Wales, property-cost test
scrapped, +2.7m households. Autumn Budget 2026 is the decision point.

## Literature

**Canonical / must-cite**

- **Levell, O'Connell & Smith (2025)**, IFS WP W25/55 / CEPR DP19939, conditionally
  accepted *JPE*. Mean welfare loss 6% of income absent policy; the actual 2022-23 UK
  package cut mean and dispersion of losses at an efficiency cost of 12% of its
  revenue cost; **optimal policy includes a strictly positive price subsidy**,
  shrinking as transfers can target income *and past energy usage*. THE reference.
- **Känzig (2023)**, NBER WP 31221. Poorer households lose more mainly through
  **income/employment**, not budget shares; GE/indirect effects are ~2/3 of the
  response. The main methodological threat to any static microsimulation — state it.
- **Fetzer, Gazze & Bishop (2024)**, *Economic Policy* 39(120):711. 22.2m EPCs + NEED
  meter data. **More affluent areas are more exposed in £ terms**; the EPG
  disproportionately benefited them. Proposes a two-tier tariff.
- **Cronin, Fullerton & Sexton (2019)**, *JAERE*. Within-decile horizontal
  redistribution exceeds between-decile vertical redistribution.
- **Douenne (2020)**, *Energy Journal*. Flat recycling progressive on average but
  large horizontal losers among rural/off-gas-grid poor. **No UK counterpart exists.**
- **Sallee (2019)**, NBER WP 25831. With observable-based transfers, compensating all
  losers is infeasible.
- **Ohlendorf et al. (2021)**, *ERE*. Meta-analysis: progressivity findings depend on
  method.
- **Goulder et al. (2019)**, *JPubE*. Use-side regressive, source-side progressive;
  microsimulation captures only the use side.
- **Pallotti, Paz-Pardo, Slacalek, Tristani & Violante (2024)**, *JME*. Nominal net
  asset positions, not the energy basket, drive most euro-area heterogeneity.

**Macro / geopolitical**

- **Caldara & Iacoviello (2022)**, *AER*. The GPR index.
- **Caldara, Conlisk, Iacoviello & Penn (2025)**, Fed Board WP. GPR **raises**
  inflation while lowering activity; commodity-price and currency channels dominate.
- **Verduzco-Bustos & Zanetti (2026)**, CFM DP 2026-08. Isolates geopolitical events
  that *reduce oil supply*; oil **inventories** the key propagation margin.
- **BoE SWP 1118 (Feb 2025)**, "Geopolitical risk shocks: when size matters."
  Separates "Acts" from "Threats"; Acts produce *negative* oil price and inflation
  responses. Contradicts the naive stagflation prior — a useful foil.
- **Kilian (2009)**, *AER*; **Baumeister & Hamilton (2019)**, *AER*; **Hamilton
  (1983)**, *JPE*. Structural oil-market foundations.
- **Dallas Fed WP 2609 (Apr 2026)**. 2026 Iran war 2–3x the size of the 1973 and 1990
  disruptions; US GDP response ~1/6th of the rest-of-world response. The UK
  counterpart is unwritten.
- **IMF WP 2026/026**, "From Ports to Prices." AIS-based port-to-port shipping times →
  a 100-hour delay raises inflation ~0.5pp at a five-month peak.
- **World Bank CMO (Apr 2026)**; **IMF WEO (Apr 2026)**, "Global Economy in the Shadow
  of War."

**Grey but heavily cited:** IMF WP/22/152 (Ari et al. 2022 — the "compensate incomes,
not prices" consensus); Bruegel fiscal tracker (only ~1/3 of European support was
targeted); Amores et al. (JRC 2023, EUROMOD); ECB EB 2/2023.

**UK baseline:** Advani, Bassi, Levell & Rasul (IFS R85, 2013); IFS W25/21 (top
expenditure decile faced inflation 7.7pp lower than the bottom at peak); ONS Household
Costs Indices.

**Method:** Deaton & Muellbauer (AIDS 1980); Banks, Blundell & Lewbel (QUAIDS 1997);
O'Donoghue et al. PRICES framework (arXiv 2310.00231).

**The publishable tension:** the grey-literature consensus (IMF/Bruegel: never
subsidise prices) is contradicted by the peer-reviewed frontier (Levell–O'Connell–Smith:
the optimum is a mix). Second gap: **no well-cited UK estimate of wholesale→retail
pass-through in the price-cap era.**

## PolicyEngine UK — what exists and what does not

**Exists (more than expected):**

- `variables/input/consumption/energy.py`: `domestic_energy_consumption`,
  `electricity_consumption`, `gas_consumption` — LCFS-imputed, calibrated to NEED 2023
  kWh, priced at Ofgem Q2-2026 rates.
- **Energy Price Guarantee fully implemented**: `gov/treasury/price_cap_subsidy/`
  against `parameters/gov/ofgem/energy_price_cap.yaml` (real Ofgem cap series). Plus
  the 2022 `energy_bills_rebate/` (£400 credit + council tax rebate).
- **A ready-made shock lever**:
  `parameters/gov/contrib/policyengine/economy/energy_bills.yaml` — "raise energy
  spending by this percentage", default 0.
- **Fuel duty** at household level: litres = spending ÷ modelled pump price, with
  DESNZ price parameters at
  `parameters/household/consumption/fuel/prices/{petrol,diesel}.yaml`.
- **Carbon tax module** (contrib/ubi_center) with sector carbon intensities over 12
  COICOP categories and a consumer/shareholder incidence split.
- **650-constituency and local-authority weight sets** (Nuffield-funded).
- Base is **FRS 2023-24**, fused by quantile regression forests with WAS (wealth),
  LCFS (12 COICOP categories, petrol/diesel, gas and electricity separately), ETB (VAT
  expenditure rate), SPI (top incomes), Advani–Summers (capital gains). Weights
  re-optimised by gradient descent against **1,512 administrative targets**. Uprated on
  OBR EFO March 2026.

**Outputs:** budgetary (incl. multi-year), income-decile and wealth-decile
average/relative impacts, intra-decile winners/losers, poverty and deep poverty by
child/working-age/senior across four UK measures (rel/abs × BHC/AHC), Gini + top
1%/10%/bottom 50% shares, cliff analysis, labour-supply response, constituency/LA hex
maps.

**Gaps — state these explicitly in the paper:**

1. **VAT enters distribution only as a delta.** `household_tax.py` includes
   `vat_change`, *not* `vat`. Baseline VAT is never deducted from net income. Fuel duty
   and carbon tax do enter at level.
2. VAT rests on a **0.38 coverage grossing factor** (`microdata_vat_coverage.yaml`,
   following IFS TAXBEN) and a 4-predictor ETB imputation. Issue **#352** open.
3. **No consumption elasticity or price pass-through.** Open issue **#1114** (notes
   UKMOD's TCO uses 0.8).
4. **No price block at all.** CPI/CPIH/RPI exist only as uprating indices.
5. **No missing duties:** alcohol, tobacco, APD, IPT, VED all absent.
6. Static, no GE, no lifecycle; WAS vintage is round 7 (2018-20); take-up is a scalar;
   constituency weights are synthetic re-optimisations of ~20k households; **microdata
   sits in a private Hugging Face repo requiring a token**.

**Citation footprint:** no peer-reviewed article, IZA/SSRN/NBER paper or Commons
Library briefing cites PolicyEngine UK. The gov.uk Algorithmic Transparency record
("HMT: Policy Engine UK") explicitly says HMT does not currently use it. A JOSS
submission (23 May 2026) is under review. One arXiv citation — Youngman et al.,
*Agent-based macroeconomics for the UK's Seventh Carbon Budget* (arXiv 2602.15607,
INET Oxford + DESNZ, Mar 2026) — cites PolicyEngine only as a **methodological
precedent for WAS→FRS wealth imputation, not as an integrated component**. CB7 made
"macro model + distributional annex" the expected format for UK official analysis, yet
nobody has built the macro→PolicyEngine link. That is the opening.

## The five-step pipeline

**Step 1 — shock the prices you can shock.** Map macro scenarios (NIESR baseline
$103/bbl vs adverse $140/bbl) into an Ofgem cap path and a pump-price path. The cap
lags forward wholesale **6–9 months** mechanically, wholesale ≈ 40–50% of a dual-fuel
bill, so a +X% wholesale move enters as ~0.45X two-to-three quarters later, quantised
into quarterly steps. **Shock gas and electricity asymmetrically** — they are separate
NEED-calibrated variables and electricity is gas-priced ~85% of the time. This is the
modelling choice that makes it a UK paper.

**Step 2 — first-order incidence.** Δcost = quantity × Δprice, no substitution. State
as the Deaton first-order approximation and an explicit upper bound. Report as % of
equivalised disposable income by decile.

**Step 3 — cross the two incidence facts.** Fetzer–Gazze–Bishop: affluent areas lose
more in **£**. Budget-share literature: poor households lose more in **%**. Show both
simultaneously and — uniquely — **by parliamentary constituency** using the 650-seat
weight sets. Two hex maps side by side is the figure that carries the paper.

**Step 4 — score the four competing policies as reforms.**

- *Social tariff*: means-tested discount on `domestic_energy_consumption` — reuse the
  EPG subsidy code path with an eligibility condition.
- *JRF universal discounted block*: discounted rate on the first 50% of typical
  consumption + per-child allowance — a two-tier version of
  `monthly_epg_consumption_level`, which already exists.
- *WHD expansion*: parameter change.
- *VAT cut on domestic fuel (5%→0%)*: `reduced_rate.yaml` — the `vat_change`
  delta-only treatment works in your favour here since you only need the delta.
- *IPPR/EPL-funded rebate*: flat per-household credit; financing scored separately.

Report: cost per £ of bottom-decile gain, share of spend reaching deciles 1–3, and —
the point everyone misses — the **share of losers left uncompensated within each
decile** (Cronin/Fullerton/Sexton; Sallee; Douenne). PolicyEngine's
This is computed directly in `uk_iran_conflict/policies.py`; PolicyEngine has no
suitable built-in for it.

**Step 5 — caveat honestly, and turn caveats into contributions.** No GE, so per Känzig
at most a third of total incidence (use-side only, per Goulder et al.); no consumption
elasticity (#1114); baseline VAT not in net income; 0.38 coverage factor unvalidated.
Then name three gaps the paper could fill next: (a) UK cap-era wholesale→retail
pass-through, (b) energy arrears/debt as an outcome (JRF: 1.5m families in arrears,
3.8m owing £7.5bn), (c) off-gas-grid/rural horizontal losers — Douenne's French result
with no UK analogue.

## Data sources

| Need | Source |
|---|---|
| Retail energy | DESNZ energy price statistics; Ofgem cap history |
| Wholesale | ICE Brent; NBP/TTF futures curves |
| Electricity marginal pricing | National Grid / Elexon (half-hourly) |
| Prices / deflators | ONS CPI/CPIH item-level; ONS Household Costs Indices |
| Household microdata | FRS; LCFS; NEED; EPC |
| Macro scenarios | NIESR Spring/Summer 2026 Outlook; OBR EFO Mar 2026; IMF WEO Apr 2026 |
| Geopolitical risk | policyuncertainty.com/gpr.html (free, daily, threats/acts subindices) |
| Chokepoint transits | IMF/UNCTAD PortWatch (portwatch.imf.org, free, daily) |
| Freight | Drewry WCI; Freightos FBX; Baltic Dry |

## Repository conventions

Structured after `PolicyEngine/uk-ai-study`:

- `uk_iran_conflict/` — importable package: scenarios, shocks, runner
- `analysis/` — scripts that produce `results/`, orchestrated by `analysis/run_incidence.py`
- `results/` — canonical JSON/CSV artifacts, committed
- `paper/` — LaTeX; `main.tex` + `sections/*.tex`; **every headline number enters prose
  as a macro from `paper/values_generated.tex`**, emitted mechanically by
  `analysis/emit_tex_values.py` from `results/`. Missing keys emit `\GENMISSING` so a
  stale tree fails visibly at build time rather than silently keeping old numbers.
- `tests/` — pytest
- Python ≥3.13, `uv`, hatchling, pinned `policyengine-uk`

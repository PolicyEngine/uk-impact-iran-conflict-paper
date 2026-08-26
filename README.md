# uk-impact-iran-conflict-paper

**From Hormuz to the Household: macro-to-micro incidence of the 2026 energy shock in the UK.** The 2026 US/Israel–Iran war pushed UK gas up 78p/therm and Brent up $42/bbl (+57%), and earned the UK the largest G7 growth markdown (−0.5pp) from both the IMF and the OECD. Every published UK assessment of what that does to households is either a macro forecast with no distribution, or a distributional estimate with no macro scenario behind it. This study builds the missing link: a macro scenario is mapped into an Ofgem price-cap path and a pump-price path, and that price vector is passed through **PolicyEngine UK**'s FRS-based microdata to produce household-level incidence — including, uniquely, incidence for all 650 Westminster constituencies on two metrics at once.

The study asks: when a geopolitical shock raises wholesale gas and oil, who in
the UK actually pays for it — in pounds, as a share of income, and in which
seats — and which of the five policy responses live before the Autumn Budget
2026 leaves the fewest losers uncompensated?

## Contribution

Nobody has built a macro→microsimulation link for the UK. The Seventh Carbon
Budget made "macro model + distributional annex" the expected format for UK
official analysis, but the annex is normally produced by a separate method on
separate data, so the two halves are not mutually consistent. Here the
household results are a deterministic function of the macro scenario, through
one explicit and auditable transmission mechanism — the Ofgem cap formula.

Two outputs have no existing UK counterpart:

1. **Paired constituency maps** — £ lost and % of income lost, same shock, same
   households, side by side across 650 seats. The two geographies are close to
   disjoint, which is the mechanism behind the compensation problem.
2. **Within-decile uncompensated-loser counts** for each competing policy. The
   conventional targeting metrics rank the policies one way; this ranks them
   differently, and the difference is the substance of the Budget argument.

## Method — exactly how we do it

The pipeline is `macro scenario → Ofgem cap path + pump-price path →
first-order household incidence → constituency crossing → policy scorecard`,
in five steps:

| Step | What happens | Where |
|---|---|---|
| 1. Price path | NIESR baseline ($103/bbl), adverse ($140/bbl) and realised-2026 oil/gas paths → quarterly Ofgem cap steps and pump prices | `uk_iran_conflict/scenarios.py`, `shocks.py` |
| 2. Incidence | Δcost = quantity × Δprice on NEED-calibrated household kWh; Deaton first-order, explicit upper bound | `uk_iran_conflict/incidence.py` |
| 3. Crossing | £ loss and %-of-income loss aggregated to 650 seats via the constituency weight sets | `analysis/constituency.py` |
| 4. Scorecard | Five reforms scored on cost per £ of D1 gain, share of spend to D1–D3, and share of losers uncompensated within each decile | `analysis/policies.py` |
| 5. Caveats | GE, elasticity, VAT-level and data-vintage gaps stated; three named extensions | `paper/sections/discussion.tex` |

### 1. From wholesale to the cap (`uk_iran_conflict/shocks.py`)

Household energy prices in GB are regulated, not market, prices, so the
transmission device is the cap formula rather than a spot price. The cap's
wholesale allowance is built from forward contracts over an averaging window
that closes before each quarterly announcement, placing a **6–9 month
mechanical lag** between a wholesale move and the bill. Wholesale is only
**~40–50% of a dual-fuel bill** (the rest is network, policy, operating, debt
and margin allowances, none of which respond at this horizon). So a +X%
wholesale move enters as

    Δln p_cap(f,t) = φ_f · Δln p_wholesale(f, t−ℓ),   φ ≈ 0.45, ℓ = 2–3 quarters

quantised into quarterly steps and annualised by consumption-weighted
averaging over the modelled year — not the peak, which is why peak-based bill
figures in the policy literature exceed annualised ones.

**Gas and electricity are shocked asymmetrically.** This is the modelling
choice that makes it a UK paper. Gas is 62% of UK final household energy
consumption (highest in the G7), and gas sets the GB electricity price ~85% of
the time, so a gas shock propagates into the electricity cap — but attenuated,
because electricity bills carry a much larger non-commodity share. We apply
`φ_elec = ψ · φ_gas` with ψ calibrated from the marginal-pricing frequency and
the wholesale share of the electricity cap. PolicyEngine UK carries
`gas_consumption` and `electricity_consumption` as separate NEED-calibrated
kWh variables, which is what makes this implementable. Symmetric shocking
would misallocate the burden away from the gas-heated majority and erase
exactly the horizontal variation the paper exists to show.

Road fuel runs in parallel: crude → product cost + margin + specific fuel duty,
all with 20% VAT on top, so pass-through is less than one-for-one from the
duty and amplified by the VAT layer. The realised 2026 outturns (petrol +20%,
diesel +36%) discipline the calibration. Litres are recovered in PolicyEngine
as spending ÷ modelled pump price, so a price path implies both a duty base
and an expenditure change.

### 2. First-order incidence (`uk_iran_conflict/incidence.py`)

`Δcost_h = Σ_f q_hf · Δp_f` — pre-shock quantities, no substitution. This is
the Deaton & Muellbauer (1980) first-order approximation to the compensating
variation and is stated throughout as an **upper bound**: substitution enters
only at second order. We do not estimate a demand system. A QUAIDS
specification (Banks, Blundell & Lewbel 1997) is not identified here — the
energy quantities are imputed, there is no household-level price variation
under a national cap, and cross-sectional expenditure variation is not price
variation. Elasticity is a **robustness check** (appendix), not the main spec.

### 3. Microdata

PolicyEngine UK, base **FRS 2023–24**, with quantile-regression-forest fusion
from WAS (wealth), LCFS (12 COICOP categories, petrol/diesel/gas/electricity
separately), ETB (VAT expenditure rate), SPI (top incomes) and
Advani–Summers (capital gains). QRF rather than conditional-mean imputation,
so imputed *distributions* survive — which matters, because dispersion in
energy quantities within deciles is the object of interest. LCFS energy
quantities are calibrated to **NEED 2023 kWh**, so households carry physical
quantities that can be repriced without double-counting. Weights are
re-optimised by gradient descent against **1,512 administrative targets**,
which is what permits the constituency aggregation. Uprated on **OBR EFO
March 2026**; baseline prices are the Ofgem Q2 2026 cap.

### 4. Policies scored

| Reform | Implementation |
|---|---|
| Means-tested social tariff | EPG subsidy code path (`gov/treasury/price_cap_subsidy/`) + eligibility condition |
| JRF universal discounted block (~£5bn) | Two-tier `monthly_epg_consumption_level`: discounted rate on first 50% of typical consumption + per-child allowance |
| Warm Home Discount expansion | Parameter change (post-Aug 2026: £150, automatic in E&W, property-cost test scrapped, +2.7m households) |
| VAT on domestic fuel 5% → 0% | `reduced_rate.yaml`; delta-only VAT treatment is harmless here |
| IPPR flat rebate (~£183) | Per-household credit; financing (EPL successor, EGL 45%→55%) scored separately |

Metrics: **cost per £ of bottom-decile gain**, **share of spend reaching
deciles 1–3**, and — the one everyone misses — **share of losers left
uncompensated within each decile**, computed directly in
`uk_iran_conflict/policies.py` alongside continuous measures (share of aggregate
loss offset, mean and median residual loss).
The anchor fact for the third: **~40% of households struggling to heat their
home are not on means-tested benefits.**

## Repo layout

```
uk_iran_conflict/   importable package: scenarios, shocks, runner
analysis/           scripts producing results/
  run_incidence.py  baseline, shock and policy scoring -> results/*.json
  run_variants.py   the specification variants -> results/robustness/
  run_sensitivity.py the three sweeps -> results/sensitivity/
  run_grid.py       the gas-oil scenario grid -> results/grid/
  emit_tex_values.py  results/ -> paper/values_generated.tex
results/            canonical JSON/CSV artifacts, committed (aggregates only)
paper/              main.tex + sections/*.tex + references.bib
tests/              pytest; runs without microdata or an HF token
docs/               RESEARCH_BRIEF.md
```

Every headline number enters the prose as a macro from
`paper/values_generated.tex`, emitted mechanically by
`analysis/emit_tex_values.py` from `results/`. A macro with no corresponding
key renders as `\GENMISSING`, so a stale results tree **fails visibly in the
PDF** rather than silently keeping superseded numbers.

## Reproduce

```bash
uv sync
export HUGGING_FACE_TOKEN=hf_...     # needs policyengine/policyengine-uk-data access
python analysis/run_incidence.py     # baseline, shock, policies -> results/
python analysis/run_variants.py      # specification variants
python analysis/emit_tex_values.py   # results/ -> paper/values_generated.tex
python -m pytest tests/              # no microdata needed
cd paper && latexmk -pdf main.tex
```

### Data access

PolicyEngine UK's enhanced microdata sit in a **private Hugging Face
repository** (`policyengine/policyengine-uk-data`) under a licence that does
not permit redistribution, so a token with access is required to regenerate
household results. Microdata are never committed. The aggregates in
`results/` are committed, so every number in the paper is traceable from the
JSON/CSV artifact to the macro that renders it without re-running the
microsimulation. `tests/` is designed to run without the token, and CI runs
without it.

## Known limitations

Stated in full in `paper/sections/methodology.tex`; the short list:

- **No general equilibrium.** Static, use-side only. Känzig (2023): poorer
  households lose mainly through income/employment, and GE/indirect effects
  are ~2/3 of the response. Goulder et al. (2019): use-side regressive,
  source-side progressive, and a consumption microsimulation sees only the
  use side. Read every headline number as a **lower bound on total household
  incidence** and an upper bound on the use-side channel.
- **No consumption elasticity or pass-through module** in PolicyEngine UK
  (open issue #1114; UKMOD's TCO uses 0.8). Main spec is zero-elasticity by
  construction; elasticity is the appendix robustness check. No price block
  exists at all — CPI/CPIH/RPI are uprating indices only.
- **Baseline VAT is not in net income.** `household_tax` includes
  `vat_change`, not `vat`, so only VAT *reforms* move the distribution.
  Fuel duty and carbon tax do enter at level — the treatment across indirect
  taxes is inconsistent.
- **VAT rests on a 0.38 coverage grossing factor** (following IFS TAXBEN) plus
  a 4-predictor ETB imputation; unvalidated, open issue #352.
- **Missing duties**: alcohol, tobacco, APD, IPT, VED all absent.
- **Static, no lifecycle** — no borrowing, no savings drawdown, and no
  arrears, which is the most visible real-world form of the 2026 shock.
- **WAS round 7 (2018–20)** wealth vintage, predating the entire rate cycle —
  a first-order problem for anything in the Pallotti et al. (2024)
  balance-sheet direction. Take-up is a scalar.
- **Constituency weights are synthetic** re-optimisations of ~20k households
  against area targets, not independent local samples. Seat *rankings* are
  more reliable than seat *levels*.
- **No well-cited UK estimate of cap-era wholesale→retail pass-through
  exists**, so φ and ℓ are calibrated from the published cap structure rather
  than estimated. Sensitivity is reported in the appendix; closing this gap is
  extension (a).

## Key references

- Levell, O'Connell & Smith (2025), IFS WP W25/55 / CEPR DP19939 (cond. acc. *JPE*) — the optimum includes a strictly positive price subsidy.
- Känzig (2023), NBER WP 31221 — the GE caveat.
- Fetzer, Gazze & Bishop (2024), *Economic Policy* 39(120) — affluent areas lose more in £.
- Cronin, Fullerton & Sexton (2019), *JAERE*; Sallee (2019), NBER WP 25831; Douenne (2020), *Energy Journal* — horizontal losers.
- Goulder, Hafstead, Kim & Long (2019), *JPubE*; Ohlendorf et al. (2021), *ERE*.
- Deaton & Muellbauer (1980); Banks, Blundell & Lewbel (1997).
- Caldara & Iacoviello (2022), *AER*; Caldara, Conlisk, Iacoviello & Penn (2025).
- NIESR Spring/Summer 2026 Outlook; JRF (Apr 2026); Resolution Foundation (Apr 2026); IPPR (Apr 2026); Cornwall Insight (Aug 2026).

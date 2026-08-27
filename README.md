# uk-impact-iran-conflict-paper

**From Hormuz to the Household: macro-to-micro incidence of the 2026 energy shock in the UK.**

Replication package for the paper. The 2026 US/Israel–Iran war pushed UK gas up
78p/therm and Brent up $42/bbl (+57%), and earned the UK the largest G7 growth
markdown (−0.5pp) from both the IMF and the OECD. Published UK assessments are
either a macro forecast with no distribution, or a distributional estimate with
no macro scenario behind it. This study builds the link: a macro scenario is
mapped into an Ofgem price-cap path and a pump-price path, and that price vector
is passed through **PolicyEngine UK**'s FRS-based microdata to produce
household-level first-order incidence, which is then used to score five policy
responses live before the Autumn Budget 2026.

The question: when a geopolitical shock raises wholesale gas and oil, who in the
UK actually pays for it — in pounds and as a share of income — and which policy
response leaves the fewest losers uncompensated?

## What this repository contains — and what it does not

The household results are a deterministic function of the macro scenario through
one explicit transmission mechanism (the Ofgem cap formula), so the macro and
distributional halves are mutually consistent. That is the contribution.

**There is no constituency analysis.** An earlier scoping brief promised
incidence for all 650 Westminster seats and paired constituency maps; the paper
**withdraws** that claim and this repository contains no code for it. The reason
is in the data: this release of the PolicyEngine UK microdata carries **no
constituency weight matrix**, and its `local_authority` column is degenerate —
every household reads `MAIDSTONE`. The finest real geography available is
**region (12)**, with country (4) above it, and that is what
`uk_iran_conflict.incidence.geography_table` and `results/figures/fig4_region.png`
report. `docs/RESEARCH_BRIEF.md` is the superseded brief, kept for provenance;
its constituency claims do not describe this package.

What the package does produce, with no direct counterpart in the UK literature:

1. **Within-decile uncompensated-loser counts** for each competing policy,
   alongside continuous measures (share of aggregate loss offset, mean and
   median residual loss). Conventional targeting metrics rank the five policies
   one way; this ranks them differently.
2. **A channel decomposition** — gas, electricity and motor fuel — reported
   across seven specifications plus two combination specifications, rather than
   as a single point estimate. The motor-fuel share is calibration-dependent and
   is reported as a range for that reason.
3. **A negative structural finding about PolicyEngine UK itself**: the model has
   no price channel, `gov.contrib.policyengine.economy.energy_bills` is a dead
   parameter, and there is no Warm Home Discount. The shock is therefore applied
   on the consumption side rather than through parameter reforms.

## Method

The pipeline is `macro scenario → Ofgem cap path + pump-price path → first-order
household incidence → policy scorecard`.

| Step | What happens | Where |
|---|---|---|
| 1. Price path | NIESR baseline ($103/bbl), adverse ($140/bbl) and realised-2026 oil/gas paths → quarterly Ofgem cap steps and pump prices | `uk_iran_conflict/scenarios.py` |
| 2. Price factors | Scenario → retail multipliers, pump multipliers and cap levels (pure functions, no microdata) | `uk_iran_conflict/reforms.py` |
| 3. Incidence | Δcost = quantity × Δprice on NEED-calibrated household kWh; Deaton first-order, explicit upper bound; decile, intra-decile and region tables | `uk_iran_conflict/incidence.py` |
| 4. Scorecard | Five reforms scored on cost per £ of decile-1 gain, share of spend to D1–D3, and share of losers uncompensated within each decile | `uk_iran_conflict/policies.py` |
| 5. Robustness | Demand-response sweep as an appendix check only; the main spec is zero-elasticity by construction | `uk_iran_conflict/elasticity.py` |
| 6. Caveats | GE, elasticity, VAT-level and data-vintage gaps stated; three named extensions | `paper/sections/discussion.tex` |

### From wholesale to the cap (`uk_iran_conflict/scenarios.py`)

Household energy prices in GB are regulated, not market, prices, so the
transmission device is the cap formula rather than a spot price. The cap's
wholesale allowance is built from forward contracts over an averaging window
that closes before each quarterly announcement, placing a **6–9 month mechanical
lag** between a wholesale move and the bill. Wholesale is only **~40–50% of a
dual-fuel bill**. So a +X% wholesale move enters as

    Δln p_cap(f,t) = φ_f · Δln p_wholesale(f, t−ℓ),   φ ≈ 0.45, ℓ = 2–3 quarters

quantised into quarterly steps and annualised by consumption-weighted averaging
over the modelled year — not the peak. This is the phase-in recorded as decision
D2 in `docs/FIXES.md`; the steady-state figure is retained as a labelled
alternative, not as the headline.

**Gas and electricity are shocked asymmetrically.** Gas is 62% of UK final
household energy consumption (highest in the G7), and gas sets the GB
electricity price ~85% of the time, so a gas shock propagates into the
electricity cap — attenuated, because electricity bills carry a larger
non-commodity share. `φ_elec = ψ · φ_gas`, with ψ calibrated from the
marginal-pricing frequency and the wholesale share of the electricity cap.
PolicyEngine UK carries `gas_consumption` and `electricity_consumption` as
separate NEED-calibrated kWh variables, which is what makes this implementable.

Road fuel runs in parallel: crude → product cost + margin + specific fuel duty,
with 20% VAT on top. A **symmetric-damping specification** is run alongside the
asymmetric one, because the motor-fuel share of the loss depends on the *ratio*
of the gas and pump damping fractions rather than on either alone.

### First-order incidence (`uk_iran_conflict/incidence.py`)

`Δcost_h = Σ_f q_hf · Δp_f` — pre-shock quantities, no substitution. This is the
Deaton & Muellbauer (1980) first-order approximation to the compensating
variation and is stated throughout as an **upper bound**. No demand system is
estimated: a QUAIDS specification is not identified here, because the energy
quantities are imputed and there is no household-level price variation under a
national cap. Elasticity is an appendix robustness check.

Percentage-of-income statistics use **equivalised AHC income** throughout
(decision D1 in `docs/FIXES.md`), with the unequivalised gradient reported once
as a robustness line.

### Microdata

PolicyEngine UK, base **FRS 2023–24**, with quantile-regression-forest fusion
from WAS (wealth), LCFS (12 COICOP categories, petrol/diesel/gas/electricity
separately), ETB (VAT expenditure rate), SPI (top incomes) and Advani–Summers
(capital gains). LCFS energy quantities are calibrated to **NEED 2023 kWh**.
Weights are re-optimised by gradient descent against 1,512 administrative
targets. Uprated on **OBR EFO March 2026**; baseline prices are the Ofgem Q2 2026
cap. Known imputation defects — motor fuel over-imputed at the bottom, domestic
energy under-imputed overall — are quantified in `docs/VALIDATION.md` and carried
as explicit robustness specifications, not buried.

### Policies scored

| Reform | Implementation |
|---|---|
| Means-tested social tariff | Bill discount on the shocked domestic bill + eligibility condition |
| JRF universal discounted block (~£5bn) | Two-tier: discounted rate on the first 50% of typical consumption + per-child allowance |
| Warm Home Discount expansion | £150, automatic in E&W, property-cost test scrapped (+2.7m households) — modelled directly, as PolicyEngine UK has no WHD |
| VAT on domestic fuel 5% → 0% | Delta-only VAT treatment |
| IPPR flat rebate (~£183) | Per-household credit; financing scored separately |

All five are also scored at a **common exchequer envelope** as well as at each
sponsor's own stated cost, so generosity and design are separable.

## Repository layout

```
uk_iran_conflict/          importable package
  scenarios.py             macro scenarios + price-path arithmetic (no microdata)
  reforms.py               retail/pump factors and cap levels (pure functions)
  incidence.py             first-order incidence; decile, intra-decile, region tables
  policies.py              the five policy instruments and the scorecard metrics
  elasticity.py            appendix demand-response module (not the main spec)

analysis/                  scripts producing results/ and paper inputs
  run_incidence.py         baseline, shock and policy scoring        -> results/<scenario>/
  run_variants.py          the seven headline specifications          -> results/robustness/
  run_combinations.py      the two combination specifications         -> results/robustness/
  run_sensitivity.py       the sweeps                                 -> results/sensitivity/
  run_grid.py              the gas-oil scenario grid                  -> results/grid/
  emit_tex_values.py       results/ -> paper/values_generated.tex
  emit_tables.py           results/ -> paper/tables/*.tex
  figures.py               results/ (+ microdata for 3 of them) -> results/figures/
  figures_sensitivity.py   results/sensitivity/ -> results/figures/  (no microdata)
  figures_grid.py          results/grid/ -> results/figures/         (no microdata)
  figstyle.py              shared matplotlib style

results/                   canonical JSON/CSV artifacts, committed (aggregates only)
paper/                     main.tex + sections/ + tables/ + references.bib
tests/                     pytest; runs with no microdata and no token
docs/                      briefs, validation, referee reports, fix list
```

Every headline number enters the prose as a macro from
`paper/values_generated.tex`, emitted mechanically by `analysis/emit_tex_values.py`
from `results/`. A macro with no corresponding key renders as `\GENMISSING`, so a
stale results tree **fails visibly in the PDF** rather than silently keeping
superseded numbers. Table bodies are emitted the same way into `paper/tables/`.

## Reproducing this paper

There is an access boundary, and it matters for what you can check.

### Without credentials — the results → tables → PDF chain

Everything from the committed `results/` tree onward reproduces with no token
and no microdata, and is **byte-reproducible**: re-running the emitters over the
committed results regenerates `paper/values_generated.tex` and `paper/tables/*.tex`
identically, and the PDF rebuilds from them.

```bash
uv sync
uv run python -m pytest tests/            # full suite, no token needed
uv run python analysis/emit_tex_values.py # results/ -> paper/values_generated.tex
uv run python analysis/emit_tables.py     # results/ -> paper/tables/*.tex
uv run python analysis/figures_sensitivity.py
uv run python analysis/figures_grid.py
cd paper && latexmk -pdf main.tex
```

This lets a reader verify that every number in the manuscript traces to a
committed JSON/CSV artifact, and that no number was hand-transcribed. It does
**not** verify the microsimulation that produced those artifacts.

### With credentials — regenerating the results tree

```bash
cp .env.example .env                      # then paste your own token into .env
uv run python analysis/run_incidence.py   # baseline, shock, policies -> results/
uv run python analysis/run_variants.py    # the seven specifications
uv run python analysis/run_combinations.py
uv run python analysis/run_sensitivity.py
uv run python analysis/run_grid.py
uv run python analysis/figures.py         # 3 of 14 figures need the microdata
```

These scripts download a **pinned revision** of the dataset, so the run is
reproducible for anyone holding access.

### Data access

The PolicyEngine UK enhanced microdata sit in a **private Hugging Face dataset**
(`policyengine/populace-uk-private`, file `populace_uk_2023.h5`, at the revision
pinned in `analysis/run_incidence.py`) under a licence that does not permit
redistribution. Access is granted by PolicyEngine; a read-scoped user access
token for an account with that grant is required. Put it in `.env` as
`HUGGING_FACE_TOKEN` — see `.env.example`. **`.env` is gitignored and no token is
distributed with this repository.** Microdata are never committed. The
aggregates in `results/` are committed, which is what makes the credential-free
chain above possible.

## What CI covers — and what it does not

CI (`.github/workflows/ci.yaml`) runs ruff lint, ruff format check, and the full
pytest suite on every push and pull request. **CI has no Hugging Face token and
no microdata, by design** — the dataset is not redistributable, so it cannot be
placed in a public runner.

Concretely, a green badge means:

- **Covered.** Scenario construction and price-path arithmetic; retail, pump and
  cap-level factors; the elasticity module's welfare bounds; the grid and sweep
  logic; the policy instruments and every scorecard metric; and the incidence
  arithmetic — all exercised on **synthetic fixtures** constructed in the tests.
- **Not covered.** Any execution against the real PolicyEngine UK microdata. No
  test loads the dataset, so nothing in CI validates the imputation quality, the
  weighting, the household-level cuts, or the committed contents of `results/`.
  The suite completes in well under a second, which is the tell.

Read the badge as "the arithmetic is consistent and the code runs", not as "the
pipeline has been validated against data". The data-side validation is a
separate, manual exercise and is written up in `docs/VALIDATION.md`.

## Documentation

| File | What it is |
|---|---|
| `docs/FIXES.md` | The authoritative fix list and the two decisions (D1, D2) the results hang on. **Referenced by code docstrings** — labels are stable. |
| `docs/VALIDATION.md` | Adversarial check of the results against ONS, DfT, DWP and the literature. **Referenced by code docstrings** — check numbers are stable. |
| `docs/REFEREE_REPORTS.md`, `docs/REFEREE_ROUND2.md` | The referee reports as received. Historical record, not edited. |
| `docs/FINDINGS.md` | The first full run. **Superseded** — kept as an audit trail; see the banner at its top. |
| `docs/RESEARCH_BRIEF.md` | The original scoping brief. **Superseded in part** — see the banner at its top. |
| `docs/PRIOR_WORK_ENERGY.md`, `docs/PRIOR_WORK_IRAN.md` | Literature scoping notes. |

## Known limitations

Stated in full in `paper/sections/methodology.tex`; the short list:

- **No general equilibrium.** Static, use-side only. Känzig (2023): poorer
  households lose mainly through income/employment, and GE/indirect effects are
  ~2/3 of the response. Goulder et al. (2019): use-side regressive, source-side
  progressive. Read every headline number as a **lower bound on total household
  incidence** and an upper bound on the use-side channel.
- **No consumption elasticity or pass-through module** in PolicyEngine UK
  (open issue #1114). Main spec is zero-elasticity by construction. No price
  block exists at all — CPI/CPIH/RPI are uprating indices only.
- **Baseline VAT is not in net income.** `household_tax` includes `vat_change`,
  not `vat`, so only VAT *reforms* move the distribution, while fuel duty and
  carbon tax enter at level — the treatment across indirect taxes is
  inconsistent.
- **VAT rests on a 0.38 coverage grossing factor** (following IFS TAXBEN) plus a
  4-predictor ETB imputation; unvalidated, open issue #352.
- **Missing duties**: alcohol, tobacco, APD, IPT, VED all absent.
- **Static, no lifecycle** — no borrowing, no savings drawdown, and no arrears,
  which is the most visible real-world form of the 2026 shock.
- **WAS round 7 (2018–20)** wealth vintage, predating the entire rate cycle.
  Take-up is a scalar.
- **Motor fuel is over-imputed at the bottom of the distribution** and domestic
  energy under-imputed overall, relative to ONS Family Spending. Both are
  quantified in `docs/VALIDATION.md` Checks 2 and 3 and carried as explicit
  robustness specifications.
- **No geography below region.** No constituency weight matrix exists in this
  data release and `local_authority` is degenerate. Region (12) is the floor.
- **No well-cited UK estimate of cap-era wholesale→retail pass-through exists**,
  so φ and ℓ are calibrated from the published cap structure rather than
  estimated. Sensitivity is reported in the appendix.

## Key references

- Levell, O'Connell & Smith (2025), IFS WP W25/55 / CEPR DP19939 (cond. acc. *JPE*) — the optimum includes a strictly positive price subsidy.
- Känzig (2023), NBER WP 31221 — the GE caveat.
- Fetzer, Gazze & Bishop (2024), *Economic Policy* 39(120) — affluent areas lose more in £.
- Cronin, Fullerton & Sexton (2019), *JAERE*; Sallee (2019), NBER WP 25831; Douenne (2020), *Energy Journal* — horizontal losers.
- Goulder, Hafstead, Kim & Long (2019), *JPubE*; Ohlendorf et al. (2021), *ERE*.
- Deaton & Muellbauer (1980); Banks, Blundell & Lewbel (1997).
- Caldara & Iacoviello (2022), *AER*; Caldara, Conlisk, Iacoviello & Penn (2025).
- NIESR Spring/Summer 2026 Outlook; JRF (Apr 2026); Resolution Foundation (Apr 2026); IPPR (Apr 2026); Cornwall Insight (Aug 2026); Ofgem cap announcement, 26 Aug 2026.

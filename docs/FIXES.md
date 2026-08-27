# Referee-response fix list

> **STABLE REFERENCE — DO NOT RENUMBER.** Docstrings throughout
> `uk_iran_conflict/` and `analysis/` cite this file by label (`docs/FIXES.md
> D1`, `docs/FIXES.md A5`, `docs/FIXES.md C14`, ...) to explain why a piece of
> code does what it does. Those labels are part of the replication package's
> interface. Rules:
>
> - **Never renumber an item.** Numbering is continuous 1–35 across sections;
>   the letter is the section it sits in (A = items 1–6, B = 7–9, C = 10–15,
>   D = 16–29, E = 30–35), so `C14` means "item 14, in section C".
> - **The two decisions are `D1` and `D2` and are *not* section-D items.**
>   Section D begins at item 16, so there is no collision, but a citation of
>   `FIXES.md D1` or `FIXES.md D2` always means a **decision**, while `D16`
>   and above mean a section-D item. Preserved as-is for the code that already
>   cites them.
> - **Never delete an item.** Mark it withdrawn or superseded in place.
> - **Never change what an item asserts.** Append a dated note instead; the
>   code that cites it was written against the original wording.
> - New items append at the end of their section with the next free number.
>
> This file is a normative reference for the code, so it ships **inside** the
> replication package. It is not a to-do list to be tidied.

Three independent referees (public economics, energy/macro, data/replication) all
returned **major revision** and all answered "no" to submission. Their reports are
in `docs/REFEREE_REPORTS.md`. This file is the authoritative work list and records
the two judgement calls the rest of the work hangs on.

Everything here has been verified against the code or the results tree. Items are
ordered by how much of the paper they touch.

---

## The two decisions

### D1. The denominator: switch to equivalised income, do not relabel

The paper claims equivalised AHC income throughout and asserts comparability with
the published distributional literature. The code divides by unequivalised
`household_net_income`; `equiv_hbai_household_net_income_ahc` is loaded and never
referenced. Relabelling would keep the numbers but forfeit the comparability the
paper trades on, and Referee 2 independently found the implied incomes to be about
double published HBAI means — which is itself evidence the wrong concept is in use.

**Decision: use equivalised AHC income as the denominator, as the paper claims.**
Re-run everything downstream. Report the unequivalised gradient once, as a
robustness line, so the change is visible rather than silent.

### D2. The headline aggregate: apply the phase-in the paper says it applies

`shock_cost` uses steady-state retail factors; the quarterly phase-in profile is
never applied, so the headline charges full pass-through for twelve months. Step 1
of the paper says the opposite: "the annual price faced by a household is the
consumption-weighted average of the quarterly cap levels prevailing over the
modelled year, not the peak."

**Decision: implement Step 1 as written.** The annual domestic factor becomes the
consumption-weighted average of the quarterly cap levels over the modelled year.
Motor fuel keeps its own (already damped) annual factor, since pump prices do not
lag. Report the steady-state figure as a clearly-labelled alternative, not as the
headline. Expect the aggregate to fall materially; that is the correct direction.

---

## A. Core computation

1. **Denominator (D1).** Use `equiv_income_ahc` in every percentage-of-income
   statistic: `decile_table`, `intra_decile_table`, `geography_table`,
   `run_scenario`. Keep the aggregate-ratio convention. Document the equivalisation
   scale explicitly. Add a robustness row on the unequivalised basis.

2. **Phase-in (D2).** Apply the consumption-weighted quarterly cap average in the
   main run. Keep a labelled steady-state variant. Reconcile with
   `results/sensitivity/cap_lag.csv` so the appendix and the headline agree — at
   present the appendix reports £205 per household where the abstract reports £343
   for the same specification.

3. **Symmetric-damping variant — the one that may overturn the headline.** The
   motor-fuel share depends only on the *ratio* of the gas and pump damping
   fractions: 57% at the paper's 0.36/0.60, and 44.5% at **any** common fraction.
   The reported 55.8–67.8% range is one-sided — both endpoints raise the share.
   Add a symmetric-damping specification and report it alongside. If the majority
   claim does not survive, the paper's central contribution must be reframed as a
   channel-composition result that is calibration-dependent, not as "motor fuel
   dominates".

4. **Identification of the gas leg.** `sustained_fraction x phase_in[0]` is what
   the Cornwall anchor identifies; only the product is pinned. Sweep the split, and
   sweep the three unswept parameters that scale the domestic channel one-for-one:
   `PREWAR_NBP_PENCE_PER_THERM` (90.0), `WHOLESALE_SHARE_GAS_BILL` (0.45),
   `WHOLESALE_SHARE_ELECTRICITY_BILL` (0.35).

5. **Elasticity: welfare, not spending.** `spend_change` returns the change in
   expenditure, so at eps = -0.8 the reported figure counts foregone heating as
   costless — the "heat or eat" fallacy the same appendix criticises. Report a
   money-metric bound (the Paasche term `q1 . dp` bounds the compensating variation
   below). The uncertainty ranking then almost certainly inverts: demand response
   becomes small, and the fuel imputation, the accounting basis and the pass-through
   calibration become the large ones. Rewrite the appendix's ranking accordingly.

6. **Decile coverage.** `decile_table` iterates deciles 1-10 and silently drops
   out-of-range households (~0.9%, 29.23m against 29.5m), while
   `share_of_total_loss` normalises on the full total. Zero and negative-income
   households are the likeliest to be dropped — which makes the "only 0.57% of
   decile one has zero or negative income" rebuttal partly circular. Report the
   excluded count and fix the normalisation.

## B. Policy modelling

7. **JRF block is mis-specified, not under-calibrated.** `_social_tariff` discounts
   the whole shocked bill; `_jrf_block` discounts only the shock component. JRF
   propose a discounted *rate on the block*, i.e. on the level. That, not
   generosity, is why our cost is £1.9bn against their £5bn. Re-specify as a level
   subsidy, and additionally score all five instruments at a **common exchequer
   envelope** as well as at each sponsor's own stated cost.

8. **Child count.** `np.clip(base.people - 2, 0, None)` is household size minus
   two, not children: a lone parent with one child gets nothing, a three-adult
   household gets an allowance. Use PolicyEngine's real child count.

9. **Continuous compensation measures.** "Share of losers uncompensated" is
   knife-edge (any shortfall counts) applied to a loss that is itself an upper
   bound. VAT zero-rating is called "compensates nobody" while delivering the
   second-highest mean gain and a lower mean residual loss than the social tariff.
   `net_loss_after_policy_gbp` is already computed and appears in no table. Add
   share of aggregate loss offset, and mean/median residual loss overall and by
   decile. Temper the rhetoric to match.

## C. Data quality and honesty

10. **ONS misattribution — factual error, three places.** The paper says "ONS
    Family Spending gives £521 against £2,230". The real ONS values, in our own
    code, are **£318 and £1,362**; £521/£2,230 are our model's output *after*
    rescaling (the routine preserves the microdata's national total and transplants
    only the ONS shape). Quote the real ONS figures and describe the rescaling
    honestly.

11. **Domestic energy is under-imputed by ~25% and this is not disclosed.** Mean
    modelled domestic spend is £1,330 against ONS £1,780 — and below Ofgem's £1,663
    typical-consumption cap, which is not credible for a mean. It runs opposite to
    the disclosed motor-fuel over-imputation and **compounds it** on the central
    claim. Disclose both directions, and add a specification that corrects both
    levels against ONS.

12. **Persist the hardcoded prose numbers.** `_PENDING_PERSIST` emits seven
    literals with no results backing, including the entire evidentiary basis of the
    denominator rebuttal, while the appendix guarantees every number is emitted
    mechanically and fails visibly if absent. Persist them or drop the guarantee.

13. **Stop swallowing exceptions under a headline finding.**
    `means_tested_flag` wraps each benefit lookup in `except Exception: continue`,
    so a renamed variable silently shrinks the means-tested population — precisely
    the symptom the paper reports as a finding (15.7% against a 7.2m UC caseload).
    Assert explicitly and log the resolved set into `results/`.

14. **Delete the contradictory second pipeline.** `uk_iran_conflict/runner.py` and
    `analysis/run_all.py` implement the shock as a parameter reform and include
    650-seat constituency aggregation described as "the paper's Step 3 result" —
    both of which the paper spends two subsections proving impossible. A replicator
    running them silently gets different numbers. Remove them.

15. **Disclose the two live policy facts.** Ofgem confirmed £1,723 on 26 August
    2026, superseding the Cornwall Insight £1,729 the paper calibrates to; and
    Ofgem's temporary removal of VAT on electricity from 1 October 2026 interacts
    directly with the scored VAT zero-rating instrument and may already sit in the
    baseline. Both are in our own notes and neither is in the paper.

## D. Corrections of fact in the text

16. Asymmetry effect is **7.6%**, not the "five per cent" stated twice in the
    appendix (the macro already says 7.6).
17. Cap-lag range is **£83** per household, not "roughly £40" (the discussion's
    "roughly £80" is right).
18. Cost per pound of bottom-decile gain: the rebate is **10.53**; 12.98 is the VAT
    cut. The sentence carries the paper's key targeting point.
19. Under the ONS calibration decile ten is **not** "unchanged at 0.36 per cent" —
    it rises from 0.26%, and its cash loss rises from £394 to £540.
20. Motor-fuel share is **not** "stable across macro scenarios": 65.9 / 66.1 / 55.8,
    a 10pp spread with the central case as the outlier.
21. The crude-to-pump "duty and VAT identity" described in Step 1 **is not
    implemented** — pump changes are hand-set percentages, and the grid then fits a
    pass-through to those stipulated points. Implement the identity or delete the
    claim.
22. Report bill versus unit rate consistently: phi is a share of the *bill*, but the
    text reports a "unit rate" 14.0% above baseline. With standing charges a +14%
    bill is roughly a +17% unit rate — and the elasticity sweep is fed the bill
    factor while the appendix claims it uses the unit rate.
23. `\genCapAnnualised` (£1,853) is the steady-state cap, not an annualised level.
24. The cap-lag "exact invariance" is an identity by construction
    (`cumulative_weight` has no lag dependence). Present it as such.
25. The grid plots named scenarios at **undamped** coordinates while every cell runs
    undamped; the realised point therefore sits above the stated frontier by the
    grid's own numbers, contradicting the claim that all three sit below it. Plot in
    damped-equivalent coordinates or drop the claim.
26. The decile-8 cash spike is narrated as mileage but is more likely the diesel
    uplift (+21.6% against petrol's +12.0%). Report petrol, diesel and diesel share
    by decile and let the data choose.
27. The regional profile is unchecked: a North East mean loss 47% above Wales and
    2.5x London, in the lowest-income English region, is implausible and probably
    the same fuel imputation defect. Benchmark or report under the ONS calibration.
28. §3.5 says the imputed energy variables are "physical quantities rather than
    deflated spending". The code treats them as £/yr throughout. Correct the claim.
29. Bibliography: R85 author list — one referee gives Advani, Johnson, Leicester and
    Stoye, another Advani, Bassi, Leicester and Stoye. Resolve against the source.

## E. Numerical hygiene (minor, none likely material at n = 535k)

30. `wquantile` interpolates on raw cumulative weight rather than `cum - 0.5w`.
31. `top_share` uses `>=`, over-counting ties.
32. `gini` omits the origin segment of the Lorenz curve.
33. `retail_factors` / `pump_price_factors` fall back to a silently zero shock;
    prefer raising.
34. Two macros carry one number (`genMeansTestedShareHouseholds`,
    `genMeansTestedSharePct`).
35. One grid cell has an empty ratio; the degenerate corner yields 0% and 100%
    fuel shares. Filter or document.

# Referee reports — round 2

Three fresh referees, same lenses as round 1, none primed by the first round or
by the internal notes.

| Referee | Round 1 | Round 2 | Submit today? |
|---|---|---|---|
| 1 — public economics | Major revision | Major revision — "close; one focused revision" | No |
| 2 — energy / macro | Major revision | Major revision — "one focused revision away" | No |
| 3 — data / replication | Major revision | Major revision — "a well-executed revision should close them" | No |

Round 1 failed the paper on three of four headline claims. Round 2 fails it on
labels and conventions rather than on the substance of the pipeline, and all
three say no new data is needed. Referee 3 re-ran both emit scripts and
confirmed `values_generated.tex` and all twelve tables regenerate
**byte-identically**; referee 1 verified twelve numeric claims and found no
fabricated figures; all three record that the paper's self-disclosure runs
against the author's interest and is handled cleanly.

## Found by all three independently

1. **The two legs are annualised over different windows.** The domestic leg is
   averaged over 2026Q4-2027Q3; the pump damping fraction is derived from a
   calendar-2026 monthly profile. They are summed and labelled "2026". This also
   breaks the Resolution Foundation comparison (£11bn is calendar 2026) and
   mechanically inflates the motor-fuel share. Round 1 raised this as the
   headline-versus-appendix gap; changing £343 to £304 narrowed it from 67% to
   48% without resolving it.
2. **The common-envelope ranking is a scaling artefact.** Scaling instrument
   outputs to a £5bn envelope implies a 138% social-tariff discount (x3.95), a
   £1,083 Warm Home Discount (x7.22) and a negative VAT rate (x2.47). The
   saturation result follows mechanically from paying households more than they
   lost, and the paper crowns "VAT zero-rating" the winner while separately and
   correctly stating the 5% rate is an arithmetic ceiling.

## Found by two of three

3. **The within-decile dispersion result is carried entirely by decile one.**
   Excluding it, mean within-decile range (2.08pp) falls below the between-decile
   range (2.27pp); eight of ten deciles are below it and the median is 2.18pp. It
   rests on the decile where 20% have non-positive equivalised income.
4. **The non-monotone cash profile does not survive either ONS calibration** —
   both are strictly monotone, peaking at decile ten. The paper says the pattern
   "is common to every specification".
5. **The reported price path is the steady-state one, the applied one is
   smaller** (14.0%/9.3% quoted, 10.2%/7.0% applied).
6. **`cost_per_pound_decile_one` is dimensionally wrong** — £bn divided by £,
   printed as pounds. Ranking survives; every quoted level is meaningless.
7. **The grid's decile ratios (11.77-12.73) lie outside every specification in
   the paper (4.41-9.25)**, unreconciled.

## Referee 3 only, and worth acting on

8. **No external source has a bibliography entry.** Resolution Foundation, NIESR,
   Ofgem, Cornwall Insight, ONS Family Spending, DfT NTS, JRF and IPPR are cited
   in prose only. `docs/VALIDATION.md` has URLs for all of them. A reader cannot
   locate the £11bn the validation rests on.
9. **`reforms.py` is a dead parallel implementation that contradicts the paper**,
   including instruments built on a Warm Home Discount parameter the paper states
   categorically does not exist. It also contains unreachable date bugs
   (`2026-06-31`, `2026-09-31`).
10. **The dataset download carries no `revision=` pin**, so the model is pinned
    exactly and the data it consumes is not.
11. **The decile ranking variable's income concept is never verified** — probably
    equivalised BHC, while burdens are measured on equivalised AHC.
12. **Three items in our own notes never reached the paper**: the
    domestic-energy-only gradient as a robustness anchor, the Fetzer/Gazze/Bishop
    self-disclaimer that they cannot compute energy as a share of income, and the
    table of published UK gradient comparators (NEF 7.5x, ONS 3.5x, IFS R85 4.8x,
    RF 1.75x).

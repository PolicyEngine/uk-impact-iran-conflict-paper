# Referee reports — round 1

Three independent referees reviewed the manuscript against the *Fiscal Studies*
bar: one applied public economist (estimand, incidence, policy scoring), one
energy/macro specialist (shock calibration, cap mechanics, pass-through), and one
data-and-replication referee (does the code do what the paper says).

Each was given the compiled PDF and the repository as the replication package, and
was explicitly told that `docs/FINDINGS.md`, `docs/VALIDATION.md` and
`docs/RESEARCH_BRIEF.md` are internal working notes, **not** part of the
submission — to be read only at the end, to judge whether anything material was
known and omitted.

## Verdicts

| Referee | Recommendation | Submit today? |
|---|---|---|
| 1 — public economics | Major revision | No |
| 2 — energy / macro | Major revision | No |
| 3 — data / replication | Major revision (reproducible with caveats) | No |

Unanimous, and — importantly — they failed the paper on *different* grounds, so
the count is three independent failures rather than three views of one flaw.

## The four findings that carry the most weight

1. **The headline is a calibration artefact.** Motor fuel's share depends only on
   the ratio of the gas and pump damping fractions: 57% at the paper's 0.36/0.60,
   44.5% at any common fraction. The reported 55.8-67.8% range is one-sided.
2. **The denominator is not the one claimed.** Every percentage divides by
   unequivalised net income while the paper claims equivalised AHC throughout.
3. **The headline aggregate skips the lag the paper says it applies**, and so
   contradicts the paper's own appendix (£343 per household against £205).
4. **A benchmark quoted four times is misattributed to ONS** — the real figures in
   our own code are £318/£1,362, not the £521/£2,230 in the text.

## What all three referees agreed survives

The macro-to-microsimulation pipeline as a contribution; the channel-limitation
argument in qualitative form; horizontal dispersion exceeding vertical; and the
capability-boundary finding about PolicyEngine UK. Referee 3 verified the
`results -> tables -> PDF` chain reproduces byte-identically and called the
transparency "exceptional". Referee 1 confirmed the earlier removal of the
spurious Resolution Foundation validation was the right call.

The full work list arising is `docs/FIXES.md`.

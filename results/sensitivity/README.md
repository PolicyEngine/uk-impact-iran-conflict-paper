# Sensitivity and robustness results

Produced by `analysis/run_sensitivity.py` from the private PolicyEngine UK
microdata, against the `realised_2026` scenario and a single baseline load.
Percentages of income are the **aggregate ratio** (weighted loss ÷ weighted
income) on the **equivalised AHC** concept, never a mean of household-level
ratios.

**This file deliberately quotes no numbers.** An earlier version narrated the
headline figures inline and went stale the moment the central specification was
re-specified, at which point it contradicted the paper. The numbers live in the
CSVs beside it and reach the manuscript only through
`analysis/emit_tex_values.py`.

## What each file contains

| File | Sweep | What it is for |
|---|---|---|
| `elasticity.csv` | Demand response, ε = 0 to −0.8, plus five published specifications | The main specification is ε = 0 by construction (first-order, an explicit upper bound). Reports both the change in **spending** and the **compensating-variation bounds** (Paasche below, Laspeyres above). The two differ enormously and only the second is a welfare measure — reporting spending alone was a real error in an earlier draft, since it counts foregone heating as costless. |
| `cap_lag.csv` | Wholesale-to-retail lag, 1–4 quarters | Separates the burden of the shock from the calendar year it is booked to. The cumulative invariance is an **identity by construction** (the phase-in weights sum to the same total at every lag), not a finding. |
| `asymmetry.csv` | Marginal-pricing share, 0.70–1.00 | How far a wholesale gas move reaches the electricity cap. Compositional: it moves the gas/electricity split of the loss, not the distributional gradient. Distinct from the gas/pump **damping ratio**, which is what moves the motor-fuel share. |
| `domestic_leg.csv` | The four parameters scaling the domestic channel | The pre-war gas reference, the two wholesale bill shares, and the split of the cap anchor between the sustained fraction and the phase-in. Only the *product* of the last two is identified by the Cornwall anchor, so the split is swept rather than assumed. |
| `policy_envelope.csv` | Five instruments at a common exchequer envelope | Scoring instruments at their sponsors' own costs compares a £1.3bn scheme with a £5.4bn one; this holds spend fixed so the comparison is about design rather than budget. |
| `fuel_by_decile.csv` | Petrol, diesel and diesel share by decile | Settles whether the decile-8 cash peak is mileage or the diesel uplift. It is both. |

## Reading these honestly

Two of the sweeps bound the paper's central claims rather than decorating them.
`domestic_leg.csv` shows the motor-fuel share moving substantially on
assumptions that are calibrations rather than estimates, and `elasticity.csv`
shows that the demand response — long presented as the paper's largest
uncertainty — is close to its smallest once measured as welfare. The ordering of
uncertainty in the paper's appendix follows from these files, not the reverse.

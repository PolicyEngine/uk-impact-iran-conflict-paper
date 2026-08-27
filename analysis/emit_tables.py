#!/usr/bin/env python3
r"""Emit LaTeX table bodies into ``paper/tables/`` from the canonical results.

Each output file is a **standalone ``tabular``** that ``paper/sections/results.tex``
can ``\input`` inside its own ``table`` float, so the caption, label, placement
and float style stay in the prose file and only the numbers are machine-written.

Usage:  ``python analysis/emit_tables.py``  ->  ``paper/tables/tab_*.tex``

Tables
------
``tab_decile.tex``        loss by income decile: £, % of income, share of the
                          total loss, households (m)
``tab_intra_decile.tex``  within-decile p10/p50/p90 loss % and the share of
                          households losing more than 5% and 10% of income
``tab_scenario.tex``      the three scenarios side by side
``tab_scorecard.tex``     the five policies on the paper's scoring metrics
``tab_region.tex``        the 12 regions: mean loss £, % of income, households
``tab_variants.tex``      four realised-2026 specifications side by side
                          (main, peak-fuel upper bound, ONS motor-fuel shape,
                          calendar 2026), from
                          ``results/robustness/comparison.csv``, with the
                          gradient's three companions as rows
``tab_specifications.tex`` all eleven specifications, one per row, ruled off
                          between the accounting choices and the two window
                          changes: the full post-referee comparison
``tab_envelope.tex``     the five instruments at a common exchequer envelope,
                          across the five row types, with the continuous
                          compensation measures. ``feasible_max`` (the
                          instrument's own ceiling) and ``common_capped``
                          (envelope absorption) are separate rows and are
                          labelled as the different questions they are
``tab_domestic_leg.tex`` the domestic-leg parameter sweep: what the Cornwall
                          anchor pins and what it does not

Conventions: ``booktabs`` rules only (no vertical rules), numeric columns
right-aligned (``r``), a machine-generated header comment in every file.
Requires ``\usepackage{booktabs}`` in the preamble.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "tables"

#: The observed shock is the paper's headline; the NIESR paths bound it.
CENTRAL_SCENARIO = "realised_2026"

#: Display order and labels for the scenario comparison table.
SCENARIO_LABELS = (
    ("niesr_baseline", "NIESR baseline"),
    ("niesr_adverse", "NIESR adverse"),
    ("realised_2026", "Realised 2026"),
)

#: Policy file stem -> short display label. Ordered as the paper argues them.
POLICY_LABELS = (
    ("social_tariff", "Social tariff"),
    ("jrf_block", "JRF block"),
    ("whd_expansion", "WHD expansion"),
    ("vat_zero", "VAT zero-rate"),
    ("ippr_rebate", "Flat rebate"),
)


#: Small-integer -> English word, so a caption reads "three rows", not "3 rows".
NUMBER_WORD = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
)


def mc(*lines: str, align: str = "c") -> str:
    r"""A stacked (multi-line) column header, without needing ``makecell``.

    A nested one-column ``tabular`` bottom-aligned to the header row keeps the
    column narrow: the widest line sets the column width instead of the whole
    header running on one line. ``booktabs`` rules are unaffected.
    """
    inner = r" \\ ".join(lines)
    return r"\begin{tabular}[b]{@{}" + align + r"@{}}" + inner + r"\end{tabular}"


def small(body: list[str]) -> list[str]:
    """Group a tabular inside ``\\small`` (and slightly tighter column
    separation) so the wide numeric tables fit the text block. The caption sits
    outside the group and keeps body size.
    """
    return [r"{\small\setlength{\tabcolsep}{4.5pt}"] + body + [r"}"]


def tight(body: list[str]) -> list[str]:
    r"""As :func:`small`, but tighter still.

    Used for the two widest tables (nine and seven numeric columns), which do
    not fit the 6.3in text block at the standard 4.5pt column separation.
    """
    return [r"{\small\setlength{\tabcolsep}{2.6pt}"] + body + [r"}"]


def jload(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def region_label(name: str) -> str:
    """``EAST_MIDLANDS`` -> ``East Midlands``, with the usual UK exceptions."""
    words = [w.capitalize() for w in str(name).split("_")]
    fixed = {"Of": "of", "And": "and", "The": "the"}
    words = [words[0]] + [fixed.get(w, w) for w in words[1:]]
    return " ".join(words)


def decile_row(shock: dict, block: str, decile: int) -> dict:
    for row in shock[block]:
        if int(row["decile"]) == decile:
            return row
    raise KeyError(f"{block} has no decile {decile}")


def header(name: str, standalone: bool = True) -> list[str]:
    return [
        f"% {name} — machine-generated by analysis/emit_tables.py",
        f"% generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"from results/ (central scenario: {CENTRAL_SCENARIO}). DO NOT EDIT BY HAND.",
        (
            "% A standalone tabular: \\input this inside a table float that "
            "carries the caption and label. Requires \\usepackage{booktabs}."
            if standalone
            else "% A complete table float, caption and label included: the "
            "appendix inputs this directly. Requires \\usepackage{booktabs}."
        ),
    ]


def write(name: str, body: list[str], standalone: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(header(name, standalone) + body) + "\n")
    print(f"wrote paper/tables/{name}")


def cost_per_pound_decile_one(policy: dict, shock: dict | None = None) -> float:
    """Total exchequer cost per £1 of gain reaching decile one.

    Stored **already dimensionless** by the rebuilt scorecard (cost divided by
    the aggregate gain reaching decile one, >= 1 by construction), so it is
    read straight through and rendered without a pound sign. See
    ``cost_per_pound_decile_one_units`` in the results files.
    """
    return float(policy["cost_per_pound_decile_one"])


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def tab_decile(shock: dict) -> None:
    body = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Decile & Loss (\pounds) & Loss (\% of income) & Share of total loss (\%) "
        r"& Households (m) \\",
        r"\midrule",
    ]
    for row in shock["decile"]:
        body.append(
            f"{int(row['decile'])} & {row['mean_loss_gbp']:,.0f} & "
            f"{row['mean_loss_pct']:.2f} & {100 * row['share_of_total_loss']:.1f} & "
            f"{row['households_m']:.2f} \\\\"
        )
    body += [
        r"\midrule",
        f"All & {shock['mean_loss_gbp']:,.0f} & {shock['mean_loss_pct']:.2f} & "
        f"100.0 & {sum(r['households_m'] for r in shock['decile']):.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_decile.tex", body)


def tab_intra_decile(shock: dict) -> None:
    body = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Loss (\% of income)} & "
        r"\multicolumn{2}{c}{Share of households (\%)} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}",
        r"Decile & p10 & p50 & p90 & Losing $>$5\% & Losing $>$10\% \\",
        r"\midrule",
    ]
    for row in shock["intra_decile"]:
        body.append(
            f"{int(row['decile'])} & {row['p10_loss_pct']:.2f} & "
            f"{row['p50_loss_pct']:.2f} & {row['p90_loss_pct']:.2f} & "
            f"{100 * row['share_above_5pct']:.1f} & "
            f"{100 * row['share_above_10pct']:.1f} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_intra_decile.tex", body)


def tab_scenario() -> None:
    cells = []
    for key, label in SCENARIO_LABELS:
        shock = jload(f"{key}/shock.json")
        cells.append((label, shock))
    body = [
        r"\begin{tabular}{l" + "r" * len(cells) + "}",
        r"\toprule",
        " & " + " & ".join(label for label, _ in cells) + r" \\",
        r"\midrule",
    ]

    def line(name: str, fn, fmt: str = "{:.2f}") -> str:
        return name + " & " + " & ".join(fmt.format(fn(s)) for _, s in cells) + r" \\"

    body += [
        line(
            r"Aggregate additional spend (\pounds bn)",
            lambda s: s["aggregate_cost_bn"],
            "{:.1f}",
        ),
        line(r"Mean loss (\pounds)", lambda s: s["mean_loss_gbp"], "{:,.0f}"),
        line(r"Mean loss (\% of income)", lambda s: s["mean_loss_pct"]),
        r"\midrule",
        line(
            r"Decile 1 loss (\pounds)",
            lambda s: decile_row(s, "decile", 1)["mean_loss_gbp"],
            "{:,.0f}",
        ),
        line(
            r"Decile 10 loss (\pounds)",
            lambda s: decile_row(s, "decile", 10)["mean_loss_gbp"],
            "{:,.0f}",
        ),
        line(
            r"Decile 1 loss (\% of income)",
            lambda s: decile_row(s, "decile", 1)["mean_loss_pct"],
        ),
        line(
            r"Decile 10 loss (\% of income)",
            lambda s: decile_row(s, "decile", 10)["mean_loss_pct"],
        ),
        r"\midrule",
        line(
            r"Gas share of loss (\%)", lambda s: 100 * s["gas_share_of_loss"], "{:.1f}"
        ),
        line(
            r"Electricity share of loss (\%)",
            lambda s: 100 * s["electricity_share_of_loss"],
            "{:.1f}",
        ),
        line(
            r"Motor fuel share of loss (\%)",
            lambda s: 100 * s["motor_fuel_share_of_loss"],
            "{:.1f}",
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_scenario.tex", body)


def tab_scorecard(shock: dict) -> None:
    body = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Policy & "
        + mc(r"Cost", r"(\pounds bn)")
        + " & "
        + mc(r"Share to", r"deciles 1--3 (\%)")
        + " & "
        + mc(r"Loss", r"offset (\%)")
        + " & "
        + mc(r"Mean res-", r"idual (\pounds)")
        + " & "
        + mc(r"Losers un-", r"compensated (\%)")
        + " & "
        # The other tail of the same distribution. The saturation result the
        # referees called a scaling artefact is, at bottom, about paying
        # households more than they lost, and an instrument can be scored as
        # generous while overshooting most of the people it reaches.
        + mc(r"Recipients over-", r"compensated (\%)")
        + r" \\",
        r"\midrule",
    ]
    for key, label in POLICY_LABELS:
        p = jload(f"{CENTRAL_SCENARIO}/{key}.json")
        body.append(
            f"{label} & {p['cost_bn']:.2f} & "
            f"{100 * p['share_to_bottom_three']:.1f} & "
            f"{100 * p['share_of_aggregate_loss_offset']:.1f} & "
            f"{p['mean_residual_loss_gbp']:,.0f} & "
            f"{100 * p['uncompensated_share_overall']:.1f} & "
            f"{100 * p['overcompensated_share_of_recipients']:.1f} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    # Seven numeric columns: tight() rather than small(), which is what the
    # 6.3in text block takes at this width.
    write("tab_scorecard.tex", tight(body))


def tab_region(shock: dict) -> None:
    rows = sorted(shock["region"], key=lambda r: -r["mean_loss_pct"])
    body = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Region & Mean loss (\pounds) & Mean loss (\% of income) & "
        r"Households (m) \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(
            f"{region_label(row['name'])} & {row['mean_loss_gbp']:,.0f} & "
            f"{row['mean_loss_pct']:.2f} & {row['households_m']:.2f} \\\\"
        )
    body += [
        r"\midrule",
        f"United Kingdom & {shock['mean_loss_gbp']:,.0f} & "
        f"{shock['mean_loss_pct']:.2f} & "
        f"{sum(r['households_m'] for r in rows):.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_region.tex", body)


#: The seven specifications, in the order the paper argues them. Keyed by the
#: ``variant`` column of ``results/robustness/comparison.csv``. Kept in one place
#: so ``tab_variants`` and ``tab_specifications`` cannot drift apart.
SPEC_ORDER = (
    ("main", ("Main", "specification")),
    ("steady_state", ("Steady", "state")),
    ("symmetric_damping", ("Symmetric", "damping")),
    ("peak_fuel", ("Peak fuel", "(upper bound)")),
    ("ons_shape", ("ONS", "shape")),
    ("ons_both_levels", ("ONS both", "levels")),
    ("unequivalised", ("Unequiv-", "alised")),
    # Round-3 additions. Rows, deliberately, rather than columns: the table is
    # already at nine numeric columns and 2.6pt column separation, which is what
    # fits the 6.3in text block. Rows are free.
    ("mt_fuel_parity", ("Means-tested", "fuel parity")),
    ("nts_participation", ("NTS partici-", "pation margin")),
    ("calendar_2026", ("Calendar", "2026")),
    ("peak_fuel_calendar_2026", ("Calendar 2026,", "peak fuel")),
)

#: Row label per variant, for the specifications table.
SPEC_ROW_LABELS = {
    "main": "Main specification",
    "steady_state": "Steady state",
    "symmetric_damping": "Symmetric damping",
    "peak_fuel": "Peak fuel (upper bound)",
    "ons_shape": "ONS shape",
    "ons_both_levels": "ONS both levels",
    "unequivalised": "Unequivalised",
    "mt_fuel_parity": "Means-tested fuel parity",
    "nts_participation": "NTS participation margin",
    "calendar_2026": "Calendar 2026",
    "peak_fuel_calendar_2026": "Calendar 2026, peak fuel",
}

#: Variants that change the *annualising window* rather than an accounting
#: choice, and so are not members of the range the paper quotes across
#: specifications. Separated by a rule in the table.
WINDOW_VARIANTS = ("calendar_2026", "peak_fuel_calendar_2026")


def comparison() -> dict[str, dict]:
    path = R / "robustness" / "comparison.csv"
    with path.open(newline="") as fh:
        return {r["variant"]: r for r in csv.DictReader(fh)}


def tab_variants() -> None:
    r"""The three headline realised-2026 specifications side by side.

    Read straight from ``results/robustness/comparison.csv`` (one row per
    variant) rather than re-deriving from the per-variant JSON, so the table and
    the audit trail cannot drift apart. Three numeric columns plus a label
    column fit the 6.3in text block comfortably at ``\small`` with stacked
    two-line headers. The full seven-specification comparison is
    ``tab_specifications``.
    """
    rows = comparison()
    order = (
        ("main", ("Main", "specification")),
        ("peak_fuel", ("Peak-fuel", "upper bound")),
        ("ons_shape", ("ONS-calibrated", "motor fuel")),
        # The calendar-2026 run, so the Resolution Foundation comparison is on
        # the page beside the specification it is being compared with rather
        # than only in the prose.
        ("calendar_2026", ("Calendar 2026", "(RF window)")),
    )
    cols = [rows[key] for key, _ in order if key in rows]
    if not cols:
        raise KeyError("comparison.csv has none of the expected variants")

    def line(name: str, fn, fmt: str = "{:.2f}") -> str:
        return name + " & " + " & ".join(fmt.format(fn(r)) for r in cols) + r" \\"

    def num(column: str):
        return lambda r: float(r[column])

    def pct(column: str):
        return lambda r: 100 * float(r[column])

    body = [
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        " & " + " & ".join(mc(*head) for _, head in order[: len(cols)]) + r" \\",
        r"\midrule",
        line(
            r"Aggregate additional spend (\pounds bn)",
            num("aggregate_cost_bn"),
            "{:.2f}",
        ),
        line(r"Mean loss (\pounds)", num("mean_loss_gbp"), "{:,.0f}"),
        line(r"Mean loss (\% of income)", num("mean_loss_pct")),
        r"\midrule",
        line(r"Decile 1 loss (\pounds)", num("decile1_loss_gbp"), "{:,.0f}"),
        line(r"Decile 1 loss (\% of income)", num("decile1_loss_pct")),
        line(r"Decile 10 loss (\pounds)", num("decile10_loss_gbp"), "{:,.0f}"),
        line(r"Decile 10 loss (\% of income)", num("decile10_loss_pct")),
        line(r"Decile 1/10 ratio (\% of income)", num("d1_d10_ratio_pct")),
        # The gradient's three companions. Decile one is where a fifth of
        # households have non-positive equivalised income, so a headline ratio
        # anchored there needs the ratio that is not, and both need a tail
        # treatment. Rows rather than a separate table: the point is that they
        # are read together.
        line(r"\quad from decile 2 (D2/D10)", num("d2_d10_ratio_pct")),
        line(r"\quad D1/D10, winsorised", num("d1_d10_ratio_pct_winsorised")),
        line(r"\quad D2/D10, winsorised", num("d2_d10_ratio_pct_winsorised")),
        r"\midrule",
        line(r"Gas share of loss (\%)", pct("gas_share_of_loss"), "{:.1f}"),
        line(
            r"Electricity share of loss (\%)",
            pct("electricity_share_of_loss"),
            "{:.1f}",
        ),
        line(
            r"Motor fuel share of loss (\%)", pct("motor_fuel_share_of_loss"), "{:.1f}"
        ),
        r"\midrule",
        # What the means-tested system actually reaches. The targeting claim in
        # the policy section scales inversely with this, so it belongs beside
        # the specifications rather than buried in an appendix.
        line(
            r"Means-tested share of loss (\%)",
            pct("means_tested_share_of_loss"),
            "{:.2f}",
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_variants.tex", small(body))


def tab_specifications() -> None:
    r"""All seven specifications, one per row.

    Specifications are the *rows* rather than the columns: nine short numeric
    columns fit the 6.3in text block at ``\small``, whereas seven wide numeric
    columns plus a row-label column would not. The motor-fuel share is the last
    column because it is the one the paper's central claim turns on and the one
    that moves most across the seven.
    """
    rows = comparison()
    body = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        "Specification & "
        + mc(r"Agg.", r"(\pounds bn)")
        + " & "
        + mc(r"Mean", r"(\pounds)")
        + " & "
        + mc(r"Mean", r"(\%)")
        + " & "
        + mc(r"D1", r"(\pounds)")
        + " & "
        + mc(r"D1", r"(\%)")
        + " & "
        + mc(r"D10", r"(\pounds)")
        + " & "
        + mc(r"D10", r"(\%)")
        + " & "
        + mc(r"D1/D10", r"(\%)")
        + " & "
        + mc(r"Motor fuel", r"(\% of loss)")
        + r" \\",
        r"\midrule",
    ]
    ruled = False
    for key, _head in SPEC_ORDER:
        r = rows.get(key)
        if r is None:
            continue
        if key in WINDOW_VARIANTS and not ruled:
            # The window changes are a different axis from the accounting
            # choices above them, and the paper's headline range is quoted over
            # the block above this rule only. A reader who takes a min and a max
            # down the whole column gets a number the paper does not claim.
            body.append(r"\midrule")
            ruled = True
        label = SPEC_ROW_LABELS[key]
        body.append(
            f"{label} & {float(r['aggregate_cost_bn']):.2f} & "
            f"{float(r['mean_loss_gbp']):,.0f} & "
            f"{float(r['mean_loss_pct']):.2f} & "
            f"{float(r['decile1_loss_gbp']):,.0f} & "
            f"{float(r['decile1_loss_pct']):.2f} & "
            f"{float(r['decile10_loss_gbp']):,.0f} & "
            f"{float(r['decile10_loss_pct']):.2f} & "
            f"{float(r['d1_d10_ratio_pct']):.2f} & "
            f"{100 * float(r['motor_fuel_share_of_loss']):.1f} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_specifications.tex", tight(body))


#: Common-envelope scorecard: the five instruments scored against one exchequer
#: envelope, with the five row types the rebuild made load-bearing.
#:
#: The round-3 referees' objection was to the second label. ``common_capped``
#: was printed as "at feasible max", and it is not one: it is
#: ``min(envelope, feasible-max cost)``, which for an instrument costing more
#: than the envelope scales generosity *down*. The genuine feasible maximum —
#: the instrument's own ceiling with no envelope applied — is now its own row,
#: and the two are labelled so a reader can see they are different questions.
#:
#: ``stated`` is omitted from this table; it is ``tab_scorecard``.
ENVELOPE_ROWS = (
    ("feasible_max", "own ceiling"),
    ("common_capped", "absorbs envelope"),
    ("common_scaled", "scaled"),
    ("common_eligibility", "wider eligibility"),
)

#: A row is treated as universal — means-tested in name only — at or above this
#: eligible share. Mirrors ``policies.UNIVERSAL_ELIGIBILITY_TOLERANCE``, written
#: literally so this emitter stays importable without the package.
UNIVERSAL_ELIGIBILITY_TOLERANCE = 0.999


def tab_envelope() -> None:
    r"""The five instruments against a common exchequer envelope.

    ``results/sensitivity/policy_envelope.csv`` carries up to five rows per
    policy. Four of them appear here, one row each, grouped under the
    instrument's name:

    ``feasible_max``        the instrument at its OWN ceiling, uncapped by any
                            envelope. This is the true feasible maximum and the
                            only row down which feasible maxima are comparable:
                            the JRF block reaches £21.9bn here, more than four
                            times the envelope.
    ``common_capped``       envelope **absorption**: ``min(envelope,
                            feasible-max cost)``. It answers how much of the
                            £5bn the instrument can take, which for an
                            instrument that already costs more than the
                            envelope scales generosity *down*. Round-3 referees
                            flagged this row being printed as "at feasible
                            max", which it is not.
    ``common_scaled``       the old proportional scaling to the full envelope,
                            kept because it is what the withdrawn claim rested
                            on, and flagged with a dagger where the implied
                            parameter is not a thing that can exist.
    ``common_eligibility``  the envelope spent on widening eligibility at the
                            sponsor's own generosity, where that is defined.
                            Marked with a double dagger where widening takes
                            the scheme all the way to universal, at which point
                            it is no longer a targeted instrument at all.

    The withdrawn claim is "VAT zero-rating wins at a common envelope". It won
    only on the scaled row, by removing more VAT points than the tax has.
    """
    with (SENS / "policy_envelope.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_key = {(r["policy"].strip(), r["envelope"].strip()): r for r in rows}
    envelope = next(
        (
            float(r["envelope_bn"])
            for r in rows
            if r["envelope"].strip() == "common_capped" and r["envelope_bn"]
        ),
        float("nan"),
    )

    def units_short(row: dict) -> str:
        """A four-character-wide unit tag for the implied-parameter column."""
        return (
            r"\%" if not row["parameter_units"].strip().startswith("£") else r"\pounds"
        )

    def param_cell(row: dict) -> str:
        value = float(row["implied_parameter"])
        text = (
            f"{value:,.0f}" if abs(value) >= 100 else f"{value:.1f}".removesuffix(".0")
        )
        unit = units_short(row)
        cell = f"{text}{unit}" if unit == r"\%" else f"{unit} {text}"
        if row["is_feasible"].strip().lower() != "true":
            cell += r"$^\dagger$"
        return cell

    def is_universal(row: dict) -> bool:
        """True where eligibility has widened to (almost) every household.

        A "means-tested" instrument whose eligibility arm ends up covering
        everyone is not a targeted instrument any more, and a table that lets
        that row sit unmarked beside the sponsor's own targeted design invites
        exactly the comparison it should be blocking.
        """
        share = row.get("eligible_share", "").strip()
        if not share:
            return False
        try:
            return float(share) >= UNIVERSAL_ELIGIBILITY_TOLERANCE
        except ValueError:
            return False

    body = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        "Instrument & Variant & "
        + mc(r"Spend", r"(\pounds bn)")
        + " & "
        + mc(r"Implied", r"setting")
        + " & "
        + mc(r"Share to", r"D1--3 (\%)")
        + " & "
        + mc(r"Loss", r"offset (\%)")
        + " & "
        + mc(r"Mean res-", r"idual (\pounds)")
        + " & "
        + mc(r"Un-comp-", r"ensated (\%)")
        + r" \\",
        r"\midrule",
    ]
    infeasible = 0
    universal = 0
    for key, label in POLICY_LABELS:
        first = True
        for envelope_key, variant in ENVELOPE_ROWS:
            r = by_key.get((key, envelope_key))
            if r is None:
                continue
            if r["is_feasible"].strip().lower() != "true":
                infeasible += 1
            variant_cell = variant
            if is_universal(r):
                universal += 1
                variant_cell += r"$^\ddagger$"
            body.append(
                f"{label if first else ''} & {variant_cell} & "
                f"{float(r['cost_bn']):.2f} & "
                f"{param_cell(r)} & "
                f"{100 * float(r['share_to_bottom_three']):.1f} & "
                f"{100 * float(r['share_of_aggregate_loss_offset']):.1f} & "
                f"{float(r['mean_residual_loss_gbp']):,.0f} & "
                f"{100 * float(r['uncompensated_share_overall']):.0f} \\\\"
            )
            first = False
        if not first:
            body.append(r"\addlinespace")
    if body[-1] == r"\addlinespace":
        body.pop()
    body += [r"\bottomrule", r"\end{tabular}"]
    # Caption inputs, read off the same rows the body was built from rather
    # than written into the sentence by hand.
    feasible_max_costs = [
        float(r["cost_bn"])
        for (policy, kind), r in by_key.items()
        if kind == "feasible_max" and r["cost_bn"]
    ]
    feasible_max_bn = max(feasible_max_costs) if feasible_max_costs else float("nan")
    saturating = sum(1 for c in feasible_max_costs if c < envelope)
    write(
        "tab_envelope.tex",
        float_wrap(
            tight(body),
            "The five instruments against a common exchequer envelope of "
            f"\\pounds {envelope:.0f}bn, realised 2026 scenario. The first two "
            "rows for each instrument answer different questions and the "
            "pre-revision table conflated them. ``Own ceiling'' is the "
            "instrument at its own feasible maximum with no envelope applied "
            "at all: the JRF block reaches "
            f"\\pounds {feasible_max_bn:.1f}bn there, more than "
            f"{feasible_max_bn / envelope:.0f} times the envelope. ``Absorbs "
            "envelope'' is the budget constraint --- the smaller of the "
            "envelope and that ceiling --- so for an instrument that already "
            "costs more than \\pounds "
            f"{envelope:.0f}bn it scales generosity \\emph{{down}}, and for "
            f"the {NUMBER_WORD[saturating]} that saturate below the envelope it "
            "reports the smaller sum they can actually spend. Neither row holds "
            "spend fixed across instruments. ``Scaled'' is proportional scaling "
            "to the full envelope regardless of feasibility; $\\dagger$ marks "
            f"the {NUMBER_WORD[infeasible]} rows whose implied setting does not "
            "exist (a bill discount above 100 per cent, more VAT points than "
            "the tax has). ``Wider eligibility'' spends the envelope on who is "
            "in the scheme rather than on how generous it is, at the sponsor's "
            "own rate, and is defined only for the two means-tested schemes; "
            f"$\\ddagger$ marks the {NUMBER_WORD[universal]} such rows where "
            "widening reaches every household, at which point the instrument is "
            "means-tested in name only. Any claim that one instrument ``wins at "
            "a common envelope'' rests on the scaled rows and is withdrawn.",
            "tab:envelope",
        ),
        standalone=False,
    )


#: Row labels for the domestic-leg parameter sweep.
LEG_LABELS = {
    "sustained_fraction_split": r"Sustained fraction",
    "prewar_nbp_pence_per_therm": "Pre-war NBP",
    "wholesale_share_gas_bill": "Wholesale share, gas",
    "wholesale_share_electricity_bill": "Wholesale share, elec.",
    # Added in round 3, and the block that matters: the monthly shape of the
    # wholesale gas peak was never swept, and it turns out to carry the whole
    # domestic leg.
    "gas_peak_monthly_profile": "Gas peak profile",
}


def leg_is_identified(row: dict) -> bool:
    """True if this sweep cell solved at all.

    Two cells of the gas-profile block do not: the two published caps fail to
    pin a pre-war counterfactual, either because the later observation window
    prices no more of the shock than the earlier one, or because the implied
    sustained fraction leaves ``(0, 1]``. Those rows are written with every
    numeric column empty. They are printed, because a sweep cell that cannot be
    solved is a result about identification and not a gap in the table, but
    they cannot be formatted as numbers.
    """
    flag = str(row.get("identified", "")).strip().lower()
    if flag in {"false", "0", "no"}:
        return False
    return str(row.get("aggregate_cost_bn", "")).strip() != ""


def leg_value_text(row: dict) -> str:
    """Render the swept value, which is numeric in four blocks and a tag in one.

    The gas-profile block sweeps a *shape*, not a scalar, so its values are
    labels like ``shift-1m`` and ``flatten0.8``. ``float()`` on those is what
    took this table down.
    """
    raw = str(row["value"]).strip()
    try:
        return f"{float(raw):g}"
    except ValueError:
        return raw.replace("_", " ")


def tab_domestic_leg() -> None:
    r"""The A4 parameter sweep: what the Cornwall anchor does and does not pin.

    The anchor identifies only the *product* of ``sustained_fraction`` and the
    first phase-in quarter, so the first block sweeps the split at a constant
    product; the remaining three blocks sweep the parameters that scale the
    domestic channel one-for-one and were previously unswept.
    """
    with (SENS / "domestic_leg.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    body = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        "Parameter & "
        + mc(r"Value", align="l")
        + " & "
        + mc(r"Agg.", r"(\pounds bn)")
        + " & "
        + mc(r"Mean", r"(\pounds)")
        + " & "
        + mc(r"Mean", r"(\%)")
        + " & "
        + mc(r"D1", r"(\%)")
        + " & "
        + mc(r"D10", r"(\%)")
        + " & "
        + mc(r"Motor fuel", r"(\% of loss)")
        + r" \\",
        r"\midrule",
    ]
    # The paper's own cell, matched against the central aggregate as it stands
    # rather than against a literal. The old code compared to a hardcoded
    # 8.957518848, which the rebuild moved: every row silently stopped being
    # marked "(paper)" and the table lost its anchor with no error anywhere.
    paper_agg = float(jload(f"{CENTRAL_SCENARIO}/shock.json")["aggregate_cost_bn"])

    last = None
    unidentified = 0
    for r in rows:
        parameter = r["parameter"].strip()
        if last is not None and parameter != last:
            body.append(r"\midrule")
        label = LEG_LABELS.get(parameter, parameter) if parameter != last else ""
        last = parameter
        shown = leg_value_text(r)
        if not leg_is_identified(r):
            unidentified += 1
            # One spanning cell rather than six repetitions of an em-dash: the
            # row failed as a whole, not column by column.
            body.append(
                f"{label} & {shown}$^\\dagger$ & "
                r"\multicolumn{6}{c}{\emph{not identified}} \\"
            )
            continue
        if abs(float(r["aggregate_cost_bn"]) - paper_agg) < 1e-6:
            shown += " (paper)"
        body.append(
            f"{label} & {shown} & "
            f"{float(r['aggregate_cost_bn']):.2f} & "
            f"{float(r['mean_loss_gbp']):,.0f} & "
            f"{float(r['mean_loss_pct']):.2f} & "
            f"{float(r['decile1_loss_pct']):.2f} & "
            f"{float(r['decile10_loss_pct']):.2f} & "
            f"{100 * float(r['motor_fuel_share_of_loss']):.1f} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    # Caption inputs from the gas-profile block itself, so the sentence and the
    # table cannot disagree.
    profile = [
        r
        for r in rows
        if r["parameter"].strip() == "gas_peak_monthly_profile" and leg_is_identified(r)
    ]

    def span(column: str, select, scale: float = 1.0) -> float:
        return scale * select(float(r[column]) for r in profile)

    cap_spread = span("prewar_counterfactual_cap_gbp", max) - span(
        "prewar_counterfactual_cap_gbp", min
    )
    agg_min = span("aggregate_cost_bn", min)
    agg_max = span("aggregate_cost_bn", max)
    fuel_min = span("motor_fuel_share_of_loss", min, 100)
    fuel_max = span("motor_fuel_share_of_loss", max, 100)
    write(
        "tab_domestic_leg.tex",
        float_wrap(
            tight(body),
            "Domestic-leg parameter sweep, realised 2026 scenario. The two "
            "published caps identify only the product of the sustained "
            "fraction and the phase-in at the anchor quarter, so the first "
            "block varies the split at a constant product; the next three vary "
            "the parameters that scale the domestic channel one for one. The "
            "last block is the one that matters and was previously unswept: "
            "the monthly shape of the wholesale gas peak, which the "
            "observation-window phase-in reads directly. Shifting that shape by "
            "a month either way, or flattening it, moves the solved pre-war "
            f"counterfactual cap over a \\pounds {cap_spread:,.0f} range, the "
            f"aggregate from \\pounds {agg_min:.1f}bn to \\pounds {agg_max:.1f}bn, "
            f"and motor fuel's share of the loss from {fuel_min:.0f} to "
            f"{fuel_max:.0f} per cent --- from a bare majority to four fifths. "
            f"$\\dagger$ marks the {NUMBER_WORD[unidentified]} cells in which "
            "the two cap observations do not identify a counterfactual at all, "
            "because the later observation window prices no more of the shock "
            "than the earlier one or the implied sustained fraction leaves "
            "$(0, 1]$. This block, not the elasticity sweep, is the paper's "
            "real identification fragility.",
            "tab:domesticleg",
        ),
        standalone=False,
    )


# ---------------------------------------------------------------------------
# appendix sweep tables (results/sensitivity/*.csv)
# ---------------------------------------------------------------------------

SENS = R / "sensitivity"


def sload(name: str) -> list[dict]:
    with (SENS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def float_wrap(body: list[str], caption: str, label: str) -> list[str]:
    """Wrap a tabular in its own float.

    The appendix inputs these three files directly rather than inside a
    ``table`` environment of its own, so the caption and label live here.
    """
    return (
        [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
        ]
        + body
        + [r"\end{table}"]
    )


#: Display names for the named elasticity specifications, in sweep order.
ELASTICITY_LABELS = {
    "labandeira_short_run": "Labandeira et al., short run",
    "labandeira_long_run": "Labandeira et al., long run",
    "priesmann_short_run": "Priesmann \\& Praktiknjo, short run",
    "priesmann_long_run": "Priesmann \\& Praktiknjo, long run",
    "prior_repo_replication": "Prior-repo replication",
}


def minus(text: str) -> str:
    """ASCII hyphen -> LaTeX math minus, for numeric cells only."""
    return text.replace("-", "$-$")


def elasticity_label(row: dict) -> str:
    spec = row["spec"]
    if spec in ELASTICITY_LABELS:
        return ELASTICITY_LABELS[spec]
    eps = float(row["epsilon_mean"])
    if eps == 0:
        return r"Flat $\varepsilon = 0$ (main spec.)"
    # already inside math mode, so the ASCII hyphen renders as a minus
    return f"Flat $\\varepsilon = {eps:.1f}$"


def tab_elasticity() -> None:
    rows = sload("elasticity.csv")
    body = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Specification & "
        + mc(r"Mean", r"$\varepsilon$")
        + " & "
        + mc(r"Mean loss", r"(\pounds)")
        + " & "
        # No per-cent-of-income column: the data rows carry the aggregate spend
        # change here, and an orphan header made the tabular 8 wide against 7
        # columns of data, which is a hard LaTeX error.
        + mc(r"Spend", r"(\pounds bn)")
        + " & "
        + mc(r"CV bounds", r"(\pounds bn)")
        + " & "
        + mc(r"Welfare", r"shaved (\%)")
        + " & "
        + mc(r"Spend", r"shaved (\%)")
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lo = float(row["cv_lower_bn"])
        hi = float(row["cv_upper_bn"])
        cells = [
            minus(f"{float(row['epsilon_mean']):.2f}"),
            f"{float(row['mean_loss_gbp']):,.0f}",
            f"{float(row['aggregate_loss_bn']):.2f}",
            f"{lo:.2f}--{hi:.2f}",
            f"{100 * float(row['welfare_share_shaved']):.1f}",
            f"{100 * float(row['share_of_upper_bound_shaved']):.0f}",
        ]
        body.append(elasticity_label(row) + " & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write(
        "tab_elasticity.tex",
        float_wrap(
            small(body),
            "Demand-response sweep, realised 2026 scenario. ``Spend'' is the "
            "change in expenditure; the money-metric statement is the pair of "
            "bounds on the compensating variation (Paasche below, Laspeyres "
            "above). Reading the spending column as the loss counts foregone "
            "heating as costless. On the welfare measure the strongest response "
            "in the grid shaves only a tenth of the loss, against four fifths of "
            "the spending change --- so demand response is a small source of "
            "uncertainty here, not the largest one.",
            "tab:elasticity",
        ),
        standalone=False,
    )


def tab_caplag() -> None:
    r"""The wholesale-to-retail lag sweep, on both anchoring rules.

    The rebuilt sweep is two-way: each lag appears once ``anchored`` (the
    sustained fraction re-solved so the Cornwall cap anchor still binds) and
    once ``unanchored`` (that fraction held at the central value). The two
    coincide up to two quarters and separate after, which is worth showing: the
    long-lag rows are the ones where the choice of anchoring rule, not the lag
    itself, is doing the work. ``lag_quarters`` is now a float and the
    cumulative columns are gone, so the table reports the annualised figures
    and the two legs behind them.
    """
    rows = sload("cap_lag.csv")
    lags = sorted({float(r["lag_quarters"]) for r in rows})
    by_key = {(float(r["lag_quarters"]), r["anchor"].strip()): r for r in rows}
    central = next(
        (r for r in rows if r["is_central_specification"].strip().lower() == "true"),
        None,
    )
    central_lag = float(central["lag_quarters"]) if central else None

    def fmt_lag(value: float) -> str:
        text = f"{value:g}"
        return f"{text} (paper)" if value == central_lag else text

    body = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Booked in the modelled year} & "
        r"\multicolumn{2}{c}{By leg (\pounds bn)} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        mc(r"Lag", r"(quarters)", align="l")
        + " & "
        + mc(r"Cap", r"anchoring", align="l")
        + " & "
        + mc(r"Mean", r"(\pounds)")
        + " & "
        + mc(r"Total", r"(\pounds bn)")
        + " & Domestic & "
        + mc(r"Motor", r"fuel")
        + " & "
        + mc(r"Motor fuel", r"share (\%)")
        + r" \\",
        r"\midrule",
    ]
    unidentified = 0
    for lag in lags:
        first = True
        for anchor in ("anchored", "unanchored"):
            r = by_key.get((lag, anchor))
            if r is None:
                continue
            # Three of the five anchored lags do not solve: re-anchoring at a
            # one-quarter lag leaves the two observation windows unable to
            # separate the published caps, and at three and four quarters the
            # implied pre-war counterfactual is not strictly below the observed
            # July cap. Their rows are blank, and printing the failure is more
            # informative than dropping the lag.
            if not leg_is_identified(
                {
                    "identified": r.get("identified", ""),
                    "aggregate_cost_bn": r["mean_loss_gbp"],
                }
            ):
                unidentified += 1
                body.append(
                    f"{fmt_lag(lag) if first else ''} & {anchor} & "
                    r"\multicolumn{5}{c}{\emph{not identified}} \\"
                )
                first = False
                continue
            body.append(
                f"{fmt_lag(lag) if first else ''} & {anchor} & "
                f"{float(r['mean_loss_gbp']):,.0f} & "
                f"{float(r['aggregate_loss_bn']):.2f} & "
                f"{float(r['domestic_loss_bn']):.2f} & "
                f"{float(r['motor_fuel_loss_bn']):.2f} & "
                f"{100 * float(r['motor_fuel_share_of_loss']):.1f} \\\\"
            )
            first = False
    body += [r"\bottomrule", r"\end{tabular}"]
    write(
        "tab_caplag.tex",
        float_wrap(
            small(body),
            "Wholesale-to-retail lag sweep, realised 2026 scenario, on both "
            "anchoring rules. ``Anchored'' re-solves the sustained fraction at "
            "each lag so the Cornwall cap anchor still binds; ``unanchored'' "
            "holds it at the central value and so varies the timing alone. The "
            "cumulative burden is very nearly invariant along the unanchored "
            "series --- the phase-in weights mostly move the burden between "
            "calendar years rather than changing its size --- so what moves in "
            "the columns below is the attribution to the modelled year. The "
            "two rules agree to two quarters and separate after, which is where "
            "the anchoring rule rather than the lag is doing the work. Anchoring "
            f"is not always possible: {NUMBER_WORD[unidentified]} of the "
            "anchored rows do not solve at all, because at those lags the two "
            "published caps' observation windows either fail to separate them "
            "or imply a pre-war counterfactual that is not strictly below the "
            "observed July cap. That the paper's own 1.5-quarter lag is one of "
            "the few that does solve is a constraint on the calibration, not a "
            "coincidence in its favour.",
            "tab:caplag",
        ),
        standalone=False,
    )


def tab_asymmetry() -> None:
    rows = sload("asymmetry.csv")
    body = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        mc(r"Marginal-", r"pricing share", align="l")
        + " & "
        + mc(r"Electricity", r"factor")
        + " & "
        + mc(r"Mean loss", r"(\pounds)")
        + " & "
        + mc(r"Loss", r"(\% inc.)")
        + " & "
        + mc(r"Decile", r"1/10 ratio")
        + " & "
        + mc(r"Gas share of", r"domestic loss (\%)")
        + " & "
        + mc(r"Aggregate", r"(\pounds bn)")
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        share = float(row["marginal_pricing_share"])
        tag = (
            f"{share:.2f} (paper)"
            if row["is_paper_central"] == "True"
            else f"{share:.2f}"
        )
        body.append(
            f"{tag} & {float(row['electricity_price_factor']):.4f} & "
            f"{float(row['mean_loss_gbp']):,.0f} & "
            f"{float(row['mean_loss_pct']):.2f} & "
            f"{float(row['decile_ratio_pct']):.2f} & "
            f"{100 * float(row['gas_share_of_domestic_loss']):.1f} & "
            f"{float(row['aggregate_loss_bn']):.2f} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    write(
        "tab_asymmetry.tex",
        float_wrap(
            small(body),
            "Marginal-pricing-share sweep, realised 2026 scenario. The gas "
            "price factor is held at 1.1404 throughout; only the electricity "
            "pass-through varies. The headline incidence barely moves, while the "
            "composition of the domestic loss moves materially.",
            "tab:asymmetry",
        ),
        standalone=False,
    )


def main() -> None:
    shock = jload(f"{CENTRAL_SCENARIO}/shock.json")
    tab_decile(shock)
    tab_intra_decile(shock)
    tab_scenario()
    tab_scorecard(shock)
    tab_region(shock)
    tab_variants()
    tab_specifications()
    tab_envelope()
    tab_domestic_leg()
    tab_elasticity()
    tab_caplag()
    tab_asymmetry()


if __name__ == "__main__":
    main()

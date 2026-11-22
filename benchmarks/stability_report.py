#!/usr/bin/env python3
"""Render the accuracy tables, CSV summary and key figures from stability.py results.

I is inazumaq2n with its moments, F is inazumaq2n with the fused sums of
fastCD, T is I and G is F with the block merged from four tiles, U is
inazumaq2n without the evaluation by definition of marked blocks, and Q and
D evaluate the definition pixel by pixel with pyfastcd's quasilinear and
direct products. Studies made before the fused sums have no F, and their
tables are rendered as they were. The same tables are written to the study
directory and, with --update-docs, into the delimited accuracy section of
docs/stability.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN STABILITY ACCURACY -->"
END = "<!-- END STABILITY ACCURACY -->"
TYPICAL = ("uniform", "gaussian")
OFFSETS = ("offset_1e3", "offset_1e6", "offset_1e9", "offset_1e12")
HAZARDS = (
    "zero_mean",
    "near_zero_mean",
    "constant_integer",
    "constant_fraction",
    "scaled_down",
    "scaled_up",
    "overflow",
)
METHODS = ("I", "F", "T", "G", "U", "Q", "D")
MAIN = ("I", "F", "Q", "D")
GUARDED = ("I", "F", "U", "Q", "D")
SIDE = 32
FAMILY_BANDS = (4, 16)
# An error above this is a different value, not a rounding error
WRONG = 1e-6
FOOTER = (
    "The archived CSV summary also gives, for every family, band count, block size "
    "and method, the count, the median, mean, p95, p99 and maximum of eta, the "
    "relative errors, the number of results that are not finite, and the share of "
    "cases in which I is at least as accurate. Groups with fewer than 100 cases "
    "have an empirical p99 close to their maximum and large sampling uncertainty."
)
FIELDS = (
    "family",
    "bands",
    "dimension",
    "side",
    "method",
    "samples",
    "median_eta",
    "mean_eta",
    "p95_eta",
    "p99_eta",
    "max_eta",
    "median_relative",
    "max_relative",
    "nonfinite",
    "marked",
    "fraction_I_at_least_as_accurate",
    "fraction_tied",
)

Rows = list[dict[str, Any]]


def fmt(value: Any) -> str:
    """Format a statistic to three significant digits, or a dash when undefined."""
    return "—" if value is None else f"{value:.3g}"


def values(rows: Rows, method: str, key: str) -> list[float]:
    """Return one measure of one method over rows, leaving out undefined values."""
    data = (r["methods"][method][key] for r in rows)
    return [v for v in data if v is not None]


def statistic(
    rows: Rows, method: str, key: str, operation: Callable[..., Any]
) -> float | None:
    """Return a statistic of one measure of one method, or None without data."""
    data = values(rows, method, key)
    return float(operation(data)) if data else None


def median(rows: Rows, method: str, key: str = "eta") -> float | None:
    return statistic(rows, method, key, np.median)


def largest(rows: Rows, method: str, key: str = "eta") -> float | None:
    return statistic(rows, method, key, max)


def p99(rows: Rows, method: str, key: str = "eta") -> float | None:
    return statistic(rows, method, key, lambda a: np.percentile(a, 99))


def joined(rows: Rows, methods: tuple[str, ...], operation: Callable[..., Any]) -> str:
    """Format one statistic for each method, separated by slashes."""
    return " / ".join(fmt(operation(rows, m)) for m in methods)


def at_least(rows: Rows, method: str) -> float:
    """Return the share of cases in which I is at least as accurate as a method."""
    return sum(r["comparisons"][method] != "worse" for r in rows) / len(rows)


def tied(rows: Rows, method: str) -> float:
    return sum(r["comparisons"][method] == "tied" for r in rows) / len(rows)


def wrong(rows: Rows, method: str) -> int:
    """Return the number of results that differ from the exact value by more than WRONG."""
    return sum(
        r["methods"][method]["absolute"] is None
        or r["methods"][method]["absolute"] > WRONG
        for r in rows
    )


def measured(document: dict[str, Any], methods: tuple[str, ...] = METHODS) -> Any:
    """Return those of some methods that a study measured, in their order."""
    samples = document["samples"]
    present = samples[0]["methods"] if samples else {}
    return tuple(m for m in methods if m in present)


def section(text: str) -> str | None:
    """Return the delimited accuracy section of a document, or None without markers."""
    if (
        text.count(BEGIN) != 1
        or text.count(END) != 1
        or text.index(BEGIN) > text.index(END)
    ):
        return None
    return text[text.index(BEGIN) + len(BEGIN) : text.index(END)].strip("\n")


def replace_section(text: str, tables: str) -> str | None:
    """Return a document with its accuracy section replaced, or None without markers."""
    if section(text) is None:
        return None
    head = text[: text.index(BEGIN) + len(BEGIN)]
    return head + "\n\n" + tables.strip("\n") + "\n\n" + text[text.index(END) :]


def select(document: dict[str, Any], families: tuple[str, ...], side: int | None = SIDE,
           bands: int | None = None) -> Rows:  # fmt: skip
    """Return the cases of some families, at one block size and band count."""
    return [
        r
        for r in document["samples"]
        if r["family"] in families
        and (side is None or r["side"] == side)
        and (bands is None or r["bands"] == bands)
    ]


def summarize(document: dict[str, Any]) -> Rows:
    """Return the statistics of every group, and of typical inputs pooled."""
    groups: dict[tuple[str, Any, int], Rows] = {}
    for row in document["samples"]:
        names = [row["family"]] + (["typical"] if row["family"] in TYPICAL else [])
        for name in names:
            groups.setdefault((name, row["bands"], row["side"]), []).append(row)
            groups.setdefault((name, "all", row["side"]), []).append(row)
    summaries = []
    for (family, bands, side), rows in groups.items():
        for method in measured(document):
            eta = values(rows, method, "eta")
            relative = values(rows, method, "relative")
            summaries.append(
                {
                    "family": family,
                    "bands": bands,
                    "dimension": rows[0]["dimension"] if bands != "all" else "",
                    "side": side,
                    "method": method,
                    "samples": len(rows),
                    "median_eta": float(np.median(eta)) if eta else None,
                    "mean_eta": float(np.mean(eta)) if eta else None,
                    "p95_eta": float(np.percentile(eta, 95)) if eta else None,
                    "p99_eta": float(np.percentile(eta, 99)) if eta else None,
                    "max_eta": max(eta) if eta else None,
                    "median_relative": float(np.median(relative)) if relative else None,
                    "max_relative": max(relative) if relative else None,
                    "nonfinite": len(rows) - len(eta),
                    "marked": sum(r["marked"] for r in rows),
                    "fraction_I_at_least_as_accurate": (
                        None if method == "I" else at_least(rows, method)
                    ),
                    "fraction_tied": None if method == "I" else tied(rows, method),
                }
            )
    families = ["typical", *document["corpus"]["families"]]

    def order(s: dict[str, Any]) -> tuple[Any, ...]:
        bands = -1 if s["bands"] == "all" else s["bands"]
        return (
            families.index(s["family"]),
            s["side"],
            bands,
            METHODS.index(s["method"]),
        )

    return sorted(summaries, key=order)


def render(document: dict[str, Any]) -> str:
    """Return the accuracy tables, without section markers."""
    if not document.get("completed"):
        raise ValueError("accuracy study did not finish; refusing a partial table")
    samples = document["samples"]
    side = max({r["side"] for r in samples}, key=lambda s: (s == SIDE, s))
    main, guarded = measured(document, MAIN), measured(document, GUARDED)
    fused = "F" in main
    names, others = " / ".join(main), " / ".join(main[1:])
    sums = measured(document, ("I", "F", "G"))
    sums_names = " / ".join(sums)
    guards = " / ".join(guarded)
    lines = [
        f"{len(samples):,} block pairs from the {document['profile']} profile. I is "
        + (
            "inazumaQ2n with its moments and F inazumaQ2n with the fused sums of "
            "fastCD; "
            if fused
            else "inazumaQ2n; "
        )
        + "Q and D evaluate the definition pixel by pixel with the products of "
        "pyfastcd's quasilinear and direct algorithms; "
        + ("G is F and " if "G" in sums else "")
        + "T is inazumaQ2n with the block "
        "merged from four tiles; U is inazumaQ2n without the evaluation by definition of "
        "the blocks its guards mark. B is the number of bands and N the dimension of the "
        "algebra. eta is the absolute error of the value of a block in units of "
        'u = 2^-53. "Typical" pools the uniform and Gaussian families. Empirical p99 '
        "values and maxima describe this corpus only.",
        "",
    ]
    typical = select(document, TYPICAL, side)
    band_counts = sorted({r["bands"] for r in typical})
    lines += [
        f"**Typical inputs, blocks of {side} by {side} pixels**",
        "",
        f"| B | N | blocks | median eta {names} | p99 eta {names} | max eta {names} "
        f"| I at least as accurate as {others} |",
        "|---:|---:|---:|:---|:---|:---|:---|",
    ]
    for bands in band_counts:
        rows = [r for r in typical if r["bands"] == bands]
        shares = " / ".join(f"{at_least(rows, m):.0%}" for m in main[1:])
        lines.append(
            f"| {bands} | {rows[0]['dimension']} | {len(rows)} | {joined(rows, main, median)} "
            f"| {joined(rows, main, p99)} | {joined(rows, main, largest)} "
            f"| {shares} |"
        )
    lines += [
        "",
        "**Relative error, typical inputs**",
        "",
        f"| B | median relative error {names} | max relative error {names} |",
        "|---:|:---|:---|",
    ]
    for bands in band_counts:
        rows = [r for r in typical if r["bands"] == bands]
        lines.append(
            f"| {bands} | {joined(rows, main, lambda r, m: median(r, m, 'relative'))} "
            f"| {joined(rows, main, lambda r, m: largest(r, m, 'relative'))} |"
        )

    sized = select(document, TYPICAL, None)
    pairs = sorted({(r["bands"], r["side"]) for r in sized})
    sizes = [(b, s) for b, s in pairs if len({t for c, t in pairs if c == b}) > 1]
    if sizes:
        lines += [
            "",
            "**Block size, typical inputs**",
            "",
            f"| B | block | blocks | median eta {names} | max eta {names} |",
            "|---:|---:|---:|:---|:---|",
        ]
        for bands, size in sizes:
            rows = [r for r in sized if r["bands"] == bands and r["side"] == size]
            lines.append(
                f"| {bands} | {size} x {size} | {len(rows)} | {joined(rows, main, median)} "
                f"| {joined(rows, main, largest)} |"
            )

    lines += [
        "",
        f"**Blocks merged from four tiles, typical inputs, blocks of {side} by {side}**",
        "",
        "| B | median eta I / T | max eta I / T | T identical to I |",
        "|---:|:---|:---|---:|",
    ]
    for bands in band_counts:
        rows = [r for r in typical if r["bands"] == bands]
        lines.append(
            f"| {bands} | {joined(rows, ('I', 'T'), median)} "
            f"| {joined(rows, ('I', 'T'), largest)} | {tied(rows, 'T'):.0%} |"
        )

    if fused:
        lines += [
            "",
            "**Fused sums of fastCD against the moments, typical inputs, blocks of "
            f"{side} by {side}**",
            "",
            f"| B | N | median eta {sums_names} | p99 eta {sums_names} "
            f"| max eta {sums_names} | F at least as accurate as I "
            "| F as accurate as I |",
            "|---:|---:|:---|:---|:---|---:|---:|",
        ]
        for bands in band_counts:
            rows = [r for r in typical if r["bands"] == bands]
            better = sum(r["comparisons"]["F"] != "better" for r in rows) / len(rows)
            lines.append(
                f"| {bands} | {rows[0]['dimension']} | {joined(rows, sums, median)} "
                f"| {joined(rows, sums, p99)} | {joined(rows, sums, largest)} "
                f"| {better:.0%} | {tied(rows, 'F'):.0%} |"
            )

    families = [f for f in document["corpus"]["families"] if f not in OFFSETS + HAZARDS]
    for bands in FAMILY_BANDS:
        rows_b = [r for r in samples if r["bands"] == bands and r["side"] == side]
        if not rows_b:
            continue
        lines += [
            "",
            f"**Families, B = {bands}**",
            "",
            f"| family | blocks | median eta {names} | max eta {names} | marked |",
            "|:---|---:|:---|:---|---:|",
        ]
        for family in families:
            rows = [r for r in rows_b if r["family"] == family]
            if rows:
                lines.append(
                    f"| {family} | {len(rows)} | {joined(rows, main, median)} "
                    f"| {joined(rows, main, largest)} | {sum(r['marked'] for r in rows)} |"
                )

    offsets = select(document, OFFSETS, side)
    if offsets:
        lines += [
            "",
            "**Means far from zero**, all band counts pooled",
            "",
            f"| family | blocks | median eta {guards} | max eta {guards} |",
            "|:---|---:|:---|:---|",
        ]
        for family in OFFSETS:
            rows = [r for r in offsets if r["family"] == family]
            lines.append(
                f"| {family} | {len(rows)} | {joined(rows, guarded, median)} "
                f"| {joined(rows, guarded, largest)} |"
            )

    hazards = select(document, HAZARDS, side)
    if hazards:
        lines += [
            "",
            "**Branches of the definition and extreme scales**, all band counts pooled",
            "",
            f"| family | blocks | median eta {guards} | results off by more than "
            f"{WRONG:g}, {guards} |",
            "|:---|---:|:---|:---|",
        ]
        for family in HAZARDS:
            rows = [r for r in hazards if r["family"] == family]
            counts = " / ".join(str(wrong(rows, m)) for m in guarded)
            lines.append(
                f"| {family} | {len(rows)} | {joined(rows, guarded, median)} | {counts} |"
            )
    lines += ["", FOOTER]
    return "\n".join(lines) + "\n"


def summary_csv(document: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summarize(document))
    return output.getvalue()


def key_figures(document: dict[str, Any]) -> dict[str, Any]:
    """Return the figures that the documentation quotes, computed from the results."""
    samples = document["samples"]
    side = max({r["side"] for r in samples}, key=lambda s: (s == SIDE, s))
    typical = select(document, TYPICAL, side)
    band_counts = sorted({r["bands"] for r in typical})
    per_bands = {b: [r for r in typical if r["bands"] == b] for b in band_counts}
    ordinary = [r for r in samples if r["family"] not in OFFSETS + HAZARDS]
    methods, guarded = measured(document), measured(document, GUARDED)
    compared = measured(document, ("F", "Q", "D"))
    ratios = {
        m: [
            median(rows, m) / median(rows, "I")
            for rows in per_bands.values()
            if median(rows, "I")
        ]
        for m in compared
    }
    figures: dict[str, Any] = {
        "profile": document["profile"],
        "cases": len(samples),
        "families": len(document["corpus"]["families"]),
        "bands": [min(r["bands"] for r in samples), max(r["bands"] for r in samples)],
        "typical_median_eta": {
            str(b): {m: median(rows, m) for m in methods}
            for b, rows in per_bands.items()
        },
        "typical_max_eta": {m: largest(typical, m) for m in methods},
        "typical_median_ratio_over_I": {
            m: [min(r), max(r)] if r else None for m, r in ratios.items()
        },
        "typical_I_at_least_as_accurate": {m: at_least(typical, m) for m in compared},
        "ordinary_max_eta": {m: largest(ordinary, m) for m in methods},
        "ordinary_I_at_least_as_accurate": {m: at_least(ordinary, m) for m in compared},
        "offsets_median_eta": {
            f: {m: median(select(document, (f,), side), m) for m in guarded}
            for f in OFFSETS
            if select(document, (f,), side)
        },
        "hazards_wrong": {
            f: {m: wrong(select(document, (f,), side), m) for m in guarded}
            for f in HAZARDS
            if select(document, (f,), side)
        },
        "hazards_cases": {
            f: len(select(document, (f,), side))
            for f in HAZARDS
            if select(document, (f,), side)
        },
        "nonfinite": {
            m: sum(r["methods"][m]["eta"] is None for r in samples) for m in methods
        },
        "marked": sum(r["marked"] for r in samples),
        "marked_ordinary": sum(r["marked"] for r in ordinary),
    }
    finite = [v for v in figures["typical_max_eta"].values() if v is not None]
    assert all(math.isfinite(v) for v in finite)
    return figures


def write(document: dict[str, Any], directory: Path) -> str:
    """Write the three report files into a study directory and return the tables."""
    tables = render(document)
    (directory / "accuracy-tables.md").write_text(tables)
    (directory / "accuracy-summary.csv").write_text(summary_csv(document))
    figures = json.dumps(key_figures(document), indent=2, allow_nan=False)
    (directory / "key-figures.json").write_text(figures + "\n")
    return tables


def update_document(path: Path, tables: str) -> bool:
    """Rewrite the accuracy section of a document; do nothing without the file or markers."""
    if not path.is_file():
        return False
    updated = replace_section(path.read_text(), tables)
    if updated is None:
        return False
    path.write_text(updated)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="stability.py JSON results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the tables, CSV summary and key figures",
    )
    parser.add_argument(
        "--update-docs", action="store_true", help="rewrite the accuracy section"
    )
    parser.add_argument("--document", type=Path, default=ROOT / "docs" / "stability.md")
    args = parser.parse_args(argv)
    tables = write(json.loads(args.results.read_text()), args.output_dir)
    if args.update_docs and not update_document(args.document, tables):
        print(f"{args.document} has no accuracy section; left unchanged")


if __name__ == "__main__":
    main()

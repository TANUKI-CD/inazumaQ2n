#!/usr/bin/env python3
"""Run the stability study end to end and write a self-describing study directory.

The study measures the accuracy of the value of a Q2n block against an exact
reference (stability.py), tabulates the results (stability_report.py) and
draws the charts (stability_plots.py). Published files sit at the top of the
study directory and are distributed with the repository; the raw results sit
under raw/ and stay on the machine that ran the study. SHA256SUMS lists the
published files, and study.json records the hashes of the raw files and of
the measured sources.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stability  # noqa: E402
import stability_plots  # noqa: E402
import stability_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = "raw"
PUBLISHED = {
    "README.md": "This description, generated from the results.",
    "SHA256SUMS": "SHA-256 of every published file of the study.",
    "study.json": "The command, the host and software, the checks of the reference, "
    "the hashes of the measured sources and of the raw files, and every step with its "
    "duration.",
    "key-figures.json": "The figures quoted in docs/stability.md, computed from the results.",
    "accuracy-tables.md": "The accuracy tables shown in docs/stability.md.",
    "accuracy-summary.csv": "Error statistics for every family, band count, block size and "
    "method.",
    "PLOTS.md": "Description of the charts and the command that drew them.",
    "plot-data.csv": "Every plotted value in tabular form.",
}
STEPS = {
    "accuracy": "Measure every error against the exact value.",
    "tables": "Tabulate the accuracy results.",
    "plots": "Draw the charts.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    """Return the SHA-256 of every file of the measured package and comparisons."""
    files = [
        p
        for p in (root / "src" / "inazumaq2n").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".so"
    ]
    files += [root / "benchmarks" / "baseline.py", root / "tests" / "exact.py"]
    # The record of the fastCD sources that the package was built from
    files += [p for p in [root / "vendor" / "fastcd" / "VENDORED.json"] if p.is_file()]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}


def published_files(directory: Path) -> list[Path]:
    """Return the files of a study directory that are distributed."""
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS"
        and RAW not in p.relative_to(directory).parts
    )


def write_checksums(directory: Path) -> None:
    lines = [
        f"{sha256(p)}  {p.relative_to(directory).as_posix()}"
        for p in published_files(directory)
    ]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def command_line(args: argparse.Namespace, name: str) -> str:
    """Return the command that reproduces the study, relative to the repository root."""
    parts = ["python", "benchmarks/stability_study.py", "--profile", args.profile]
    parts += ["--seed", str(args.seed), "--jobs", str(args.jobs)]
    if args.digits != stability.parse_arguments([]).digits:
        parts += ["--digits", str(args.digits)]
    parts += ["--output", f"benchmark-results/{name}"]
    if args.update_docs:
        parts.append("--update-docs")
    return " ".join(parts)


def readme(study: dict[str, Any], figures: dict[str, Any], files: list[str]) -> str:
    """Return the description of a study directory."""
    environment = study["environment"]
    runtime = environment["runtime"]
    lines = [
        f"# Stability study, {study['date']}",
        "",
        "This directory holds the measurements behind "
        "[docs/stability.md](../../docs/stability.md), which defines the error measures "
        "and interprets the results.",
        "",
        "## What was measured",
        "",
        "The floating-point accuracy of the value of one block of a Q2n index, against "
        "the exact value of the definition for the stored samples.",
        "",
        "- **I, inazumaQ2n:** the block evaluated from its centered moments, as `q2n` "
        "and `q2n_map` evaluate blocks that do not overlap.",
    ]
    if "F" in figures["nonfinite"]:
        lines += [
            "- **F, inazumaQ2n with fastCD:** the block evaluated by one fused sum of "
            'products of fastCD, as with `algorithm="fastcd"`, behind the same guards '
            "as I.",
            "- **G, inazumaQ2n with fastCD from tiles:** F with the records and the "
            "statistics of the block merged from four tiles, as for blocks that overlap "
            "with a stride of half the block size.",
        ]
    lines += [
        "- **T, inazumaQ2n from tiles:** the block merged from four tiles, as for blocks "
        "that overlap with a stride of half the block size.",
        "- **U, inazumaQ2n without its guards:** I without the evaluation by definition "
        "of the blocks that the guards mark.",
        "- **Q and D, the definition pixel by pixel:** every normalized reference pixel "
        "multiplied by the conjugate of the normalized test pixel, with the products of "
        "pyfastcd's quasilinear (Q) and direct (D) algorithms, in "
        "`benchmarks/baseline.py`.",
        "",
        f"The {study['profile']} profile measured {figures['cases']:,} block pairs from "
        f"{figures['families']} families. The reference converts every binary64 sample to "
        "an integer times a power of two, forms the moments of the block in "
        "arbitrary-precision integers, decides every branch of the definition on them, "
        f"and evaluates square roots and quotients to {study['digits']} significant "
        "digits.",
        "",
        "## Host and software",
        "",
        "| | |",
        "|:---|:---|",
        f"| CPU | {environment['cpu_model']} |",
        f"| Operating system | {environment['platform']} |",
        f"| Python | {environment['python'].split()[0]} |",
        f"| NumPy | {environment['numpy']} |",
        f"| inazumaQ2n | {environment['inazumaq2n']}, kernels {runtime['simd']}, "
        "matrix products by NumPy |",
    ]
    if "fastcd" in runtime:
        lines += [
            f"| fastCD | {runtime['fastcd']}, compiled into inazumaQ2n, fused sums by "
            f"its {runtime['fastcd_kernel']} kernel |",
        ]
    lines += [
        f"| pyfastcd | {environment['pyfastcd']} |",
        f"| Matplotlib | {environment['matplotlib']} |",
        f"| Worker processes | {study['jobs']} |",
        "",
        "## How it was produced",
        "",
        "```sh",
        study["command"],
        "```",
        "",
        "| Step | What it does | Duration |",
        "|:---|:---|---:|",
    ]
    lines += [f"| {n} | {STEPS[n]} | {s:.0f} s |" for n, s in study["steps"].items()]
    lines += [
        "",
        f"The samples derive from the root seed {study['seed']}. study.json records the "
        "SHA-256 of every file of the measured package, of the comparison and of the "
        "reference.",
        "",
        "## Files",
        "",
        "| File | Contents |",
        "|:---|:---|",
    ]
    lines += [f"| {name} | {text} |" for name, text in PUBLISHED.items()]
    charts = sorted({name.rsplit(".", 1)[0] for name in files if name.endswith(".png")})
    lines.append(f"| {', '.join(charts)} | The charts, each as PNG, SVG and PDF. |")
    lines += [
        "",
        f"The raw per-case results are written to `{RAW}/`. They are not distributed; "
        "study.json records their SHA-256, and `tools/check_archives.py --full` checks "
        "them where they are present. Every case records its seed and the SHA-256 of its "
        "samples, from which the samples can be generated again.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    """Run every step of the study and return the study directory."""
    import matplotlib

    directory = args.output.resolve()
    raw = directory / RAW
    raw.mkdir(parents=True, exist_ok=True)
    steps: dict[str, float] = {}

    start = time.perf_counter()
    arguments = ["--output", str(raw / "accuracy.json"), "--profile", args.profile]
    arguments += ["--seed", str(args.seed), "--jobs", str(args.jobs)]
    arguments += ["--digits", str(args.digits)]
    document = stability.run(stability.parse_arguments(arguments))
    steps["accuracy"] = time.perf_counter() - start

    start = time.perf_counter()
    tables = stability_report.write(document, directory)
    figures = stability_report.key_figures(document)
    if args.update_docs and not stability_report.update_document(args.document, tables):
        print(f"{args.document} has no accuracy section; left unchanged")
    steps["tables"] = time.perf_counter() - start

    start = time.perf_counter()
    stability_plots.main([str(directory)])
    steps["plots"] = time.perf_counter() - start

    study = {
        "schema": "inazumaq2n-stability-study-v1",
        "date": args.date,
        "profile": document["profile"],
        "seed": document["seed"],
        "digits": document["digits"],
        "jobs": document["jobs"],
        "command": command_line(args, directory.name),
        "environment": {
            **document["environment"],
            "matplotlib": matplotlib.__version__,
        },
        "oracle_validation": document["oracle_validation"],
        "precision_checks": document["precision_checks"],
        "corpus": document["corpus"],
        "steps": steps,
        "source": {"files_sha256": source_hashes(ROOT)},
        "tools_sha256": {
            p.name: sha256(p)
            for p in sorted(Path(__file__).parent.glob("stability*.py"))
        },
        "raw_sha256": {
            p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(raw.iterdir())
            if p.is_file()
        },
    }
    (directory / "study.json").write_text(
        json.dumps(study, indent=2, allow_nan=False) + "\n"
    )
    names = [p.relative_to(directory).as_posix() for p in published_files(directory)]
    (directory / "README.md").write_text(readme(study, figures, names))
    write_checksums(directory)
    return directory


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    today = datetime.date.today()
    defaults = stability.parse_arguments([])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / f"stability-{today:%Y%m%d}",
        help="study directory",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--jobs", type=int, default=defaults.jobs, help="worker processes"
    )
    parser.add_argument("--digits", type=int, default=defaults.digits)
    parser.add_argument(
        "--update-docs", action="store_true", help="rewrite the accuracy section"
    )
    parser.add_argument("--document", type=Path, default=ROOT / "docs" / "stability.md")
    args = parser.parse_args(argv)
    args.date = today.isoformat()
    return args


if __name__ == "__main__":
    print(run(parse_arguments(sys.argv[1:])))

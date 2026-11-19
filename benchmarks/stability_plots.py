#!/usr/bin/env python3
"""Draw the accuracy charts of a stability study from its CSV summary.

The script draws three charts, each as PNG, SVG and PDF: the median error eta
and the median relative error of inazumaQ2n (I) and of the evaluations of the
definition pixel by pixel with pyfastcd's quasilinear (Q) and direct (D)
products on typical inputs, against the number of bands; and the median eta
of these and of inazumaQ2n without its guards (U) on images whose mean is far
from zero. Every plotted value is also written to plot-data.csv, and PLOTS.md
lists the charts. Needs Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

FORMATS = ("png", "svg", "pdf")
# The block size of the charts: 32 by 32 when the study has it, else the largest
SIDE = 32
OFFSETS = {"offset_1e3": 1e3, "offset_1e6": 1e6, "offset_1e9": 1e9, "offset_1e12": 1e12}

# One colour for each method, kept across every chart
COLORS = {"I": "#2a78d6", "Q": "#eb6834", "D": "#1baf7a", "U": "#8a5bd0"}
MARKERS = {"I": "o", "Q": "s", "D": "^", "U": "D"}
NAMES = {
    "I": "I, inazumaQ2n",
    "Q": "Q, per pixel, pyfastcd quasilinear",
    "D": "D, per pixel, pyfastcd direct",
    "U": "U, inazumaQ2n without its guards",
}
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

FIELDS = ("figure", "series", "x", "value", "samples")


def read_summary(path: Path) -> list[dict[str, Any]]:
    """Read the CSV summary, converting its numeric columns."""
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["side"] = int(row["side"])
            row["samples"] = int(row["samples"])
            for key in ("median_eta", "median_relative"):
                row[key] = float(row[key]) if row[key] else None
            rows.append(row)
    return rows


def style(
    axis: Any, ticks: list[float], labels: list[str], linear_top: float | None
) -> None:
    """Apply the shared axis style: recessive grid and frame, logarithmic x axis.

    With *linear_top* the value axis is linear from zero to a little above it;
    without it the axis is logarithmic.
    """
    axis.set_facecolor(SURFACE)
    axis.set_xscale("log", base=2 if ticks and ticks[-1] < 1000 else 10)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if linear_top is None:
        axis.set_yscale("log")
    else:
        axis.set_ylim(0, 1.15 * linear_top if linear_top > 0 else 1)
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(which="both", colors=MUTED, labelsize=9)


def draw(axis: Any, lines: dict[str, list[tuple[float, float, int]]]) -> None:
    """Draw each method as a line with markers and label it at its last point."""
    for method, points in lines.items():
        if not points:
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        axis.plot(xs, ys, color=COLORS[method], marker=MARKERS[method], linewidth=2,
                  markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
                  label=NAMES[method])  # fmt: skip
        axis.annotate(method, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                      color=INK, fontsize=10, va="center")  # fmt: skip


def save(figure: Any, directory: Path, name: str) -> None:
    # Dates are left out of the vector files so that redrawing reproduces them
    metadata = {"png": None, "svg": {"Date": None}, "pdf": {"CreationDate": None}}
    for extension in FORMATS:
        figure.savefig(directory / f"{name}.{extension}", dpi=160, facecolor=SURFACE,
                       metadata=metadata[extension])  # fmt: skip
    plt.close(figure)


def band_chart(rows: list[dict[str, Any]], directory: Path, key: str, name: str,
               ylabel: str, data: list[dict[str, Any]]) -> bool:  # fmt: skip
    """Draw I, Q and D on typical inputs against the number of bands."""
    lines = {}
    for method in ("I", "Q", "D"):
        lines[method] = sorted(
            (float(r["bands"]), r[key], r["samples"])
            for r in rows
            if r["family"] == "typical" and r["side"] == SIDE and r["bands"] != "all"
            and r["method"] == method and r[key] is not None
        )  # fmt: skip
    if not any(lines.values()):
        return False
    bands = sorted({p[0] for points in lines.values() for p in points})
    log = key == "median_relative" and all(p[1] > 0 for v in lines.values() for p in v)
    top = None if log else max(p[1] for v in lines.values() for p in v)
    figure, axis = plt.subplots(
        figsize=(7.2, 4.4), facecolor=SURFACE, layout="constrained"
    )
    draw(axis, lines)
    style(axis, bands, [str(int(b)) for b in bands], top)
    axis.set_xlabel("B, number of bands", color=SECONDARY, fontsize=10)
    axis.set_ylabel(ylabel, color=SECONDARY, fontsize=10)
    axis.set_title(f"{ylabel}, typical inputs, blocks of {SIDE} by {SIDE}", color=INK,
                   fontsize=11, loc="left")  # fmt: skip
    axis.legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="upper left")
    for method, points in lines.items():
        data.extend({"figure": name, "series": method, "x": int(x), "value": repr(v),
                     "samples": n} for x, v, n in points)  # fmt: skip
    save(figure, directory, name)
    return True


def offset_chart(rows: list[dict[str, Any]], directory: Path, name: str,
                 data: list[dict[str, Any]]) -> bool:  # fmt: skip
    """Draw I, U, Q and D on images whose mean is far from zero, against the offset."""
    lines = {}
    for method in ("I", "U", "Q", "D"):
        lines[method] = sorted(
            (OFFSETS[r["family"]], r["median_eta"], r["samples"])
            for r in rows
            if r["family"] in OFFSETS and r["bands"] == "all" and r["side"] == SIDE
            and r["method"] == method and r["median_eta"]
        )  # fmt: skip
    if not any(lines.values()):
        return False
    offsets = sorted({p[0] for points in lines.values() for p in points})
    figure, axis = plt.subplots(
        figsize=(7.2, 4.4), facecolor=SURFACE, layout="constrained"
    )
    draw(axis, lines)
    style(axis, offsets, [f"1e{round(len(str(int(o))) - 1)}" for o in offsets], None)
    axis.set_xlabel("offset of the samples; their variation is about 0.3", color=SECONDARY,
                    fontsize=10)  # fmt: skip
    axis.set_ylabel("Median eta", color=SECONDARY, fontsize=10)
    axis.set_title("Median eta, images whose mean is far from zero", color=INK, fontsize=11,
                   loc="left")  # fmt: skip
    axis.legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="upper left")
    for method, points in lines.items():
        data.extend({"figure": name, "series": method, "x": x, "value": repr(v),
                     "samples": n} for x, v, n in points)  # fmt: skip
    save(figure, directory, name)
    return True


def plots_document(figures: list[tuple[str, str]], command: str) -> str:
    """Return the Markdown description of the charts."""
    lines = [
        "# Accuracy charts",
        "",
        "The charts show the accuracy of the value of a Q2n block, measured against the "
        "exact value of the definition. I is inazumaQ2n, Q and D evaluate the definition "
        "pixel by pixel with the products of pyfastcd's quasilinear and direct "
        "algorithms, and U is inazumaQ2n without the evaluation by definition of the "
        "blocks its guards mark.",
        "",
        "| Chart | PNG | SVG | PDF |",
        "|:---|:---|:---|:---|",
    ]
    lines += [
        f"| {t} | [PNG]({n}.png) | [SVG]({n}.svg) | [PDF]({n}.pdf) |"
        for n, t in figures
    ]
    lines += [
        "",
        "## Values",
        "",
        "Each point of the band charts is the median over the uniform and Gaussian block "
        f"pairs of {SIDE} by {SIDE} pixels with one number of bands; each point of the "
        "offset chart is the median over every number of bands of one offset family. The "
        "error measures are defined in docs/stability.md. Every plotted value, with the "
        "number of samples behind it, is in plot-data.csv.",
        "",
        "## Axes",
        "",
        "The number of bands is on a base-two logarithmic axis and the offset on a "
        "base-ten one. The eta axis of the band chart is linear and starts at zero; the "
        "other value axes are logarithmic. Medians do not describe worst-case errors; the "
        "maxima are in accuracy-tables.md.",
        "",
        "## Reproduction",
        "",
        "```sh",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    global SIDE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "directory", type=Path, help="study directory with accuracy-summary.csv"
    )
    args = parser.parse_args(argv)
    rows = read_summary(args.directory / "accuracy-summary.csv")
    SIDE = max({r["side"] for r in rows}, key=lambda s: (s == 32, s))
    data: list[dict[str, Any]] = []
    figures = []
    charts = (
        ("eta-bands", "Median eta against the number of bands",
         lambda n: band_chart(rows, args.directory, "median_eta", n, "Median eta", data)),
        ("relative-error-bands", "Median relative error against the number of bands",
         lambda n: band_chart(rows, args.directory, "median_relative", n,
                              "Median relative error", data)),
        ("eta-offset", "Median eta against the offset of the samples",
         lambda n: offset_chart(rows, args.directory, n, data)),
    )  # fmt: skip
    for name, title, chart in charts:
        if chart(name):
            figures.append((name, title))
    with (args.directory / "plot-data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    command = (
        f"python benchmarks/stability_plots.py benchmark-results/{args.directory.name}"
    )
    (args.directory / "PLOTS.md").write_text(plots_document(figures, command))


if __name__ == "__main__":
    main()

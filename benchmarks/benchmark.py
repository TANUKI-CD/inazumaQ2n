"""Timings of complete Q2n calls.

Three protocols are provided. ``calls`` times :func:`inazumaq2n.q2n` and a
reused :class:`inazumaq2n.Q2nPlan` on a fixed set of images, beside the
evaluation of the definition pixel by pixel with the products of pyfastcd,
with its quasilinear and with its direct algorithm (``baseline.py``).
``kernels`` times, for images with at most four bands, the kernels that need
no matrix product against the general route through a matrix product, in
every instruction set the processor supports. ``algorithms`` times the two
algorithms of the package, the real cross moments and the fused sums of
fastCD, against each other and against the recursive product of the
reference toolboxes (``toolbox.py``), over sweeps of the dimension, the band
count, the overlap, the sample type and the block size; it is the
measurement behind ``algorithm="auto"``. ``toolboxes`` times the package
beside the code of the published toolboxes themselves, the MATLAB toolbox
under GNU Octave and its Python ports (``toolboxes.py``), and records how far
their values are from those of the package.

Every measurement is a complete call on images that already exist: the
conversion of the input, the allocation of work arrays, the moments, the
evaluation of the blocks and the mean are inside the timed region. The
variants of a case are run in rotating order within each round, so that a
change in the load of the machine affects all of them, and the report gives
the median over the rounds with every observation kept in the output file.

Run ``python benchmarks/benchmark.py --help`` for the options, and see
``docs/benchmark.md`` for the protocol and the archived results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import inazumaq2n
from inazumaq2n import Q2nPlan, _core, _plan, q2n, runtime_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

# The comparisons: the definition evaluated pixel by pixel with pyfastcd
BASELINES = {"fastCD quasilinear": "quasilinear", "fastCD direct": "direct"}

# name, shape, block size, stride, sample type
CALL_CASES: list[tuple[str, tuple[int, ...], int, int, str]] = [
    ("gray 512x512, uint8", (512, 512), 32, 32, "uint8"),
    ("RGB 512x512, uint8", (512, 512, 3), 32, 32, "uint8"),
    ("RGB 512x512, uint8, block 64, stride 32", (512, 512, 3), 64, 32, "uint8"),
    ("4 bands 512x512, uint16", (512, 512, 4), 32, 32, "uint16"),
    ("8 bands 512x512, uint16", (512, 512, 8), 32, 32, "uint16"),
    ("16 bands 512x512, float64", (512, 512, 16), 32, 32, "float64"),
    (
        "16 bands 512x512, float64, block 64, stride 32",
        (512, 512, 16),
        64,
        32,
        "float64",
    ),
    ("32 bands 256x256, float64", (256, 256, 32), 32, 32, "float64"),
    (
        "32 bands 256x256, float64, block 64, stride 32",
        (256, 256, 32),
        64,
        32,
        "float64",
    ),
    ("159 bands 200x200, float64", (200, 200, 159), 32, 32, "float64"),
    (
        "159 bands 200x200, float64, block 64, stride 32",
        (200, 200, 159),
        64,
        32,
        "float64",
    ),
]

# The comparison of the algorithms protocol: the recursion of the toolboxes
TOOLBOX = "toolbox"

# The sweeps of the algorithms protocol
SWEEPS = ("dimension", "bands", "overlap", "types", "blocks", "workers")

# A toolbox call longer than this is timed from its first call, without a warm-up
SLOW_SECONDS = 2.0

# The largest difference allowed between the plans, and against the toolbox
PLAN_TOLERANCE = 1e-12
TOOLBOX_TOLERANCE = 1e-10


def algorithm_side(dimension: int) -> int:
    """Return the side of the square image used for an algebra of a dimension.

    The images of every dimension exceed the second-level cache of a core,
    as real images do, and hold between 2 and 34 MiB of float64 samples.
    """
    if dimension <= 16:
        return 512
    return 256 if dimension <= 64 else 128 if dimension <= 256 else 64


def algorithm_cases(
    sweeps: tuple[str, ...] = SWEEPS, max_dimension: int = 1024
) -> list[dict[str, Any]]:
    """Return the cases of the algorithms protocol.

    ``dimension`` sweeps the dimension ``N`` with ``B = N`` bands, where both
    algorithms do their full work. ``bands`` sweeps the band count within one
    dimension, where the moments do less and the fused sums do not.
    ``overlap`` sweeps ``N`` with blocks merged from 4 and from 16 tiles,
    which the moments share and the fused sums repeat, and the band count
    below the larger dimensions with 4 tiles. ``types`` reads
    narrower samples, ``blocks`` changes the block size, and ``workers`` the
    number of threads. Only the first sweep includes the toolbox by default.
    """
    dimensions = [
        n for n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= max_dimension
    ]
    cases: list[dict[str, Any]] = []

    def add(
        sweep: str, bands: int, ws: int = 32, stride: int = 32, **options: Any
    ) -> None:
        dimension = 1 << (bands - 1).bit_length()
        side = options.pop("side", algorithm_side(dimension))
        cases.append(
            {
                "sweep": sweep,
                "bands": bands,
                "dimension": dimension,
                "shape": (side, side, bands),
                "ws": ws,
                "stride": stride,
                "sample_type": options.pop("sample_type", "float64"),
                "workers": options.pop("workers", None),
                "toolbox": options.pop("toolbox", False),
            }
        )

    if "dimension" in sweeps:
        for n in dimensions:
            add("dimension", n, toolbox=True)
    if "bands" in sweeps:
        for n in dimensions:
            if n >= 8:
                for bands in sorted({n // 2 + 1, 5 * n // 8, 6 * n // 8, 7 * n // 8}):
                    add("bands", bands)
    if "overlap" in sweeps:
        for n in dimensions:
            if n >= 8:
                add("overlap", n, 32, 16)
        for n in dimensions:
            if n >= 64:
                add("overlap", n, 64, 16)
        for n in dimensions:
            if n >= 128:
                for bands in (5 * n // 8, 6 * n // 8, 7 * n // 8):
                    add("overlap", bands, 32, 16)
    if "types" in sweeps:
        for sample_type in ("float32", "uint16", "uint8"):
            for n in (8, 32, 128, 512):
                if n <= max_dimension:
                    add("types", n, sample_type=sample_type)
    if "blocks" in sweeps:
        for ws in (8, 16, 64):
            for n in (8, 32, 128, 512):
                if n <= max_dimension:
                    add("blocks", n, ws, ws, side=max(algorithm_side(n), 2 * ws))
    if "workers" in sweeps:
        for workers in (1, 2, 4, 8):
            for n in (32, 256):
                if n <= max_dimension:
                    add("workers", n, workers=workers, side=2 * algorithm_side(n))
    return cases


# bands, height, width, block size, stride, sample type
KERNEL_CASES: list[tuple[int, int, int, int, int, str]] = [
    (1, 512, 512, 32, 32, "uint8"),
    (1, 193, 207, 24, 12, "float64"),
    (2, 384, 448, 32, 32, "float64"),
    (2, 193, 207, 24, 12, "uint8"),
    (2, 512, 512, 64, 32, "float32"),
    (3, 512, 512, 32, 32, "uint8"),
    (3, 95, 117, 32, 12, "float64"),
    (3, 384, 448, 64, 64, "float32"),
    (4, 512, 512, 32, 32, "uint16"),
    (4, 512, 512, 32, 32, "float64"),
    (4, 384, 448, 64, 64, "uint8"),
    (4, 193, 207, 24, 12, "float32"),
    (4, 95, 117, 32, 12, "float64"),
    (4, 1024, 1024, 32, 32, "uint16"),
    (4, 256, 320, 16, 8, "uint8"),
    (4, 300, 300, 8, 8, "float64"),
]


def synthetic_pair(
    rng: np.random.Generator, shape: tuple[int, ...], dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return a uniform reference image and a copy with Gaussian noise."""
    x = rng.uniform(0, 255, shape)
    y = x + rng.normal(0, 20, shape)
    if np.dtype(dtype).kind == "u":
        x, y = np.clip(x, 0, 255), np.clip(y, 0, 255)
    return x.astype(dtype), y.astype(dtype)


def measure(
    variants: dict[str, Callable[[], float]],
    rounds: int,
    limits: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Return the value and the timings in seconds of every variant.

    Each variant is called once untimed, and then once per round in an
    order that rotates from round to round. A variant named in ``limits`` is
    timed in that many rounds at most, and when its first call takes longer
    than ``SLOW_SECONDS`` that call is its first observation: a call of
    seconds has nothing left to warm up.
    """
    names = list(variants)
    limits = limits or {}
    values = {}
    samples: dict[str, list[float]] = {name: [] for name in names}
    for name in names:
        start = time.perf_counter()
        values[name] = float(variants[name]())
        elapsed = time.perf_counter() - start
        if name in limits and elapsed > SLOW_SECONDS:
            samples[name].append(elapsed)
    for index in range(rounds):
        shift = index % len(names)
        for name in names[shift:] + names[:shift]:
            if len(samples[name]) >= limits.get(name, rounds):
                continue
            start = time.perf_counter()
            variants[name]()
            samples[name].append(time.perf_counter() - start)
    return values, samples


def summarize(
    values: dict[str, float], samples: dict[str, list[float]], reference: str
) -> dict[str, Any]:
    """Return the medians, and the ratios and paired wins against one variant."""
    medians = {name: float(np.median(times)) for name, times in samples.items()}
    return {
        "values": values,
        "max_abs_difference": max(values.values()) - min(values.values()),
        "median_seconds": medians,
        "seconds": samples,
        "ratio_to_" + reference: {
            name: medians[name] / medians[reference] for name in medians
        },
        "rounds_slower_than_" + reference: {
            name: sum(a < b for a, b in zip(samples[reference], times, strict=True))
            for name, times in samples.items()
            if name != reference
        },
    }


def cpu_model() -> str:
    """Return the name of the processor, where the system reports one."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict[str, Any]:
    """Return the software, hardware and sources of this run."""
    package = Path(inazumaq2n.__file__).resolve().parent
    names = ("*.py", "*.c", "*.h")
    sources = sorted(path for pattern in names for path in package.glob(pattern))
    try:
        from threadpoolctl import threadpool_info

        # The paths of the libraries are local to the machine and are left out
        pools: Any = [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in threadpool_info()
        ]
    except ImportError:
        pools = None
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "runtime": runtime_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "affinity": affinity,
        "thread_pools": pools,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            + [
                Path(__file__).resolve().with_name(name)
                for name in ("benchmark.py", "toolbox.py", "toolboxes.py")
            ]
        },
    }


def run_calls(
    rounds: int,
    backend: str = "auto",
    workers: int = 1,
    baselines: bool = True,
    selected: list[str] | None = None,
    cases: list[tuple[str, tuple[int, ...], int, int, str]] | None = None,
) -> dict[str, Any]:
    """Time one-shot calls and reused plans on every case, beside the baselines."""
    if baselines:
        import baseline
        import pyfastcd
    rng = np.random.default_rng(663819)
    rows = []
    for name, shape, ws, stride, dtype in CALL_CASES if cases is None else cases:
        if selected and not any(text in name for text in selected):
            continue
        x, y = synthetic_pair(rng, shape, dtype)
        options: dict[str, Any] = {"backend": backend, "workers": workers}
        profile: dict[str, Any] = {}
        with Q2nPlan(x.shape, ws, stride, **options) as plan:
            variants: dict[str, Callable[[], float]] = {
                "one-shot": lambda: q2n(x, y, ws, stride, **options),  # noqa: B023
                "plan": lambda: plan.score(x, y),  # noqa: B023
            }
            other = None
            if (
                plan.backend == "mkl"
                and plan.algorithm == "moments"
                and not plan._direct
            ):
                # The same plan with the library of NumPy, in the same rounds
                other = Q2nPlan(x.shape, ws, stride, backend="numpy", workers=workers)
                variants["plan, numpy"] = lambda: other.score(x, y)  # noqa: B023
            for label, algorithm in BASELINES.items() if baselines else ():
                variants[label] = partial(
                    baseline.q2n, x, y, ws, stride, algorithm=algorithm
                )
            values, samples = measure(variants, rounds)
            plan._run(x, y, profile)
            if other is not None:
                other.close()
        row = {
            "case": name,
            "shape": list(x.shape),
            "ws": ws,
            "stride": stride,
            "sample_type": str(x.dtype),
            "profile": profile,
        }
        row.update(summarize(values, samples, "plan"))
        if row["max_abs_difference"] > 1e-12:
            raise AssertionError(f"{name}: the variants disagree, {values}")
        rows.append(row)
        report = ", ".join(
            f"{key} {1e3 * value:.3f} ms"
            for key, value in row["median_seconds"].items()
        )
        print(f"{name}: {report}", flush=True)
    return {
        "protocol": "calls",
        "scope": "complete calls; plan construction is excluded for 'plan' only",
        "rounds": rounds,
        "backend": backend,
        "workers": workers,
        "baselines": (
            {"pyfastcd": pyfastcd.__version__, "algorithms": BASELINES}
            if baselines
            else None
        ),
        "rows": rows,
    }


def general_plan(shape: tuple[int, ...], ws: int, stride: int, backend: str) -> Q2nPlan:
    """Return a plan that takes the matrix-product route for any band count."""
    saved = _plan._DIRECT_BANDS
    _plan._DIRECT_BANDS = 0
    try:
        return Q2nPlan(shape, ws, stride, backend=backend, algorithm="moments")
    finally:
        _plan._DIRECT_BANDS = saved


def run_algorithms(
    rounds: int,
    backend: str = "auto",
    workers: int = 1,
    toolbox: bool = True,
    toolbox_rounds: int = 3,
    sweeps: tuple[str, ...] = SWEEPS,
    max_dimension: int = 1024,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Time the moments, the fused sums of fastCD and the toolbox on every case.

    The package is timed through reused plans, one for each algorithm and
    one for ``algorithm="auto"``; with MKL, the moments are also timed with
    the library of NumPy. The toolbox is a complete call of ``toolbox.q2n``
    and is timed in ``toolbox_rounds`` rounds at most.
    """
    if toolbox:
        import toolbox as reference_toolbox
    rng = np.random.default_rng(271828)
    rows = []
    for case in algorithm_cases(sweeps, max_dimension) if cases is None else cases:
        shape, ws, stride = tuple(case["shape"]), case["ws"], case["stride"]
        x, y = synthetic_pair(rng, shape, case["sample_type"])
        threads = case["workers"] or workers
        plans = {
            "moments": Q2nPlan(
                shape, ws, stride, backend=backend, workers=threads, algorithm="moments"
            ),
            "fastcd": Q2nPlan(shape, ws, stride, workers=threads, algorithm="fastcd"),
            "auto": Q2nPlan(shape, ws, stride, backend=backend, workers=threads),
        }
        if plans["moments"].backend == "mkl" and not plans["moments"]._direct:
            plans["moments, numpy"] = Q2nPlan(
                shape, ws, stride, backend="numpy", workers=threads, algorithm="moments"
            )
        variants: dict[str, Callable[[], float]] = {
            name: partial(plan.score, x, y) for name, plan in plans.items()
        }
        limits = {}
        if toolbox and case["toolbox"]:
            variants[TOOLBOX] = partial(reference_toolbox.q2n, x, y, ws, stride)
            limits[TOOLBOX] = toolbox_rounds
        values, samples = measure(variants, rounds, limits)
        chosen = plans["auto"].algorithm
        direct = plans["moments"]._direct
        tiles = plans["moments"]._ids.shape[1]
        blocks = int(np.prod(plans["auto"].block_shape))
        for plan in plans.values():
            plan.close()
        medians = {name: float(np.median(times)) for name, times in samples.items()}
        inside = [values[name] for name in plans]
        own = {name: medians[name] for name in ("moments", "fastcd")}
        fastest = min(own, key=lambda name: own[name])
        row = {
            **{
                key: case[key]
                for key in ("sweep", "bands", "dimension", "ws", "stride")
            },
            "shape": list(shape),
            "sample_type": case["sample_type"],
            "workers": threads,
            "blocks": blocks,
            "tiles_per_block": tiles,
            "moments_kernel": "direct" if direct else "matrix product",
            "values": values,
            "max_abs_difference": max(inside) - min(inside),
            "toolbox_abs_difference": (
                max(abs(values[TOOLBOX] - value) for value in inside)
                if TOOLBOX in values
                else None
            ),
            "median_seconds": medians,
            "seconds": samples,
            "nanoseconds_per_pixel": {
                name: 1e9 * median / (blocks * ws * ws)
                for name, median in medians.items()
            },
            "moments_over_fastcd": medians["moments"] / medians["fastcd"],
            "toolbox_over": (
                {name: medians[TOOLBOX] / medians[name] for name in own}
                if TOOLBOX in medians
                else None
            ),
            "rounds_fastcd_faster": sum(
                a < b
                for a, b in zip(samples["fastcd"], samples["moments"], strict=True)
            ),
            "auto": chosen,
            "fastest": fastest,
            # The time of the algorithm auto selects over that of the faster one
            "auto_regret": own[chosen] / own[fastest],
        }
        if row["max_abs_difference"] > PLAN_TOLERANCE:
            raise AssertionError(f"{case}: the algorithms disagree, {values}")
        if TOOLBOX in values and row["toolbox_abs_difference"] > TOOLBOX_TOLERANCE:
            raise AssertionError(f"{case}: the toolbox disagrees, {values}")
        rows.append(row)
        report = ", ".join(
            f"{key} {1e3 * value:.3f} ms" for key, value in medians.items()
        )
        print(
            f"{case['sweep']}: {case['bands']} bands (N = {case['dimension']}) "
            f"{shape[0]}x{shape[1]} {case['sample_type']} {ws}/{stride} "
            f"workers {threads}: {report}; auto {chosen}, fastest {fastest}",
            flush=True,
        )
    return {
        "protocol": "algorithms",
        "scope": "reused plans; the toolbox is a complete call",
        "rounds": rounds,
        "toolbox_rounds": toolbox_rounds if toolbox else None,
        "backend": backend,
        "workers": workers,
        "sweeps": list(sweeps),
        "rows": rows,
    }


def run_toolboxes(
    rounds: int,
    backend: str = "auto",
    toolbox_rounds: int = 3,
    max_dimension: int = 1024,
    names: list[str] | None = None,
    cases: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Time the package beside the published toolboxes, and compare their values.

    Every case is an image of ``uint16`` samples with ``B = N`` bands and
    blocks of 32 pixels without overlap, the setting of the toolbox, where
    all the implementations compute the same index. The package is timed
    through a reused plan with ``algorithm="auto"``, and ``toolbox.py``, the
    NumPy port that the algorithms protocol times, as a complete call. An
    external implementation is timed by its adapter in ``toolboxes.py``, in
    ``toolbox_rounds`` rounds at most and in one when a round takes more
    than ``SLOW_SECONDS``: the MATLAB toolbox inside GNU Octave, around its
    ``q2n`` alone, and a Python port around its call. An implementation
    that cannot run here is recorded with the reason.
    """
    import toolbox as port
    import toolboxes

    names = list(toolboxes.SOURCES) if names is None else names
    reasons = {name: toolboxes.unavailable(name) for name in names}
    rng = np.random.default_rng(314159)
    dimensions = [
        n for n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= max_dimension
    ]
    if cases is None:
        cases = [(n, algorithm_side(n)) for n in dimensions]
    rows = []
    for bands, side in cases:
        shape = (side, side, bands)
        x, y = synthetic_pair(rng, shape, "uint16")
        with Q2nPlan(shape, 32, backend=backend) as plan:
            expected = plan.map(x, y)
            values, samples = measure(
                {
                    "inazumaQ2n": partial(plan.score, x, y),
                    "port": partial(port.q2n, x, y, 32),
                },
                rounds,
                {"port": toolbox_rounds},
            )
            algorithm = plan.algorithm
        ported = port.q2n_map(x, y, 32)
        differences = {"port": float(np.max(np.abs(ported - expected)))}
        failures = {}
        for name in names:
            if reasons[name] is not None:
                continue
            adapter = toolboxes.ADAPTERS[name]
            try:
                first = adapter(x, y, 32, 32, rounds=1)
                seconds = list(first.seconds)
                if seconds[0] <= SLOW_SECONDS and toolbox_rounds > 1:
                    seconds = adapter(x, y, 32, 32, rounds=toolbox_rounds).seconds
            except Exception as error:  # a port may not support a case
                failures[name] = f"{type(error).__name__}: {error}"
                continue
            values[name] = first.index
            samples[name] = seconds
            differences[name] = (
                float(np.max(np.abs(first.map - expected)))
                if first.map.shape == expected.shape
                else float("nan")
            )
        medians = {name: float(np.median(times)) for name, times in samples.items()}
        row = {
            "bands": bands,
            "dimension": 1 << (bands - 1).bit_length(),
            "shape": list(shape),
            "ws": 32,
            "stride": 32,
            "sample_type": "uint16",
            "algorithm": algorithm,
            "values": values,
            "largest_block_difference": differences,
            "failures": failures,
            "median_seconds": medians,
            "seconds": samples,
            "ratio_to_inazumaQ2n": {
                name: medians[name] / medians["inazumaQ2n"] for name in medians
            },
        }
        rows.append(row)
        report = ", ".join(
            f"{name} {1e3 * value:.3f} ms"
            + (f" (off by {differences[name]:.1e})" if name in differences else "")
            for name, value in medians.items()
        )
        print(f"{bands} bands {side}x{side} uint16 32/32: {report}", flush=True)
    return {
        "protocol": "toolboxes",
        "scope": "a reused plan; complete calls of the port; q2n alone inside Octave",
        "rounds": rounds,
        "toolbox_rounds": toolbox_rounds,
        "backend": backend,
        "sources": {
            name: {
                key: toolboxes.SOURCES[name][key]
                for key in ("title", "repository", "commit", "license", "files")
            }
            for name in names
        },
        "not_run": {name: reason for name, reason in reasons.items() if reason},
        "octave": toolboxes.octave_version(),
        "rows": rows,
    }


def toolboxes_report(result: dict[str, Any]) -> str:
    """Return the Markdown table of a toolboxes result."""
    titles = {"inazumaQ2n": "inazumaQ2n", "port": "NumPy port (`toolbox.py`)"}
    titles.update({name: source["title"] for name, source in result["sources"].items()})
    present = [
        name
        for name in titles
        if any(name in row["median_seconds"] for row in result["rows"])
    ]
    others = present[1:]
    head = ["N", "image", "algorithm"]
    head += [titles[name] for name in present]
    head += [f"{titles[name]} / inazumaQ2n" for name in others]
    head += [f"difference, {titles[name]}" for name in others]
    lines = [
        "# inazumaQ2n and the published toolboxes",
        "",
        f"Median milliseconds over {result['rounds']} rounds for inazumaQ2n, on a "
        f"reused plan, and over at most {result['toolbox_rounds']} for the others. "
        "Images of `uint16` samples with B = N bands, blocks of 32 pixels without "
        "overlap. The MATLAB toolbox is timed inside GNU Octave"
        + (f" {result['octave']}" if result.get("octave") else "")
        + ", around its `q2n` alone. The difference is the largest difference of the "
        "value of a block from that of inazumaQ2n.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + " ---: |" * len(head),
    ]

    def cell(value: float | None, pattern: str) -> str:
        return "" if value is None else format(value, pattern)

    for row in result["rows"]:
        medians, ratios = row["median_seconds"], row["ratio_to_inazumaQ2n"]
        cells = [
            str(row["dimension"]),
            f"{row['shape'][0]} x {row['shape'][1]}",
            row["algorithm"],
        ]
        cells += [
            cell(1e3 * medians[name] if name in medians else None, ",.3g")
            for name in present
        ]
        cells += [cell(ratios.get(name), ",.3g") for name in others]
        cells += [
            cell(row["largest_block_difference"].get(name), ".1e") for name in others
        ]
        lines.append("| " + " | ".join(cells) + " |")
    for reason in result["not_run"].values():
        lines += ["", f"Not run: {reason}."]
    failed = sorted({name for row in result["rows"] for name in row["failures"]})
    if failed:
        lines += ["", "Failed on some cases: " + ", ".join(failed) + "."]
    return "\n".join(lines) + "\n"


def algorithms_report(result: dict[str, Any]) -> str:
    """Return the Markdown tables of an algorithms result."""
    runtime = result.get("environment", {}).get("runtime", {})
    lines = [
        "# Algorithms of inazumaQ2n",
        "",
        f"Median milliseconds of a call on reused plans over {result['rounds']} rounds"
        + (
            f"; the toolbox is a complete call, timed in at most "
            f"{result['toolbox_rounds']} rounds."
            if result["toolbox_rounds"]
            else "."
        ),
        "`moments` contracts real cross moments, `fastcd` evaluates one fused sum of "
        "fastCD per block, and `toolbox` is the recursive product of the reference "
        "toolboxes in NumPy. B is the number of bands and N the dimension of the "
        'algebra. `auto` is the algorithm that `algorithm="auto"` selects, and its '
        "regret is the time of that algorithm over the time of the faster one.",
    ]
    if runtime:
        lines += [
            "",
            f"inazumaQ2n {runtime.get('version')}, kernels {runtime.get('simd')}, "
            f"matrix products by {runtime.get('backend')} "
            f"{runtime.get('backend_version')}, fastCD {runtime.get('fastcd')} with its "
            f"{runtime.get('fastcd_kernel')} fused-sum kernel; "
            f"{result.get('environment', {}).get('cpu')}.",
        ]

    def ms(row: dict[str, Any], name: str) -> str:
        value = row["median_seconds"].get(name)
        return "" if value is None else f"{1e3 * value:,.3g}"

    titles = {
        "dimension": "Dimension, B = N, blocks of 32 without overlap, float64",
        "bands": "Band counts below the dimension, blocks of 32 without overlap, float64",
        "overlap": "Overlapping blocks, float64",
        "types": "Sample types, B = N, blocks of 32 without overlap",
        "blocks": "Block sizes, B = N, no overlap, float64",
        "workers": "Workers, B = N, blocks of 32 without overlap, float64",
    }
    for sweep in result["sweeps"]:
        rows = [row for row in result["rows"] if row["sweep"] == sweep]
        if not rows:
            continue
        numpy_column = any("moments, numpy" in row["median_seconds"] for row in rows)
        toolbox_column = any(TOOLBOX in row["median_seconds"] for row in rows)
        head = ["B", "N", "image", "block / stride"]
        head += ["type"] if sweep == "types" else []
        head += ["workers"] if sweep == "workers" else []
        head += ["toolbox"] if toolbox_column else []
        head += ["moments"] + (["moments, NumPy"] if numpy_column else []) + ["fastcd"]
        head += ["toolbox / moments", "toolbox / fastcd"] if toolbox_column else []
        head += ["moments / fastcd", "auto", "regret"]
        lines += ["", f"## {titles[sweep]}", "", "| " + " | ".join(head) + " |"]
        lines.append("|" + "|".join(" ---: " for _ in head) + "|")
        for row in rows:
            cells = [
                str(row["bands"]),
                str(row["dimension"]),
                f"{row['shape'][0]} x {row['shape'][1]}",
                f"{row['ws']} / {row['stride']}",
            ]
            cells += [row["sample_type"]] if sweep == "types" else []
            cells += [str(row["workers"])] if sweep == "workers" else []
            cells += [ms(row, TOOLBOX)] if toolbox_column else []
            cells += [ms(row, "moments")]
            cells += [ms(row, "moments, numpy")] if numpy_column else []
            cells += [ms(row, "fastcd")]
            if toolbox_column:
                over = row["toolbox_over"]
                cells += (
                    ["", ""]
                    if over is None
                    else [f"{over['moments']:,.3g}", f"{over['fastcd']:,.3g}"]
                )
            cells += [
                f"{row['moments_over_fastcd']:.2f}",
                row["auto"],
                f"{row['auto_regret']:.2f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    worst = max(row["auto_regret"] for row in result["rows"])
    agree = max(row["max_abs_difference"] for row in result["rows"])
    lines += [
        "",
        f"The values of the plans agree to {agree:.1e} in every case, and the largest "
        f"regret of `auto` is {worst:.2f}.",
    ]
    return "\n".join(lines) + "\n"


def run_kernels(
    rounds: int,
    cases: list[tuple[int, int, int, int, int, str]] | None = None,
) -> dict[str, Any]:
    """Time the direct kernels of every instruction set against the general route."""
    best = _core.simd()
    levels = [level for level in _core.simd_levels() if level != "baseline"]
    backends = ["numpy"] + (["mkl"] if runtime_info()["backend"] == "mkl" else [])
    rng = np.random.default_rng(90417)
    rows = []
    try:
        for bands, h, w, ws, stride, dtype in KERNEL_CASES if cases is None else cases:
            shape = (h, w, bands)
            x, y = synthetic_pair(rng, shape, dtype)
            plans: dict[str, tuple[str, Q2nPlan]] = {}
            for level in levels:
                _core.set_simd(level)
                plans[f"direct, {level}"] = (level, Q2nPlan(shape, ws, stride))
            _core.set_simd(best)
            for backend in backends:
                plan = general_plan(shape, ws, stride, backend)
                plans[f"matrix product, {backend}"] = (best, plan)

            def call(level: str, plan: Q2nPlan) -> float:
                _core.set_simd(level)
                return plan.score(x, y)  # noqa: B023

            variants = {
                name: (lambda level=level, plan=plan: call(level, plan))
                for name, (level, plan) in plans.items()
            }
            values, samples = measure(variants, rounds)
            for _, plan in plans.values():
                plan.close()
            medians = {name: float(np.median(times)) for name, times in samples.items()}
            general = [name for name in plans if name.startswith("matrix")]
            fastest = min(medians[name] for name in general)
            paired = np.min([samples[name] for name in general], axis=0)
            row = {
                "bands": bands,
                "shape": list(shape),
                "ws": ws,
                "stride": stride,
                "sample_type": dtype,
                "values": values,
                "max_abs_difference": max(values.values()) - min(values.values()),
                "median_seconds": medians,
                "seconds": samples,
                "speedup_over_matrix_product": {
                    name: fastest / medians[name]
                    for name in plans
                    if name not in general
                },
                "rounds_faster_than_matrix_product": {
                    name: int(np.sum(np.asarray(samples[name]) < paired))
                    for name in plans
                    if name not in general
                },
            }
            if row["max_abs_difference"] > 1e-12:
                raise AssertionError(f"{shape}: the routes disagree, {values}")
            rows.append(row)
            report = ", ".join(f"{k} {1e3 * v:.3f} ms" for k, v in medians.items())
            print(f"{bands} bands {h}x{w} {ws}/{stride} {dtype}: {report}", flush=True)
    finally:
        _core.set_simd(best)
    summary = {}
    for level in levels:
        name = f"direct, {level}"
        ratios = [row["speedup_over_matrix_product"][name] for row in rows]
        summary[name] = {
            "geometric_mean_speedup": float(np.exp(np.mean(np.log(ratios)))),
            "minimum_speedup": float(min(ratios)),
        }
    return {
        "protocol": "kernels",
        "scope": "reused plans; the general route uses the faster library per round",
        "rounds": rounds,
        "summary": summary,
        "rows": rows,
    }


def main(arguments: list[str] | None = None) -> int:
    """Run a protocol and write its results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--protocol",
        choices=("calls", "kernels", "algorithms", "toolboxes"),
        default="calls",
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--backend", choices=("auto", "numpy", "mkl"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--simd", help="restrict the kernels to this instruction set (calls only)"
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="leave out the pyfastcd baselines (calls only)",
    )
    parser.add_argument(
        "--cases", nargs="*", help="run the cases whose names contain one of these"
    )
    parser.add_argument(
        "--sweeps",
        nargs="*",
        choices=SWEEPS,
        default=list(SWEEPS),
        help="the sweeps to run (algorithms only)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="the largest dimension of the sweeps (algorithms and toolboxes)",
    )
    parser.add_argument(
        "--no-toolbox",
        action="store_true",
        help="leave out the toolbox recursion (algorithms only)",
    )
    parser.add_argument(
        "--toolbox-rounds",
        type=int,
        default=3,
        help="the most rounds in which a toolbox is timed (algorithms, toolboxes)",
    )
    options = parser.parse_args(arguments)
    if options.rounds < 1 or options.toolbox_rounds < 1:
        parser.error("rounds must be positive")
    if options.simd:
        _core.set_simd(options.simd)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        parser.error("the benchmark needs threadpoolctl; install inazumaq2n[benchmark]")
    # One thread in every matrix-product library, whichever package loaded it
    with threadpool_limits(limits=1, user_api="blas"):
        if options.protocol == "calls":
            result = run_calls(
                options.rounds,
                options.backend,
                options.workers,
                not options.no_baselines,
                options.cases,
            )
        elif options.protocol == "kernels":
            result = run_kernels(options.rounds)
        elif options.protocol == "toolboxes":
            result = run_toolboxes(
                options.rounds,
                options.backend,
                options.toolbox_rounds,
                options.max_dimension,
            )
        else:
            result = run_algorithms(
                options.rounds,
                options.backend,
                options.workers,
                not options.no_toolbox,
                options.toolbox_rounds,
                tuple(options.sweeps),
                options.max_dimension,
            )
        result["environment"] = environment()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {options.output}")
    if options.protocol in ("algorithms", "toolboxes"):
        text = (
            algorithms_report(result)
            if options.protocol == "algorithms"
            else toolboxes_report(result)
        )
        report = options.output.with_suffix(".md")
        report.write_text(text, encoding="utf-8")
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

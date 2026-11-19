"""Timings of complete Q2n calls.

Two protocols are provided. ``calls`` times :func:`inazumaq2n.q2n` and a
reused :class:`inazumaq2n.Q2nPlan` on a fixed set of images, beside the
evaluation of the definition pixel by pixel with the products of pyfastcd,
with its quasilinear and with its direct algorithm (``baseline.py``).
``kernels`` times, for images with at most four bands, the kernels that need
no matrix product against the general route through a matrix product, in
every instruction set the processor supports.

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
    variants: dict[str, Callable[[], float]], rounds: int
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Return the value and the timings in seconds of every variant.

    Each variant is called once untimed, and then once per round in an
    order that rotates from round to round.
    """
    names = list(variants)
    values = {name: float(variants[name]()) for name in names}
    samples: dict[str, list[float]] = {name: [] for name in names}
    for index in range(rounds):
        shift = index % len(names)
        for name in names[shift:] + names[:shift]:
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
            for path in sources + [Path(__file__).resolve()]
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
            if plan.backend == "mkl" and not plan._direct:
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
        return Q2nPlan(shape, ws, stride, backend=backend)
    finally:
        _plan._DIRECT_BANDS = saved


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
    parser.add_argument("--protocol", choices=("calls", "kernels"), default="calls")
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
    options = parser.parse_args(arguments)
    if options.rounds < 1:
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
        else:
            result = run_kernels(options.rounds)
        result["environment"] = environment()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

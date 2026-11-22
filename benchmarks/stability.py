#!/usr/bin/env python3
"""Exact-reference accuracy study of the value of a Q2n block.

Every Q2n index is the mean of the values of its blocks, so the study
measures the value of one block, computed seven ways from the same stored
samples:

- I: inazumaq2n, the block evaluated from its own moments, as ``q2n`` and
  ``q2n_map`` evaluate blocks that do not overlap with
  ``algorithm="moments"``;
- F: inazumaq2n, the block evaluated by a fused sum of fastCD, as with
  ``algorithm="fastcd"``, behind the same guards as I;
- T: inazumaq2n, the block merged from four tiles, as for blocks that
  overlap with a stride of half the block size;
- G: F with the block merged from four tiles, as ``algorithm="fastcd"``
  evaluates blocks that overlap with a stride of half the block size;
- U: I without the evaluation by definition of the blocks that the guards of
  the package mark, to show what the guards change;
- Q and D: the definition evaluated pixel by pixel in NumPy, with the pixel
  products of pyfastcd's quasilinear and direct algorithms
  (``benchmarks/baseline.py``).

The reference, ``tests/exact.py``, evaluates the definition exactly. It
converts every binary64 sample to an integer times a power of two, forms the
sums, the sums of squares and the cross sums of the block in
arbitrary-precision integers, decides every branch of the definition (a zero
mean, a zero standard deviation, a zero variance) on these exact values, and
evaluates the square roots and quotients with decimal arithmetic at 100
significant digits by default. It uses no inazumaq2n or pyfastcd arithmetic,
and derives the signs of the algebra from the doubling formula on its own.
Before the study, it is checked against a literal evaluation of the
definition, pixel by pixel with the recursive product, in decimal
arithmetic.

Needs NumPy, pyfastcd and threadpoolctl.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests"))

import baseline  # noqa: E402
from exact import exact_value, literal_value  # noqa: E402

import inazumaq2n  # noqa: E402
from inazumaq2n import Q2nPlan, _core, q2n_map  # noqa: E402

SCHEMA = "inazumaq2n-stability-v2"
UNIT = Decimal(2) ** -53
METHODS = ("I", "F", "T", "G", "U", "Q", "D")
COMPARED = ("F", "T", "G", "U", "Q", "D")

TYPICAL = ("uniform", "gaussian")
OFFSETS = {"offset_1e3": 1e3, "offset_1e6": 1e6, "offset_1e9": 1e9, "offset_1e12": 1e12}
SCALES = {"scaled_down": 2.0**-530, "scaled_up": 2.0**490, "overflow": 2.0**515}
FAMILIES = (
    "uniform",
    "gaussian",
    "uint8",
    "uint16",
    "identical",
    "affine",
    "inverted",
    "independent",
    *OFFSETS,
    "zero_mean",
    "near_zero_mean",
    "constant_integer",
    "constant_fraction",
    *SCALES,
)
DESCRIPTIONS = {
    "uniform": "uniform samples in [0, 255), test = reference + N(0, 20^2)",
    "gaussian": "Gaussian bands with their own means and spreads, test = reference + N(0, 10^2)",
    "uint8": "integers in [0, 255], test = reference + rounded N(0, 20^2), clipped",
    "uint16": "integers in [0, 65535], test = reference + rounded N(0, 2000^2), clipped",
    "identical": "test = reference, uniform samples",
    "affine": "test = 0.8 reference + 2, uniform samples",
    "inverted": "test = 255 - reference, uniform samples",
    "independent": "independent uniform reference and test",
    "offset_1e3": "reference = 1e3 + U[0, 1), test = reference + N(0, 0.1^2)",
    "offset_1e6": "reference = 1e6 + U[0, 1), test = reference + N(0, 0.1^2)",
    "offset_1e9": "reference = 1e9 + U[0, 1), test = reference + N(0, 0.1^2)",
    "offset_1e12": "reference = 1e12 + U[0, 1), test = reference + N(0, 0.1^2)",
    "zero_mean": "reference bands of exactly zero mean in random pixel order, test = 0.8 reference + 2",
    "near_zero_mean": "zero_mean with one sample changed by a relative 2^-40",
    "constant_integer": "uniform, with reference band 0 constant at 7",
    "constant_fraction": "uniform, with reference band 0 constant at 0.1",
    "scaled_down": "uniform scaled by 2^-530; products of samples are subnormal",
    "scaled_up": "uniform scaled by 2^490",
    "overflow": "uniform scaled by 2^515; squares of deviations overflow",
}

BANDS = (1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 159, 256)
SIDE = 32
BLOCK_BANDS = (3, 16)
BLOCK_SIDES = (8, 16, 64)
SMOKE_BANDS = (1, 3, 5)
SMOKE_SIDE = 4


# The corpus


def case_seed(root: int, *labels: int) -> int:
    """Return the seed of one case, derived from the root seed and its labels."""
    return int(np.random.SeedSequence([root, *labels]).generate_state(1)[0])


def make_case(
    family: str, bands: int, side: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference and test blocks of one case, of shape (side, side, bands)."""
    rng = np.random.default_rng(seed)
    shape = (side, side, bands)
    x = rng.uniform(0, 255, shape)
    noise = rng.normal(0, 20, shape)
    if family in ("uniform", "constant_integer", "constant_fraction"):
        y = x + noise
        if family != "uniform":
            x[..., 0] = 7.0 if family == "constant_integer" else 0.1
    elif family == "gaussian":
        mean = rng.uniform(50, 200, bands)
        spread = rng.uniform(5, 40, bands)
        x = mean + spread * rng.standard_normal(shape)
        y = x + rng.normal(0, 10, shape)
    elif family == "uint8":
        x = rng.integers(0, 256, shape).astype(np.float64)
        y = np.clip(x + np.round(noise), 0, 255)
    elif family == "uint16":
        x = rng.integers(0, 65536, shape).astype(np.float64)
        y = np.clip(x + np.round(100 * noise), 0, 65535)
    elif family == "identical":
        y = x.copy()
    elif family == "affine":
        y = 0.8 * x + 2
    elif family == "inverted":
        y = 255 - x
    elif family == "independent":
        y = rng.uniform(0, 255, shape)
    elif family in OFFSETS:
        x = OFFSETS[family] + rng.uniform(0, 1, shape)
        y = x + rng.normal(0, 0.1, shape)
    elif family in ("zero_mean", "near_zero_mean"):
        pixels = side * side
        half = rng.normal(0, 10, (pixels // 2, bands))
        values = np.concatenate([half, -half, np.zeros((pixels % 2, bands))])
        x = values[rng.permutation(pixels)].reshape(shape)
        if family == "near_zero_mean":
            x[0, 0] += np.ldexp(x[0, 0], -40)
        y = 0.8 * x + 2
    elif family in SCALES:
        x, y = x * SCALES[family], (x + noise) * SCALES[family]
    else:
        raise ValueError(f"unknown family {family}")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def sample_count(family: str, bands: int, side: int, profile: str) -> int:
    """Return the number of cases of one family, band count and block size."""
    if profile == "smoke":
        return 1
    if side != SIDE:
        return 64
    if family in TYPICAL:
        return 256 if bands <= 16 else 64 if bands <= 64 else 16
    return 32 if bands <= 16 else 8 if bands <= 64 else 4


def groups(profile: str) -> Iterator[tuple[str, int, int]]:
    """Yield the (family, bands, side) groups of a profile."""
    if profile == "smoke":
        for family in FAMILIES:
            for bands in SMOKE_BANDS:
                yield family, bands, SMOKE_SIDE
        return
    for family in FAMILIES:
        for bands in BANDS:
            yield family, bands, SIDE
    for family in TYPICAL:
        for bands in BLOCK_BANDS:
            for side in BLOCK_SIDES:
                yield family, bands, side


# The measured methods


def measured_values(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], bool]:
    """Return the value of the block by every method, and whether the guards mark it."""
    side = x.shape[0]
    values = {}
    with Q2nPlan(x.shape, side, algorithm="fastcd") as plan:
        plan._run(x, y, None)
        values["F"] = float(plan._values[0])
        fused_marked = bool(plan._risk[0])
    with Q2nPlan(x.shape, side, backend="numpy", algorithm="moments") as plan:
        plan._run(x, y, None)
        values["I"] = float(plan._values[0])
        marked = bool(plan._risk[0])
        if fused_marked != marked:
            raise AssertionError("the guards of the two algorithms mark differently")
        unguarded = np.empty(1)
        _core.finish(
            plan._grams,
            plan._stats,
            plan._ids,
            plan._ids.shape[1],
            plan._pixels,
            plan._bands,
            plan._dimension,
            plan._signs,
            unguarded,
            np.empty_like(plan._scratch),
            np.empty(1, dtype=np.uint8),
        )
        values["U"] = float(unguarded[0])
    values["T"] = float(
        q2n_map(x, y, side, side // 2, backend="numpy", algorithm="moments")[0, 0]
    )
    values["G"] = float(q2n_map(x, y, side, side // 2, algorithm="fastcd")[0, 0])
    for name, algorithm in (("Q", "quasilinear"), ("D", "direct")):
        values[name] = float(baseline.q2n_map(x, y, side, algorithm=algorithm)[0, 0])
    return values, marked


def error_of(value: float, exact: Decimal) -> Decimal | None:
    """Return the exact absolute error of a computed value, or None if it is not finite."""
    return abs(Decimal(value) - exact) if math.isfinite(value) else None


def measures(value: float, error: Decimal | None, exact: Decimal) -> dict[str, Any]:
    """Return the error measures of one computed value."""
    if error is None:
        return {"value": None, "absolute": None, "eta": None, "relative": None}
    return {
        "value": value,
        "absolute": float(error),
        "eta": float(error / UNIT),
        "relative": float(error / exact) if exact else None,
    }


def comparison(first: Decimal | None, second: Decimal | None) -> str:
    """Compare the error of I with that of another method: better, tied or worse."""
    if first is None:
        return "tied" if second is None else "worse"
    if second is None or first < second:
        return "better"
    return "tied" if first == second else "worse"


def run_case(
    family: str, bands: int, side: int, index: int, root: int, digits: int, check: bool
) -> dict[str, Any]:
    """Measure one case."""
    seed = case_seed(root, FAMILIES.index(family), bands, side, index)
    x, y = make_case(family, bands, side, seed)
    exact = exact_value(x, y, digits)
    values, marked = measured_values(x, y)
    errors = {name: error_of(value, exact) for name, value in values.items()}
    row: dict[str, Any] = {
        "family": family,
        "bands": bands,
        "dimension": 1 << (bands - 1).bit_length(),
        "side": side,
        "index": index,
        "seed": seed,
        "operands_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        "exact": f"{exact:.25e}",
        "marked": marked,
        "methods": {
            name: measures(values[name], errors[name], exact) for name in METHODS
        },
        "comparisons": {
            name: comparison(errors["I"], errors[name]) for name in COMPARED
        },
    }
    if check:
        finer = exact_value(x, y, 2 * digits)
        row["precision_check"] = float(abs(finer - exact) / max(abs(finer), Decimal(1)))
    return row


def run_group(task: tuple[str, int, int, int, int, int, str]) -> list[dict[str, Any]]:
    """Measure every case of one group, with one thread in every BLAS."""
    from threadpoolctl import threadpool_limits

    family, bands, side, count, root, digits, _ = task
    with threadpool_limits(limits=1), np.errstate(all="ignore"):
        return [
            run_case(family, bands, side, index, root, digits, check=index == 0)
            for index in range(count)
        ]


def validate_oracle(root: int, digits: int) -> dict[str, Any]:
    """Compare exact_value with the literal evaluation of the definition."""
    worst = 0.0
    checks = 0
    families = ("uniform", "identical", "independent", "zero_mean", "near_zero_mean")
    families += ("constant_integer", "constant_fraction", "offset_1e6", "scaled_down")
    for family in families:
        for bands in (1, 2, 3, 4, 5, 8):
            for side in (2, 3, 4):
                seed = case_seed(root, 1000 + FAMILIES.index(family), bands, side)
                x, y = make_case(family, bands, side, seed)
                fast = exact_value(x, y, digits)
                slow = literal_value(x, y, digits)
                scale = max(abs(slow), Decimal(1))
                worst = max(worst, float(abs(fast - slow) / scale))
                checks += 1
    tolerance = 10.0 ** (-digits // 2)
    if not worst <= tolerance:
        raise AssertionError(
            f"the exact reference differs from the definition by {worst}"
        )
    return {"checks": checks, "max_relative_difference": worst, "tolerance": tolerance}


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
    """Return the host and the software of the study."""
    import pyfastcd

    return {
        "cpu_model": cpu_model(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pyfastcd": pyfastcd.__version__,
        "inazumaq2n": inazumaq2n.__version__,
        "runtime": inazumaq2n.runtime_info(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the study and return its results; with --output, also write them."""
    import json

    start = time.perf_counter()
    oracle = validate_oracle(args.seed, args.digits)
    tasks = [
        (family, bands, side, sample_count(family, bands, side, args.profile), args.seed,
         args.digits, args.profile)
        for family, bands, side in groups(args.profile)
    ]  # fmt: skip
    # The largest groups first, so that the workers finish together
    tasks.sort(key=lambda task: -task[3] * task[1] * task[1])
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_group, tasks))
    else:
        results = [run_group(task) for task in tasks]
    order = {name: k for k, name in enumerate(FAMILIES)}
    samples = sorted(
        (row for rows in results for row in rows),
        key=lambda r: (order[r["family"]], r["bands"], r["side"], r["index"]),
    )
    checks = [r["precision_check"] for r in samples if "precision_check" in r]
    document = {
        "schema": SCHEMA,
        "methods": list(METHODS),
        "completed": True,
        "profile": args.profile,
        "seed": args.seed,
        "digits": args.digits,
        "jobs": args.jobs,
        "environment": environment(),
        "oracle_validation": oracle,
        "precision_checks": {
            "cases": len(checks),
            "digits": 2 * args.digits,
            "max_relative_difference": max(checks, default=0.0),
        },
        "corpus": {
            "cases": len(samples),
            "families": {f: DESCRIPTIONS[f] for f in FAMILIES},
            "bands": sorted({r["bands"] for r in samples}),
            "sides": sorted({r["side"] for r in samples}),
        },
        "seconds": time.perf_counter() - start,
        "samples": samples,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, allow_nan=False) + "\n")
    return document


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the options of the study."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="JSON file for the results")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--digits", type=int, default=100, help="significant digits of the reference"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    document = run(parse_arguments())
    print(f"{document['corpus']['cases']} cases in {document['seconds']:.0f} s")

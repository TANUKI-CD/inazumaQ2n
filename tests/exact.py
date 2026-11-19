"""An exact evaluation of the value of a Q2n block, from its definition.

This module imports nothing from inazumaq2n and needs only NumPy and the
standard library. It is the reference of the tests of conditioned inputs and
of the stability study in ``benchmarks/stability.py``.

``exact_value`` converts every binary64 sample to an integer times a power
of two, forms the sums, the sums of squares and the cross sums of the block
in arbitrary-precision integers, decides every branch of the definition (a
zero mean, a zero standard deviation, a zero variance) on these exact
values, and evaluates the square roots and quotients with decimal
arithmetic at a chosen number of significant digits. It derives the signs
of the algebra from the doubling formula on its own. ``literal_value``
evaluates the definition literally, pixel by pixel with the recursive
product, in decimal arithmetic, and serves to check ``exact_value`` on small
blocks.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np

# Bits of one limb of the integer encoding; P * 2^(2 LIMB) must stay below 2^53
LIMB = 20
MAX_PIXELS = 8192


@cache
def sign_table(size: int) -> tuple[tuple[int, ...], ...]:
    """Derive the basis signs from (a,b)(c,d) = (ac - conj(d) b, da + b conj(c)).

    Entry (i, j) is the sign of e_i e_j = sign e_(i XOR j). The table is built
    by the doubling recurrence and does not use the sign table of inazumaq2n.
    """
    if size < 1 or size & (size - 1):
        raise ValueError("dimension must be a positive power of two")
    if size == 1:
        return ((1,),)
    s = sign_table(size // 2)
    half = size // 2
    eps = [1] + [-1] * (half - 1)
    rows = [list(s[i]) + [s[j][i] for j in range(half)] for i in range(half)]
    rows += [
        [eps[j] * s[i][j] for j in range(half)]
        + [-eps[j] * s[j][i] for j in range(half)]
        for i in range(half)
    ]
    return tuple(tuple(row) for row in rows)


def encode(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return limbs and exponents that represent a (pixels, bands) array.

    The representation is exact: column ``i`` of ``values`` equals
    ``sum_k limbs[k][:, i] 2^(LIMB k) 2^exponents[i]``. Every limb is a
    float64 holding an integer of magnitude below ``2^LIMB``, with the sign
    of the value it belongs to.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    mantissa, exponent = np.frexp(values)
    integer = np.ldexp(mantissa, 53).astype(np.int64)
    nonzero = integer != 0
    lowest = np.where(nonzero, integer & -integer, 1)
    trailing = np.frexp(lowest.astype(np.float64))[1] - 1
    shift = np.where(
        nonzero, exponent.astype(np.int64) - 53 + trailing, np.iinfo(np.int64).max
    )
    exponents = shift.min(axis=0)
    exponents = np.where(exponents == np.iinfo(np.int64).max, 0, exponents)
    with np.errstate(over="ignore"):
        scaled = np.ldexp(values, -exponents)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("the exponents of a band span too wide a range")
    limbs = []
    rest = scaled
    while True:
        low = np.fmod(rest, 2.0**LIMB)
        limbs.append(low)
        rest = (rest - low) / 2.0**LIMB
        if not np.any(rest):
            break
    return np.stack(limbs), [int(e) for e in exponents]


def grouped(products: np.ndarray) -> np.ndarray:
    """Sum the limb products of axes 0 and 1 that share the power 2^(LIMB (a + b))."""
    first, second = products.shape[:2]
    totals = np.zeros((first + second - 1,) + products.shape[2:], dtype=np.int64)
    for a in range(first):
        for b in range(second):
            totals[a + b] += products[a, b].astype(np.int64)
    return totals


def combine(totals: np.ndarray) -> np.ndarray:
    """Return sum_s totals[s] 2^(LIMB s) as an array of Python integers."""
    result = totals[-1].astype(object)
    for s in range(len(totals) - 2, -1, -1):
        result = result * (1 << LIMB) + totals[s].astype(object)
    return result


def exact_moments(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Return the exact sums, sums of squares and cross sums of two blocks.

    The blocks are (pixels, bands) arrays.

    In the result, ``sum_p x[p, i] = sx[i] 2^ex[i]``,
    ``sum_p x[p, i]^2 = sxx[i] 2^(2 ex[i])`` and
    ``sum_p x[p, i] y[p, j] = sxy[i, j] 2^(ex[i] + ey[j])``, with integers
    ``sx``, ``sxx`` and ``sxy``, and likewise for ``y``. Every limb product
    is below 2^(2 LIMB) and a column has at most MAX_PIXELS entries, so every
    partial sum of a float64 matrix product of limbs is an integer below
    2^53, and the products are exact in any order of summation.
    """
    pixels = x.shape[0]
    if pixels > MAX_PIXELS:
        raise ValueError(f"at most {MAX_PIXELS} pixels per block")
    lx, ex = encode(x)
    ly, ey = encode(y)
    kx, ky, bands = len(lx), len(ly), x.shape[1]
    left = lx.transpose(1, 0, 2).reshape(pixels, kx * bands)
    right = ly.transpose(1, 0, 2).reshape(pixels, ky * bands)
    cross = (left.T @ right).reshape(kx, bands, ky, bands).transpose(0, 2, 1, 3)
    return {
        "sx": combine(lx.sum(axis=1).astype(np.int64)),
        "sy": combine(ly.sum(axis=1).astype(np.int64)),
        "sxx": combine(grouped(np.einsum("apb,cpb->acb", lx, lx))),
        "syy": combine(grouped(np.einsum("apb,cpb->acb", ly, ly))),
        "sxy": combine(grouped(cross)),
        "ex": ex,
        "ey": ey,
    }


def exact_value(x: np.ndarray, y: np.ndarray, digits: int = 100) -> Decimal:
    """Return the value of the block (side, side, bands) of the definition.

    With P pixels, the means a_i and b_i, and the centered sums Vx_i, Vy_i and
    C_ij of docs/mathematics.md, the value is
    2 bias |mu(D C D')| / sum_i (Vx_i / c_i^2 + Vy_i / c'_i^2). Every branch
    is decided on exact integers; only square roots and quotients are
    rounded, to ``digits`` significant digits.
    """
    side, _, bands = x.shape
    pixels = side * side
    dimension = 1 << (bands - 1).bit_length()
    m = exact_moments(x.reshape(pixels, bands), y.reshape(pixels, bands))
    sx, sy, ex, ey = m["sx"], m["sy"], m["ex"], m["ey"]
    # P Vx_i 2^(-2 ex_i), P Vy_i 2^(-2 ey_i) and P C_ij 2^(-ex_i - ey_j)
    pvx = [pixels * m["sxx"][i] - sx[i] * sx[i] for i in range(bands)]
    pvy = [pixels * m["syy"][i] - sy[i] * sy[i] for i in range(bands)]
    pc = pixels * m["sxy"] - np.multiply.outer(sx, sy)
    signs = sign_table(dimension)
    with localcontext() as context:
        context.prec = digits
        context.Emax, context.Emin = 10**6, -(10**6)
        two, p = Decimal(2), Decimal(pixels)
        eps = two**-52
        c, cp = [], []
        for i in range(bands):
            if pvx[i] == 0:
                c.append(eps)
            else:
                c.append((Decimal(pvx[i]) * two ** (2 * ex[i]) / (p * (p - 1))).sqrt())
            cp.append(Decimal(1) if sx[i] == 0 else c[i])

        # The norm of the mean of the conjugated, normalized test pixels
        n2 = Decimal(dimension - bands)
        for i in range(bands):
            e = min(ex[i], ey[i])
            difference = (sy[i] << (ey[i] - e)) - (sx[i] << (ex[i] - e))
            term = 1 + Decimal(difference) * two**e / (p * cp[i])
            n2 += term * term
        n = Decimal(dimension)
        bias = 2 * (n * n2).sqrt() / (n + n2)

        # The sum of the variances, exactly
        variance = Fraction(0)
        for i in range(bands):
            if pvx[i] != 0:
                variance += pixels - 1
            vy = Fraction(pvy[i]) * Fraction(2) ** (2 * ey[i]) / pixels
            if sx[i] == 0:
                variance += vy
            elif pvx[i] == 0:
                variance += vy * Fraction(2) ** 104
            else:
                variance += (
                    Fraction(pixels - 1)
                    * vy
                    * pixels
                    / (Fraction(pvx[i]) * Fraction(2) ** (2 * ex[i]))
                )
        if variance == 0:
            return abs(bias)

        # The hypercomplex covariance, contracted with the signs
        u = [two ** ex[i] / (p * c[i]) for i in range(bands)]
        v = [(1 if j == 0 else -1) * two ** ey[j] / cp[j] for j in range(bands)]
        covariance = [Decimal(0)] * dimension
        for i in range(bands):
            row = signs[i]
            for j in range(bands):
                if pc[i, j]:
                    covariance[i ^ j] += row[j] * Decimal(pc[i, j]) * u[i] * v[j]
        norm = sum(value * value for value in covariance).sqrt()
        total = Decimal(variance.numerator) / Decimal(variance.denominator)
        return 2 * bias * norm / total


def literal_value(x: np.ndarray, y: np.ndarray, digits: int = 100) -> Decimal:
    """Return the value of a block by evaluating the definition literally.

    The bands are padded with zeros, normalized and conjugated, every pixel
    is multiplied with the recursive Cayley--Dickson product, and the means
    are averaged, all in decimal arithmetic. This is slow and serves to
    check exact_value on small blocks.
    """
    side, _, bands = x.shape
    pixels = side * side
    dimension = 1 << (bands - 1).bit_length()

    def product(a: list[Decimal], b: list[Decimal]) -> list[Decimal]:
        if len(a) == 1:
            return [a[0] * b[0]]
        h = len(a) // 2
        p, q, r, s = a[:h], a[h:], b[:h], b[h:]
        conj_s = s[:1] + [-t for t in s[1:]]
        conj_r = r[:1] + [-t for t in r[1:]]
        first = [g - k for g, k in zip(product(p, r), product(conj_s, q), strict=True)]
        second = [g + k for g, k in zip(product(s, p), product(q, conj_r), strict=True)]
        return first + second

    with localcontext() as context:
        context.prec = digits
        context.Emax, context.Emin = 10**6, -(10**6)
        flat_x = x.reshape(pixels, bands)
        flat_y = y.reshape(pixels, bands)
        xs = [[Decimal(0)] * dimension for _ in range(pixels)]
        zs = [[Decimal(0)] * dimension for _ in range(pixels)]
        p = Decimal(pixels)
        for i in range(dimension):
            column_x = (
                [Decimal(float(v)) for v in flat_x[:, i]]
                if i < bands
                else [Decimal(0)] * pixels
            )
            column_y = (
                [Decimal(float(v)) for v in flat_y[:, i]]
                if i < bands
                else [Decimal(0)] * pixels
            )
            mean = sum(column_x) / p
            sd = (sum((v - mean) ** 2 for v in column_x) / (p - 1)).sqrt()
            if sd == 0:
                sd = Decimal(2) ** -52
            scale = Decimal(1) if mean == 0 else sd
            for k in range(pixels):
                xs[k][i] = (column_x[k] - mean) / sd + 1
                value = (column_y[k] - mean) / scale + 1
                zs[k][i] = value if i == 0 else -value
        m1 = [sum(px[i] for px in xs) / p for i in range(dimension)]
        m2 = [sum(pz[i] for pz in zs) / p for i in range(dimension)]
        mean_product = [Decimal(0)] * dimension
        for k in range(pixels):
            for i, value in enumerate(product(xs[k], zs[k])):
                mean_product[i] += value / p
        factor = p / (p - 1)
        a1 = sum(v * v for v in m1).sqrt()
        a2 = sum(v * v for v in m2).sqrt()
        t4 = a1 * a1 + a2 * a2
        squares = sum(sum(v * v for v in px) for px in xs) / p
        squares += sum(sum(v * v for v in pz) for pz in zs) / p
        t3 = factor * squares - factor * t4
        bias = 2 * a1 * a2 / t4
        if t3 == 0:
            return abs(bias)
        means = product(m1, m2)
        q = [
            (factor * g - factor * h) * bias * 2 / t3
            for g, h in zip(mean_product, means, strict=True)
        ]
        return sum(v * v for v in q).sqrt()


def exact_map(
    reference: np.ndarray,
    test: np.ndarray,
    ws: int,
    stride: int | None = None,
    digits: int = 100,
) -> list[list[Decimal]]:
    """Return the exact value of every block, with bottom and right reflection."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(test, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    h, w, _ = x.shape
    stride = ws if stride is None else stride
    rows, columns = math.ceil(h / stride), math.ceil(w / stride)
    pad = (
        (0, (rows - 1) * stride + ws - h),
        (0, (columns - 1) * stride + ws - w),
        (0, 0),
    )
    x, y = np.pad(x, pad, mode="symmetric"), np.pad(y, pad, mode="symmetric")
    return [
        [
            exact_value(
                np.ascontiguousarray(x[r : r + ws, c : c + ws]),
                np.ascontiguousarray(y[r : r + ws, c : c + ws]),
                digits,
            )
            for c in range(0, columns * stride, stride)
        ]
        for r in range(0, rows * stride, stride)
    ]

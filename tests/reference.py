"""A slow, independent evaluation of Q2n from its definition.

This module imports nothing from inazumaq2n. It pads the images, normalizes
every band of every block, multiplies the pixels by the recursive
Cayley--Dickson product and averages, as the reference implementations of
the index do. No sign table, cross-moment matrix or compiled code is used,
so agreement with the package checks the identities the package relies on.

The reference implementations follow their product by a fixed pattern of
sign changes on the coordinates. The pattern preserves the norm, which is
all the index takes, so it is omitted here.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def conjugate(x: Array) -> Array:
    """Return the conjugates of the elements on the last axis of ``x``."""
    result = x.copy()
    result[..., 1:] *= -1
    return result


def multiply(x: Array, y: Array) -> Array:
    """Return the Cayley--Dickson products of the elements on the last axis.

    The product is ``(a, b)(c, d) = (ac - conj(d) b, d a + b conj(c))``.
    """
    if x.shape[-1] == 1:
        return x * y
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c, d = y[..., :half], y[..., half:]
    return np.concatenate(
        (
            multiply(a, c) - multiply(conjugate(d), b),
            multiply(d, a) + multiply(b, conjugate(c)),
        ),
        axis=-1,
    )


def block_value(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Return the Q2n value of one block of shape ``(side, side, bands)``."""
    xn = np.array(x, dtype=np.float64)
    yn = np.array(y, dtype=np.float64)
    side, _, bands = xn.shape
    dimension = 1 << (bands - 1).bit_length()
    xn = np.pad(xn, ((0, 0), (0, 0), (0, dimension - bands)))
    yn = np.pad(yn, ((0, 0), (0, 0), (0, dimension - bands)))
    mx, my = np.empty(dimension), np.empty(dimension)
    normx, normy = np.zeros((side, side)), np.zeros((side, side))
    for i in range(dimension):
        mean, sd = xn[:, :, i].mean(), xn[:, :, i].std(ddof=1)
        if sd == 0:
            sd = np.finfo(float).eps
        xn[:, :, i] = (xn[:, :, i] - mean) / sd + 1
        yn[:, :, i] = (yn[:, :, i] - mean) / (1 if mean == 0 else sd) + 1
        if i:
            yn[:, :, i] *= -1
        mx[i], my[i] = xn[:, :, i].mean(), yn[:, :, i].mean()
        normx += xn[:, :, i] ** 2
        normy += yn[:, :, i] ** 2
    ax, ay = np.sqrt(sum(mx**2)), np.sqrt(sum(my**2))
    normx, normy = np.sqrt(normx), np.sqrt(normy)
    factor = side**2 / (side**2 - 1.0)
    mean_norm = ax**2 + ay**2
    variance = factor * (np.mean(normx**2) + np.mean(normy**2)) - factor * mean_norm
    bias = 2 * ax * ay / mean_norm if mean_norm > 0 else 0.0
    if variance == 0:
        return float(abs(bias))
    covariance = factor * np.mean(multiply(xn, yn), axis=(0, 1)) - factor * multiply(
        mx, my
    )
    q = covariance * bias * (2 / variance)
    return float(np.sqrt(np.sum(q * q)))


def q2n_map(
    x: npt.ArrayLike, y: npt.ArrayLike, ws: int = 32, stride: int | None = None
) -> Array:
    """Return the value of every block, with symmetric bottom and right padding."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if xa.ndim == 2:
        xa, ya = xa[..., None], ya[..., None]
    h, w, _ = xa.shape
    stride = ws if stride is None else stride
    nr, nc = math.ceil(h / stride), math.ceil(w / stride)
    ph, pw = (nr - 1) * stride + ws - h, (nc - 1) * stride + ws - w
    xa = np.pad(xa, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    ya = np.pad(ya, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    return np.array(
        [
            [
                block_value(xa[r : r + ws, c : c + ws], ya[r : r + ws, c : c + ws])
                for c in range(0, nc * stride, stride)
            ]
            for r in range(0, nr * stride, stride)
        ]
    )


def q2n(
    x: npt.ArrayLike, y: npt.ArrayLike, ws: int = 32, stride: int | None = None
) -> float:
    """Return the index: the mean of the values of the blocks."""
    return float(np.mean(q2n_map(x, y, ws, stride)))

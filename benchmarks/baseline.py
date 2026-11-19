"""Q2n evaluated from its definition, pixel by pixel, with the products of pyfastcd.

This is the comparison that the benchmark and the stability study use. It
computes the index as the definition states it (see docs/mathematics.md):
in every block it normalizes the bands of both images with the statistics of
the reference, multiplies every normalized reference pixel by the conjugate
of the normalized test pixel in the Cayley--Dickson algebra, and averages.
The products are those of pyfastcd, the Python package of the fastCD
library, with its quasilinear algorithm or its direct algorithm.

The evaluation is vectorized in NumPy over the pixels, the bands and a batch
of blocks, and the products of a batch are one call of pyfastcd. Bands beyond
the image are not stored as zeros and normalized: their normalized values,
one in the reference and minus one in the conjugated test image, are written
directly, which is exact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pyfastcd
from numpy.lib.stride_tricks import sliding_window_view

ALGORITHMS = ("quasilinear", "direct")

# Target size of the arrays of one batch of blocks
WORKSPACE_BYTES = 8 << 20


def q2n_map(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    algorithm: str = "quasilinear",
) -> npt.NDArray[np.float64]:
    """Return the value of every block, with pixel products by pyfastcd.

    The parameters are those of ``inazumaq2n.q2n_map``, and ``algorithm`` is
    ``"quasilinear"`` or ``"direct"``, the multiplication algorithm of
    pyfastcd.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(test, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("the images must have one shape, (height, width[, bands])")
    stride = ws if stride is None else stride
    height, width, bands = x.shape
    dimension = 1 << (bands - 1).bit_length()
    rows, columns = math.ceil(height / stride), math.ceil(width / stride)
    pad = (
        (0, (rows - 1) * stride + ws - height),
        (0, (columns - 1) * stride + ws - width),
        (0, 0),
    )
    windows = [
        sliding_window_view(np.pad(image, pad, mode="symmetric"), (ws, ws), axis=(0, 1))
        for image in (x, y)
    ]
    # Shape (rows, columns, bands, ws, ws): every block, without a copy
    blocks = [w[::stride, ::stride][:rows, :columns] for w in windows]
    algebra = pyfastcd.Algebra(dimension=dimension)
    pixels = ws * ws
    batch = max(1, WORKSPACE_BYTES // (32 * pixels * dimension))
    origins = [(r, c) for r in range(rows) for c in range(columns)]
    values = np.empty(len(origins))
    for first in range(0, len(origins), batch):
        chosen = origins[first : first + batch]
        xb = np.stack([blocks[0][r, c] for r, c in chosen]).reshape(-1, bands, pixels)
        yb = np.stack([blocks[1][r, c] for r, c in chosen]).reshape(-1, bands, pixels)
        values[first : first + len(chosen)] = block_values(
            xb, yb, dimension, algebra, algorithm
        )
    return values.reshape(rows, columns)


def q2n(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    algorithm: str = "quasilinear",
) -> float:
    """Return the index, the mean of the values of the blocks."""
    return float(np.mean(q2n_map(reference, test, ws, stride, algorithm=algorithm)))


def block_values(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    dimension: int,
    algebra: Any,
    algorithm: str,
) -> npt.NDArray[np.float64]:
    """Return the values of a batch of blocks of shape (blocks, bands, pixels)."""
    count, bands, pixels = x.shape
    eps = np.finfo(np.float64).eps
    # Squares that overflow give the result the definition gives in floating point
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean = x.mean(axis=2, keepdims=True)
        sd = x.std(axis=2, ddof=1, keepdims=True)
        sd[sd == 0] = eps
        xn = (x - mean) / sd + 1
        yn = (y - mean) / np.where(mean == 0, 1.0, sd) + 1

        # Pixels as elements of the algebra; the test image is conjugated
        xs = np.ones((count, pixels, dimension))
        zs = -np.ones((count, pixels, dimension))
        xs[:, :, :bands] = xn.transpose(0, 2, 1)
        zs[:, :, :bands] = -yn.transpose(0, 2, 1)
        zs[:, :, 0] = yn[:, 0, :]
        products = np.empty_like(xs)
        algebra.multiply(xs, zs, out=products, algorithm=algorithm)

        # Every mean over the pixels runs along a contiguous axis, where NumPy
        # sums pairwise, as it does over the pixels of one block
        m1 = np.ascontiguousarray(xs.transpose(0, 2, 1)).mean(axis=2)
        m2 = np.ascontiguousarray(zs.transpose(0, 2, 1)).mean(axis=2)
        mean_product = np.ascontiguousarray(products.transpose(0, 2, 1)).mean(axis=2)
        means = np.empty_like(m1)
        algebra.multiply(m1, m2, out=means, algorithm=algorithm)
        factor = pixels / (pixels - 1.0)
        a1 = np.sqrt(np.sum(m1 * m1, axis=1))
        a2 = np.sqrt(np.sum(m2 * m2, axis=1))
        t4 = a1 * a1 + a2 * a2
        squares = np.mean(np.sum(xs * xs, axis=2), axis=1)
        squares += np.mean(np.sum(zs * zs, axis=2), axis=1)
        t3 = factor * squares - factor * t4
        bias = 2 * a1 * a2 / t4
        q = (factor * mean_product - factor * means) * (bias * 2 / t3)[:, None]
        return np.where(t3 == 0, np.abs(bias), np.sqrt(np.sum(q * q, axis=1)))

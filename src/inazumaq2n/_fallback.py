"""The Cayley--Dickson sign table and the evaluation of a block by definition.

The compiled kernels evaluate a block from its centered moments, using the
fact that the normalized reference image has mean exactly one in every
band. That holds in exact arithmetic, but in floating point it treats the
reference and the test image inconsistently when a band has a mean far
larger than its standard deviation. The kernels also cannot decide the
branches of the definition exactly: whether the mean of a reference band is
zero, and whether the band is constant. And their raw moments can underflow
or overflow at extreme scales. The kernels mark the blocks where any of this
may matter, and :func:`block_value` evaluates those blocks from the
normalized samples instead, deciding every branch exactly.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt


@lru_cache(maxsize=32)
def sign_table(bands: int) -> npt.NDArray[np.int8]:
    """Return the signs of the products of the first ``bands`` basis elements.

    Entry ``(i, j)`` is ``sigma(i, j)`` in ``e_i e_j = sigma(i, j) e_(i XOR j)``,
    for the Cayley--Dickson algebra whose dimension is the least power of two
    that is at least ``bands``. The table is built by the doubling recurrence
    and is read-only.
    """
    dimension = 1 << (bands - 1).bit_length()
    table = np.ones((1, 1), dtype=np.int8)
    while len(table) < dimension:
        eps = -np.ones(len(table), dtype=np.int8)
        eps[0] = 1
        table = np.block([[table, table.T], [table * eps, -table.T * eps]])
    result = np.ascontiguousarray(table[:bands, :bands])
    result.flags.writeable = False
    return result


def standard_deviation(deviations: npt.NDArray[np.float64]) -> float:
    """Return the unbiased standard deviation of centered samples.

    The samples are scaled by a power of two before they are squared, which
    is exact, so the squares neither overflow nor underflow.
    """
    largest = float(np.max(np.abs(deviations)))
    if largest == 0:
        return 0.0
    exponent = math.frexp(largest)[1]
    scaled = np.ldexp(deviations, -exponent)
    variance = float(scaled @ scaled) / (len(deviations) - 1)
    return math.ldexp(math.sqrt(variance), exponent)


def block_value(x: npt.NDArray[np.generic], y: npt.NDArray[np.generic]) -> float:
    """Return the Q2n value of one block, evaluated from its normalized samples.

    Parameters
    ----------
    x, y : ndarray
        The reference and test blocks, of shape ``(side, side, bands)``.

    Returns
    -------
    float
        The value of the block.

    Notes
    -----
    The mean of each reference band is its correctly rounded sum divided by
    the number of pixels, so it is zero exactly when the exact mean is zero,
    and a band is constant, with standard deviation replaced by the machine
    epsilon, exactly when all its samples are equal. The standard deviation
    is computed without overflow or underflow. The samples of both images
    are normalized with the same rounded statistics, as the definition does,
    and the covariance and the variances are formed from the normalized
    samples minus their means, so that no difference of large sums is taken.
    """
    side, _, bands = x.shape
    pixels = side * side
    dimension = 1 << (bands - 1).bit_length()
    xb = np.asarray(x, dtype=np.float64).reshape(pixels, bands)
    yb = np.asarray(y, dtype=np.float64).reshape(pixels, bands)
    eps = float(np.finfo(np.float64).eps)
    # Bands beyond the image normalize to one, and to minus one when conjugated
    xn = np.ones((pixels, dimension))
    zn = -np.ones((pixels, dimension))
    for i in range(bands):
        column = xb[:, i]
        if np.all(column == column[0]):
            mean, sd = float(column[0]), eps
        else:
            mean = math.fsum(column) / pixels
            sd = standard_deviation(column - mean) or eps
        xn[:, i] = (column - mean) / sd + 1
        normalized = (yb[:, i] - mean) / (1.0 if mean == 0 else sd) + 1
        zn[:, i] = normalized if i == 0 else -normalized
    m1, m2 = xn.mean(axis=0), zn.mean(axis=0)
    xc, zc = xn - m1, zn - m2
    a1 = math.sqrt(float(m1 @ m1))
    a2 = math.sqrt(float(m2 @ m2))
    t4 = a1 * a1 + a2 * a2
    t3 = (float(np.sum(xc * xc)) + float(np.sum(zc * zc))) / (pixels - 1)
    bias = 2 * a1 * a2 / t4 if t4 > 0 else 0.0
    if t3 == 0:
        return abs(bias)
    signs = sign_table(dimension)
    index = np.arange(dimension)
    targets = np.bitwise_xor(index[:, None], index[None, :]).ravel()
    weights = ((xc.T @ zc) * signs).ravel()
    covariance = np.bincount(targets, weights=weights, minlength=dimension)
    q = covariance / (pixels - 1) * (bias * 2 / t3)
    return float(np.sqrt(np.sum(q * q)))

"""Q2n as the reference toolboxes compute it: the recursive product, block by block.

The benchmark compares inazumaq2n with this module. It is a NumPy port of the
structure of the MATLAB implementation of Q2n in the pansharpening toolbox of
Vivone et al. (IEEE Transactions on Geoscience and Remote Sensing 53(5),
2015), which its Python ports keep: the image is padded to the blocks and to
a power-of-two number of bands, and for every block ``onions_quality``
normalizes the bands one at a time, multiplies every reference pixel by the
conjugate of the test pixel with ``onion_mult2D``, and averages.
``onion_mult2D`` is the doubling recursion on arrays of pixels: a product in
dimension ``N`` is four products in dimension ``N / 2``, down to the complex
numbers, so a pixel costs ``N^2`` real multiplications, and the recursion
allocates its operands at every level.

The port differs from the toolbox in two ways, neither of which changes the
work done. It does not round or convert the padded images to integers, a
step that changes the value of the index and that inazumaq2n does not
reproduce (see docs/mathematics.md). And it returns the value of every block
as well as their mean. Its values agree with ``tests/reference.py`` and with
inazumaq2n to rounding error, which the benchmark checks on every case.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def conjugate(onion: Array) -> Array:
    """Negate every coordinate of the last axis after the first."""
    return np.concatenate((onion[..., :1], -onion[..., 1:]), axis=-1)


def onion_mult(onion1: Array, onion2: Array) -> Array:
    """Return the recursive product of two elements, as the toolbox forms it."""
    size = onion1.size
    if size == 1:
        return onion1 * onion2
    half = size // 2
    a, b = onion1[:half], conjugate(onion1[half:])
    c, d = onion2[:half], conjugate(onion2[half:])
    if size == 2:
        return np.concatenate((a * c - d * b, a * d + c * b))
    ris1 = onion_mult(a, c)
    ris2 = onion_mult(d, conjugate(b))
    ris3 = onion_mult(conjugate(a), d)
    ris4 = onion_mult(c, b)
    return np.concatenate((ris1 - ris2, ris3 + ris4))


def onion_mult2D(onion1: Array, onion2: Array) -> Array:  # noqa: N802
    """Return the recursive products of the pixels of two blocks."""
    size = onion1.shape[2]
    if size == 1:
        return onion1 * onion2
    half = size // 2
    a, b = onion1[:, :, :half], conjugate(onion1[:, :, half:])
    c, d = onion2[:, :, :half], conjugate(onion2[:, :, half:])
    if size == 2:
        return np.concatenate((a * c - d * b, a * d + c * b), axis=2)
    ris1 = onion_mult2D(a, c)
    ris2 = onion_mult2D(d, conjugate(b))
    ris3 = onion_mult2D(conjugate(a), d)
    ris4 = onion_mult2D(c, b)
    return np.concatenate((ris1 - ris2, ris3 + ris4), axis=2)


def norm_blocco(x: Array) -> tuple[Array, float, float]:
    """Normalize one band of a block; return it with its mean and deviation."""
    mean = float(np.mean(x))
    deviation = float(np.std(x, ddof=1))
    if deviation == 0:
        deviation = float(np.finfo(np.float64).eps)
    return (x - mean) / deviation + 1, mean, deviation


def onions_quality(dat1: Array, dat2: Array, size: int) -> Array:
    """Return the hypercomplex value of one block, as the toolbox forms it."""
    dat1 = dat1.astype(np.float64)
    dat2 = conjugate(dat2.astype(np.float64))
    bands = dat1.shape[2]
    for i in range(bands):
        dat1[:, :, i], mean, deviation = norm_blocco(dat1[:, :, i])
        scale = 1.0 if mean == 0 else deviation
        if i == 0:
            dat2[:, :, i] = (dat2[:, :, i] - mean) / scale + 1
        else:
            dat2[:, :, i] = -((-dat2[:, :, i] - mean) / scale + 1)
    m1, m2 = np.zeros(bands), np.zeros(bands)
    mod_q1m = mod_q2m = 0.0
    mod_q1, mod_q2 = np.zeros((size, size)), np.zeros((size, size))
    for i in range(bands):
        m1[i] = np.mean(dat1[:, :, i])
        m2[i] = np.mean(dat2[:, :, i])
        mod_q1m += m1[i] ** 2
        mod_q2m += m2[i] ** 2
        mod_q1 += dat1[:, :, i] ** 2
        mod_q2 += dat2[:, :, i] ** 2
    mod_q1m, mod_q2m = math.sqrt(mod_q1m), math.sqrt(mod_q2m)
    mod_q1, mod_q2 = np.sqrt(mod_q1), np.sqrt(mod_q2)
    factor = size**2 / (size**2 - 1)
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    int1 = factor * float(np.mean(mod_q1**2))
    int2 = factor * float(np.mean(mod_q2**2))
    termine3 = int1 + int2 - factor * (mod_q1m**2 + mod_q2m**2)
    mean_bias = 2 * termine2 / termine4
    if termine3 == 0:
        q = np.zeros(bands)
        q[bands - 1] = mean_bias
        return q
    qu = onion_mult2D(dat1, dat2)
    qm = onion_mult(m1, m2)
    qv = np.array([factor * np.mean(qu[:, :, i]) for i in range(bands)])
    return (qv - factor * qm) * mean_bias * (2 / termine3)


def q2n_map(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
) -> Array:
    """Return the value of every block of an image pair."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(test, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    stride = ws if stride is None else stride
    height, width, bands = x.shape
    rows, columns = math.ceil(height / stride), math.ceil(width / stride)
    bottom = (rows - 1) * stride + ws - height
    right = (columns - 1) * stride + ws - width
    if bottom or right:
        pad = ((0, bottom), (0, right), (0, 0))
        x, y = np.pad(x, pad, mode="symmetric"), np.pad(y, pad, mode="symmetric")
    dimension = 1 << (bands - 1).bit_length()
    if dimension != bands:
        pad = ((0, 0), (0, 0), (0, dimension - bands))
        x, y = np.pad(x, pad), np.pad(y, pad)
    values = np.zeros((rows, columns, dimension))
    for j in range(rows):
        for i in range(columns):
            r, c = j * stride, i * stride
            values[j, i] = onions_quality(
                x[r : r + ws, c : c + ws], y[r : r + ws, c : c + ws], ws
            )
    result: Array = np.sqrt(np.sum(values**2, axis=2))
    return result


def q2n(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
) -> float:
    """Return the index: the mean of the values of the blocks."""
    return float(np.mean(q2n_map(reference, test, ws, stride)))

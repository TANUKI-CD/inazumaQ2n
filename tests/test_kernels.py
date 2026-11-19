"""The compiled kernels: instruction sets, sample types and argument checks."""

from __future__ import annotations

import ctypes
import mmap
import sys

import numpy as np
import pytest

from inazumaq2n import _core

LEVELS = _core.simd_levels()
TYPES = [np.float64, np.float32, np.uint8, np.uint16]


def tiles(h, w, side):
    """Return origins that cover the image and its reflection."""
    step = max(1, side - 3)
    return np.array(
        [
            (r, c)
            for r in range(0, 2 * h - side + 1, step)
            for c in range(0, 2 * w - side + 1, step)
        ],
        dtype=np.int32,
    )


def images(rng, h, w, bands, dtype):
    x = rng.uniform(0, 255, (h, w, bands)).astype(dtype)
    y = rng.uniform(0, 255, (h, w, bands)).astype(dtype)
    return x, y


def direct(x, y, origins, side):
    h, w, bands = x.shape
    grams = np.zeros((len(origins), bands, bands))
    stats = np.zeros((len(origins), 4, bands))
    _core.direct(x, y, h, w, bands, origins, side, grams, stats)
    return grams, stats


def prepare(x, y, origins, side):
    h, w, bands = x.shape
    xc = np.zeros((len(origins), side * side, bands))
    yc = np.zeros_like(xc)
    stats = np.zeros((len(origins), 4, bands))
    _core.prepare(x, y, h, w, bands, origins, side, xc, yc, stats)
    return xc, yc, stats


def test_levels():
    assert LEVELS[0] == "baseline"
    assert _core.simd() == LEVELS[-1]
    with pytest.raises(ValueError):
        _core.set_simd("unknown")


@pytest.mark.parametrize("bands", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", TYPES)
def test_direct_is_identical_in_every_instruction_set(rng, bands, dtype):
    for h, w in ((19, 23), (40, 33)):
        x, y = images(rng, h, w, bands, dtype)
        for side in (1, 4, 7, 8, 9, 16, 21, 32):
            origins = tiles(h, w, side)
            results = {}
            for level in LEVELS:
                _core.set_simd(level)
                results[level] = direct(x, y, origins, side)
                # Samples read in place convert exactly
                converted = direct(x.astype(float), y.astype(float), origins, side)
                for a, b in zip(results[level], converted, strict=True):
                    np.testing.assert_array_equal(a, b)
            for level in LEVELS[1:]:
                for a, b in zip(results[level], results[LEVELS[0]], strict=True):
                    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bands", [1, 2, 5, 8, 17])
@pytest.mark.parametrize("dtype", TYPES)
def test_prepare_is_identical_in_every_instruction_set(rng, bands, dtype):
    x, y = images(rng, 19, 23, bands, dtype)
    for side in (1, 5, 8, 13):
        origins = tiles(19, 23, side)
        results = {}
        for level in LEVELS:
            _core.set_simd(level)
            results[level] = prepare(x, y, origins, side)
            converted = prepare(x.astype(float), y.astype(float), origins, side)
            for a, b in zip(results[level], converted, strict=True):
                np.testing.assert_array_equal(a, b)
        for level in LEVELS[1:]:
            for a, b in zip(results[level], results[LEVELS[0]], strict=True):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bands", [1, 2, 3, 4])
def test_direct_moments(rng, bands):
    h, w, side = 21, 26, 11
    x, y = images(rng, h, w, bands, np.float64)
    x += 1e6
    origins = tiles(h, w, side)
    grams, stats = direct(x, y, origins, side)
    xc, yc, general = prepare(x, y, origins, side)
    for k, (r, c) in enumerate(origins):
        rows, cols = np.arange(r, r + side), np.arange(c, c + side)
        rows = np.where(rows < h, rows, 2 * h - 1 - rows)
        cols = np.where(cols < w, cols, 2 * w - 1 - cols)
        bx = x[rows[:, None], cols].astype(np.longdouble).reshape(-1, bands)
        by = y[rows[:, None], cols].astype(np.longdouble).reshape(-1, bands)
        cx, cy = bx - bx.mean(0), by - by.mean(0)
        scale = float(np.abs(cx).max() * np.abs(cy).max() * side * side)
        np.testing.assert_allclose(stats[k, 0], bx.mean(0).astype(float), rtol=1e-14)
        np.testing.assert_allclose(stats[k, 1], by.mean(0).astype(float), rtol=1e-14)
        np.testing.assert_allclose(
            stats[k, 2], (cx * cx).sum(0).astype(float), rtol=1e-9
        )
        np.testing.assert_allclose(
            grams[k], (cx.T @ cy).astype(float), atol=1e-9 * scale, rtol=0
        )
        # The centered samples of the general route give the same moments
        gram = xc[k].T @ yc[k]
        np.testing.assert_allclose(gram, grams[k], atol=1e-9 * scale, rtol=0)
        np.testing.assert_allclose(
            general[k, 2], (cx * cx).sum(0).astype(float), rtol=1e-9
        )
    np.testing.assert_allclose(general[:, :2], stats[:, :2], rtol=1e-14)


@pytest.mark.skipif(sys.platform == "win32", reason="needs mprotect")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("bands", [1, 2, 3, 4, 5])
def test_no_read_past_the_end_of_an_image(rng, dtype, bands):
    """Place the last sample of each image before an inaccessible page."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    page = mmap.PAGESIZE
    h, w = 9, 16 + bands
    size = h * w * bands * np.dtype(dtype).itemsize
    pages = -(-size // page)
    arrays, maps = [], []
    for _ in range(2):
        region = mmap.mmap(-1, (pages + 1) * page)
        address = ctypes.addressof(ctypes.c_char.from_buffer(region))
        # Protection 0 is PROT_NONE: any access to the page faults
        assert libc.mprotect(address + pages * page, page, 0) == 0
        offset = pages * page - size
        array = np.frombuffer(region, dtype=dtype, count=h * w * bands, offset=offset)
        array = array.reshape(h, w, bands)
        array[...] = rng.uniform(0, 255, array.shape).astype(dtype)
        arrays.append(array)
        maps.append((region, address))
    x, y = arrays
    copies = x.copy(), y.copy()
    try:
        for side in (3, 8, 9):
            origins = tiles(h, w, side)
            for level in LEVELS:
                _core.set_simd(level)
                if bands <= 4:
                    guarded = direct(x, y, origins, side)
                    plain = direct(*copies, origins, side)
                else:
                    guarded = prepare(x, y, origins, side)
                    plain = prepare(*copies, origins, side)
                for a, b in zip(guarded, plain, strict=True):
                    np.testing.assert_array_equal(a, b)
    finally:
        del x, y, arrays, array
        for _, address in maps:
            access = mmap.PROT_READ | mmap.PROT_WRITE
            libc.mprotect(address + pages * page, page, access)


def test_argument_checks(rng):
    x, y = images(rng, 16, 16, 3, np.float64)
    origins = np.zeros((1, 2), dtype=np.int32)
    grams, stats = np.zeros((1, 3, 3)), np.zeros((1, 4, 3))
    xc = np.zeros((1, 64, 3))
    _core.direct(x, y, 16, 16, 3, origins, 8, grams, stats)
    with pytest.raises(TypeError):
        _core.direct(x, y.astype(np.float32), 16, 16, 3, origins, 8, grams, stats)
    with pytest.raises(TypeError):
        xi, yi = x.astype(np.int16), y.astype(np.int16)
        _core.direct(xi, yi, 16, 16, 3, origins, 8, grams, stats)
    with pytest.raises(TypeError):
        _core.direct(x, y, 16, 16, 3, origins.astype(np.int64), 8, grams, stats)
    with pytest.raises(TypeError):
        _core.direct(x, y, 16, 16, 3, origins, 8, grams.astype(np.float32), stats)
    with pytest.raises(ValueError):
        _core.direct(x, y, 17, 16, 3, origins, 8, grams, stats)
    with pytest.raises(ValueError):
        _core.direct(x, y, 16, 16, 3, origins, 8, grams[:, :2], stats)
    with pytest.raises(ValueError):
        _core.direct(x, y, 16, 16, 3, origins, 8, grams, stats[:, :3])
    with pytest.raises(ValueError):
        _core.direct(x, y, 16, 16, 5, origins, 8, grams, stats)
    with pytest.raises(ValueError):
        _core.direct(x, y, 16, 16, 3, origins, 0, grams, stats)
    for origin in ((-1, 0), (0, -1), (25, 0), (0, 25)):
        outside = np.array([origin], dtype=np.int32)
        with pytest.raises(ValueError):
            _core.direct(x, y, 16, 16, 3, outside, 8, grams, stats)
        with pytest.raises(ValueError):
            _core.prepare(x, y, 16, 16, 3, outside, 8, xc, xc.copy(), stats)
    with pytest.raises(ValueError):
        _core.prepare(x, y, 16, 16, 3, origins, 8, xc[:, :63], xc.copy(), stats)
    read_only = grams.copy()
    read_only.flags.writeable = False
    with pytest.raises((BufferError, ValueError)):
        _core.direct(x, y, 16, 16, 3, origins, 8, read_only, stats)
    with pytest.raises((BufferError, ValueError)):
        _core.direct(x[:, ::2], y[:, ::2], 16, 8, 3, origins, 8, grams, stats)


def test_finish_argument_checks():
    bands, dimension = 3, 4
    grams, stats = np.ones((2, 3, 3)), np.ones((2, 4, 3))
    ids = np.array([[0], [1]], dtype=np.int32)
    sigma = np.ones((3, 3), dtype=np.int8)
    values, scratch = np.zeros(2), np.zeros(6 * bands + dimension)
    risk = np.zeros(2, dtype=np.uint8)

    def call(**changes):
        arguments = dict(
            grams=grams, stats=stats, ids=ids, nt=1, tile_pixels=64, bands=bands,
            dimension=dimension, sigma=sigma, values=values, scratch=scratch, risk=risk,
        )  # fmt: skip
        arguments.update(changes)
        _core.finish(*arguments.values())

    call()
    with pytest.raises(ValueError):
        call(ids=np.array([[0], [2]], dtype=np.int32))
    with pytest.raises(ValueError):
        call(ids=np.array([[0], [-1]], dtype=np.int32))
    with pytest.raises(ValueError):
        call(dimension=3)
    with pytest.raises(ValueError):
        call(dimension=2)
    with pytest.raises(ValueError):
        call(scratch=scratch[:-1])
    with pytest.raises(ValueError):
        call(tile_pixels=1)
    with pytest.raises(ValueError):
        call(grams=grams[:1])
    with pytest.raises(TypeError):
        call(sigma=sigma.astype(np.int16))
    with pytest.raises(ValueError):
        call(risk=risk[:1])

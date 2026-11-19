"""The index against an independent evaluation of its definition."""

from __future__ import annotations

import doctest

import numpy as np
import pytest
from conftest import pair
from exact import exact_map
from reference import conjugate, multiply
from reference import q2n_map as reference_map

import inazumaq2n
from inazumaq2n import Q2nPlan, _plan, q2n, q2n_map

TOLERANCE = 1e-12


def check(x, y, ws, stride=None, **options):
    """Compare the block map and the index with the reference."""
    expected = reference_map(x, y, ws, stride)
    actual = q2n_map(x, y, ws, stride, **options)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)
    assert abs(q2n(x, y, ws, stride, **options) - expected.mean()) <= TOLERANCE


def check_exact(x, y, ws, stride=None, tolerance=TOLERANCE):
    """Compare the block map with the exact value of the definition."""
    expected = np.array(exact_map(x, y, ws, stride), dtype=np.float64)
    actual = q2n_map(x, y, ws, stride)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("bands", [1, 2, 3, 4, 5, 6, 8, 9, 16, 31, 32])
def test_band_counts(rng, bands):
    x, y = pair(rng, (11, 13, bands))
    check(x, y, 8, backend="numpy")


def test_two_dimensional_image_is_one_band(rng):
    x, y = pair(rng, (19, 23))
    check(x, y, 8)
    assert q2n(x, y, 8) == q2n(x[..., None], y[..., None], 8)


@pytest.mark.parametrize("bands", [1, 3, 4, 5])
@pytest.mark.parametrize("ws, stride", [(8, 4), (16, 8), (16, 6), (8, 7), (12, 3)])
def test_overlap_and_irregular_stride(rng, bands, ws, stride):
    x, y = pair(rng, (19, 23, bands))
    check(x, y, ws, stride)


@pytest.mark.parametrize("bands", [1, 3, 4, 7])
@pytest.mark.parametrize("ws", [2, 3, 7, 8, 9, 17])
def test_block_sizes(rng, bands, ws):
    x, y = pair(rng, (21, 20, bands))
    check(x, y, ws)


def test_scores_above_one_are_preserved(rng):
    a = rng.choice([-1.0, 1.0], (1024, 16))
    b = rng.choice([-1.0, 1.0], (1024, 16))
    ratios = np.linalg.norm(multiply(a, conjugate(b)), axis=1) / 16
    best = int(np.argmax(ratios))
    assert ratios[best] > 1.05
    signs = np.tile(np.array([1.0, -1.0]), 32).reshape(8, 8, 1)
    x, y = 100 + signs * a[best], 100 + signs * b[best]
    actual = q2n(x, y, ws=8)
    assert actual > 1.05
    assert actual == pytest.approx(ratios[best], abs=TOLERANCE)


@pytest.mark.parametrize("bands", [3, 6])
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize("layout", ["c", "fortran", "slice"])
def test_sample_types_and_layouts(rng, bands, dtype, layout):
    x, y = pair(rng, (32, 40, bands))
    a, b = x.astype(dtype), y.astype(dtype)
    if layout == "fortran":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == "slice":
        a, b = a[::2, ::-1], b[::2, ::-1]
    check(a, b, 8)


@pytest.mark.parametrize("bands", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_samples_read_in_place_equal_their_conversion(rng, bands, dtype):
    x, y = pair(rng, (33, 47, bands))
    a = np.clip(x, 0, 255).astype(dtype)
    b = np.clip(y, 0, 255).astype(dtype)
    for ws, stride in ((16, 8), (9, 9)):
        native = q2n_map(a, b, ws, stride, backend="numpy")
        af, bf = a.astype(float), b.astype(float)
        converted = q2n_map(af, bf, ws, stride, backend="numpy")
        np.testing.assert_array_equal(native, converted)


def test_mixed_sample_types(rng):
    x, y = pair(rng, (16, 16, 3))
    a = np.clip(x, 0, 255).astype(np.uint8)
    check(a, y.astype(np.float32), 8)
    check(a.tolist(), y, 8)
    check(a > 100, y, 8)


def test_zero_means_and_conditioning(rng):
    # The floating-point evaluation of tests/reference.py takes the wrong
    # branch or rounds too much on these inputs, so they use the exact value
    half = rng.normal(0, 10, (32, 3))
    zero_mean = np.r_[half, -half].reshape(8, 8, 3)
    x, y = pair(rng, (8, 8, 3))
    families = [
        (zero_mean, 0.8 * zero_mean + 2),
        (x * 1e-160, y * 1e-160),
        (np.full_like(x, 4), y),
    ]
    for a, b in families:
        check_exact(a, b, 8)
    # A mean 1e12 times the variation: the value is that sensitive to rounding
    check_exact(1e12 + x / 100, 1e12 + y / 100, 8, tolerance=1e-7)


@pytest.mark.parametrize("bands", [1, 3, 5])
@pytest.mark.parametrize("side", [8, 12])
def test_zero_mean_in_any_pixel_order(rng, bands, side):
    """Exactly zero means take the branch of the definition, however they are summed."""
    pixels = side * side
    half = rng.normal(0, 10, (pixels // 2, bands))
    x = np.concatenate([half, -half])[rng.permutation(pixels)].reshape(
        side, side, bands
    )
    check_exact(x, 0.8 * x + 2, side)
    if bands == 1:
        # With one band, the value is 0.96 c / (1 + 0.64 c^2), c the deviation
        c = float(np.std(x, ddof=1))
        assert q2n(x, 0.8 * x + 2, side) == pytest.approx(
            0.96 * c / (1 + 0.64 * c * c), abs=TOLERANCE
        )


@pytest.mark.parametrize("bands", [1, 3, 5])
@pytest.mark.parametrize("scale", [2.0**-530, 2.0**515, 2.0**600])
def test_extreme_scales(rng, bands, scale):
    """Squares that underflow or overflow do not change the value."""
    x, y = pair(rng, (16, 16, bands))
    expected = q2n_map(x, y, 8)
    np.testing.assert_allclose(
        q2n_map(x * scale, y * scale, 8), expected, atol=TOLERANCE, rtol=0
    )
    check_exact(x * scale, y * scale, 8)


@pytest.mark.parametrize("bands", [3, 5])
@pytest.mark.parametrize("value", [0.1, 7.0])
def test_constant_band_of_any_value(rng, bands, value):
    """A constant reference band takes the branch of the definition."""
    x, y = pair(rng, (24, 24, bands))
    x[..., 0] = value
    check_exact(x, y, 12)


def test_zero_mean_blocks_under_overlap(rng):
    # Every tile mean is nonzero but the means of the blocks are zero
    half = rng.choice([-1.0, 1.0], (8, 8, 3)) * 10
    tile = np.block([[half, -half], [-half, half]])
    x = np.tile(tile, (3, 3, 1))
    check(x, 0.8 * x + 2, 16, 8)
    check_exact(x, 0.8 * x + 2, 16, 8)


def test_identical_images(rng):
    x, _ = pair(rng, (32, 32, 5))
    assert q2n(x, x, 8) == pytest.approx(1.0, abs=TOLERANCE)
    # The zero-mean branch of the definition scores an identical pair below one
    a = np.tile(np.array([2.0, -2.0]), 32).reshape(8, 8)
    check(a, a, 8)
    assert q2n(a, a, ws=8) < 0.9


@pytest.mark.parametrize("value", [0, 1, 255])
def test_constant_reference(value):
    a = np.full((16, 16, 3), value, dtype=np.uint8)
    b = np.clip(a.astype(float) + 2, 0, 255).astype(np.uint8)
    check(a, b, 16)
    check_exact(a, b, 16)


def test_map_is_the_index_by_block(rng):
    x, y = pair(rng, (40, 70, 4))
    block_map = q2n_map(x, y, 16, 8)
    assert block_map.shape == (5, 9)
    assert block_map.dtype == np.float64
    assert q2n(x, y, 16, 8) == float(np.mean(block_map))


def test_matrix_product_route_agrees_with_direct_kernels(rng, monkeypatch):
    for bands in (1, 2, 3, 4):
        x, y = pair(rng, (40, 33, bands))
        direct = q2n_map(x, y, 16, 8, backend="numpy")
        with monkeypatch.context() as patch:
            patch.setattr(_plan, "_DIRECT_BANDS", 0)
            with Q2nPlan(x.shape, 16, 8, backend="numpy") as plan:
                assert not plan._direct
                general = plan.map(x, y)
        np.testing.assert_allclose(direct, general, atol=TOLERANCE, rtol=0)
        np.testing.assert_allclose(general, reference_map(x, y, 16, 8), atol=TOLERANCE)


def test_invalid_images(rng):
    x, y = pair(rng, (32, 32, 3))
    with pytest.raises(TypeError):
        q2n(x.astype(complex), y)
    with pytest.raises(TypeError):
        q2n(x, y.astype(str))
    with pytest.raises(ValueError):
        q2n(x, y[:-1])
    with pytest.raises(ValueError):
        q2n(x[0, 0], y[0, 0])
    with pytest.raises(ValueError):
        q2n(x[:4, :4], y[:4, :4])


@pytest.mark.parametrize(
    "options",
    [{"ws": 1}, {"ws": 3.5}, {"ws": True}, {"stride": 0}, {"workers": 0}],
)
def test_invalid_options(rng, options):
    x, y = pair(rng, (32, 32, 3))
    with pytest.raises(ValueError):
        q2n(x, y, **options)
    with pytest.raises(ValueError):
        q2n(x, y, backend="unknown")


def test_stride_that_leaves_pixels_uncovered(rng):
    x, y = pair(rng, (100, 100))
    with pytest.raises(ValueError):
        q2n(x, y, ws=8, stride=64)


def test_docstring_examples():
    for module in (inazumaq2n, _plan):
        assert doctest.testmod(module).failed == 0

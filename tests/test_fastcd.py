"""The fused sums of fastCD: agreement, selection and argument checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import pair
from exact import exact_map
from reference import q2n_map as reference_map

from inazumaq2n import Q2nPlan, _auto, _core, _plan, q2n, q2n_map, runtime_info

TOLERANCE = 1e-12
LEVELS = _core.simd_levels()
VENDOR_TOOL = Path(__file__).resolve().parents[1] / "tools" / "vendor_fastcd.py"


def fused(x, y, ws, stride=None, **options):
    return q2n_map(x, y, ws, stride, algorithm="fastcd", **options)


def moments(x, y, ws, stride=None, **options):
    return q2n_map(x, y, ws, stride, algorithm="moments", backend="numpy", **options)


@pytest.mark.parametrize("bands", [1, 2, 3, 4, 5, 6, 8, 9, 16, 31, 32])
def test_band_counts_against_the_definition(rng, bands):
    x, y = pair(rng, (11, 13, bands))
    expected = reference_map(x, y, 8)
    np.testing.assert_allclose(fused(x, y, 8), expected, atol=TOLERANCE, rtol=0)
    assert abs(q2n(x, y, 8, algorithm="fastcd") - expected.mean()) <= TOLERANCE


@pytest.mark.parametrize("bands", [1, 3, 5, 8, 12])
@pytest.mark.parametrize("ws, stride", [(8, 4), (16, 8), (16, 6), (8, 7), (12, 3)])
def test_overlap_and_irregular_stride(rng, bands, ws, stride):
    x, y = pair(rng, (19, 23, bands))
    expected = reference_map(x, y, ws, stride)
    np.testing.assert_allclose(
        fused(x, y, ws, stride), expected, atol=TOLERANCE, rtol=0
    )


@pytest.mark.parametrize("bands", [7, 33, 40, 64, 71, 100, 129])
def test_many_bands_against_the_moments(rng, bands):
    # Band counts on both sides of every width of the pixel kernels
    x, y = pair(rng, (37, 29, bands))
    for ws, stride in ((16, 16), (16, 8), (12, 5), (34, 34)):
        np.testing.assert_allclose(
            fused(x, y, ws, stride), moments(x, y, ws, stride), atol=TOLERANCE, rtol=0
        )


@pytest.mark.parametrize("bands", [1, 5, 8, 13, 32, 37, 70])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint8, np.uint16])
def test_identical_in_every_instruction_set_and_sample_type(rng, bands, dtype):
    x, y = pair(rng, (33, 26, bands))
    a = np.clip(x, 0, 255).astype(dtype)
    b = np.clip(y, 0, 255).astype(dtype)
    for ws, stride in ((16, 8), (9, 9), (30, 30)):
        results = []
        for level in LEVELS:
            _core.set_simd(level)
            results.append(fused(a, b, ws, stride))
            # Samples read in place convert exactly
            converted = fused(a.astype(float), b.astype(float), ws, stride)
            np.testing.assert_array_equal(results[-1], converted)
        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])


@pytest.mark.parametrize("bands", [1, 5, 8, 13, 33])
@pytest.mark.parametrize(
    "shape, ws, stride",
    [((40, 33), 16, 8), ((37, 45), 16, 4), ((30, 30), 12, 6), ((21, 64), 8, 2)],
)
def test_shared_tiles(rng, monkeypatch, bands, shape, ws, stride):
    x, y = pair(rng, (*shape, bands))
    with Q2nPlan(x.shape, ws, stride, algorithm="fastcd") as plan:
        assert plan._shared == ws // stride
        expected = plan.map(x, y)
    np.testing.assert_allclose(
        expected, moments(x, y, ws, stride), atol=TOLERANCE, rtol=0
    )
    # Neither the workers nor the width of a strip of blocks change a value
    for workers in (2, 5):
        np.testing.assert_array_equal(
            fused(x, y, ws, stride, workers=workers), expected
        )
    for budget in (1, 1 << 16, 1 << 30):
        monkeypatch.setattr(_plan, "_RECORD_BYTES", budget)
        np.testing.assert_array_equal(fused(x, y, ws, stride), expected)
    a = np.clip(x, 0, 255).astype(np.uint8)
    b = np.clip(y, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(
        fused(a, b, ws, stride), fused(a.astype(float), b.astype(float), ws, stride)
    )


def test_blocks_that_share_no_tiles_are_summed_alone(rng):
    x, y = pair(rng, (30, 31, 9))
    with Q2nPlan(x.shape, 12, 5, algorithm="fastcd") as plan:
        assert plan._shared == 0
        np.testing.assert_allclose(
            plan.map(x, y), moments(x, y, 12, 5), atol=TOLERANCE, rtol=0
        )


def test_statistics_equal_those_of_the_moments(rng):
    # The same guards mark the same blocks, from bit-identical statistics
    x, y = pair(rng, (64, 48, 13))
    x[:16, :16] += 1e6
    x[16:32, :16, 3] = 7.0
    half = rng.normal(0, 10, (128, 13))
    x[32:48, :16] = np.r_[half, -half].reshape(16, 16, 13)
    with (
        Q2nPlan(x.shape, 16, algorithm="fastcd") as a,
        Q2nPlan(x.shape, 16, algorithm="moments", backend="numpy") as b,
    ):
        a._run(x, y, None)
        b._run(x, y, None)
        np.testing.assert_array_equal(a._risk, b._risk)
        assert 0 < a._risk.sum() < a._risk.size
        marked = a._risk.astype(bool)
        # Marked blocks are evaluated from their samples by both
        np.testing.assert_array_equal(a._values[marked], b._values[marked])
        np.testing.assert_allclose(a._values, b._values, atol=TOLERANCE, rtol=0)
    # Blocks merged from tiles: the same merged statistics, the same marks
    with (
        Q2nPlan(x.shape, 16, 8, algorithm="fastcd") as a,
        Q2nPlan(x.shape, 16, 8, algorithm="moments", backend="numpy") as b,
    ):
        a._run(x, y, None)
        b._run(x, y, None)
        np.testing.assert_array_equal(a._risk, b._risk)
        assert 0 < a._risk.sum() < a._risk.size
        marked = a._risk.astype(bool)
        np.testing.assert_array_equal(a._values[marked], b._values[marked])
        np.testing.assert_allclose(a._values, b._values, atol=TOLERANCE, rtol=0)


def test_zero_means_and_conditioning(rng):
    half = rng.normal(0, 10, (32, 5))
    zero_mean = np.r_[half, -half].reshape(8, 8, 5)
    x, y = pair(rng, (8, 8, 5))
    constant = x.copy()
    constant[..., 0] = 7.0
    families = [
        (zero_mean, 0.8 * zero_mean + 2),
        (x * 1e-160, y * 1e-160),
        (x * 1e150, y * 1e150),
        (x + 1e9, y + 1e9),
        (constant, y),
    ]
    for a, b in families:
        expected = np.array(exact_map(a, b, 8), dtype=np.float64)
        np.testing.assert_allclose(fused(a, b, 8), expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            fused(a, b, 8), moments(a, b, 8), atol=TOLERANCE, rtol=0
        )


def test_special_values(rng):
    x, y = pair(rng, (16, 16, 9))
    assert q2n(x, x, 8, algorithm="fastcd") == pytest.approx(1.0, abs=TOLERANCE)
    flat = np.full_like(x, 3.0)
    np.testing.assert_array_equal(fused(flat, flat, 8), moments(flat, flat, 8))
    for value in (np.nan, np.inf):
        bad = y.copy()
        bad[2, 3, 4] = value
        with np.errstate(invalid="ignore"):
            for result in (fused(x, bad, 8), fused(bad, x, 8)):
                assert np.isnan(result[0, 0])
                assert np.all(np.isfinite(result.ravel()[1:]))


def test_workers_and_reuse(rng):
    x, y = pair(rng, (45, 52, 17))
    expected = fused(x, y, 16, 8)
    for workers in (2, 3, 64):
        np.testing.assert_array_equal(fused(x, y, 16, 8, workers=workers), expected)
    with Q2nPlan(x.shape, 16, 8, algorithm="fastcd", workers=3) as plan:
        assert plan.algorithm == "fastcd"
        assert "algorithm='fastcd'" in repr(plan)
        for _ in range(3):
            np.testing.assert_array_equal(plan.map(x, y), expected)
        assert plan.score(x, y) == float(np.mean(expected))
        profile: dict = {}
        plan._run(x, y, profile)
        assert profile["algorithm"] == "fastcd"
        assert profile["kernel"] == "fused sum, shared tiles"
    with pytest.raises(RuntimeError):
        plan.score(x, y)


def test_algorithm_argument():
    with pytest.raises(ValueError, match="algorithm"):
        Q2nPlan((16, 16, 5), algorithm="quasilinear")
    with Q2nPlan((16, 16, 5), algorithm="moments") as plan:
        assert plan.algorithm == "moments"
    with Q2nPlan((16, 16, 3), ws=8) as plan:
        # At most four bands always take the kernels without a matrix product
        assert plan.algorithm == "moments"


def test_runtime_info_reports_fastcd():
    info = runtime_info()
    assert info["fastcd"] == "1.5.0"
    assert info["fastcd_kernel"] in ("avx512", "avx2", "portable")
    assert (info["fastcd"], info["fastcd_kernel"]) == _core.fastcd()


def test_auto_follows_the_measured_table():
    kernel = _core.fastcd()[1]
    for shape, ws, stride in (
        ((64, 64, 3), 32, 32),
        ((64, 64, 8), 32, 32),
        ((64, 64, 64), 32, 32),
        ((64, 64, 40), 32, 32),
        ((64, 64, 64), 32, 16),
        ((64, 64, 200), 32, 24),
    ):
        shared = (ws // stride) ** 2 if ws % stride == 0 else 1
        dimension = 1 << (shape[2] - 1).bit_length()
        with Q2nPlan(shape, ws, stride) as plan:
            assert plan.algorithm == _auto.choose(shape[2], dimension, shared, kernel)


def test_choice_of_auto():
    choose = _auto.choose
    for kernel in ("avx512", "avx2", "portable", "unknown"):
        # The kernels without a matrix product are never displaced
        for bands in (1, 2, 3, 4):
            for shared in (1, 4, 16):
                assert (
                    choose(bands, 4 if bands > 2 else bands, shared, kernel)
                    == "moments"
                )
        # More shared tiles never favour the fused sums, nor do fewer bands
        for dimension in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            for shared in (1, 4, 9, 16):
                full = choose(dimension, dimension, shared, kernel)
                if choose(dimension, dimension, 4 * shared, kernel) == "fastcd":
                    assert full == "fastcd"
                if choose(dimension // 2 + 1, dimension, shared, kernel) == "fastcd":
                    assert full == "fastcd"
    assert choose(1024, 1024, 1, "portable") == "moments"
    assert choose(4096, 4096, 1, "avx512") == "fastcd"


def test_fused_argument_checks():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 2, (16, 16, 3))
    y = x.copy()
    origins = np.array([[0, 0], [8, 8]], dtype=np.int32)
    work = np.zeros(_core.fused_work(64, 4))
    values = np.zeros(2)
    risk = np.zeros(2, dtype=np.uint8)

    def call(**changes):
        arguments = dict(
            x=x, y=y, height=16, width=16, bands=3, dimension=4, origins=origins,
            side=8, work=work, values=values, risk=risk,
        )  # fmt: skip
        arguments.update(changes)
        _core.fused(*arguments.values())

    call()
    assert np.all(np.abs(values - 1.0) < TOLERANCE) and not risk.any()
    with pytest.raises(ValueError):
        call(dimension=3)
    with pytest.raises(ValueError):
        call(dimension=2)
    with pytest.raises(ValueError):
        call(side=1)
    with pytest.raises(ValueError):
        call(work=work[:-1])
    with pytest.raises(ValueError):
        call(values=values[:1])
    with pytest.raises(ValueError):
        call(risk=risk[:1])
    with pytest.raises(ValueError):
        call(bands=4)
    with pytest.raises(TypeError):
        call(y=y.astype(np.float32))
    with pytest.raises(TypeError):
        call(risk=risk.astype(np.int16))
    for origin in ((-1, 0), (0, -1), (25, 0), (0, 25)):
        with pytest.raises(ValueError):
            call(origins=np.array([origin], dtype=np.int32))
    read_only = values.copy()
    read_only.flags.writeable = False
    with pytest.raises((BufferError, ValueError)):
        call(values=read_only)
    with pytest.raises(ValueError):
        _core.fused_work(64, 3)
    with pytest.raises(ValueError):
        _core.fused_work(0, 4)
    assert _core.fused_work(64, 8) > 2 * 64 * 8


def test_vendored_fastcd_matches_its_record():
    if not VENDOR_TOOL.is_file():
        pytest.skip("the tools are not beside the tests")
    result = subprocess.run(
        [sys.executable, str(VENDOR_TOOL), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert _core.fastcd()[0] in result.stdout


def test_fused_tiles_argument_checks():
    rng = np.random.default_rng(6)
    x = rng.uniform(1, 2, (16, 16, 3))
    y = x.copy()
    work = np.zeros(_core.fused_tiles_work(16, 4, 2, 3))
    values = np.zeros(9)
    risk = np.zeros(9, dtype=np.uint8)

    def call(**changes):
        arguments = dict(
            x=x, y=y, height=16, width=16, bands=3, dimension=4, stride=4, q=2,
            row0=0, rows=3, col0=0, cols=3, pitch=3, work=work, values=values,
            risk=risk,
        )  # fmt: skip
        arguments.update(changes)
        _core.fused_tiles(*arguments.values())

    call()
    assert np.all(np.abs(values - 1.0) < TOLERANCE) and not risk.any()
    # A rectangle inside the grid writes only its own blocks
    values[:] = -1.0
    call(row0=1, rows=1, col0=1, cols=2)
    assert np.all(values[[0, 1, 2, 3, 6, 7, 8]] == -1.0)
    assert np.all(np.abs(values[[4, 5]] - 1.0) < TOLERANCE)
    for changes in (
        dict(dimension=3),
        dict(q=1),
        dict(stride=0),
        dict(rows=0),
        dict(cols=4),
        dict(pitch=2),
        dict(row0=-1),
        dict(rows=8),
        dict(cols=8, pitch=8),
        dict(work=work[:-1]),
        dict(values=values[:8]),
        dict(risk=risk[:8]),
        dict(bands=4),
    ):
        with pytest.raises(ValueError):
            call(**changes)
    with pytest.raises(TypeError):
        call(y=y.astype(np.float32))
    with pytest.raises(ValueError):
        _core.fused_tiles_work(16, 3, 2, 3)
    with pytest.raises(ValueError):
        _core.fused_tiles_work(16, 4, 1, 3)
    assert _core.fused_tiles_work(16, 4, 2, 4) > _core.fused_tiles_work(16, 4, 2, 3)

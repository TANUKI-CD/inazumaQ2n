"""Plans: reuse, lifetime, workers and attributes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import pair
from reference import q2n as reference_q2n

from inazumaq2n import Q2nPlan, _plan, q2n, q2n_map


def test_attributes():
    with Q2nPlan((70, 95, 5), ws=16, stride=8, workers=2, backend="numpy") as plan:
        assert plan.shape == (70, 95, 5)
        assert (plan.ws, plan.stride) == (16, 8)
        assert (plan.bands, plan.dimension) == (5, 8)
        assert plan.block_shape == (9, 12)
        assert (plan.workers, plan.backend) == (2, "numpy")
        assert "Q2nPlan((70, 95, 5), ws=16, stride=8" in repr(plan)
    assert Q2nPlan((8, 8), ws=8).bands == 1
    assert Q2nPlan((8, 8), ws=4).stride == 4
    assert Q2nPlan([8, 8, 31], ws=np.int64(4)).dimension == 32


@pytest.mark.parametrize("bands", [3, 8])
def test_reuse_gives_the_values_of_single_calls(rng, bands):
    x, y = pair(rng, (32, 40, bands))
    with Q2nPlan(x.shape, ws=8) as plan:
        first = plan.score(x, y)
        block_map = plan.map(x, y)
        saved = block_map.copy()
        second = plan.score(x, x)
        assert first == q2n(x, y, ws=8)
        assert second > first
        # A map is a copy, not a view of the work arrays of the plan
        np.testing.assert_array_equal(block_map, saved)
        np.testing.assert_array_equal(block_map, q2n_map(x, y, ws=8))
        assert plan.score(x, y) == first


def test_closed_plan(rng):
    x, y = pair(rng, (16, 16, 3))
    plan = Q2nPlan(x.shape, ws=8)
    plan.score(x, y)
    plan.close()
    plan.close()
    with pytest.raises(RuntimeError):
        plan.score(x, y)
    with pytest.raises(RuntimeError):
        plan.map(x, y)
    with pytest.raises(RuntimeError), plan:
        pass


def test_shape_must_match(rng):
    x, y = pair(rng, (16, 16, 3))
    with Q2nPlan((16, 16, 4), ws=8) as plan:
        with pytest.raises(ValueError):
            plan.score(x, y)
    with Q2nPlan((16, 16), ws=8) as plan:
        with pytest.raises(ValueError):
            plan.score(x, y)


@pytest.mark.parametrize("bands", [3, 16])
@pytest.mark.parametrize("workers", [2, 3])
def test_workers_and_small_workspace(rng, monkeypatch, bands, workers):
    x, y = pair(rng, (64, 64, bands))
    expected = q2n_map(x, y, 16, 8, backend="numpy")
    assert abs(expected.mean() - reference_q2n(x, y, 16, 8)) < 1e-12
    # One tile per chunk exercises every chunk boundary
    monkeypatch.setattr(_plan, "_WORKSPACE_BYTES", 1)
    with Q2nPlan(x.shape, 16, 8, backend="numpy", workers=workers) as plan:
        np.testing.assert_array_equal(plan.map(x, y), expected)
        assert abs(plan.score(x, x) - 1) < 1e-12
        np.testing.assert_array_equal(plan.map(x, y), expected)


def test_calls_from_several_threads_are_serialized(rng):
    x, y = pair(rng, (32, 40, 8))
    with Q2nPlan(x.shape, ws=8) as plan:
        first, second = plan.score(x, y), plan.score(x, x)
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda p: plan.score(*p), [(x, y), (x, x)] * 8))
        assert values == [first, second] * 8


@pytest.mark.parametrize(
    "shape", [(0, 32), (32, 0), (32, 32, 0), (32,), (32, 32, 3, 1), (32.0, 32)]
)
def test_invalid_shapes(shape):
    with pytest.raises(ValueError):
        Q2nPlan(shape)


def test_padding_larger_than_the_image():
    with pytest.raises(ValueError):
        Q2nPlan((4, 4), ws=32)
    # A padding equal to the image is the largest that reflection defines
    assert Q2nPlan((4, 4), ws=8).block_shape == (1, 1)

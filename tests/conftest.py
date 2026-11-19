"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from inazumaq2n import _core


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator with a fixed seed."""
    return np.random.default_rng(181739)


@pytest.fixture(autouse=True)
def restore_simd() -> Iterator[None]:
    """Restore the instruction set a test may have changed."""
    level = _core.simd()
    yield
    _core.set_simd(level)


def pair(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reference image and a noisy copy of it."""
    x = rng.uniform(20, 220, shape)
    return x, x + rng.normal(0, 20, shape)

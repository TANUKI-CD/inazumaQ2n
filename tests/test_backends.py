"""The matrix-product libraries and the report of what was selected."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
from conftest import pair

import inazumaq2n
from inazumaq2n import Q2nPlan, _blas, _core, q2n, runtime_info


def test_runtime_info():
    info = runtime_info()
    assert info["version"] == inazumaq2n.__version__
    assert info["simd"] == _core.simd()
    assert info["simd"] in ("baseline", "neon", "avx2", "avx512")
    assert info["direct_kernels"] == _core.direct_is_fast()
    assert info["backend"] in ("numpy", "mkl")
    assert isinstance(info["backend_version"], str)
    assert info["numpy"] == np.__version__


def test_auto_without_mkl(rng, monkeypatch):
    def unavailable():
        raise RuntimeError("not installed")

    monkeypatch.setattr(_blas, "_mkl", None)
    monkeypatch.setattr(_blas, "_mkl_failure", None)
    monkeypatch.setattr(_blas, "Mkl", unavailable)
    x, y = pair(rng, (16, 16, 5))
    with Q2nPlan(x.shape, ws=8) as plan:
        assert plan.backend == "numpy"
        assert plan.score(x, y) == q2n(x, y, ws=8, backend="numpy")
    assert runtime_info()["backend"] == "numpy"
    with pytest.raises(RuntimeError, match="not installed"):
        Q2nPlan(x.shape, ws=8, backend="mkl")


def test_unknown_backend():
    with pytest.raises(ValueError):
        Q2nPlan((16, 16, 5), backend="blis")


def test_mkl_agrees_and_restores_its_thread_setting(rng):
    try:
        plan = Q2nPlan((32, 32, 5), backend="mkl")
    except RuntimeError as error:
        pytest.skip(str(error))
    with plan:
        assert plan.backend == "mkl"
        assert plan._mkl is not None
        getter = plan._mkl.library.MKL_Get_Max_Threads
        getter.restype = ctypes.c_int
        before = getter()
        x, y = pair(rng, plan.shape)
        assert abs(plan.score(x, y) - q2n(x, y, backend="numpy")) < 1e-12
        assert getter() == before

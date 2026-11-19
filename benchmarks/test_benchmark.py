"""The benchmark driver and the pyfastcd baseline on small cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("threadpoolctl")
pytest.importorskip("pyfastcd")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import baseline  # noqa: E402
import benchmark  # noqa: E402
import reference  # noqa: E402

from inazumaq2n import _core  # noqa: E402


def test_calls_protocol():
    cases = [
        ("RGB", (40, 48, 3), 16, 8, "uint8"),
        ("six bands", (40, 48, 6), 16, 16, "float64"),
    ]
    result = benchmark.run_calls(3, backend="numpy", cases=cases)
    assert [row["case"] for row in result["rows"]] == ["RGB", "six bands"]
    assert result["baselines"]["algorithms"] == benchmark.BASELINES
    variants = {"one-shot", "plan", *benchmark.BASELINES}
    for row in result["rows"]:
        assert set(row["median_seconds"]) == variants
        assert all(len(times) == 3 for times in row["seconds"].values())
        assert row["max_abs_difference"] <= 1e-12
        assert row["ratio_to_plan"]["plan"] == 1.0
        assert row["profile"]["tiles"] > 0
    selected = benchmark.run_calls(
        1, backend="numpy", baselines=False, cases=cases, selected=["six"]
    )
    assert len(selected["rows"]) == 1
    assert set(selected["rows"][0]["median_seconds"]) == {"one-shot", "plan"}
    assert selected["baselines"] is None


@pytest.mark.parametrize("algorithm", baseline.ALGORITHMS)
@pytest.mark.parametrize("bands", [1, 2, 3, 5, 8])
def test_baseline_evaluates_the_definition(algorithm, bands):
    rng = np.random.default_rng(bands)
    x = rng.uniform(20, 220, (19, 23, bands))
    y = x + rng.normal(0, 20, x.shape)
    for ws, stride in ((8, 8), (8, 4), (16, 6)):
        expected = reference.q2n_map(x, y, ws, stride)
        actual = baseline.q2n_map(x, y, ws, stride, algorithm=algorithm)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert baseline.q2n(x[..., 0], y[..., 0], 8, algorithm=algorithm) == pytest.approx(
        reference.q2n(x[..., 0], y[..., 0], 8), abs=1e-12
    )


def test_baseline_batches_do_not_change_values(monkeypatch):
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 255, (40, 40, 6))
    y = x + rng.normal(0, 20, x.shape)
    whole = baseline.q2n_map(x, y, 8)
    monkeypatch.setattr(baseline, "WORKSPACE_BYTES", 1)
    np.testing.assert_array_equal(baseline.q2n_map(x, y, 8), whole)
    with pytest.raises(ValueError):
        baseline.q2n_map(x, y, 8, algorithm="cariow_cariowa")
    with pytest.raises(ValueError):
        baseline.q2n_map(x, y[:-1], 8)


def test_kernels_protocol():
    level = _core.simd()
    cases = [(2, 40, 48, 16, 8, "uint8"), (4, 33, 37, 8, 8, "float64")]
    result = benchmark.run_kernels(2, cases=cases)
    assert _core.simd() == level
    levels = [name for name in _core.simd_levels() if name != "baseline"]
    assert set(result["summary"]) == {f"direct, {name}" for name in levels}
    for row in result["rows"]:
        assert "matrix product, numpy" in row["median_seconds"]
        assert row["max_abs_difference"] <= 1e-12


def test_command_line(tmp_path):
    output = tmp_path / "result.json"
    arguments = ["--rounds", "1", "--backend", "numpy", "--cases", "gray"]
    assert benchmark.main([*arguments, "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["protocol"] == "calls"
    assert len(result["rows"]) == 1
    assert result["environment"]["runtime"]["simd"] == _core.simd()
    assert "_plan.py" in result["environment"]["source_sha256"]

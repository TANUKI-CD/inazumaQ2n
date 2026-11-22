"""The benchmark driver, the pyfastcd baseline and the toolbox port on small cases."""

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
import toolbox  # noqa: E402

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


@pytest.mark.parametrize("bands", [1, 2, 3, 5, 8, 11, 16])
def test_toolbox_evaluates_the_definition(bands):
    rng = np.random.default_rng(100 + bands)
    x = rng.uniform(20, 220, (19, 23, bands))
    y = x + rng.normal(0, 20, x.shape)
    for ws, stride in ((8, 8), (8, 4), (16, 6)):
        expected = reference.q2n_map(x, y, ws, stride)
        actual = toolbox.q2n_map(x, y, ws, stride)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert toolbox.q2n(x[..., 0], y[..., 0], 8) == pytest.approx(
        reference.q2n(x[..., 0], y[..., 0], 8), abs=1e-12
    )


def test_toolbox_branches_of_the_definition():
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 255, (8, 8, 4))
    x[..., 1] = 5.0
    half = rng.normal(0, 10, 32)
    x[..., 2] = np.r_[half, -half].reshape(8, 8)
    y = x + rng.normal(0, 2, x.shape)
    assert toolbox.q2n(x, y, 8) == pytest.approx(reference.q2n(x, y, 8), abs=1e-12)
    flat = np.full((8, 8, 3), 2.0)
    assert toolbox.q2n(flat, flat, 8) == pytest.approx(reference.q2n(flat, flat, 8))
    # The recursion on single elements is the recursion on blocks of one pixel
    a, b = rng.normal(size=16), rng.normal(size=16)
    np.testing.assert_array_equal(
        toolbox.onion_mult(a, b),
        toolbox.onion_mult2D(a.reshape(1, 1, 16), b.reshape(1, 1, 16)).ravel(),
    )


def test_algorithm_cases():
    cases = benchmark.algorithm_cases()
    assert {case["sweep"] for case in cases} == set(benchmark.SWEEPS)
    assert all(case["dimension"] >= case["bands"] for case in cases)
    assert all(case["toolbox"] == (case["sweep"] == "dimension") for case in cases)
    keys = [
        (c["sweep"], c["bands"], c["ws"], c["stride"], c["sample_type"], c["workers"])
        for c in cases
    ]
    assert len(set(keys)) == len(keys)
    small = benchmark.algorithm_cases(("dimension", "bands"), max_dimension=16)
    assert max(case["dimension"] for case in small) == 16
    assert [c["bands"] for c in small if c["sweep"] == "dimension"] == [2, 4, 8, 16]


def test_algorithms_protocol(tmp_path):
    def case(sweep, bands, ws, stride, toolbox_too):
        return {
            "sweep": sweep,
            "bands": bands,
            "dimension": 1 << (bands - 1).bit_length(),
            "shape": (40, 48, bands),
            "ws": ws,
            "stride": stride,
            "sample_type": "float64",
            "workers": None,
            "toolbox": toolbox_too,
        }

    cases = [case("dimension", 8, 16, 16, True), case("overlap", 6, 16, 8, False)]
    result = benchmark.run_algorithms(
        3,
        backend="numpy",
        toolbox_rounds=2,
        sweeps=("dimension", "overlap"),
        cases=cases,
    )
    first, second = result["rows"]
    assert set(first["median_seconds"]) == {"moments", "fastcd", "auto", "toolbox"}
    assert set(second["median_seconds"]) == {"moments", "fastcd", "auto"}
    assert len(first["seconds"]["toolbox"]) == 2
    assert len(first["seconds"]["fastcd"]) == 3
    assert first["tiles_per_block"] == 1 and second["tiles_per_block"] == 4
    for row in result["rows"]:
        assert row["max_abs_difference"] <= 1e-12
        assert row["auto"] in ("moments", "fastcd")
        assert row["fastest"] in ("moments", "fastcd")
        assert row["auto_regret"] >= 1.0
    assert first["toolbox_abs_difference"] <= 1e-10
    assert second["toolbox_over"] is None
    report = benchmark.algorithms_report(result)
    assert "## Dimension" in report and "## Overlapping blocks" in report
    assert "| 8 | 8 | 40 x 48 | 16 / 16 |" in report
    without = benchmark.run_algorithms(1, backend="numpy", toolbox=False, cases=cases)
    assert "toolbox" not in without["rows"][0]["median_seconds"]
    assert "toolbox" not in benchmark.algorithms_report(without).split("\n## ", 1)[1]
    output = tmp_path / "algorithms.json"
    arguments = ["--protocol", "algorithms", "--rounds", "1", "--backend", "numpy"]
    arguments += ["--sweeps", "blocks", "--max-dimension", "8", "--no-toolbox"]
    assert benchmark.main([*arguments, "--output", str(output)]) == 0
    written = json.loads(output.read_text())
    assert written["protocol"] == "algorithms" and written["sweeps"] == ["blocks"]
    assert output.with_suffix(".md").read_text().startswith("# Algorithms")


def test_toolboxes_protocol(tmp_path):
    import toolboxes

    result = benchmark.run_toolboxes(
        2, backend="numpy", toolbox_rounds=2, cases=[(4, 64), (6, 64)]
    )
    assert result["protocol"] == "toolboxes"
    assert set(result["sources"]) == set(toolboxes.SOURCES)
    ran = [name for name in toolboxes.SOURCES if name not in result["not_run"]]
    for row in result["rows"]:
        assert {"inazumaQ2n", "port"} <= set(row["median_seconds"])
        assert row["largest_block_difference"]["port"] <= 1e-12
        assert row["ratio_to_inazumaQ2n"]["inazumaQ2n"] == 1.0
        assert len(row["seconds"]["port"]) == 2
        if "matlab" in ran:
            assert row["largest_block_difference"]["matlab"] <= 1e-12
            assert len(row["seconds"]["matlab"]) == 2
    report = benchmark.toolboxes_report(result)
    assert report.startswith("# inazumaQ2n and the published toolboxes")
    assert "| 4 | 64 x 64 | moments |" in report and "| 8 | 64 x 64 |" in report
    for name in result["not_run"]:
        assert f"Not run: {name}" in report
    # Without an implementation that can run, the port is still compared
    alone = benchmark.run_toolboxes(1, backend="numpy", names=[], cases=[(2, 64)])
    assert set(alone["rows"][0]["median_seconds"]) == {"inazumaQ2n", "port"}


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

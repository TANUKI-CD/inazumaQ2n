"""inazumaq2n against the code of the published toolboxes.

The comparisons run the q2n of the MATLAB pansharpening toolbox under GNU
Octave, from files that ``benchmarks/toolboxes.py fetch`` downloads at a
pinned commit. They are skipped where Octave or the files are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import toolbox  # noqa: E402
import toolboxes  # noqa: E402
from exact import exact_map  # noqa: E402

from inazumaq2n import q2n, q2n_map  # noqa: E402

# The toolbox and the package evaluate one definition in different orders
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def matlab():
    """The q2n of the MATLAB toolbox, or a skip where it cannot run."""
    reason = toolboxes.unavailable("matlab")
    if reason is not None:
        pytest.skip(reason)
    return toolboxes.matlab_q2n


def integer_pair(seed, shape, top=2048, noise=60):
    """Return a reference of integer samples and a noisy integer copy of it."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, top, shape).astype(np.float64)
    y = np.clip(x + np.round(rng.normal(0, noise, shape)), 0, 65535)
    return x, y


def test_sources_are_pinned():
    for name, source in toolboxes.SOURCES.items():
        assert len(source["commit"]) == 40 and int(source["commit"], 16) >= 0
        assert source["license"] and source["files"]
        for file, digest in source["files"].items():
            assert len(digest) == 64 and int(digest, 16) >= 0
            address = toolboxes.url(name, file)
            assert address.startswith("https://raw.githubusercontent.com/")
            assert source["commit"] in address and " " not in address
        assert name in toolboxes.ADAPTERS


def test_missing_and_altered_files_are_never_run(tmp_path):
    assert "fetched" in toolboxes.unavailable("matlab", tmp_path)
    with pytest.raises(toolboxes.Unavailable):
        toolboxes.matlab_q2n(np.ones((8, 8)), np.ones((8, 8)), 8, directory=tmp_path)
    folder = tmp_path / "matlab"
    folder.mkdir()
    for file in toolboxes.SOURCES["matlab"]["files"]:
        (folder / file).write_text("disp('not the toolbox')\n")
    assert "SHA-256" in toolboxes.unavailable("matlab", tmp_path)
    with pytest.raises(toolboxes.Unavailable, match="SHA-256"):
        toolboxes.location("matlab", tmp_path)


def test_matlab_uint16_conversion():
    values = [-3.2, -0.5, 0.0, 0.4, 0.5, 1.5, 2.5, 65534.5, 65535.4, 7e4, np.nan]
    expected = [0, 0, 0, 0, 1, 2, 3, 65535, 65535, 65535, 0]
    np.testing.assert_array_equal(toolboxes.matlab_uint16(values), expected)


@pytest.mark.parametrize("bands", [1, 2, 3, 4, 5, 8, 13, 16])
@pytest.mark.parametrize("algorithm", ["moments", "fastcd"])
def test_package_agrees_with_the_matlab_toolbox(matlab, bands, algorithm):
    x, y = integer_pair(bands, (64, 64, bands))
    theirs = matlab(x, y, 32, 32)
    ours = q2n_map(x, y, 32, 32, algorithm=algorithm)
    assert theirs.map.shape == ours.shape
    np.testing.assert_allclose(ours, theirs.map, atol=TOLERANCE, rtol=0)
    assert abs(q2n(x, y, 32, 32, algorithm=algorithm) - theirs.index) <= TOLERANCE


@pytest.mark.parametrize(
    "shape, ws, stride",
    [
        ((64, 64, 4), 32, 16),  # overlapping blocks
        ((70, 81, 4), 32, 32),  # bottom and right padding
        ((50, 77, 5), 16, 8),  # both, with bands padded to eight
        ((45, 40, 3), 16, 12),  # a shift that does not divide the block
        ((64, 64, 8), 8, 8),
        ((96, 64, 6), 64, 32),
    ],
)
def test_blocks_shifts_and_padding(matlab, shape, ws, stride):
    x, y = integer_pair(sum(shape) + ws + stride, shape)
    theirs = matlab(x, y, ws, stride)
    for algorithm in ("moments", "fastcd"):
        ours = q2n_map(x, y, ws, stride, algorithm=algorithm)
        assert theirs.map.shape == ours.shape
        np.testing.assert_allclose(ours, theirs.map, atol=TOLERANCE, rtol=0)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_sample_types(matlab, dtype):
    top = int(np.iinfo(dtype).max) + 1
    x, y = integer_pair(top, (64, 96, 4), top=top, noise=top / 30)
    y = np.clip(y, 0, top - 1)
    theirs = matlab(x, y, 32, 32)
    ours = q2n_map(x.astype(dtype), y.astype(dtype), 32, 32)
    np.testing.assert_allclose(ours, theirs.map, atol=TOLERANCE, rtol=0)


def test_branches_of_the_definition(matlab):
    x, y = integer_pair(77, (32, 64, 4))
    x[:, :32, 1] = 0.0  # a band of zero mean, and constant
    x[:, 32:, 2] = 900.0  # a constant band
    y[:, :, 3] = x[:, :, 3]  # an identical band
    theirs = matlab(x, y, 32, 32)
    expected = np.array(exact_map(x, y, 32, 32), dtype=np.float64)
    np.testing.assert_allclose(theirs.map, expected, atol=1e-9, rtol=0)
    for algorithm in ("moments", "fastcd"):
        ours = q2n_map(x, y, 32, 32, algorithm=algorithm)
        np.testing.assert_allclose(ours, expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(ours, theirs.map, atol=1e-9, rtol=0)
    same = matlab(x, x, 32, 32)
    np.testing.assert_allclose(same.map, q2n_map(x, x, 32, 32), atol=TOLERANCE, rtol=0)


def test_the_toolbox_rounds_its_inputs_and_the_package_does_not(matlab):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 255, (64, 64, 4))
    y = x + rng.normal(0, 20, x.shape)
    theirs = matlab(x, y, 32, 32)
    # On samples that are not integers the two differ, by the rounding alone
    assert np.max(np.abs(q2n_map(x, y, 32, 32) - theirs.map)) > 1e-6
    rounded = q2n_map(toolboxes.matlab_uint16(x), toolboxes.matlab_uint16(y), 32, 32)
    np.testing.assert_allclose(rounded, theirs.map, atol=TOLERANCE, rtol=0)


def test_the_port_of_the_benchmark_is_the_toolbox(matlab):
    # benchmarks/toolbox.py, which the benchmark times, against the original
    for seed, shape, ws, stride in ((1, (64, 64, 8), 32, 32), (2, (50, 77, 5), 16, 8)):
        x, y = integer_pair(seed, shape)
        theirs = matlab(x, y, ws, stride)
        ported = toolbox.q2n_map(x, y, ws, stride)
        np.testing.assert_allclose(ported, theirs.map, atol=TOLERANCE, rtol=0)


def test_rounds_are_timed_inside_the_interpreter(matlab):
    x, y = integer_pair(3, (32, 32, 2))
    result = matlab(x, y, 32, rounds=3)
    assert len(result.seconds) == 3 and all(0 < t < 60 for t in result.seconds)
    assert result.map.shape == (1, 1)
    with pytest.raises(ValueError):
        matlab(x, y[:-1], 32)


def test_report_and_command_line(matlab, capsys):
    table = toolboxes.report(["matlab"])
    rows = [line for line in table.splitlines() if line.startswith("| matlab")]
    assert len(rows) == 18
    assert all(float(row.split("|")[-2]) <= TOLERANCE for row in rows)
    assert toolboxes.main(["status", "matlab"]) == 0
    assert "runnable" in capsys.readouterr().out


def port(name):
    """Return the adapter of a Python port, or skip where it cannot run."""
    reason = toolboxes.unavailable(name)
    if reason is not None:
        pytest.skip(reason)
    return toolboxes.ADAPTERS[name]


# The ports are not exact: one normalizes with the population standard
# deviation, the other stores the value of a block in single precision
PORT_TOLERANCE = 1e-6


@pytest.mark.parametrize("name", ["pancollection", "hyperspectral"])
@pytest.mark.parametrize("bands", [1, 4, 8, 16])
def test_python_ports_in_the_setting_of_the_toolbox(name, bands):
    # A power-of-two number of bands and blocks of 32 that divide the image
    adapter = port(name)
    x, y = integer_pair(11 + bands, (64, 96, bands))
    result = adapter(x, y, 32, 32)
    expected = q2n_map(x, y, 32, 32)
    assert result.map.shape == expected.shape
    np.testing.assert_allclose(result.map, expected, atol=PORT_TOLERANCE, rtol=0)
    assert abs(result.index - float(np.mean(expected))) <= PORT_TOLERANCE
    # Close, and not the same computation: neither port is exact
    assert np.max(np.abs(result.map - expected)) > TOLERANCE


def test_pancollection_outside_that_setting():
    adapter = port("pancollection")
    # Its padding of the bands fails for a number that is not a power of two
    x, y = integer_pair(21, (64, 64, 3))
    with pytest.raises(TypeError):
        adapter(x, y, 32, 32)
    # Its padding of the image changes the value where blocks need it
    for shape, ws, stride in (((64, 64, 4), 32, 16), ((70, 81, 4), 32, 32)):
        x, y = integer_pair(22, shape)
        difference = np.max(
            np.abs(adapter(x, y, ws, stride).map - q2n_map(x, y, ws, stride))
        )
        assert PORT_TOLERANCE < difference < 0.05


def test_hyperspectral_toolbox_outside_that_setting():
    adapter = port("hyperspectral")
    # Other band counts and overlapping blocks agree as far as it stores them
    for shape, ws, stride in (((64, 64, 13), 32, 32), ((64, 64, 3), 32, 16)):
        x, y = integer_pair(23, shape)
        expected = q2n_map(x, y, ws, stride)
        np.testing.assert_allclose(
            adapter(x, y, ws, stride).map, expected, atol=PORT_TOLERANCE, rtol=0
        )
    # Its padding of the image changes the value where blocks need it
    x, y = integer_pair(24, (70, 81, 4))
    difference = np.max(np.abs(adapter(x, y, 32, 32).map - q2n_map(x, y, 32, 32)))
    assert PORT_TOLERANCE < difference < 0.05

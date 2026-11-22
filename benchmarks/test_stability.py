"""Tests of the exact reference, the families of the corpus and the study tooling."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyfastcd")
pytest.importorskip("threadpoolctl")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests"))

import exact  # noqa: E402
import stability  # noqa: E402
import stability_report  # noqa: E402
import stability_study  # noqa: E402

CHECKER = ROOT / "tools" / "check_archives.py"


def test_encode_is_exact():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 5)) * np.ldexp(1.0, rng.integers(-60, 60, (40, 5)))
    values[rng.random((40, 5)) < 0.2] = 0.0
    # Subnormal values, in a band of their own
    values[:, 4] = rng.integers(-(2**20), 2**20, 40) * 5e-324
    limbs, exponents = exact.encode(values)
    assert np.all(np.abs(limbs) < 2**exact.LIMB)
    for i in range(values.shape[1]):
        for p in range(values.shape[0]):
            total = sum(
                int(limbs[k, p, i]) << (exact.LIMB * k) for k in range(len(limbs))
            )
            assert Fraction(total) * Fraction(2) ** exponents[i] == Fraction(
                values[p, i]
            )
    with pytest.raises(ValueError):
        exact.encode(np.array([[np.inf]]))
    # A band spanning more than the double range cannot be scaled to integers
    with pytest.raises(ValueError):
        exact.encode(np.array([[5e-324], [1e300]]))


def test_exact_moments_match_integer_arithmetic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(37, 3)) * np.ldexp(1.0, rng.integers(-40, 40, (37, 3)))
    y = rng.normal(size=(37, 3)) * 2.0**-1060
    m = exact.exact_moments(x, y)
    for i in range(3):
        column = [Fraction(v) for v in x[:, i]]
        assert Fraction(int(m["sx"][i])) * Fraction(2) ** m["ex"][i] == sum(column)
        assert Fraction(int(m["sxx"][i])) * Fraction(2) ** (2 * m["ex"][i]) == sum(
            v * v for v in column
        )
        for j in range(3):
            cross = sum(
                Fraction(u) * Fraction(v) for u, v in zip(x[:, i], y[:, j], strict=True)
            )
            scale = Fraction(2) ** (m["ex"][i] + m["ey"][j])
            assert Fraction(int(m["sxy"][i, j])) * scale == cross


def doubling(x: list[int], y: list[int]) -> list[int]:
    """Multiply by the defining recursion (a,b)(c,d) = (ac - conj(d) b, da + b conj(c))."""
    if len(x) == 1:
        return [x[0] * y[0]]
    h = len(x) // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    conj_d = d[:1] + [-v for v in d[1:]]
    conj_c = c[:1] + [-v for v in c[1:]]
    upper = [u - v for u, v in zip(doubling(a, c), doubling(conj_d, b), strict=True)]
    lower = [u + v for u, v in zip(doubling(d, a), doubling(b, conj_c), strict=True)]
    return upper + lower


def test_sign_table_matches_the_doubling_formula():
    for size in (1, 2, 4, 8, 16):
        signs = exact.sign_table(size)
        for i in range(size):
            for j in range(size):
                left, right = [0] * size, [0] * size
                left[i], right[j] = 1, 1
                assert doubling(left, right)[i ^ j] == signs[i][j]


def test_oracle_validation_passes():
    result = stability.validate_oracle(5, 60)
    assert result["checks"] > 100
    assert result["max_relative_difference"] <= result["tolerance"]


def test_exact_values_of_known_blocks():
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 255, (8, 8, 5))
    assert abs(exact.exact_value(x, x) - 1) < Decimal(10) ** -90
    # One band of exactly zero mean, test = 0.8 reference + 2: 0.96 c / (1 + 0.64 c^2)
    half = rng.normal(0, 10, 32)
    band = np.concatenate([half, -half])[rng.permutation(64)].reshape(8, 8, 1)
    c = (sum(Fraction(v) ** 2 for v in band.ravel()) / 63).limit_denominator(10**40)
    c = Decimal(c.numerator) / Decimal(c.denominator)
    value = exact.exact_value(band, 0.8 * band + 2)
    expected = Decimal("0.96") * c.sqrt() / (1 + Decimal("0.64") * c)
    # The test image is 0.8 x + 2 rounded to double precision
    assert abs(value - expected) < Decimal(10) ** -14


def test_precision_does_not_change_the_value():
    for family in ("uniform", "offset_1e12", "constant_fraction", "scaled_down"):
        x, y = stability.make_case(family, 3, 8, 11)
        coarse, fine = exact.exact_value(x, y, 60), exact.exact_value(x, y, 120)
        assert abs(coarse - fine) <= Decimal(10) ** -55 * max(abs(fine), Decimal(1))


@pytest.mark.parametrize("family", stability.FAMILIES)
def test_every_family_is_deterministic_and_well_formed(family):
    for bands in (1, 4, 6):
        x, y = stability.make_case(family, bands, 8, 3)
        again = stability.make_case(family, bands, 8, 3)
        assert np.array_equal(x, again[0]) and np.array_equal(y, again[1])
        assert x.shape == y.shape == (8, 8, bands)
        assert x.flags.c_contiguous and y.flags.c_contiguous
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        if family == "zero_mean":
            assert all(math.fsum(x[..., b].ravel()) == 0 for b in range(bands))
        if family == "near_zero_mean":
            assert all(math.fsum(x[..., b].ravel()) != 0 for b in range(bands))
        if family.startswith("constant"):
            assert np.all(x[..., 0] == x[0, 0, 0])
        if family == "identical":
            assert np.array_equal(x, y)


def test_seeds_depend_on_every_label():
    seeds = {
        stability.case_seed(1, f, b, s, i)
        for f in range(3)
        for b in (1, 2)
        for s in (8, 16)
        for i in range(3)
    }
    assert len(seeds) == 36
    assert stability.case_seed(1, 0, 1, 8, 0) != stability.case_seed(2, 0, 1, 8, 0)


def test_measured_methods_are_accurate():
    row = stability.run_case("uniform", 3, 8, 0, 4, 60, check=True)
    for method in stability.METHODS:
        assert row["methods"][method]["eta"] < 100
    assert row["precision_check"] < 1e-50
    assert set(row["comparisons"]) == set(stability.COMPARED)
    assert row["marked"] is False
    assert {"I", "F"} <= set(row["methods"])


def test_fused_sums_are_guarded_like_the_moments():
    for family in ("zero_mean", "overflow", "scaled_down", "offset_1e9"):
        row = stability.run_case(family, 5, 8, 0, 4, 60, check=False)
        assert row["marked"]
        # A marked block is evaluated from its samples whatever the algorithm
        assert row["methods"]["F"]["value"] == row["methods"]["I"]["value"]
        assert row["comparisons"]["F"] == "tied"
        # And so is the block merged from tiles
        assert row["methods"]["G"]["value"] == row["methods"]["T"]["value"]


def test_studies_without_fused_sums_render_as_before():
    rows = [
        stability.run_case("uniform", 3, 8, k, 4, 60, check=False) for k in range(2)
    ]
    document = {
        "completed": True,
        "profile": "smoke",
        "samples": rows,
        "corpus": {"families": {"uniform": ""}},
    }
    current = stability_report.render(document)
    assert "I / F / Q / D" in current and "Fused sums of fastCD" in current
    assert "median eta I / F / G" in current
    assert stability_report.measured(document) == stability.METHODS
    for row in rows:
        for method in ("F", "G"):
            del row["methods"][method], row["comparisons"][method]
    legacy = stability_report.render(document)
    assert "F" not in stability_report.measured(document)
    assert "I / Q / D" in legacy and "fastCD" not in legacy
    assert "typical_median_eta" in stability_report.key_figures(document)


def test_hazards_are_handled_by_the_package():
    for family in ("zero_mean", "overflow", "scaled_down"):
        row = stability.run_case(family, 3, 8, 0, 4, 60, check=False)
        assert row["marked"]
        assert row["methods"]["I"]["eta"] < 100, family


def test_section_round_trip():
    text = (
        "# Title\n\n" + stability_report.BEGIN + "\nold\n" + stability_report.END + "\n"
    )
    updated = stability_report.replace_section(text, "new tables\n")
    assert stability_report.section(updated) == "new tables"
    assert stability_report.replace_section("no markers", "tables") is None


def test_update_document_without_file_or_markers(tmp_path):
    assert not stability_report.update_document(tmp_path / "absent.md", "tables")
    plain = tmp_path / "plain.md"
    plain.write_text("no markers\n")
    assert not stability_report.update_document(plain, "tables")
    assert plain.read_text() == "no markers\n"


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Run the smoke profile once into a fresh results directory."""
    results = tmp_path_factory.mktemp("results")
    arguments = ["--profile", "smoke", "--jobs", "1"]
    arguments += ["--output", str(results / "stability-20260101")]
    return stability_study.run(stability_study.parse_arguments(arguments))


def test_published_files(study):
    names = {p.name for p in study.iterdir() if p.is_file()}
    for name in stability_study.PUBLISHED:
        assert name in names
    for chart in ("eta-bands", "relative-error-bands", "eta-offset"):
        for extension in ("png", "svg", "pdf"):
            assert f"{chart}.{extension}" in names
    assert (study / "raw" / "accuracy.json").is_file()


def test_checksums_cover_the_published_files(study):
    lines = (study / "SHA256SUMS").read_text().splitlines()
    recorded = {line.split("  ", 1)[1] for line in lines}
    published = {
        p.relative_to(study).as_posix() for p in stability_study.published_files(study)
    }
    assert recorded == published
    assert not any(name.startswith("raw/") for name in recorded)


def test_study_record(study):
    record = json.loads((study / "study.json").read_text())
    assert record["profile"] == "smoke"
    assert record["oracle_validation"]["max_relative_difference"] < 1e-50
    assert "src/inazumaq2n/_plan.py" in record["source"]["files_sha256"]
    assert "benchmarks/baseline.py" in record["source"]["files_sha256"]
    assert set(record["raw_sha256"]) == {"raw/accuracy.json"}
    assert "--profile smoke" in record["command"]
    readme = (study / "README.md").read_text()
    assert record["command"] in readme


def test_tables_are_generated_from_the_raw_results(study):
    document = json.loads((study / "raw" / "accuracy.json").read_text())
    assert document["completed"]
    assert (
        stability_report.render(document) == (study / "accuracy-tables.md").read_text()
    )
    figures = json.loads((study / "key-figures.json").read_text())
    assert figures["cases"] == len(document["samples"])
    assert figures["hazards_wrong"]["zero_mean"]["I"] == 0


def run_checker(*arguments):
    command = [sys.executable, str(CHECKER), *map(str, arguments)]
    return subprocess.run(command, capture_output=True, text=True)


def test_checker_accepts_the_study(study, tmp_path):
    document = tmp_path / "stability.md"
    tables = (study / "accuracy-tables.md").read_text()
    document.write_text(
        f"# Doc\n\n{stability_report.BEGIN}\n\n{tables}\n{stability_report.END}\n"
    )
    result = run_checker("--results", study.parent, "--document", document, "--full")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["studies"][0]["tables_regenerated"]
    assert "matches" in report["document"]


def test_checker_rejects_changes(study, tmp_path):
    document = tmp_path / "stability.md"
    document.write_text(
        f"{stability_report.BEGIN}\n\nother tables\n\n{stability_report.END}\n"
    )
    result = run_checker("--results", study.parent, "--document", document)
    assert result.returncode == 1
    tables = study / "accuracy-tables.md"
    saved = tables.read_text()
    try:
        tables.write_text(saved + "changed\n")
        result = run_checker(
            "--results", study.parent, "--document", tmp_path / "none.md"
        )
        assert result.returncode == 1
        assert "checksum differs" in result.stderr
    finally:
        tables.write_text(saved)


def test_checker_without_studies(tmp_path):
    result = run_checker("--results", tmp_path, "--document", tmp_path / "none.md")
    assert result.returncode == 0
    assert "no stability studies" in result.stdout

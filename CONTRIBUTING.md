# Contributing to inazumaq2n

The canonical repository is
[git.tanuki-cd.com/algebraity/inazumaq2n](https://git.tanuki-cd.com/algebraity/inazumaq2n).

## Building and testing

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m mypy
python -m ruff check src tests benchmarks tools setup.py
python -m ruff format --check src tests benchmarks tools setup.py
python tools/generate_kernels.py --check
python tools/check_archives.py
```

Building needs a C11 compiler and the Python development headers, for the
extension module `inazumaq2n._core`. After changing a C source, run
`python -m pip install -e .` again to rebuild it. The tests under
`benchmarks/` cover the baseline, the benchmark and the stability study,
and need pyfastcd, threadpoolctl and Matplotlib, which `dev` includes.

## Layout

- `src/inazumaq2n/_core.c`: the extension module. It holds the routine that
  merges tiles and evaluates blocks, the call into an external `dgemm`, the
  detection of the instruction sets of the processor, and the functions
  that check every buffer before a kernel reads or writes it.
- `src/inazumaq2n/_isa.h`: the kernels of one instruction set. `_core.c`
  includes it once per instruction set. It defines an eight-lane vector of
  doubles and its operations for AVX-512, AVX2, NEON and plain C, and then
  includes the two files below once per sample type.
- `src/inazumaq2n/_prepare.h`: centering of tiles with any number of bands.
- `src/inazumaq2n/_direct.h`: the kernels for one to four bands, generated
  by `tools/generate_kernels.py`. Do not edit it; change the generator and
  run it.
- `src/inazumaq2n/_plan.py`: `Q2nPlan`, the public functions, the geometry
  of blocks and tiles, and the dispatch between the kernels.
- `src/inazumaq2n/_blas.py`: the optional MKL provider.
- `src/inazumaq2n/_fallback.py`: the sign table, and the evaluation of one
  marked block from its normalized samples.
- `tests/reference.py`: an evaluation of the definition in floating point
  that imports nothing from the package and uses neither a sign table nor a
  matrix of moments. Tests of values on ordinary inputs compare against it.
- `tests/exact.py`: the exact value of a block, from moments in
  arbitrary-precision integers. Tests of conditioned inputs and the
  stability study compare against it.
- `benchmarks/baseline.py`: the definition evaluated pixel by pixel with
  the products of pyfastcd, the comparison of the benchmark and of the
  stability study.
- `benchmarks/benchmark.py`: the speed benchmark.
- `benchmarks/stability.py`, `stability_report.py`, `stability_plots.py`
  and `stability_study.py`: the stability study, its tables, its charts and
  the driver that writes an archive.
- `tools/check_archives.py`: checks the archives under `benchmark-results/`
  and the tables of `docs/stability.md`.

## Conventions

- In `src/inazumaq2n`, every function has a docstring that begins with a
  one-line summary, and public classes and functions have NumPy-style
  docstrings. Examples in docstrings must run; `tests/test_q2n.py` runs
  them. Comments have no closing punctuation, except multi-line
  explanations of an algorithm, which are written as full sentences where
  the algorithm is implemented.
- The C sources follow the conventions of fastCD: four-space block
  indentation, eight-space continuations, pointers attached to their types,
  and a short purpose comment before each function. They must build without
  warnings under GCC, Clang and MSVC. The build disables floating-point
  contraction and value-changing optimization; the only fused operations
  are the explicit ones of the kernels for at most four bands.
- A kernel must give bit-identical results in every instruction set and for
  every sample type that converts exactly. An operation on the eight-lane
  vector must act on each lane as the plain C version does, and sums over
  lanes must use the fixed order of the reduction in `_isa.h`.
  `tests/test_kernels.py` checks both properties on every instruction set
  the processor supports.
- Every buffer a function of `_core` receives is checked for its element
  type, contiguity, alignment and length, and every tile for its position,
  before a kernel runs. A kernel may read exactly the samples of its tiles
  and nothing beyond them; `tests/test_kernels.py` places images against an
  inaccessible page to check this.
- Every public behavior change needs a test, a documentation update, and a
  line in `CHANGELOG.md`.

## Changing what a kernel or a default does

The choices the package makes for the caller are the band count up to which
the workspace-free kernels are used (`_DIRECT_BANDS` in `_plan.py` and
`Q2N_DIRECT_BANDS` in `_core.c`), the size of the workspace
(`_WORKSPACE_BYTES`), the preference for MKL under `backend="auto"`, and the
conditions under which a block is evaluated again from its normalized
samples (in `finish` in `_core.c`). Each rests on measurements that are
recorded in `docs/benchmark.md` and `docs/stability.md`.

Change a performance default only on the basis of a new benchmark run on
complete calls, with the variants interleaved, and update
`docs/benchmark.md` in the same commit. A kernel that is faster in isolation
is not enough: the effect has to survive into complete calls.

Change a guard or the arithmetic of a kernel only together with the
stability study: add to `benchmarks/stability.py` the family of inputs that
motivates the change, run the study, and update `docs/stability.md` with
`--update-docs`. The baseline in `benchmarks/baseline.py` must stay a
faithful evaluation of the definition: its tests compare it with
`tests/reference.py`.

## Benchmark archives

Never edit the archived results under `benchmark-results/`, including their
checksum records. A new benchmark run goes into a new directory named for
the machine and the date, with a `SHA256SUMS` file, and a new stability
study into `stability-<date>/`, written by `stability_study.py`. The tables
of `docs/benchmark.md`, `docs/stability.md` and the README are updated to
them in the same commit. Raw study data under `raw/` is not distributed;
`tools/check_archives.py` checks it where it is present, and `--full`
requires it and regenerates the tables from it.

## Release checks

Update the version in `src/inazumaq2n/_version.py`, the README heading, and
the version in `CITATION.cff`, where the release date is added as
`date-released`. Move the pending changelog entries into a dated release
section.

Build and check the distributions in a development environment:

```sh
python -m build
python -m twine check --strict dist/*
```

CI builds the source distribution and a wheel from it, installs the wheel,
and runs the tests from the extracted source distribution outside the
checkout, so that the distributed files are what is tested. The release
workflow builds the wheels for every supported platform with cibuildwheel,
whose configuration is in `pyproject.toml`, and runs the tests against each
wheel. A pushed version tag publishes to PyPI from the GitHub mirror; a
manual run of the workflow publishes to TestPyPI.

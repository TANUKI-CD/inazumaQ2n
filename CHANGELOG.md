# Changelog

All notable changes to inazumaq2n are recorded in this file. The project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html); the
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The
version is defined in `src/inazumaq2n/_version.py`.

## [Unreleased]

### Added

- `q2n`: the Q2n index of a test image against a reference image, for
  images with any number of bands, as the MATLAB pansharpening toolbox
  defines it, in double precision.
- `q2n_map`: the value of every block.
- `Q2nPlan`: a reusable computation for image pairs of one shape, with
  `score`, `map` and `close`, usable as a context manager.
- `runtime_info`: the instruction set and matrix-product library selected.
- Evaluation from centered real moments of the blocks: one `B` by `B`
  matrix of cross moments per tile for an image with `B` bands, contracted
  with the Cayley–Dickson signs, with no hypercomplex product of pixels and
  no padded copy of the image or of its bands.
- Shared tiles for overlapping blocks whose size is a multiple of the
  stride, so that each pixel is read once.
- Kernels for one to four bands that compute the moments of a tile in two
  passes with no workspace and no matrix product, for AVX-512, AVX2 and
  NEON, with bit-identical results in all three and in the baseline code.
- In-place reading of `float64`, `float32`, `uint8` and `uint16` images.
- Selection of the instruction set at import; the module is compiled for the
  baseline instruction set of the platform.
- Optional use of Intel MKL for the matrix products, through `backend` and
  the `mkl` extra, without changing the global settings of MKL.
- `workers`: division of the tiles among threads.
- A second evaluation, from the normalized samples, of the blocks on which
  the moment shortcut may be inaccurate: a mean far larger than the standard
  deviation, a mean within rounding of zero, and moments that underflow or
  overflow. It decides the zero-mean and constant-band branches of the
  definition exactly and computes standard deviations without overflow or
  underflow.
- A speed benchmark against the definition evaluated pixel by pixel with
  the direct and quasilinear products of pyfastcd (`benchmarks/benchmark.py`,
  `benchmarks/baseline.py`), archived in `benchmark-results/t440-20261113/`.
- A stability study against an exact reference (`benchmarks/stability*.py`,
  `tests/exact.py`), archived in `benchmark-results/stability-20261113/`,
  and `tools/check_archives.py`.

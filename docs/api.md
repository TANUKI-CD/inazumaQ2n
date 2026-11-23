# Python API reference

inazumaQ2n computes the Q2n image quality index of a test image against a
reference image with any number of bands, and it is installed as the Python
package `inazumaq2n`. Its functions take images as NumPy-compatible arrays and
evaluate the index in compiled kernels, which are selected for the processor
when the package is imported, and errors are reported as Python exceptions.

This page documents every public name of `inazumaq2n` and the rules that govern
them, while the [quick start](quickstart.md) introduces them by example and is
the place to begin, and [Conventions and results](mathematics.md) defines the
index and explains how it is computed. The public names are `q2n`, `q2n_map`,
`Q2nPlan`, `runtime_info`, and `__version__`, and names that begin with an
underscore are private and may change between releases. After the dependencies
and the terms used throughout, the page documents the module functions, then the
class `Q2nPlan`, then the choice of the algorithm, the matrix-product library,
and the instruction set, and finally exceptions, threads, versioning, and
typing.

---

## Dependencies

`inazumaq2n` depends on:

- CPython 3.10 or newer;
- NumPy 1.26 or newer.

The extension module contains the fused sums of fastCD 1.5.0, compiled from the
sources in `vendor/fastcd`, so neither fastCD nor its Python package pyfastcd
needs to be installed, and pyfastcd is not used. On Linux x86-64, the optional
extra `mkl` installs Intel MKL, which the package then uses for the matrix
products of images with more than four bands (see [Backends](#backends)).

---

## Terms and notation

The rest of this page uses the following terms.

**Images.** An image is anything `numpy.asarray` turns into a real array of
shape `(height, width)` or `(height, width, bands)`, where a two-dimensional
image has one band. The two images of a call, the *reference* and the *test*
image, must have the same shape, and the reference supplies the normalization of
both, so the index is not symmetric in them.

**Sample types.** Images of four sample types are read from the memory of the
caller without a converted copy, and other images are converted to `float64`
first:

| Sample type | Handling |
| --- | --- |
| `float64`, `float32`, `uint8`, `uint16` | Read in place, if both images have the same type, are C-contiguous and aligned, and have native byte order |
| Other integer and floating types, `bool` | Converted to `float64` |
| A pair of different types | Both converted to `float64` |
| Arrays that are not C-contiguous | Converted to contiguous `float64` |
| Complex, string, object | `TypeError` |

Conversion to `float64` is exact for every integer type whose values are below
`2^53` in absolute value, and reading in place gives the same bits as converting
first.

**Blocks.** The index is the mean of the values of square blocks of `ws` by `ws`
pixels, whose origins lie `stride` pixels apart in each direction. `ws` is an
integer of at least 2, and `stride` an integer of at least 1, which defaults to
`ws`, so that blocks do not overlap. They are the `Q_blocks_size` and the
`Q_shift` of the MATLAB toolbox, whose evaluation script uses 32 for both, and
both are part of the definition of the index, so another block size or stride
gives another value. There are `ceil(height / stride)` rows and
`ceil(width / stride)` columns of blocks, the *block shape*, and blocks beyond
the bottom and the right edge read the image through its symmetric reflection.

The last row and the last column of blocks must reach the bottom and the right
edge of the image, and no block may extend beyond an edge by more than the size
of the image in that direction, or `ValueError` is raised. The first condition
can fail only when `stride` exceeds `ws`, in which case the pixels between two
blocks belong to neither, and the second fails when `ws` is more than twice the
height or the width.

**Tiles.** When `ws` is a multiple of `stride`, the image is cut into tiles of
`stride` by `stride` pixels, and overlapping blocks share the work on their
tiles; otherwise every block is its own tile.

**Algorithms.** The covariance of a block is obtained either from the real cross
moments of its tiles, the *moments*, or from one *fused sum* of products of
fastCD (see [Choosing the algorithm](#choosing-the-algorithm)), and the
parameter `algorithm` selects between them.

Below, `reference` and `test` are images, and `B` is the number of bands.

---

## Module contents

- `__version__: str`

  The version of the package.

- `q2n(reference, test, ws=32, stride=None, *, workers=1, backend="auto",
  algorithm="auto") -> float`

  Return the Q2n index of `test` against `reference`, the mean of the values of
  the blocks.

  - **Parameters:** `reference` and `test` are images of the same shape, and
    `ws` and `stride` describe the blocks, where `stride=None` means `ws`.
    `workers` is the number of threads over which the tiles are divided, at
    least 1. `backend` is `"auto"`, `"numpy"`, or `"mkl"`, the matrix-product
    library for images with more than four bands (see [Backends](#backends)),
    and `algorithm` is `"auto"`, `"moments"`, or `"fastcd"` (see [Choosing the
    algorithm](#choosing-the-algorithm)).
  - **Returns:** the index, as a Python `float`.
  - **Notes:** The index is not clipped, so it can exceed one for images with
    more than eight bands, and it is NaN if a block contains a value that is not
    finite. A call is equivalent to creating a `Q2nPlan` for the shape of
    `reference`, calling `score` once, and closing the plan, and it gives the
    same bits.

- `q2n_map(reference, test, ws=32, stride=None, *, workers=1, backend="auto",
  algorithm="auto") -> numpy.ndarray`

  Return the value of every block.

  - **Parameters:** those of `q2n`.
  - **Returns:** a new `float64` array of the block shape, whose entry `(r, c)`
    is the value of the block with origin `(r * stride, c * stride)`, and whose
    mean is the index that `q2n` returns.

- `runtime_info() -> dict`

  Return what the package selected on this machine.

  - **Returns:** a dictionary with these keys:

    | Key | Value |
    | --- | --- |
    | `version` | The version of the package |
    | `simd` | The instruction set of the compiled kernels: `"avx512"`, `"avx2"`, `"neon"`, or `"baseline"` |
    | `direct_kernels` | Whether images with at most four bands use the kernels that need no matrix product, which is false only where the processor has no fused multiply-add instruction |
    | `backend` | The library that `backend="auto"` selects: `"mkl"` or `"numpy"` |
    | `backend_version` | The version of MKL, or of NumPy |
    | `numpy` | The version of NumPy |
    | `fastcd` | The version of fastCD compiled into the package |
    | `fastcd_kernel` | The kernel that fastCD runs its fused sums with on this processor: `"avx512"`, `"avx2"`, or `"portable"` |

---

## Q2nPlan

A `Q2nPlan` is a reusable computation for image pairs of one shape, which holds
the block geometry, the work arrays, and, with more than one worker, the worker
threads, so that scoring many pairs does not repeat their setup.

### Constructor

```python
Q2nPlan(shape, ws=32, stride=None, *, workers=1, backend="auto", algorithm="auto")
```

Create a plan for images of shape `shape`, which is `(height, width)` or
`(height, width, bands)`, with the other parameters of `q2n`. The constructor
resolves `backend` and `algorithm`, so the plan records the library and the
algorithm that it uses, and it raises what `q2n` raises for its arguments.

### Properties

All properties are read-only.

- `shape: tuple[int, ...]`

  The image shape.

- `ws: int`, `stride: int`

  The side of a block and the distance between blocks.

- `bands: int`

  The number of bands.

- `dimension: int`

  The dimension of the algebra, the least power of two that is at least `bands`.

- `block_shape: tuple[int, int]`

  The rows and columns of blocks, which is the shape of a map.

- `workers: int`

  The number of workers.

- `backend: str`

  The matrix-product library selected, `"numpy"` or `"mkl"`.

- `algorithm: str`

  The algorithm selected, `"moments"` or `"fastcd"`, never `"auto"`.

### Methods

- `score(reference, test) -> float`

  Return the index of the pair.

- `map(reference, test) -> numpy.ndarray`

  Return the value of every block, as a new array of shape `block_shape`.

- `close() -> None`

  Release the worker threads and the work arrays. It may be called more than
  once.

`score` and `map` raise `ValueError` if an image does not have the shape of the
plan, `TypeError` if an image is not real, and `RuntimeError` if the plan is
closed. The two images of a call may have any supported sample type, and the
type may change from call to call.

A plan is a context manager, so entering returns the plan and leaving closes it:

```python
with Q2nPlan((512, 512, 4), ws=32) as plan:
    scores = [plan.score(reference, test) for test in tests]
```

### Memory

What a plan holds depends on its algorithm. With `algorithm="moments"`, it holds
`8 (bands^2 + 4 bands)` bytes for each tile, and for images with more than four
bands each worker also holds a workspace for the centered samples of a batch of
tiles, about 8 MiB, or the samples of one tile of both images if that is more,
while images with at most four bands need no workspace.

With `algorithm="fastcd"`, each worker holds the pixels of one block of both
images as records of `dimension` doubles, `16 ws^2 dimension` bytes, and the
workspace of a fused sum, which grows like `dimension log dimension` and does
not depend on the block, and nothing is held for each block. When `ws` is a
multiple of `stride`, a worker instead holds the records of the tiles of a strip
of adjacent block columns, about 4 MiB, or those of one block if that is more.

---

## Choosing the algorithm

Every block needs one hypercomplex quantity, the covariance of its normalized
pixels, and `algorithm` selects how it is obtained. Both algorithms give the
value of the definition to rounding error, but they round differently, so their
values differ in the last digits.

| `algorithm` | Evaluation |
| --- | --- |
| `"moments"` | The matrix of real cross moments of every tile, contracted with the signs of the algebra. Its cost grows with the square of the number of bands, and blocks that overlap share their tiles |
| `"fastcd"` | One fused sum of products of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) for each block, which adds up the products of its pixels in `O(N log N)` operations for each pixel, for an algebra of dimension `N`, without forming them. Blocks that overlap share the records and the statistics of their tiles, but each block is summed on its own |
| `"auto"` | The one that was measured to be faster for the number of bands, the number of tiles a block is merged from, and the kernel of the fused sums on this processor |

The fused sums need no matrix-product library and ignore `backend`. They read
`float64`, `float32`, `uint8`, and `uint16` images in place, divide the blocks
among the `workers`, and mark the same blocks for evaluation from their
normalized samples as the moments do. `"auto"` never selects them for images
with at most four bands, and it selects them only where the benchmarks measured
them to be faster: without overlap from eight bands on with the AVX-512 and AVX2
kernels of fastCD, and with overlap only in high dimensions, while with the
portable kernel of fastCD `"auto"` always selects the moments. [How the results
set the defaults](benchmark.md#how-the-results-set-the-defaults) gives the rule
and the measurements behind it.

---

## Backends

`backend` matters only for `algorithm="moments"` and images with more than four
bands, whose cross moments are matrix products.

| `backend` | Library |
| --- | --- |
| `"numpy"` | `numpy.matmul`, with the BLAS that NumPy was built with |
| `"mkl"` | The `dgemm` of Intel MKL, loaded from the installed `mkl` package, on Linux only; `RuntimeError` if it cannot be used |
| `"auto"` | MKL if it can be used, and NumPy otherwise |

MKL is called through its entry point with 64-bit integers, with one MKL thread
set for the calling thread only and restored afterwards, so the integer
interface and the thread count that other code in the process sees are
unchanged. With NumPy, the thread count of its BLAS is whatever the process has
configured, and the [quick start](quickstart.md#threads) shows how to combine it
with `workers`.

---

## Instruction sets

The compiled kernels are chosen when the package is imported, as the widest
instruction set that both the build and the processor support: AVX-512 when the
processor has AVX-512F, DQ, BW, and VL, AVX2 when it has AVX2 and FMA, NEON on
AArch64, and the baseline code otherwise. fastCD chooses the kernel of its fused
sums by its own rule, which is AVX-512 with AVX-512F and AVX2 with AVX2 and FMA,
in builds for x86-64 with GCC or Clang, and its portable kernel otherwise.
`runtime_info` reports both choices.

---

## Exceptions

`inazumaq2n` reports errors with standard Python exceptions:

- `TypeError`

  An image that does not hold real numbers, such as a complex, string, or object
  array.

- `ValueError`

  - Images that differ in shape, or that do not have the shape of a plan.
  - A shape that is not `(height, width)` or `(height, width, bands)`.
  - `ws`, `stride`, `workers`, or an extent of a shape that is not an integer of
    at least its minimum, including a boolean.
  - Blocks whose last row or column does not reach the edge of the image, or
    that extend beyond an edge by more than the size of the image.
  - An image side, block size, or band count above `2^30`, or an image with too
    many blocks.
  - A `backend` or `algorithm` that is not one of its names.

- `RuntimeError`

  - `backend="mkl"` where MKL cannot be used.
  - A call on a closed plan.

### Floating-point conditions

No exception is raised for values that are not finite. A block that contains a
NaN or an infinity has the value NaN, and the index is then NaN, while samples
whose sum over a block overflows, above about `10^308` divided by the number of
pixels, are outside the range of the package.

---

## Threads

Calls on one plan run one at a time, so a second thread that calls a plan waits
for the first, and to score pairs simultaneously, each thread needs its own
plan. The compiled kernels release the global interpreter lock (GIL), so other
threads can run during them. A plan with `workers > 1` divides the tiles of each
call among its own threads, and the result does not depend on the number of
workers.

---

## Versioning and stability

The package follows semantic versioning, and its public interface is what this
page documents. Values of the index may change in the last digits between
releases, when the order of floating-point operations changes, while a change
beyond rounding is a breaking change.

---

## Typing

The package ships inline type annotations, a stub for its compiled module, and a
`py.typed` marker. Images are annotated as `numpy.typing.ArrayLike`, and the
functions and the methods of `Q2nPlan` have concrete return annotations.

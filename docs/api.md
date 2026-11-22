# Python API reference

```python
import inazumaq2n
from inazumaq2n import q2n, q2n_map, Q2nPlan, runtime_info
```

The package exports two functions, one class, one diagnostic and
`__version__`. Everything whose name begins with an underscore is private
and may change between releases.

## Images

An image is anything `numpy.asarray` turns into a real array of shape
`(height, width)` or `(height, width, bands)`. A two-dimensional image has
one band. The reference and the test image of a call must have the same
shape.

| Sample type | Handling |
| --- | --- |
| `float64`, `float32`, `uint8`, `uint16` | Read in place, if both images have the same type, are C-contiguous and aligned, and have native byte order |
| Other integer and floating types, `bool` | Converted to `float64` |
| A pair of different types | Both converted to `float64` |
| Arrays that are not C-contiguous | Converted to contiguous `float64` |
| Complex, string, object | `TypeError` |

Conversion to `float64` is exact for every integer type whose values are
below `2^53` in absolute value. Reading in place gives the same bits as
converting first.

## Blocks

`ws` is the side of a block, an integer of at least 2. `stride` is the
distance between the origins of adjacent blocks, an integer of at least 1,
and defaults to `ws`. They are the `Q_blocks_size` and the `Q_shift` of the
MATLAB toolbox, whose evaluation script uses 32 for both. Both are part of
the definition of the index: another block size or stride gives another
value. There are `ceil(height / stride)` rows and
`ceil(width / stride)` columns of blocks. Blocks beyond the bottom and right
edges read the image through its symmetric reflection.

The blocks must cover the image, and no block may extend beyond an edge by
more than the size of the image in that direction; otherwise `ValueError`
is raised. The first condition fails when `stride` exceeds `ws` and leaves
pixels at an edge outside every block. The second fails when `ws` is more
than twice the height or the width.

When `ws` is a multiple of `stride`, overlapping blocks share the work on
their pixels, and the cost of a call is close to that of blocks of size
`stride` without overlap. This holds for `algorithm="moments"`. With
`algorithm="fastcd"` the blocks share the reading of their pixels and the
statistics of their tiles, but every block is summed on its own (see
[Algorithms](#algorithms)).

## `q2n`

```python
q2n(reference, test, ws=32, stride=None, *, workers=1, backend="auto",
    algorithm="auto") -> float
```

Return the Q2n index of `test` against `reference`: the mean of the values
of the blocks, as a Python float.

| Parameter | Meaning |
| --- | --- |
| `reference`, `test` | The images. The reference supplies the normalization of both |
| `ws` | The side of a block |
| `stride` | The distance between blocks; `ws` when `None` |
| `workers` | The number of threads over which the tiles are divided, at least 1 |
| `backend` | `"auto"`, `"numpy"` or `"mkl"`: the matrix-product library for images with more than four bands |
| `algorithm` | `"auto"`, `"moments"` or `"fastcd"`: how the covariance of a block is obtained; see [Algorithms](#algorithms) |

Raises `ValueError` if the images differ in shape, if an argument is out of
range, or if the blocks do not fit the image as described above; `TypeError`
if an image is not real; `RuntimeError` if `backend="mkl"` and MKL cannot be
used.

The index is not clipped. It can exceed one for images with more than eight
bands, and it is NaN if a block contains a value that is not finite.

## `q2n_map`

```python
q2n_map(reference, test, ws=32, stride=None, *, workers=1, backend="auto",
        algorithm="auto") -> numpy.ndarray
```

Return the value of every block, as a new `float64` array of shape
`(ceil(height / stride), ceil(width / stride))`. Entry `(r, c)` is the value
of the block with origin `(r * stride, c * stride)`. The parameters and
exceptions are those of `q2n`, and `q2n` returns the mean of this array.

## `Q2nPlan`

```python
Q2nPlan(shape, ws=32, stride=None, *, workers=1, backend="auto", algorithm="auto")
```

A reusable computation for image pairs of one shape. A plan holds the block
geometry, the work arrays and, with more than one worker, the worker
threads, so that scoring many pairs does not repeat their setup. `shape` is
`(height, width)` or `(height, width, bands)`; the other parameters are
those of `q2n`. The constructor raises what `q2n` raises for its arguments.

`q2n(x, y, ...)` is equivalent to creating a plan for the shape of `x`,
calling `score(x, y)` and closing the plan, and gives the same bits.

### Methods

| Method | Result |
| --- | --- |
| `score(reference, test) -> float` | The index of the pair |
| `map(reference, test) -> numpy.ndarray` | The value of every block, as a new array of shape `block_shape` |
| `close() -> None` | Release the worker threads and work arrays. May be called more than once |

`score` and `map` raise `ValueError` if an image does not have the shape of
the plan, `TypeError` if an image is not real, and `RuntimeError` if the
plan is closed. The two images of a call may have any supported sample
type, and the type may change from call to call.

A plan is a context manager. Entering returns the plan, and leaving closes
it:

```python
with Q2nPlan((512, 512, 4), ws=32) as plan:
    scores = [plan.score(reference, test) for test in tests]
```

### Attributes

All attributes are read-only.

| Attribute | Value |
| --- | --- |
| `shape` | The image shape, a tuple |
| `ws`, `stride` | The side of a block and the distance between blocks |
| `bands` | The number of bands |
| `dimension` | The dimension of the algebra, the least power of two not below `bands` |
| `block_shape` | `(rows, columns)` of blocks, the shape of a map |
| `workers` | The number of workers |
| `backend` | The matrix-product library selected, `"numpy"` or `"mkl"` |
| `algorithm` | The algorithm selected, `"moments"` or `"fastcd"`; never `"auto"` |

### Threads and plans

Calls on one plan run one at a time: a second thread that calls a plan
waits for the first. To score pairs simultaneously, create one plan per
thread. The compiled kernels release the global interpreter lock.

A plan with `workers > 1` divides the tiles of each call among its threads.
The result does not depend on the number of workers.

### Memory

With `algorithm="moments"`, a plan holds `8 (bands^2 + 4 bands)` bytes per
tile, a tile being one block, or one `stride` by `stride` square when `ws`
is a multiple of `stride`. For images with more than four bands, each worker
also holds a workspace for the centered samples of a batch of tiles: about
8 MiB, or the samples of one tile of both images if that is more. Images
with at most four bands need no workspace.

With `algorithm="fastcd"`, each worker holds the pixels of one block of both
images as records of `dimension` doubles, `16 ws^2 dimension` bytes, and the
workspace of a fused sum, which grows like `dimension log dimension` and
does not depend on the block. Nothing is held per block. When `ws` is a
multiple of `stride`, a worker instead holds the records of the tiles of a
strip of adjacent block columns, about 4 MiB or those of one block if that
is more.

## `runtime_info`

```python
runtime_info() -> dict
```

Return what the package selected on this machine.

| Key | Value |
| --- | --- |
| `version` | The version of the package |
| `simd` | The instruction set of the compiled kernels: `"avx512"`, `"avx2"`, `"neon"` or `"baseline"` |
| `direct_kernels` | Whether images with at most four bands use the kernels that need no matrix product. False only where the processor has no fused multiply-add instruction |
| `backend` | The library that `backend="auto"` selects: `"mkl"` or `"numpy"` |
| `backend_version` | The version of MKL, or of NumPy |
| `numpy` | The version of NumPy |
| `fastcd` | The version of fastCD compiled into the package |
| `fastcd_kernel` | The kernel that fastCD runs its fused sums with on this processor: `"avx512"`, `"avx2"` or `"portable"` |

The instruction set is chosen when the package is imported, as the widest
that both the build and the processor support. AVX-512 is used when the
processor has AVX-512F, DQ, BW and VL; AVX2 when it has AVX2 and FMA.
fastCD chooses the kernel of its fused sums by its own rule: AVX-512 with
AVX-512F, AVX2 with AVX2 and FMA, in builds for x86-64 with GCC or Clang,
and its portable kernel otherwise.

## Algorithms

Every block needs one hypercomplex quantity, the covariance of its
normalized pixels. The package obtains it in one of two ways, and
`algorithm` selects between them. Both give the value of the definition to
rounding error; they round differently, so their values differ in the last
digits.

| `algorithm` | Evaluation |
| --- | --- |
| `"moments"` | The matrix of real cross moments of every tile, contracted with the signs of the algebra. Its cost grows with the square of the number of bands, and blocks that overlap share their tiles |
| `"fastcd"` | One fused sum of products of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) per block: the sum of the products of its pixels, in `O(N log N)` operations per pixel for an algebra of dimension `N`, without forming the products. Blocks that overlap share the records and the statistics of their tiles, and each block is summed on its own |
| `"auto"` | The one that was measured to be faster for the number of bands, the number of tiles a block is merged from, and the kernel of the fused sums on this processor |

The fused sums need no matrix-product library and ignore `backend`. They
read `float64`, `float32`, `uint8` and `uint16` images in place, divide the
blocks among the `workers`, and mark the same blocks for evaluation from
their normalized samples as the moments do. `"auto"` never selects them for
images with at most four bands, and selects them only where
[the benchmark](benchmark.md#algorithms) measured them faster: without
overlap from eight bands on with the AVX-512 and AVX2 kernels of fastCD, and
with overlap only in high dimensions. With the portable kernel of fastCD
`"auto"` is always `"moments"`.

fastCD 1.5.0 is compiled into the extension module from the sources in
`vendor/fastcd`; the `pyfastcd` package is not needed and is not used.

## Backends

`backend` matters only for `algorithm="moments"` and images with more than
four bands, whose cross moments are matrix products.

| `backend` | Library |
| --- | --- |
| `"numpy"` | `numpy.matmul`, with the BLAS that NumPy was built with |
| `"mkl"` | The `dgemm` of Intel MKL, loaded from the installed `mkl` package. `RuntimeError` if it cannot be used. Linux only |
| `"auto"` | MKL if it can be used, and NumPy otherwise |

MKL is called through its entry point with 64-bit integers, with one MKL
thread set for the calling thread only and restored afterwards, so the
integer interface and the thread count that other code in the process sees
are unchanged. With NumPy, the thread count of its BLAS is whatever the
process has configured; see the [quick start](quickstart.md) for combining
it with `workers`.

## Versioning

The package follows semantic versioning. The public interface is what this
page documents. Values of the index may change in the last digits between
releases, when the order of floating-point operations changes; a change
beyond rounding is a breaking change or a bug.

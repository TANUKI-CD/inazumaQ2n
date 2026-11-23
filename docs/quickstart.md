# Quick start

inazumaQ2n computes the Q2n index, which measures how closely a test image, such
as the result of pansharpening, fusion, or restoration, reproduces a reference
image with any number of bands. The index compares the two images block by
block, treating every pixel as an element of a Cayley-Dickson algebra, and
inazumaQ2n evaluates it in compiled kernels without forming the products of
pixels that its definition calls for; it is installed as the Python package
`inazumaq2n`, which takes images as NumPy arrays.

This guide introduces the basics of how to use inazumaQ2n one idea at a time,
from installation and the index of one image pair through sample types, blocks,
and maps to plans for many pairs, the choice of the algorithm, threads, and what
the package selected on a machine. Each example builds on the examples that come
before it and shows its results in comments. Users of inazumaQ2n are recommended
to start here and read this guide in its entirety, running the examples for
themselves before using the package in production. The [Python API
reference](api.md) gives the complete rules for everything shown here and should
be read for a more detailed reference if more information is needed than what
this quick start guide can provide.

## Installation

inazumaQ2n requires CPython 3.10 or newer and NumPy 1.26 or newer. Once it is
released on PyPI, which it has not been yet, it will be installed with:

```sh
pip install inazumaq2n
```

To build and install it from a source checkout instead, you also need a C11
compiler and the Python development headers. Run the following command from the
repository root, with the Python interpreter that will run your code:

```sh
python -m pip install .
```

For images with more than four bands the package multiplies matrices with the
library that NumPy was built with, unless Intel MKL is installed as the `mkl`
package, which was faster for images with many bands in the
[benchmarks](benchmark.md#numpy-and-mkl) and which the extra `mkl` installs on
Linux x86-64:

```sh
python -m pip install ".[mkl]"
```

## The index of an image pair

An image is an array of shape `(height, width, bands)`, or `(height, width)` for
one band, and `q2n` returns the index of a test image against a reference image
of the same shape:

```python
import numpy as np
from inazumaq2n import q2n

rng = np.random.default_rng(0)
reference = rng.uniform(0, 255, (256, 256, 8))       # 8 bands
test = reference + rng.normal(0, 10, reference.shape)

round(q2n(reference, test), 4)                       # 0.9909
q2n(reference, test) == q2n(test, reference)         # False
```

A value of one means that the test image reproduces the reference in every
block, and lower values mean larger departures. The reference image comes first
because it supplies the normalization of both images, so exchanging the two
arguments changes the value, here in the seventh digit.

## Images and sample types

Arrays of `float64`, `float32`, `uint8`, and `uint16` samples are read in place
when both images have the same type and are C-contiguous, and anything else that
holds real numbers is accepted and converted to `float64`: other integer types,
lists, strided views, Fortran-ordered arrays, and pairs of different types.

```python
image8 = np.clip(reference, 0, 255).astype(np.uint8)
noisy8 = np.clip(test, 0, 255).astype(np.uint8)
round(q2n(image8, noisy8), 4)                        # 0.9912, with no float64 copy

channels_first = np.moveaxis(reference, -1, 0)       # (bands, height, width)
q2n(np.moveaxis(channels_first, 0, -1), test) == q2n(reference, test)   # True
```

Each sample is converted to a double exactly as it is read, so an integer image
gives the same bits as its conversion to `float64`. A channel-first array, as
PyTorch uses, has to be moved to channel-last first, and since the moved array
is not contiguous, it is copied once, which `np.ascontiguousarray` beforehand
makes explicit.

## Blocks and maps

The index is the mean of a value computed on every block of `ws` by `ws` pixels,
where blocks start every `stride` pixels, and `q2n_map` returns the value of
every block:

```python
from inazumaq2n import q2n_map

block_map = q2n_map(reference, test)                 # blocks of 32, no overlap
block_map.shape                                      # (8, 8)
block_map.mean() == q2n(reference, test)             # True

q2n_map(reference, test, ws=64, stride=32).shape     # (8, 8): blocks of 64, half overlapping
```

`ws` defaults to 32 and `stride` to `ws`, the setting of the evaluation script
of the MATLAB pansharpening toolbox, and both are part of the definition of the
index, so another block size or stride gives another value. Entry `(r, c)` of a
map is the value of the block whose origin is `(r * stride, c * stride)`, which
locates the regions where the test image departs from the reference, and the
blocks that extend beyond the bottom and the right edge read the image through
its symmetric reflection.

## Many pairs of one shape

Every call of `q2n` sets up the block geometry and the work arrays for the shape
of its images, so scoring many pairs of one shape repeats that setup, which a
`Q2nPlan` keeps instead:

```python
from inazumaq2n import Q2nPlan

candidates = [reference + rng.normal(0, sigma, reference.shape) for sigma in (5, 10, 20)]
with Q2nPlan(reference.shape) as plan:
    scores = [plan.score(reference, candidate) for candidate in candidates]
    worst = plan.map(reference, candidates[int(np.argmin(scores))])

[round(score, 3) for score in scores]                # [0.998, 0.991, 0.964]
worst.shape                                          # (8, 8)
```

A plan saves about a tenth of a millisecond for each call, which matters for
images with one to four bands, where a call of 512 by 512 pixels takes well
under a millisecond for each band (see [Complete
calls](benchmark.md#complete-calls-against-the-definition)). A plan scores one
pair at a time, so to score pairs from several threads, each thread needs its
own plan.

## Choosing the algorithm

The package obtains the covariance of each block either from real cross moments
of its bands or from one fused sum of products of fastCD, and by default it
selects the one that was measured to be faster for the image, the blocks, and
the processor:

```python
moments = q2n_map(reference, test, algorithm="moments")   # real cross moments
fused = q2n_map(reference, test, algorithm="fastcd")      # fused sums of fastCD
np.abs(moments - fused).max() < 1e-15                     # True
(moments == fused).all()                                  # False

Q2nPlan(reference.shape).algorithm                        # 'fastcd'
```

The default, `algorithm="auto"`, selects the moments for images with at most
four bands and the fused sums for images with eight bands or more whose blocks
do not overlap, and for overlapping blocks only in high dimensions, so the last
line gives `'fastcd'` on a processor where fastCD runs a vector kernel, and
`'moments'` elsewhere ([Choosing the algorithm](api.md#choosing-the-algorithm)
has the details). The two algorithms give the same index to rounding error but
not the same bits, so to compare scores between machines to the last digit, pass
`algorithm` explicitly, and with `algorithm="moments"` also `backend="numpy"`,
because MKL and the library of NumPy round differently too.

## Threads

`workers` divides the tiles of the image among threads, and the result does not
depend on their number:

```python
q2n(reference, test, workers=4) == q2n(reference, test)   # True
```

With the moments, more than four bands, and NumPy as the matrix-product library,
the library has threads of its own, and the two levels multiply, so the library
should then be set to one thread, with `OPENBLAS_NUM_THREADS=1` in the
environment or, if it is installed, with
[threadpoolctl](https://github.com/joblib/threadpoolctl):

```python
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1, user_api="blas"):
    score = q2n(reference, test, workers=4)
```

With MKL the package sets one MKL thread for each worker itself, for its own
calls only, and for calls that take a millisecond or less, handing the tiles to
threads costs about as much as it saves, so `workers` is best left at one there.

## What the package selected

The compiled kernels, the matrix-product library, and the kernel of the fused
sums are all selected when the package is imported, and `runtime_info` reports
the choices:

```python
from inazumaq2n import runtime_info

info = runtime_info()
# {'version': '0.1.0', 'simd': 'avx512', 'direct_kernels': True,
#  'backend': 'mkl', 'backend_version': '2026.2.1', 'numpy': '2.5.4',
#  'fastcd': '1.5.0', 'fastcd_kernel': 'avx512'}
```

`simd` is the instruction set of the compiled kernels, `backend` the
matrix-product library that `backend="auto"` uses, and `fastcd_kernel` the
kernel that fastCD runs its fused sums with, and the values in the comment are
those of the machine on which the benchmarks were run.

## Things to know about the index

- **Values are not clipped.** With more than eight bands the value of a block,
  and so the index, can exceed one.
- **An image compared with itself scores one,** except where a band of the
  reference has a mean of exactly zero over a block, which changes the
  normalization of the test band (see [Conventions and
  results](mathematics.md)).
- **Images should be finite.** A block that contains a NaN or an infinity has
  the value NaN, and so has the index.
- **Constant reference bands are fragile.** A reference band that is exactly
  constant over a block is normalized by the machine epsilon, and the value of
  the block changes completely if the band is perturbed by a single rounding
  error (see [Numerical stability](stability.md)).

## Where to go next

- The [Python API reference](api.md) documents every function, class, and rule.
- [Conventions and results](mathematics.md) gives the definition of the index
  and explains how it is computed without products of pixels.
- The [benchmarks](benchmark.md) and the [stability study](stability.md) report
  the speed and the accuracy of the package.

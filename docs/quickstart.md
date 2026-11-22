# Quick start

## Installation

inazumaq2n requires CPython 3.10 or newer and NumPy 1.26 or newer.

```sh
pip install inazumaq2n
```

Building from a checkout also needs a C11 compiler and the Python
development headers:

```sh
python -m pip install .
```

For images with more than four bands the package multiplies matrices with
the library NumPy was built with. If Intel MKL is installed as the `mkl`
package it is used instead, which was faster for images with many bands in
the [benchmark](benchmark.md). On Linux x86-64 it can be installed with the
package:

```sh
pip install "inazumaq2n[mkl]"
```

## The index of an image pair

```python
import numpy as np
from inazumaq2n import q2n

rng = np.random.default_rng(0)
reference = rng.uniform(0, 255, (256, 256, 8))
test = reference + rng.normal(0, 10, reference.shape)

score = q2n(reference, test)                       # blocks of 32, no overlap
score = q2n(reference, test, ws=64, stride=32)     # blocks of 64, half overlapping
```

An image is an array of shape `(height, width, bands)`, or `(height, width)`
for one band. The reference image comes first: it supplies the normalization
of both images, so `q2n(a, b)` and `q2n(b, a)` differ. `ws` is the side of a
block and `stride` the distance between blocks, `ws` by default. The image is
extended at the bottom and at the right by reflection where the last blocks
need it.

Arrays of `float64`, `float32`, `uint8` and `uint16` are read in place when
both images have the same type and are C-contiguous. Anything else that
holds real numbers is accepted and converted to `float64`: other integer
types, lists, views with strides, Fortran-ordered arrays, and pairs of
different types.

```python
image = np.asarray(some_uint16_sensor_image)       # (rows, cols, bands), uint16
score = q2n(image, fused)                          # no float64 copy is made
```

A channel-first array, as PyTorch uses, has to be moved to channel-last
first, for example with `np.moveaxis(tensor.numpy(), 0, -1)`. The result is
then not contiguous and is copied once; `np.ascontiguousarray` beforehand
makes that copy explicit.

## The value of every block

```python
from inazumaq2n import q2n_map

block_map = q2n_map(reference, test, ws=32)        # shape (8, 8)
block_map.mean()                                   # equals q2n(reference, test, ws=32)
```

Entry `(r, c)` is the value of the block whose origin is
`(r * stride, c * stride)`. The map locates the regions where the test image
departs from the reference.

## Many pairs of one shape

`q2n` sets up the block geometry and work arrays on every call. A
`Q2nPlan` keeps them:

```python
from inazumaq2n import Q2nPlan

with Q2nPlan(reference.shape, ws=32) as plan:
    scores = [plan.score(reference, candidate) for candidate in candidates]
    worst = plan.map(reference, candidates[int(np.argmin(scores))])
```

For images with one to four bands of 512 by 512 pixels, where a call takes
well under a millisecond per band, a plan removes a noticeable share of the
time; see the [benchmark](benchmark.md). A plan scores one pair at a time.
To score pairs from several threads, give each thread its own plan.

## Threads

```python
score = q2n(reference, test, workers=4)
```

`workers` divides the tiles of the image among threads. It defaults to one.
With more than four bands, the moments and NumPy as the matrix-product
library, the library has threads of its own, and the two levels multiply. Set the
library to one thread in that case, with `OPENBLAS_NUM_THREADS=1` in the
environment or with
[threadpoolctl](https://github.com/joblib/threadpoolctl):

```python
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1, user_api="blas"):
    score = q2n(reference, test, workers=4)
```

With MKL the package sets one MKL thread per worker itself, for its own
calls only.

## What the package selected

```python
from inazumaq2n import runtime_info

runtime_info()
# {'version': '0.1.0', 'simd': 'avx512', 'direct_kernels': True,
#  'backend': 'mkl', 'backend_version': '2026.2.1', 'numpy': '2.5.4',
#  'fastcd': '1.5.0', 'fastcd_kernel': 'avx512'}
```

`simd` is the instruction set of the compiled kernels, chosen for the
processor when the package is imported. `backend` is the matrix-product
library that `backend="auto"` uses, and `fastcd_kernel` the kernel that
fastCD runs its fused sums with.

## Two algorithms

```python
score = q2n(reference, test)                        # algorithm="auto"
score = q2n(reference, test, algorithm="moments")   # real cross moments
score = q2n(reference, test, algorithm="fastcd")    # fused sums of fastCD

with Q2nPlan(reference.shape) as plan:
    plan.algorithm                                  # "moments" or "fastcd"
```

The package evaluates a block either from real cross moments or from one
fused sum of products of fastCD, and by default selects the one that was
measured to be faster for the image and the blocks: the fused sums for
images with eight bands or more whose blocks do not overlap, and for
overlapping blocks in high dimensions; see the
[API reference](api.md#algorithms).

The two give the same index to rounding error, but not the same bits, and
which one `"auto"` selects depends on the processor. To compare scores
between machines to the last digit, pass `algorithm` explicitly, and with
`algorithm="moments"` also `backend="numpy"`: MKL and the library of NumPy
round differently too.

## Things to know about the index

- Values are not clipped. With more than eight bands a block, and so the
  index, can exceed one.
- An image compared with itself scores one, except where a band of the
  reference has mean exactly zero over a block; see
  [Conventions and results](mathematics.md).
- Images should be finite. A block that contains a NaN or an infinity has
  the value NaN, and so has the index.
- A reference band that is exactly constant over a block is normalized by
  the machine epsilon, and the value of the block changes completely if the
  band is perturbed by a single rounding error; see
  [Numerical stability](stability.md).

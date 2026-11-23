# inazumaQ2n - v0.1.0 (unreleased)

inazumaQ2n is a fast implementation of the Q2n image quality index for images with any number of bands, written for researchers and engineers who evaluate pansharpening, fusion, and restoration of multispectral and hyperspectral images. Q2n compares a test image with a reference image block by block, treating every pixel as an element of a Cayley-Dickson algebra (the complex numbers for two bands, the quaternions for four, the octonions for eight, and so on), and its definition multiplies the pixels of every block in that algebra.

inazumaQ2n computes the same index **without forming a single product of pixels**: it evaluates each block either from real moments, one small matrix of cross moments per tile contracted with the signs of the algebra, or, for images with many bands, from one *fused sum* of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD), which adds up the products of the pixels of a block in `O(N log N)` operations per pixel without forming them. Both run in compiled kernels that are selected for the processor at run time, and the faster of the two is chosen for each image.

It is installed as the Python package `inazumaq2n`, whose only dependency is NumPy.

The canonical source repository is
[git.tanuki-cd.com/algebraity/inazumaq2n](https://git.tanuki-cd.com/algebraity/inazumaq2n).

## Definition

A pixel with `B` bands is an element of the Cayley-Dickson algebra of dimension `N`, the least power of two that is at least `B`, in the convention of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD). In each block of `ws` by `ws` pixels, the reference image `X` and the test image `Y` are normalized band by band with the mean and the unbiased standard deviation of the reference, giving `x` and `y`, and the index is

```text
Q2n(X, Y) := mean over blocks of |q|,
q := 2 bias cov(x, conj(y)) / (var(x) + var(y)),   bias := 2 |mean x| |mean conj(y)| / (|mean x|^2 + |mean conj(y)|^2),
```

where the covariance is the hypercomplex one, with products in the algebra, the variances are sums over the bands, and `|.|` is the Euclidean norm.

Blocks start at multiples of `stride` and are completed by symmetric reflection at the bottom and right edges. This is the index of Garzelli and Nencini (2009) as the reference MATLAB implementation defines it, in double precision.

[Conventions and results](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/mathematics.md) gives the full definition, including its treatment of bands with zero mean or zero variance, and derives the evaluation from moments and from fused sums.

## Installation

inazumaQ2n requires CPython 3.10 or newer and NumPy 1.26 or newer. Once it is released on PyPI, which it has not been yet, it will be installed with `pip` by running:

```sh
pip install inazumaq2n
```

Wheels will be provided for CPython 3.10 through 3.14 on Linux, macOS, and Windows. To build and install it from a checkout of this repository, which also requires a C11 compiler and the Python development headers, run:

```sh
python -m pip install .
```

The build compiles fastCD 1.5.0 into the extension module from the sources in [vendor/fastcd](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/vendor/fastcd), so neither fastCD nor pyfastcd needs to be installed. For images with more than four bands, Intel MKL is used for the matrix products of the moments when the `mkl` package is installed, which the extra `mkl` does on Linux x86-64.

## Usage

After importing `inazumaq2n`, you can score an image pair with `q2n`, obtain the value of every block with `q2n_map`, and score many pairs of one shape with a `Q2nPlan`. Here is an example with an image of eight bands:

```python
import numpy as np
from inazumaq2n import q2n, q2n_map, Q2nPlan, runtime_info

rng = np.random.default_rng(0)
reference = rng.uniform(0, 255, (256, 256, 8))    # (height, width, bands)
test = reference + rng.normal(0, 10, reference.shape)

score = q2n(reference, test)                       # blocks of 32, no overlap
score = q2n(reference, test, ws=64, stride=32)     # overlapping blocks
score = q2n(reference, test, algorithm="fastcd")   # or "moments"; "auto" selects
block_map = q2n_map(reference, test)               # the value of every block, shape (8, 8)

with Q2nPlan(reference.shape, ws=32) as plan:      # many pairs of one shape
    scores = [plan.score(reference, test + k) for k in range(10)]

runtime_info()                                     # instruction sets and libraries in use
```

`q2n` returns the index and `q2n_map` the value of every block, while a `Q2nPlan` keeps the block geometry and the work arrays for scoring many pairs of one shape. Images of `float64`, `float32`, `uint8`, and `uint16` samples are read in place, and any other real array is converted. See the [quick start](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/quickstart.md) and the [API reference](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/api.md) for more information.

## Benchmark

The benchmark compares inazumaQ2n with the published code that computes Q2n: the MATLAB pansharpening toolbox, run unchanged under GNU Octave, and two Python ports of it, PanCollection and the hyperspectral pansharpening toolbox, all fetched at pinned commits and verified before they are run. It also times the definition evaluated pixel by pixel and the two algorithms of inazumaQ2n against each other, which sets the automatic choices of the package. The [benchmark documentation](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md) describes the method, gives the full results, and explains how to rerun them.

Median milliseconds on 2026-11-21 on one logical CPU of an Intel Xeon Gold 6148, for images of 16-bit samples with `B = N` bands and blocks of 32 without overlap, the setting of the evaluation script of the toolbox, where the factor in parentheses is the time divided by that of inazumaQ2n:

| Bands | image | inazumaQ2n | MATLAB toolbox in Octave | PanCollection | Hyperspectral toolbox, PyTorch |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | 512 x 512 | 0.41 | 622 (1,505x) | 59.0 (143x) | 142 (343x) |
| 4 | 512 x 512 | 1.54 | 1,189 (771x) | 148 (96x) | 283 (184x) |
| 8 | 512 x 512 | 4.63 | 2,591 (560x) | 422 (91x) | 754 (163x) |
| 16 | 512 x 512 | 7.94 | 6,461 (813x) | 1,306 (164x) | 2,366 (298x) |
| 32 | 256 x 256 | 4.03 | 4,176 (1,037x) | 1,072 (266x) | 2,107 (523x) |
| 64 | 256 x 256 | 11.6 | 13,142 (1,132x) | 3,943 (340x) | 8,033 (692x) |
| 128 | 128 x 128 | 7.70 | 11,020 (1,430x) | 3,792 (492x) | 7,874 (1,022x) |
| 256 | 128 x 128 | 19.4 | 41,076 (2,117x) | 14,805 (763x) | 31,483 (1,622x) |
| 512 | 64 x 64 | 9.64 | 38,552 (4,001x) | 14,711 (1,527x) | 31,355 (3,254x) |
| 1024 | 64 x 64 | 27.8 | 151,316 (5,439x) | 57,249 (2,058x) | 122,969 (4,420x) |

inazumaQ2n is 91 to 2,100 times faster than PanCollection, 160 to 4,400 times faster than the PyTorch toolbox on one thread, and 560 to 5,400 times faster than the MATLAB toolbox under Octave, which interprets the toolbox without the compiler of MATLAB. Its values agree with those of the MATLAB toolbox to `1.7e-15`, while the two Python ports differ from the toolbox in details of their own (see [Against the published toolboxes](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md#against-the-published-toolboxes)).

## Numerical stability

inazumaQ2n forms the covariance of a block from centered quantities, so it never subtracts large uncentered sums, and it evaluates again from the normalized samples the blocks where its identities may fail in floating point. Measured against exact values on 8,988 block pairs, with the error `eta = |q_hat - q| / u` of the value of a block and `u = 2^-53`, the median error on typical inputs, with blocks of 32 by 32 pixels, is:

| Bands | 1 | 3 | 4 | 8 | 16 | 64 | 159 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inazumaQ2n, moments | 1.19 | 1.00 | 0.96 | 2.41 | 1.82 | 2.32 | 2.96 | 4.26 |
| inazumaQ2n, fused sums of fastCD | 4.01 | 2.23 | 1.98 | 1.51 | 1.43 | 1.89 | 3.32 | 3.14 |
| per pixel, pyfastcd quasilinear | 1.61 | 1.66 | 1.89 | 1.77 | 1.63 | 1.86 | 2.03 | 2.10 |
| per pixel, pyfastcd direct | 1.61 | 1.50 | 1.83 | 1.83 | 1.78 | 1.88 | 2.47 | 1.92 |

On typical inputs every evaluation is accurate to a few units in the last place, with largest errors of 15.9 for the moments, 13.5 for the fused sums where `auto` can select them, and 10.8 for the per-pixel evaluation, and on inputs that strain the definition, such as means a million times the variation or samples whose products underflow, inazumaQ2n is the more accurate by orders of magnitude, while on 16-bit integer images its long sums reach an error of 155, about `1.7e-14`. These figures describe the inputs tested, not a worst case, and [Numerical stability](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/stability.md) gives the method and the full results.

## Documentation

- [Quick start](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/quickstart.md)
- [Python API reference](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/mathematics.md)
- [Benchmarks](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md)
- [Numerical stability](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/stability.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/LICENSE) and is presented with no warranty or guarantees. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/CHANGELOG.md).

The Q2n index is due to A. Garzelli and F. Nencini, *Hypercomplex quality assessment of multi/hyperspectral images*, IEEE Geoscience and Remote Sensing Letters 6(4), 2009. The definition implemented here is that of the MATLAB pansharpening toolbox of G. Vivone et al., *A critical comparison among pansharpening algorithms*, IEEE Transactions on Geoscience and Remote Sensing 53(5), 2015. The Cayley-Dickson conventions, the fused sums, and the products of the per-pixel evaluation in the benchmark are those of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) and its quasilinear algorithm, [*Quasilinear multiplication in the Cayley-Dickson tower*](https://arxiv.org/abs/2609.11588).

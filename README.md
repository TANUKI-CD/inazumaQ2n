![inaaumaQ2n](https://git.tanuki-cd.com/algebraity/inazumaq2n/raw/branch/main/assets/inazumaq2n.png)
# inazumaQ2n - v0.1.0 (unreleased)

inazumaQ2n is a fast implementation of the Q2n image quality index for images with any number of bands, written for researchers and engineers who evaluate pansharpening, fusion and restoration of multispectral and hyperspectral images. Q2n compares a test image with a reference image block by block, treating every pixel as an element of a Cayley–Dickson algebra (the complex numbers for two bands, the quaternions for four, the octonions for eight, and so on), and its definition multiplies the pixels of every block in that algebra. inazumaQ2n computes the same index **without forming a single product of pixels**: it evaluates each block either from real moments, one small matrix of cross moments per tile contracted with the signs of the algebra, or, for images with many bands, from one *fused sum* of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD), which adds up the products of the pixels of a block in `O(N log N)` operations per pixel without forming them. Both run in compiled kernels that are selected for the processor at run time, and the faster of the two is chosen for each image. It is installed as the Python package `inazumaq2n`, whose only dependency is NumPy.

The canonical source repository is
[git.tanuki-cd.com/algebraity/inazumaq2n](https://git.tanuki-cd.com/algebraity/inazumaq2n).

## Definition

```text
Q2n(X, Y) := mean over blocks of |q|,
q := 2 bias cov(x, conj(y)) / (var(x) + var(y)),   bias := 2 |mean x| |mean conj(y)| / (|mean x|^2 + |mean conj(y)|^2)
```

A pixel with `B` bands is an element of the Cayley–Dickson algebra of dimension `N`, the least power of two that is at least `B`, in the convention of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD). In each block of `ws` by `ws` pixels, `x` and `y` are the reference and the test image normalized band by band with the mean and the unbiased standard deviation of the reference, the covariance is the hypercomplex one, and the variances are sums over the bands. Blocks start at multiples of `stride` and are completed by symmetric reflection at the bottom and right edges. This is the index of Garzelli and Nencini (2009) as the reference MATLAB implementation defines it, in double precision. [Conventions and results](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/mathematics.md) gives the full definition, including its treatment of bands with zero mean or zero variance, and derives the evaluation from moments.

## Installation

inazumaQ2n requires CPython 3.10 or newer and NumPy 1.26 or newer. When it is released on PyPI (not yet released), install it with `pip`, run:

```sh
pip install inazumaq2n
```

Wheels are provided for CPython 3.10 through 3.14 on Linux, macOS and Windows. To build and install it from a checkout of this repository, which also requires a C11 compiler and the Python development headers, run:

```sh
python -m pip install .
```

The build compiles fastCD 1.5.0 into the extension module from the sources in [vendor/fastcd](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/vendor/fastcd), so neither fastCD nor pyfastcd needs to be installed. For images with more than four bands, Intel MKL is used for the matrix products of the moments when the `mkl` package is installed, which `pip install "inazumaq2n[mkl]"` does on Linux x86-64.

## Usage

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

`q2n` returns the index and `q2n_map` the value of every block. A `Q2nPlan` keeps the block geometry and work arrays for scoring many pairs of one shape. Images of `float64`, `float32`, `uint8` and `uint16` samples are read in place, and any other real array is converted. See the [quick start](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/quickstart.md) and the [API reference](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/api.md).

## Benchmark

The natural way to compute Q2n is to follow its definition: normalize every block, multiply every reference pixel by the conjugate of the corresponding test pixel in the Cayley–Dickson algebra, and average. [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) is a C library for fast Cayley–Dickson multiplication, and [pyfastcd](https://pypi.org/project/pyfastcd) is its Python package; it multiplies in `O(N^2)` operations with its direct algorithm and in `O(N log N)` with its quasilinear algorithm. The benchmark compares inazumaQ2n with that evaluation of the definition, vectorized in NumPy over the pixels, the bands and batches of blocks, with every pixel product computed by pyfastcd 1.4.0, once with each algorithm. The comparison is part of the benchmark, in [benchmarks/baseline.py](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmarks/baseline.py).

The table gives the median milliseconds of a complete call over 9 rounds, measured on 2026-11-13 on one logical CPU of an Intel Xeon Gold 6148 with AVX-512, using Python 3.14.8, NumPy 2.5.4 and MKL 2026.2.1, on a shared host. Every call includes the handling of the input, the allocation of work arrays and, for inazumaQ2n, the construction of its plan. The last column divides the time of the faster per-pixel evaluation by that of inazumaQ2n. Blocks are 32 by 32 pixels without overlap unless stated. The protocol, the results for AVX2 and for several threads, and the comparison of the matrix-product libraries are in [Benchmark](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md).

| Image | inazumaQ2n | per pixel, quasilinear | per pixel, direct | speedup |
| --- | ---: | ---: | ---: | ---: |
| gray 512 x 512, uint8 | 0.343 | 22.6 | 22.8 | 66 |
| RGB 512 x 512, uint8 | 0.910 | 67.7 | 68.6 | 74 |
| RGB 512 x 512, uint8, block 64, stride 32 | 1.07 | 215.0 | 215.8 | 201 |
| 4 bands 512 x 512, uint16 | 2.38 | 65.9 | 65.6 | 28 |
| 8 bands 512 x 512, uint16 | 13.5 | 107.7 | 107.0 | 7.9 |
| 16 bands 512 x 512, float64 | 17.5 | 163.4 | 181.9 | 9.4 |
| 16 bands 512 x 512, float64, block 64, stride 32 | 23.7 | 565.1 | 653.2 | 24 |
| 32 bands 256 x 256, float64 | 10.6 | 83.3 | 104.1 | 7.8 |
| 159 bands 200 x 200, float64 | 66.9 | 478.8 | 1,268.7 | 7.2 |
| 159 bands 200 x 200, float64, block 64, stride 32 | 100.7 | 2,643.4 | 5,694.7 | 26 |

inazumaQ2n was faster in all 9 rounds of every case. Its advantage is largest for images with at most four bands, where its kernels form the moments of a tile in two passes without a workspace or a matrix product, and for overlapping blocks, where it reads every pixel once however many blocks contain it.

That table was measured with the real moments alone. For images with more bands inazumaQ2n now also evaluates a block as one fused sum of fastCD 1.5.0, which is compiled into the package, and `algorithm="auto"` selects the faster of the two. The reference toolboxes instead multiply the pixels of every block with a recursive product that costs `N^2` multiplications per pixel; [benchmarks/toolbox.py](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmarks/toolbox.py) is a NumPy port of it. Median milliseconds on 2026-11-21 on the same core, for images with `B = N` bands, blocks of 32 without overlap and reused plans:

| Bands | image | toolbox recursion | moments | fused sums of fastCD | toolbox / selected |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 512 x 512 | 122 | **1.65** | 3.84 | 74 |
| 8 | 512 x 512 | 344 | 11.0 | **5.62** | 61 |
| 16 | 512 x 512 | 1,052 | 21.1 | **12.3** | 86 |
| 32 | 256 x 256 | 915 | 13.1 | **5.89** | 155 |
| 64 | 256 x 256 | 3,476 | 26.0 | **15.9** | 219 |
| 128 | 128 x 128 | 3,397 | 19.3 | **9.46** | 359 |
| 256 | 128 x 128 | 13,034 | 47.4 | **21.1** | 617 |
| 1024 | 64 x 64 | 50,799 | 136 | **34.9** | 1,456 |

The time that `auto` selects is bold. Without overlap the fused sums are 1.6 to 3.9 times faster than the moments from eight bands on with fastCD's AVX-512 kernel, and 1.2 to 2.9 times with its AVX2 kernel. Blocks that overlap share the matrices of their tiles with the moments but must each be summed in full, so with a stride of half the block the fused sums are selected from dimension 128 with AVX-512 and from 512 with AVX2; and since a fused sum costs the same for every band count of a dimension, they are selected only when the bands fill most of it. [Algorithms](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md#algorithms) gives the sweeps behind the selection.

### Against the published toolboxes

The same index can be computed with the code of the toolboxes themselves, and the benchmark does so: the `q2n` of the MATLAB pansharpening toolbox, run unchanged under GNU Octave, two Python ports of it, and a NumPy port of its recursion written for this repository. None of the external code is distributed here; [benchmarks/toolboxes.py](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmarks/toolboxes.py) fetches each at a pinned commit and verifies it before it is run.

Median milliseconds on 2026-11-21 on one logical CPU of the same host, for images of 16-bit samples with `B = N` bands and blocks of 32 without overlap, the setting of the evaluation script of the toolbox. The factor in parentheses is the time divided by that of inazumaQ2n, which selects its kernels for at most four bands at 2 and 4 bands and the fused sums of fastCD from 8 bands on:

| Bands | image | inazumaQ2n | NumPy port of the toolbox | MATLAB toolbox in Octave | PanCollection | Hyperspectral toolbox, PyTorch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 512 x 512 | 0.41 | 46.2 (112x) | 622 (1,505x) | 59.0 (143x) | 142 (343x) |
| 4 | 512 x 512 | 1.54 | 123 (79x) | 1,189 (771x) | 148 (96x) | 283 (184x) |
| 8 | 512 x 512 | 4.63 | 345 (75x) | 2,591 (560x) | 422 (91x) | 754 (163x) |
| 16 | 512 x 512 | 7.94 | 1,101 (139x) | 6,461 (813x) | 1,306 (164x) | 2,366 (298x) |
| 32 | 256 x 256 | 4.03 | 939 (233x) | 4,176 (1,037x) | 1,072 (266x) | 2,107 (523x) |
| 64 | 256 x 256 | 11.6 | 3,447 (297x) | 13,142 (1,132x) | 3,943 (340x) | 8,033 (692x) |
| 128 | 128 x 128 | 7.70 | 3,418 (444x) | 11,020 (1,430x) | 3,792 (492x) | 7,874 (1,022x) |
| 256 | 128 x 128 | 19.4 | 13,209 (681x) | 41,076 (2,117x) | 14,805 (763x) | 31,483 (1,622x) |
| 512 | 64 x 64 | 9.64 | 12,991 (1,348x) | 38,552 (4,001x) | 14,711 (1,527x) | 31,355 (3,254x) |
| 1024 | 64 x 64 | 27.8 | 51,983 (1,868x) | 151,316 (5,439x) | 57,249 (2,058x) | 122,969 (4,420x) |

inazumaQ2n is 75 to 1,900 times faster than the recursion in NumPy, 91 to 2,100 times faster than PanCollection, 160 to 4,400 times faster than the PyTorch toolbox on one thread, and 560 to 5,400 times faster than the MATLAB toolbox under Octave, and the factor grows with the number of bands. The times of the MATLAB toolbox are those of Octave, which interprets the recursion without the compiler of MATLAB; MATLAB itself was not available and would be faster.

All five compute the same values, as far as each implementation allows:

| Implementation | Source | Largest difference of a block from inazumaQ2n, in the table | Outside the setting of the table |
| --- | --- | ---: | --- |
| MATLAB pansharpening toolbox | `q2n.m` and its helpers, in the copy of [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox), run unchanged under GNU Octave | `1.7e-15` | Agrees to `1e-12` in every geometry of the tests: overlap, padding, any number of bands, zero and constant bands |
| NumPy port of the toolbox | [benchmarks/toolbox.py](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmarks/toolbox.py), in this repository | `1.3e-15` | Agrees to `1e-12` likewise |
| PanCollection | `my_q2n.py` of [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection), NumPy | `1.4e-07` | Fails when the number of bands is not a power of two; off by up to `6e-3` when blocks overlap or do not divide the image |
| Hyperspectral pansharpening toolbox | `Q2n` of [hyperspectral_pansharpening_toolbox](https://github.com/matciotola/hyperspectral_pansharpening_toolbox), PyTorch | `1.4e-07` | Agrees to `1e-7` with overlap and with other numbers of bands; off by up to `2e-4` when the image needs a different padding at the bottom and at the right |

The MATLAB toolbox and inazumaQ2n agree to rounding error, and the tests require it, for the moments and for the fused sums. The toolbox rounds its inputs to `uint16` and inazumaQ2n does not, so on images whose samples are not integers the toolbox agrees with inazumaQ2n applied to the rounded images. The two Python ports are close but not exact in their own setting: PanCollection normalizes with the population standard deviation where the toolbox uses the unbiased one, and the PyTorch toolbox stores the value of a block in single precision. See [Toolboxes](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md#toolboxes).

## Numerical stability

Accuracy was measured against an exact reference on 8,988 block pairs from 19 families, for 1 to 256 bands. The error of the computed value `q_hat` of a block with exact value `q` is reported as `eta = |q_hat - q| / u`, with `u = 2^-53`. The median `eta` on typical inputs, with blocks of 32 by 32 pixels, is:

| Bands | 1 | 3 | 4 | 8 | 16 | 64 | 159 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inazumaQ2n, moments | 1.19 | 1.00 | 0.96 | 2.41 | 1.82 | 2.32 | 2.96 | 4.26 |
| inazumaQ2n, fused sums of fastCD | 4.01 | 2.23 | 1.98 | 1.51 | 1.43 | 1.89 | 3.32 | 3.14 |
| per pixel, pyfastcd quasilinear | 1.61 | 1.66 | 1.89 | 1.77 | 1.63 | 1.86 | 2.03 | 2.10 |
| per pixel, pyfastcd direct | 1.61 | 1.50 | 1.83 | 1.83 | 1.78 | 1.88 | 2.47 | 1.92 |

Where `auto` can select the fused sums, from 5 to 256 bands, their median `eta` on typical inputs is 1.4 to 3.3 and their largest 13.5; they are at least as accurate as the moments in 59% to 79% of the blocks of every band count. No result of theirs is further from the exact value than `2.6e-14` outside the families whose mean is far from zero, where both routes return the same guarded value. With one to four bands, where `auto` always uses the moments, they are the less accurate of the two.

On typical inputs both inazumaQ2n and the per-pixel evaluation are accurate to a few units in the last place: with its moments, inazumaQ2n is the more accurate up to four bands and somewhat less accurate above, and the largest errors in these families are 15.9 for inazumaQ2n and 10.8 for the per-pixel evaluation. On 16-bit integer images the long sums of inazumaQ2n round the same way at every step, and its largest error rises to 155, about `1.7e-14`. On inputs that strain the definition it is the more accurate: its error stays at a few units where the mean of an image is a million times its variation (median 5.8 against 101), where products of samples underflow (1.9 against 7,500), where squares of samples overflow, and where a reference band has a mean of exactly zero; in the last two families nearly every per-pixel value was wrong. The compiled kernels give bit-identical results with AVX-512, AVX2, NEON and the baseline code, and an integer or single-precision image gives the same bits as its conversion to `float64`. The method, the corpus and the full results are in [Numerical stability](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/stability.md).

## Documentation

- [Quick start](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/quickstart.md)
- [Python API reference](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/mathematics.md)
- [Benchmark](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/benchmark.md)
- [Numerical stability](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/docs/stability.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/LICENSE) and is presented with no warranty or guarantees. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/CHANGELOG.md).

The Q2n index is due to A. Garzelli and F. Nencini, *Hypercomplex quality assessment of multi/hyperspectral images*, IEEE Geoscience and Remote Sensing Letters 6(4), 2009. The definition implemented here is that of the MATLAB pansharpening toolbox of G. Vivone et al., *A critical comparison among pansharpening algorithms*, IEEE Transactions on Geoscience and Remote Sensing 53(5), 2015. The Cayley–Dickson conventions, the fused sums, and the products of the per-pixel evaluation in the benchmark, are those of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD) and its quasilinear algorithm, [*Quasilinear multiplication in the Cayley–Dickson tower*](https://arxiv.org/abs/2609.11588).

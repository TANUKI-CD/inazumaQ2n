# Benchmarks

The value of every block of a Q2n index needs the hypercomplex covariance of the
normalized pixels of the block, and there are three ways to obtain it: by
multiplying the pixels in the Cayley-Dickson algebra, as the definition does,
or, as inazumaQ2n does, from a matrix of real cross moments, the *moments*, or
from one *fused sum* of products of fastCD (see [Conventions and
results](mathematics.md)). Their operation counts differ widely, from one
product in the algebra for every pixel of every block to `B^2` multiply-adds or
`O(N log N)` operations for each pixel, for `B` bands in an algebra of dimension
`N`, but a smaller operation count does not by itself make a call faster: the
time of a call also depends on how often each pixel is read, on the memory that
the work arrays occupy, on the matrix-product library, and, for small images, on
the fixed cost of a call from Python. The only way to know which evaluation is
faster for a given image is to time the implementations themselves.

The benchmark does this for two purposes. It justifies the automatic choices of
the package with reproducible data: the compiled kernels that need no matrix
product for images with at most four bands, the choice of `algorithm="auto"`
between the moments and the fused sums, and the choice of `backend="auto"` of
the library for the matrix products. It also measures how much faster inazumaQ2n
is than the evaluations that users would otherwise run, the definition evaluated
pixel by pixel and the published toolboxes that compute the index, and it checks
that their values agree. It does not try to prove that one evaluation is faster
than another in general; instead, it measures the implementations as they are,
on one host.

The measurements show that inazumaQ2n is 6.9 to 204 times faster than the
definition evaluated pixel by pixel with the products of fastCD, and 91 to 5,400
times faster than the published toolboxes, with values that agree with the
MATLAB toolbox to `1.7e-15`. Between its own algorithms, the kernels for at most
four bands are the fastest evaluation where they apply, and from eight bands on
the fused sums are 1.6 to 3.9 times faster than the moments when blocks do not
overlap, while the moments remain the faster for overlapping blocks below high
dimensions and for band counts that leave much of the dimension empty. The
sections below describe what is measured and how the timings are made, report
the results for complete calls, for the kernels, for the moments against the
fused sums, for the matrix-product libraries and the workers, and against the
published toolboxes, show how these results set the automatic choices, and end
with the limitations of the measurements and with instructions for running the
benchmark.

## What is measured

The driver `benchmarks/benchmark.py` times complete calls of the public
interface, in four experiments, called *protocols*, each of which answers one
question:

- **Complete calls** (`calls`): how long does a call of `q2n` take for typical
  images, compared with the definition evaluated pixel by pixel?
- **Kernels** (`kernels`): for images with one to four bands, how much faster
  are the kernels that need no matrix product than the route through a matrix
  product, in each instruction set?
- **Algorithms** (`algorithms`): which of the moments and the fused sums is
  faster, depending on the dimension, the number of bands, the overlap of the
  blocks, the sample type, the block size, and the number of workers, and how do
  both compare with the recursive product of the reference toolboxes?
- **Toolboxes** (`toolboxes`): how long do the published toolboxes take on the
  same images, and how far are their values from those of inazumaQ2n?

Every measurement is one call on a pair of images that already exist in memory,
timed with `time.perf_counter`. The conversion of the input, the allocation of
the work arrays, the moments or the fused sums, the evaluation of the blocks,
and the mean are inside the timed region, and so, for `q2n`, are the
construction and the release of its plan, while the variant `plan` reuses one
plan, constructed once outside the timed region.

## How timings are made

All four protocols prepare, time, and check their cases in the same way, so that
the variants of a case see the same images and the same conditions of the host.

**Images.** The images are uniform noise in `[0, 255]` and the same image with
Gaussian noise of standard deviation 20 added, clipped for integer types, from a
fixed seed. Running time does not depend on the values, except through the
blocks that inazumaQ2n evaluates a second time, of which these images have none.

**Rounds.** Every variant of a case is called once before the rounds begin, so
that nothing is compiled or loaded during a timed call, and then once in every
round, in an order that rotates from round to round, so that a change in the
load of the host falls on all variants alike. The report gives the median over
the rounds, and the output file keeps every observation and, for each variant,
the number of rounds in which it was slower than the reused plan.

**Verification.** Before timing, every variant of a case is evaluated and the
values are compared, and the run stops if two differ by more than `1e-12`.

**Threads.** The driver limits every matrix-product library in the process to
one thread, with threadpoolctl, so that the single-thread results are those of
one core, and `--workers` sets the number of workers of inazumaQ2n, while the
libraries stay at one thread each.

## The archived runs

All results on this page were measured on a Dell PowerEdge T440 with an Intel
Xeon Gold 6148, a Skylake-SP processor with AVX-512 and 1 MiB of second-level
cache for each core, under Linux 7.2.11, with GCC 16.2.1, CPython 3.14.8, NumPy
2.5.4 with its bundled OpenBLAS 0.3.34, MKL 2026.2.1, and pyfastcd 1.4.0.
inazumaQ2n 0.1.0, with fastCD 1.5.0 compiled into it, was built as a wheel is
built: for the baseline instruction set, with the kernels for wider instruction
sets selected at run time. The host is shared with other work, and no frequency
governor, turbo, or SMT settings were changed. The results come from three
archives, whose READMEs record the commands, and whose JSON files record every
observation and the hash of every measured source file:

| Archive | Protocols | Date |
| --- | --- | --- |
| [benchmark-results/t440-20261122/](../benchmark-results/t440-20261122/) | `calls`, `kernels`, and `calls` with 1, 2, and 4 workers | 2026-11-22 |
| [benchmark-results/fastcd-20261121/](../benchmark-results/fastcd-20261121/) | `algorithms`, with each kernel of fastCD | 2026-11-21 |
| [benchmark-results/toolboxes-20261121/](../benchmark-results/toolboxes-20261121/) | `toolboxes`, with NumPy 2.5.5, PyTorch 2.15.0, and GNU Octave 11.3.0 | 2026-11-21 |

## Complete calls against the definition

The definition of Q2n can be evaluated as it is written: in every block, the
bands of both images are normalized with the statistics of the reference, every
normalized reference pixel is multiplied by the conjugate of the normalized test
pixel in the Cayley-Dickson algebra, and the products are averaged. The
benchmark measures inazumaQ2n against that evaluation, as
`benchmarks/baseline.py` implements it with the help of
[fastCD](https://git.tanuki-cd.com/algebraity/fastCD), a C library for
Cayley-Dickson multiplication, and its Python package
[pyfastcd](https://pypi.org/project/pyfastcd), whose direct algorithm multiplies
in `O(N^2)` operations and whose quasilinear algorithm in `O(N log N)`. The
baseline computes every pixel product of a batch of blocks in one call of
pyfastcd, with one algorithm or the other, and does everything else in NumPy,
vectorized over the pixels, the bands, and the blocks of batches of about 8 MiB,
and its values agree with an independent evaluation of the definition to `1e-12`
in the tests.

The table gives the median milliseconds over 9 rounds on one logical CPU. `q2n`
includes the construction of its plan, `plan` reuses one, and the column
"algorithm" names the algorithm that `auto` selected, while the speedup divides
the time of the faster per-pixel evaluation by that of `q2n`. Blocks are 32 by
32 pixels without overlap unless a block and a stride are given.

| Image | `q2n` | plan | algorithm | per pixel, quasilinear | per pixel, direct | speedup |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| gray 512 x 512, uint8 | 0.348 | 0.231 | moments | 22.6 | 23.4 | 65.0 |
| RGB 512 x 512, uint8 | 0.888 | 0.758 | moments | 67.9 | 65.9 | 74.2 |
| RGB 512 x 512, uint8, block 64, stride 32 | 1.06 | 0.868 | moments | 220.0 | 215.1 | 204 |
| 4 bands 512 x 512, uint16 | 2.13 | 1.99 | moments | 66.2 | 63.9 | 30.1 |
| 8 bands 512 x 512, uint16 | 5.52 | 5.44 | fastcd | 104.4 | 106.2 | 18.9 |
| 16 bands 512 x 512, float64 | 12.5 | 12.3 | fastcd | 159.0 | 180.8 | 12.7 |
| 16 bands 512 x 512, float64, block 64, stride 32 | 19.0 | 22.2 | moments | 556.3 | 628.5 | 29.3 |
| 32 bands 256 x 256, float64 | 5.84 | 6.00 | fastcd | 75.9 | 97.9 | 13.0 |
| 32 bands 256 x 256, float64, block 64, stride 32 | 15.1 | 14.7 | moments | 310.4 | 401.0 | 20.5 |
| 159 bands 200 x 200, float64 | 69.2 | 70.9 | moments | 475.1 | 1,237.7 | 6.87 |
| 159 bands 200 x 200, float64, block 64, stride 32 | 104.4 | 99.9 | moments | 2,728.2 | 5,848.1 | 26.1 |

**Against the definition.** `q2n` and the plan were faster than both per-pixel
evaluations in all 9 rounds of every case, and the values of all four variants
agree to `4.4e-16` in every case.

**Why.** Three things account for the difference. inazumaQ2n forms no product of
pixels, so the hypercomplex work of a block is one contraction of a `B` by `B`
matrix or one fused sum, it does every step for each band in compiled code,
where the per-pixel evaluation spends most of its time for small `N` in NumPy
normalizing and rearranging the bands, and for overlapping blocks it reads every
pixel once, where the per-pixel evaluation normalizes and multiplies every pixel
once for each block that contains it, which is why its advantage is largest
there.

**The two product algorithms.** The quasilinear and direct products of pyfastcd
take about the same time up to eight bands, where the products are a small part
of the per-pixel evaluation, while from 32 bands on the quasilinear algorithm is
faster in every round, by a factor that reaches 2.6 at 159 bands.

**Plans.** A plan saves the setup of a call, about a tenth of a millisecond,
which matters for images with at most four bands, while for larger images it
makes no consistent difference: in this run, where every call followed a
per-pixel evaluation that touches hundreds of megabytes, the plan happened to be
slower than `q2n` for some of the images with 16 bands or more.

**Where the time goes.** With the fused sums, the kernel that writes the records
and sums the blocks takes more than 99% of a call. With the moments and more
than four bands, the profile of a call divides as follows:

| Image | centering | matrix products | evaluating blocks |
| --- | ---: | ---: | ---: |
| 16 bands 512 x 512, float64, block 64, stride 32 | 73% | 24% | 3% |
| 32 bands 256 x 256, float64, block 64, stride 32 | 61% | 34% | 4% |
| 159 bands 200 x 200, float64 | 39% | 58% | 4% |
| 159 bands 200 x 200, float64, block 64, stride 32 | 33% | 57% | 10% |

Up to about 32 bands the centering pass, which reads the image and writes the
workspace, takes most of the time, and it is limited by memory traffic more than
by arithmetic, while at 159 bands the matrix products take the larger share.

## The kernels for at most four bands

For images with one to four bands, the package computes the moments of a tile in
two passes over its pixels, with no workspace and no matrix product (see [The
kernels](mathematics.md#the-kernels)), instead of the route through a matrix
product that it takes for more bands. The `kernels` protocol times both routes
on the same images, as a reused plan over 15 rounds, where "matrix product" is
the general route forced on the same image with the faster of NumPy and MKL, and
the AVX2 column is the AVX2 kernel run on this AVX-512 processor:

| Image | block / stride | AVX-512 | AVX2 | matrix product |
| --- | --- | ---: | ---: | ---: |
| 1 band 512 x 512, uint8 | 32 / 32 | 0.201 | 0.258 | 2.954 |
| 2 bands 384 x 448, float64 | 32 / 32 | 0.373 | 0.463 | 2.153 |
| 2 bands 512 x 512, float32 | 64 / 32 | 0.572 | 0.812 | 4.279 |
| 3 bands 512 x 512, uint8 | 32 / 32 | 0.856 | 1.227 | 5.117 |
| 3 bands 95 x 117, float64 | 32 / 12 | 0.235 | 0.367 | 1.251 |
| 4 bands 512 x 512, uint16 | 32 / 32 | 1.124 | 1.937 | 6.159 |
| 4 bands 512 x 512, float64 | 32 / 32 | 1.616 | 2.423 | 5.064 |
| 4 bands 1024 x 1024, uint16 | 32 / 32 | 7.18 | 11.77 | 25.50 |
| 4 bands 300 x 300, float64 | 8 / 8 | 0.771 | 0.954 | 1.600 |

**The result.** Over the 16 cases of the protocol, the kernels are faster than
the matrix product by a geometric mean of 4.5 times with AVX-512 and 3.2 times
with AVX2, and by at least 2.1 and 1.7 times, and they were faster than both
libraries in all 15 rounds of every case. The NEON kernel is the same generated
code and gives the same bits, which was checked under emulation, but it has not
been timed.

## Moments and fused sums

The `algorithms` protocol is the measurement behind the choice of
`algorithm="auto"`. It times reused plans with `algorithm="moments"` and with
`algorithm="fastcd"`, and a plan with `algorithm="auto"`, to record what it
selects and how far the selected algorithm is from the faster one, and in its
dimension sweep it also times the recursive product of the reference toolboxes
as a complete call.

**The recursion of the toolboxes.** `benchmarks/toolbox.py` is a NumPy port of
the structure of the MATLAB implementation in the pansharpening toolbox, which
its Python ports keep: for every block it normalizes the bands one at a time and
multiplies every reference pixel by the conjugate of the test pixel with
`onion_mult2D`, the doubling recursion on arrays of pixels. A product in
dimension `N` is four products in dimension `N / 2`, so a pixel costs `N^2` real
multiplications, and every level of the recursion allocates its operands. The
port leaves out the rounding of the images to integers, which inazumaQ2n does
not reproduce, and its values agree with those of the package to `7e-16` in
every case of the run.

**The sweeps.** The protocol runs six sweeps over images of `float64` samples,
unless stated, whose side depends on the dimension so that every image exceeds
the second-level cache of a core: 512 pixels through `N = 16`, 256 through 64,
128 through 256, and 64 above.

| `--sweeps` | Cases |
| --- | --- |
| `dimension` | `B = N` bands for `N` from 2 to 1024, blocks of 32 without overlap, with the toolbox |
| `bands` | `B = N / 2 + 1`, `5N / 8`, `3N / 4`, and `7N / 8` for `N` from 8 to 1024: the moments do less work with fewer bands, the fused sums the same |
| `overlap` | Blocks of 32 with stride 16 and of 64 with stride 16, merged from 4 and 16 tiles, with `B = N` and, for 4 tiles, with fewer bands |
| `types` | `float32`, `uint16`, and `uint8` samples |
| `blocks` | Blocks of 8, 16, and 64 pixels without overlap |
| `workers` | 1, 2, 4, and 8 workers |

The toolbox is timed in three rounds at most, and a call of more than two
seconds is timed from its first call, while before timing, the values of the
plans must agree to `1e-12` and that of the toolbox to `1e-10`.

**Kernels of fastCD.** fastCD selects the kernel of its fused sums at run time
and offers no switch, so one build measures one kernel on a given processor. The
AVX2 and portable kernels were therefore measured on builds with fastCD's tuning
macro, which `CFLAGS` passes to `setup.py`: `CFLAGS=-DFASTCD_SUM_MAX_KERNEL=2`
withholds its AVX-512 kernel and `=1` leaves the portable one, and the builds
were run with `--simd avx2` and `--simd baseline`, so that the kernels of
inazumaQ2n match.

### The dimension sweep

Median milliseconds with the AVX-512 kernels, for `B = N` and blocks of 32
without overlap, on one logical CPU:

| N | image | toolbox | moments | fastcd | toolbox / moments | toolbox / fastcd | moments / fastcd | `auto` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 512 x 512 | 45.5 | 0.57 | 3.20 | 80.3 | 14.2 | 0.18 | moments |
| 4 | 512 x 512 | 122 | 1.65 | 3.84 | 73.9 | 31.8 | 0.43 | moments |
| 8 | 512 x 512 | 344 | 11.0 | 5.62 | 31.3 | 61.2 | 1.96 | fastcd |
| 16 | 512 x 512 | 1,052 | 21.1 | 12.3 | 49.9 | 85.6 | 1.71 | fastcd |
| 32 | 256 x 256 | 915 | 13.1 | 5.89 | 69.8 | 155 | 2.23 | fastcd |
| 64 | 256 x 256 | 3,476 | 26.0 | 15.9 | 134 | 219 | 1.64 | fastcd |
| 128 | 128 x 128 | 3,397 | 19.3 | 9.46 | 176 | 359 | 2.04 | fastcd |
| 256 | 128 x 128 | 13,034 | 47.4 | 21.1 | 275 | 617 | 2.25 | fastcd |
| 512 | 64 x 64 | 12,815 | 40.8 | 13.1 | 314 | 975 | 3.10 | fastcd |
| 1024 | 64 x 64 | 50,799 | 136 | 34.9 | 373 | 1,456 | 3.90 | fastcd |

**Against the recursion.** Both algorithms of the package are faster than the
recursion of the toolboxes at every dimension, the moments by 31 to 373 times,
the fused sums by 14 to 1,456 times, and the algorithm that `auto` selects by 61
times at least, and the advantage grows with the dimension, as `N^2` against
`B^2` with an optimized matrix product and against `N log N`.

**Between the two.** The kernels for at most four bands, which need neither a
workspace nor a product, are 2.3 to 5.6 times faster than a fused sum, while
from eight bands on the fused sums are 1.6 to 3.9 times faster than the moments,
in all 9 rounds at every dimension.

### Overlap and the kernels of fastCD

Overlapping blocks change the comparison, because the moments form the matrix of
a tile once for all the blocks that share it, while a pixel enters the fused sum
of every block that contains it, and the two share only the reading of the
pixels and the statistics of the tiles. The table gives the time of the moments
divided by that of the fused sums, with each kernel of fastCD, for `B = N`
without overlap and with blocks of 32 or 64 at a stride of 16, where `auto`
selects the fused sums at the ratios in bold:

| N | AVX-512, no overlap | AVX2, no overlap | portable, no overlap | AVX-512, 32 / 16 | AVX2, 32 / 16 | portable, 32 / 16 | AVX-512, 64 / 16 | AVX2, 64 / 16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.18 | 0.22 | 0.94 |  |  |  |  |  |
| 4 | 0.43 | 0.54 | 1.02 |  |  |  |  |  |
| 8 | **1.96** | **1.59** | 0.98 | 0.82 | 0.67 | 0.34 |  |  |
| 16 | **1.71** | **1.33** | 0.78 | 0.84 | 0.55 | 0.28 |  |  |
| 32 | **2.23** | **1.48** | 0.72 | 0.90 | 0.59 | 0.25 |  |  |
| 64 | **1.64** | **1.19** | 0.59 | 0.86 | 0.59 | 0.26 | 0.39 | 0.24 |
| 128 | **2.04** | **1.41** | 0.71 | **1.30** | 0.67 | 0.35 | 0.71 | 0.39 |
| 256 | **2.25** | **1.49** | 0.76 | **1.39** | 0.80 | 0.39 | 0.88 | 0.47 |
| 512 | **3.10** | **1.96** | 0.94 | **2.29** | **1.36** | 0.60 | **1.36** | 0.85 |
| 1024 | **3.90** | **2.87** | 1.46 | **3.12** | **2.02** | 1.03 | **1.67** | **1.09** |

**Without overlap** the fused sums are the faster from eight bands on with
either vector kernel. **With overlap** they are the faster from dimension 128
with AVX-512 and from 512 with AVX2 at a stride of half the block, and from 512
and 1024 at a quarter. **The portable kernel**, which on this processor rounds
products and sums separately and uses no vector instructions, is faster than the
moments at dimension 1024 only, and `auto` does not select it, while with the
portable kernels of both, images with two and four bands take about as long by
either route.

### Fewer bands than the dimension

A fused sum costs the same for every number of bands of a dimension, while the
moments cost less for fewer bands, so the fused sums need the bands to fill most
of the dimension. The same ratio for fewer bands than the dimension, without
overlap, where at `N = 8` the first two columns are the same image, with five
bands:

| N | AVX-512: N / 2 + 1 | 5N / 8 | 3N / 4 | 7N / 8 | N | AVX2: N / 2 + 1 | 5N / 8 | 3N / 4 | 7N / 8 | N |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | **1.10** | **1.10** | **1.44** | **1.35** | **1.96** | 0.96 | 0.96 | 1.05 | **1.20** | **1.59** |
| 16 | 1.06 | 1.17 | **1.20** | **1.31** | **1.71** | 0.78 | 0.87 | 1.01 | 1.03 | **1.33** |
| 32 | 0.96 | 1.10 | **1.60** | **1.53** | **2.23** | 0.64 | 0.72 | 0.91 | 0.99 | **1.48** |
| 64 | 0.93 | **1.11** | **1.33** | **1.44** | **1.64** | 0.63 | 0.73 | 0.86 | 0.94 | **1.19** |
| 128 | 0.87 | **1.17** | **1.33** | **1.71** | **2.04** | 0.60 | 0.70 | 0.83 | 1.13 | **1.41** |
| 256 | 1.13 | **1.44** | **1.57** | **1.93** | **2.25** | 0.68 | 0.94 | 0.97 | **1.29** | **1.49** |
| 512 | 1.10 | **1.51** | **1.84** | **2.34** | **3.10** | 0.70 | 0.95 | **1.24** | **1.53** | **1.96** |
| 1024 | **1.34** | **1.97** | **2.60** | **3.22** | **3.90** | 0.96 | **1.25** | **1.77** | **2.22** | **2.87** |

**The share of the dimension.** The fused sums need five eighths to three
quarters of the dimension with AVX-512, and nearly all of it with AVX2 below
dimension 256, while with a stride of half the block the AVX-512 kernel needs
`B = N` at dimension 128, seven eighths at 256, three quarters at 512, and five
eighths at 1024.

**Sample types and block sizes.** With narrower samples and other block sizes
the fused sums gain as much or more: over the `types` and `blocks` sweeps the
ratio is 1.6 to 3.5 and 1.7 to 7.4 with AVX-512, and 1.1 to 2.4 and 1.2 to 4.9
with AVX2, for the dimensions 8, 32, 128, and 512.

### Workers

Median milliseconds of the two algorithms with several workers on eight physical
cores:

| N | image | workers | moments | fastcd | moments / fastcd |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | 512 x 512 | 1 | 55.6 | 30.3 | 1.83 |
| 32 | 512 x 512 | 2 | 34.9 | 16.4 | 2.12 |
| 32 | 512 x 512 | 4 | 20.1 | 6.56 | 3.06 |
| 32 | 512 x 512 | 8 | 14.7 | 5.14 | 2.86 |
| 256 | 256 x 256 | 1 | 177 | 76.7 | 2.31 |
| 256 | 256 x 256 | 2 | 132 | 48.8 | 2.69 |
| 256 | 256 x 256 | 4 | 76.7 | 28.0 | 2.74 |
| 256 | 256 x 256 | 8 | 55.2 | 22.6 | 2.44 |

**Scaling.** The fused sums keep or widen their advantage with more workers, by
factors of 2.1 to 3.1 with two or more workers, against 1.8 and 2.3 with one.

## NumPy and MKL

For images with more than four bands, the moments take the cross moments of each
tile as a matrix product, from NumPy, with the BLAS it was built with, or from
Intel MKL when the `mkl` package is installed. The `calls` protocol times a
reused plan with each library in the same rounds, for the images where `auto`
selects the moments:

| Image | NumPy (OpenBLAS 0.3.34) | MKL 2026.2 | rounds won by MKL |
| --- | ---: | ---: | ---: |
| 16 bands 512 x 512, float64, block 64, stride 32 | 22.4 | 22.2 | 6 / 9 |
| 32 bands 256 x 256, float64, block 64, stride 32 | 16.0 | 14.7 | 9 / 9 |
| 159 bands 200 x 200, float64 | 83.0 | 70.9 | 9 / 9 |
| 159 bands 200 x 200, float64, block 64, stride 32 | 118.8 | 99.9 | 9 / 9 |

**The result.** MKL is 17% to 19% faster at 159 bands, where the matrix products
are most of the time, and 8.5% faster at 32 bands, while at 16 bands, where the
centering of the samples dominates, the two libraries are within the variation
of the host.

## Workers

The `calls` protocol was run again without the per-pixel evaluation, with 1, 2,
and 4 workers on four physical cores, each worker count in a separate run, and
the table gives the median milliseconds of a reused plan over 9 rounds:

| Image | algorithm | 1 worker | 2 workers | 4 workers |
| --- | --- | ---: | ---: | ---: |
| gray 512 x 512, uint8 | moments | 0.363 | 0.410 | 0.405 |
| RGB 512 x 512, uint8 | moments | 0.649 | 0.925 | 0.778 |
| 4 bands 512 x 512, uint16 | moments | 1.044 | 1.501 | 0.767 |
| 8 bands 512 x 512, uint16 | fastcd | 3.05 | 2.04 | 1.23 |
| 16 bands 512 x 512, float64 | fastcd | 11.79 | 7.53 | 3.38 |
| 32 bands 256 x 256, float64 | fastcd | 6.22 | 3.27 | 1.86 |
| 159 bands 200 x 200, float64 | moments | 70.04 | 52.60 | 27.37 |
| 159 bands 200 x 200, float64, block 64, stride 32 | moments | 98.49 | 74.23 | 44.70 |

**The result.** Four workers are 2.2 to 3.5 times faster than one for images
with more than four bands, while for calls that take a millisecond or less,
handing the tiles to threads costs about as much as it saves, and workers should
be left at one. The 8-band image took 3.05 ms with one worker here and 5.44 ms
in the `calls` run, where every call followed a per-pixel evaluation, which
shows how much a call of a few milliseconds depends on the state of the caches.

## Against the published toolboxes

The comparisons above use code written for this repository, the definition
evaluated with the products of pyfastcd and a NumPy port of the recursion of the
toolboxes. Users of Q2n, however, compute it with the published code of a few
toolboxes: the reference MATLAB implementation in the pansharpening toolbox of
Vivone et al., and Python ports of it in frameworks for pansharpening with deep
learning. The `toolboxes` protocol runs that published code itself and records
two things for it, how long it takes and how far its values are from those of
inazumaQ2n, so that it measures what a user of those toolboxes gains and
confirms that the values agree.

**The implementations.** `benchmarks/toolboxes.py` lists each implementation as
files of a public repository at a pinned commit, with the SHA-256 of every file.
`python benchmarks/toolboxes.py fetch` downloads them into
`benchmarks/.toolboxes`, which is not committed or distributed, and nothing is
run that does not match its hash.

| Name | Implementation | Runs with |
| --- | --- | --- |
| `matlab` | `q2n.m`, `onions_quality.m`, `onion_mult.m`, `onion_mult2D.m`, and `norm_blocco.m` of the MATLAB pansharpening toolbox, in the copy of [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) | GNU Octave |
| `pancollection` | `my_q2n.py` of [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection), a NumPy port | NumPy, and PyTorch, which the file imports |
| `hyperspectral` | `Q2n` of the [hyperspectral pansharpening toolbox](https://github.com/matciotola/hyperspectral_pansharpening_toolbox), a PyTorch port | PyTorch and torchvision |

**Running the MATLAB code.** The MATLAB code runs unchanged under GNU Octave. It
calls `mean2` and `std2` of the Image Processing Toolbox, which Octave has only
with its image package, so `benchmarks/octave/shims` supplies the two where they
are missing. The driver passes the images as doubles and times the call of `q2n`
inside the interpreter, so that neither the start of the interpreter nor the
transfer of the images is counted. The toolbox converts both images to `uint16`
first, so the comparison uses images of `uint16` samples, on which it and
inazumaQ2n compute the same index (see [The MATLAB toolbox and other
implementations](mathematics.md#the-matlab-toolbox-and-other-implementations)).

The table gives the median milliseconds on one thread, for `B = N` bands and
blocks of 32 without overlap, the setting of the evaluation script of the
toolbox. inazumaQ2n is a reused plan with `algorithm="auto"`, which selects the
kernels for at most four bands at 2 and 4 bands and the fused sums from 8 on,
and the factor in parentheses is the time divided by that of inazumaQ2n.

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

**The result.** inazumaQ2n is 75 to 1,900 times faster than the recursion in
NumPy, 91 to 2,100 times faster than PanCollection, 160 to 4,400 times faster
than the PyTorch toolbox, and 560 to 5,400 times faster than the MATLAB toolbox
under Octave, and the factor grows with the number of bands.

**The four implementations of the recursion.** The times of the MATLAB toolbox
are those of Octave, which interprets the recursion without the compiler of
MATLAB, so it takes 3 to 13 times as long as the same recursion in NumPy, the
more the smaller the dimension, where the calls and not the arithmetic are the
cost; MATLAB was not available for this measurement and would be faster than
Octave. The NumPy port is the fastest of the four and therefore the most
favorable comparison for the recursion, which is why the `algorithms` protocol
uses it, while PanCollection, the same recursion in NumPy, takes 1.1 to 1.3
times as long as the port, and the PyTorch toolbox, on one thread and on the
CPU, 2.1 to 3.1 times.

### Agreement

The largest difference of the value of a block from that of inazumaQ2n, over the
dimensions of the table, and what `python benchmarks/toolboxes.py report` and
the tests find outside its setting:

| Implementation | Largest difference of a block from inazumaQ2n, in the table | Outside the setting of the table |
| --- | ---: | --- |
| MATLAB pansharpening toolbox, under GNU Octave | `1.7e-15` | Agrees to `1e-12` in every geometry of the tests: overlap, padding, any number of bands, and zero and constant bands |
| NumPy port of the toolbox, [benchmarks/toolbox.py](../benchmarks/toolbox.py) | `1.3e-15` | Agrees to `1e-12` likewise |
| PanCollection | `1.4e-07` | Fails when the number of bands is not a power of two; off by up to `6e-3` when blocks overlap or do not divide the image |
| Hyperspectral pansharpening toolbox | `1.4e-07` | Agrees to `1e-7` with overlap and with other numbers of bands; off by up to `2e-4` when the image needs a different padding at the bottom and at the right |

**The MATLAB toolbox.** Run as it is distributed, the MATLAB toolbox agrees with
inazumaQ2n to rounding error at every dimension from 2 to 1024, and
`benchmarks/test_toolboxes.py` requires that agreement to `1e-12`, for the
moments and for the fused sums: for 1 to 16 bands, with and without overlap,
with padding at the bottom and the right, with a stride that does not divide the
block, with 8- and 16-bit samples, and with bands that are zero, constant, or
identical.

**The Python ports.** The two ports are not exact, each for a reason that can be
read in its source, and these are the differences that the tests and `report`
exhibit on the pinned commits, stated to explain the numbers, not as a review of
either project:

- PanCollection normalizes every band with the population standard deviation,
  `x.std()`, where `std2` of the toolbox divides by `P - 1`, an effect of `1e-7`
  or less in the setting of the table. Its padding of the image does not reflect
  the rows and columns as the toolbox does, which changes the blocks at the
  bottom and the right when blocks overlap or do not divide the image, and its
  padding of the bands raises an error for a number of bands that is not a power
  of two.
- The hyperspectral pansharpening toolbox writes the value of every block into a
  single-precision tensor, which limits its agreement to about `1e-7`. It pads
  the bottom of the image by the amount meant for the right and the right by the
  amount meant for the bottom, which changes the values when the two differ, so
  an image that needs the same padding on both sides agrees. It also normalizes
  a block in place after converting it to double, and since for images passed as
  double tensors that conversion does not copy, overlapping blocks would
  overwrite each other, so the driver passes single-precision tensors, as the
  toolbox itself does.

## How the results set the defaults

Each of the three automatic choices of the package follows a rule drawn from the
results above.

**Kernels for at most four bands.** Images with one to four bands use the
kernels that need no matrix product wherever the processor has a fused
multiply-add instruction, because they were faster than the matrix product in
every case and every round of the `kernels` protocol.

**`algorithm="auto"`.** `src/inazumaq2n/_auto.py` holds, for each vector kernel
of fastCD, each measured number of tiles per block (1, 4, and 16), and each
dimension, the least share of the dimension that the bands must fill for the
fused sums to have been at least 1.1 times as fast as the moments, and `auto`
selects the fused sums from that share on. A number of tiles that was not
measured uses the entry of the next larger one, more than 16 tiles select the
moments, and a dimension above 1024 uses the entry of 1024, while images with at
most four bands, and every image where fastCD runs its portable kernel, use the
moments. In the 180 cases of the two single-core runs with vector kernels,
`auto` selected the faster algorithm in 168, and in the other 12 the algorithm
it selected took at most 1.17 times as long as the faster one.

**`backend="auto"`.** The moments use MKL for their matrix products whenever it
is installed, because it was as fast as NumPy with OpenBLAS or faster in every
measured case, and markedly faster where the matrix products dominate, while
`backend="numpy"` keeps the library of NumPy.

These rules reflect measurements of one processor with MKL as the matrix-product
library, so they should be reviewed against the results of any new run, and on
another machine `algorithm="moments"` or `algorithm="fastcd"` overrides the
choice.

## Limitations

- All results come from one shared Skylake-SP host, with one compiler and one
  version of Python, NumPy, and MKL, and the AVX2 kernels were measured on that
  AVX-512 processor, not on a processor without AVX-512. The NEON kernels have
  not been timed.
- Medians of the same case can differ by 10% to 30% between runs, most for calls
  that take less than a few milliseconds, and by more when the calls are
  interleaved with the per-pixel evaluation, as the two measurements of the
  8-band image show.
- The images are noise, which exercises no block that the guards evaluate a
  second time, and the `algorithms` protocol covers the dimensions, band counts,
  overlaps, sample types, and block sizes of its sweeps only.
- MATLAB itself was not available, so the MATLAB toolbox was timed under GNU
  Octave, and the toolboxes ran in a separate environment with NumPy 2.5.5 and
  PyTorch 2.15.0.

## Running the benchmark

The benchmark needs the `benchmark` extras, which install pyfastcd,
threadpoolctl, and Matplotlib, and each run should be pinned to one core, with
the matrix-product libraries limited to one thread:

```sh
python -m pip install ".[benchmark]"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 6 python benchmarks/benchmark.py --rounds 9 --output calls.json
taskset -c 6 python benchmarks/benchmark.py --protocol kernels --rounds 15 --output kernels.json
taskset -c 6 python benchmarks/benchmark.py --protocol algorithms --rounds 9 \
    --sweeps dimension bands overlap types blocks --output algorithms.json
```

The `algorithms` protocol writes the report `algorithms.md` beside its JSON
file, and `--no-toolbox`, `--toolbox-rounds N`, and `--max-dimension N` shorten
it. The `toolboxes` protocol needs GNU Octave and PyTorch, and the toolboxes are
fetched once before it runs:

```sh
python benchmarks/toolboxes.py fetch
taskset -c 6 python benchmarks/benchmark.py --protocol toolboxes --rounds 9 --output toolboxes.json
```

The options of the driver are:

| Option | Meaning |
| --- | --- |
| `--protocol NAME` | `calls` (the default), `kernels`, `algorithms`, or `toolboxes` |
| `--rounds N` | Rounds per case, 9 by default |
| `--backend {auto,numpy,mkl}` | The matrix-product library of inazumaQ2n |
| `--workers N` | Workers of inazumaQ2n |
| `--simd NAME` | Restrict the kernels to one instruction set, for example `avx2` on a processor with AVX-512 (`calls` only) |
| `--no-baselines` | Leave out the per-pixel evaluation (`calls` only) |
| `--cases TEXT ...` | Run only the cases whose names contain one of the texts |
| `--sweeps NAME ...` | The sweeps of the `algorithms` protocol |
| `--max-dimension N` | The largest dimension of the sweeps (`algorithms` and `toolboxes`) |
| `--no-toolbox`, `--toolbox-rounds N` | Leave out the toolbox recursion, or limit its rounds (`algorithms`, and `toolboxes` for the second) |
| `--output FILE` | The JSON file to write |

The output records the versions of inazumaQ2n, pyfastcd, Python, and NumPy, the
instruction set and the library selected, the processor, the CPU affinity, the
thread pools in the process, and the SHA-256 of every source file of the
package.

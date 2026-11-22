# Benchmark

The driver `benchmarks/benchmark.py` times complete calls of the public
interface. It has four protocols:

| `--protocol` | Measurements |
| --- | --- |
| `calls` (default) | `q2n` and a reused `Q2nPlan` on a fixed set of images, beside the evaluation of the definition pixel by pixel with the products of pyfastcd's quasilinear and direct algorithms |
| `kernels` | For images with one to four bands: the kernels that need no matrix product, in every instruction set of the processor, against the route through a matrix product |
| `algorithms` | The real cross moments, the fused sums of fastCD and the recursive product of the reference toolboxes, over sweeps of the dimension, the band count, the overlap, the sample type, the block size and the workers; see [Algorithms](#algorithms) |
| `toolboxes` | The published toolboxes themselves, the MATLAB one under GNU Octave, beside the package, with the difference of their values; see [Toolboxes](#toolboxes) |

This page describes the comparison, how the protocols measure, how to run
them, and the archived results. The accuracy of the values is covered in
[Numerical stability](stability.md).

## The comparison

Q2n is defined pixel by pixel: in every block, the bands of both images are
normalized with the statistics of the reference, every normalized reference
pixel is multiplied by the conjugate of the normalized test pixel in the
Cayley–Dickson algebra, and the products are averaged (see
[Conventions and results](mathematics.md)). The benchmark measures
inazumaQ2n against that evaluation, as `benchmarks/baseline.py` implements
it.

[fastCD](https://git.tanuki-cd.com/algebraity/fastCD) is a C library for
Cayley–Dickson multiplication over the real and complex numbers and finite
fields, and [pyfastcd](https://pypi.org/project/pyfastcd) is its Python
package. Its direct algorithm multiplies in `O(N^2)` operations and its
quasilinear algorithm in `O(N log N)`, where `N` is the dimension of the
algebra. The baseline computes every pixel product of a batch of blocks in
one call of pyfastcd, with one algorithm or the other, and does everything
else in NumPy, vectorized over the pixels, the bands and the blocks of the
batch, with batches of about 8 MiB. Its values agree with an independent
evaluation of the definition to `1e-12` in the tests, and its accuracy
against the exact value is that of the definition evaluated in floating
point (see [Numerical stability](stability.md)).

## How measurements are made

### What is timed

Every measurement is one call on a pair of images that already exist in
memory, timed with `time.perf_counter`. The conversion of the input, the
allocation of work arrays, the moments, the evaluation of the blocks and
the mean are inside the timed region; for `q2n` so are the construction and
the release of its plan. For the `plan` variant the plan is constructed
once, outside the timed region. Nothing is compiled or loaded during a
timed call: each variant is called once before the rounds begin.

### Rounds

The variants of a case are called once per round, in an order that rotates
from round to round, so that a change in the load of the machine falls on
all of them. The report gives the median over the rounds. The output file
keeps every observation, and for each variant the number of rounds in which
it was slower than the plan.

### Images

The images are uniform noise in `[0, 255]` and the same image with Gaussian
noise of standard deviation 20 added, clipped for integer types, from a
fixed seed. Running time does not depend on the values, except through the
blocks that inazumaQ2n evaluates a second time, of which these images have
none.

### Threads

The driver limits every matrix-product library in the process to one
thread, with threadpoolctl, so that the single-thread results are those of
one core. `--workers` sets the number of workers of inazumaQ2n; the
libraries stay at one thread each.

### Verification

Before timing, every variant of a case is evaluated and the values are
compared, and the run stops if two differ by more than `1e-12`.

## Running the benchmark

```sh
python -m pip install ".[benchmark]"
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 6 \
    python benchmarks/benchmark.py --rounds 9 --output calls.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 6 \
    python benchmarks/benchmark.py --protocol kernels --rounds 15 --output kernels.json
```

| Option | Meaning |
| --- | --- |
| `--rounds N` | Rounds per case, 9 by default |
| `--backend {auto,numpy,mkl}` | The matrix-product library of inazumaQ2n |
| `--workers N` | Workers of inazumaQ2n |
| `--simd NAME` | Restrict the kernels to one instruction set, for example `avx2` on a processor with AVX-512 |
| `--no-baselines` | Leave out the pyfastcd baselines |
| `--cases TEXT ...` | Run only the cases whose names contain one of the texts |
| `--output FILE` | The JSON file to write |

The output records the versions of inazumaQ2n, pyfastcd, Python and NumPy,
the instruction set and library selected, the processor, the CPU affinity,
the thread pools in the process, and the SHA-256 of every source file of the
package.

## Results

The results of this section were measured before the fused sums of fastCD
were added, when every image took the route that `algorithm="moments"` now
names; for images with more than four bands, `q2n` now selects the faster of
the two routes, which [Algorithms](#algorithms) measures.

All results below were measured on 2026-11-13 on a Dell PowerEdge T440 with
an Intel Xeon Gold 6148 (Skylake-SP, AVX-512, 1 MiB of second-level cache
per core) under Linux 7.2.11, with GCC 16.2.1, CPython 3.14.8, NumPy 2.5.4
with its bundled OpenBLAS 0.3.34, MKL 2026.2.1 and pyfastcd 1.4.0.
inazumaQ2n was built as a wheel is: for the baseline instruction set, with
the kernels for wider instruction sets selected at run time. The host is
shared, and no governor, turbo or SMT setting was changed. Medians of the
same case can differ by 10% to 30% between runs, most for calls that take
less than a few milliseconds. The raw observations are archived in
[benchmark-results/t440-20261113/](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmark-results/t440-20261113/).

### Complete calls on one core

Median milliseconds over 9 rounds on one logical CPU, from `calls.json`.
`q2n` includes the construction of its plan, and `plan` reuses one. The
speedup divides the time of the faster per-pixel evaluation by that of
`q2n`. Blocks are 32 by 32 pixels without overlap unless a block and stride
are given.

| Image | `q2n` | plan | per pixel, quasilinear | per pixel, direct | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| gray 512 x 512, uint8 | 0.343 | 0.206 | 22.6 | 22.8 | 66 |
| RGB 512 x 512, uint8 | 0.910 | 0.799 | 67.7 | 68.6 | 74 |
| RGB 512 x 512, uint8, block 64, stride 32 | 1.07 | 0.947 | 215.0 | 215.8 | 201 |
| 4 bands 512 x 512, uint16 | 2.38 | 2.25 | 65.9 | 65.6 | 28 |
| 8 bands 512 x 512, uint16 | 13.5 | 13.4 | 107.7 | 107.0 | 7.9 |
| 16 bands 512 x 512, float64 | 17.5 | 21.0 | 163.4 | 181.9 | 9.4 |
| 16 bands 512 x 512, float64, block 64, stride 32 | 23.7 | 28.3 | 565.1 | 653.2 | 24 |
| 32 bands 256 x 256, float64 | 10.6 | 17.1 | 83.3 | 104.1 | 7.8 |
| 32 bands 256 x 256, float64, block 64, stride 32 | 13.1 | 16.1 | 293.8 | 384.3 | 22 |
| 159 bands 200 x 200, float64 | 66.9 | 68.3 | 478.8 | 1,268.7 | 7.2 |
| 159 bands 200 x 200, float64, block 64, stride 32 | 100.7 | 100.2 | 2,643.4 | 5,694.7 | 26 |

`q2n` and the plan were faster than both per-pixel evaluations in all 9
rounds of every case, and the values of all four variants agree to
`4.4e-16` in every case.

Three things account for the difference. inazumaQ2n forms no product of
pixels: the hypercomplex work of a block is one contraction of a `B` by `B`
matrix. It does every per-band step in compiled code, where the per-pixel
evaluation spends most of its time for small `N` in NumPy normalizing and
rearranging the bands. And for overlapping blocks it reads every pixel once,
where the per-pixel evaluation normalizes and multiplies every pixel once per
block that contains it, which is why its advantage is largest there.

The two product algorithms of pyfastcd take about the same time up to eight
bands, where the products are a small part of the per-pixel evaluation;
from 16 bands on the quasilinear algorithm is faster in every round, by a
factor of 2.6 at 159 bands.

A plan saves the setup of a call, about 0.1 ms, which matters for images with
at most four bands. For larger images it makes no measurable difference: in
this run, where every call followed a per-pixel evaluation that touches
hundreds of megabytes, the plan happened to be slower for the 16- and
32-band images, while in a separate interleaved run without the baselines
the two were within 10% of each other in both directions.

### Instruction sets, for at most four bands

Median milliseconds of a reused plan over 15 rounds, from `kernels.json`.
`matrix product` is the general route forced on the same image, with the
faster of NumPy and MKL. The AVX2 column is the AVX2 kernel run on this
AVX-512 processor.

| Image | block / stride | AVX-512 | AVX2 | matrix product |
| --- | --- | ---: | ---: | ---: |
| 1 band 512 x 512, uint8 | 32 / 32 | 0.185 | 0.235 | 2.814 |
| 2 bands 384 x 448, float64 | 32 / 32 | 0.357 | 0.457 | 2.041 |
| 2 bands 512 x 512, float32 | 64 / 32 | 0.590 | 0.846 | 4.186 |
| 3 bands 512 x 512, uint8 | 32 / 32 | 0.695 | 1.057 | 4.974 |
| 3 bands 95 x 117, float64 | 32 / 12 | 0.220 | 0.329 | 1.246 |
| 4 bands 512 x 512, uint16 | 32 / 32 | 1.127 | 1.944 | 6.285 |
| 4 bands 512 x 512, float64 | 32 / 32 | 1.372 | 2.005 | 4.739 |
| 4 bands 1024 x 1024, uint16 | 32 / 32 | 7.44 | 12.16 | 27.22 |
| 4 bands 300 x 300, float64 | 8 / 8 | 0.758 | 0.947 | 1.493 |

Over the 16 cases of the protocol, the kernels are faster than the matrix
product by a geometric mean of 4.6 times with AVX-512 and 3.3 times with
AVX2, and by at least 2.0 and 1.6 times; they were faster than both
libraries in all 15 rounds of every case. This is the measurement on which
images with at most four bands use them. The NEON kernel is the same
generated code and gives the same bits, which was checked under emulation,
but it has not been timed.

### NumPy and MKL, for more than four bands

Median milliseconds of a reused plan, timed in the same rounds, from
`calls.json`.

| Image | NumPy (OpenBLAS 0.3.34) | MKL 2026.2 | rounds won by MKL |
| --- | ---: | ---: | ---: |
| 8 bands 512 x 512, uint16 | 12.7 | 13.4 | 3 / 9 |
| 16 bands 512 x 512, float64 | 20.7 | 21.0 | 3 / 9 |
| 32 bands 256 x 256, float64 | 13.9 | 17.1 | 0 / 9 |
| 159 bands 200 x 200, float64 | 83.0 | 68.3 | 9 / 9 |
| 159 bands 200 x 200, float64, block 64, stride 32 | 123.8 | 100.2 | 8 / 9 |

MKL is about 20% faster at 159 bands, where the matrix products are most of
the time. With fewer bands the centering of the samples dominates, and the
two libraries are within the variation of the host. `backend="auto"`
selects MKL whenever it is installed; pass `backend="numpy"` to use the
library of NumPy instead.

### Workers

Median milliseconds of a reused plan with 1, 2 and 4 workers on four
physical cores, 9 rounds each, from `workers-1.json`, `workers-2.json` and
`workers-4.json`. Each worker count is a separate run.

| Image | 1 worker | 2 workers | 4 workers |
| --- | ---: | ---: | ---: |
| gray 512 x 512, uint8 | 0.393 | 0.322 | 0.446 |
| RGB 512 x 512, uint8 | 0.710 | 0.885 | 0.761 |
| 4 bands 512 x 512, uint16 | 1.136 | 1.112 | 0.851 |
| 8 bands 512 x 512, uint16 | 12.05 | 8.94 | 4.60 |
| 16 bands 512 x 512, float64 | 21.39 | 16.43 | 9.07 |
| 32 bands 256 x 256, float64 | 15.87 | 8.82 | 5.70 |
| 159 bands 200 x 200, float64 | 75.41 | 50.98 | 27.58 |
| 159 bands 200 x 200, float64, block 64, stride 32 | 111.10 | 77.49 | 41.79 |

Four workers are 2.4 to 2.8 times faster than one for images with more than
four bands. For calls that take a millisecond or less, handing the tiles to
threads costs about as much as it saves, and workers should be left at one.

## Algorithms

The package obtains the covariance of a block either from real cross moments
or from one fused sum of products of fastCD (see
[Conventions and results](mathematics.md)), and `algorithm="auto"` selects
between them. The `algorithms` protocol is the measurement behind that
selection. It times three evaluations of the same image pairs:

| Variant | Evaluation |
| --- | --- |
| `toolbox` | The recursive product of the reference toolboxes, block by block in NumPy, as a complete call (`benchmarks/toolbox.py`) |
| `moments` | A reused plan with `algorithm="moments"`; with MKL installed, also with `backend="numpy"` |
| `fastcd` | A reused plan with `algorithm="fastcd"` |

and a plan with `algorithm="auto"`, to record what it selects and how far
the selected algorithm is from the faster one.

`benchmarks/toolbox.py` is a NumPy port of the structure of the MATLAB
implementation in the pansharpening toolbox, which its Python ports keep:
for every block it normalizes the bands one at a time and multiplies every
reference pixel by the conjugate of the test pixel with `onion_mult2D`, the
doubling recursion on arrays of pixels. A product in dimension `N` is four
products in dimension `N / 2`, so a pixel costs `N^2` real multiplications,
and every level of the recursion allocates its operands. The port leaves
out the rounding of the padded images to integers, which changes the value
of the index and which inazumaQ2n does not reproduce; its values agree with
those of the package to `7e-16` in every case of the run.

### Sweeps

| `--sweeps` | Cases |
| --- | --- |
| `dimension` | `B = N` bands for `N` from 2 to 1024, blocks of 32 without overlap, with the toolbox |
| `bands` | `B = N / 2 + 1`, `5N / 8`, `3N / 4` and `7N / 8` for `N` from 8 to 1024: the moments do less work with fewer bands, the fused sums the same |
| `overlap` | Blocks of 32 with stride 16 and of 64 with stride 16, merged from 4 and 16 tiles, with `B = N` and, for 4 tiles, with fewer bands |
| `types` | `float32`, `uint16` and `uint8` samples |
| `blocks` | Blocks of 8, 16 and 64 pixels without overlap |
| `workers` | 1, 2, 4 and 8 workers |

The images are `float64` unless stated, and their side depends on the
dimension so that every image exceeds the second-level cache of a core: 512
pixels through `N = 16`, 256 through 64, 128 through 256 and 64 above. The
toolbox is timed in three rounds at most, and a call of more than two
seconds is timed from its first call. Before timing, the values of the
plans must agree to `1e-12` and that of the toolbox to `1e-10`.

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 6 \
    python benchmarks/benchmark.py --protocol algorithms --rounds 9 \
    --sweeps dimension bands overlap types blocks --output algorithms.json
```

`--no-toolbox`, `--toolbox-rounds N` and `--max-dimension N` shorten a run.
The driver writes the report `algorithms.md` beside the JSON file.

fastCD selects the kernel of its fused sums at run time and offers no
switch, so one build measures one kernel on a given processor. The AVX2 and
portable kernels were measured on builds with fastCD's tuning macro, which
`CFLAGS` passes to `setup.py`: `CFLAGS=-DFASTCD_SUM_MAX_KERNEL=2` withholds
its AVX-512 kernel and `=1` leaves the portable one; they were run with
`--simd avx2` and `--simd baseline`, so that the kernels of inazumaQ2n match.

### Measured results

Measured on 2026-11-21 on the host of the other results, from the working
tree that introduced the fused sums, with fastCD 1.5.0 and MKL 2026.2.1 on
one logical CPU. The observations are archived in
[benchmark-results/fastcd-20261121/](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmark-results/fastcd-20261121/).

Median milliseconds with the AVX-512 kernels, `B = N`, blocks of 32 without
overlap, from `algorithms-avx512.json`:

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

Both algorithms of the package are faster than the recursion of the
toolboxes at every dimension: the moments by 31 to 373 times, the fused sums
by 14 to 1,456 times, and the algorithm that `auto` selects by 61 times at
least. The advantage grows with the dimension, as `N^2` against `B^2` with
an optimized matrix product, and against `N log N`. Between the two, the
kernels for at most four bands, which need neither a workspace nor a
product, are 2.3 to 5.6 times faster than a fused sum, and from eight bands
on the fused sums are 1.6 to 3.9 times faster than the moments, in all 9
rounds at every dimension.

The time of the moments divided by that of the fused sums, with each kernel
of fastCD, for `B = N`; `auto` selects the fused sums where the ratio is
bold:

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

Without overlap the fused sums are the faster from eight bands on with
either vector kernel. With overlap the moments form the matrix of a tile
once for the blocks that share it, while a pixel enters the fused sum of
every block that contains it; the two share only the reading of the pixels
and the statistics of the tiles. With a stride of half the block the fused
sums are then the faster from dimension 128 with AVX-512 and from 512 with
AVX2, and with a quarter from 512 and 1024. The portable kernel, which on
this processor rounds products and sums separately and uses no vector
instructions, is faster than the moments at dimension 1024 only, and `auto`
does not select it. With the portable kernels of both, images with two and
four bands take about as long by either route.

The same ratio for fewer bands than the dimension, without overlap:

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

At `N = 8` the first two columns are the same image, with five bands. A
fused sum costs the same for every band count of a dimension, and the
moments less for fewer bands, so the fused sums need the bands to fill most
of the dimension: five eighths to three quarters of it with AVX-512, and
nearly all of it with AVX2 below dimension 256. With a stride of half the block the AVX-512 kernel
needs `B = N` at dimension 128, seven eighths at 256, three quarters at 512
and five eighths at 1024.

With narrower samples and other block sizes the fused sums gain as much or
more: over the `types` and `blocks` sweeps the ratio is 1.6 to 3.5 and 1.7
to 7.4 with AVX-512, and 1.1 to 2.4 and 1.2 to 4.9 with AVX2, for the
dimensions 8, 32, 128 and 512.

Median milliseconds with several workers on eight physical cores, from
`algorithms-workers.json`:

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

### What `auto` selects

`src/inazumaq2n/_auto.py` holds, for each vector kernel of fastCD, each
measured number of tiles per block (1, 4 and 16) and each dimension, the
least share of the dimension that the bands must fill for the fused sums to
have been at least 1.1 times as fast as the moments; `auto` selects the
fused sums from that share on. A tile count that was not measured uses the
entry of the next larger one, more than 16 tiles select the moments, and a
dimension above 1024 uses the entry of 1024. Images with at most four
bands, and every image where fastCD runs its portable kernel, use the
moments.

In the 180 cases of the two single-core runs with vector kernels, `auto`
selected the faster algorithm in 168; in the other 12 the algorithm it
selected took at most 1.17 times as long as the faster one. The table is a
measurement of one processor with MKL as the matrix-product library; on
another machine `algorithm="moments"` or `algorithm="fastcd"` overrides it.

## Toolboxes

The comparisons above use code written for this repository: the definition
evaluated with the products of pyfastcd, and a NumPy port of the recursion
of the toolboxes. The `toolboxes` protocol runs the published code itself,
and records two things for it: how long it takes, and how far its values
are from those of inazumaQ2n.

`benchmarks/toolboxes.py` lists each implementation as files of a public
repository at a pinned commit, with the SHA-256 of every file.
`python benchmarks/toolboxes.py fetch` downloads them into
`benchmarks/.toolboxes`, which is not committed or distributed, and nothing
is run that does not match its hash.

| Name | Implementation | Runs with |
| --- | --- | --- |
| `matlab` | `q2n.m`, `onions_quality.m`, `onion_mult.m`, `onion_mult2D.m` and `norm_blocco.m` of the MATLAB pansharpening toolbox, in the copy of [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) | GNU Octave |
| `pancollection` | `my_q2n.py` of [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection), a NumPy port | NumPy, and PyTorch, which the file imports |
| `hyperspectral` | `Q2n` of the [hyperspectral pansharpening toolbox](https://github.com/matciotola/hyperspectral_pansharpening_toolbox), a PyTorch port | PyTorch and torchvision |

The MATLAB code runs unchanged under Octave. It calls `mean2` and `std2` of
the Image Processing Toolbox, which Octave has only with its image package;
`benchmarks/octave/shims` supplies the two where they are missing. The
driver passes the images as doubles and times the call of `q2n` inside the
interpreter, so that neither its start nor the transfer of the images is
counted.

The toolbox converts both images to `uint16` first, so the comparison uses
images of `uint16` samples, on which it and inazumaQ2n compute the same
index (see [Conventions and results](mathematics.md#agreement-with-the-toolbox)).
`benchmarks/test_toolboxes.py` asserts the agreement to `1e-12` on more
geometries; the benchmark records the largest difference it met.

```sh
python benchmarks/toolboxes.py fetch
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 6 \
    python benchmarks/benchmark.py --protocol toolboxes --rounds 9 --output toolboxes.json
```

Median milliseconds on 2026-11-21 on the host of the other results, with
GNU Octave 11.3.0 and PyTorch 2.15.0 on one thread, for `B = N` bands and
blocks of 32 without overlap, the setting of the evaluation script of the
toolbox. inazumaQ2n is a reused plan with `algorithm="auto"`, which selects
the kernels for at most four bands at 2 and 4 bands and the fused sums from
8 on. The factor in parentheses is the time divided by that of inazumaQ2n.
The archive is
[benchmark-results/toolboxes-20261121/](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmark-results/toolboxes-20261121/).

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

inazumaQ2n is 75 to 1,900 times faster than the recursion in NumPy, 91 to
2,100 times faster than PanCollection, 160 to 4,400 times faster than the
PyTorch toolbox and 560 to 5,400 times faster than the MATLAB toolbox under
Octave.

The times of the MATLAB toolbox are those of Octave, which interprets the
recursion without the compiler of MATLAB: it takes 3 to 13 times as long as
the same recursion in NumPy, the more the smaller the dimension, where the
calls and not the arithmetic are the cost. MATLAB was not available for
this measurement and would be faster than Octave. The NumPy port is the
fastest of the four and therefore the most favourable comparison for the
recursion, and it is the one the algorithms protocol uses. PanCollection is
the same recursion in NumPy and takes 1.1 to 1.3 times as long as the port;
the PyTorch toolbox, on one thread and on the CPU, 2.1 to 3.1 times.

### Agreement

The largest difference of the value of a block from that of inazumaQ2n,
over the dimensions of the table, and what `python benchmarks/toolboxes.py
report` and the tests find outside its setting:

| Implementation | Source | Largest difference of a block from inazumaQ2n, in the table | Outside the setting of the table |
| --- | --- | ---: | --- |
| MATLAB pansharpening toolbox | `q2n.m` and its helpers, in the copy of [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox), run unchanged under GNU Octave | `1.7e-15` | Agrees to `1e-12` in every geometry of the tests: overlap, padding, any number of bands, zero and constant bands |
| NumPy port of the toolbox | [benchmarks/toolbox.py](https://git.tanuki-cd.com/algebraity/inazumaq2n/src/branch/main/benchmarks/toolbox.py), in this repository | `1.3e-15` | Agrees to `1e-12` likewise |
| PanCollection | `my_q2n.py` of [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection), NumPy | `1.4e-07` | Fails when the number of bands is not a power of two; off by up to `6e-3` when blocks overlap or do not divide the image |
| Hyperspectral pansharpening toolbox | `Q2n` of [hyperspectral_pansharpening_toolbox](https://github.com/matciotola/hyperspectral_pansharpening_toolbox), PyTorch | `1.4e-07` | Agrees to `1e-7` with overlap and with other numbers of bands; off by up to `2e-4` when the image needs a different padding at the bottom and at the right |

The MATLAB toolbox, run as it is distributed, agrees with inazumaQ2n to
rounding error at every dimension from 2 to 1024, and
`benchmarks/test_toolboxes.py` requires that agreement to `1e-12`, for the
moments and for the fused sums. The two Python ports are not exact, each
for a reason that can be read in its source:

- PanCollection normalizes every band with the population standard
  deviation, `x.std()`, where `std2` of the toolbox divides by `P - 1`. In
  the setting of the table the effect is `1e-7` or less. Its padding of the
  image does not reflect the rows and columns as the toolbox does, which
  changes the blocks at the bottom and the right when blocks overlap or do
  not divide the image, and its padding of the bands raises an error for a
  number of bands that is not a power of two.
- The hyperspectral pansharpening toolbox writes the value of every block
  into a single-precision tensor, which limits its agreement to about
  `1e-7`. It pads the bottom of the image by the amount meant for the right
  and the right by the amount meant for the bottom, which changes the
  values when the two differ; an image that needs the same padding on both
  sides agrees. It also normalizes a block in place after
  converting it to double: for images passed as double tensors that
  conversion does not copy, and overlapping blocks then overwrite each
  other, so the adapter passes single-precision tensors, as the toolbox
  itself does.

These are the differences that `tests` and `report` exhibit on the pinned
commits; they are stated to explain the numbers, not as a review of either
project.

## Where the time goes

For images with more than four bands, the profile of a call in `calls.json`
divides as follows on this machine.

| Image | centering | matrix products | evaluating blocks |
| --- | ---: | ---: | ---: |
| 8 bands 512 x 512, uint16 | 85% | 14% | 1% |
| 16 bands 512 x 512, float64 | 74% | 24% | 1% |
| 32 bands 256 x 256, float64 | 75% | 23% | 1% |
| 159 bands 200 x 200, float64 | 37% | 59% | 4% |
| 159 bands 200 x 200, float64, block 64, stride 32 | 33% | 58% | 9% |

From 5 to about 32 bands the centering pass, which reads the image and
writes the workspace, is most of the time, and it is limited by memory
traffic more than by arithmetic: the instruction set changes its speed
little. Images with at most four bands avoid that pass altogether, which is
why the step from four to eight bands is the largest in the table of
complete calls.

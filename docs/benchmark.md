# Benchmark

The driver `benchmarks/benchmark.py` times complete calls of the public
interface. It has two protocols:

| `--protocol` | Measurements |
| --- | --- |
| `calls` (default) | `q2n` and a reused `Q2nPlan` on a fixed set of images, beside the evaluation of the definition pixel by pixel with the products of pyfastcd's quasilinear and direct algorithms |
| `kernels` | For images with one to four bands: the kernels that need no matrix product, in every instruction set of the processor, against the route through a matrix product |

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

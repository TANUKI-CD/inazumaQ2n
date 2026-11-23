# Numerical stability

The value of each block of a Q2n index is a ratio of statistics of the
normalized pixels of the block, and inazumaQ2n evaluates it in a different order
from its definition: from centered moments of the bands, or from one fused sum
of products of the centered and scaled pixels, where the definition normalizes
every sample, multiplies the pixels in the Cayley-Dickson algebra, and averages
the products (see [Conventions and results](mathematics.md)). Every operation in
either order rounds its result in binary64 arithmetic, and this page
investigates whether inazumaQ2n loses accuracy compared with the definition
evaluated pixel by pixel, on ordinary images and on inputs that strain the
definition.

The conclusion is that, measured against exact values on 8,988 block pairs, both
algorithms of inazumaQ2n are accurate to a few units in the last place on
typical inputs, as the per-pixel evaluation is: with its moments, the median
error of inazumaQ2n is between 0.96 and 4.3 units of the unit roundoff at every
number of bands from 1 to 256, against 1.1 to 2.5 for the per-pixel evaluation.
On inputs that strain the definition, such as images whose mean is far larger
than their variation and samples whose products underflow or whose squares
overflow, inazumaQ2n is the more accurate, often by orders of magnitude, while
on 16-bit integer images its long sums lose more accuracy than those of the
per-pixel evaluation. The sections below explain where each evaluation rounds,
describe how accuracy was measured, report and interpret the results, state
which results are identical bit for bit on every machine, and end with what the
results mean for code that uses inazumaQ2n, with the limitations of the study,
and with how to check and reproduce it.

## Where rounding enters

The three evaluations add their long sums in different orders and over different
quantities, and five properties of these sums predict the measurements below.

**The moments of inazumaQ2n.** For images with at most four bands, one kernel
sums the centered squares and cross products of a tile in eight lanes, so that
each lane adds `P / 8` products in sequence for a block of `P` pixels. For more
bands, a kernel sums the centered squares of each band in sequence over the
pixels, and the cross moments are a matrix product, whose order of summation
belongs to the BLAS library. The value of the block is then formed from these
moments with a few dozen further roundings for each band, independently of `P`.

**The fused sums of inazumaQ2n.** The statistics of the bands are the same
sequential sums as those of the moments for more than four bands, bit for bit,
at every number of bands. The covariance is then one fused sum of fastCD over
the pixels of the block, each centered and scaled as it is read, in which the
products of fastCD's algorithm are accumulated in eight lanes with fused
multiply-adds and its additions are applied once to the accumulated sums. No
matrix is formed and nothing is contracted, so neither the rounding of a BLAS
library nor the `B^2` terms of the contraction enter, while with one to four
bands, where the moments use their eight-lane kernel, the sequential sums of the
statistics make the fused sums the less accurate of the two.

**The per-pixel evaluation.** The definition normalizes every sample, multiplies
every pixel, and averages the products, with NumPy forming each average by
pairwise summation. It then subtracts the product of the means from the mean of
the products, and since both are of the order of the dimension of the algebra
while their difference, the covariance, can be much smaller, the rounding error
of the difference is measured against the larger terms.

**Integer images.** When the samples are integers and the number of pixels is a
power of two, the mean of a band over a block is an exact binary fraction, and
every centered square of the band has the same low-order bits. A long sum of
such squares, once it exceeds `2^53` units of those bits, rounds in the same
direction at every step, so its error grows with the length of the sum instead
of averaging out. 8-bit images stay below that threshold and are summed exactly,
while 16-bit images do not, which causes the larger errors of inazumaQ2n on the
`uint16` family, in its own sums, in those of the BLAS library, and in the
statistics of the fused sums. The pairwise sums of the per-pixel evaluation are
short enough not to be affected, and so are the sums over tiles: a block merged
from four tiles has a largest error of 16.9 with the moments and 13.9 with the
fused sums.

**Guarded blocks.** The fast evaluation relies on identities that hold in exact
arithmetic but not always in floating point, so the blocks where they may fail
are evaluated again from their normalized samples (see [Blocks evaluated from
their normalized
samples](mathematics.md#blocks-evaluated-from-their-normalized-samples)). Both
algorithms compute the statistics that the guards test in the same way, so they
mark the same blocks, which the study checks on every case, and it also measures
inazumaQ2n without the second evaluation, to show what the guards change.

## How accuracy was measured

### Error measures

The value of a block is dimensionless: it is at most one for up to eight bands,
typical values lie between 0 and 1, and values are compared and reported as
absolute numbers, so the study measures the absolute error of the value. For a
computed value `q_hat` with exact value `q`, and with `u = 2^-53` the unit
roundoff of binary64, it records

```text
E_abs = |q_hat - q|
E_rel = E_abs / q                          (q nonzero)
eta   = E_abs / u
```

where `eta`, the absolute error in units of `u`, is the main measure, while the
relative error is reported as well and is large where the value itself is small.

### Exact reference

`tests/exact.py` evaluates the value of a block exactly, without inazumaQ2n or
pyfastcd arithmetic. It converts every binary64 sample to an integer times a
power of two, forms the sums, the sums of squares, and the cross sums of the
block in arbitrary-precision integers, decides every branch of the definition (a
mean of zero, a standard deviation of zero, and a variance of zero) on these
exact values, and evaluates the square roots and quotients with decimal
arithmetic at 100 significant digits, while the signs of the algebra used in
this reference are derived from the doubling formula on their own. Before the
study runs, `benchmarks/stability.py` checks the reference against a literal
evaluation of the definition in decimal arithmetic, pixel by pixel with the
recursive product, on 162 small blocks from nine families, where the largest
relative difference was `6.5e-96`, and the study also evaluated one case of
every group again at 200 digits, where the largest relative change was
`3.3e-99`.

### Methods compared

Every case is one block of `side` by `side` pixels with `B` bands, scored as an
image of one block, and all seven methods read the same stored samples:

| | Method |
| --- | --- |
| I | inazumaQ2n, `q2n_map` with `algorithm="moments"` and blocks that do not overlap |
| F | inazumaQ2n, `q2n_map` with `algorithm="fastcd"` and blocks that do not overlap: one fused sum of fastCD, behind the same guards as I |
| T | I with the block merged from four tiles, as for blocks that overlap with a stride of half the block size |
| G | F with the records and the statistics of the block merged from four tiles |
| U | I without the second evaluation of the blocks its guards mark |
| Q | the definition evaluated pixel by pixel, with the products of pyfastcd's quasilinear algorithm |
| D | the same, with the products of pyfastcd's direct algorithm |

Q and D are `benchmarks/baseline.py`, which evaluates the definition in NumPy,
vectorized over the pixels and the bands of a batch of blocks, with every pixel
product computed by [pyfastcd](https://pypi.org/project/pyfastcd), the Python
package of fastCD. For I, T, and U inazumaQ2n used NumPy for its matrix products
and the AVX-512 kernels of the processor, which give the same bits as every
other instruction set, and for F and G the fused sums of the fastCD compiled
into it, run with fastCD's AVX-512 kernel, which gives the same bits as its AVX2
kernel.

### Corpus

The corpus has 8,988 block pairs from 19 families, generated from seeds derived
from `numpy.random.SeedSequence(20261113, ...)`, each family exercising one kind
of input:

- **Typical:** `uniform`, samples uniform in `[0, 255)` with a test image equal
  to the reference plus Gaussian noise of standard deviation 20, and `gaussian`,
  Gaussian bands with their own means and spreads.
- **Integer:** `uint8` and `uint16`, integer samples over the full range of each
  type, with rounded noise.
- **Relations:** `identical`, `affine` (`0.8 x + 2`), `inverted` (`255 - x`),
  and `independent` test images.
- **Means far from zero:** `offset_1e3` to `offset_1e12`, samples equal to the
  offset plus a uniform variation in `[0, 1)`, so that the mean is about 3.5
  times the offset times the standard deviation.
- **Branches of the definition:** `zero_mean`, reference bands whose mean is
  exactly zero, in random pixel order; `near_zero_mean`, the same with one
  sample changed by a relative `2^-40`; and `constant_integer` and
  `constant_fraction`, a reference band constant at 7 or at 0.1.
- **Extreme scales:** `scaled_down` (`2^-530`, where products of samples are
  subnormal), `scaled_up` (`2^490`), and `overflow` (`2^515`, where squares of
  deviations overflow).

The numbers of bands are 1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 159, and 256, with
blocks of 32 by 32 pixels. Each typical family has 256 cases for each number of
bands up to 16, 64 up to 64, and 16 above, and each other family 32, 8, and 4,
while the typical families were also measured at blocks of 8, 16, and 64 pixels
with 3 and 16 bands, 64 cases each.

### Host and software

The study ran on 2026-11-21 on an Intel Xeon Gold 6148 under Linux with Python
3.14.8, NumPy 2.5.4 and its bundled OpenBLAS, pyfastcd 1.4.0, and inazumaQ2n
0.1.0 with fastCD 1.5.0, and its record lists the hash of every measured source
file, of the record of the fastCD sources, of the baseline, and of the
reference.

## Accuracy of block values

The tables below are generated by `benchmarks/stability_report.py` from the
archived results of the *full profile* of the study, its complete run, as
opposed to the short smoke profile that only tests the tooling. They report the
typical inputs at every number of bands, first by absolute and then by relative
error, then the effect of the block size, of merging a block from tiles, and of
the fused sums against the moments, and finally every family at 4 and at 16
bands, and the families that strain the definition, pooled over the numbers of
bands.

In every table, a cell of the form "I / F / Q / D" gives one statistic for each
of the methods named, "p99" is the 99th percentile, and a column "I at least as
accurate as F / Q / D" gives the fraction of cases in which the exact error of I
is no larger than that of each other method. "F as accurate as I" gives the
fraction in which the two errors are equal, "marked" counts the blocks that the
guards mark, and "results off by more than 1e-06" counts values that are wrong
rather than merely inaccurate.

<!-- BEGIN STABILITY ACCURACY -->

8,988 block pairs from the full profile. I is inazumaQ2n with its moments and F inazumaQ2n with the fused sums of fastCD; Q and D evaluate the definition pixel by pixel with the products of pyfastcd's quasilinear and direct algorithms; G is F and T is inazumaQ2n with the block merged from four tiles; U is inazumaQ2n without the evaluation by definition of the blocks its guards mark. B is the number of bands and N the dimension of the algebra. eta is the absolute error of the value of a block in units of u = 2^-53. "Typical" pools the uniform and Gaussian families. Empirical p99 values and maxima describe this corpus only.

**Typical inputs, blocks of 32 by 32 pixels**

| B | N | blocks | median eta I / F / Q / D | p99 eta I / F / Q / D | max eta I / F / Q / D | I at least as accurate as F / Q / D |
|---:|---:|---:|:---|:---|:---|:---|
| 1 | 1 | 512 | 1.19 / 4.01 / 1.61 / 1.61 | 5.01 / 13.9 / 6.93 / 6.93 | 5.97 / 16.7 / 8.19 / 8.19 | 84% / 65% / 65% |
| 2 | 2 | 512 | 1.1 / 2.81 / 1.82 / 1.82 | 4.32 / 10.4 / 7.37 / 7.37 | 7.08 / 14.4 / 8.09 / 8.09 | 81% / 73% / 73% |
| 3 | 4 | 512 | 1 / 2.23 / 1.66 / 1.5 | 3.93 / 8.04 / 5.66 / 5.69 | 5.62 / 10.4 / 8.16 / 7.13 | 80% / 73% / 71% |
| 4 | 4 | 512 | 0.96 / 1.98 / 1.89 / 1.83 | 4.37 / 7.65 / 6.9 / 6.9 | 5.05 / 9.31 / 7.3 / 7.33 | 74% / 74% / 72% |
| 5 | 8 | 512 | 3.13 / 1.87 / 2.02 / 1.91 | 10.7 / 8.01 / 7.46 / 7.85 | 15.9 / 9.98 / 8.89 / 10.8 | 40% / 40% / 38% |
| 8 | 8 | 512 | 2.41 / 1.51 / 1.77 / 1.83 | 9.23 / 5.83 / 6.65 / 7.69 | 11.9 / 6.51 / 8.92 / 9.03 | 45% / 46% / 48% |
| 16 | 16 | 512 | 1.82 / 1.43 / 1.63 / 1.78 | 7.75 / 5.24 / 6.66 / 7.32 | 9.87 / 6.05 / 7.28 / 10.4 | 51% / 54% / 55% |
| 32 | 32 | 128 | 1.48 / 1.55 / 1.55 / 1.62 | 6.33 / 5.33 / 6.64 / 6.68 | 9.99 / 6.99 / 8.66 / 6.91 | 67% / 58% / 56% |
| 64 | 64 | 128 | 2.32 / 1.89 / 1.86 / 1.88 | 7.94 / 6.45 / 6.41 / 6.41 | 9 / 7.13 / 7.2 / 7.2 | 55% / 51% / 48% |
| 128 | 128 | 32 | 3.3 / 2.15 / 1.21 / 1.06 | 9.07 / 5.6 / 5 / 4.28 | 9.47 / 5.76 / 5.24 / 4.48 | 34% / 28% / 25% |
| 159 | 256 | 32 | 2.96 / 3.32 / 2.03 / 2.47 | 10.9 / 12.3 / 6.22 / 5.12 | 11.5 / 13.5 / 6.29 / 5.2 | 56% / 44% / 41% |
| 256 | 256 | 32 | 4.26 / 3.14 / 2.1 / 1.92 | 11.8 / 9.78 / 5.38 / 4.15 | 12.2 / 10.2 / 5.46 / 4.19 | 38% / 25% / 25% |

**Relative error, typical inputs**

| B | median relative error I / F / Q / D | max relative error I / F / Q / D |
|---:|:---|:---|
| 1 | 1.55e-16 / 5.04e-16 / 2.04e-16 / 2.04e-16 | 6.88e-16 / 3.46e-15 / 9.44e-16 / 9.44e-16 |
| 2 | 1.36e-16 / 3.42e-16 / 2.24e-16 / 2.24e-16 | 1.01e-15 / 1.66e-15 / 1.09e-15 / 1.09e-15 |
| 3 | 1.25e-16 / 2.77e-16 / 2.07e-16 / 1.94e-16 | 6.47e-16 / 1.2e-15 / 1.24e-15 / 9.39e-16 |
| 4 | 1.21e-16 / 2.51e-16 / 2.35e-16 / 2.32e-16 | 5.82e-16 / 1.36e-15 / 1.06e-15 / 1.06e-15 |
| 5 | 3.91e-16 / 2.32e-16 / 2.5e-16 / 2.4e-16 | 1.83e-15 / 1.15e-15 / 1.02e-15 / 1.39e-15 |
| 8 | 3.11e-16 / 1.94e-16 / 2.23e-16 / 2.28e-16 | 1.47e-15 / 1.01e-15 / 1.03e-15 / 1.07e-15 |
| 16 | 2.34e-16 / 1.77e-16 / 2.09e-16 / 2.26e-16 | 1.31e-15 / 7.26e-16 / 8.57e-16 / 1.2e-15 |
| 32 | 1.86e-16 / 1.9e-16 / 2.05e-16 / 2.14e-16 | 1.34e-15 / 9.37e-16 / 9.98e-16 / 7.96e-16 |
| 64 | 3.01e-16 / 2.42e-16 / 2.35e-16 / 2.54e-16 | 1.2e-15 / 8.83e-16 / 8.64e-16 / 8.29e-16 |
| 128 | 4.54e-16 / 2.91e-16 / 1.43e-16 / 1.34e-16 | 1.09e-15 / 6.7e-16 / 7.16e-16 / 6.19e-16 |
| 159 | 4.04e-16 / 4.36e-16 / 2.7e-16 / 2.98e-16 | 1.33e-15 / 1.56e-15 / 8.41e-16 / 6.5e-16 |
| 256 | 5.42e-16 / 4.08e-16 / 2.69e-16 / 2.42e-16 | 1.74e-15 / 1.46e-15 / 6.31e-16 / 4.91e-16 |

**Block size, typical inputs**

| B | block | blocks | median eta I / F / Q / D | max eta I / F / Q / D |
|---:|---:|---:|:---|:---|
| 3 | 8 x 8 | 128 | 0.909 / 0.966 / 1.7 / 1.7 | 4.1 / 4.83 / 8.31 / 8.31 |
| 3 | 16 x 16 | 128 | 0.887 / 1.23 / 1.62 / 1.63 | 3.28 / 6.05 / 7.12 / 6.37 |
| 3 | 32 x 32 | 512 | 1 / 2.23 / 1.66 / 1.5 | 5.62 / 10.4 / 8.16 / 7.13 |
| 3 | 64 x 64 | 128 | 1.08 / 4.21 / 1.76 / 1.65 | 4.63 / 18.9 / 7.22 / 7.22 |
| 16 | 8 x 8 | 128 | 1.2 / 1.11 / 1.85 / 1.8 | 4.74 / 4.98 / 6.96 / 9.03 |
| 16 | 16 x 16 | 128 | 1.42 / 0.928 / 1.71 / 1.53 | 4.69 / 4.3 / 6.52 / 7.83 |
| 16 | 32 x 32 | 512 | 1.82 / 1.43 / 1.63 / 1.78 | 9.87 / 6.05 / 7.28 / 10.4 |
| 16 | 64 x 64 | 128 | 2.14 / 2.31 / 1.74 / 1.97 | 9.31 / 9.31 / 9.78 / 7.78 |

**Blocks merged from four tiles, typical inputs, blocks of 32 by 32**

| B | median eta I / T | max eta I / T | T identical to I |
|---:|:---|:---|---:|
| 1 | 1.19 / 0.891 | 5.97 / 4.96 | 25% |
| 2 | 1.1 / 0.947 | 7.08 / 5.23 | 33% |
| 3 | 1 / 0.992 | 5.62 / 4.67 | 25% |
| 4 | 0.96 / 1.06 | 5.05 / 5.33 | 31% |
| 5 | 3.13 / 1.44 | 15.9 / 6.42 | 10% |
| 8 | 2.41 / 1.5 | 11.9 / 6.17 | 13% |
| 16 | 1.82 / 1.47 | 9.87 / 5.88 | 13% |
| 32 | 1.48 / 1.88 | 9.99 / 8.62 | 14% |
| 64 | 2.32 / 2.53 | 9 / 12.4 | 5% |
| 128 | 3.3 / 4.08 | 9.47 / 11.6 | 9% |
| 159 | 2.96 / 4.53 | 11.5 / 22.5 | 9% |
| 256 | 4.26 / 6.8 | 12.2 / 18.2 | 6% |

**Fused sums of fastCD against the moments, typical inputs, blocks of 32 by 32**

| B | N | median eta I / F / G | p99 eta I / F / G | max eta I / F / G | F at least as accurate as I | F as accurate as I |
|---:|---:|:---|:---|:---|---:|---:|
| 1 | 1 | 1.19 / 4.01 / 1.27 | 5.01 / 13.9 / 4.93 | 5.97 / 16.7 / 5.65 | 23% | 7% |
| 2 | 2 | 1.1 / 2.81 / 1.28 | 4.32 / 10.4 / 4.7 | 7.08 / 14.4 / 5.85 | 31% | 11% |
| 3 | 4 | 1 / 2.23 / 1.09 | 3.93 / 8.04 / 4.57 | 5.62 / 10.4 / 5.26 | 33% | 13% |
| 4 | 4 | 0.96 / 1.98 / 1.06 | 4.37 / 7.65 / 4.28 | 5.05 / 9.31 / 5.69 | 40% | 15% |
| 5 | 8 | 3.13 / 1.87 / 1.14 | 10.7 / 8.01 / 4.02 | 15.9 / 9.98 / 5.2 | 79% | 19% |
| 8 | 8 | 2.41 / 1.51 / 1.09 | 9.23 / 5.83 / 4.47 | 11.9 / 6.51 / 6.15 | 76% | 21% |
| 16 | 16 | 1.82 / 1.43 / 1.07 | 7.75 / 5.24 / 4.14 | 9.87 / 6.05 / 5.12 | 73% | 24% |
| 32 | 32 | 1.48 / 1.55 / 1.32 | 6.33 / 5.33 / 4.9 | 9.99 / 6.99 / 6.62 | 68% | 35% |
| 64 | 64 | 2.32 / 1.89 / 1.87 | 7.94 / 6.45 / 6.07 | 9 / 7.13 / 7.36 | 77% | 33% |
| 128 | 128 | 3.3 / 2.15 / 2.1 | 9.07 / 5.6 / 6.29 | 9.47 / 5.76 / 6.76 | 75% | 9% |
| 159 | 256 | 2.96 / 3.32 / 2.89 | 10.9 / 12.3 / 12.7 | 11.5 / 13.5 / 14.5 | 59% | 16% |
| 256 | 256 | 4.26 / 3.14 / 2.65 | 11.8 / 9.78 / 10.5 | 12.2 / 10.2 / 11.2 | 78% | 16% |

**Families, B = 4**

| family | blocks | median eta I / F / Q / D | max eta I / F / Q / D | marked |
|:---|---:|:---|:---|---:|
| uniform | 256 | 1.04 / 1.98 / 1.97 / 2.02 | 5.05 / 8.57 / 7.29 / 7.33 | 0 |
| gaussian | 256 | 0.869 / 1.98 / 1.82 / 1.65 | 4.83 / 9.31 / 7.3 / 7.18 | 0 |
| uint8 | 32 | 1.06 / 0.88 / 1.8 / 1.8 | 2.51 / 2.95 / 5.6 / 5.6 | 0 |
| uint16 | 32 | 9.6 / 22 / 2.6 / 1.7 | 33.6 / 131 / 5.99 / 5.78 | 0 |
| identical | 32 | 0 / 2 / 1.8e-84 / 0 | 1.8e-84 / 8 / 6 / 2 | 0 |
| affine | 32 | 0.985 / 3.03 / 1.06 / 1 | 3.95 / 13.6 / 5.16 / 5.16 | 0 |
| inverted | 32 | 0.444 / 3.09 / 1.05 / 1.33 | 2.44 / 9.84 / 6.41 / 5.11 | 0 |
| independent | 32 | 0.0544 / 0.0718 / 0.727 / 0.705 | 0.173 / 0.314 / 1.52 / 1.74 | 0 |

**Families, B = 16**

| family | blocks | median eta I / F / Q / D | max eta I / F / Q / D | marked |
|:---|---:|:---|:---|---:|
| uniform | 256 | 2.04 / 1.38 / 1.96 / 2.12 | 9.56 / 6.05 / 7.28 / 10.4 | 0 |
| gaussian | 256 | 1.66 / 1.48 / 1.43 / 1.49 | 9.87 / 5.23 / 6.27 / 8.39 | 0 |
| uint8 | 32 | 1.11 / 1.17 / 1.84 / 2.48 | 3.36 / 3.8 / 5.39 / 4.93 | 0 |
| uint16 | 32 | 20.8 / 16.6 / 2.09 / 1.95 | 70.1 / 56.6 / 5.61 / 6.2 | 0 |
| identical | 32 | 0 / 2 / 0 / 0 | 1 / 6 / 6 / 4 | 0 |
| affine | 32 | 2.66 / 2.26 / 1.64 / 1.18 | 9.18 / 7.18 / 5.64 / 5.64 | 0 |
| inverted | 32 | 0.417 / 2 / 1.57 / 1.6 | 2.14 / 6.93 / 4.01 / 4.01 | 0 |
| independent | 32 | 0.0625 / 0.0324 / 0.275 / 0.344 | 0.246 / 0.14 / 1.69 / 1.38 | 0 |

**Means far from zero**, all band counts pooled

| family | blocks | median eta I / F / U / Q / D | max eta I / F / U / Q / D |
|:---|---:|:---|:---|
| offset_1e3 | 252 | 2.31 / 2.31 / 4.89 / 2.05 / 2.12 | 10.7 / 10.7 / 150 / 11.8 / 9.55 |
| offset_1e6 | 252 | 5.77 / 5.77 / 3.02e+03 / 101 / 101 | 1.15e+03 / 1.15e+03 / 1.59e+05 / 3.34e+03 / 3.34e+03 |
| offset_1e9 | 252 | 4.31e+03 / 4.31e+03 / 3.42e+06 / 5.91e+04 / 5.91e+04 | 4.36e+05 / 4.36e+05 / 1.16e+08 / 8.63e+06 / 8.63e+06 |
| offset_1e12 | 252 | 5.52e+06 / 5.52e+06 / 1.37e+10 / 1.15e+08 / 1.15e+08 | 8.03e+08 / 8.03e+08 / 2.09e+11 / 1e+10 / 1e+10 |

**Branches of the definition and extreme scales**, all band counts pooled

| family | blocks | median eta I / F / U / Q / D | results off by more than 1e-06, I / F / U / Q / D |
|:---|---:|:---|:---|
| zero_mean | 252 | 0.474 / 0.474 / 7.17e+15 / 7.3e+15 / 7.3e+15 | 0 / 0 / 232 / 248 / 248 |
| near_zero_mean | 252 | 2.63 / 2.63 / 1.36 / 2.58 / 2.64 | 0 / 0 / 2 / 1 / 1 |
| constant_integer | 252 | 1.67e-35 / 1.67e-35 / 7.89e-36 / 5.86e-35 / 5.99e-35 | 0 / 0 / 0 / 0 / 0 |
| constant_fraction | 252 | 1.4e-35 / 1.4e-35 / 1.72e-35 / 1.24e-18 / 1.24e-18 | 0 / 0 / 0 / 0 / 0 |
| scaled_down | 252 | 1.86 / 1.86 / 2.82e+05 / 7.51e+03 / 7.51e+03 | 0 / 0 / 0 / 0 / 0 |
| scaled_up | 252 | 1.68 / 2.29 / 1.68 / 1.75 / 1.56 | 0 / 0 / 0 / 0 / 0 |
| overflow | 252 | 2.09 / 2.09 / — / 3.29e+14 / 3.29e+14 | 0 / 0 / 252 / 252 / 252 |

The archived CSV summary also gives, for every family, band count, block size and method, the count, the median, mean, p95, p99 and maximum of eta, the relative errors, the number of results that are not finite, and the share of cases in which I is at least as accurate. Groups with fewer than 100 cases have an empirical p99 close to their maximum and large sampling uncertainty.

<!-- END STABILITY ACCURACY -->

### Reading the tables

**Typical inputs.** Up to four bands, where inazumaQ2n forms its moments in one
pass over each tile without a matrix product, its median error is about 1,
against 1.5 to 1.9 for the per-pixel evaluation, and it is at least as accurate
in 65% to 74% of the blocks. From five bands on, where the moments come from a
sequential sum and a matrix product, its median error is 1.5 to 4.3, against 1.1
to 2.5, and it is at least as accurate in 25% to 58% of the blocks. All of these
are a few units in the last place: the largest relative error of inazumaQ2n on
typical inputs is `1.8e-15`.

**Fused sums.** From five bands on, where `algorithm="auto"` can select them,
the median error of the fused sums is 1.4 to 3.3, below that of the moments at
every number of bands but 32 and 159, and close to that of the per-pixel
evaluation up to 64 bands. They are at least as accurate as the moments in 59%
to 79% of the blocks, and their largest error, 13.5, is below the 15.9 of the
moments. With one to four bands the order is reversed, because there the moments
come from the eight-lane kernel and have a median error of about 1, while the
fused sums, whose statistics are sequential sums over the whole block, have 2.0
to 4.0, which is why `auto` uses the moments for those images on every
processor. The two algorithms return the same bits in 17% of the typical blocks.

**Block size.** From 8 by 8 to 64 by 64 pixels, the median error of the
per-pixel evaluation stays between 1.5 and 2.0. That of inazumaQ2n stays about 1
with 3 bands and grows from 1.2 to 2.1 with 16 bands, where its sums run over
all the pixels of a tile, and that of the fused sums grows with the block for
the same reason, from 1.0 to 4.2 with 3 bands and from about 1 to 2.3 with 16.

**Tiles.** Merging a block from four tiles changes its value in the last bits in
80% of the blocks. Up to 16 bands T is as accurate as I or more, and from 32
bands on its median error is 1.1 to 1.6 times that of I. With the fused sums the
merged block is the more accurate at every number of bands, the median error of
G being 1.1 to 2.9 against 1.4 to 4.0 for F, since its statistics are sums over
a quarter of the pixels, merged, and its sum of products is the same.

**Families.** inazumaQ2n scores an image against itself as exactly one in most
blocks, 87% with the moments, while a fused sum does not see that its two
operands are equal and returns exactly one in 15% of them and a value within 21
units of it in the rest, with a median error of 2. On independent images, where
the value is small, the error of either algorithm is several times smaller than
that of the per-pixel evaluation up to 64 bands, since neither subtracts the
product of the means from the mean of the products. The `uint16` family is the
exception described in [Where rounding enters](#where-rounding-enters).

**Means far from zero.** The value of a block whose mean is far larger than its
variation is sensitive to every rounding of the samples, so every method loses
accuracy roughly in proportion to the ratio. With its guard, inazumaQ2n
evaluates such blocks from their normalized samples, centered before they are
multiplied, and from an offset of `10^6` on its median error is 14 to 21 times
smaller than that of the per-pixel evaluation, while without the guard (U) it
would be 500 to 2,500 times larger than with it. The fused sums are behind the
same guard, and their values on these families are those of I, bit for bit.

**Branches of the definition.** A reference band whose mean is exactly zero
changes the normalization of the test band, so the value is discontinuous there.
A floating-point mean of samples whose exact mean is zero is rarely exactly
zero, so the per-pixel evaluation takes the wrong branch in 248 of 252 blocks,
an error of order one, whereas inazumaQ2n decides the branch on the correctly
rounded sum of the band, which is zero exactly when the exact sum is, and takes
the right branch in every block. With a mean merely close to zero, every method
takes the right branch except in a few blocks whose floating-point mean happens
to be exactly zero.

**Constant bands.** The value of a block with a constant reference band is
determined, the standard deviation being replaced by the machine epsilon, and
every method computed it in this corpus. The value changes completely, however,
if one sample of the band changes by a single rounding error, which makes its
standard deviation a tiny number instead of the machine epsilon, so such blocks
should not be relied on, in this package or in any evaluation of the definition.

**Extreme scales.** The value does not depend on a common scale of the images.
When products of samples are subnormal, the standard deviations of the per-pixel
evaluation lose precision, and its median error is 7,500, and when squares of
deviations overflow, its standard deviations are infinite and every value it
returns is one. inazumaQ2n computes the standard deviations of such blocks from
samples scaled by a power of two, and its median errors are 1.9 and 2.1.

## Which results are identical bit for bit

Scores computed on different machines, or with different options, can be
compared to the last digit only where they are identical bit for bit, and the
kernels are written so that most properties of the machine do not change the
result, while a few choices do.

### Identical

- **Instruction sets.** The compiled kernels give the same bits with AVX-512,
  with AVX2, with NEON, and with the baseline code. The kernels for at most four
  bands achieve this by construction, with eight fixed lanes and a fixed order
  of reduction, and the centering kernel has no dependence between the bands it
  handles together. The tests compare every instruction set the processor
  supports, and a freestanding build of the kernels was run under emulation of
  AArch64 and gave the same checksum as on x86-64.
- **Sample types.** An image of `uint8`, `uint16`, or `float32` samples gives
  the same bits as its conversion to `float64`, because each sample is converted
  exactly before any arithmetic.
- **Workers and plans.** The number of workers, the size of the batches of
  tiles, and the reuse of a plan do not change the result, and with the fused
  sums neither does the width of the strips of blocks that share tiles.
- **The vector kernels of fastCD.** fastCD runs its fused sums with an AVX-512
  or an AVX2 kernel, which perform the same operations in the same order. Builds
  restricted to each gave the same bits on the host of the study, for blocks
  with and without overlap, so a build that contains them, as those with GCC or
  Clang do, gives the same values with `algorithm="fastcd"` on every x86-64
  processor with AVX2 and FMA.

For images with at most four bands, on a processor with a fused multiply-add
instruction, these together mean that the value of every block that is not
evaluated a second time is the same on every machine, and the index is the mean
of the values of the blocks, taken by NumPy.

### Not identical

- **The algorithm.** The moments and the fused sums round differently, and
  `algorithm="auto"` selects between them by the number of bands, the overlap of
  the blocks, and the kernel that fastCD runs on the processor, so
  `algorithm="moments"` or `algorithm="fastcd"` keeps one.
- **The portable kernel of fastCD.** Where fastCD has no vector kernel, and the
  processor no fused multiply-add instruction, its fused sums round products and
  sums separately, and the values differ in the last digits from those of its
  vector kernels.
- **The matrix-product library.** With the moments and more than four bands the
  cross moments are a matrix product, whose order of summation belongs to the
  library, so NumPy with OpenBLAS and MKL, different versions of either, and
  different processors can differ in the last digits. `backend="numpy"` keeps
  one library, which fixes the library but not its internal choices.
- **Tiles.** With `ws` a multiple of `stride` the moments of a block are merged
  from its tiles, and otherwise computed in one piece, and the two differ in
  rounding, as do the fused sums, whose statistics are merged from the tiles in
  the same way.
- **Processors without a fused multiply-add.** On x86-64 processors that lack
  AVX2 or FMA, images with at most four bands take the route through a matrix
  product, which rounds differently from the kernels.

## What this means for your code

- **Read Q2n values to a few digits.** On ordinary images every method in the
  study agrees with the exact value to about `10^-14` or better, and differences
  beyond that are rounding, not quality.
- **Expect identical bits only from the same route and library,** as listed
  above, and compare scores across machines with a tolerance.
- **Avoid blocks whose reference is constant,** as in masked or synthetic
  images, or treat their values as undetermined.
- **Subtract a large common offset first.** For images whose values lie far from
  zero compared with their variation, such as a small signal on a large offset,
  subtracting a constant from both images before scoring does not change the
  value of the index, except through the zero-mean branch, and improves its
  accuracy in every implementation.
- **Keep images finite and within range.** A block that contains a NaN or an
  infinity has the value NaN, and so has the index, with no exception raised,
  and samples whose sum over a block overflows, above about `10^308` divided by
  the number of pixels, are outside the range of the package.

## Limitations

- The study is empirical: its medians and maxima describe this corpus, not all
  inputs, and no bound on the error of either algorithm is proved. Only the
  typical groups through 64 bands, the block sizes, and the families pooled over
  the numbers of bands have 100 cases or more, and the other groups are too
  small to estimate tails.
- Blocks of 8 to 64 pixels with 1 to 256 bands were measured, and blocks merged
  from four tiles only, so overlaps that merge a block from more tiles were not
  measured.
- The matrix products of the moments were computed by NumPy with OpenBLAS and
  the fused sums by the AVX-512 kernel of fastCD, so the accuracy with MKL, or
  with the portable kernel of fastCD, was not measured.
- Every case is a single block, so the mean that forms the index from the values
  of the blocks is not part of the measurement.

## Checking and reproducing the results

The study is archived in
[benchmark-results/stability-20261121/](../benchmark-results/stability-20261121/)
with its record `study.json`, the summary tables and key figures, the charts in
PNG, SVG, and PDF, and `SHA256SUMS`, while the raw per-case results are not
distributed; every case records its seed and the SHA-256 of its samples, from
which the samples can be generated again.

To check the study, run from the repository root:

```sh
python tools/check_archives.py
```

The checker uses only the Python standard library. It compares every published
file with `SHA256SUMS` and checks that the tables on this page are the tables of
the study, and with `--full` it also requires the raw files, checks them against
their recorded hashes, and regenerates the tables from them, which needs NumPy.

To repeat the study, install the `benchmark` extras and run:

```sh
python benchmarks/stability_study.py --profile full --jobs 8 --update-docs
```

`--profile smoke` runs one case of every family at small sizes in seconds, to
test the tooling, and `--update-docs` rewrites the tables on this page from the
new results.

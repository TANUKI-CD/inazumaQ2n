# Conventions and results

This page states the definition of the Q2n index that inazumaQ2n computes, and
it explains how the package evaluates that definition without forming a single
product of pixels in a Cayley-Dickson algebra, as the definition itself does.
The index is due to Garzelli and Nencini, *Hypercomplex quality assessment of
multi/hyperspectral images*, IEEE Geoscience and Remote Sensing Letters 6(4),
2009, who introduced it as the extension of the universal image quality index,
and of its four-band form Q4, to any number of bands. inazumaQ2n computes it in
double precision as the reference MATLAB implementation in the pansharpening
toolbox of Vivone et al., *A critical comparison among pansharpening
algorithms*, IEEE Transactions on Geoscience and Remote Sensing 53(5), 2015,
defines it. The speed of the evaluation is reported in the
[benchmarks](benchmark.md) and its accuracy in the [stability
study](stability.md).

The page has four parts:

1. **The index.** The blocks, the algebra, and the value of a block, the
   properties that follow from the definition, and how the MATLAB toolbox
   relates to it.
2. **Evaluation from real moments.** The covariance of a block is a signed
   contraction of a matrix of real cross moments, and every other term of its
   value is a real statistic of the bands.
3. **Evaluation by fused sums.** The same covariance is one sum of products in
   the algebra, which fastCD computes without forming the products.
4. **The kernels, floating point, and cost.** How the compiled kernels obtain
   the statistics, which blocks are evaluated a second time because the
   identities can fail in floating point, and what each route costs.

## The index

### Images and blocks

Let the reference image `X` and the test image `Y` have `H` rows, `W` columns,
and `B` bands. The index is the mean of a value computed on each block of `ws`
by `ws` pixels, whose origins are the multiples of `stride` in each direction,
so that there are `ceil(H / stride)` rows and `ceil(W / stride)` columns of
blocks. A block that extends beyond the bottom or the right edge reads the image
through its symmetric reflection, in which row `H + k` is row `H - 1 - k` and
column `W + k` is column `W - 1 - k`, and the package requires that no block
extend beyond an edge by more than the size of the image, so that every row and
column it reads is a single reflection.

### The algebra

A pixel is treated as an element of the Cayley-Dickson algebra `A_n` of
dimension `N = 2^n`, where `N` is the least power of two that is at least `B`,
with its bands as the first `B` coordinates and zeros as the rest. In the
convention of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD), the algebra
is built by repeated doubling, with the conjugation and the product

```text
A_0 = R,   A_n = A_(n-1) x A_(n-1),
conj((a, b)) = (conj(a), -b),   (a, b)(c, d) = (ac - conj(d) b,  d a + b conj(c)),
```

so that the product of two basis elements is a signed basis element whose index
is the bitwise XOR of the two indices, `e_i e_j = sigma(i, j) e_(i XOR j)`.
Starting from `sigma_1(0, 0) = 1`, the signs in dimension `N` follow from those
in dimension `N/2` by

```text
sigma_N(i,       j)       =  sigma_(N/2)(i, j)          sigma_N(i,       N/2 + j) =  sigma_(N/2)(j, i)
sigma_N(N/2 + i, j)       =  eps(j) sigma_(N/2)(i, j)   sigma_N(N/2 + i, N/2 + j) = -eps(j) sigma_(N/2)(j, i)
```

for `i` and `j` below `N/2`, where `eps(0) = 1` and `eps(j) = -1` for `j > 0`.

The reference implementations multiply with a recursion that equals this product
followed by a fixed pattern of sign changes on the coordinates. Since the index
takes only the Euclidean norm of a linear combination of products, and a pattern
of sign changes preserves that norm, the pattern has no effect on the index, and
inazumaQ2n does not compute it.

### The value of a block

Each block is normalized with the statistics of its reference image. Let the
block have `P = ws^2` pixels, and for each band `i` let `a_i` be the mean of the
reference band over the block and `c_i` its unbiased standard deviation,
replaced by the machine epsilon when it is zero. The normalized images are

```text
x_i = (X_i - a_i) / c_i + 1,
y_i = (Y_i - a_i) / c_i + 1,     or   y_i = Y_i + 1   when a_i is exactly zero,
```

and the covariance is taken against the conjugate `z = conj(y)`, whose
coordinates are `z_0 = y_0` and `z_i = -y_i` for `i > 0`. The second form of
`y_i` is the behavior of the reference implementations, which test the mean of
the band and not its standard deviation.

With `m1` and `m2` the means of `x` and `z` over the block, as elements of the
algebra, with products taken in the algebra, and with `|.|` the Euclidean norm
of the coordinates, the toolbox forms

```text
t4   = |m1|^2 + |m2|^2,
t3   = P / (P - 1) (mean |x|^2 + mean |z|^2) - P / (P - 1) t4,
bias = 2 |m1| |m2| / t4,
q    = (P / (P - 1) mean (x z) - P / (P - 1) m1 m2) bias 2 / t3,
```

where `t3` is the sum over the bands of the unbiased variances of `x` and `z`,
`t4` the sum of the squared norms of their means, and the first factor of `q`
the hypercomplex covariance of `x` and `z`; the names are those of the toolbox.
The value of the block is `|q|`, or `|bias|` when `t3` is zero, and the index is
the mean of the values of all blocks.

### Properties that follow from the definition

- **Asymmetry.** The index is not symmetric in its arguments, because the
  reference image supplies the normalization of both.
- **Values above one.** With at most eight bands the algebra is
  norm-multiplicative and the value of a block is at most one, but from sixteen
  bands on it is not, and a block can score above one: the test suite constructs
  a 16-band pair that scores above 1.05, and the package does not clip.
- **Zero means.** A reference band whose mean over a block is exactly zero
  changes the normalization of the test band, so an image compared with itself
  can score below one.
- **Constant bands.** A reference band that is constant over a block is divided
  by the machine epsilon. The value of such a block is determined, but it
  changes completely when the band is perturbed by a single rounding error,
  which makes its normalization a tiny standard deviation instead of the machine
  epsilon (see [Numerical stability](stability.md)).

### The MATLAB toolbox and other implementations

The `q2n` of the MATLAB toolbox converts both images to `uint16` before it does
anything else, which rounds every sample to the nearest integer and limits it to
the range 0 to 65535, while inazumaQ2n evaluates the same definition on the
samples as they are given. On images whose samples are integers in that range,
as the products of the sensors are, the two compute the same index, and on any
other image the toolbox computes what inazumaQ2n computes for the rounded
images. The evaluation script of the toolbox calls `q2n` with blocks of 32
pixels and a stride equal to the block, that is, without overlap, which are the
defaults of the package, and [Against the published
toolboxes](benchmark.md#against-the-published-toolboxes) describes how the
agreement is tested against the code of the toolbox itself.

Other ports of the MATLAB toolbox differ from this definition in details that
change the value, such as the rounding of the inputs, the replacement of a zero
standard deviation, the estimator of the standard deviation, and the side on
which the image is padded, and inazumaQ2n does not reproduce their values.

## Evaluation from real moments

The definition multiplies every normalized pixel of a block in the algebra, but
since the product is bilinear, the covariance that the value needs can be
obtained from real sums over the pixels instead, and the rest of the value from
real statistics of the bands.

### The covariance is a contraction of a real matrix

By bilinearity of the product, the covariance term of `q` is a sum of products
of centered pixels,

```text
P / (P - 1) (mean (x z) - m1 m2) = 1 / (P - 1) sum over p of (x_p - m1)(z_p - m2),
```

where `p` runs over the pixels of the block. For elements `u` and `v` of the
algebra the coordinates of the product are `(u v)_k = sum over i XOR j = k of
sigma(i, j) u_i v_j`, so a sum of products over the pixels is a signed
contraction of the matrix of real cross moments,

```text
sum over p of u_p v_p = mu(G),     G[i][j] = sum over p of u_p[i] v_p[j],
mu(G)_k = sum over i XOR j = k of sigma(i, j) G[i][j],
```

and computing `G` is one matrix product, the transpose of the `P` by `N` matrix
of the `u_p` times that of the `v_p`, in which no product of two pixels is
formed.

### Only the bands of the image enter the matrix

Because the normalization is affine in each band, the centered normalized pixels
are scaled centered samples,

```text
(x_p - m1)_i = (X_p[i] - a_i) / c_i,
(z_p - m2)_j = s_j (Y_p[j] - b_j) / c'_j,
```

where `b_j` is the mean of the test band, `c'_j` is `c_j`, or 1 when `a_j` is
exactly zero, and `s_0 = 1` and `s_j = -1` for `j > 0` are the signs of the
conjugate. A band beyond the image is constant after normalization, so its
centered samples vanish, and with the `B` by `B` matrix of centered cross
moments of the block,

```text
C[i][j] = sum over p of (X_p[i] - a_i)(Y_p[j] - b_j),     0 <= i, j < B,
```

the hypercomplex covariance of the block is

```text
1 / (P - 1) mu(D C D'),     D = diag(1 / c_i),   D' = diag(s_j / c'_j).
```

`C` has `B` rows and columns, not `N`, so an image with 159 bands needs a 159 by
159 matrix although its algebra has dimension 256, and the scales and the signs
of the conjugate are applied to `C` once per block, in the contraction.

### The remaining terms are real statistics

The other terms of the value need no matrix at all. In exact arithmetic every
coordinate of `m1` is one, including those beyond the image, where the
normalized sample is the constant one, so `|m1|^2 = N`. A coordinate of `m2`
within the image is `s_i (1 + (b_i - a_i) / c'_i)`, and one beyond it has
absolute value one, so that

```text
n2 = |m2|^2 = (N - B) + sum over i < B of (1 + (b_i - a_i) / c'_i)^2,
bias = 2 sqrt(N n2) / (N + n2).
```

The variance term `t3` is the sum over the bands of the unbiased variances of
`x_i` and `z_i`, which are formed from the centered sums of squares

```text
Vx_i = sum over p of (X_p[i] - a_i)^2,     Vy_i = sum over p of (Y_p[i] - b_i)^2,
```

scaled as `Vx_i / c_i^2` and `Vy_i / c'_i^2` and divided by `P - 1`. Since the
factors `1 / (P - 1)` of the covariance and of `t3` cancel, the value of the
block is

```text
2 bias |mu(D C D')| / sum over i < B of (Vx_i / c_i^2 + Vy_i / c'_i^2),
```

so a block is determined by `C` and by four statistics of each band, the means
`a_i` and `b_i` and the centered sums of squares `Vx_i` and `Vy_i`.

### Blocks that overlap share tiles

When `ws` is a multiple of `stride` and larger than it, the image is cut into
tiles of `stride` by `stride` pixels, and each block is the union of
`T = (ws / stride)^2` tiles, whose moments are computed once and merged for
every block that contains them. With `a_t`, `b_t`, and `C_t` the means and the
centered matrix of tile `t`, `P_t` its number of pixels, and `a` and `b` the
means of the block,

```text
C    = sum over tiles of (C_t + P_t (a_t - a)(b_t - b)^T),
Vx_i = sum over tiles of (Vx_t,i + P_t (a_t,i - a_i)^2),
```

and likewise for `Vy`. Every term is a centered quantity, so no difference of
large raw moments is formed, and each pixel is read and multiplied once however
many blocks contain it, which with `ws = 2 stride`, the usual overlap, is a
quarter of the work of evaluating each block separately. The normalization is
applied after the merge, because it belongs to the block, and when `ws` is not a
multiple of `stride`, every block is its own tile.

## Evaluation by fused sums

The covariance of a block is also what its definition says it is, a sum of `P`
products in the algebra, and fastCD computes such a sum without forming its
products, at a cost that grows with `N log N` for each pixel rather than with
`B^2`.

### A sum of products without the products

In the notation above,

```text
mu(D C D') = sum over p of u_p v_p,     u_p = (X_p - a) D,     v_p = (Y_p - b) D',
```

with the coordinates beyond the bands of the image equal to zero. A
multiplication algorithm with `K` real multiplications computes `u v` as fixed
linear combinations of `K` products `l_m(u) r_m(v)` of linear forms of its
operands, so that

```text
sum over p of u_p v_p = combination of ( sum over p of l_m(u_p) r_m(v_p) ),     m = 1, ..., K,
```

where the `K` products are accumulated over the pixels and the combination, the
additions that follow the multiplications in the algorithm, is applied once to
the accumulated sums. [fastCD](https://git.tanuki-cd.com/algebraity/fastCD)
calls this a *fused sum* and provides it for its quasilinear algorithm, in which
`K` grows like `N log N`, so a block costs `O(P N log N)` operations, against
the `P B^2` of the matrix `C`, and no matrix is formed: the result is the vector
`mu(D C D')` itself.

### Records and the affine map

For this route, a kernel writes the pixels of one block of both images as
records of `N` doubles, sums every band, and accumulates the centered sums of
squares in a second pass over the records, with the same operations in the same
order as the kernel that centers the samples for the moments, so that a block
has the same statistics by either route. The means and the scales `D` and `D'`
then enter the fused sum as the affine map that fastCD applies to its operands
as it reads them, in which coordinate `i` of a record of `X` is read as
`(X_p[i] - a_i) / c_i`, and the value of the block follows as before, from the
norm of the sum and from the statistics.

The fused sums are those of fastCD 1.5.0, compiled into the extension module
from `vendor/fastcd` with fastCD's own compiler options, and fastCD runs them
with an AVX-512 or an AVX2 kernel on x86-64, selected at run time, and with a
portable kernel elsewhere.

### When the fused sums are faster

Two properties decide when this route is the faster one. Its cost depends on the
dimension `N` and not on the number of bands `B`, so an image with few bands
above a power of two pays for the whole dimension, while the matrix `C` has only
`B` rows and columns. Moreover, the normalization `D` belongs to the block and
is applied before the products, so the sum of a block cannot be put together
from sums of its tiles, as its matrix `C` can, and every block is summed on its
own. The [benchmarks](benchmark.md#moments-and-fused-sums) measure both effects,
and `algorithm="auto"` follows them.

What blocks that overlap do share is everything before the sum. The records of a
tile are its raw samples, and the means and scales of a block reach its sum only
through the affine map, so when `ws` is a multiple of `stride`, the kernel
writes the records and the statistics of each tile once, merges the statistics
of a block from those of its tiles with the identities above, and repeats only
the sum. It moves down the image keeping `ws / stride` rows of tiles, stored so
that the tiles of every block are one contiguous run of records, and a block
remains one fused sum.

## The kernels

Both routes start from the statistics of the bands, which compiled kernels
obtain tile by tile, and both end in one routine for each block that turns the
statistics into the value of the block.

### At most four bands

For images with one to four bands, one kernel computes `C` and the statistics of
a tile in two passes over its pixels, with no workspace and no matrix product.
The first pass sums the differences from an anchor, the first pixel of the tile,
which gives the means without accumulating large values, and the second
subtracts the means and accumulates the `B^2` cross products and the `2B`
squares with fused multiply-adds, so that each pair of bands costs one fused
operation for each pixel, beside `2B` subtractions and `2B` additions for each
pixel in the two passes.

The kernel handles eight pixels of a row at once, with one eight-lane
accumulator for each moment: one 512-bit register with AVX-512, two 256-bit
registers with AVX2, and four 128-bit registers with NEON. The columns of a tile
row beyond the last multiple of eight go to one scalar accumulator for each
moment, and the eight lanes are added in a fixed order at the end of the tile,
so every instruction set performs the same operations in the same order, and the
kernel is generated once, by `tools/generate_kernels.py`, in terms of an
eight-lane vector that each instruction set defines.

### More bands

For other images, a kernel centers the samples of a batch of tiles into a
workspace in two passes, the first for the means and the second for the centered
samples and their sums of squares. The matrix `C` of each tile is then the
product of the transpose of its centered reference samples with its centered
test samples, a `B` by `P` times `P` by `B` product computed by NumPy's `matmul`
or, when it is installed, directly by the `dgemm` of Intel MKL, and the batch is
sized so that the workspace of both images stays near 8 MiB for each worker.

### Samples read in place

All kernels are compiled for samples of type `float64`, `float32`, `uint8`, and
`uint16`, and read images of those types from the memory of the caller without a
converted copy. Each sample is converted to a double as it is loaded, which is
exact for all four types, so the result is identical to that for the converted
image, while images of other real types, images that are not C-contiguous, and
pairs of different types are converted to `float64` first.

### Evaluating the blocks

With the moments, one routine merges the tiles of each block, forms the scales
of the bands, contracts `D C D'` with the signs, and evaluates the value of the
block, the signs being a table of `B^2` bytes built once by the recurrence
above. With the fused sums there is nothing to merge or contract, and the
routine that sums a block forms the same scales and evaluates its value
directly.

## Blocks evaluated from their normalized samples

The identities above hold in exact arithmetic, but in floating point they can
fail in three ways, so the routines that evaluate the blocks, by either route,
mark the blocks where they may fail and evaluate those blocks again in the
manner of the definition.

**Where the identities fail.** The identity `|m1|^2 = N` treats the two images
inconsistently when a band has a mean far larger than its standard deviation:
the definition normalizes both images with the same rounded mean, so its
rounding errors largely cancel, while the identity assumes the exact mean for
the reference only. The branches of the definition, a mean that is exactly zero
and a band that is exactly constant, cannot be decided from rounded moments, and
raw moments can underflow or overflow at extreme scales.

**Which blocks are marked.** A block is marked when, for some band,

- the absolute mean of the reference exceeds 1000 times its standard deviation;
- the reference is not constant and its absolute mean is at most `P` times the
  machine epsilon times its standard deviation; or
- a moment is not finite, a nonzero sum of squares is below `P` times the
  smallest normal double, or a nonzero mean is below the square root of the
  smallest normal double in absolute value, so that products of samples may have
  lost information to underflow.

**How a marked block is evaluated.** The statistics of a marked block are
computed so that they decide the branches of the definition exactly: the mean of
each reference band is its correctly rounded sum divided by the number of
pixels, so it is zero exactly when the exact mean is zero; a band counts as
constant, with its standard deviation replaced by the machine epsilon, exactly
when all its samples are equal; and the standard deviation is computed from
samples scaled by a power of two, so that their squares neither overflow nor
underflow. Both images are then normalized with these statistics, as the
definition does, the means of the normalized samples are computed rather than
assumed, and the covariance and the variances are formed from the normalized
samples minus those means, so that no difference of large sums is taken. The
thresholds are conservative choices, and the [stability study](stability.md)
measures the effect of this evaluation on every family of inputs that triggers
it.

## Operation counts

For a tile of `P` pixels and `B` bands, the kernels of the moments perform the
following operations, in the kernel for at most four bands and in the centering
kernel with its matrix product for more bands:

| Stage | At most four bands | More bands |
| --- | --- | --- |
| Means | `2PB` subtractions, `2PB` additions | `2PB` additions |
| Centering | `2PB` subtractions | `2PB` subtractions, `2PB` stores |
| Squares | `2PB` fused multiply-adds | `2PB` multiplications, `2PB` additions |
| Cross moments | `P B^2` fused multiply-adds | one `B x P` by `P x B` matrix product |

The evaluation of a block of `T` tiles then costs `T B^2` multiply-adds for the
merge and the contraction, independently of `P`, while the definition costs one
product in an algebra of dimension `N` for every pixel of every block.

With the fused sums, a block of `P` pixels costs `2PB` stores and `2PB`
additions for the records and the means, `2PB` subtractions, multiplications,
and additions for the squares, and one fused sum of `P` products, whose
multiplications, `O(N log N)` of them in fastCD's quasilinear algorithm, are
performed once for every pixel and whose additions are performed once for the
block, so that nothing depends on `B^2`. When blocks of `T` tiles overlap, the
records and the statistics are those of the tiles and are formed once, and each
pixel enters `T` sums.

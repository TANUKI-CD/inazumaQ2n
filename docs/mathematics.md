# Conventions and results

This page states the definition of Q2n that inazumaq2n computes, the
identities by which it is computed without forming a hypercomplex product of
pixels, and what each compiled kernel does. The measured accuracy of the
result is in [Numerical stability](stability.md).

## The index

Q2n was introduced by Garzelli and Nencini, *Hypercomplex quality assessment
of multi/hyperspectral images*, IEEE Geoscience and Remote Sensing Letters
6(4), 2009, as the extension of the universal image quality index, and of its
four-band form Q4, to any number of bands. inazumaq2n computes it in double
precision as the reference MATLAB implementation in the pansharpening
toolbox of Vivone et al. (IEEE Transactions on Geoscience and Remote Sensing
53(5), 2015) defines it.

### Blocks

Let the reference image `X` and the test image `Y` have `H` rows, `W` columns
and `B` bands. The index is the mean of a value computed on each block of
`ws` by `ws` pixels. Blocks have their origins at the multiples of `stride`
in each direction, so there are `ceil(H / stride)` rows and
`ceil(W / stride)` columns of blocks. A block that extends beyond the bottom
or the right edge reads the image through its symmetric reflection, in which
row `H + k` is row `H - 1 - k` and column `W + k` is column `W - 1 - k`. The
package requires that no block extends beyond an edge by more than the image
size, so that every row and column it reads is a single reflection.

### The algebra

Let `N` be the least power of two that is at least `B`. A pixel is treated as
an element of the Cayley–Dickson algebra `A_n` of dimension `N = 2^n`, with
its bands as the first `B` coordinates and zeros as the rest. The convention
is that of [fastCD](https://git.tanuki-cd.com/algebraity/fastCD):

```text
A_0 := R,   A_n := A_(n-1) x A_(n-1),   (a, b)(c, d) := (ac - conj(d) b, d a + b conj(c)),
conj((a, b)) := (conj(a), -b),          e_i e_j = sigma(i, j) e_(i XOR j)
```

The signs `sigma(i, j)` of the products of basis elements satisfy, with
`M = N / 2`, `eps(0) = 1` and `eps(j) = -1` for `j > 0`:

```text
sigma_N(i,     j)     =  sigma_M(i, j)          sigma_N(i,     M + j) =  sigma_M(j, i)
sigma_N(M + i, j)     =  eps(j) sigma_M(i, j)   sigma_N(M + i, M + j) = -eps(j) sigma_M(j, i)
```

The reference implementations multiply with a recursion that equals this
product followed by a fixed pattern of sign changes on the coordinates. The
index takes only the Euclidean norm of a linear combination of products, and
a pattern of sign changes preserves that norm, so the pattern has no effect
on the index and is not computed.

### The value of a block

Let a block have `P = ws^2` pixels. For each band `i`, let `a_i` be the mean
of the reference band over the block and `c_i` its unbiased standard
deviation, replaced by the machine epsilon when it is zero. Both images are
normalized with the statistics of the reference:

```text
x_i = (X_i - a_i) / c_i + 1
y_i = (Y_i - a_i) / c_i + 1,     or   y_i = Y_i + 1   when a_i is exactly zero
z   = conj(y):   z_0 = y_0,   z_i = -y_i for i > 0
```

The second form of `y_i` is the behaviour of the reference implementations,
which test the mean of the band and not its standard deviation. With `m1`
and `m2` the means of `x` and `z` over the block, as elements of the algebra,

```text
t4   = |m1|^2 + |m2|^2
t3   = P / (P - 1) (mean |x|^2 + mean |z|^2) - P / (P - 1) t4
bias = 2 |m1| |m2| / t4
q    = (P / (P - 1) mean (x z) - P / (P - 1) m1 m2) bias 2 / t3
```

and the value of the block is `|q|`, or `|bias|` when `t3` is zero. The
products `x z` and `m1 m2` are products in the algebra.

### Properties that follow from the definition

- The index is not symmetric in its arguments: the reference image supplies
  the normalization of both.
- With at most eight bands the algebra is norm-multiplicative and the value
  of a block is at most one. From sixteen bands on it is not, and a block can
  score above one; the test suite constructs a 16-band pair that scores above
  1.05. The package does not clip.
- A reference band whose mean over a block is exactly zero changes the
  normalization of the test band, so an image compared with itself can score
  below one.
- A reference band that is constant over a block is divided by the machine
  epsilon. The value of such a block is determined, but it changes
  completely when the band is perturbed by a single rounding error, which
  makes its normalization a tiny standard deviation instead of the machine
  epsilon; see [Numerical stability](stability.md).
- Other ports of the MATLAB toolbox differ from this definition in details
  that change the value, such as the rounding of the inputs, the replacement
  of a zero standard deviation, the estimator of the standard deviation and
  the side on which the image is padded. Their values are not reproduced.

## Evaluation from real moments

### The covariance is a contraction of a real matrix

By bilinearity of the product,

```text
P / (P - 1) (mean (x z) - m1 m2) = 1 / (P - 1) sum_p (x_p - m1)(z_p - m2).
```

For `u` and `v` in the algebra, `(u v)_k = sum over i XOR j = k of
sigma(i, j) u_i v_j`. A sum of products over the pixels of a block is
therefore a signed contraction of the matrix of real cross moments:

```text
sum_p u_p v_p = mu(G),     G[i][j] = sum_p u_p[i] v_p[j],
mu(G)_k = sum over i XOR j = k of sigma(i, j) G[i][j].
```

No product of two pixels is formed. Computing `G` is one matrix product:
the transpose of the `P` by `N` matrix of the `u_p` times that of the `v_p`.

### Only the bands of the image enter the matrix

The normalization is affine in each band, so the centered normalized pixels
are scaled centered samples:

```text
(x_p - m1)_i = (X_p[i] - a_i) / c_i
(z_p - m2)_j = s_j (Y_p[j] - b_j) / c'_j
```

where `b_j` is the mean of the test band, `c'_j` is `c_j`, or 1 when `a_j` is
exactly zero, and `s_0 = 1`, `s_j = -1` for `j > 0` is the conjugate. A band
beyond the image is constant after normalization, so its centered samples
vanish. With

```text
C[i][j] = sum_p (X_p[i] - a_i)(Y_p[j] - b_j),     0 <= i, j < B,
```

the hypercomplex covariance of the block is

```text
1 / (P - 1) mu(D C D'),     D = diag(1 / c_i),   D' = diag(s_j / c'_j).
```

`C` has `B` rows and columns, not `N`: an image with 159 bands needs a 159 by
159 matrix, although its algebra has dimension 256. The scales and the signs
of the conjugate are applied to `C` once per block, in the contraction.

### The remaining terms are real statistics

In exact arithmetic every coordinate of `m1` is one, including those beyond
the image, where the normalized sample is the constant one. Hence
`|m1|^2 = N`. A coordinate of `m2` within the image is
`s_i (1 + (b_i - a_i) / c'_i)`, and one beyond it has absolute value one, so

```text
n2 := |m2|^2 = (N - B) + sum over i < B of (1 + (b_i - a_i) / c'_i)^2
bias = 2 sqrt(N n2) / (N + n2).
```

`t3` is the sum over the bands of the unbiased variances of `x_i` and `z_i`,
which are the centered sums of squares

```text
Vx_i = sum_p (X_p[i] - a_i)^2,     Vy_i = sum_p (Y_p[i] - b_i)^2
```

scaled as `Vx_i / c_i^2` and `Vy_i / c'_i^2` and divided by `P - 1`. The
factors `1 / (P - 1)` of the covariance and of `t3` cancel, and the value of
the block is

```text
2 bias |mu(D C D')| / sum over i < B of (Vx_i / c_i^2 + Vy_i / c'_i^2).
```

A block is thus determined by `C` and by four statistics of each band: the
means `a_i` and `b_i` and the centered sums of squares `Vx_i` and `Vy_i`.

### Blocks that overlap share tiles

When `ws` is a multiple of `stride` and larger than it, the image is cut
into tiles of `stride` by `stride` pixels and a block is the union of
`(ws / stride)^2` tiles. The moments are computed once per tile and merged
per block. With `a_t`, `b_t`, `C_t` the means and the centered matrix of
tile `t`, with `P_t` its number of pixels and `a`, `b` the means of the
block,

```text
C    = sum over tiles of (C_t + P_t (a_t - a)(b_t - b)^T)
Vx_i = sum over tiles of (Vx_t,i + P_t (a_t,i - a_i)^2),
```

and likewise for `Vy`. Every term is a centered quantity, so no difference
of large raw moments is formed. Each pixel is read and multiplied once
however many blocks contain it; with `ws = 2 stride`, the usual overlap,
that is a quarter of the work of evaluating each block separately. The
normalization is applied after the merge, because it belongs to the block.

When `ws` is not a multiple of `stride`, every block is its own tile.

## The kernels

The package cuts the image into tiles, obtains `C` and the four statistics
of every tile, and then evaluates the blocks.

### At most four bands: no workspace and no matrix product

For images with one to four bands, one kernel computes `C` and the
statistics of a tile in two passes over its pixels. The first pass sums the
differences from an anchor, the first pixel of the tile, which gives the
means without accumulating large values. The second pass subtracts the
means and accumulates the `B^2` cross products and the `2B` squares with
fused multiply-adds. Per pixel and band pair this is one fused operation,
with `2B` subtractions and `2B` additions per pixel for the two passes.

The kernel handles eight pixels of a row at once, with one eight-lane
accumulator per moment: one 512-bit register with AVX-512, two 256-bit
registers with AVX2, four 128-bit registers with NEON. The columns of a tile
row beyond the last multiple of eight go to one scalar accumulator per
moment, and the eight lanes are added in a fixed order at the end of the
tile. All instruction sets therefore perform the same operations in the
same order, and the kernel is generated once, by
`tools/generate_kernels.py`, in terms of an eight-lane vector that each
instruction set defines.

### More bands: centering and one matrix product per tile

For other images, a kernel centers the samples of a batch of tiles into a
workspace in two passes, the first for the means and the second for the
centered samples and their sums of squares. The matrix `C` of each tile is
then the product of the transpose of its centered reference samples with
its centered test samples, a `B` by `P` times `P` by `B` product computed by
NumPy's `matmul` or, when it is installed, directly by the `dgemm` of Intel
MKL. The batch is sized so that the workspace of both images stays near
8 MiB per worker.

### Samples are read in place

Both kernels are compiled for samples of type `float64`, `float32`, `uint8`
and `uint16`, and read images of those types from the memory of the caller
without a converted copy. Each sample is converted to a double as it is
loaded, which is exact for all four types, so the result is identical to
that for the converted image. Images of other real types, images that are
not C-contiguous, and pairs of different types are converted to `float64`
first.

### Evaluating the blocks

One routine merges the tiles of each block, forms the scales of the bands,
contracts `D C D'` with the signs, and evaluates the value of the block.
The signs are a table of `B^2` bytes, built once by the recurrence above.

### Blocks evaluated from their normalized samples

The identity `|m1|^2 = N` holds in exact arithmetic. In floating point it
treats the two images inconsistently when a band has a mean far larger than
its standard deviation: the definition normalizes both images with the same
rounded mean, so its rounding errors largely cancel, while the shortcut
assumes the exact mean for the reference only. The branches of the
definition, a mean exactly zero and a band exactly constant, cannot be
decided from rounded moments, and raw moments can underflow or overflow at
extreme scales. The routine that evaluates the blocks therefore marks a
block when, for some band,

- the absolute mean of the reference exceeds 1000 times its standard
  deviation;
- the reference is not constant and its absolute mean is at most
  `P` times the machine epsilon times its standard deviation; or
- a moment is not finite, a nonzero sum of squares is below `P` times the
  smallest normal double, or a nonzero mean is below the square root of the
  smallest normal double in absolute value, so that products of samples may
  have lost information to underflow.

A marked block is evaluated again from its normalized samples. The mean of
each reference band is its correctly rounded sum divided by the number of
pixels, so it is zero exactly when the exact mean is zero; a band is
constant, and its standard deviation replaced by the machine epsilon,
exactly when all its samples are equal; and the standard deviation is
computed from samples scaled by a power of two, so that their squares
neither overflow nor underflow. Both images are normalized with these
statistics, as the definition does, the means of the normalized samples are
computed rather than assumed, and the covariance and the variances are
formed from the normalized samples minus those means, so that no difference
of large sums is taken. The thresholds are conservative choices; the
[stability study](stability.md) measures the effect of the evaluation on
every family of inputs that triggers them.

## Operation counts

For a tile of `P` pixels and `B` bands, the kernels perform:

| Stage | At most four bands | More bands |
| --- | --- | --- |
| Means | `2PB` subtractions, `2PB` additions | `2PB` additions |
| Centering | `2PB` subtractions | `2PB` subtractions, `2PB` stores |
| Squares | `2PB` fused multiply-adds | `2PB` multiplications, `2PB` additions |
| Cross moments | `P B^2` fused multiply-adds | one `B x P` by `P x B` matrix product |

and the evaluation of a block of `T` tiles costs `T B^2` multiply-adds for
the merge and the contraction, independent of `P`. The count of the
definition, by comparison, is one product in an algebra of dimension `N` per
pixel of every block.

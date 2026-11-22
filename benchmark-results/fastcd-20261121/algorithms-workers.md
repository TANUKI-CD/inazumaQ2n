# Algorithms of inazumaQ2n

Median milliseconds of a call on reused plans over 9 rounds.
`moments` contracts real cross moments, `fastcd` evaluates one fused sum of fastCD per block, and `toolbox` is the recursive product of the reference toolboxes in NumPy. B is the number of bands and N the dimension of the algebra. `auto` is the algorithm that `algorithm="auto"` selects, and its regret is the time of that algorithm over the time of the faster one.

inazumaQ2n 0.1.0, kernels avx512, matrix products by mkl 2026.2.1, fastCD 1.5.0 with its avx512 fused-sum kernel; Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

## Workers, B = N, blocks of 32 without overlap, float64

| B | N | image | block / stride | workers | moments | moments, NumPy | fastcd | moments / fastcd | auto | regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 512 x 512 | 32 / 32 | 1 | 55.6 | 56.5 | 30.3 | 1.83 | fastcd | 1.00 |
| 256 | 256 | 256 x 256 | 32 / 32 | 1 | 177 | 222 | 76.7 | 2.31 | fastcd | 1.00 |
| 32 | 32 | 512 x 512 | 32 / 32 | 2 | 34.9 | 39.7 | 16.4 | 2.12 | fastcd | 1.00 |
| 256 | 256 | 256 x 256 | 32 / 32 | 2 | 132 | 139 | 48.8 | 2.69 | fastcd | 1.00 |
| 32 | 32 | 512 x 512 | 32 / 32 | 4 | 20.1 | 24.8 | 6.56 | 3.06 | fastcd | 1.00 |
| 256 | 256 | 256 x 256 | 32 / 32 | 4 | 76.7 | 87.8 | 28 | 2.74 | fastcd | 1.00 |
| 32 | 32 | 512 x 512 | 32 / 32 | 8 | 14.7 | 15.8 | 5.14 | 2.86 | fastcd | 1.00 |
| 256 | 256 | 256 x 256 | 32 / 32 | 8 | 55.2 | 55.8 | 22.6 | 2.44 | fastcd | 1.00 |

The values of the plans agree to 1.1e-16 in every case, and the largest regret of `auto` is 1.00.

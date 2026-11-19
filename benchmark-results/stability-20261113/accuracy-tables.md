8,988 block pairs from the full profile. I is inazumaQ2n; Q and D evaluate the definition pixel by pixel with the products of pyfastcd's quasilinear and direct algorithms; T is inazumaQ2n with the block merged from four tiles; U is inazumaQ2n without the evaluation by definition of the blocks its guards mark. B is the number of bands and N the dimension of the algebra. eta is the absolute error of the value of a block in units of u = 2^-53. "Typical" pools the uniform and Gaussian families. Empirical p99 values and maxima describe this corpus only.

**Typical inputs, blocks of 32 by 32 pixels**

| B | N | blocks | median eta I / Q / D | p99 eta I / Q / D | max eta I / Q / D | I at least as accurate as Q / D |
|---:|---:|---:|:---|:---|:---|:---|
| 1 | 1 | 512 | 1.19 / 1.61 / 1.61 | 5.01 / 6.93 / 6.93 | 5.97 / 8.19 / 8.19 | 65% / 65% |
| 2 | 2 | 512 | 1.1 / 1.82 / 1.82 | 4.32 / 7.37 / 7.37 | 7.08 / 8.09 / 8.09 | 73% / 73% |
| 3 | 4 | 512 | 1 / 1.66 / 1.5 | 3.93 / 5.66 / 5.69 | 5.62 / 8.16 / 7.13 | 73% / 71% |
| 4 | 4 | 512 | 0.96 / 1.89 / 1.83 | 4.37 / 6.9 / 6.9 | 5.05 / 7.3 / 7.33 | 74% / 72% |
| 5 | 8 | 512 | 3.13 / 2.02 / 1.91 | 10.7 / 7.46 / 7.85 | 15.9 / 8.89 / 10.8 | 40% / 38% |
| 8 | 8 | 512 | 2.41 / 1.77 / 1.83 | 9.23 / 6.65 / 7.69 | 11.9 / 8.92 / 9.03 | 46% / 48% |
| 16 | 16 | 512 | 1.82 / 1.63 / 1.78 | 7.75 / 6.66 / 7.32 | 9.87 / 7.28 / 10.4 | 54% / 55% |
| 32 | 32 | 128 | 1.48 / 1.55 / 1.62 | 6.33 / 6.64 / 6.68 | 9.99 / 8.66 / 6.91 | 58% / 56% |
| 64 | 64 | 128 | 2.32 / 1.86 / 1.88 | 7.94 / 6.41 / 6.41 | 9 / 7.2 / 7.2 | 51% / 48% |
| 128 | 128 | 32 | 3.3 / 1.21 / 1.06 | 9.07 / 5 / 4.28 | 9.47 / 5.24 / 4.48 | 28% / 25% |
| 159 | 256 | 32 | 2.96 / 2.03 / 2.47 | 10.9 / 6.22 / 5.12 | 11.5 / 6.29 / 5.2 | 44% / 41% |
| 256 | 256 | 32 | 4.26 / 2.1 / 1.92 | 11.8 / 5.38 / 4.15 | 12.2 / 5.46 / 4.19 | 25% / 25% |

**Relative error, typical inputs**

| B | median relative error I / Q / D | max relative error I / Q / D |
|---:|:---|:---|
| 1 | 1.55e-16 / 2.04e-16 / 2.04e-16 | 6.88e-16 / 9.44e-16 / 9.44e-16 |
| 2 | 1.36e-16 / 2.24e-16 / 2.24e-16 | 1.01e-15 / 1.09e-15 / 1.09e-15 |
| 3 | 1.25e-16 / 2.07e-16 / 1.94e-16 | 6.47e-16 / 1.24e-15 / 9.39e-16 |
| 4 | 1.21e-16 / 2.35e-16 / 2.32e-16 | 5.82e-16 / 1.06e-15 / 1.06e-15 |
| 5 | 3.91e-16 / 2.5e-16 / 2.4e-16 | 1.83e-15 / 1.02e-15 / 1.39e-15 |
| 8 | 3.11e-16 / 2.23e-16 / 2.28e-16 | 1.47e-15 / 1.03e-15 / 1.07e-15 |
| 16 | 2.34e-16 / 2.09e-16 / 2.26e-16 | 1.31e-15 / 8.57e-16 / 1.2e-15 |
| 32 | 1.86e-16 / 2.05e-16 / 2.14e-16 | 1.34e-15 / 9.98e-16 / 7.96e-16 |
| 64 | 3.01e-16 / 2.35e-16 / 2.54e-16 | 1.2e-15 / 8.64e-16 / 8.29e-16 |
| 128 | 4.54e-16 / 1.43e-16 / 1.34e-16 | 1.09e-15 / 7.16e-16 / 6.19e-16 |
| 159 | 4.04e-16 / 2.7e-16 / 2.98e-16 | 1.33e-15 / 8.41e-16 / 6.5e-16 |
| 256 | 5.42e-16 / 2.69e-16 / 2.42e-16 | 1.74e-15 / 6.31e-16 / 4.91e-16 |

**Block size, typical inputs**

| B | block | blocks | median eta I / Q / D | max eta I / Q / D |
|---:|---:|---:|:---|:---|
| 3 | 8 x 8 | 128 | 0.909 / 1.7 / 1.7 | 4.1 / 8.31 / 8.31 |
| 3 | 16 x 16 | 128 | 0.887 / 1.62 / 1.63 | 3.28 / 7.12 / 6.37 |
| 3 | 32 x 32 | 512 | 1 / 1.66 / 1.5 | 5.62 / 8.16 / 7.13 |
| 3 | 64 x 64 | 128 | 1.08 / 1.76 / 1.65 | 4.63 / 7.22 / 7.22 |
| 16 | 8 x 8 | 128 | 1.2 / 1.85 / 1.8 | 4.74 / 6.96 / 9.03 |
| 16 | 16 x 16 | 128 | 1.42 / 1.71 / 1.53 | 4.69 / 6.52 / 7.83 |
| 16 | 32 x 32 | 512 | 1.82 / 1.63 / 1.78 | 9.87 / 7.28 / 10.4 |
| 16 | 64 x 64 | 128 | 2.14 / 1.74 / 1.97 | 9.31 / 9.78 / 7.78 |

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

**Families, B = 4**

| family | blocks | median eta I / Q / D | max eta I / Q / D | marked |
|:---|---:|:---|:---|---:|
| uniform | 256 | 1.04 / 1.97 / 2.02 | 5.05 / 7.29 / 7.33 | 0 |
| gaussian | 256 | 0.869 / 1.82 / 1.65 | 4.83 / 7.3 / 7.18 | 0 |
| uint8 | 32 | 1.06 / 1.8 / 1.8 | 2.51 / 5.6 / 5.6 | 0 |
| uint16 | 32 | 9.6 / 2.6 / 1.7 | 33.6 / 5.99 / 5.78 | 0 |
| identical | 32 | 0 / 1.8e-84 / 0 | 1.8e-84 / 6 / 2 | 0 |
| affine | 32 | 0.985 / 1.06 / 1 | 3.95 / 5.16 / 5.16 | 0 |
| inverted | 32 | 0.444 / 1.05 / 1.33 | 2.44 / 6.41 / 5.11 | 0 |
| independent | 32 | 0.0544 / 0.727 / 0.705 | 0.173 / 1.52 / 1.74 | 0 |

**Families, B = 16**

| family | blocks | median eta I / Q / D | max eta I / Q / D | marked |
|:---|---:|:---|:---|---:|
| uniform | 256 | 2.04 / 1.96 / 2.12 | 9.56 / 7.28 / 10.4 | 0 |
| gaussian | 256 | 1.66 / 1.43 / 1.49 | 9.87 / 6.27 / 8.39 | 0 |
| uint8 | 32 | 1.11 / 1.84 / 2.48 | 3.36 / 5.39 / 4.93 | 0 |
| uint16 | 32 | 20.8 / 2.09 / 1.95 | 70.1 / 5.61 / 6.2 | 0 |
| identical | 32 | 0 / 0 / 0 | 1 / 6 / 4 | 0 |
| affine | 32 | 2.66 / 1.64 / 1.18 | 9.18 / 5.64 / 5.64 | 0 |
| inverted | 32 | 0.417 / 1.57 / 1.6 | 2.14 / 4.01 / 4.01 | 0 |
| independent | 32 | 0.0625 / 0.275 / 0.344 | 0.246 / 1.69 / 1.38 | 0 |

**Means far from zero**, all band counts pooled

| family | blocks | median eta I / U / Q / D | max eta I / U / Q / D |
|:---|---:|:---|:---|
| offset_1e3 | 252 | 2.31 / 4.89 / 2.05 / 2.12 | 10.7 / 150 / 11.8 / 9.55 |
| offset_1e6 | 252 | 5.77 / 3.02e+03 / 101 / 101 | 1.15e+03 / 1.59e+05 / 3.34e+03 / 3.34e+03 |
| offset_1e9 | 252 | 4.31e+03 / 3.42e+06 / 5.91e+04 / 5.91e+04 | 4.36e+05 / 1.16e+08 / 8.63e+06 / 8.63e+06 |
| offset_1e12 | 252 | 5.52e+06 / 1.37e+10 / 1.15e+08 / 1.15e+08 | 8.03e+08 / 2.09e+11 / 1e+10 / 1e+10 |

**Branches of the definition and extreme scales**, all band counts pooled

| family | blocks | median eta I / U / Q / D | results off by more than 1e-06, I / U / Q / D |
|:---|---:|:---|:---|
| zero_mean | 252 | 0.474 / 7.17e+15 / 7.3e+15 / 7.3e+15 | 0 / 232 / 248 / 248 |
| near_zero_mean | 252 | 2.63 / 1.36 / 2.58 / 2.64 | 0 / 2 / 1 / 1 |
| constant_integer | 252 | 1.67e-35 / 7.89e-36 / 5.86e-35 / 5.99e-35 | 0 / 0 / 0 / 0 |
| constant_fraction | 252 | 1.4e-35 / 1.72e-35 / 1.24e-18 / 1.24e-18 | 0 / 0 / 0 / 0 |
| scaled_down | 252 | 1.86 / 2.82e+05 / 7.51e+03 / 7.51e+03 | 0 / 0 / 0 / 0 |
| scaled_up | 252 | 1.68 / 1.68 / 1.75 / 1.56 | 0 / 0 / 0 / 0 |
| overflow | 252 | 2.09 / — / 3.29e+14 / 3.29e+14 | 0 / 252 / 252 / 252 |

The archived CSV summary also gives, for every family, band count, block size and method, the count, the median, mean, p95, p99 and maximum of eta, the relative errors, the number of results that are not finite, and the share of cases in which I is at least as accurate. Groups with fewer than 100 cases have an empirical p99 close to their maximum and large sampling uncertainty.

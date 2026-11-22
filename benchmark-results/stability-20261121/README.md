# Stability study, 2026-11-21

This directory holds the measurements behind [docs/stability.md](../../docs/stability.md), which defines the error measures and interprets the results.

## What was measured

The floating-point accuracy of the value of one block of a Q2n index, against the exact value of the definition for the stored samples.

- **I, inazumaQ2n:** the block evaluated from its centered moments, as `q2n` and `q2n_map` evaluate blocks that do not overlap.
- **F, inazumaQ2n with fastCD:** the block evaluated by one fused sum of products of fastCD, as with `algorithm="fastcd"`, behind the same guards as I.
- **G, inazumaQ2n with fastCD from tiles:** F with the records and the statistics of the block merged from four tiles, as for blocks that overlap with a stride of half the block size.
- **T, inazumaQ2n from tiles:** the block merged from four tiles, as for blocks that overlap with a stride of half the block size.
- **U, inazumaQ2n without its guards:** I without the evaluation by definition of the blocks that the guards mark.
- **Q and D, the definition pixel by pixel:** every normalized reference pixel multiplied by the conjugate of the normalized test pixel, with the products of pyfastcd's quasilinear (Q) and direct (D) algorithms, in `benchmarks/baseline.py`.

The full profile measured 8,988 block pairs from 19 families. The reference converts every binary64 sample to an integer times a power of two, forms the moments of the block in arbitrary-precision integers, decides every branch of the definition on them, and evaluates square roots and quotients to 100 significant digits.

## Host and software

| | |
|:---|:---|
| CPU | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz |
| Operating system | Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 |
| Python | 3.14.8 |
| NumPy | 2.5.4 |
| inazumaQ2n | 0.1.0, kernels avx512, matrix products by NumPy |
| fastCD | 1.5.0, compiled into inazumaQ2n, fused sums by its avx512 kernel |
| pyfastcd | 1.4.0 |
| Matplotlib | 3.11.2 |
| Worker processes | 16 |

## How it was produced

```sh
python benchmarks/stability_study.py --profile full --seed 20261113 --jobs 16 --output benchmark-results/stability-20261121 --update-docs
```

| Step | What it does | Duration |
|:---|:---|---:|
| accuracy | Measure every error against the exact value. | 16 s |
| tables | Tabulate the accuracy results. | 1 s |
| plots | Draw the charts. | 2 s |

The samples derive from the root seed 20261113. study.json records the SHA-256 of every file of the measured package, of the comparison and of the reference.

## Files

| File | Contents |
|:---|:---|
| README.md | This description, generated from the results. |
| SHA256SUMS | SHA-256 of every published file of the study. |
| study.json | The command, the host and software, the checks of the reference, the hashes of the measured sources and of the raw files, and every step with its duration. |
| key-figures.json | The figures quoted in docs/stability.md, computed from the results. |
| accuracy-tables.md | The accuracy tables shown in docs/stability.md. |
| accuracy-summary.csv | Error statistics for every family, band count, block size and method. |
| PLOTS.md | Description of the charts and the command that drew them. |
| plot-data.csv | Every plotted value in tabular form. |
| eta-bands, eta-offset, relative-error-bands | The charts, each as PNG, SVG and PDF. |

The raw per-case results are written to `raw/`. They are not distributed; study.json records their SHA-256, and `tools/check_archives.py --full` checks them where they are present. Every case records its seed and the SHA-256 of its samples, from which the samples can be generated again.

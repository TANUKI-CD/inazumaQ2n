# Accuracy charts

The charts show the accuracy of the value of a Q2n block, measured against the exact value of the definition. I is inazumaQ2n, Q and D evaluate the definition pixel by pixel with the products of pyfastcd's quasilinear and direct algorithms, and U is inazumaQ2n without the evaluation by definition of the blocks its guards mark.

| Chart | PNG | SVG | PDF |
|:---|:---|:---|:---|
| Median eta against the number of bands | [PNG](eta-bands.png) | [SVG](eta-bands.svg) | [PDF](eta-bands.pdf) |
| Median relative error against the number of bands | [PNG](relative-error-bands.png) | [SVG](relative-error-bands.svg) | [PDF](relative-error-bands.pdf) |
| Median eta against the offset of the samples | [PNG](eta-offset.png) | [SVG](eta-offset.svg) | [PDF](eta-offset.pdf) |

## Values

Each point of the band charts is the median over the uniform and Gaussian block pairs of 32 by 32 pixels with one number of bands; each point of the offset chart is the median over every number of bands of one offset family. The error measures are defined in docs/stability.md. Every plotted value, with the number of samples behind it, is in plot-data.csv.

## Axes

The number of bands is on a base-two logarithmic axis and the offset on a base-ten one. The eta axis of the band chart is linear and starts at zero; the other value axes are logarithmic. Medians do not describe worst-case errors; the maxima are in accuracy-tables.md.

## Reproduction

```sh
python benchmarks/stability_plots.py benchmark-results/stability-20261113
```

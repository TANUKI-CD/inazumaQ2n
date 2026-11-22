# inazumaQ2n and the published toolboxes

Median milliseconds over 9 rounds for inazumaQ2n, on a reused plan, and over at most 3 for the others. Images of `uint16` samples with B = N bands, blocks of 32 pixels without overlap. The MATLAB toolbox is timed inside GNU Octave 11.3.0, around its `q2n` alone. The difference is the largest difference of the value of a block from that of inazumaQ2n.

| N | image | algorithm | inazumaQ2n | NumPy port (`toolbox.py`) | MATLAB pansharpening toolbox, under GNU Octave | PanCollection, NumPy port | Hyperspectral pansharpening toolbox, PyTorch port | NumPy port (`toolbox.py`) / inazumaQ2n | MATLAB pansharpening toolbox, under GNU Octave / inazumaQ2n | PanCollection, NumPy port / inazumaQ2n | Hyperspectral pansharpening toolbox, PyTorch port / inazumaQ2n | difference, NumPy port (`toolbox.py`) | difference, MATLAB pansharpening toolbox, under GNU Octave | difference, PanCollection, NumPy port | difference, Hyperspectral pansharpening toolbox, PyTorch port |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 512 x 512 | moments | 0.413 | 46.2 | 622 | 59 | 142 | 112 | 1.51e+03 | 143 | 343 | 1.0e-15 | 1.0e-15 | 1.4e-07 | 7.5e-08 |
| 4 | 512 x 512 | moments | 1.54 | 123 | 1.19e+03 | 148 | 283 | 79.4 | 771 | 95.8 | 184 | 1.1e-15 | 8.9e-16 | 8.6e-08 | 7.3e-08 |
| 8 | 512 x 512 | fastcd | 4.63 | 345 | 2.59e+03 | 422 | 754 | 74.5 | 560 | 91.1 | 163 | 8.9e-16 | 1.0e-15 | 3.7e-08 | 1.2e-07 |
| 16 | 512 x 512 | fastcd | 7.94 | 1.1e+03 | 6.46e+03 | 1.31e+03 | 2.37e+03 | 139 | 813 | 164 | 298 | 1.1e-15 | 8.9e-16 | 1.5e-08 | 1.2e-07 |
| 32 | 256 x 256 | fastcd | 4.03 | 939 | 4.18e+03 | 1.07e+03 | 2.11e+03 | 233 | 1.04e+03 | 266 | 523 | 7.8e-16 | 7.8e-16 | 5.0e-09 | 1.0e-07 |
| 64 | 256 x 256 | fastcd | 11.6 | 3.45e+03 | 1.31e+04 | 3.94e+03 | 8.03e+03 | 297 | 1.13e+03 | 340 | 692 | 1.0e-15 | 1.1e-15 | 2.8e-09 | 1.4e-07 |
| 128 | 128 x 128 | fastcd | 7.7 | 3.42e+03 | 1.1e+04 | 3.79e+03 | 7.87e+03 | 444 | 1.43e+03 | 492 | 1.02e+03 | 7.8e-16 | 8.9e-16 | 1.7e-09 | 7.8e-08 |
| 256 | 128 x 128 | fastcd | 19.4 | 1.32e+04 | 4.11e+04 | 1.48e+04 | 3.15e+04 | 681 | 2.12e+03 | 763 | 1.62e+03 | 1.1e-15 | 1.7e-15 | 1.0e-09 | 9.1e-08 |
| 512 | 64 x 64 | fastcd | 9.64 | 1.3e+04 | 3.86e+04 | 1.47e+04 | 3.14e+04 | 1.35e+03 | 4e+03 | 1.53e+03 | 3.25e+03 | 1.3e-15 | 1.0e-15 | 3.6e-10 | 5.8e-08 |
| 1024 | 64 x 64 | fastcd | 27.8 | 5.2e+04 | 1.51e+05 | 5.72e+04 | 1.23e+05 | 1.87e+03 | 5.44e+03 | 2.06e+03 | 4.42e+03 | 1.2e-15 | 8.9e-16 | 3.4e-10 | 9.2e-08 |

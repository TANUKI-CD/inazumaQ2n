# inazumaQ2n and the published toolboxes, 2026-11-21

This directory holds the measurement behind the comparison table of the README and the Toolboxes section of [docs/benchmark.md](../../docs/benchmark.md): the Q2n of the MATLAB pansharpening toolbox, run as it is distributed under GNU Octave, and of two Python ports of it, beside inazumaQ2n and beside `benchmarks/toolbox.py`, the NumPy port of the same recursion that the algorithms protocol times.

| File | Contents |
|:---|:---|
| `toolboxes.json` | Every observation, the value of the index by every implementation, the largest difference of a block from inazumaQ2n, the pinned sources, the host and the hashes of the measured files |
| `toolboxes.md` | The report written by the driver |
| `toolboxes.log` | The output of the driver |

| Implementation | Source | License of the source |
|:---|:---|:---|
| MATLAB pansharpening toolbox, under GNU Octave | `q2n.m`, `onions_quality.m`, `onion_mult.m`, `onion_mult2D.m`, `norm_blocco.m` of [liangjiandeng/DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) at commit `a34f884af88967580e3ef4f61a89e12fdfe4cda5` | GPL-3.0 |
| PanCollection, NumPy port | `my_q2n.py` of [XiaoXiao-Woo/PanCollection](https://github.com/XiaoXiao-Woo/PanCollection) at commit `4bd170a4713b1fa3a713964225c1d37ed16b24ae` | GPL-2.0 |
| Hyperspectral pansharpening toolbox, PyTorch port | `metrics_rr.py` of [matciotola/hyperspectral_pansharpening_toolbox](https://github.com/matciotola/hyperspectral_pansharpening_toolbox) at commit `679c33ae254a72850589ac028bbbe723c3b6f499` | CC BY-NC-SA 4.0 |

The files of the three sources are fetched and verified by `benchmarks/toolboxes.py` and are not distributed here.

| | |
|:---|:---|
| CPU | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz |
| Operating system | Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 |
| Python | 3.14.8 |
| NumPy | 2.5.5 |
| PyTorch | 2.15.0 (CPU), torchvision 0.29.2, one thread |
| GNU Octave | 11.3.0, without the image package: `mean2` and `std2` come from `benchmarks/octave/shims` |
| inazumaQ2n | 0.1.0, working tree of 2026-11-21, `algorithm="auto"`, kernels avx512, fastCD 1.5.0 (avx512) |

```sh
python benchmarks/toolboxes.py fetch
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 30 \
    python benchmarks/benchmark.py --protocol toolboxes --rounds 9 \
    --toolbox-rounds 3 --output toolboxes.json
```

The run used an environment with PyTorch and without MKL, which does not matter here: with `algorithm="auto"` these images take the kernels for at most four bands or the fused sums of fastCD, and neither uses a matrix-product library. The images have `uint16` samples, on which the toolbox and inazumaQ2n compute the same index. inazumaQ2n is timed in 9 rounds on a reused plan. The others are timed in three rounds where a round takes less than two seconds and in one otherwise: the MATLAB toolbox inside Octave, around its `q2n` alone, and the Python implementations around their call. Octave interprets the recursion without the compiler of MATLAB, so its times are those of Octave and not of MATLAB.

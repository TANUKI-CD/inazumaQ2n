# Complete calls, kernels, and workers, 2026-11-22

This directory holds the measurements behind the sections Complete calls against the definition, The kernels for at most four bands, NumPy and MKL, and Workers of [docs/benchmark.md](../../docs/benchmark.md). The measurements behind `algorithm="auto"` are in [fastcd-20261121](../fastcd-20261121/README.md), and the comparison with the published toolboxes is in [toolboxes-20261121](../toolboxes-20261121/README.md).

| Files | Run |
|:---|:---|
| `calls.*` | `--protocol calls`, 9 rounds, on one logical CPU: `q2n` and a reused plan with `algorithm="auto"` and `backend="auto"`, a reused plan with `backend="numpy"` where the moments use a matrix product, and the definition evaluated pixel by pixel with the products of pyfastcd's quasilinear and direct algorithms (`benchmarks/baseline.py`). |
| `kernels.*` | `--protocol kernels`, 15 rounds, on one logical CPU: for images with one to four bands, the kernels that need no matrix product with AVX-512 and with AVX2, against the route through a matrix product with NumPy and with MKL. |
| `workers-1.*`, `workers-2.*`, `workers-4.*` | `--protocol calls --no-baselines` with 1, 2, and 4 workers, 9 rounds each, on four physical cores. |
| `host-load.log` | The load average of the host before the first run and after every run. |

Every `.json` file keeps every observation and records the host, the software, the thread pools in the process, and the SHA-256 of every measured source file, and the `.log` file beside it is the output of the driver.

| | |
|:---|:---|
| CPU | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz |
| Operating system | Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 |
| Python | 3.14.8 |
| NumPy | 2.5.4, with its bundled OpenBLAS 0.3.34 |
| Matrix products | mkl 2026.2.1 |
| pyfastcd | 1.4.0, for the per-pixel evaluation only |
| inazumaQ2n | 0.1.0, commit `7848d40`, built as a wheel is built, kernels avx512 |
| fastCD | 1.5.0, compiled from `vendor/fastcd`, fused sums by its avx512 kernel |

```sh
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 6 python benchmarks/benchmark.py --rounds 9 --output calls.json
taskset -c 6 python benchmarks/benchmark.py --protocol kernels --rounds 15 --output kernels.json
for w in 1 2 4; do
    taskset -c 6,8,10,12 python benchmarks/benchmark.py --rounds 9 --workers $w \
        --no-baselines --output workers-$w.json
done
```

The host is shared, and no governor, turbo, or SMT setting was changed; the load average stayed between 1.9 and 2.7 during the runs.

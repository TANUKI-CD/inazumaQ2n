# The fused sums of fastCD in inazumaQ2n, 2026-11-21

This directory holds the measurements behind `algorithm="auto"` (`src/inazumaq2n/_auto.py`) and the Algorithms section of [docs/benchmark.md](../../docs/benchmark.md). The accuracy of the fused sums is measured in [stability-20261121](../stability-20261121/README.md).

## Speed

`benchmarks/benchmark.py --protocol algorithms` times, on reused plans, the real cross moments (`moments`), the fused sums of fastCD (`fastcd`) and `algorithm="auto"`, and in the dimension sweep the recursive product of the reference toolboxes (`benchmarks/toolbox.py`) as a complete call. Every `.json` file keeps every observation and records the host, the software and the SHA-256 of the measured sources; the `.md` file beside it is the report the driver wrote, and the `.log` file its output.

| Files | Run |
|:---|:---|
| `algorithms-avx512.*` | the build as a wheel is built, on one logical CPU; fastCD selects its AVX-512 kernel. With the toolbox recursion in the dimension sweep. |
| `algorithms-workers.*` | the same build on eight physical cores, with 1, 2, 4 and 8 workers. |
| `algorithms-avx2.*` | a build with `-DFASTCD_SUM_MAX_KERNEL=2`, which withholds fastCD's AVX-512 kernel, run with `--simd avx2`, on one logical CPU. |
| `algorithms-portable.*` | a build with `-DFASTCD_SUM_MAX_KERNEL=1`, which leaves fastCD's portable kernel, run with `--simd baseline`, on one logical CPU; 5 rounds. |

| | |
|:---|:---|
| CPU | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz |
| Operating system | Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 |
| Python | 3.14.8 |
| NumPy | 2.5.4 |
| Matrix products | mkl 2026.2.1 |
| inazumaQ2n | 0.1.0, working tree of 2026-11-21 |
| fastCD | 1.5.0, compiled from `vendor/fastcd` |

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 30 \
    python benchmarks/benchmark.py --protocol algorithms --rounds 9 \
    --sweeps dimension bands overlap types blocks --output algorithms-avx512.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 30-37 \
    python benchmarks/benchmark.py --protocol algorithms --rounds 9 \
    --no-toolbox --sweeps workers --output algorithms-workers.json
```

The other two runs add `--no-toolbox --simd avx2` and `--no-toolbox --simd baseline --rounds 5 --sweeps dimension bands overlap` on their builds. The host is shared, and no governor, turbo or SMT setting was changed.

Every result records the SHA-256 of the sources it measured. They are those of the commit that added this directory, with two exceptions: the docstring of `src/inazumaq2n/__init__.py` was completed after the runs, and the toolboxes protocol was added to `benchmarks/benchmark.py`, without a change to the algorithms protocol.

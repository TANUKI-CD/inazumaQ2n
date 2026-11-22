# fastCD 1.5.0, as compiled into inazumaQ2n

This directory is a snapshot of the real-coefficient kernel of
[fastCD](https://git.tanuki-cd.com/algebraity/fastCD), version 1.5.0, under
its MIT license (`LICENSE`). inazumaQ2n compiles these files into its
extension module and calls `fastcd_mul_sum_f64`, the fused sum of products,
for the hypercomplex covariance of a block.

Nothing here is edited by hand. `tools/vendor_fastcd.py --source <checkout>`
writes the snapshot and `VENDORED.json`, which records the commit, the
sources `setup.py` compiles with the options of each, and the SHA-256 of
every file; `tools/vendor_fastcd.py --check` verifies the snapshot against
that record.

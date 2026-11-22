"""The choice of ``algorithm="auto"``: moments or the fused sums of fastCD.

The package has two ways to obtain the hypercomplex covariance of a block.

``"moments"``
    The matrix of real cross moments of every tile, contracted with the signs
    of the algebra. Its cost grows with the square of the number of bands
    ``B``, and overlapping blocks share the moments of their tiles.

``"fastcd"``
    One fused sum of products of fastCD for every block. Its cost grows like
    ``N log N`` with the dimension ``N`` of the algebra, the least power of
    two that is at least ``B``, whatever ``B`` is. Blocks that overlap share
    the records and the statistics of their tiles, but every block is summed
    on its own, so a pixel enters as many sums as blocks contain it.

Which is faster therefore depends on ``N``, on how far ``B`` lies below
``N``, on how many tiles a block is merged from, and on the kernel fastCD
runs its sums with on this processor. :func:`choose` is a table of the
measurements of ``benchmarks/benchmark.py --protocol algorithms``, archived
in ``benchmark-results/fastcd-20261121``; see ``docs/benchmark.md``.
"""

from __future__ import annotations

ALGORITHMS = ("auto", "moments", "fastcd")

# _MIN_EIGHTHS[kernel][tiles][N] is the least number of eighths of the
# dimension N that the bands must fill for the fused sums to be selected,
# for a fused-sum kernel of fastCD and a number of tiles per block. An entry
# of 8 selects them for B = N only, and one of 4 for every band count of the
# dimension, since B always exceeds N / 2.
#
# An entry is the least fill at which the fused sums were measured at least
# 1.1 times as fast as the moments, on one core of a Xeon Gold 6148 with MKL
# as the matrix-product library, for float64 images and tiles of 16 or 32
# pixels a side; narrower sample types, and smaller tiles under overlap,
# were measured to favour the fused sums as much or more. A dimension below
# the first of a table is not selected, and one above the last uses the
# last. Tile counts were measured at 1 (no overlap, or blocks that share no
# tiles), 4 (a stride of half the block) and 16 (a quarter); another count
# uses the next larger one, and more than 16 tiles, like a kernel that is
# not listed, select the moments. The portable kernel of fastCD is not
# listed: on x86 it was faster than the moments at dimension 1024 only, and
# it has not been measured elsewhere.
_MIN_EIGHTHS: dict[str, dict[int, dict[int, int]]] = {
    "avx512": {
        1: {8: 5, 16: 6, 32: 6, 64: 5, 128: 5, 256: 5, 512: 5, 1024: 4},
        4: {128: 8, 256: 7, 512: 6, 1024: 5},
        16: {512: 8},
    },
    "avx2": {
        1: {8: 7, 16: 8, 32: 8, 64: 8, 128: 8, 256: 7, 512: 6, 1024: 5},
        4: {512: 8, 1024: 6},
        16: {1024: 8},
    },
}


def choose(bands: int, dimension: int, tiles_per_block: int, sum_kernel: str) -> str:
    """Return ``"moments"`` or ``"fastcd"`` for an image and a block geometry.

    Parameters
    ----------
    bands : int
        The number of bands of the image.
    dimension : int
        The dimension of the algebra.
    tiles_per_block : int
        The number of tiles whose moments a block is merged from; 1 when
        every block is its own tile.
    sum_kernel : str
        The fused-sum kernel of fastCD in use: ``"avx512"``, ``"avx2"`` or
        ``"portable"``.
    """
    tables = _MIN_EIGHTHS.get(sum_kernel, {})
    listed = [count for count in tables if count >= tiles_per_block]
    if not listed:
        return "moments"
    table = tables[min(listed)]
    if dimension < min(table):
        return "moments"
    eighths = table[min(dimension, max(table))]
    return "fastcd" if 8 * bands >= eighths * dimension else "moments"

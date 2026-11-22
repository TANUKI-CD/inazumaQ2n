"""A fast implementation of the Q2n image quality index.

Q2n compares a test image with a reference image block by block, treating a
pixel with ``B`` bands as an element of the Cayley--Dickson algebra whose
dimension is the least power of two that is at least ``B``. ``inazumaq2n``
evaluates the index from real moments of the blocks, with compiled kernels
and one matrix product per tile, or, for images with many bands, from one
fused sum of products of fastCD per block, and never forms a hypercomplex
product of pixels::

    import numpy as np
    from inazumaq2n import q2n

    rng = np.random.default_rng(0)
    reference = rng.uniform(0, 255, (256, 256, 16))
    test = reference + rng.normal(0, 10, reference.shape)
    score = q2n(reference, test)

Public API
----------
q2n
    The index of an image pair.
q2n_map
    The value of every block of an image pair.
Q2nPlan
    A reusable computation for many image pairs of one shape.
runtime_info
    The instruction set, the matrix-product library and the kernel of
    fastCD selected on this machine.
"""

from ._plan import Q2nPlan, q2n, q2n_map, runtime_info
from ._version import __version__

__all__ = ["Q2nPlan", "__version__", "q2n", "q2n_map", "runtime_info"]

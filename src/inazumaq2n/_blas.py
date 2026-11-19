"""The matrix-product library used for images with many bands.

The cross moments of a tile with ``B`` bands are one ``B`` by ``B`` matrix
product. By default it is NumPy's ``matmul``, which calls the BLAS that
NumPy was built with. When Intel MKL is installed as the ``mkl`` package,
its ``dgemm`` is called directly instead, through an entry point that takes
64-bit integers and with a thread setting local to the calling thread, so
that neither the integer interface nor the thread count of MKL is changed
for other users of the library in the same process.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import numpy as np
import numpy.typing as npt

from . import _core

BACKENDS = ("auto", "numpy", "mkl")

_lock = threading.Lock()
_mkl: Mkl | None = None
_mkl_failure: str | None = None


class Mkl:
    """Intel MKL, loaded from the installed ``mkl`` package."""

    name = "mkl"

    def __init__(self) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("the MKL backend is supported on Linux only")
        package = distribution("mkl")
        candidates = [
            package.locate_file(entry)
            for entry in package.files or ()
            if Path(str(entry)).name.startswith("libmkl_rt.so.")
        ]
        if not candidates:
            raise RuntimeError("the installed mkl package has no runtime library")
        self.path = str(Path(str(candidates[0])).resolve())
        self.library = ctypes.CDLL(self.path)
        gemm = ctypes.cast(self.library.cblas_dgemm_64, ctypes.c_void_p).value
        threads = ctypes.cast(
            self.library.MKL_Set_Num_Threads_Local, ctypes.c_void_p
        ).value
        if gemm is None or threads is None:
            raise RuntimeError("the MKL runtime library lacks a required entry point")
        self.gemm = gemm
        self.set_threads = threads
        self.version = package.version
        self._verify()

    def gram(
        self,
        xc: npt.NDArray[np.float64],
        yc: npt.NDArray[np.float64],
        grams: npt.NDArray[np.float64],
    ) -> None:
        """Write the cross moments of the centered tiles ``xc`` and ``yc``.

        ``xc`` and ``yc`` have shape ``(tiles, pixels, bands)`` and ``grams``
        has shape ``(tiles, bands, bands)``. One MKL thread is used.
        """
        _, pixels, bands = xc.shape
        _core.gram_batch(self.gemm, self.set_threads, 1, xc, yc, grams, pixels, bands)

    def _verify(self) -> None:
        """Check the entry point on a product whose result is exact."""
        x = np.arange(30, dtype=np.float64).reshape(2, 5, 3) / 8
        y = x[:, ::-1].copy() - 1
        grams = np.empty((2, 3, 3))
        self.gram(x, y, grams)
        if not np.array_equal(grams, x.transpose(0, 2, 1) @ y):
            raise RuntimeError("the 64-bit dgemm of MKL failed its check")


def resolve(backend: str) -> Mkl | None:
    """Return the MKL provider for a backend name, or None for NumPy.

    ``"auto"`` selects MKL when it is installed and passes its check, and
    NumPy otherwise. ``"mkl"`` raises ``RuntimeError`` when MKL cannot be
    used.
    """
    global _mkl, _mkl_failure
    if backend not in BACKENDS:
        raise ValueError("backend must be 'auto', 'numpy' or 'mkl'")
    if backend == "numpy":
        return None
    with _lock:
        if _mkl is None and (backend == "mkl" or _mkl_failure is None):
            try:
                _mkl = Mkl()
            except (
                PackageNotFoundError,
                OSError,
                AttributeError,
                RuntimeError,
            ) as error:
                _mkl_failure = str(error) or type(error).__name__
        if _mkl is None and backend == "mkl":
            raise RuntimeError(
                "the MKL backend is unavailable; install inazumaq2n[mkl] "
                f"or use backend='numpy' ({_mkl_failure})"
            )
        return _mkl

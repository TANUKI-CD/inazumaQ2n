"""Plans: the geometry, work arrays and dispatch of a Q2n computation."""

from __future__ import annotations

import math
import numbers
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from ._blas import resolve
from ._fallback import block_value, sign_table
from ._version import __version__

# Sample types that the kernels read in place; other inputs become float64
_IN_PLACE = tuple(np.dtype(t) for t in (np.float64, np.float32, np.uint8, np.uint16))

# Target size of the centered samples of both images that one worker holds
_WORKSPACE_BYTES = 8 << 20

# The largest image side, block size or band count
_MAX_EXTENT = 1 << 30

# The largest band count of the kernels that need no matrix product
_DIRECT_BANDS = 4


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    """Return ``value`` as an int, requiring an integer of at least ``minimum``."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class Q2nPlan:
    """A reusable Q2n computation for image pairs of one shape.

    A plan holds the block geometry, the work arrays and the worker threads
    for one image shape, so that scoring many pairs of that shape does not
    repeat their setup. :func:`q2n` and :func:`q2n_map` create a plan, use it
    once and close it.

    Parameters
    ----------
    shape : tuple of int
        ``(height, width)`` or ``(height, width, bands)``.
    ws : int, default 32
        The side of a block, at least 2.
    stride : int, optional
        The distance between the origins of adjacent blocks. The default is
        ``ws``, which gives blocks that do not overlap.
    workers : int, default 1
        The number of threads over which the tiles are divided.
    backend : {"auto", "numpy", "mkl"}, default "auto"
        The matrix-product library for images that have neither one nor
        three bands. ``"auto"`` uses Intel MKL when the ``mkl`` package is
        installed, and NumPy otherwise.

    Raises
    ------
    ValueError
        If an argument is out of range, or if the blocks do not cover the
        image with a bottom and right padding of at most its height and width.
    RuntimeError
        If ``backend="mkl"`` and MKL cannot be used.

    Notes
    -----
    Calls on one plan run one at a time; use one plan per thread to score
    pairs simultaneously. A plan is a context manager, and leaving the
    context closes it.

    Examples
    --------
    >>> import numpy as np
    >>> from inazumaq2n import Q2nPlan
    >>> image = np.arange(64 * 64 * 4, dtype=np.float64).reshape(64, 64, 4) % 251
    >>> with Q2nPlan(image.shape, ws=32) as plan:
    ...     score = plan.score(image, image)
    >>> round(score, 12)
    1.0
    """

    def __init__(
        self,
        shape: Iterable[int],
        ws: int = 32,
        stride: int | None = None,
        *,
        workers: int = 1,
        backend: str = "auto",
    ) -> None:
        shape = tuple(shape)
        if len(shape) not in (2, 3):
            raise ValueError("shape must be (height, width) or (height, width, bands)")
        self._shape = tuple(_integer(v, "every extent of shape") for v in shape)
        self._ws = _integer(ws, "ws", 2)
        self._stride = self._ws if stride is None else _integer(stride, "stride")
        self._workers = _integer(workers, "workers")
        h, w = self._shape[:2]
        self._bands = self._shape[2] if len(self._shape) == 3 else 1
        if max(h, w, self._bands, self._ws) > _MAX_EXTENT:
            raise ValueError("the image or the block is too large")
        self._dimension = 1 << (self._bands - 1).bit_length()
        nr, nc = math.ceil(h / self._stride), math.ceil(w / self._stride)
        self._block_shape = (nr, nc)
        ph = (nr - 1) * self._stride + self._ws - h
        pw = (nc - 1) * self._stride + self._ws - w
        if ph < 0 or pw < 0 or ph > h or pw > w:
            raise ValueError(
                "the blocks must cover the image with a bottom and right padding "
                "of at most its height and width"
            )

        # Tiles are stride-sized and shared between blocks when they tile a block
        reuse = self._stride < self._ws and self._ws % self._stride == 0
        self._side = self._stride if reuse else self._ws
        self._pixels = self._side * self._side
        if reuse:
            q = self._ws // self._side
            tr, tc = nr + q - 1, nc + q - 1
            first = np.arange(nr)[:, None] * tc + np.arange(nc)[None, :]
            offset = np.arange(q)[:, None] * tc + np.arange(q)[None, :]
            ids = first.reshape(-1, 1) + offset.reshape(1, -1)
        else:
            tr, tc = nr, nc
            ids = np.arange(nr * nc).reshape(-1, 1)
        if tr * tc > np.iinfo(np.int32).max or ids.size > np.iinfo(np.int32).max:
            raise ValueError("the image has too many blocks")
        origins = np.empty((tr, tc, 2), dtype=np.int32)
        origins[:, :, 0] = np.arange(tr)[:, None] * self._stride
        origins[:, :, 1] = np.arange(tc)[None, :] * self._stride
        self._origins = np.ascontiguousarray(origins.reshape(-1, 2))
        self._ids = np.ascontiguousarray(ids, dtype=np.int32)
        self._tiles = tr * tc

        self._mkl = resolve(backend)
        self._direct = self._bands <= _DIRECT_BANDS and _core.direct_is_fast()
        if self._direct:
            self._chunk = math.ceil(self._tiles / self._workers)
        else:
            fit = _WORKSPACE_BYTES // (16 * self._pixels * self._bands)
            self._chunk = max(1, min(self._tiles, fit))
            if self._workers > 1:
                self._chunk = min(self._chunk, math.ceil(self._tiles / self._workers))

        self._signs = sign_table(self._bands)
        self._grams: npt.NDArray[np.float64]
        self._stats: npt.NDArray[np.float64]
        self._values: npt.NDArray[np.float64]
        self._scratch: npt.NDArray[np.float64]
        self._grams = np.empty((self._tiles, self._bands, self._bands))
        self._stats = np.empty((self._tiles, 4, self._bands))
        self._values = np.empty(nr * nc)
        self._risk = np.empty(nr * nc, dtype=np.uint8)
        self._scratch = np.empty(6 * self._bands + self._dimension)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        if self._workers == 1 and not self._direct:
            self._buffers()

    @property
    def shape(self) -> tuple[int, ...]:
        """The image shape the plan was made for."""
        return self._shape

    @property
    def ws(self) -> int:
        """The side of a block."""
        return self._ws

    @property
    def stride(self) -> int:
        """The distance between the origins of adjacent blocks."""
        return self._stride

    @property
    def bands(self) -> int:
        """The number of bands of an image."""
        return self._bands

    @property
    def dimension(self) -> int:
        """The dimension of the algebra, the least power of two not below bands."""
        return self._dimension

    @property
    def block_shape(self) -> tuple[int, int]:
        """The number of rows and columns of blocks, the shape of a block map."""
        return self._block_shape

    @property
    def workers(self) -> int:
        """The number of threads over which the tiles are divided."""
        return self._workers

    @property
    def backend(self) -> str:
        """The matrix-product library selected: ``"numpy"`` or ``"mkl"``."""
        return "numpy" if self._mkl is None else self._mkl.name

    def __repr__(self) -> str:
        return (
            f"Q2nPlan({self._shape}, ws={self._ws}, stride={self._stride}, "
            f"workers={self._workers}, backend={self.backend!r})"
        )

    def __enter__(self) -> Q2nPlan:
        if self._closed:
            raise RuntimeError("the plan is closed")
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the worker threads and the work arrays.

        A closed plan cannot score. Closing a plan twice is allowed.
        """
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            empty = np.empty(0)
            self._grams = self._stats = self._values = self._scratch = empty
            self._local = threading.local()
            self._closed = True

    def score(self, reference: npt.ArrayLike, test: npt.ArrayLike) -> float:
        """Return the Q2n index of a test image against a reference image.

        Parameters
        ----------
        reference, test : array_like
            Real images of the shape of the plan.

        Returns
        -------
        float
            The mean of the values of the blocks.

        Raises
        ------
        ValueError
            If an image does not have the shape of the plan.
        TypeError
            If an image is not real.
        RuntimeError
            If the plan is closed.
        """
        with self._lock:
            self._run(reference, test, None)
            return float(np.mean(self._values))

    def map(
        self, reference: npt.ArrayLike, test: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Return the Q2n value of every block.

        Parameters
        ----------
        reference, test : array_like
            Real images of the shape of the plan.

        Returns
        -------
        ndarray
            A new array of shape :attr:`block_shape`. Entry ``(r, c)`` is the
            value of the block whose origin is ``(r * stride, c * stride)``,
            and the mean of the array is :meth:`score`.

        Raises
        ------
        ValueError
            If an image does not have the shape of the plan.
        TypeError
            If an image is not real.
        RuntimeError
            If the plan is closed.
        """
        with self._lock:
            self._run(reference, test, None)
            return self._values.reshape(self._block_shape).copy()

    def _buffers(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the centered-sample arrays of the calling thread."""
        buffers: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None
        buffers = getattr(self._local, "buffers", None)
        if buffers is None:
            shape = (self._chunk, self._pixels, self._bands)
            buffers = self._local.buffers = (np.empty(shape), np.empty(shape))
        return buffers

    def _images(
        self, reference: npt.ArrayLike, test: npt.ArrayLike
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Return both images as contiguous arrays of one supported sample type."""
        x, y = np.asarray(reference), np.asarray(test)
        if x.shape != self._shape or y.shape != self._shape:
            raise ValueError(
                f"both images must have the shape of the plan, {self._shape}"
            )
        if x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf":
            raise TypeError("the images must hold real numbers")
        in_place = (
            x.dtype == y.dtype
            and x.dtype in _IN_PLACE
            and x.flags.c_contiguous
            and y.flags.c_contiguous
            and x.flags.aligned
            and y.flags.aligned
        )
        if not in_place:
            x = np.ascontiguousarray(x, dtype=np.float64)
            y = np.ascontiguousarray(y, dtype=np.float64)
        if x.ndim == 2:
            x, y = x[..., None], y[..., None]
        return x, y

    def _moments(
        self, x: npt.NDArray[Any], y: npt.NDArray[Any], first: int
    ) -> tuple[float, float]:
        """Fill the moments of one chunk of tiles and return the stage times."""
        h, w = self._shape[:2]
        last = min(first + self._chunk, self._tiles)
        origins = self._origins[first:last]
        grams = self._grams[first:last]
        stats = self._stats[first:last]
        start = time.perf_counter()
        if self._direct:
            _core.direct(x, y, h, w, self._bands, origins, self._side, grams, stats)
            return time.perf_counter() - start, 0.0
        xc, yc = self._buffers()
        xc, yc = xc[: last - first], yc[: last - first]
        _core.prepare(x, y, h, w, self._bands, origins, self._side, xc, yc, stats)
        middle = time.perf_counter()
        if self._mkl is not None:
            self._mkl.gram(xc, yc, grams)
        else:
            # Products that overflow mark their block for evaluation by definition
            with np.errstate(over="ignore", invalid="ignore"):
                np.matmul(xc.transpose(0, 2, 1), yc, out=grams)
        return middle - start, time.perf_counter() - middle

    def _run(
        self,
        reference: npt.ArrayLike,
        test: npt.ArrayLike,
        profile: dict[str, Any] | None,
    ) -> None:
        """Write the value of every block of an image pair to the plan."""
        if self._closed:
            raise RuntimeError("the plan is closed")
        begin = time.perf_counter()
        x, y = self._images(reference, test)
        converted = time.perf_counter()
        firsts = range(0, self._tiles, self._chunk)
        if self._workers == 1:
            times = [self._moments(x, y, first) for first in firsts]
        else:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._workers, thread_name_prefix="inazumaq2n"
                )
            times = list(
                self._pool.map(lambda first: self._moments(x, y, first), firsts)
            )
        gathered = time.perf_counter()
        _core.finish(
            self._grams,
            self._stats,
            self._ids,
            self._ids.shape[1],
            self._pixels,
            self._bands,
            self._dimension,
            self._signs,
            self._values,
            self._scratch,
            self._risk,
        )
        finished = time.perf_counter()

        # Marked blocks are evaluated again from their normalized samples
        risky = np.flatnonzero(self._risk)
        h, w = self._shape[:2]
        for k in risky:
            r, c = divmod(int(k), self._block_shape[1])
            rows = np.arange(r * self._stride, r * self._stride + self._ws)
            cols = np.arange(c * self._stride, c * self._stride + self._ws)
            rows = np.where(rows < h, rows, 2 * h - 1 - rows)
            cols = np.where(cols < w, cols, 2 * w - 1 - cols)
            self._values[k] = block_value(
                x[rows[:, None], cols], y[rows[:, None], cols]
            )
        if profile is not None:
            end = time.perf_counter()
            profile.update(
                conversion=converted - begin,
                moments=sum(t[0] for t in times),
                matrix_products=sum(t[1] for t in times),
                moments_wall=gathered - converted,
                finish=finished - gathered,
                fallback=end - finished,
                fallback_blocks=len(risky),
                total=end - begin,
                sample_type=str(x.dtype),
                kernel="direct" if self._direct else "matrix product",
                tiles=self._tiles,
                chunk_tiles=self._chunk,
                simd=_core.simd(),
                backend=self.backend,
            )


def q2n(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    workers: int = 1,
    backend: str = "auto",
) -> float:
    """Return the Q2n index of a test image against a reference image.

    Parameters
    ----------
    reference, test : array_like
        Real images of one shape, ``(height, width)`` or
        ``(height, width, bands)``.
    ws : int, default 32
        The side of a block, at least 2.
    stride : int, optional
        The distance between the origins of adjacent blocks; ``ws`` by default.
    workers : int, default 1
        The number of threads over which the tiles are divided.
    backend : {"auto", "numpy", "mkl"}, default "auto"
        The matrix-product library for images that have neither one nor
        three bands.

    Returns
    -------
    float
        The mean of the values of the blocks. The index is not clipped: with
        more than eight bands it can exceed one.

    Raises
    ------
    ValueError
        If the images differ in shape or an argument is out of range.
    TypeError
        If an image is not real.

    See Also
    --------
    q2n_map : The value of every block.
    Q2nPlan : Reuse the setup for many pairs of one shape.

    Examples
    --------
    >>> import numpy as np
    >>> from inazumaq2n import q2n
    >>> rng = np.random.default_rng(0)
    >>> reference = rng.uniform(0, 255, (64, 64, 8))
    >>> round(q2n(reference, reference), 12)
    1.0
    """
    x = np.asarray(reference)
    with Q2nPlan(x.shape, ws, stride, workers=workers, backend=backend) as plan:
        return plan.score(x, test)


def q2n_map(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    workers: int = 1,
    backend: str = "auto",
) -> npt.NDArray[np.float64]:
    """Return the Q2n value of every block of an image pair.

    The parameters are those of :func:`q2n`.

    Returns
    -------
    ndarray
        An array of shape ``(ceil(height / stride), ceil(width / stride))``
        whose entry ``(r, c)`` is the value of the block with origin
        ``(r * stride, c * stride)``. Its mean is the value of :func:`q2n`.

    Examples
    --------
    >>> import numpy as np
    >>> from inazumaq2n import q2n_map
    >>> rng = np.random.default_rng(0)
    >>> reference = rng.uniform(0, 255, (64, 96, 3))
    >>> q2n_map(reference, reference + 1.0).shape
    (2, 3)
    """
    x = np.asarray(reference)
    with Q2nPlan(x.shape, ws, stride, workers=workers, backend=backend) as plan:
        return plan.map(x, test)


def runtime_info() -> dict[str, Any]:
    """Return what the package selected on this machine.

    Returns
    -------
    dict
        ``version`` is the version of the package. ``simd`` is the
        instruction set of the compiled kernels: ``"avx512"``, ``"avx2"``,
        ``"neon"`` or ``"baseline"``. ``direct_kernels`` tells whether images
        with one or three bands use the kernels that need no matrix product.
        ``backend`` is the library that ``backend="auto"`` selects for other
        images, ``"mkl"`` or ``"numpy"``, and ``backend_version`` is its
        version. ``numpy`` is the version of NumPy.
    """
    mkl = resolve("auto")
    return {
        "version": __version__,
        "simd": _core.simd(),
        "direct_kernels": _core.direct_is_fast(),
        "backend": "numpy" if mkl is None else mkl.name,
        "backend_version": np.__version__ if mkl is None else mkl.version,
        "numpy": np.__version__,
    }

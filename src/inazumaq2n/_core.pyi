"""Type stubs for the compiled kernels."""

from typing import Any

def prepare(
    x: Any,
    y: Any,
    height: int,
    width: int,
    bands: int,
    origins: Any,
    side: int,
    xc: Any,
    yc: Any,
    stats: Any,
    /,
) -> None: ...
def direct(
    x: Any,
    y: Any,
    height: int,
    width: int,
    bands: int,
    origins: Any,
    side: int,
    grams: Any,
    stats: Any,
    /,
) -> None: ...
def finish(
    grams: Any,
    stats: Any,
    ids: Any,
    nt: int,
    tile_pixels: int,
    bands: int,
    dimension: int,
    sigma: Any,
    values: Any,
    scratch: Any,
    risk: Any,
    /,
) -> None: ...
def fused(
    x: Any,
    y: Any,
    height: int,
    width: int,
    bands: int,
    dimension: int,
    origins: Any,
    side: int,
    work: Any,
    values: Any,
    risk: Any,
    /,
) -> None: ...
def fused_tiles(
    x: Any,
    y: Any,
    height: int,
    width: int,
    bands: int,
    dimension: int,
    stride: int,
    q: int,
    row0: int,
    rows: int,
    col0: int,
    cols: int,
    pitch: int,
    work: Any,
    values: Any,
    risk: Any,
    /,
) -> None: ...
def fused_tiles_work(tile_pixels: int, dimension: int, q: int, cols: int, /) -> int: ...
def fused_work(pixels: int, dimension: int, /) -> int: ...
def fastcd() -> tuple[str, str]: ...
def gram_batch(
    gemm: int,
    set_threads: int | None,
    threads: int,
    xc: Any,
    yc: Any,
    grams: Any,
    pixels: int,
    bands: int,
    /,
) -> None: ...
def simd() -> str: ...
def simd_levels() -> list[str]: ...
def set_simd(name: str, /) -> None: ...
def direct_is_fast() -> bool: ...

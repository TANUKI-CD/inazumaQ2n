#!/usr/bin/env python3
"""The Q2n of published pansharpening toolboxes, run as they are distributed.

inazumaq2n computes the index that the MATLAB pansharpening toolbox defines.
This module makes that statement checkable: it runs the code of the toolbox
itself, under GNU Octave, and Python ports of it, on the same images as the
package, so that the tests can assert the agreement and the benchmark can
time the three side by side.

None of that code is part of this repository. Each implementation is a
list of files in a public repository at a pinned commit, with the SHA-256
of every file:

    python benchmarks/toolboxes.py fetch      # download and verify
    python benchmarks/toolboxes.py status     # what is present and runnable
    python benchmarks/toolboxes.py report     # differences from inazumaq2n

``fetch`` writes the files under ``benchmarks/.toolboxes``, or under the
directory named by the environment variable ``INAZUMAQ2N_TOOLBOXES``, and
nothing is run that does not match its recorded hash. The files keep the
licenses of their repositories, which ``SOURCES`` records; they are fetched
for comparison and are not distributed with inazumaq2n.

``matlab``
    ``q2n.m`` and its helpers from the MATLAB toolbox, in the copy that the
    DLPan-Toolbox distributes. It is run by ``octave/run_q2n.m`` under GNU
    Octave, which lacks only ``mean2`` and ``std2`` of the Image Processing
    Toolbox when its image package is not installed; ``octave/shims``
    supplies the two.
``pancollection`` and ``hyperspectral``
    Python ports, in NumPy and in PyTorch. They are run in this process and
    need PyTorch, and the second also torchvision.

The toolbox converts both images to ``uint16`` before anything else, which
rounds every sample to the nearest integer and limits it to 0 to 65535, and
inazumaq2n does not. The two therefore agree on images whose samples are
integers in that range, and on other images the toolbox agrees with
inazumaq2n applied to :func:`matlab_uint16` of the images.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
OCTAVE = HERE / "octave"
DIRECTORY = Path(os.environ.get("INAZUMAQ2N_TOOLBOXES", HERE / ".toolboxes"))
RAW = "https://raw.githubusercontent.com"

SOURCES: dict[str, dict[str, Any]] = {
    "matlab": {
        "title": "MATLAB pansharpening toolbox, under GNU Octave",
        "repository": "liangjiandeng/DLPan-Toolbox",
        "commit": "a34f884af88967580e3ef4f61a89e12fdfe4cda5",
        "license": "GPL-3.0",
        "root": "02-Test-toolbox-for-traditional-and-DL(Matlab)/Quality_Indices",
        "files": {
            "q2n.m": "cdd94dc89d908e9ac8d72658ce07681157e5f813fd2f06cfdf2b88007b61a22c",
            "onions_quality.m": "eedfc9c6f0c9d2906fddb4c9d9f08d53aa8f6620001392f71df10ab1bb89e12f",
            "onion_mult.m": "0b93847824e7a8cc42e10a807cd5564acf7dfb53aceccb24d2368f5cc80d5019",
            "onion_mult2D.m": "1d16ac812d976a3d1d6bbcc48ca0b6b823bf58e6f298f5b4549aa55c312f1e08",
            "norm_blocco.m": "2b12a299860e492a1296f96443af4b99777e11d5e996b0b2a2db4257fb2d19bf",
        },
    },
    "pancollection": {
        "title": "PanCollection, NumPy port",
        "repository": "XiaoXiao-Woo/PanCollection",
        "commit": "4bd170a4713b1fa3a713964225c1d37ed16b24ae",
        "license": "GPL-2.0",
        "root": "pancollection/common/FS_index",
        "files": {
            "my_q2n.py": "63dda597aa04d24b6d789557f5a4c29845bb03a4dfa51c24d6d68f5f454700fc",
        },
    },
    "hyperspectral": {
        "title": "Hyperspectral pansharpening toolbox, PyTorch port",
        "repository": "matciotola/hyperspectral_pansharpening_toolbox",
        "commit": "679c33ae254a72850589ac028bbbe723c3b6f499",
        "license": "CC BY-NC-SA 4.0",
        "root": "Metrics",
        "files": {
            "metrics_rr.py": "313142b25a09dfc8cd9d94edb47bb7a5a2a70caa147cc2015934eb18bf2a28ab",
        },
    },
}


class Result(NamedTuple):
    """The index, the value of every block and the seconds of every round."""

    index: float
    map: npt.NDArray[np.float64]
    seconds: list[float]


class Unavailable(RuntimeError):
    """An implementation cannot run here: its files or its interpreter are missing."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def url(name: str, file: str) -> str:
    """Return the address of one file of an implementation at its pinned commit."""
    source = SOURCES[name]
    path = urllib.parse.quote(f"{source['root']}/{file}")
    return f"{RAW}/{source['repository']}/{source['commit']}/{path}"


def location(name: str, directory: Path | None = None) -> Path:
    """Return the directory of the files of an implementation, verified.

    Raises :class:`Unavailable` when a file is missing or does not have its
    recorded hash, so that nothing else is ever run.
    """
    folder = (DIRECTORY if directory is None else directory) / name
    for file, digest in SOURCES[name]["files"].items():
        path = folder / file
        if not path.is_file():
            raise Unavailable(
                f"{name}: {file} has not been fetched; "
                "run python benchmarks/toolboxes.py fetch"
            )
        if sha256(path) != digest:
            raise Unavailable(f"{name}: {path} does not have its recorded SHA-256")
    return folder


def fetch(names: list[str] | None = None, directory: Path | None = None) -> list[Path]:
    """Download the files of some implementations and verify their hashes."""
    written = []
    for name in SOURCES if names is None else names:
        folder = (DIRECTORY if directory is None else directory) / name
        folder.mkdir(parents=True, exist_ok=True)
        for file, digest in SOURCES[name]["files"].items():
            path = folder / file
            if not (path.is_file() and sha256(path) == digest):
                with urllib.request.urlopen(url(name, file), timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(
                        f"{url(name, file)} does not have its recorded SHA-256"
                    )
                path.write_bytes(data)
            written.append(path)
        notice = folder / "SOURCE.txt"
        source = SOURCES[name]
        notice.write_text(
            f"{source['title']}\n"
            f"https://github.com/{source['repository']} at commit {source['commit']}\n"
            f"License of the repository: {source['license']}\n"
            "Fetched by benchmarks/toolboxes.py of inazumaq2n for comparison; "
            "not part of inazumaq2n.\n",
            encoding="utf-8",
        )
    return written


def matlab_uint16(image: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return what MATLAB's ``uint16`` makes of an image, as float64.

    The conversion rounds to the nearest integer, halves away from zero,
    and limits the result to 0 to 65535; a NaN becomes zero.
    """
    values = np.asarray(image, dtype=np.float64)
    rounded = np.floor(np.where(np.isnan(values), 0.0, values) + 0.5)
    return np.clip(rounded, 0.0, 65535.0)


def octave_program() -> str | None:
    """Return the Octave interpreter on the path, or None."""
    return shutil.which("octave-cli") or shutil.which("octave")


def octave_version() -> str | None:
    """Return the version of the Octave interpreter on the path, or None."""
    program = octave_program()
    if program is None:
        return None
    done = subprocess.run([program, "--version"], capture_output=True, text=True)
    words = done.stdout.splitlines()[0].split() if done.stdout else []
    return words[-1] if words else None


def _pair(
    reference: npt.ArrayLike, test: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return both images as float64 arrays of shape (height, width, bands)."""
    x = np.array(reference, dtype=np.float64)
    y = np.array(test, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("the images must have one shape, (height, width[, bands])")
    return x, y


def matlab_q2n(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    rounds: int = 1,
    directory: Path | None = None,
) -> Result:
    """Run the ``q2n`` of the MATLAB toolbox under GNU Octave.

    The images are passed as doubles, as a MATLAB caller would pass them;
    the toolbox converts them to ``uint16`` itself. Only the call of
    ``q2n`` is timed, not the start of the interpreter or the transfer of
    the images.
    """
    toolbox = location("matlab", directory)
    program = octave_program()
    if program is None:
        raise Unavailable("matlab: GNU Octave is not on the path")
    x, y = _pair(reference, test)
    stride = ws if stride is None else stride

    def quoted(path: Path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "images.bin", Path(scratch) / "result.bin"
        with source.open("wb") as stream:
            np.array([*x.shape, ws, stride], dtype="<i8").tofile(stream)
            x.astype("<f8").ravel(order="F").tofile(stream)
            y.astype("<f8").ravel(order="F").tofile(stream)
        command = (
            f"addpath({quoted(OCTAVE)}); run_q2n({quoted(source)}, {quoted(target)}, "
            f"{quoted(toolbox)}, {quoted(OCTAVE / 'shims')}, {int(rounds)});"
        )
        done = subprocess.run(
            [program, "--quiet", "--no-init-file", "--eval", command],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0 or not target.is_file():
            raise RuntimeError(f"Octave failed: {done.stderr.strip() or done.stdout}")
        raw = target.read_bytes()
    rows, columns, count = (int(v) for v in np.frombuffer(raw, "<i8", 3))
    values = np.frombuffer(raw, "<f8", offset=24)
    blocks = values[1 : 1 + rows * columns].reshape((rows, columns), order="F")
    return Result(
        float(values[0]), blocks.copy(), values[1 + rows * columns :].tolist()
    )


def _module(name: str, file: str, directory: Path | None) -> Any:
    """Load one fetched Python file as a module of its own."""
    path = location(name, directory) / file
    spec = importlib.util.spec_from_file_location(f"inazumaq2n_toolbox_{name}", path)
    if spec is None or spec.loader is None:
        raise Unavailable(f"{name}: {path} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise Unavailable(f"{name}: {error}") from error
    return module


def _timed(call: Callable[[], Any], rounds: int) -> tuple[Any, list[float]]:
    import time

    seconds = []
    value = None
    for _ in range(rounds):
        start = time.perf_counter()
        value = call()
        seconds.append(time.perf_counter() - start)
    return value, seconds


def pancollection_q2n(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    rounds: int = 1,
    directory: Path | None = None,
) -> Result:
    """Run the ``q2n`` of PanCollection, a NumPy port of the toolbox."""
    module = _module("pancollection", "my_q2n.py", directory)
    x, y = _pair(reference, test)
    stride = ws if stride is None else stride
    (index, blocks), seconds = _timed(
        lambda: module.q2n(x.copy(), y.copy(), ws, stride), rounds
    )
    return Result(float(index), np.asarray(blocks, dtype=np.float64), seconds)


def hyperspectral_q2n(
    reference: npt.ArrayLike,
    test: npt.ArrayLike,
    ws: int = 32,
    stride: int | None = None,
    *,
    rounds: int = 1,
    directory: Path | None = None,
) -> Result:
    """Run the ``Q2n`` of the hyperspectral pansharpening toolbox, in PyTorch."""
    module = _module("hyperspectral", "metrics_rr.py", directory)
    import torch

    x, y = _pair(reference, test)
    stride = ws if stride is None else stride
    metric = module.Q2n(ws, stride)

    def call() -> Any:
        # Batches of single-precision images with the bands first, as the
        # toolbox passes them. The metric converts every block to double and
        # then normalizes it in place: a block of a double tensor is not
        # copied by that conversion, so overlapping blocks would overwrite
        # each other's samples.
        labels = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32))
        outputs = torch.from_numpy(y.transpose(2, 0, 1)[None].astype(np.float32))
        return metric(outputs, labels)

    (index, blocks), seconds = _timed(call, rounds)
    return Result(float(index), blocks[0].numpy().astype(np.float64), seconds)


ADAPTERS: dict[str, Callable[..., Result]] = {
    "matlab": matlab_q2n,
    "pancollection": pancollection_q2n,
    "hyperspectral": hyperspectral_q2n,
}


def unavailable(name: str, directory: Path | None = None) -> str | None:
    """Return why an implementation cannot run here, or None if it can."""
    try:
        location(name, directory)
    except Unavailable as error:
        return str(error)
    if name == "matlab":
        return None if octave_program() else "matlab: GNU Octave is not on the path"
    needed = ["torch"] + (["torchvision"] if name == "hyperspectral" else [])
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    return f"{name}: needs {' and '.join(missing)}" if missing else None


def report(names: list[str] | None = None, directory: Path | None = None) -> str:
    """Return a table of the differences of every runnable implementation.

    Each is run on images of integer samples, where the toolbox and
    inazumaq2n compute the same index, and compared with the block map of
    inazumaq2n. An implementation that fails on a case is reported as such.
    """
    import inazumaq2n

    rng = np.random.default_rng(20261121)
    cases = [
        (bands, shape, ws, stride)
        for bands in (1, 4, 8, 16, 3, 13)
        for shape, ws, stride in (
            ((64, 64), 32, 32),
            ((64, 64), 32, 16),
            ((70, 81), 32, 32),
        )
    ]
    lines = [
        "| implementation | bands | image | block / stride | largest difference of a block |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for name in SOURCES if names is None else names:
        reason = unavailable(name, directory)
        if reason is not None:
            lines.append(f"| {name} | | | | not run: {reason} |")
            continue
        for bands, shape, ws, stride in cases:
            x = rng.integers(0, 2048, (*shape, bands)).astype(np.float64)
            y = np.clip(x + np.round(rng.normal(0, 60, x.shape)), 0, 65535)
            expected = inazumaq2n.q2n_map(x, y, ws, stride)
            try:
                blocks = ADAPTERS[name](x, y, ws, stride, directory=directory).map
                if blocks.shape != expected.shape:
                    outcome = f"map of shape {blocks.shape}, not {expected.shape}"
                else:
                    outcome = f"{float(np.max(np.abs(blocks - expected))):.1e}"
            except Exception as error:  # a port may not support a geometry
                outcome = f"{type(error).__name__}: {error}"[:80]
            lines.append(
                f"| {name} | {bands} | {shape[0]} x {shape[1]} | {ws} / {stride} | {outcome} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "status", "report"))
    parser.add_argument("names", nargs="*", choices=[*SOURCES, []], default=[])
    parser.add_argument("--directory", type=Path, help="where the files are kept")
    args = parser.parse_args(argv)
    names = args.names or list(SOURCES)
    if args.command == "fetch":
        for path in fetch(names, args.directory):
            print(path)
    elif args.command == "status":
        for name in names:
            reason = unavailable(name, args.directory)
            source = SOURCES[name]
            print(
                f"{name}: {source['title']}, {source['repository']} at "
                f"{source['commit'][:7]}, {source['license']}: "
                + ("runnable" if reason is None else f"not runnable ({reason})")
            )
    else:
        print(report(names, args.directory), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

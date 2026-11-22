#!/usr/bin/env python3
"""Copy the fastCD sources that inazumaq2n compiles, or check the copy.

inazumaq2n builds the real-coefficient kernel of fastCD into its extension
module, from the snapshot under ``vendor/fastcd``. This tool is the only way
the snapshot is written:

    python tools/vendor_fastcd.py --source /path/to/fastCD

copies the files listed in ``FILES`` from a checkout of fastCD whose
``FASTCD_VERSION`` is the pinned ``VERSION``, and writes
``vendor/fastcd/VENDORED.json`` with the version, the commit of the checkout
the sources to compile and the SHA-256 of every file. Without ``--source``,

    python tools/vendor_fastcd.py --check

verifies that the snapshot holds exactly the recorded files with the recorded
hashes; it needs no checkout and only the standard library. ``setup.py``
reads the sources to compile, and the options of each, from the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "fastcd"
RECORD = "VENDORED.json"

# The release of fastCD that inazumaq2n is built on
VERSION = "1.5.0"

# The translation units of the real-coefficient library, and the compiler
# options that fastCD's CMakeLists.txt gives each on x86 with GCC or Clang
SOURCES = (
    "src/fastcd.c",
    "src/quasilinear.c",
    "src/cariow_cariowa.c",
    "src/direct.c",
    "src/fused_sum.c",
)
SIMD_SOURCES = {
    "src/fused_sum_avx2.c": ("-mavx2", "-mfma"),
    "src/fused_sum_avx512.c": ("-mavx512f",),
}
HEADERS = (
    "include/fastcd/fastcd.h",
    "src/balanced_sum.h",
    "src/cariow_cariowa_impl.h",
    "src/direct_impl.h",
    "src/fastcd_internal.h",
    "src/fused_sum.h",
    "src/fused_sum_impl.h",
    "src/pair.h",
    "src/quasilinear_impl.h",
    "src/quasilinear_simd.h",
    "src/generated/cariow_cariowa_kernels.h",
    "src/generated/cariow_cariowa_masks.h",
    "src/generated/direct_kernels.h",
    "src/generated/fused_sum_kernels.h",
    "src/generated/quasilinear_kernels.h",
)
FILES = ("LICENSE", *SOURCES, *SIMD_SOURCES, *HEADERS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_version(source: Path) -> str:
    """Return the FASTCD_VERSION of a checkout."""
    text = (source / "src" / "fastcd.c").read_text(encoding="utf-8")
    match = re.search(r'^#define FASTCD_VERSION "([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise ValueError("the checkout does not define FASTCD_VERSION")
    return match.group(1)


def source_commit(source: Path) -> dict[str, object]:
    """Return the commit of a checkout and whether the copied files are unmodified."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "modified_files": git("status", "--porcelain", "--", *FILES).splitlines(),
    }


def vendor(source: Path) -> None:
    """Replace the snapshot with the files of a checkout."""
    version = source_version(source)
    if version != VERSION:
        raise ValueError(f"the checkout is fastCD {version}, not the pinned {VERSION}")
    origin = source_commit(source)
    if origin["modified_files"]:
        raise ValueError(f"the checkout has modified files: {origin['modified_files']}")
    # README.md describes the snapshot and is not part of it
    for name in ("include", "src"):
        if (VENDOR / name).exists():
            shutil.rmtree(VENDOR / name)
    for name in FILES:
        target = VENDOR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    record = {
        "project": "fastCD",
        "repository": "https://git.tanuki-cd.com/algebraity/fastCD",
        "version": VERSION,
        "commit": origin["commit"],
        "sources": list(SOURCES),
        "simd_sources": {name: list(flags) for name, flags in SIMD_SOURCES.items()},
        "files_sha256": {name: sha256(VENDOR / name) for name in FILES},
    }
    (VENDOR / RECORD).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


def check() -> list[str]:
    """Return the differences between the snapshot and its record."""
    record_path = VENDOR / RECORD
    if not record_path.is_file():
        return [f"{record_path} is missing"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems = []
    if record.get("version") != VERSION:
        problems.append(f"recorded version {record.get('version')} is not {VERSION}")
    recorded: dict[str, str] = record.get("files_sha256", {})
    if sorted(recorded) != sorted(FILES):
        problems.append("the recorded files are not the files this tool lists")
    if record.get("sources") != list(SOURCES) or record.get("simd_sources") != {
        name: list(flags) for name, flags in SIMD_SOURCES.items()
    }:
        problems.append("the recorded sources are not the sources this tool lists")
    present = {
        path.relative_to(VENDOR).as_posix()
        for path in VENDOR.rglob("*")
        if path.is_file() and path.name not in (RECORD, "README.md")
    }
    problems += [f"unrecorded file: {name}" for name in sorted(present - set(recorded))]
    for name, digest in recorded.items():
        path = VENDOR / name
        if not path.is_file():
            problems.append(f"missing file: {name}")
        elif sha256(path) != digest:
            problems.append(f"checksum differs: {name}")
    if (VENDOR / "src" / "fastcd.c").is_file() and source_version(VENDOR) != VERSION:
        problems.append("the snapshot does not define the pinned FASTCD_VERSION")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="a checkout of fastCD to copy from")
    parser.add_argument("--check", action="store_true", help="verify the snapshot")
    args = parser.parse_args(argv)
    if args.source is not None:
        try:
            vendor(args.source)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"vendoring failed: {error}", file=sys.stderr)
            return 1
        print(f"vendored fastCD {VERSION} into {VENDOR}")
    elif not args.check:
        parser.error("give --source or --check")
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"vendor/fastcd matches its record: fastCD {VERSION}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

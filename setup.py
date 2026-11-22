"""Build configuration for the compiled kernels of inazumaq2n.

Package metadata lives in pyproject.toml. This file declares the extension
module ``inazumaq2n._core``, which holds the kernels that center tiles,
accumulate their moments and evaluate the index of each block. The module
is compiled for the baseline instruction set of the platform; the kernels
for wider instruction sets carry their own target attributes and are
selected when the module is imported.

The module also contains fastCD, whose fused sums of products evaluate the
hypercomplex covariance of a block for images with many bands. The sources
are the snapshot of fastCD under ``vendor/fastcd``, and they are compiled
here as fastCD's own build compiles them: every translation unit with the
default instruction set, except the AVX2 and AVX-512 kernels of the fused
sums, which are compiled with their instruction sets on x86 with GCC or
Clang and selected by fastCD at run time. ``vendor/fastcd/VENDORED.json``
lists the sources and those options.
"""

import json
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

FASTCD = Path("vendor") / "fastcd"

if sys.platform == "win32":
    compile_args = ["/O2", "/fp:precise"]
    fastcd_args = ["/O2", "/fp:precise", "/std:c11"]
    link_args = []
    libraries = []
else:
    compile_args = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fopenmp-simd"]
    # fastCD's own flags; its symbols stay private to the module, and the
    # sections let the linker drop the products the module never calls
    fastcd_args = [
        "-O3",
        "-std=c11",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fvisibility=hidden",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    link_args = ["-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"]
    libraries = ["m"]


class BuildExt(build_ext):
    """Compile fastCD into objects and link them into the extension module."""

    def build_extension(self, ext):
        if ext.name == "inazumaq2n._core":
            objects, simd = self.fastcd_objects()
            ext.extra_objects = list(ext.extra_objects or []) + objects
            if simd:
                ext.define_macros = list(ext.define_macros or []) + [
                    ("Q2N_FASTCD_SUM_SIMD", "1")
                ]
        super().build_extension(ext)

    def fastcd_objects(self):
        """Return the objects of fastCD and whether its x86 sum kernels are built."""
        record = json.loads((FASTCD / "VENDORED.json").read_text(encoding="utf-8"))
        include = [str(FASTCD / "include"), str(FASTCD / "src")]
        depends = [str(FASTCD / name) for name in record["files_sha256"]]
        # As in fastCD's CMakeLists.txt: x86 with GCC or Clang only
        simd = self.compiler.compiler_type != "msvc" and self.plat_name.endswith(
            ("x86_64", "amd64")
        )
        macros = [("FASTCD_SUM_SIMD", "1")] if simd else []

        def compile_group(names, flags):
            return self.compiler.compile(
                [str(FASTCD / name) for name in names],
                output_dir=self.build_temp,
                macros=macros,
                include_dirs=include,
                debug=self.debug,
                extra_postargs=fastcd_args + list(flags),
                depends=depends,
            )

        objects = compile_group(record["sources"], [])
        if simd:
            for name, flags in record["simd_sources"].items():
                objects += compile_group([name], flags)
        return objects, simd


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Extension(
            "inazumaq2n._core",
            sources=["src/inazumaq2n/_core.c"],
            depends=[
                "src/inazumaq2n/_isa.h",
                "src/inazumaq2n/_prepare.h",
                "src/inazumaq2n/_direct.h",
                "src/inazumaq2n/_fused.h",
                str(FASTCD / "include" / "fastcd" / "fastcd.h"),
            ],
            include_dirs=[str(FASTCD / "include")],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            libraries=libraries,
        )
    ],
)

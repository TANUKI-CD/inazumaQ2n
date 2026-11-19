"""Build configuration for the compiled kernels of inazumaq2n.

Package metadata lives in pyproject.toml. This file declares the extension
module ``inazumaq2n._core``, which holds the kernels that center tiles,
accumulate their moments and evaluate the index of each block. The module
is compiled for the baseline instruction set of the platform; the kernels
for wider instruction sets carry their own target attributes and are
selected when the module is imported.
"""

import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/O2", "/fp:precise"]
else:
    compile_args = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fopenmp-simd"]

setup(
    ext_modules=[
        Extension(
            "inazumaq2n._core",
            sources=["src/inazumaq2n/_core.c"],
            depends=[
                "src/inazumaq2n/_isa.h",
                "src/inazumaq2n/_prepare.h",
                "src/inazumaq2n/_direct.h",
            ],
            extra_compile_args=compile_args,
        )
    ]
)

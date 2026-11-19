# Security policy

inazumaq2n is a numerical library with no network access, file I/O, or
dynamic code execution. Security-relevant issues are therefore
memory-safety defects in its extension module `inazumaq2n._core`:
out-of-bounds reads or writes, integer overflow in size computations, or
reference-counting errors.

The optional MKL backend loads the runtime library of the `mkl` package
that is installed in the same environment, located through the metadata of
that package, and calls its `dgemm` directly. No other library is loaded,
and nothing is loaded when the package is absent or `backend="numpy"` is
selected.

Running time and memory access depend on the shape of the images and, for
blocks that need evaluation from their normalized samples, on their
values.

Please report issues privately to the maintainer through the contact details
on the canonical repository,
[git.tanuki-cd.com/algebraity/inazumaq2n](https://git.tanuki-cd.com/algebraity/inazumaq2n),
rather than in a public issue. Include the shape and sample type of the
images, the block size and stride, the output of
`inazumaq2n.runtime_info()`, and a minimal input that reproduces the
problem. A fix will be released as a patch version and noted in
`CHANGELOG.md`.

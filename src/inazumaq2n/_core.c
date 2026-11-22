#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <fastcd/fastcd.h>

/*
 * Compiled kernels of inazumaq2n.
 *
 * Q2n treats a pixel with B bands as an element of the Cayley--Dickson
 * algebra of dimension N, the least power of two that is at least B, and
 * needs one hypercomplex quantity for each block of an image pair: the
 * covariance sum_p (x_p - mean x)(conj(y_p) - mean conj(y)) of the
 * normalized pixels. By bilinearity that sum is a signed contraction of
 * the B by B matrix of real cross moments
 *
 *     G[i][j] = sum_p (x_p[i] - mean x[i]) (y_p[j] - mean y[j]),
 *
 * so no hypercomplex product of pixels is ever formed. The work is divided
 * as follows.
 *
 *   prepare   centers the samples of a batch of tiles into a workspace and
 *             accumulates the means and sums of squares of every band. The
 *             caller then forms G for each tile with a matrix product.
 *   direct    computes G and the same statistics for images with at most
 *             four bands in two passes over each tile, with no workspace
 *             and no matrix product.
 *   finish    merges the tiles of each block, applies the normalization of
 *             every band to G, contracts it with the Cayley--Dickson signs
 *             and evaluates the index of the block.
 *
 * The same covariance is also a sum of P hypercomplex products, one for each
 * pixel of the block, and fastCD evaluates such a sum without forming the
 * products: its fused sum accumulates the multiplications of its
 * quasilinear algorithm over the pixels and applies the additions of the
 * algorithm once. That costs O(P N log N) for a block instead of the
 * O(P B^2) of the matrix G, which is less when the image has many bands.
 *
 *   fused     writes the pixels of one block as records of N doubles with
 *             the statistics of every band, obtains the covariance of the
 *             normalized block from one fused sum of fastCD, and evaluates
 *             the index of the block.
 *   fused_tiles  does the same for blocks that overlap and are unions of
 *             stride-sized tiles: the records and the statistics of a tile
 *             are written once and shared by the blocks that contain it,
 *             and only the sum is repeated for every block.
 *
 * A tile is a block when blocks do not overlap. When the block size is a
 * multiple of the stride, tiles are stride-sized and each block is the
 * union of the tiles it covers, so every pixel is multiplied once however
 * many blocks contain it. Rows and columns beyond the bottom and right
 * edges of an image are read through the symmetric reflection, and bands
 * beyond B enter only through constants in finish, so no padded copy of an
 * image is made.
 *
 * The kernels that traverse pixels are compiled once for every instruction
 * set the build supports, from the templates in _isa.h, and the best one
 * the processor provides is selected when the module is imported. They are
 * also compiled for every supported sample type, so that images of 8- or
 * 16-bit integers and of single-precision numbers are read in place. Every
 * conversion of a sample to a double is exact.
 *
 * Arithmetic is performed in a fixed order with no value-changing
 * optimization, and the build disables floating-point contraction, so the
 * only fused operations are the explicit ones of the direct kernels.
 */

#if defined(_MSC_VER) && !defined(__clang__)
#define Q2N_MSVC 1
#define Q2N_RESTRICT __restrict
#define Q2N_INLINE static __forceinline
#define Q2N_SIMD_LOOP
#else
#define Q2N_RESTRICT restrict
#define Q2N_INLINE static inline __attribute__((always_inline))
#define Q2N_SIMD_LOOP _Pragma("omp simd")
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define Q2N_X86 1
#include <immintrin.h>
#ifdef Q2N_MSVC
#include <intrin.h>
#endif
#elif defined(__aarch64__)
#define Q2N_ARM 1
#include <arm_neon.h>
#endif

#define Q2N_ISA_BASELINE 0
#define Q2N_ISA_NEON 1
#define Q2N_ISA_AVX2 2
#define Q2N_ISA_AVX512 3
#define Q2N_ISA_COUNT 4

// The largest image side, tile side or band count the kernels accept
#define Q2N_MAX_EXTENT ((Py_ssize_t)1 << 30)

// The largest band count of the kernels that need no workspace
#define Q2N_DIRECT_BANDS 4

enum { DT_F64, DT_F32, DT_U8, DT_U16, DT_COUNT };

typedef void (*PrepareFn)(const void*, const void*, ptrdiff_t, ptrdiff_t, ptrdiff_t,
        const int32_t*, ptrdiff_t, ptrdiff_t, double*, double*, double*);
typedef void (*DirectFn)(const void*, const void*, ptrdiff_t, ptrdiff_t, ptrdiff_t,
        const int32_t*, ptrdiff_t, ptrdiff_t, double*, double*);
typedef void (*FusedFn)(const void*, const void*, ptrdiff_t, ptrdiff_t, ptrdiff_t,
        ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, double*, double*, double*);

// The pixel kernels of one instruction set, indexed by sample type
typedef struct {
    int id;
    int direct_is_fast;
    PrepareFn prepare[DT_COUNT];
    DirectFn direct[DT_COUNT];
    FusedFn fused[DT_COUNT];
} Kernels;

/*
 * The baseline kernels use only the instructions the whole module is
 * compiled for. Their kernels for at most four bands call fma, which is fast
 * only where the compiler can emit a fused multiply-add instruction for it.
 */
#define Q2N_ISA baseline
#define Q2N_ISA_ID Q2N_ISA_BASELINE
#define Q2N_TARGET
#ifdef FP_FAST_FMA
#define Q2N_DIRECT 1
#else
#define Q2N_DIRECT 0
#endif
#include "_isa.h"
#undef Q2N_ISA
#undef Q2N_ISA_ID
#undef Q2N_TARGET
#undef Q2N_DIRECT

#ifdef Q2N_X86

#define Q2N_ISA avx2
#define Q2N_ISA_ID Q2N_ISA_AVX2
#ifdef Q2N_MSVC
#define Q2N_TARGET
#else
#define Q2N_TARGET __attribute__((target("avx2,fma")))
#endif
#define Q2N_DIRECT 1
#include "_isa.h"
#undef Q2N_ISA
#undef Q2N_ISA_ID
#undef Q2N_TARGET
#undef Q2N_DIRECT

#define Q2N_ISA avx512
#define Q2N_ISA_ID Q2N_ISA_AVX512
#ifdef Q2N_MSVC
#define Q2N_TARGET
#else
#define Q2N_TARGET __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma")))
#endif
#define Q2N_DIRECT 1
#include "_isa.h"
#undef Q2N_ISA
#undef Q2N_ISA_ID
#undef Q2N_TARGET
#undef Q2N_DIRECT

#endif

#ifdef Q2N_ARM

#define Q2N_ISA neon
#define Q2N_ISA_ID Q2N_ISA_NEON
#define Q2N_TARGET
#define Q2N_DIRECT 1
#include "_isa.h"
#undef Q2N_ISA
#undef Q2N_ISA_ID
#undef Q2N_TARGET
#undef Q2N_DIRECT

#endif

static const char* const isa_names[Q2N_ISA_COUNT] = {"baseline", "neon", "avx2", "avx512"};

// The kernels compiled for each instruction set, or NULL
static const Kernels* const isa_kernels[Q2N_ISA_COUNT] = {
    &kernels_baseline,
#ifdef Q2N_ARM
    &kernels_neon,
#else
    NULL,
#endif
#ifdef Q2N_X86
    &kernels_avx2,
    &kernels_avx512,
#else
    NULL,
    NULL,
#endif
};

// The instruction sets this processor supports, and the one in use
static int isa_supported[Q2N_ISA_COUNT];
static const Kernels* active = &kernels_baseline;

// Record which of the compiled instruction sets the processor and system support
static void detect_instruction_sets(void) {
    isa_supported[Q2N_ISA_BASELINE] = 1;
#if defined(Q2N_ARM)
    isa_supported[Q2N_ISA_NEON] = 1;
#elif defined(Q2N_X86) && defined(Q2N_MSVC)
    int info[4];
    int fma, avx2, avx512;
    unsigned long long state;

    __cpuid(info, 0);
    if (info[0] < 7) return;
    __cpuid(info, 1);
    // The system must save the vector registers: OSXSAVE and AVX
    if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0) return;
    fma = (info[2] & (1 << 12)) != 0;
    state = _xgetbv(0);
    if ((state & 0x6) != 0x6) return;
    __cpuidex(info, 7, 0);
    avx2 = (info[1] & (1 << 5)) != 0;
    avx512 = (info[1] & (1 << 16)) != 0 && (info[1] & (1 << 17)) != 0
            && (info[1] & (1 << 30)) != 0 && ((unsigned)info[1] >> 31) != 0;
    isa_supported[Q2N_ISA_AVX2] = avx2 && fma;
    isa_supported[Q2N_ISA_AVX512] = avx2 && fma && avx512 && (state & 0xE0) == 0xE0;
#elif defined(Q2N_X86)
    __builtin_cpu_init();
    isa_supported[Q2N_ISA_AVX2] =
            __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    isa_supported[Q2N_ISA_AVX512] = isa_supported[Q2N_ISA_AVX2]
            && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")
            && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");
#endif
}

// 1 if the raw products of a band could lose information to underflow
static int raw_range_risky(double mx, double my, double vx, double vy, double pixels) {
    // The square root of the smallest normal double
    const double tiny_mean = ldexp(1.0, -511);

    return !isfinite(vx) || !isfinite(vy) || (vx > 0 && vx < DBL_MIN * pixels)
            || (vy > 0 && vy < DBL_MIN * pixels) || (mx != 0 && fabs(mx) < tiny_mean)
            || (my != 0 && fabs(my) < tiny_mean);
}

/*
 * Each block is the union of nt tiles of tile_pixels pixels, listed in ids.
 * The statistics of a tile are its means and centered sums of squares, and
 * its matrix G is centered at the means of the tile. Tiles are merged with
 * the identity
 *
 *     sum over the block of (x - m)(y - m')
 *         = sum over tiles of (G_t + P_t (m_t - m)(m'_t - m')),
 *
 * in which m and m' are the means of the block, so no difference of large
 * raw moments is formed. The normalization of the reference image is then
 * applied to the merged moments: band i of x is scaled by 1 / sd_i, and
 * band j of y by 1 / sd_j, or by 1 when the mean of the reference band is
 * exactly zero, with the sign of every band of y after the first reversed
 * for the conjugate. The N - B bands beyond the image are constant after
 * normalization and contribute only to the norms of the mean vectors.
 *
 * A block is marked in risk when this evaluation may be inaccurate: when a
 * band has a mean far above its standard deviation, a mean within rounding of
 * zero, so that the branch of an exactly zero mean cannot be decided here,
 * or moments that are not finite or close to underflow. The caller evaluates
 * such blocks again from their normalized samples.
 *
 * scratch holds 6 * bands + dimension doubles.
 */
// Merge tiles into blocks, normalize, contract with the signs and score every block
static void finish(const double* grams, const double* stats, const int32_t* ids,
        ptrdiff_t blocks, ptrdiff_t nt, ptrdiff_t tile_pixels, ptrdiff_t bands,
        ptrdiff_t dimension, const int8_t* sigma, double* values, double* scratch,
        uint8_t* risk) {
    double* mx = scratch;
    double* my = mx + bands;
    double* vx = my + bands;
    double* vy = vx + bands;
    double* sx = vy + bands;
    double* sy = sx + bands;
    double* cov = sy + bands;
    const double pixels = (double)(tile_pixels * nt);
    const double tile = (double)tile_pixels;
    const double count = (double)nt;
    const double n = (double)dimension;

    for (ptrdiff_t block = 0; block < blocks; block++) {
        const int32_t* tiles = ids + block * nt;
        double n2 = (double)(dimension - bands), variance = 0.0, norm = 0.0, bias;

        risk[block] = 0;
        for (ptrdiff_t b = 0; b < 4 * bands; b++) mx[b] = 0.0;
        for (ptrdiff_t t = 0; t < nt; t++) {
            const double* s = stats + (ptrdiff_t)tiles[t] * 4 * bands;

            for (ptrdiff_t b = 0; b < bands; b++) {
                mx[b] += s[b] / count;
                my[b] += s[b + bands] / count;
            }
        }
        for (ptrdiff_t t = 0; t < nt; t++) {
            const double* s = stats + (ptrdiff_t)tiles[t] * 4 * bands;

            for (ptrdiff_t b = 0; b < bands; b++) {
                const double dx = s[b] - mx[b], dy = s[b + bands] - my[b];

                vx[b] += s[b + 2 * bands] + tile * dx * dx;
                vy[b] += s[b + 3 * bands] + tile * dy * dy;
            }
        }
        for (ptrdiff_t b = 0; b < bands; b++) {
            double sd = sqrt(vx[b] / (pixels - 1.0));
            double mean;

            if (sd == 0) sd = DBL_EPSILON;
            if (fabs(mx[b]) > 1000 * sd
                    || (vx[b] > 0 && fabs(mx[b]) <= pixels * DBL_EPSILON * sd)) {
                risk[block] = 1;
            }
            if (raw_range_risky(mx[b], my[b], vx[b], vy[b], pixels)) risk[block] = 1;
            sx[b] = 1 / sd;
            sy[b] = mx[b] == 0 ? 1 : sx[b];
            mean = 1 + (my[b] - mx[b]) * sy[b];
            n2 += mean * mean;
            variance += vx[b] * sx[b] * sx[b] + vy[b] * sy[b] * sy[b];
            // The conjugate of y is folded into its scale
            if (b != 0) sy[b] = -sy[b];
        }
        bias = 2 * sqrt(n * n2) / (n + n2);
        for (ptrdiff_t k = 0; k < dimension; k++) cov[k] = 0.0;
        for (ptrdiff_t t = 0; t < nt; t++) {
            const double* g = grams + (ptrdiff_t)tiles[t] * bands * bands;
            const double* s = stats + (ptrdiff_t)tiles[t] * 4 * bands;

            for (ptrdiff_t i = 0; i < bands; i++) {
                const double dx = tile * (s[i] - mx[i]);

                for (ptrdiff_t j = 0; j < bands; j++) {
                    const double cij = g[i * bands + j] + dx * (s[j + bands] - my[j]);

                    cov[i ^ j] += (double)sigma[i * bands + j] * cij * sx[i] * sy[j];
                }
            }
        }
        for (ptrdiff_t k = 0; k < dimension; k++) norm += cov[k] * cov[k];
        values[block] = variance == 0 ? fabs(bias) : 2 * bias * sqrt(norm) / variance;
    }
}

/*
 * The evaluation of blocks by fused sums. For each block, the pixel kernel
 * writes the records of both images and the statistics of every band, and
 * the scales of the normalization follow from the statistics exactly as in
 * finish, with the same marks in risk. The covariance of the normalized
 * block is then
 *
 *     sum_p ((x_p - m) D) ((y_p - m') D'),
 *
 * one fused sum of fastCD over the records, in which the means are the
 * shifts and the scales, with the signs of the conjugate, are the scales of
 * its affine map. The coordinates beyond the bands of the image are zero in
 * every record and are shifted by zero, so they contribute nothing. A block
 * that is marked, or whose variances are all zero, needs no sum.
 *
 * work holds fused_work(pixels, dimension) doubles: the records of both
 * images, nine arrays of dimension doubles, and the workspace of the sum.
 */

// The number of doubles in the arrays of dimension doubles of the fused kernel
#define Q2N_FUSED_SMALL 9

// Write the workspace length of the fused sums, or return -1 if it is unknown
static int fused_sum_workspace(ptrdiff_t dimension, size_t* length) {
    return fastcd_mul_sum_workspace_size((size_t)dimension, length) == FASTCD_SUCCESS
            ? 0 : -1;
}

// Return the doubles of work the fused kernel needs, or -1 if they overflow
static Py_ssize_t fused_work(Py_ssize_t pixels, Py_ssize_t dimension) {
    const Py_ssize_t limit = PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(double);
    size_t sum;
    Py_ssize_t records, rest;

    if (fused_sum_workspace(dimension, &sum) != 0 || sum > (size_t)limit) return -1;
    if (pixels > limit / 2 / dimension) return -1;
    records = 2 * pixels * dimension;
    rest = Q2N_FUSED_SMALL * dimension + (Py_ssize_t)sum;
    return records > limit - rest ? -1 : records + rest;
}

// Evaluate count blocks by fused sums; returns 0, or the failing status of fastCD
static int fused(const Kernels* kernels, int type, const void* x, const void* y,
        ptrdiff_t height, ptrdiff_t width, ptrdiff_t bands, ptrdiff_t dimension,
        const int32_t* origins, ptrdiff_t count, ptrdiff_t side, double* work,
        double* values, uint8_t* risk) {
    const ptrdiff_t tile_pixels = side * side;
    const double pixels = (double)tile_pixels;
    const double n = (double)dimension;
    double* xr = work;
    double* yr = xr + tile_pixels * dimension;
    double* stats = yr + tile_pixels * dimension;
    double* mx = stats;
    double* my = mx + bands;
    double* vx = my + bands;
    double* vy = vx + bands;
    double* shift_x = stats + 4 * dimension;
    double* shift_y = shift_x + dimension;
    double* sx = shift_y + dimension;
    double* sy = sx + dimension;
    double* cov = sy + dimension;
    FastCDWorkspaceF64 workspace;
    FastCDSumMapF64 map;
    size_t length;

    if (fused_sum_workspace(dimension, &length) != 0) return (int)FASTCD_INVALID_DIMENSION;
    workspace.data = cov + dimension;
    workspace.length = length;
    map.x_shift = shift_x;
    map.x_scale = sx;
    map.y_shift = shift_y;
    map.y_scale = sy;
    // The coordinates beyond the image: zero samples, shifted by zero
    for (ptrdiff_t p = 0; p < tile_pixels; p++) {
        for (ptrdiff_t b = bands; b < dimension; b++) {
            xr[p * dimension + b] = 0.0;
            yr[p * dimension + b] = 0.0;
        }
    }
    for (ptrdiff_t b = bands; b < dimension; b++) {
        shift_x[b] = shift_y[b] = 0.0;
        sx[b] = sy[b] = 1.0;
    }

    for (ptrdiff_t block = 0; block < count; block++) {
        double n2 = (double)(dimension - bands), variance = 0.0, norm = 0.0, bias;
        FastCDStatus status;

        kernels->fused[type](x, y, height, width, bands, origins[2 * block],
                origins[2 * block + 1], side, dimension, xr, yr, stats);
        risk[block] = 0;
        for (ptrdiff_t b = 0; b < bands; b++) {
            double sd = sqrt(vx[b] / (pixels - 1.0));
            double mean;

            if (sd == 0) sd = DBL_EPSILON;
            if (fabs(mx[b]) > 1000 * sd
                    || (vx[b] > 0 && fabs(mx[b]) <= pixels * DBL_EPSILON * sd)) {
                risk[block] = 1;
            }
            if (raw_range_risky(mx[b], my[b], vx[b], vy[b], pixels)) risk[block] = 1;
            sx[b] = 1 / sd;
            sy[b] = mx[b] == 0 ? 1 : sx[b];
            mean = 1 + (my[b] - mx[b]) * sy[b];
            n2 += mean * mean;
            variance += vx[b] * sx[b] * sx[b] + vy[b] * sy[b] * sy[b];
            // The conjugate of y is folded into its scale
            if (b != 0) sy[b] = -sy[b];
            shift_x[b] = mx[b];
            shift_y[b] = my[b];
        }
        bias = 2 * sqrt(n * n2) / (n + n2);
        if (risk[block] || variance == 0) {
            // A marked block is evaluated by the caller from its samples
            values[block] = fabs(bias);
            continue;
        }
        status = fastcd_mul_sum_f64(cov, xr, (size_t)dimension, yr, (size_t)dimension,
                (size_t)tile_pixels, (size_t)dimension, &map, &workspace);
        if (status != FASTCD_SUCCESS) return (int)status;
        for (ptrdiff_t k = 0; k < dimension; k++) norm += cov[k] * cov[k];
        values[block] = 2 * bias * sqrt(norm) / variance;
    }
    return 0;
}

/*
 * Blocks that are unions of q by q tiles of `stride` pixels a side share the
 * records and the statistics of their tiles. The normalization belongs to
 * the block, so the sum of a block cannot be put together from sums of its
 * tiles; but the records of a tile are raw samples, and the means and the
 * scales of a block enter its sum only through the affine map. The kernel
 * therefore writes each tile once and repeats only the sum.
 *
 * It evaluates a rectangle of blocks, rows row0 to row0 + rows and columns
 * col0 to col0 + cols of the grid of blocks, row by row. A block row needs
 * q rows of tiles, and consecutive block rows share q - 1 of them, so the
 * kernel keeps q rows of cols + q - 1 tiles and replaces the oldest row
 * when it moves down. The records of tile row t and tile column c are
 * stored in place c * q + t mod q: the q rows of one tile column are
 * adjacent, so the q * q tiles of a block are one contiguous run of
 * records, whichever of the rows is the oldest, and a block is one fused
 * sum. The statistics of a block are merged from those of its tiles with
 * the identities, and in the order, of finish, and the same blocks are
 * marked.
 *
 * work holds fused_tiles_work(tile_pixels, dimension, q, cols) doubles.
 */

// Return the doubles of work the shared-tile kernel needs, or -1 if they overflow
static Py_ssize_t fused_tiles_work(Py_ssize_t tile_pixels, Py_ssize_t dimension,
        Py_ssize_t q, Py_ssize_t cols) {
    const Py_ssize_t limit = PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(double);
    size_t sum;
    Py_ssize_t tiles, per_tile, rest;

    if (fused_sum_workspace(dimension, &sum) != 0 || sum > (size_t)limit) return -1;
    if (cols > limit - q || cols + q - 1 > limit / q) return -1;
    tiles = (cols + q - 1) * q;
    // Two records for every pixel of a tile, and four statistics of every band
    if (tile_pixels > (limit / 2 - 2) / dimension) return -1;
    per_tile = (2 * tile_pixels + 4) * dimension;
    rest = Q2N_FUSED_SMALL * dimension + (Py_ssize_t)sum;
    if (tiles > (limit - rest) / per_tile) return -1;
    return tiles * per_tile + rest;
}

// Evaluate a rectangle of overlapping blocks by fused sums over shared tiles
static int fused_tiles(const Kernels* kernels, int type, const void* x, const void* y,
        ptrdiff_t height, ptrdiff_t width, ptrdiff_t bands, ptrdiff_t dimension,
        ptrdiff_t stride, ptrdiff_t q, ptrdiff_t row0, ptrdiff_t rows, ptrdiff_t col0,
        ptrdiff_t cols, ptrdiff_t pitch, double* work, double* values, uint8_t* risk) {
    const ptrdiff_t tile_pixels = stride * stride;
    const ptrdiff_t nt = q * q;
    const ptrdiff_t tile_columns = cols + q - 1;
    // The doubles of the records of one tile, and of its statistics
    const ptrdiff_t slot = tile_pixels * dimension;
    const ptrdiff_t entry = 4 * dimension;
    const double pixels = (double)(tile_pixels * nt);
    const double tile = (double)tile_pixels;
    const double count = (double)nt;
    const double n = (double)dimension;
    double* xr = work;
    double* yr = xr + tile_columns * q * slot;
    double* tiles = yr + tile_columns * q * slot;
    double* stats = tiles + tile_columns * q * entry;
    double* mx = stats;
    double* my = mx + bands;
    double* vx = my + bands;
    double* vy = vx + bands;
    double* shift_x = stats + 4 * dimension;
    double* shift_y = shift_x + dimension;
    double* sx = shift_y + dimension;
    double* sy = sx + dimension;
    double* cov = sy + dimension;
    FastCDWorkspaceF64 workspace;
    FastCDSumMapF64 map;
    size_t length;

    if (fused_sum_workspace(dimension, &length) != 0) return (int)FASTCD_INVALID_DIMENSION;
    workspace.data = cov + dimension;
    workspace.length = length;
    map.x_shift = shift_x;
    map.x_scale = sx;
    map.y_shift = shift_y;
    map.y_scale = sy;
    // The coordinates beyond the image: zero samples, shifted by zero
    if (bands < dimension) {
        for (ptrdiff_t p = 0; p < tile_columns * q * tile_pixels; p++) {
            for (ptrdiff_t b = bands; b < dimension; b++) {
                xr[p * dimension + b] = 0.0;
                yr[p * dimension + b] = 0.0;
            }
        }
    }
    for (ptrdiff_t b = bands; b < dimension; b++) {
        shift_x[b] = shift_y[b] = 0.0;
        sx[b] = sy[b] = 1.0;
    }

    for (ptrdiff_t br = row0; br < row0 + rows; br++) {
        // The first block row needs q rows of tiles, every other one a new row
        for (ptrdiff_t tr = br == row0 ? br : br + q - 1; tr < br + q; tr++) {
            for (ptrdiff_t tc = 0; tc < tile_columns; tc++) {
                const ptrdiff_t place = tc * q + tr % q;

                kernels->fused[type](x, y, height, width, bands, tr * stride,
                        (col0 + tc) * stride, stride, dimension, xr + place * slot,
                        yr + place * slot, tiles + place * entry);
            }
        }
        for (ptrdiff_t bc = 0; bc < cols; bc++) {
            const ptrdiff_t block = br * pitch + col0 + bc;
            double n2 = (double)(dimension - bands), variance = 0.0, norm = 0.0, bias;
            FastCDStatus status;

            for (ptrdiff_t b = 0; b < 4 * bands; b++) mx[b] = 0.0;
            for (ptrdiff_t u = 0; u < q; u++) {
                for (ptrdiff_t v = 0; v < q; v++) {
                    const double* s = tiles + ((bc + v) * q + (br + u) % q) * entry;

                    for (ptrdiff_t b = 0; b < bands; b++) {
                        mx[b] += s[b] / count;
                        my[b] += s[b + bands] / count;
                    }
                }
            }
            for (ptrdiff_t u = 0; u < q; u++) {
                for (ptrdiff_t v = 0; v < q; v++) {
                    const double* s = tiles + ((bc + v) * q + (br + u) % q) * entry;

                    for (ptrdiff_t b = 0; b < bands; b++) {
                        const double dx = s[b] - mx[b], dy = s[b + bands] - my[b];

                        vx[b] += s[b + 2 * bands] + tile * dx * dx;
                        vy[b] += s[b + 3 * bands] + tile * dy * dy;
                    }
                }
            }
            risk[block] = 0;
            for (ptrdiff_t b = 0; b < bands; b++) {
                double sd = sqrt(vx[b] / (pixels - 1.0));
                double mean;

                if (sd == 0) sd = DBL_EPSILON;
                if (fabs(mx[b]) > 1000 * sd
                        || (vx[b] > 0 && fabs(mx[b]) <= pixels * DBL_EPSILON * sd)) {
                    risk[block] = 1;
                }
                if (raw_range_risky(mx[b], my[b], vx[b], vy[b], pixels)) risk[block] = 1;
                sx[b] = 1 / sd;
                sy[b] = mx[b] == 0 ? 1 : sx[b];
                mean = 1 + (my[b] - mx[b]) * sy[b];
                n2 += mean * mean;
                variance += vx[b] * sx[b] * sx[b] + vy[b] * sy[b] * sy[b];
                // The conjugate of y is folded into its scale
                if (b != 0) sy[b] = -sy[b];
                shift_x[b] = mx[b];
                shift_y[b] = my[b];
            }
            bias = 2 * sqrt(n * n2) / (n + n2);
            if (risk[block] || variance == 0) {
                // A marked block is evaluated by the caller from its samples
                values[block] = fabs(bias);
                continue;
            }
            status = fastcd_mul_sum_f64(cov, xr + bc * q * slot, (size_t)dimension,
                    yr + bc * q * slot, (size_t)dimension, (size_t)(nt * tile_pixels),
                    (size_t)dimension, &map, &workspace);
            if (status != FASTCD_SUCCESS) return (int)status;
            for (ptrdiff_t k = 0; k < dimension; k++) norm += cov[k] * cov[k];
            values[block] = 2 * bias * sqrt(norm) / variance;
        }
    }
    return 0;
}

/*
 * The kernel of the fused sums that fastCD selects on this processor. fastCD
 * chooses among its kernels at run time and reports the choice through no
 * public function, so this follows its rule: with GCC or Clang on x86, where
 * the build compiles them, AVX-512 with avx512f and AVX2 with avx2 and fma;
 * the portable kernel otherwise. FASTCD_SUM_MAX_KERNEL is fastCD's tuning
 * macro that withholds the wider kernels from a build made for measurement.
 */
#ifndef FASTCD_SUM_MAX_KERNEL
#define FASTCD_SUM_MAX_KERNEL 3
#endif

// Return the name of the fused-sum kernel of fastCD in use
static const char* fused_sum_kernel(void) {
#if defined(Q2N_FASTCD_SUM_SIMD) && defined(Q2N_X86) && !defined(Q2N_MSVC)
    if (FASTCD_SUM_MAX_KERNEL >= 3 && __builtin_cpu_supports("avx512f")) return "avx512";
    if (FASTCD_SUM_MAX_KERNEL >= 2 && __builtin_cpu_supports("avx2")
            && __builtin_cpu_supports("fma")) {
        return "avx2";
    }
#endif
    return "portable";
}

// The signature of cblas_dgemm with 64-bit integers
typedef void (*GemmFn)(int, int, int, int64_t, int64_t, int64_t, double, const double*,
        int64_t, const double*, int64_t, double, double*, int64_t);
typedef int (*ThreadsFn)(int);

/*
 * xc and yc hold count tiles of pixels rows and bands columns. For every
 * tile, the product of the transpose of its xc block with its yc block is
 * written to grams. set_threads, when it is not NULL, sets the number of
 * threads of the library for the calling thread and returns the previous
 * setting, which is restored.
 */
// Form the cross-moment matrix of every tile with an external dgemm
static void gram_batch(GemmFn gemm, ThreadsFn set_threads, int threads, const double* xc,
        const double* yc, double* grams, ptrdiff_t count, ptrdiff_t pixels,
        ptrdiff_t bands) {
    const ptrdiff_t step = pixels * bands;
    int previous = 0;

    if (set_threads != NULL) previous = set_threads(threads);
    for (ptrdiff_t t = 0; t < count; t++) {
        // Row-major, transpose of the first operand, second operand as stored
        gemm(101, 112, 111, bands, bands, pixels, 1.0, xc + t * step, bands,
                yc + t * step, bands, 0.0, grams + t * bands * bands, bands);
    }
    if (set_threads != NULL) set_threads(previous);
}

enum { KIND_F64, KIND_I32, KIND_I8, KIND_U8 };

// Return the type character of a buffer of native single elements, or 0
static char format_code(const Py_buffer* view) {
    const char* format = view->format == NULL ? "B" : view->format;
    const int one = 1;
    const char native = *(const char*)&one == 1 ? '<' : '>';

    if (*format == '@' || *format == '=' || *format == native) format++;
    if (format[0] == '\0' || format[1] != '\0') return '\0';
    return format[0];
}

/*
 * Acquire a C-contiguous buffer and check its element type and alignment,
 * so that the kernels never reinterpret storage of another type. Returns 0,
 * or -1 with an exception set and no buffer held.
 */
// Acquire a buffer of the given element kind
static int acquire(PyObject* object, Py_buffer* view, int writable, int kind,
        const char* name) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
    Py_ssize_t size;
    char code;
    int matches;

    if (PyObject_GetBuffer(object, view, flags) != 0) return -1;
    code = format_code(view);
    if (kind == KIND_F64) {
        size = 8;
        matches = code == 'd';
    } else if (kind == KIND_I32) {
        size = 4;
        matches = code == 'i' || code == 'l';
    } else if (kind == KIND_I8) {
        size = 1;
        matches = code == 'b';
    } else {
        size = 1;
        matches = code == 'B';
    }
    if (!matches || view->itemsize != size) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s does not hold elements of the required type",
                name);
        return -1;
    }
    if (((uintptr_t)view->buf % (size_t)size) != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError, "%s is not aligned", name);
        return -1;
    }
    return 0;
}

// Acquire an image buffer and return its sample type, or -1 with an exception set
static int acquire_image(PyObject* object, Py_buffer* view, const char* name) {
    int type = -1;
    char code;

    if (PyObject_GetBuffer(object, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return -1;
    code = format_code(view);
    if (code == 'd' && view->itemsize == 8) {
        type = DT_F64;
    } else if (code == 'f' && view->itemsize == 4) {
        type = DT_F32;
    } else if (code == 'B' && view->itemsize == 1) {
        type = DT_U8;
    } else if (code == 'H' && view->itemsize == 2) {
        type = DT_U16;
    }
    if (type < 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError,
                "%s must hold float64, float32, uint8 or uint16 samples", name);
        return -1;
    }
    if (((uintptr_t)view->buf % (size_t)view->itemsize) != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError, "%s is not aligned", name);
        return -1;
    }
    return type;
}

// Release the first count buffers of an array
static void release(Py_buffer* views, int count) {
    for (int i = 0; i < count; i++) PyBuffer_Release(&views[i]);
}

// Check that a buffer holds at least a * b * c elements, each factor at least 1
static int require(const Py_buffer* view, Py_ssize_t a, Py_ssize_t b, Py_ssize_t c,
        const char* name) {
    Py_ssize_t items = view->len / view->itemsize;

    if (a < 1 || b < 1 || c < 1 || items / a / b < c) {
        PyErr_Format(PyExc_ValueError, "%s is too small", name);
        return -1;
    }
    return 0;
}

// Check the extents of an image and of a tile
static int check_geometry(Py_ssize_t height, Py_ssize_t width, Py_ssize_t bands,
        Py_ssize_t side) {
    if (height < 1 || width < 1 || bands < 1 || side < 1 || height > Q2N_MAX_EXTENT
            || width > Q2N_MAX_EXTENT || bands > Q2N_MAX_EXTENT
            || side > Q2N_MAX_EXTENT) {
        PyErr_SetString(PyExc_ValueError, "image or tile extent is out of range");
        return -1;
    }
    return 0;
}

/*
 * A tile may extend beyond the bottom and right edges by at most the height
 * and the width, where every row and column it reads is the reflection of
 * one inside the image.
 */
// Check that every tile lies within the image and its reflection
static int check_origins(const Py_buffer* origins, Py_ssize_t height, Py_ssize_t width,
        Py_ssize_t side) {
    const int32_t* data = (const int32_t*)origins->buf;
    Py_ssize_t count = origins->len / 8;

    if (origins->len % 8 != 0) {
        PyErr_SetString(PyExc_ValueError, "origins must hold pairs of int32");
        return -1;
    }
    for (Py_ssize_t t = 0; t < count; t++) {
        Py_ssize_t r = data[2 * t], c = data[2 * t + 1];

        if (r < 0 || c < 0 || r > 2 * height - side || c > 2 * width - side) {
            PyErr_SetString(PyExc_ValueError, "a tile lies outside the image");
            return -1;
        }
    }
    return 0;
}

// prepare(x, y, height, width, bands, origins, side, xc, yc, stats)
static PyObject* core_prepare(PyObject* module, PyObject* args) {
    PyObject *xo, *yo, *oo, *xco, *yco, *so;
    Py_ssize_t height, width, bands, side, count;
    Py_buffer views[6];
    const Kernels* kernels = active;
    int held = 0, type, other;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOnnnOnOOO", &xo, &yo, &height, &width, &bands, &oo,
            &side, &xco, &yco, &so)) {
        return NULL;
    }
    if (check_geometry(height, width, bands, side) != 0) return NULL;
    if ((type = acquire_image(xo, &views[0], "x")) < 0) goto fail;
    held = 1;
    if ((other = acquire_image(yo, &views[1], "y")) < 0) goto fail;
    held = 2;
    if (other != type) {
        PyErr_SetString(PyExc_TypeError, "x and y must hold samples of one type");
        goto fail;
    }
    if (acquire(oo, &views[2], 0, KIND_I32, "origins") != 0) goto fail;
    held = 3;
    if (acquire(xco, &views[3], 1, KIND_F64, "xc") != 0) goto fail;
    held = 4;
    if (acquire(yco, &views[4], 1, KIND_F64, "yc") != 0) goto fail;
    held = 5;
    if (acquire(so, &views[5], 1, KIND_F64, "stats") != 0) goto fail;
    held = 6;
    if (check_origins(&views[2], height, width, side) != 0) goto fail;
    count = views[2].len / 8;
    if (count > 0) {
        if (require(&views[0], height, width, bands, "x") != 0
                || require(&views[1], height, width, bands, "y") != 0
                || require(&views[3], count, side * side, bands, "xc") != 0
                || require(&views[4], count, side * side, bands, "yc") != 0
                || require(&views[5], count, 4, bands, "stats") != 0) {
            goto fail;
        }
        Py_BEGIN_ALLOW_THREADS
        kernels->prepare[type](views[0].buf, views[1].buf, height, width, bands,
                (const int32_t*)views[2].buf, count, side, (double*)views[3].buf,
                (double*)views[4].buf, (double*)views[5].buf);
        Py_END_ALLOW_THREADS
    }
    release(views, held);
    Py_RETURN_NONE;

fail:
    release(views, held);
    return NULL;
}

// direct(x, y, height, width, bands, origins, side, grams, stats)
static PyObject* core_direct(PyObject* module, PyObject* args) {
    PyObject *xo, *yo, *oo, *go, *so;
    Py_ssize_t height, width, bands, side, count;
    Py_buffer views[5];
    const Kernels* kernels = active;
    int held = 0, type, other;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOnnnOnOO", &xo, &yo, &height, &width, &bands, &oo,
            &side, &go, &so)) {
        return NULL;
    }
    if (check_geometry(height, width, bands, side) != 0) return NULL;
    if (bands > Q2N_DIRECT_BANDS) {
        PyErr_SetString(PyExc_ValueError, "bands must be at most 4");
        return NULL;
    }
    if ((type = acquire_image(xo, &views[0], "x")) < 0) goto fail;
    held = 1;
    if ((other = acquire_image(yo, &views[1], "y")) < 0) goto fail;
    held = 2;
    if (other != type) {
        PyErr_SetString(PyExc_TypeError, "x and y must hold samples of one type");
        goto fail;
    }
    if (acquire(oo, &views[2], 0, KIND_I32, "origins") != 0) goto fail;
    held = 3;
    if (acquire(go, &views[3], 1, KIND_F64, "grams") != 0) goto fail;
    held = 4;
    if (acquire(so, &views[4], 1, KIND_F64, "stats") != 0) goto fail;
    held = 5;
    if (check_origins(&views[2], height, width, side) != 0) goto fail;
    count = views[2].len / 8;
    if (count > 0) {
        if (require(&views[0], height, width, bands, "x") != 0
                || require(&views[1], height, width, bands, "y") != 0
                || require(&views[3], count, bands, bands, "grams") != 0
                || require(&views[4], count, 4, bands, "stats") != 0) {
            goto fail;
        }
        Py_BEGIN_ALLOW_THREADS
        kernels->direct[type](views[0].buf, views[1].buf, height, width, bands,
                (const int32_t*)views[2].buf, count, side, (double*)views[3].buf,
                (double*)views[4].buf);
        Py_END_ALLOW_THREADS
    }
    release(views, held);
    Py_RETURN_NONE;

fail:
    release(views, held);
    return NULL;
}

// finish(grams, stats, ids, nt, tile_pixels, bands, dimension, sigma, values, scratch, risk)
static PyObject* core_finish(PyObject* module, PyObject* args) {
    PyObject *go, *so, *io, *sgo, *vo, *sco, *ro;
    Py_ssize_t nt, tile_pixels, bands, dimension, blocks, tiles;
    Py_buffer views[7];
    const int32_t* ids;
    int held = 0;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOOnnnnOOOO", &go, &so, &io, &nt, &tile_pixels, &bands,
            &dimension, &sgo, &vo, &sco, &ro)) {
        return NULL;
    }
    if (nt < 1 || tile_pixels < 1 || bands < 1 || nt > Q2N_MAX_EXTENT
            || tile_pixels > Q2N_MAX_EXTENT || bands > Q2N_MAX_EXTENT
            || tile_pixels > PY_SSIZE_T_MAX / nt || tile_pixels * nt < 2) {
        PyErr_SetString(PyExc_ValueError, "block extent is out of range");
        return NULL;
    }
    if (dimension < bands || dimension > 2 * Q2N_MAX_EXTENT
            || (dimension & (dimension - 1)) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "dimension must be a power of two that is at least bands");
        return NULL;
    }
    if (acquire(go, &views[0], 0, KIND_F64, "grams") != 0) goto fail;
    held = 1;
    if (acquire(so, &views[1], 0, KIND_F64, "stats") != 0) goto fail;
    held = 2;
    if (acquire(io, &views[2], 0, KIND_I32, "ids") != 0) goto fail;
    held = 3;
    if (acquire(sgo, &views[3], 0, KIND_I8, "sigma") != 0) goto fail;
    held = 4;
    if (acquire(vo, &views[4], 1, KIND_F64, "values") != 0) goto fail;
    held = 5;
    if (acquire(sco, &views[5], 1, KIND_F64, "scratch") != 0) goto fail;
    held = 6;
    if (acquire(ro, &views[6], 1, KIND_U8, "risk") != 0) goto fail;
    held = 7;
    blocks = views[4].len / 8;
    tiles = views[1].len / 8 / 4 / bands;
    if (blocks > 0) {
        if (require(&views[0], tiles > 0 ? tiles : 1, bands, bands, "grams") != 0
                || require(&views[2], blocks, nt, 1, "ids") != 0
                || require(&views[3], bands, bands, 1, "sigma") != 0
                || require(&views[5], 6 * bands + dimension, 1, 1, "scratch") != 0
                || require(&views[6], blocks, 1, 1, "risk") != 0) {
            goto fail;
        }
        ids = (const int32_t*)views[2].buf;
        for (Py_ssize_t k = 0; k < blocks * nt; k++) {
            if (ids[k] < 0 || ids[k] >= tiles) {
                PyErr_SetString(PyExc_ValueError, "a tile index is out of range");
                goto fail;
            }
        }
        Py_BEGIN_ALLOW_THREADS
        finish((const double*)views[0].buf, (const double*)views[1].buf, ids, blocks, nt,
                tile_pixels, bands, dimension, (const int8_t*)views[3].buf,
                (double*)views[4].buf, (double*)views[5].buf, (uint8_t*)views[6].buf);
        Py_END_ALLOW_THREADS
    }
    release(views, held);
    Py_RETURN_NONE;

fail:
    release(views, held);
    return NULL;
}

// fused(x, y, height, width, bands, dimension, origins, side, work, values, risk)
static PyObject* core_fused(PyObject* module, PyObject* args) {
    PyObject *xo, *yo, *oo, *wo, *vo, *ro;
    Py_ssize_t height, width, bands, dimension, side, count, needed;
    Py_buffer views[6];
    const Kernels* kernels = active;
    int held = 0, type, other, status = 0;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOnnnnOnOOO", &xo, &yo, &height, &width, &bands,
            &dimension, &oo, &side, &wo, &vo, &ro)) {
        return NULL;
    }
    if (check_geometry(height, width, bands, side) != 0) return NULL;
    if (side < 2 || side > Q2N_MAX_EXTENT / side) {
        PyErr_SetString(PyExc_ValueError, "block extent is out of range");
        return NULL;
    }
    if (dimension < bands || dimension > 2 * Q2N_MAX_EXTENT
            || (dimension & (dimension - 1)) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "dimension must be a power of two that is at least bands");
        return NULL;
    }
    needed = fused_work(side * side, dimension);
    if (needed < 0) {
        PyErr_SetString(PyExc_ValueError, "the block is too large for a fused sum");
        return NULL;
    }
    if ((type = acquire_image(xo, &views[0], "x")) < 0) goto fail;
    held = 1;
    if ((other = acquire_image(yo, &views[1], "y")) < 0) goto fail;
    held = 2;
    if (other != type) {
        PyErr_SetString(PyExc_TypeError, "x and y must hold samples of one type");
        goto fail;
    }
    if (acquire(oo, &views[2], 0, KIND_I32, "origins") != 0) goto fail;
    held = 3;
    if (acquire(wo, &views[3], 1, KIND_F64, "work") != 0) goto fail;
    held = 4;
    if (acquire(vo, &views[4], 1, KIND_F64, "values") != 0) goto fail;
    held = 5;
    if (acquire(ro, &views[5], 1, KIND_U8, "risk") != 0) goto fail;
    held = 6;
    if (check_origins(&views[2], height, width, side) != 0) goto fail;
    count = views[2].len / 8;
    if (count > 0) {
        if (require(&views[0], height, width, bands, "x") != 0
                || require(&views[1], height, width, bands, "y") != 0
                || require(&views[3], needed, 1, 1, "work") != 0
                || require(&views[4], count, 1, 1, "values") != 0
                || require(&views[5], count, 1, 1, "risk") != 0) {
            goto fail;
        }
        Py_BEGIN_ALLOW_THREADS
        status = fused(kernels, type, views[0].buf, views[1].buf, height, width, bands,
                dimension, (const int32_t*)views[2].buf, count, side,
                (double*)views[3].buf, (double*)views[4].buf, (uint8_t*)views[5].buf);
        Py_END_ALLOW_THREADS
        if (status != 0) {
            PyErr_Format(PyExc_RuntimeError, "fastCD failed: %s",
                    fastcd_status_string((FastCDStatus)status));
            goto fail;
        }
    }
    release(views, held);
    Py_RETURN_NONE;

fail:
    release(views, held);
    return NULL;
}

// fused_tiles(x, y, height, width, bands, dimension, stride, q, row0, rows, col0,
//             cols, pitch, work, values, risk)
static PyObject* core_fused_tiles(PyObject* module, PyObject* args) {
    PyObject *xo, *yo, *wo, *vo, *ro;
    Py_ssize_t height, width, bands, dimension, stride, q, row0, rows, col0, cols, pitch;
    Py_ssize_t needed;
    Py_buffer views[5];
    const Kernels* kernels = active;
    int held = 0, type, other, status = 0;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOnnnnnnnnnnnOOO", &xo, &yo, &height, &width, &bands,
            &dimension, &stride, &q, &row0, &rows, &col0, &cols, &pitch, &wo, &vo, &ro)) {
        return NULL;
    }
    if (check_geometry(height, width, bands, stride) != 0) return NULL;
    if (q < 2 || q > Q2N_MAX_EXTENT / stride || stride > Q2N_MAX_EXTENT / stride
            || q > Q2N_MAX_EXTENT / q) {
        PyErr_SetString(PyExc_ValueError, "block extent is out of range");
        return NULL;
    }
    if (dimension < bands || dimension > 2 * Q2N_MAX_EXTENT
            || (dimension & (dimension - 1)) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "dimension must be a power of two that is at least bands");
        return NULL;
    }
    /* Every tile of the rectangle lies within the image and its reflection:
     * the last one starts q - 1 tiles after the last block. */
    if (row0 < 0 || col0 < 0 || rows < 1 || cols < 1 || pitch < 1
            || rows > Q2N_MAX_EXTENT - row0 || cols > Q2N_MAX_EXTENT - col0
            || col0 + cols > pitch
            || row0 + rows + q - 1 > (2 * height) / stride
            || col0 + cols + q - 1 > (2 * width) / stride) {
        PyErr_SetString(PyExc_ValueError, "a block lies outside the image");
        return NULL;
    }
    needed = fused_tiles_work(stride * stride, dimension, q, cols);
    if (needed < 0) {
        PyErr_SetString(PyExc_ValueError, "the blocks are too large for fused sums");
        return NULL;
    }
    if ((type = acquire_image(xo, &views[0], "x")) < 0) goto fail;
    held = 1;
    if ((other = acquire_image(yo, &views[1], "y")) < 0) goto fail;
    held = 2;
    if (other != type) {
        PyErr_SetString(PyExc_TypeError, "x and y must hold samples of one type");
        goto fail;
    }
    if (acquire(wo, &views[2], 1, KIND_F64, "work") != 0) goto fail;
    held = 3;
    if (acquire(vo, &views[3], 1, KIND_F64, "values") != 0) goto fail;
    held = 4;
    if (acquire(ro, &views[4], 1, KIND_U8, "risk") != 0) goto fail;
    held = 5;
    if (require(&views[0], height, width, bands, "x") != 0
            || require(&views[1], height, width, bands, "y") != 0
            || require(&views[2], needed, 1, 1, "work") != 0
            || require(&views[3], row0 + rows, pitch, 1, "values") != 0
            || require(&views[4], row0 + rows, pitch, 1, "risk") != 0) {
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    status = fused_tiles(kernels, type, views[0].buf, views[1].buf, height, width, bands,
            dimension, stride, q, row0, rows, col0, cols, pitch, (double*)views[2].buf,
            (double*)views[3].buf, (uint8_t*)views[4].buf);
    Py_END_ALLOW_THREADS
    if (status != 0) {
        PyErr_Format(PyExc_RuntimeError, "fastCD failed: %s",
                fastcd_status_string((FastCDStatus)status));
        goto fail;
    }
    release(views, held);
    Py_RETURN_NONE;

fail:
    release(views, held);
    return NULL;
}

// fused_tiles_work(tile_pixels, dimension, q, cols): the doubles of work of fused_tiles
static PyObject* core_fused_tiles_work(PyObject* module, PyObject* args) {
    Py_ssize_t tile_pixels, dimension, q, cols, needed;

    (void)module;
    if (!PyArg_ParseTuple(args, "nnnn", &tile_pixels, &dimension, &q, &cols)) return NULL;
    if (tile_pixels < 1 || q < 2 || cols < 1 || dimension < 1
            || dimension > 2 * Q2N_MAX_EXTENT || (dimension & (dimension - 1)) != 0
            || (needed = fused_tiles_work(tile_pixels, dimension, q, cols)) < 0) {
        PyErr_SetString(PyExc_ValueError, "the blocks are too large for fused sums");
        return NULL;
    }
    return PyLong_FromSsize_t(needed);
}

// fused_work(pixels, dimension): the doubles of work that fused needs
static PyObject* core_fused_work(PyObject* module, PyObject* args) {
    Py_ssize_t pixels, dimension, needed;

    (void)module;
    if (!PyArg_ParseTuple(args, "nn", &pixels, &dimension)) return NULL;
    if (pixels < 1 || dimension < 1 || dimension > 2 * Q2N_MAX_EXTENT
            || (dimension & (dimension - 1)) != 0
            || (needed = fused_work(pixels, dimension)) < 0) {
        PyErr_SetString(PyExc_ValueError, "the block is too large for a fused sum");
        return NULL;
    }
    return PyLong_FromSsize_t(needed);
}

// fastcd(): the version of fastCD in the module and its fused-sum kernel in use
static PyObject* core_fastcd(PyObject* module, PyObject* args) {
    (void)module;
    (void)args;
    return Py_BuildValue("(ss)", fastcd_version_string(), fused_sum_kernel());
}

// gram_batch(gemm, set_threads, threads, xc, yc, grams, pixels, bands)
static PyObject* core_gram_batch(PyObject* module, PyObject* args) {
    PyObject *fo, *to, *xco, *yco, *go;
    Py_ssize_t pixels, bands, count;
    Py_buffer views[3];
    void* gemm;
    void* set_threads = NULL;
    int threads, held = 0;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOiOOOnn", &fo, &to, &threads, &xco, &yco, &go, &pixels,
            &bands)) {
        return NULL;
    }
    gemm = PyLong_AsVoidPtr(fo);
    if (gemm == NULL) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "gemm is a null pointer");
        return NULL;
    }
    if (to != Py_None) {
        set_threads = PyLong_AsVoidPtr(to);
        if (set_threads == NULL && PyErr_Occurred()) return NULL;
    }
    if (pixels < 1 || bands < 1 || pixels > Q2N_MAX_EXTENT || bands > Q2N_MAX_EXTENT) {
        PyErr_SetString(PyExc_ValueError, "tile extent is out of range");
        return NULL;
    }
    if (acquire(xco, &views[0], 0, KIND_F64, "xc") != 0) goto fail;
    held = 1;
    if (acquire(yco, &views[1], 0, KIND_F64, "yc") != 0) goto fail;
    held = 2;
    if (acquire(go, &views[2], 1, KIND_F64, "grams") != 0) goto fail;
    held = 3;
    count = views[2].len / 8 / bands / bands;
    if (count > 0) {
        if (require(&views[0], count, pixels, bands, "xc") != 0
                || require(&views[1], count, pixels, bands, "yc") != 0) {
            goto fail;
        }
        Py_BEGIN_ALLOW_THREADS
        // A data pointer is converted to a function pointer through an integer
        gram_batch((GemmFn)(uintptr_t)gemm, (ThreadsFn)(uintptr_t)set_threads, threads,
                (const double*)views[0].buf, (const double*)views[1].buf,
                (double*)views[2].buf, count, pixels, bands);
        Py_END_ALLOW_THREADS
    }
    release(views, held);
    Py_RETURN_NONE;

fail:
    release(views, held);
    return NULL;
}

// simd(): the name of the instruction set in use
static PyObject* core_simd(PyObject* module, PyObject* args) {
    (void)module;
    (void)args;
    return PyUnicode_FromString(isa_names[active->id]);
}

// simd_levels(): the names of the instruction sets this processor supports
static PyObject* core_simd_levels(PyObject* module, PyObject* args) {
    PyObject* result = PyList_New(0);

    (void)module;
    (void)args;
    if (result == NULL) return NULL;
    for (int id = 0; id < Q2N_ISA_COUNT; id++) {
        PyObject* name;

        if (!isa_supported[id] || isa_kernels[id] == NULL) continue;
        name = PyUnicode_FromString(isa_names[id]);
        if (name == NULL || PyList_Append(result, name) != 0) {
            Py_XDECREF(name);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(name);
    }
    return result;
}

// set_simd(name): select one of the supported instruction sets
static PyObject* core_set_simd(PyObject* module, PyObject* args) {
    const char* name;

    (void)module;
    if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
    for (int id = 0; id < Q2N_ISA_COUNT; id++) {
        if (strcmp(name, isa_names[id]) != 0) continue;
        if (!isa_supported[id] || isa_kernels[id] == NULL) break;
        active = isa_kernels[id];
        Py_RETURN_NONE;
    }
    PyErr_Format(PyExc_ValueError, "instruction set %s is not available", name);
    return NULL;
}

// direct_is_fast(): whether the kernels for at most four bands should be used
static PyObject* core_direct_is_fast(PyObject* module, PyObject* args) {
    (void)module;
    (void)args;
    return PyBool_FromLong(active->direct_is_fast);
}

static PyMethodDef core_methods[] = {
    {"prepare", core_prepare, METH_VARARGS,
            "prepare(x, y, height, width, bands, origins, side, xc, yc, stats)\n\n"
            "Center the tiles at origins into xc and yc and write the means and\n"
            "the centered sums of squares of every band to stats."},
    {"direct", core_direct, METH_VARARGS,
            "direct(x, y, height, width, bands, origins, side, grams, stats)\n\n"
            "Write the centered cross moments and the statistics of the tiles at\n"
            "origins, for images with at most four bands."},
    {"finish", core_finish, METH_VARARGS,
            "finish(grams, stats, ids, nt, tile_pixels, bands, dimension, sigma,\n"
            "       values, scratch, risk)\n\n"
            "Merge the tiles of every block and write its Q2n value, marking in\n"
            "risk the blocks that need evaluation from their normalized samples."},
    {"fused", core_fused, METH_VARARGS,
            "fused(x, y, height, width, bands, dimension, origins, side, work,\n"
            "      values, risk)\n\n"
            "Write the Q2n value of the blocks at origins, each from one fused\n"
            "sum of fastCD, marking in risk the blocks that need evaluation from\n"
            "their normalized samples."},
    {"fused_work", core_fused_work, METH_VARARGS,
            "fused_work(pixels, dimension)\n\n"
            "Return the number of doubles of work that fused needs for blocks\n"
            "of pixels pixels in the algebra of the given dimension."},
    {"fused_tiles", core_fused_tiles, METH_VARARGS,
            "fused_tiles(x, y, height, width, bands, dimension, stride, q, row0,\n"
            "            rows, col0, cols, pitch, work, values, risk)\n\n"
            "Write the Q2n value of the blocks of q by q tiles in rows row0 to\n"
            "row0 + rows and columns col0 to col0 + cols of a grid with pitch\n"
            "blocks to a row, sharing the records of the tiles between blocks."},
    {"fused_tiles_work", core_fused_tiles_work, METH_VARARGS,
            "fused_tiles_work(tile_pixels, dimension, q, cols)\n\n"
            "Return the number of doubles of work that fused_tiles needs for\n"
            "cols columns of blocks of q by q tiles."},
    {"fastcd", core_fastcd, METH_NOARGS,
            "fastcd()\n\n"
            "Return the version of fastCD compiled into the module and the name\n"
            "of its fused-sum kernel in use: avx512, avx2 or portable."},
    {"gram_batch", core_gram_batch, METH_VARARGS,
            "gram_batch(gemm, set_threads, threads, xc, yc, grams, pixels, bands)\n\n"
            "Form the cross moments of centered tiles with the dgemm at the\n"
            "address gemm, which takes 64-bit integers."},
    {"simd", core_simd, METH_NOARGS,
            "simd()\n\nReturn the name of the instruction set in use."},
    {"simd_levels", core_simd_levels, METH_NOARGS,
            "simd_levels()\n\n"
            "Return the names of the instruction sets this processor supports."},
    {"set_simd", core_set_simd, METH_VARARGS,
            "set_simd(name)\n\nSelect one of the supported instruction sets."},
    {"direct_is_fast", core_direct_is_fast, METH_NOARGS,
            "direct_is_fast()\n\n"
            "Return whether the kernels for at most four bands have a fused\n"
            "multiply-add instruction in the instruction set in use."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT,
    "inazumaq2n._core",
    "Compiled kernels of inazumaq2n.",
    -1,
    core_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__core(void) {
    detect_instruction_sets();
    for (int id = 0; id < Q2N_ISA_COUNT; id++) {
        if (isa_supported[id] && isa_kernels[id] != NULL) active = isa_kernels[id];
    }
    return PyModule_Create(&core_module);
}

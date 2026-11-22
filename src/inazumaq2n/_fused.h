/*
 * The records and the statistics of one block, for the evaluation of its
 * covariance by a fused sum of products. _isa.h includes this file once for
 * every instruction set without a sample type, for the part that reads only
 * doubles, and then once for every sample type Q2N_T.
 *
 * A fused sum reads the pixels of a block as records: `dimension` doubles
 * for each pixel, the bands of the image first and zeros in the coordinates
 * beyond them. The first pass over the block writes the records of both
 * images, in the order of the pixels, and adds up every band; the second
 * pass, over the records, accumulates the centered sums of squares. The
 * statistics are stored as mean x, mean y, sum of squares x, sum of squares
 * y, each of length bands. Rows and columns beyond the image are read
 * through the symmetric reflection. The coordinates of the records beyond
 * bands are not written: the caller clears them once.
 *
 * Both passes keep the sums of 32 bands at a time in four eight-lane
 * vectors while they traverse a run of pixels, and write them back at the
 * end of the run. An accumulator held in memory would be loaded and stored
 * once for every pixel, and each load would wait for the store before it; in
 * a register the sums of a run cost one addition per pixel and vector. A
 * run is as many pixels as keep its samples and records in the first-level
 * cache, so that the traversals of a run, one for every 32 bands, read them
 * from there. The bands beyond the last multiple of eight are covered by
 * one more vector over the last eight bands, of which only the lanes of
 * those bands are kept; an image with fewer than eight bands is summed
 * pixel by pixel.
 *
 * Every band is still summed over the pixels in their order, one addition
 * and one rounding at a time, whatever the length of a run and the width of
 * a vector. The statistics are therefore bit-identical between instruction
 * sets, and identical to those of _prepare.h for the same block.
 */

#ifndef Q2N_T

// The pixels of a run: 16 KiB of records
#define Q2N_FUSED_RUN(dimension) ((dimension) < 2048 ? 2048 / (dimension) : 1)

// Add the squares of the centered records of one image to v
static Q2N_TARGET void ISA(fused_squares)(const double* Q2N_RESTRICT records,
        ptrdiff_t pixels, ptrdiff_t bands, ptrdiff_t dimension,
        const double* Q2N_RESTRICT mean, double* Q2N_RESTRICT v) {
    const ptrdiff_t run = Q2N_FUSED_RUN(dimension);

    for (ptrdiff_t first = 0; first < pixels; first += run) {
        const ptrdiff_t count = pixels - first < run ? pixels - first : run;
        const double* Q2N_RESTRICT base = records + first * dimension;
        ptrdiff_t k = 0;

        for (; k + 32 <= bands; k += 32) {
            const V8 m0 = V8_LOAD_F64(mean + k), m1 = V8_LOAD_F64(mean + k + 8);
            const V8 m2 = V8_LOAD_F64(mean + k + 16), m3 = V8_LOAD_F64(mean + k + 24);
            V8 s0 = V8_LOAD_F64(v + k), s1 = V8_LOAD_F64(v + k + 8);
            V8 s2 = V8_LOAD_F64(v + k + 16), s3 = V8_LOAD_F64(v + k + 24);

            for (ptrdiff_t p = 0; p < count; p++) {
                const double* o = base + p * dimension + k;
                const V8 a0 = V8_SUB(V8_LOAD_F64(o), m0);
                const V8 a1 = V8_SUB(V8_LOAD_F64(o + 8), m1);
                const V8 a2 = V8_SUB(V8_LOAD_F64(o + 16), m2);
                const V8 a3 = V8_SUB(V8_LOAD_F64(o + 24), m3);

                s0 = V8_ADD(s0, V8_MUL(a0, a0));
                s1 = V8_ADD(s1, V8_MUL(a1, a1));
                s2 = V8_ADD(s2, V8_MUL(a2, a2));
                s3 = V8_ADD(s3, V8_MUL(a3, a3));
            }
            V8_STORE(v + k, s0);
            V8_STORE(v + k + 8, s1);
            V8_STORE(v + k + 16, s2);
            V8_STORE(v + k + 24, s3);
        }
        for (; k + 8 <= bands; k += 8) {
            const V8 m0 = V8_LOAD_F64(mean + k);
            V8 s0 = V8_LOAD_F64(v + k);

            for (ptrdiff_t p = 0; p < count; p++) {
                const V8 a0 = V8_SUB(V8_LOAD_F64(base + p * dimension + k), m0);

                s0 = V8_ADD(s0, V8_MUL(a0, a0));
            }
            V8_STORE(v + k, s0);
        }
        if (k < bands && bands >= 8) {
            // The last eight bands; the lanes before band k are discarded
            const ptrdiff_t last = bands - 8;
            const V8 m0 = V8_LOAD_F64(mean + last);
            V8 s0 = V8_LOAD_F64(v + last);
            double lanes[8];

            for (ptrdiff_t p = 0; p < count; p++) {
                const V8 a0 = V8_SUB(V8_LOAD_F64(base + p * dimension + last), m0);

                s0 = V8_ADD(s0, V8_MUL(a0, a0));
            }
            V8_STORE(lanes, s0);
            for (ptrdiff_t b = k; b < bands; b++) v[b] = lanes[b - last];
        } else if (k < bands) {
            for (ptrdiff_t p = 0; p < count; p++) {
                const double* Q2N_RESTRICT o = base + p * dimension;

                Q2N_SIMD_LOOP
                for (ptrdiff_t b = k; b < bands; b++) {
                    const double a = o[b] - mean[b];

                    v[b] += a * a;
                }
            }
        }
    }
}

#else

/*
 * The pixels of a run are count samples of every band, `step` samples
 * apart: bands for a run read forwards, minus bands for one read backwards
 * through the reflection of the columns.
 */
// Write the records of a run of pixels of one image and add its samples to sums
static Q2N_TARGET void KNAME(fused_records)(const Q2N_T* Q2N_RESTRICT px, ptrdiff_t step,
        ptrdiff_t count, ptrdiff_t bands, ptrdiff_t dimension, double* Q2N_RESTRICT out,
        double* Q2N_RESTRICT sums) {
    ptrdiff_t k = 0;

    for (; k + 32 <= bands; k += 32) {
        V8 s0 = V8_LOAD_F64(sums + k), s1 = V8_LOAD_F64(sums + k + 8);
        V8 s2 = V8_LOAD_F64(sums + k + 16), s3 = V8_LOAD_F64(sums + k + 24);

        for (ptrdiff_t p = 0; p < count; p++) {
            const Q2N_T* i = px + p * step + k;
            double* o = out + p * dimension + k;
            const V8 a0 = V8_LOAD(i), a1 = V8_LOAD(i + 8);
            const V8 a2 = V8_LOAD(i + 16), a3 = V8_LOAD(i + 24);

            V8_STORE(o, a0);
            V8_STORE(o + 8, a1);
            V8_STORE(o + 16, a2);
            V8_STORE(o + 24, a3);
            s0 = V8_ADD(s0, a0);
            s1 = V8_ADD(s1, a1);
            s2 = V8_ADD(s2, a2);
            s3 = V8_ADD(s3, a3);
        }
        V8_STORE(sums + k, s0);
        V8_STORE(sums + k + 8, s1);
        V8_STORE(sums + k + 16, s2);
        V8_STORE(sums + k + 24, s3);
    }
    for (; k + 8 <= bands; k += 8) {
        V8 s0 = V8_LOAD_F64(sums + k);

        for (ptrdiff_t p = 0; p < count; p++) {
            const V8 a0 = V8_LOAD(px + p * step + k);

            V8_STORE(out + p * dimension + k, a0);
            s0 = V8_ADD(s0, a0);
        }
        V8_STORE(sums + k, s0);
    }
    if (k < bands && bands >= 8) {
        // The last eight bands; the lanes before band k are discarded
        const ptrdiff_t last = bands - 8;
        V8 s0 = V8_LOAD_F64(sums + last);
        double lanes[8];

        for (ptrdiff_t p = 0; p < count; p++) {
            const V8 a0 = V8_LOAD(px + p * step + last);

            V8_STORE(out + p * dimension + last, a0);
            s0 = V8_ADD(s0, a0);
        }
        V8_STORE(lanes, s0);
        for (ptrdiff_t b = k; b < bands; b++) sums[b] = lanes[b - last];
    } else if (k < bands) {
        for (ptrdiff_t p = 0; p < count; p++) {
            const Q2N_T* Q2N_RESTRICT i = px + p * step;
            double* Q2N_RESTRICT o = out + p * dimension;

            Q2N_SIMD_LOOP
            for (ptrdiff_t b = k; b < bands; b++) {
                const double a = (double)i[b];

                o[b] = a;
                sums[b] += a;
            }
        }
    }
}

// Write the records of one block of an image pair and its band statistics
static Q2N_TARGET void KNAME(fused)(const void* xv, const void* yv, ptrdiff_t height,
        ptrdiff_t width, ptrdiff_t bands, ptrdiff_t r, ptrdiff_t c, ptrdiff_t side,
        ptrdiff_t dimension, double* Q2N_RESTRICT xr, double* Q2N_RESTRICT yr,
        double* Q2N_RESTRICT stats) {
    const Q2N_T* x = (const Q2N_T*)xv;
    const Q2N_T* y = (const Q2N_T*)yv;
    const ptrdiff_t pixels = side * side;
    const ptrdiff_t run = Q2N_FUSED_RUN(dimension);
    // The columns of a row inside the image, then those of its reflection
    const ptrdiff_t inside = c >= width ? 0 : side < width - c ? side : width - c;
    double* mx = stats;
    double* my = mx + bands;
    double* vx = my + bands;
    double* vy = vx + bands;

    // The four statistics of the block are adjacent
    for (ptrdiff_t b = 0; b < 4 * bands; b++) mx[b] = 0.0;
    for (ptrdiff_t u = 0; u < side; u++) {
        const ptrdiff_t rr = r + u < height ? r + u : 2 * height - 1 - r - u;

        for (ptrdiff_t v = 0; v < side;) {
            const ptrdiff_t part = v < inside ? inside - v : side - v;
            const ptrdiff_t count = part < run ? part : run;
            const ptrdiff_t cc = v < inside ? c + v : 2 * width - 1 - c - v;
            const ptrdiff_t step = v < inside ? bands : -bands;
            const ptrdiff_t first = (rr * width + cc) * bands;
            const ptrdiff_t record = (u * side + v) * dimension;

            KNAME(fused_records)(x + first, step, count, bands, dimension, xr + record, mx);
            KNAME(fused_records)(y + first, step, count, bands, dimension, yr + record, my);
            v += count;
        }
    }
    for (ptrdiff_t b = 0; b < bands; b++) {
        mx[b] /= (double)pixels;
        my[b] /= (double)pixels;
    }
    ISA(fused_squares)(xr, pixels, bands, dimension, mx, vx);
    ISA(fused_squares)(yr, pixels, bands, dimension, my, vy);
}

#endif

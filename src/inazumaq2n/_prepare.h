/*
 * Centering of tiles with any number of bands. This file is included by
 * _isa.h once for every sample type Q2N_T and every instruction set.
 *
 * For each tile, the first pass over its pixels gives the mean of every
 * band of both images, and the second writes the centered samples to xc and
 * yc, pixel by pixel with the bands adjacent, and accumulates the centered
 * sums of squares. The statistics of a tile are stored as mean x, mean y,
 * sum of squares x, sum of squares y, each of length bands. Rows and columns
 * beyond the image are read through the symmetric reflection.
 *
 * The loops over bands have no dependence between iterations, so the
 * instruction set changes how many bands are handled at once and nothing
 * else: the results are bit-identical between instruction sets.
 */

// Center count tiles of an image pair and accumulate their band statistics
static Q2N_TARGET void KNAME(prepare)(const void* xv, const void* yv, ptrdiff_t height,
        ptrdiff_t width, ptrdiff_t bands, const int32_t* origins, ptrdiff_t count,
        ptrdiff_t side, double* Q2N_RESTRICT xc, double* Q2N_RESTRICT yc,
        double* Q2N_RESTRICT stats) {
    const Q2N_T* Q2N_RESTRICT x = (const Q2N_T*)xv;
    const Q2N_T* Q2N_RESTRICT y = (const Q2N_T*)yv;
    const ptrdiff_t pixels = side * side;

    for (ptrdiff_t t = 0; t < count; t++) {
        double* mx = stats + t * 4 * bands;
        double* my = mx + bands;
        double* vx = my + bands;
        double* vy = vx + bands;
        const ptrdiff_t r = origins[2 * t], c = origins[2 * t + 1];

        // The four statistics of the tile are adjacent
        for (ptrdiff_t b = 0; b < 4 * bands; b++) mx[b] = 0.0;
        for (ptrdiff_t u = 0; u < side; u++) {
            const ptrdiff_t rr = r + u < height ? r + u : 2 * height - 1 - r - u;

            for (ptrdiff_t v = 0; v < side; v++) {
                const ptrdiff_t cc = c + v < width ? c + v : 2 * width - 1 - c - v;
                const Q2N_T* Q2N_RESTRICT px = x + (rr * width + cc) * bands;
                const Q2N_T* Q2N_RESTRICT py = y + (rr * width + cc) * bands;

                Q2N_SIMD_LOOP
                for (ptrdiff_t b = 0; b < bands; b++) {
                    mx[b] += (double)px[b];
                    my[b] += (double)py[b];
                }
            }
        }
        for (ptrdiff_t b = 0; b < bands; b++) {
            mx[b] /= (double)pixels;
            my[b] /= (double)pixels;
        }
        for (ptrdiff_t u = 0; u < side; u++) {
            const ptrdiff_t rr = r + u < height ? r + u : 2 * height - 1 - r - u;

            for (ptrdiff_t v = 0; v < side; v++) {
                const ptrdiff_t cc = c + v < width ? c + v : 2 * width - 1 - c - v;
                const Q2N_T* Q2N_RESTRICT px = x + (rr * width + cc) * bands;
                const Q2N_T* Q2N_RESTRICT py = y + (rr * width + cc) * bands;
                double* Q2N_RESTRICT ox = xc + (t * pixels + u * side + v) * bands;
                double* Q2N_RESTRICT oy = yc + (t * pixels + u * side + v) * bands;

                Q2N_SIMD_LOOP
                for (ptrdiff_t b = 0; b < bands; b++) {
                    const double a = (double)px[b] - mx[b];
                    const double d = (double)py[b] - my[b];

                    ox[b] = a;
                    oy[b] = d;
                    vx[b] += a * a;
                    vy[b] += d * d;
                }
            }
        }
    }
}

/*
 * The kernels of one instruction set. _core.c includes this file once for
 * every instruction set it can dispatch to, with
 *
 *   Q2N_ISA      the name of the instruction set, used as a suffix;
 *   Q2N_ISA_ID   one of the Q2N_ISA_* constants of _core.c;
 *   Q2N_TARGET   the function attribute that enables its instructions;
 *   Q2N_DIRECT   1 if the kernels for at most four bands are fast here.
 *
 * The file defines an eight-lane vector of doubles, V8, and then includes
 * _prepare.h and the generated _direct.h for every sample type. V8 is one
 * 512-bit register with AVX-512, two 256-bit registers with AVX2, four
 * 128-bit registers with NEON, and eight doubles otherwise. Every operation
 * acts on each lane separately and V8_FMADD is a fused multiply-add in all
 * four, so a kernel gives bit-identical results with every instruction set.
 */

#define Q2N_PASTE_(a, b) a##_##b
#define Q2N_PASTE(a, b) Q2N_PASTE_(a, b)
#define ISA(name) Q2N_PASTE(name, Q2N_ISA)

#define V8 ISA(v8)
#define V8_ZERO ISA(v8_zero)
#define V8_SET1 ISA(v8_set1)
#define V8_ADD ISA(v8_add)
#define V8_SUB ISA(v8_sub)
#define V8_FMADD ISA(v8_fmadd)
#define V8_REVERSE ISA(v8_reverse)
#define V8_STORE ISA(v8_store)
#define V8_REDUCE ISA(v8_reduce)
#define V8_DEINTERLEAVE2 ISA(v8_deinterleave2)
#define V8_DEINTERLEAVE3 ISA(v8_deinterleave3)
#define V8_DEINTERLEAVE4 ISA(v8_deinterleave4)
#define V8_LOAD_F64 ISA(v8_load_f64)
#define V8_LOAD_F32 ISA(v8_load_f32)
#define V8_LOAD_U8 ISA(v8_load_u8)
#define V8_LOAD_U16 ISA(v8_load_u16)

#if Q2N_ISA_ID == Q2N_ISA_AVX512

typedef __m512d V8;

Q2N_INLINE Q2N_TARGET V8 V8_ZERO(void) {
    return _mm512_setzero_pd();
}

Q2N_INLINE Q2N_TARGET V8 V8_SET1(double value) {
    return _mm512_set1_pd(value);
}

Q2N_INLINE Q2N_TARGET V8 V8_ADD(V8 a, V8 b) {
    return _mm512_add_pd(a, b);
}

Q2N_INLINE Q2N_TARGET V8 V8_SUB(V8 a, V8 b) {
    return _mm512_sub_pd(a, b);
}

// a * b + c with one rounding
Q2N_INLINE Q2N_TARGET V8 V8_FMADD(V8 a, V8 b, V8 c) {
    return _mm512_fmadd_pd(a, b, c);
}

Q2N_INLINE Q2N_TARGET V8 V8_REVERSE(V8 a) {
    return _mm512_permutexvar_pd(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), a);
}

Q2N_INLINE Q2N_TARGET void V8_STORE(double* out, V8 a) {
    _mm512_storeu_pd(out, a);
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F64(const double* p) {
    return _mm512_loadu_pd(p);
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F32(const float* p) {
    return _mm512_cvtps_pd(_mm256_loadu_ps(p));
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U8(const uint8_t* p) {
    return _mm512_cvtepi32_pd(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U16(const uint16_t* p) {
    return _mm512_cvtepi32_pd(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)));
}

// Split 24 interleaved values x, y, z into the values at 3k, 3k + 1 and 3k + 2
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE3(V8 x, V8 y, V8 z, V8* a, V8* b, V8* d) {
    const __m512i i0 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
    const __m512i i1 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    const __m512i i2 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    const __m512i j0 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 13);
    const __m512i j1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, 11, 14);
    const __m512i j2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, 12, 15);

    *a = _mm512_permutex2var_pd(_mm512_permutex2var_pd(x, i0, y), j0, z);
    *b = _mm512_permutex2var_pd(_mm512_permutex2var_pd(x, i1, y), j1, z);
    *d = _mm512_permutex2var_pd(_mm512_permutex2var_pd(x, i2, y), j2, z);
}

// Split 16 interleaved values x, y into the values at 2k and 2k + 1
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE2(V8 x, V8 y, V8* a, V8* b) {
    *a = _mm512_permutex2var_pd(x, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), y);
    *b = _mm512_permutex2var_pd(x, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), y);
}

// Split 32 interleaved values into the values at 4k, 4k + 1, 4k + 2 and 4k + 3
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE4(V8 x0, V8 x1, V8 x2, V8 x3, V8* a, V8* b,
        V8* c, V8* d) {
    const __m512i even = _mm512_setr_epi64(0, 4, 8, 12, 1, 5, 9, 13);
    const __m512i odd = _mm512_setr_epi64(2, 6, 10, 14, 3, 7, 11, 15);
    const __m512i low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    const __m512i high = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
    const __m512d t01 = _mm512_permutex2var_pd(x0, even, x1);
    const __m512d u01 = _mm512_permutex2var_pd(x0, odd, x1);
    const __m512d t23 = _mm512_permutex2var_pd(x2, even, x3);
    const __m512d u23 = _mm512_permutex2var_pd(x2, odd, x3);

    *a = _mm512_permutex2var_pd(t01, low, t23);
    *b = _mm512_permutex2var_pd(t01, high, t23);
    *c = _mm512_permutex2var_pd(u01, low, u23);
    *d = _mm512_permutex2var_pd(u01, high, u23);
}

#elif Q2N_ISA_ID == Q2N_ISA_AVX2

typedef struct {
    __m256d lo, hi;
} V8;

Q2N_INLINE Q2N_TARGET V8 V8_ZERO(void) {
    V8 r;

    r.lo = _mm256_setzero_pd();
    r.hi = _mm256_setzero_pd();
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_SET1(double value) {
    V8 r;

    r.lo = _mm256_set1_pd(value);
    r.hi = r.lo;
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_ADD(V8 a, V8 b) {
    V8 r;

    r.lo = _mm256_add_pd(a.lo, b.lo);
    r.hi = _mm256_add_pd(a.hi, b.hi);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_SUB(V8 a, V8 b) {
    V8 r;

    r.lo = _mm256_sub_pd(a.lo, b.lo);
    r.hi = _mm256_sub_pd(a.hi, b.hi);
    return r;
}

// a * b + c with one rounding
Q2N_INLINE Q2N_TARGET V8 V8_FMADD(V8 a, V8 b, V8 c) {
    V8 r;

    r.lo = _mm256_fmadd_pd(a.lo, b.lo, c.lo);
    r.hi = _mm256_fmadd_pd(a.hi, b.hi, c.hi);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_REVERSE(V8 a) {
    V8 r;

    r.lo = _mm256_permute4x64_pd(a.hi, 0x1B);
    r.hi = _mm256_permute4x64_pd(a.lo, 0x1B);
    return r;
}

Q2N_INLINE Q2N_TARGET void V8_STORE(double* out, V8 a) {
    _mm256_storeu_pd(out, a.lo);
    _mm256_storeu_pd(out + 4, a.hi);
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F64(const double* p) {
    V8 r;

    r.lo = _mm256_loadu_pd(p);
    r.hi = _mm256_loadu_pd(p + 4);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F32(const float* p) {
    V8 r;

    r.lo = _mm256_cvtps_pd(_mm_loadu_ps(p));
    r.hi = _mm256_cvtps_pd(_mm_loadu_ps(p + 4));
    return r;
}

// Convert eight 32-bit integers to doubles
Q2N_INLINE Q2N_TARGET V8 ISA(v8_from_i32)(__m256i values) {
    V8 r;

    r.lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(values));
    r.hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(values, 1));
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U8(const uint8_t* p) {
    return ISA(v8_from_i32)(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U16(const uint16_t* p) {
    return ISA(v8_from_i32)(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)));
}

// Split 12 interleaved values into the values at 3k, 3k + 1 and 3k + 2
Q2N_INLINE Q2N_TARGET void ISA(v4_deinterleave3)(__m256d v0, __m256d v1, __m256d v2,
        __m256d* a, __m256d* b, __m256d* d) {
    const __m256d p0 = _mm256_permute2f128_pd(v0, v1, 0x30);
    const __m256d p1 = _mm256_permute2f128_pd(v0, v2, 0x21);
    const __m256d p2 = _mm256_permute2f128_pd(v1, v2, 0x30);

    *a = _mm256_shuffle_pd(p0, p1, 0xA);
    *b = _mm256_shuffle_pd(p0, p2, 0x5);
    *d = _mm256_shuffle_pd(p1, p2, 0xA);
}

// Split 24 interleaved values x, y, z into the values at 3k, 3k + 1 and 3k + 2
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE3(V8 x, V8 y, V8 z, V8* a, V8* b, V8* d) {
    ISA(v4_deinterleave3)(x.lo, x.hi, y.lo, &a->lo, &b->lo, &d->lo);
    ISA(v4_deinterleave3)(y.hi, z.lo, z.hi, &a->hi, &b->hi, &d->hi);
}

// Split 8 interleaved values into the values at 2k and 2k + 1
Q2N_INLINE Q2N_TARGET void ISA(v4_deinterleave2)(__m256d v0, __m256d v1, __m256d* a,
        __m256d* b) {
    *a = _mm256_permute4x64_pd(_mm256_unpacklo_pd(v0, v1), 0xD8);
    *b = _mm256_permute4x64_pd(_mm256_unpackhi_pd(v0, v1), 0xD8);
}

// Split 16 interleaved values x, y into the values at 2k and 2k + 1
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE2(V8 x, V8 y, V8* a, V8* b) {
    ISA(v4_deinterleave2)(x.lo, x.hi, &a->lo, &b->lo);
    ISA(v4_deinterleave2)(y.lo, y.hi, &a->hi, &b->hi);
}

// Transpose four rows of four values
Q2N_INLINE Q2N_TARGET void ISA(v4_transpose)(__m256d r0, __m256d r1, __m256d r2,
        __m256d r3, __m256d* a, __m256d* b, __m256d* c, __m256d* d) {
    const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    const __m256d t3 = _mm256_unpackhi_pd(r2, r3);

    *a = _mm256_permute2f128_pd(t0, t2, 0x20);
    *b = _mm256_permute2f128_pd(t1, t3, 0x20);
    *c = _mm256_permute2f128_pd(t0, t2, 0x31);
    *d = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// Split 32 interleaved values into the values at 4k, 4k + 1, 4k + 2 and 4k + 3
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE4(V8 x0, V8 x1, V8 x2, V8 x3, V8* a, V8* b,
        V8* c, V8* d) {
    ISA(v4_transpose)(x0.lo, x0.hi, x1.lo, x1.hi, &a->lo, &b->lo, &c->lo, &d->lo);
    ISA(v4_transpose)(x2.lo, x2.hi, x3.lo, x3.hi, &a->hi, &b->hi, &c->hi, &d->hi);
}

#elif Q2N_ISA_ID == Q2N_ISA_NEON

typedef struct {
    float64x2_t v[4];
} V8;

Q2N_INLINE Q2N_TARGET V8 V8_ZERO(void) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vdupq_n_f64(0.0);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_SET1(double value) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vdupq_n_f64(value);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_ADD(V8 a, V8 b) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vaddq_f64(a.v[k], b.v[k]);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_SUB(V8 a, V8 b) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vsubq_f64(a.v[k], b.v[k]);
    return r;
}

// a * b + c with one rounding
Q2N_INLINE Q2N_TARGET V8 V8_FMADD(V8 a, V8 b, V8 c) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vfmaq_f64(c.v[k], a.v[k], b.v[k]);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_REVERSE(V8 a) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vextq_f64(a.v[3 - k], a.v[3 - k], 1);
    return r;
}

Q2N_INLINE Q2N_TARGET void V8_STORE(double* out, V8 a) {
    for (int k = 0; k < 4; k++) vst1q_f64(out + 2 * k, a.v[k]);
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F64(const double* p) {
    V8 r;

    for (int k = 0; k < 4; k++) r.v[k] = vld1q_f64(p + 2 * k);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F32(const float* p) {
    const float32x4_t lo = vld1q_f32(p), hi = vld1q_f32(p + 4);
    V8 r;

    r.v[0] = vcvt_f64_f32(vget_low_f32(lo));
    r.v[1] = vcvt_high_f64_f32(lo);
    r.v[2] = vcvt_f64_f32(vget_low_f32(hi));
    r.v[3] = vcvt_high_f64_f32(hi);
    return r;
}

// Convert eight 16-bit unsigned integers to doubles
Q2N_INLINE Q2N_TARGET V8 ISA(v8_from_u16)(uint16x8_t values) {
    const uint32x4_t lo = vmovl_u16(vget_low_u16(values));
    const uint32x4_t hi = vmovl_high_u16(values);
    V8 r;

    r.v[0] = vcvtq_f64_u64(vmovl_u32(vget_low_u32(lo)));
    r.v[1] = vcvtq_f64_u64(vmovl_high_u32(lo));
    r.v[2] = vcvtq_f64_u64(vmovl_u32(vget_low_u32(hi)));
    r.v[3] = vcvtq_f64_u64(vmovl_high_u32(hi));
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U8(const uint8_t* p) {
    return ISA(v8_from_u16)(vmovl_u8(vld1_u8(p)));
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U16(const uint16_t* p) {
    return ISA(v8_from_u16)(vld1q_u16(p));
}

// Split 24 interleaved values x, y, z into the values at 3k, 3k + 1 and 3k + 2
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE3(V8 x, V8 y, V8 z, V8* a, V8* b, V8* d) {
    float64x2_t q[12];

    for (int k = 0; k < 4; k++) {
        q[k] = x.v[k];
        q[4 + k] = y.v[k];
        q[8 + k] = z.v[k];
    }
    for (int k = 0; k < 4; k++) {
        a->v[k] = vcopyq_laneq_f64(q[3 * k], 1, q[3 * k + 1], 1);
        b->v[k] = vextq_f64(q[3 * k], q[3 * k + 2], 1);
        d->v[k] = vcopyq_laneq_f64(q[3 * k + 1], 1, q[3 * k + 2], 1);
    }
}

// Split 16 interleaved values x, y into the values at 2k and 2k + 1
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE2(V8 x, V8 y, V8* a, V8* b) {
    for (int k = 0; k < 2; k++) {
        a->v[k] = vuzp1q_f64(x.v[2 * k], x.v[2 * k + 1]);
        b->v[k] = vuzp2q_f64(x.v[2 * k], x.v[2 * k + 1]);
        a->v[2 + k] = vuzp1q_f64(y.v[2 * k], y.v[2 * k + 1]);
        b->v[2 + k] = vuzp2q_f64(y.v[2 * k], y.v[2 * k + 1]);
    }
}

// Split 32 interleaved values into the values at 4k, 4k + 1, 4k + 2 and 4k + 3
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE4(V8 x0, V8 x1, V8 x2, V8 x3, V8* a, V8* b,
        V8* c, V8* d) {
    float64x2_t q[16];

    for (int k = 0; k < 4; k++) {
        q[k] = x0.v[k];
        q[4 + k] = x1.v[k];
        q[8 + k] = x2.v[k];
        q[12 + k] = x3.v[k];
    }
    for (int k = 0; k < 4; k++) {
        a->v[k] = vuzp1q_f64(q[4 * k], q[4 * k + 2]);
        b->v[k] = vuzp2q_f64(q[4 * k], q[4 * k + 2]);
        c->v[k] = vuzp1q_f64(q[4 * k + 1], q[4 * k + 3]);
        d->v[k] = vuzp2q_f64(q[4 * k + 1], q[4 * k + 3]);
    }
}

#else

typedef struct {
    double v[8];
} V8;

Q2N_INLINE Q2N_TARGET V8 V8_ZERO(void) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = 0.0;
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_SET1(double value) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = value;
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_ADD(V8 a, V8 b) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = a.v[k] + b.v[k];
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_SUB(V8 a, V8 b) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = a.v[k] - b.v[k];
    return r;
}

// a * b + c with one rounding
Q2N_INLINE Q2N_TARGET V8 V8_FMADD(V8 a, V8 b, V8 c) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = fma(a.v[k], b.v[k], c.v[k]);
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_REVERSE(V8 a) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = a.v[7 - k];
    return r;
}

Q2N_INLINE Q2N_TARGET void V8_STORE(double* out, V8 a) {
    for (int k = 0; k < 8; k++) out[k] = a.v[k];
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F64(const double* p) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = p[k];
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_F32(const float* p) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = (double)p[k];
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U8(const uint8_t* p) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = (double)p[k];
    return r;
}

Q2N_INLINE Q2N_TARGET V8 V8_LOAD_U16(const uint16_t* p) {
    V8 r;

    for (int k = 0; k < 8; k++) r.v[k] = (double)p[k];
    return r;
}

// Split 24 interleaved values x, y, z into the values at 3k, 3k + 1 and 3k + 2
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE3(V8 x, V8 y, V8 z, V8* a, V8* b, V8* d) {
    double t[24];

    for (int k = 0; k < 8; k++) {
        t[k] = x.v[k];
        t[8 + k] = y.v[k];
        t[16 + k] = z.v[k];
    }
    for (int k = 0; k < 8; k++) {
        a->v[k] = t[3 * k];
        b->v[k] = t[3 * k + 1];
        d->v[k] = t[3 * k + 2];
    }
}

// Split 16 interleaved values x, y into the values at 2k and 2k + 1
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE2(V8 x, V8 y, V8* a, V8* b) {
    double t[16];

    for (int k = 0; k < 8; k++) {
        t[k] = x.v[k];
        t[8 + k] = y.v[k];
    }
    for (int k = 0; k < 8; k++) {
        a->v[k] = t[2 * k];
        b->v[k] = t[2 * k + 1];
    }
}

// Split 32 interleaved values into the values at 4k, 4k + 1, 4k + 2 and 4k + 3
Q2N_INLINE Q2N_TARGET void V8_DEINTERLEAVE4(V8 x0, V8 x1, V8 x2, V8 x3, V8* a, V8* b,
        V8* c, V8* d) {
    double t[32];

    for (int k = 0; k < 8; k++) {
        t[k] = x0.v[k];
        t[8 + k] = x1.v[k];
        t[16 + k] = x2.v[k];
        t[24 + k] = x3.v[k];
    }
    for (int k = 0; k < 8; k++) {
        a->v[k] = t[4 * k];
        b->v[k] = t[4 * k + 1];
        c->v[k] = t[4 * k + 2];
        d->v[k] = t[4 * k + 3];
    }
}

#endif

/*
 * The sum of the eight lanes, in an order that does not depend on the
 * instruction set: lanes four apart, then two apart, then the last pair.
 */
// Add the eight lanes of a vector
Q2N_INLINE Q2N_TARGET double V8_REDUCE(V8 a) {
    double t[8];

    V8_STORE(t, a);
    return ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]));
}

#define KNAME(name) Q2N_PASTE(Q2N_PASTE(name, Q2N_DT), Q2N_ISA)

#define Q2N_T double
#define Q2N_DT f64
#define V8_LOAD V8_LOAD_F64
#include "_prepare.h"
#include "_direct.h"
#undef Q2N_T
#undef Q2N_DT
#undef V8_LOAD

#define Q2N_T float
#define Q2N_DT f32
#define V8_LOAD V8_LOAD_F32
#include "_prepare.h"
#include "_direct.h"
#undef Q2N_T
#undef Q2N_DT
#undef V8_LOAD

#define Q2N_T uint8_t
#define Q2N_DT u8
#define V8_LOAD V8_LOAD_U8
#include "_prepare.h"
#include "_direct.h"
#undef Q2N_T
#undef Q2N_DT
#undef V8_LOAD

#define Q2N_T uint16_t
#define Q2N_DT u16
#define V8_LOAD V8_LOAD_U16
#include "_prepare.h"
#include "_direct.h"
#undef Q2N_T
#undef Q2N_DT
#undef V8_LOAD

#undef KNAME

// The kernels of this instruction set, indexed by sample type
static const Kernels ISA(kernels) = {
    Q2N_ISA_ID,
    Q2N_DIRECT,
    {ISA(prepare_f64), ISA(prepare_f32), ISA(prepare_u8), ISA(prepare_u16)},
    {ISA(direct_f64), ISA(direct_f32), ISA(direct_u8), ISA(direct_u16)},
};

#undef V8
#undef V8_ZERO
#undef V8_SET1
#undef V8_ADD
#undef V8_SUB
#undef V8_FMADD
#undef V8_REVERSE
#undef V8_STORE
#undef V8_REDUCE
#undef V8_DEINTERLEAVE2
#undef V8_DEINTERLEAVE3
#undef V8_DEINTERLEAVE4
#undef V8_LOAD_F64
#undef V8_LOAD_F32
#undef V8_LOAD_U8
#undef V8_LOAD_U16
#undef ISA
#undef Q2N_PASTE
#undef Q2N_PASTE_

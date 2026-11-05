#pragma once
// ruquant W4A4 GEMM for prefill widths, shaped for gfx906 (no matrix unit: v_dot8_i32_i4 on VGPRs).
// A workgroup of 256 threads owns 128 channels x 64 tokens and runs its slice of K one tile of 8 code words (64
// inputs, the smallest activation group) at a time. Everything a tile needs goes through LDS, double buffered and
// fetched a tile ahead: the code words, and the group terms already decoded (8 - z and m per token, colsum and the
// block multiplier k per channel). Each thread multiplies 4 channels by 8 tokens from registers (per word one 16 B
// weight read and two 16 B activation reads for 32 dot8s). The integer math is the skinny kernel's
// (rq-w4a4-skinny.cuh): per activation group sum = (8 - z) colsum + code products, tot += (k m sum) >> RQ_TERM_SHIFT,
// so the result is bit-exact with the CPU reference. Split-K (gridDim.z > 1) adds int32 partials atomically
// (exact in any order) and the last workgroup of a tile writes it out.
#include "rq-w4a4-skinny.cuh"

constexpr int RQ_GEMM_BN      = 128; // channels per workgroup
constexpr int RQ_GEMM_BT      = 64;  // tokens per workgroup
constexpr int RQ_GEMM_KT      = 8;   // code words per K tile
constexpr int RQ_GEMM_THREADS = 256;

// pins 8 values in registers at this point of the program (an empty volatile asm that reads and writes them)
static __device__ __forceinline__ void rq_keep8(int (&x)[8]) {
#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]));
#else
    asm volatile("" : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3]), "+r"(x[4]), "+r"(x[5]), "+r"(x[6]), "+r"(x[7]));
#endif
}

struct rq_gemm_stage {
    int32_t w[RQ_GEMM_KT][RQ_GEMM_BN]; // code words, word-major
    int32_t a[RQ_GEMM_KT][RQ_GEMM_BT];
    int32_t zc[RQ_GEMM_BT];            // 8 - z_g per token
    int32_t m[RQ_GEMM_BT];             // m_g per token
    int32_t cs[RQ_GEMM_BN];            // colsum_g per channel
    int32_t kb[RQ_GEMM_BN];            // k_b per channel
};

// Aq [Mp][Kw] token-major codes, mz [K / G][Mp] (m_g | z_g << 8), sa [Mp]: Mp a multiple of RQ_GEMM_BT, rows past M
// are read but never written out. Wt in the rq_wt_index layout, sw [N], colsum [K / G][N], wk [K / 64][N], D [M][ldd].
// GSH: log2 of the K tiles per activation group (G / 64); kz: K tiles per split (a multiple of a group). N a
// multiple of RQ_GEMM_BN;
// grid (N / RQ_GEMM_BN, Mp / RQ_GEMM_BT, splits). With splits, acc [Mp][N] and cnt [tiles] start at zero.
template <int GSH>
static __global__ void __launch_bounds__(RQ_GEMM_THREADS)
rq_gemm_kernel(const int32_t * __restrict__ Aq, const int32_t * __restrict__ mz, const float * __restrict__ sa,
               const int32_t * __restrict__ Wt, const float * __restrict__ sw, const int16_t * __restrict__ colsum,
               const int8_t * __restrict__ wk, float * __restrict__ D, int ldd, int N, int M, int Mp, int Kw, int kz,
               int32_t * __restrict__ acc, int32_t * __restrict__ cnt) {
    __shared__ rq_gemm_stage st[2];
    constexpr int gt = 1 << GSH;
    const int tid = threadIdx.x, wid = rq_wave_id();
    const int n0 = blockIdx.x * RQ_GEMM_BN, t0 = blockIdx.y * RQ_GEMM_BT;
    const int cq = tid % 32, tq = tid / 32; // this thread's outputs: channels n0 + 4 cq + c, tokens t0 + 8 tq + t
    const int k_lo = blockIdx.z * kz, k_hi = min(k_lo + kz, Kw / RQ_GEMM_KT);

    // per tile, each thread fetches 4 weight words (channel group gi, step si, lane li of the rq_wt_index layout),
    // 2 activation words (token at, words aw, aw + 1) and one group term: threads 0..127 a channel's colsum and k,
    // 128..191 a token's m | z << 8 (split by wave, so the addresses stay scalar)
    const int li = tid % RQ_WAVE, si = (tid / RQ_WAVE) % 2, gi = tid / (2 * RQ_WAVE);
    const rq_words<RQ_KW> * wsrc = reinterpret_cast<const rq_words<RQ_KW> *>(Wt) +
                                   ((size_t) (n0 / RQ_WAVE + gi) * (Kw / RQ_KW) + si) * RQ_WAVE + li;
    const int wn = gi * RQ_WAVE + li;
    const int at = tid / 4, aw = 2 * (tid % 4);
    const int2 * asrc = reinterpret_cast<const int2 *>(Aq + (size_t) (t0 + at) * Kw + aw);

    rq_words<RQ_KW> wv;
    int2 av;
    int p0 = 0, p1 = 0;
#define RQ_GEMM_FETCH(kt_)                                                          \
    do {                                                                            \
        wv = wsrc[(size_t) (kt_) * 2 * RQ_WAVE];                                    \
        av = asrc[(kt_) * RQ_GEMM_KT / 2];                                          \
        if (wid < RQ_GEMM_BN / RQ_WAVE) {                                           \
            p0 = colsum[(size_t) ((kt_) >> GSH) * N + n0 + tid];                    \
            p1 = wk[(size_t) (kt_) * N + n0 + tid];                                 \
        } else if (wid < (RQ_GEMM_BN + RQ_GEMM_BT) / RQ_WAVE) {                     \
            p0 = mz[(size_t) ((kt_) >> GSH) * Mp + t0 + tid - RQ_GEMM_BN];          \
        }                                                                           \
    } while (0)
#define RQ_GEMM_STASH(s_)                                                           \
    do {                                                                            \
        _Pragma("unroll") for (int i = 0; i < 4; ++i) (s_).w[si * 4 + i][wn] = wv.v[i]; \
        (s_).a[aw][at] = av.x;                                                      \
        (s_).a[aw + 1][at] = av.y;                                                  \
        if (wid < RQ_GEMM_BN / RQ_WAVE) {                                           \
            (s_).cs[tid] = p0;                                                      \
            (s_).kb[tid] = p1;                                                      \
        } else if (wid < (RQ_GEMM_BN + RQ_GEMM_BT) / RQ_WAVE) {                     \
            (s_).zc[tid - RQ_GEMM_BN] = 8 - (p0 >> 8);                              \
            (s_).m[tid - RQ_GEMM_BN] = p0 & 0xff;                                   \
        }                                                                           \
    } while (0)

    int sum[4][8], tot[4][8];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int t = 0; t < 8; ++t) sum[c][t] = tot[c][t] = 0;

    if (k_lo < k_hi) {
        RQ_GEMM_FETCH(k_lo);
        RQ_GEMM_STASH(st[0]);
    }
    __syncthreads();
    for (int kt = k_lo; kt < k_hi; ++kt) {
        const rq_gemm_stage & s = st[(kt - k_lo) & 1];
        const bool more = kt + 1 < k_hi;
        if (more) RQ_GEMM_FETCH(kt + 1); // in flight while this tile runs
        if ((kt & (gt - 1)) == 0) { // a group starts: its zero-point term
            const int4 c4 = *reinterpret_cast<const int4 *>(&s.cs[4 * cq]);
            const int4 z0 = *reinterpret_cast<const int4 *>(&s.zc[8 * tq]);
            const int4 z1 = *reinterpret_cast<const int4 *>(&s.zc[8 * tq + 4]);
            const int cs[4] = { c4.x, c4.y, c4.z, c4.w };
            const int zc[8] = { z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w };
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int t = 0; t < 8; ++t) sum[c][t] = __mul24(zc[t], cs[c]);
        }
        // one word ahead: word j + 1's LDS reads are in flight during word j's dot8s. The fences keep the compiler
        // from hoisting all of them and sinking the dot8s to the fold (it did, at G = 64: 200 VGPRs, one wave per
        // SIMD): the sums pinned per word, sched_barrier for the machine scheduler, the empty asm for the IR passes
        int4 w = *reinterpret_cast<const int4 *>(&s.w[0][4 * cq]);
        int4 a0 = *reinterpret_cast<const int4 *>(&s.a[0][8 * tq]);
        int4 a1 = *reinterpret_cast<const int4 *>(&s.a[0][8 * tq + 4]);
#pragma unroll
        for (int j = 0; j < RQ_GEMM_KT; ++j) {
            const int jn = j + 1 < RQ_GEMM_KT ? j + 1 : j;
            const int4 wn  = *reinterpret_cast<const int4 *>(&s.w[jn][4 * cq]);
            const int4 a0n = *reinterpret_cast<const int4 *>(&s.a[jn][8 * tq]);
            const int4 a1n = *reinterpret_cast<const int4 *>(&s.a[jn][8 * tq + 4]);
            const int wc[4] = { w.x, w.y, w.z, w.w };
            const int a8[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int t = 0; t < 8; ++t) sum[c][t] = rq_dot8(a8[t], wc[c], sum[c][t]);
#pragma unroll
            for (int c = 0; c < 4; ++c) rq_keep8(sum[c]);
            rq_block_fence();
            asm volatile("" ::: "memory");
            w = wn;
            a0 = a0n;
            a1 = a1n;
        }
        if (((kt + 1) & (gt - 1)) == 0) { // the group ends: fold it in
            const int4 k4 = *reinterpret_cast<const int4 *>(&s.kb[4 * cq]);
            const int4 m0 = *reinterpret_cast<const int4 *>(&s.m[8 * tq]);
            const int4 m1 = *reinterpret_cast<const int4 *>(&s.m[8 * tq + 4]);
            // narrowed (k 1..15, m 1..255), so k m is a known 24-bit product: unnarrowed, the outer __mul24 cost a
            // quarter-rate multiply (an inline-asm fold without the compiler's extra shift per output was slower)
            const int kb[4] = { k4.x & 0x7fff, k4.y & 0x7fff, k4.z & 0x7fff, k4.w & 0x7fff };
            const int m[8] = { m0.x & 0xff, m0.y & 0xff, m0.z & 0xff, m0.w & 0xff, m1.x & 0xff, m1.y & 0xff, m1.z & 0xff, m1.w & 0xff };
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int t = 0; t < 8; ++t) tot[c][t] += __mul24(kb[c] * m[t], sum[c][t]) >> RQ_TERM_SHIFT;
        }
        if (more) RQ_GEMM_STASH(st[(kt + 1 - k_lo) & 1]);
        __syncthreads();
    }

    if (gridDim.z > 1) { // split-K: add the partial sums; the last workgroup of the tile writes them out
        int32_t * at_ = acc + (size_t) (t0 + 8 * tq) * N + n0 + 4 * cq;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int c = 0; c < 4; ++c) atomicAdd(at_ + (size_t) t * N + c, tot[c][t]);
        __threadfence();
        __shared__ int last;
        __syncthreads();
        if (tid == 0) last = atomicAdd(&cnt[blockIdx.y * gridDim.x + blockIdx.x], 1) == (int) gridDim.z - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int c = 0; c < 4; ++c) tot[c][t] = rq_load_l2(at_ + (size_t) t * N + c);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int tok = t0 + 8 * tq + t;
        if (tok >= M) break;
        const float u = sa[tok];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = n0 + 4 * cq + c;
            D[(size_t) tok * ldd + n] =
                __fmul_rn(__fmul_rn(__fmul_rn(__int2float_rn(tot[c][t]), (float) (1 << RQ_TERM_SHIFT)), u), sw[n]);
        }
    }
#undef RQ_GEMM_FETCH
#undef RQ_GEMM_STASH
}

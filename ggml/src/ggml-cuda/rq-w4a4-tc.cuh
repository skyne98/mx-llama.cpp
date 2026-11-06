#pragma once
// ruquant W4A4 on NVIDIA INT4 tensor cores (Turing, Ampere, Ada: mma.sync with .s4 operands; Hopper and Blackwell
// dropped them). One activation group of 64 inputs is exactly one m16n8k64 MMA, so the group's code products come
// out of the tensor core as an exact int32 and the rest is the skinny kernel's integer fold (rq-w4a4-skinny.cuh):
// sum = (8 - z) colsum + code products, here preloaded into the MMA accumulator; tot += (k m sum) >> RQ_TERM_SHIFT;
// y = ((fp32(tot) * 2^RQ_TERM_SHIFT) * u) * s_w. Bit-exact with the CPU reference and the gfx906 dot8 kernels.
//
// The weight is the A operand (16 channels x 64 inputs), the activation the B operand (64 inputs x 8 tokens), so a
// decode step (1 token) or a DFlash2 verify (8 tokens) is one n8 tile. Weights are repacked once, in place, into the
// MMA's fragment order: tile (channel tile c16, group g) is 32 lanes x 16 B, lane = 4 * gid + tig holding the code
// words (channel gid, word tig), (gid + 8, tig), (gid, tig + 4), (gid + 8, tig + 4) of the group's 8 words. A warp
// reads its 512 B tile with one coalesced 16 B load per lane and feeds it to the MMA as is. Next to it, per tile, one
// word per gid: the group's colsum and weight multiplier for channels gid and gid + 8 (12 + 4 bits each).
// Layout of the repacked tensor: [codes N K / 2 B | s_w N floats | ck (N / 16) (K / 64) 8 words].
#include <stdint.h>

#define RQ_TC_MIN_CC 750 // Turing: m8n8k32 .s4
#define RQ_TC_MAX_CC 900 // Hopper and later have no INT4 MMA

static __host__ __device__ __forceinline__ size_t rq_tc_w_index(int channel, int k_word, int Kw) {
    const int c16 = channel / 16, r = channel % 16, g = k_word / 8, w = k_word % 8;
    const int lane = (r % 8) * 4 + w % 4, slot = r / 8 + 2 * (w / 4);
    return (((size_t) c16 * (Kw / 8) + g) * 32 + lane) * 4 + slot;
}

// ck word of tile (c16, g) for gid: channel gid in the low half, gid + 8 in the high half, each colsum (12-bit signed,
// |colsum| <= 512) | k << 12 (k in 1..15)
static __host__ __device__ __forceinline__ size_t rq_tc_ck_index(int c16, int g, int gid, int ng) {
    return ((size_t) c16 * ng + g) * 8 + gid;
}

// d = a b + c over one group: a the weight fragment, b0 b1 the activation words of this lane's token, exact int32
static __device__ __forceinline__ void rq_tc_mma(int (&d)[4], const int4 a, const int b0, const int b1, const int (&c)[4]) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#elif !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    // Turing: the same fragments as four m8n8k32 (rows 0-7 / 8-15, inputs 0-31 / 32-63)
    int t0, t1, t2, t3;
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(t0), "=r"(t1) : "r"(a.x), "r"(b0), "r"(c[0]), "r"(c[1]));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(a.z), "r"(b1), "r"(t0), "r"(t1));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(t2), "=r"(t3) : "r"(a.y), "r"(b0), "r"(c[2]), "r"(c[3]));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(d[2]), "=r"(d[3]) : "r"(a.w), "r"(b1), "r"(t2), "r"(t3));
#else
    GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
    NO_DEVICE_CODE;
#endif
}

// One weight per launch. A warp owns MT channel tiles (16 each) x NT token tiles (8 each); a block WN x WT warps.
// Aq [Mp][Kw] token-major codes, mz [K / 64][Mp] (m | z << 8), sa [Mp]; Wf, sw, ck the repacked weight; D [M][ldd].
// gz: groups per K split (gridDim.z splits); with splits, acc [Mp][N] and cnt [gridDim.x * gridDim.y] start at zero.
// grid (ceil(N / (16 MT WN)), Mp / (8 NT WT), splits); Mp a multiple of 8 NT WT; N a multiple of 16.
template <int MT, int NT, int WN, int WT>
static __global__ void __launch_bounds__(32 * WN * WT)
rq_tc_kernel(const int32_t * __restrict__ Aq, const int32_t * __restrict__ mz, const float * __restrict__ sa,
             const int4 * __restrict__ Wf, const float * __restrict__ sw, const uint32_t * __restrict__ ck,
             float * __restrict__ D, int ldd, int N, int M, int Mp, int Kw, int gz,
             int32_t * __restrict__ acc, int32_t * __restrict__ cnt) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int lane = threadIdx.x % 32, gid = lane / 4, tig = lane % 4;
    const int warp = threadIdx.x / 32, wn = warp % WN, wt = warp / WN;
    const int c16_0 = (blockIdx.x * WN + wn) * MT;
    const int tok0 = (blockIdx.y * WT + wt) * NT * 8;
    const int ng = Kw / 8;
    const int g_lo = blockIdx.z * gz, g_hi = min(g_lo + gz, ng);
    const int n_c16 = N / 16;

    int tot[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) tot[mt][nt][i] = 0;

    // tiles past N (the last block of a shape that is not a multiple of the block) re-read the last tile
    int c16[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) c16[mt] = min(c16_0 + mt, n_c16 - 1);
    const int32_t * ap[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) ap[nt] = Aq + (size_t) (tok0 + nt * 8 + gid) * Kw + tig;

    // Decode streams the weights at ~720 GB/s (ncu), as fast as the skinny dot8 kernel. Neither register double
    // buffering (40 -> 69 registers, prefill 139: one block per SM; both slower) nor rotating each warp's start group
    // against DRAM channel conflicts helped
#pragma unroll 2
    for (int g = g_lo; g < g_hi; ++g) {
        int4 w[MT];
        uint32_t ckw[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            w[mt]   = Wf[((size_t) c16[mt] * ng + g) * 32 + lane];
            ckw[mt] = ck[rq_tc_ck_index(c16[mt], g, gid, ng)];
        }
        int b0[NT], b1[NT];
        int2 mzv[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            b0[nt]  = ap[nt][g * 8];
            b1[nt]  = ap[nt][g * 8 + 4];
            mzv[nt] = *reinterpret_cast<const int2 *>(mz + (size_t) g * Mp + tok0 + nt * 8 + 2 * tig);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int cs0 = (int) (ckw[mt] << 20) >> 20, k0 = (ckw[mt] >> 12) & 0xF;
            const int cs1 = (int) (ckw[mt] << 4) >> 20,  k1 = ckw[mt] >> 28;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int zc0 = 8 - (mzv[nt].x >> 8), m0 = mzv[nt].x & 0xff;
                const int zc1 = 8 - (mzv[nt].y >> 8), m1 = mzv[nt].y & 0xff;
                const int c[4] = { zc0 * cs0, zc1 * cs0, zc0 * cs1, zc1 * cs1 };
                int d[4];
                rq_tc_mma(d, w[mt], b0[nt], b1[nt], c);
                // k m <= 15 * 255 and |sum| <= 2^15: the products fit int32, as in the dot8 kernels
                tot[mt][nt][0] += (k0 * m0 * d[0]) >> RQ_TERM_SHIFT;
                tot[mt][nt][1] += (k0 * m1 * d[1]) >> RQ_TERM_SHIFT;
                tot[mt][nt][2] += (k1 * m0 * d[2]) >> RQ_TERM_SHIFT;
                tot[mt][nt][3] += (k1 * m1 * d[3]) >> RQ_TERM_SHIFT;
            }
        }
    }

    // output (mt, nt, i): channel 16 c16 + gid + 8 (i / 2), token tok0 + 8 nt + 2 tig + i % 2
    if (gridDim.z > 1) { // split-K: add the partial sums; the last block of the tile writes them out
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (c16_0 + mt >= n_c16) continue;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int t = tok0 + nt * 8 + 2 * tig + i % 2, n = (c16_0 + mt) * 16 + gid + 8 * (i / 2);
                    if (t < M) atomicAdd(acc + (size_t) t * N + n, tot[mt][nt][i]);
                }
        }
        __threadfence();
        __shared__ int last;
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(&cnt[blockIdx.y * gridDim.x + blockIdx.x], 1) == (int) gridDim.z - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (c16_0 + mt >= n_c16) continue;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int t = tok0 + nt * 8 + 2 * tig + i % 2, n = (c16_0 + mt) * 16 + gid + 8 * (i / 2);
                    if (t < M) tot[mt][nt][i] = __ldcg(acc + (size_t) t * N + n);
                }
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        if (c16_0 + mt >= n_c16) continue;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = tok0 + nt * 8 + 2 * tig + i % 2, n = (c16_0 + mt) * 16 + gid + 8 * (i / 2);
                if (t < M) {
                    D[(size_t) t * ldd + n] =
                        __fmul_rn(__fmul_rn(__fmul_rn(__int2float_rn(tot[mt][nt][i]), (float) (1 << RQ_TERM_SHIFT)), sa[t]), sw[n]);
                }
            }
    }
#else
    GGML_UNUSED(Aq); GGML_UNUSED(mz); GGML_UNUSED(sa); GGML_UNUSED(Wf); GGML_UNUSED(sw); GGML_UNUSED(ck); GGML_UNUSED(D);
    GGML_UNUSED(ldd); GGML_UNUSED(N); GGML_UNUSED(M); GGML_UNUSED(Mp); GGML_UNUSED(Kw); GGML_UNUSED(gz); GGML_UNUSED(acc);
    GGML_UNUSED(cnt);
    NO_DEVICE_CODE;
#endif
}

// repack: thread per (channel, code word) from the original blocks (src) into the fragment order; word 0 also s_w
static __global__ void rq_tc_repack_kernel(const uint8_t * __restrict__ src, int32_t * __restrict__ wf, float * __restrict__ sw,
                                           int N, int K) {
    const int Kw = K / 8;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) N * Kw) return;
    const int c = (int) (i / Kw), kw = (int) (i % Kw);
    const uint8_t * row = src + (size_t) c * (K / QK_W4A4) * sizeof(block_w4a4);
    wf[rq_tc_w_index(c, kw, Kw)] = *reinterpret_cast<const int32_t *>(row + (kw / 32) * sizeof(block_w4a4) + 16 + (kw % 32) * 4);
    if (kw == 0) {
        sw[c] = reinterpret_cast<const block_w4a4 *>(row)->d;
    }
}

// ck: thread per (channel tile, group, gid)
static __global__ void rq_tc_ck_kernel(const uint8_t * __restrict__ src, uint32_t * __restrict__ ck, int N, int K) {
    const int ng = K / 64;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) (N / 16) * ng * 8) return;
    const int gid = (int) (i % 8), g = (int) ((i / 8) % ng), c16 = (int) (i / 8 / ng);
    uint32_t word = 0;
    for (int h = 0; h < 2; ++h) {
        const int c = c16 * 16 + gid + 8 * h;
        const block_w4a4 * b = reinterpret_cast<const block_w4a4 *>(src + ((size_t) c * (K / QK_W4A4) + g / 4) * sizeof(block_w4a4));
        int sum = 0;
        for (int w = (g % 4) * 8; w < (g % 4) * 8 + 8; ++w)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum += (int32_t) (((b->qs[w] >> (4 * j)) & 0xF) << 28) >> 28;
        const int k = RQ_W_MULT_G(b->sum, g % 4);
        word |= (((uint32_t) sum & 0xFFFu) | ((uint32_t) k << 12)) << (16 * h);
    }
    ck[rq_tc_ck_index(c16, g, gid, ng)] = word;
}

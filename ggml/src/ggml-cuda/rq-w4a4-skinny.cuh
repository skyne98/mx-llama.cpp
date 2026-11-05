#pragma once
// ruquant skinny W4A4 GEMM for decode / speculative-verify widths (T = 1..64 tokens), shaped for gfx906
// (from ruquant's gfx906/w4a4_skinny.h, measured on MI50).
//
// gfx906 has no matrix unit, so every instruction that is not v_dot8_i32_i4 takes a VALU slot that could
// have done 8 multiply-adds. The inner loop is therefore weight loads + dot8 only:
//   - each lane owns one output channel per channel group (R groups per wave): no cross-lane reduction;
//   - activations are wave-uniform, so they come in through scalar loads and feed dot8 from SGPRs. They are
//     stored step-major so one step's T*KA words are contiguous;
//   - weights are repacked (once, in place, on first use) so a wave's load is one contiguous 16 B/lane run;
//   - split-K in two levels: the waves of a workgroup reduce through LDS, workgroups add int32 partials
//     atomically. Integer adds are exact, so the result is bit-exact with the CPU reference.
// On CUDA (wave = 32, dot8 emulated) the same source runs on the 3090 to check the indexing (RQ_SKINNY=1).
#include <stddef.h>
#include <stdint.h>

#if defined(GGML_USE_HIP)
#define RQ_WAVE 64
static __device__ __forceinline__ int rq_dot8(int a, int b, int c) {
#if defined(__gfx906__)
    return __builtin_amdgcn_sdot8(a, b, c, false); // v_dot8_i32_i4
#else
#pragma unroll
    for (int j = 0; j < 8; ++j) c += ((a << (28 - 4 * j)) >> 28) * ((b << (28 - 4 * j)) >> 28);
    return c;
#endif
}
// Without this the scheduler hoists every token block's s_loads to the top of the step, which needs more
// SGPRs than a wave has and spills them through v_writelane/v_readlane.
static __device__ __forceinline__ void rq_block_fence() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_sched_barrier(0);
#endif
}
// wave index in an SGPR, so per-wave K offsets keep the activation loads scalar
static __device__ __forceinline__ int rq_wave_id() {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_readfirstlane(threadIdx.x / RQ_WAVE);
#else
    return threadIdx.x / RQ_WAVE;
#endif
}
#else
#define RQ_WAVE 32
static __device__ __forceinline__ int rq_dot8(int a, int b, int c) {
#pragma unroll
    for (int j = 0; j < 8; ++j) c += ((a << (28 - 4 * j)) >> 28) * ((b << (28 - 4 * j)) >> 28);
    return c;
}
static __device__ __forceinline__ void rq_block_fence() {}
static __device__ __forceinline__ int rq_wave_id() { return threadIdx.x / RQ_WAVE; }
#endif

// a load that skips the (non-coherent) L1, for data other workgroups just wrote
static __device__ __forceinline__ int32_t rq_load_l2(const int32_t * p) {
#if defined(GGML_USE_HIP)
    return __builtin_nontemporal_load(p);
#else
    return __ldcg(p);
#endif
}

template <int N> struct alignas(16) rq_words {
    int32_t v[N];
};

// Weights, 4 k-words (32 int4) per lane per step: Wt[group][step][lane][4], group = channel / RQ_WAVE
constexpr int RQ_KW = 4;
static __host__ __device__ __forceinline__ size_t rq_wt_index(int channel, int k_word, int Kw) {
    const int group = channel / RQ_WAVE, lane = channel % RQ_WAVE;
    return (((size_t) group * (Kw / RQ_KW) + k_word / RQ_KW) * RQ_WAVE + lane) * RQ_KW + k_word % RQ_KW;
}

// Verify-width presets: R channel groups per wave, KA activation words per token per chunk, TB tokens per
// scalar load. Picked from the gfx906 asm: fewest instructions per dot8 without SGPR spills.
template <int T> struct rq_cfg;
template <> struct rq_cfg<1>  { static constexpr int R = 4, KA = 4, TB = 1; };
template <> struct rq_cfg<2>  { static constexpr int R = 4, KA = 4, TB = 2; };
template <> struct rq_cfg<4>  { static constexpr int R = 4, KA = 4, TB = 4; };
template <> struct rq_cfg<8>  { static constexpr int R = 2, KA = 4, TB = 8; };
template <> struct rq_cfg<16> { static constexpr int R = 2, KA = 4, TB = 16; };
template <> struct rq_cfg<24> { static constexpr int R = 1, KA = 1, TB = 24; };
template <> struct rq_cfg<32> { static constexpr int R = 1, KA = 1, TB = 32; };
template <> struct rq_cfg<48> { static constexpr int R = 1, KA = 1, TB = 24; };
template <> struct rq_cfg<64> { static constexpr int R = 1, KA = 1, TB = 32; };

// Activations, step-major: Ac[chunk][t][KA], chunk = k_word / KA
static __host__ __device__ __forceinline__ size_t rq_act_index(int t, int k_word, int T, int KA) {
    return ((size_t) (k_word / KA) * T + t) * KA + k_word % KA;
}

// Activation groups (ggml-quants.c, rq_quantize_row_a): each G activations of a token (G = 32 << gsh: 256, 128 or 64)
// have a zero point z_g and a scale m_g * u, m_g an integer, so the dot product is the exact integer total =
// sum_g k_b * m_g * (acc_g + (8 - z_g) * colsum_g), k_b the integer multiplier of the weight block (256 inputs)
// holding group g. The spec epilogue, fused:
// y = ((fp32(total) * 16) * u) * s_w into D [M][ldd], for the M real tokens. Each group's term k m S fits int32
// (|S| <= 2^15, k m <= 3825) and goes into the total as term >> RQ_TERM_SHIFT, so the total stays in int32 for any
// input (rotated massive activations against aligned rows passed 2^31 unshifted). With split-K (gridDim.y > 1) the workgroups add into acc [M][N] in L2 (integers: any order is exact) and bump
// cnt[blockIdx.x]; the last one to arrive reads the totals and writes D. acc and cnt start at zero (the activation
// quantizer clears them).
struct rq_epi {
    const float * s_a;      // u per token
    const int32_t * mul;    // m_g | z_g << 8, [K / 256][T]
    const float * sw;
    const int16_t * colsum; // [K / G][N]
    const int8_t * wk;      // weight multipliers k per 64 inputs, [K / 64][N]
    float * D;
    int ldd;
    int M;
    int32_t * cnt;
};

static __device__ __forceinline__ float rq_epi_value(const rq_epi & e, int32_t acc, int t, int n) {
    return __fmul_rn(__fmul_rn(__fmul_rn(__int2float_rn(acc), (float) (1 << RQ_TERM_SHIFT)), e.s_a[t]), e.sw[n]);
}

// One weight matrix of a launch. Up to three matrices that read the same activation (gate and up, ...) share a
// launch, each over its own range of workgroups along x, so a workgroup's job is uniform.
struct rq_job {
    const int32_t * Wt;
    int32_t * acc; // split-K sums [M][N]
    int N;
    int nblk;      // workgroups along x: N / (RQ_WAVE * R)
    rq_epi epi;
};
struct rq_jobs {
    rq_job j[3];
    int n;
    int gsh; // steps (of RQ_KW words) per activation group, log2
};

template <int T, int R> constexpr size_t rq_lds_bytes(int W) {
    return W > 1 ? (size_t) R * T * RQ_WAVE * sizeof(int32_t) : 0;
}

// Ac: activation words (element 8w+j in bits [4j+3:4j], stored as q-8), rq_act_index
// per job: Wt weight words (rq_wt_index), acc [M][N] int32 split-K sums (gridDim.y > 1), epi the fused epilogue.
// grid (sum of the jobs' nblk, ceil(Kw / kw_split)), block RQ_WAVE*W (W <= 8), rq_lds_bytes<T, R>(W) dynamic
// LDS; Kw and kw_split multiples of an activation group (RQ_KW << jobs.gsh words).
// PF: activations one token block ahead, from T = 24 where a block is only 32 dot8s.
template <int T, int R, int KA = rq_cfg<T>::KA, int TB = rq_cfg<T>::TB, bool PF = (T >= 24 && TB * KA <= 32)>
static __global__ void __launch_bounds__(RQ_WAVE * 8)
rq_skinny_kernel(const int32_t * __restrict__ Ac, const rq_jobs jobs, int Kw, int kw_split) {
    static_assert(RQ_KW % KA == 0 && T % TB == 0, "bad preset");
    int bx = blockIdx.x, ji = 0;
    if (bx >= jobs.j[0].nblk) {
        bx -= jobs.j[0].nblk;
        ji = 1;
        if (bx >= jobs.j[1].nblk) {
            bx -= jobs.j[1].nblk;
            ji = 2;
        }
    }
    // field by field with uniform selects: indexing the argument array would copy it to scratch
#define RQ_JOB(f) (ji == 0 ? jobs.j[0].f : ji == 1 ? jobs.j[1].f : jobs.j[2].f)
    const int32_t * __restrict__ Wt = RQ_JOB(Wt);
    int32_t * __restrict__ acc = RQ_JOB(acc);
    const int N = RQ_JOB(N);
    const rq_epi epi = { RQ_JOB(epi.s_a), RQ_JOB(epi.mul), RQ_JOB(epi.sw), RQ_JOB(epi.colsum), RQ_JOB(epi.wk),
                         RQ_JOB(epi.D), RQ_JOB(epi.ldd), RQ_JOB(epi.M), RQ_JOB(epi.cnt) };
#undef RQ_JOB
    constexpr int CHUNKS = RQ_KW / KA;
    constexpr int NB = CHUNKS * (T / TB); // token blocks per step
    const int lane = threadIdx.x % RQ_WAVE;
    const int wave = rq_wave_id(), W = blockDim.x / RQ_WAVE;
    const int g0 = bx * R;
    const int wg_begin = blockIdx.y * kw_split;
    const int wg_steps = (min(wg_begin + kw_split, Kw) - wg_begin) / RQ_KW;
    const int per_wave = (((wg_steps >> jobs.gsh) + W - 1) / W) << jobs.gsh; // whole activation groups
    const int s_lo = min(wave * per_wave, wg_steps);
    const int steps = min(s_lo + per_wave, wg_steps) - s_lo;
    const int w_begin = wg_begin + s_lo * RQ_KW;

    const rq_words<RQ_KW> * wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wp[r] = reinterpret_cast<const rq_words<RQ_KW> *>(Wt) +
                ((size_t) (g0 + r) * (Kw / RQ_KW) + w_begin / RQ_KW) * RQ_WAVE + lane;
    }
    const rq_words<TB * KA> * ap = reinterpret_cast<const rq_words<TB * KA> *>(Ac) + (size_t) (w_begin / KA) * (T / TB);

    // sum: the current activation group's code products; tot: the groups so far, each times its multipliers.
    // Every factor fits 24 bits, so the products are __mul24 (gfx906: v_mul_lo_u32 is quarter rate).
    int sum[R][T], tot[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) sum[r][t] = tot[r][t] = 0;
    const int gstep0 = w_begin / RQ_KW; // a group is 1 << gsh steps
    const int gsh = jobs.gsh, gmask = (1 << gsh) - 1;
    int kw[R]; // the current weight block's multiplier, per channel group (a block is 8 steps)
#pragma unroll
    for (int r = 0; r < R; ++r) kw[r] = steps > 0 ? epi.wk[(size_t) (gstep0 >> 1) * N + (g0 + r) * RQ_WAVE + lane] : 0;

    // software pipeline: the next step's weights are in flight while this step's dot8s run
    rq_words<RQ_KW> cur[R], nxt[R];
    if (steps > 0)
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = wp[r][0];

    const int nblocks = steps * NB;
    rq_words<TB * KA> a_cur;
    if (PF && steps > 0) a_cur = ap[0];

    for (int s = 0; s < steps; ++s) {
        if (((gstep0 + s) & gmask) == 0) { // a group starts: its zero-point term
            const int g = (gstep0 + s) >> gsh;
            const int32_t * mz = epi.mul + (size_t) g * T;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int cs = epi.colsum[(size_t) g * N + (g0 + r) * RQ_WAVE + lane];
                kw[r] = epi.wk[(size_t) ((gstep0 + s) >> 1) * N + (g0 + r) * RQ_WAVE + lane];
#pragma unroll
                for (int t = 0; t < T; ++t) sum[r][t] = __mul24(8 - (mz[t] >> 8), cs);
            }
        }
        const int s_next = s + 1 < steps ? s + 1 : s; // last step re-reads, no branch
#pragma unroll
        for (int r = 0; r < R; ++r) nxt[r] = wp[r][(size_t) s_next * RQ_WAVE];

#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int c = q / (T / TB), b = q % (T / TB);
            const int qi = s * NB + q;
            rq_words<TB * KA> a, a_nxt;
            if (PF) {
                a = a_cur;
                sum[0][b * TB] = rq_dot8(a.v[0], cur[0].v[c * KA], sum[0][b * TB]);
                rq_block_fence();
                a_nxt = ap[qi + 1 < nblocks ? qi + 1 : qi]; // last block re-reads, no branch
            } else {
                a = ap[qi];
            }
#pragma unroll
            for (int tt = 0; tt < TB; ++tt)
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int j = 0; j < KA; ++j) {
                        if (PF && tt == 0 && r == 0 && j == 0) continue; // done above
                        const int t = b * TB + tt;
                        sum[r][t] = rq_dot8(a.v[tt * KA + j], cur[r].v[c * KA + j], sum[r][t]);
                    }
            rq_block_fence();
            if (PF) a_cur = a_nxt;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = nxt[r];
        if (s == steps - 1 || ((gstep0 + s + 1) & gmask) == 0) { // a group ends here (or this wave's part of it)
            const int32_t * mg = epi.mul + (size_t) ((gstep0 + s) >> gsh) * T;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int m = mg[t] & 0xff;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    // full-rate 24-bit ops: k * m <= 15 * 255, |sum| <= 2^15
                    tot[r][t] += __mul24(__mul24(kw[r], m), sum[r][t]) >> RQ_TERM_SHIFT;
                    sum[r][t] = 0;
                }
            }
        }
    }

    // reduce the workgroup's K slices in one LDS buffer: wave 0 stores, the other waves add atomically
    extern __shared__ int32_t rq_lds[];
    if (W > 1) {
        if (wave == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int t = 0; t < T; ++t) rq_lds[(r * T + t) * RQ_WAVE + lane] = tot[r][t];
        __syncthreads();
        if (wave > 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int t = 0; t < T; ++t) atomicAdd(&rq_lds[(r * T + t) * RQ_WAVE + lane], tot[r][t]);
        __syncthreads();
        if (wave == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int t = 0; t < T; ++t) tot[r][t] = rq_lds[(r * T + t) * RQ_WAVE + lane];
    }
    if (wave != 0) return;
    if (gridDim.y == 1) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int n = (g0 + r) * RQ_WAVE + lane;
                if (t < epi.M) epi.D[(size_t) t * epi.ldd + n] = rq_epi_value(epi, tot[r][t], t, n);
            }
        return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (t < epi.M) atomicAdd(&acc[(size_t) t * N + (g0 + r) * RQ_WAVE + lane], tot[r][t]);
        }
    __threadfence();
    int prev = 0;
    if (lane == 0) prev = atomicAdd(&epi.cnt[bx], 1);
    prev = __shfl_sync(0xffffffff, prev, 0, RQ_WAVE);
    if (prev != (int) gridDim.y - 1) return;
    __threadfence();
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int n = (g0 + r) * RQ_WAVE + lane;
            if (t < epi.M) epi.D[(size_t) t * epi.ldd + n] = rq_epi_value(epi, rq_load_l2(&acc[(size_t) t * N + n]), t, n);
        }
}

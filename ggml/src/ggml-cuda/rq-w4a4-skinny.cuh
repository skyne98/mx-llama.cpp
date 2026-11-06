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
    int rot; // RQ_SKINNY_ROT: each wave starts its K slice at a different whole group (0: in order)
};

template <int T, int R> constexpr size_t rq_lds_bytes(int W) {
    return W > 1 ? (size_t) R * T * RQ_WAVE * sizeof(int32_t) : 0;
}
// stream-K: one sum buffer for each of the (at most) two tiles a workgroup touches
template <int T, int R> constexpr size_t rq_sk_lds_bytes() {
    return 2 * (size_t) R * T * RQ_WAVE * sizeof(int32_t);
}

// The job of launch tile bx (a workgroup's R channel groups along x), with bx made relative to it. Its fields by
// uniform selects in a macro: indexing the argument array, or taking its address (a reference parameter), copies the
// whole argument to scratch.
struct rq_job_view {
    const int32_t * Wt;
    int32_t * acc;
    int N;
    rq_epi epi;
};
static __device__ __forceinline__ int rq_job_index(int nblk0, int nblk1, int & bx) {
    if (bx < nblk0) {
        return 0;
    }
    bx -= nblk0;
    if (bx < nblk1) {
        return 1;
    }
    bx -= nblk1;
    return 2;
}
#define RQ_JOB(jobs, ji, f) ((ji) == 0 ? (jobs).j[0].f : (ji) == 1 ? (jobs).j[1].f : (jobs).j[2].f)
#define RQ_JOB_VIEW(jobs, ji)                                                                                            \
    rq_job_view { RQ_JOB(jobs, ji, Wt), RQ_JOB(jobs, ji, acc), RQ_JOB(jobs, ji, N),                                     \
                  { RQ_JOB(jobs, ji, epi.s_a), RQ_JOB(jobs, ji, epi.mul), RQ_JOB(jobs, ji, epi.sw),                     \
                    RQ_JOB(jobs, ji, epi.colsum), RQ_JOB(jobs, ji, epi.wk), RQ_JOB(jobs, ji, epi.D),                    \
                    RQ_JOB(jobs, ji, epi.ldd), RQ_JOB(jobs, ji, epi.M), RQ_JOB(jobs, ji, epi.cnt) } }

// One wave's integer totals tot[r][t] for channel groups g0 .. g0 + R - 1 over `steps` steps of K from word w_begin,
// a whole number of activation groups: each group's code products times its multipliers, shifted (RQ_TERM_SHIFT).
// The steps go in rotated order, from step `phase` (a whole group) round to phase - 1: the sum doesn't depend on the
// order, and waves that start at different offsets keep their weight streams off the same DRAM channels.
template <int T, int R, int KA, int TB, bool PF>
static __device__ __forceinline__ void rq_skinny_dot(const int32_t * __restrict__ Ac, const int32_t * __restrict__ Wt,
                                                     const rq_epi & epi, int N, int Kw, int g0, int w_begin, int steps,
                                                     int gsh, int phase, int (&tot)[R][T]) {
    static_assert(RQ_KW % KA == 0 && T % TB == 0, "bad preset");
    constexpr int CHUNKS = RQ_KW / KA;
    constexpr int NB = CHUNKS * (T / TB); // token blocks per step
    const int lane = threadIdx.x % RQ_WAVE;

    const rq_words<RQ_KW> * wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wp[r] = reinterpret_cast<const rq_words<RQ_KW> *>(Wt) +
                ((size_t) (g0 + r) * (Kw / RQ_KW) + w_begin / RQ_KW) * RQ_WAVE + lane;
    }
    const rq_words<TB * KA> * ap = reinterpret_cast<const rq_words<TB * KA> *>(Ac) + (size_t) (w_begin / KA) * (T / TB);

    // sum: the current activation group's code products; tot: the groups so far, each times its multipliers.
    // Every factor fits 24 bits, so the products are __mul24 (gfx906: v_mul_lo_u32 is quarter rate).
    int sum[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) sum[r][t] = tot[r][t] = 0;
    const int gstep0 = w_begin / RQ_KW; // a group is 1 << gsh steps
    const int gmask = (1 << gsh) - 1;
    int kw[R]; // the current weight block's multiplier, per channel group (a block is 8 steps)
#pragma unroll
    for (int r = 0; r < R; ++r) kw[r] = steps > 0 ? epi.wk[(size_t) (gstep0 >> 1) * N + (g0 + r) * RQ_WAVE + lane] : 0;

    auto rot = [&](int s) {
        const int t = s + phase;
        return t < steps ? t : t - steps;
    };
    // software pipeline: the next step's weights are in flight while this step's dot8s run
    rq_words<RQ_KW> cur[R], nxt[R];
    if (steps > 0)
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = wp[r][(size_t) phase * RQ_WAVE];

    rq_words<TB * KA> a_cur;
    if (PF && steps > 0) a_cur = ap[(size_t) phase * NB];

    for (int s = 0; s < steps; ++s) {
        const int st = rot(s), st_next = rot(s + 1 < steps ? s + 1 : s); // the last step re-reads, no branch
        // a group starts: its zero-point term. Loading the next group's colsum and multiplier a group ahead instead
        // costs 4 VGPRs at T = 8 (64 -> 68: 3 waves per SIMD instead of 4) and made every shape 23-28% slower.
        if (((gstep0 + st) & gmask) == 0) {
            const int g = (gstep0 + st) >> gsh;
            const int32_t * mz = epi.mul + (size_t) g * T;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int cs = epi.colsum[(size_t) g * N + (g0 + r) * RQ_WAVE + lane];
                kw[r] = epi.wk[(size_t) ((gstep0 + st) >> 1) * N + (g0 + r) * RQ_WAVE + lane];
#pragma unroll
                for (int t = 0; t < T; ++t) sum[r][t] = __mul24(8 - (mz[t] >> 8), cs);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) nxt[r] = wp[r][(size_t) st_next * RQ_WAVE];

#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int c = q / (T / TB), b = q % (T / TB);
            const int qi = st * NB + q;
            rq_words<TB * KA> a, a_nxt;
            if (PF) {
                a = a_cur;
                sum[0][b * TB] = rq_dot8(a.v[0], cur[0].v[c * KA], sum[0][b * TB]);
                rq_block_fence();
                a_nxt = ap[q + 1 < NB ? qi + 1 : st_next * NB]; // the next step's first block
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
        if (s == steps - 1 || ((gstep0 + st + 1) & gmask) == 0) { // a group ends here (or this wave's part of it)
            const int32_t * mg = epi.mul + (size_t) ((gstep0 + st) >> gsh) * T;
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
}

// Ac: activation words (element 8w+j in bits [4j+3:4j], stored as q-8), rq_act_index
// per job: Wt weight words (rq_wt_index), acc [M][N] int32 split-K sums (gridDim.y > 1), epi the fused epilogue.
// grid (sum of the jobs' nblk, ceil(Kw / kw_split)), block RQ_WAVE*W (W <= 8), rq_lds_bytes<T, R>(W) dynamic
// LDS; Kw and kw_split multiples of an activation group (RQ_KW << jobs.gsh words).
// PF: activations one token block ahead, from T = 24 where a block is only 32 dot8s.
template <int T, int R, int KA = rq_cfg<T>::KA, int TB = rq_cfg<T>::TB, bool PF = (T >= 24 && TB * KA <= 32)>
static __global__ void __launch_bounds__(RQ_WAVE * 8)
rq_skinny_kernel(const int32_t * __restrict__ Ac, const rq_jobs jobs, int Kw, int kw_split) {
    const int lane = threadIdx.x % RQ_WAVE;
    const int wave = rq_wave_id(), W = blockDim.x / RQ_WAVE;
    const int wg_begin = blockIdx.y * kw_split;
    const int wg_steps = (min(wg_begin + kw_split, Kw) - wg_begin) / RQ_KW;
    const int per_wave = (((wg_steps >> jobs.gsh) + W - 1) / W) << jobs.gsh; // whole activation groups
    const int s_lo = min(wave * per_wave, wg_steps);
    const int steps = min(s_lo + per_wave, wg_steps) - s_lo;
    const int w_begin = wg_begin + s_lo * RQ_KW;
    // RQ_SKINNY_ROT: this wave's first group, from its index in the grid
    const int ngw = steps >> jobs.gsh;
    const int phase = jobs.rot != 0 && ngw > 0 ?
        (int) (((blockIdx.y * gridDim.x + blockIdx.x) * W + wave) * (unsigned) jobs.rot % ngw) << jobs.gsh : 0;
    int bx = blockIdx.x;
    const int ji = rq_job_index(jobs.j[0].nblk, jobs.j[1].nblk, bx);
    const rq_job_view job = RQ_JOB_VIEW(jobs, ji);
    int tot[R][T];
    rq_skinny_dot<T, R, KA, TB, PF>(Ac, job.Wt, job.epi, job.N, Kw, bx * R, w_begin, steps, jobs.gsh, phase, tot);
    int32_t * __restrict__ acc = job.acc;
    const int N = job.N;
    const rq_epi & epi = job.epi;
    const int g0 = bx * R;

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

// Stream-K (env RQ_SKINNY_SK=1, off by default). The launch's work, its tiles (R channel groups of one job) times their
// activation groups, is split evenly over the grid (as many workgroups as are resident, or a multiple), and each
// workgroup's share evenly over its waves. A share is at most one tile's K (gridDim.x >= the launch's tiles), so a
// workgroup touches at most two tiles: its waves add their totals into two LDS buffers, then waves 0 and 1 add those
// into acc and count the groups in cnt. The workgroup that completes a tile's ngroups writes D; one that covers a
// whole tile writes D directly.
// Measured on MI50 (REPORT-2026-11-02.md, "stream-K for the skinny kernel"): at the same grid size it times like the
// split-K kernel above, and the verify forward did not get faster at any grid size. Empty slots are not what limits
// these launches; shares that are a power-of-two number of activation groups line the waves' streams up on the same
// DRAM channels and run up to 2x slower.
// Integer adds in any order: bit-exact with the CPU reference, as the split-K path.
// grid (nwg >= tiles), block RQ_WAVE*W (W <= 8), rq_sk_lds_bytes<T, R>() dynamic LDS; acc and cnt start at zero.
template <int T, int R, int KA = rq_cfg<T>::KA, int TB = rq_cfg<T>::TB, bool PF = (T >= 24 && TB * KA <= 32)>
static __global__ void __launch_bounds__(RQ_WAVE * 8)
rq_skinny_sk_kernel(const int32_t * __restrict__ Ac, const rq_jobs jobs, int Kw, int ngroups) {
    constexpr int NL = R * T * RQ_WAVE; // one tile's sums
    const int lane = threadIdx.x % RQ_WAVE;
    const int wave = rq_wave_id(), W = blockDim.x / RQ_WAVE;
    const int units = (jobs.j[0].nblk + jobs.j[1].nblk + jobs.j[2].nblk) * ngroups;
    const int u0 = (int) ((int64_t) units * blockIdx.x / gridDim.x);
    const int u1 = (int) ((int64_t) units * (blockIdx.x + 1) / gridDim.x);
    const int t0 = u0 / ngroups;
    extern __shared__ int32_t rq_lds[];
    for (int i = threadIdx.x; i < 2 * NL; i += blockDim.x) rq_lds[i] = 0;
    __syncthreads();
    // this wave's groups [v, v1): one segment per tile, two at most
    const int v1 = u0 + (u1 - u0) * (wave + 1) / W;
#pragma unroll 1
    for (int v = u0 + (u1 - u0) * wave / W; v < v1;) {
        const int tile = v / ngroups, ve = min(v1, (tile + 1) * ngroups);
        int bx = tile;
        const int ji = rq_job_index(jobs.j[0].nblk, jobs.j[1].nblk, bx);
        const rq_job_view job = RQ_JOB_VIEW(jobs, ji);
        int tot[R][T];
        rq_skinny_dot<T, R, KA, TB, PF>(Ac, job.Wt, job.epi, job.N, Kw, bx * R, ((v - tile * ngroups) << jobs.gsh) * RQ_KW,
                                        (ve - v) << jobs.gsh, jobs.gsh, 0, tot);
        int32_t * l = rq_lds + (tile - t0) * NL;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < T; ++t) atomicAdd(&l[(r * T + t) * RQ_WAVE + lane], tot[r][t]);
        v = ve;
    }
    __syncthreads();
    if (wave > 1) return;
    const int tile = t0 + wave;
    const int lo = max(u0, tile * ngroups), hi = min(u1, (tile + 1) * ngroups);
    if (lo >= hi) return; // the share ends in its first tile
    int bx = tile;
    const int ji = rq_job_index(jobs.j[0].nblk, jobs.j[1].nblk, bx);
    const rq_job_view job = RQ_JOB_VIEW(jobs, ji);
    const rq_epi & epi = job.epi;
    const int32_t * l = rq_lds + wave * NL;
    const int g0 = bx * R;
    if (hi - lo == ngroups) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int n = (g0 + r) * RQ_WAVE + lane;
                if (t < epi.M) epi.D[(size_t) t * epi.ldd + n] = rq_epi_value(epi, l[(r * T + t) * RQ_WAVE + lane], t, n);
            }
        return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (t < epi.M) atomicAdd(&job.acc[(size_t) t * job.N + (g0 + r) * RQ_WAVE + lane], l[(r * T + t) * RQ_WAVE + lane]);
        }
    __threadfence();
    int prev = 0;
    if (lane == 0) prev = atomicAdd(&epi.cnt[bx], hi - lo);
    prev = __shfl_sync(0xffffffff, prev, 0, RQ_WAVE);
    if (prev + (hi - lo) != ngroups) return;
    __threadfence();
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int n = (g0 + r) * RQ_WAVE + lane;
            if (t < epi.M) epi.D[(size_t) t * epi.ldd + n] = rq_epi_value(epi, rq_load_l2(&job.acc[(size_t) t * job.N + n]), t, n);
        }
}

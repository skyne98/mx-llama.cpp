#include "rq-w4a4.cuh"
#if defined(GGML_USE_HIP)
#pragma clang fp contract(off) // the _rn intrinsics below are plain ops on HIP: keep mul + add unfused
#endif
#include "rq-w4a4-skinny.cuh"
#include "rq-w4a4-gemm.cuh"
#include "rq-w4a4-tc.cuh"

#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" float ggml_rq_act_clip_for(float tensor_clip, bool hadamard); // ggml-quants.c
extern "C" int   ggml_rq_act_group(void);                                // ggml-quants.c
extern "C" int64_t ggml_rq_rot_check(int32_t n, int64_t k);              // ggml-quants.c

// ruquant W4A4 on CUDA and HIP. Numerics follow the CPU reference in ggml-quants.c op for op (explicit _rn
// intrinsics, same Hadamard summation order, exact integer accumulation over the activation groups), so CPU and GPU
// give identical outputs. On NVIDIA GPUs with INT4 tensor cores the GEMM is rq-w4a4-tc.cuh at every width; elsewhere
// the skinny kernel (rq-w4a4-skinny.cuh) at verify widths and the tiled one (rq-w4a4-gemm.cuh) at prefill widths.

// same factorization as ggml_rq_hadamard_factor (ggml-quants.c)
static bool rq_factor(int64_t n, int & p, int & m) {
    for (int pp = 1; pp <= 128 && pp <= n; ++pp) {
        if (pp > 1) {
            if (pp % 4 != 0 || (pp - 1) % 4 != 3) continue;
            bool prime = pp - 1 >= 2;
            for (int d = 2; d * d <= pp - 1; ++d) if ((pp - 1) % d == 0) { prime = false; break; }
            if (!prime) continue;
        }
        if (n % pp != 0) continue;
        const int64_t mm = n / pp;
        if (mm & (mm - 1)) continue;
        p = pp;
        m = (int) mm;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------- online Hadamard
// Sylvester over each chunk of m, from src (row stride lds) into x (row stride ldx), then x *= scale if there is
// no Paley stage
static __global__ void rq_fwht_kernel(const float * __restrict__ src, int lds, float * __restrict__ x, int ldx, int m, int p,
                                      float scale) {
    extern __shared__ float buf[];
    const float * in = src + (size_t) blockIdx.x * lds + (size_t) blockIdx.y * m;
    float * row = x + (size_t) blockIdx.x * ldx + (size_t) blockIdx.y * m;
    for (int i = threadIdx.x; i < m; i += blockDim.x) buf[i] = in[i];
    for (int h = 1; h < m; h <<= 1) {
        __syncthreads();
        for (int t = threadIdx.x; t < m / 2; t += blockDim.x) {
            const int j = (t / h) * 2 * h + (t % h);
            const float a = buf[j], b = buf[j + h];
            buf[j]     = __fadd_rn(a, b);
            buf[j + h] = __fsub_rn(a, b);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) row[i] = p == 1 ? __fmul_rn(buf[i], scale) : buf[i];
}

// Paley across the p chunks: y[a] = sum_i x[i] A[i][a], i ascending from 0, then *= scale.
// A = I + S with S[0][j>0] = 1, S[i>0][0] = -1, S[i][j] = chi(j - i mod q). One thread per column j.
static __global__ void rq_paley_kernel(float * __restrict__ x, int ldx, int m, int p, float scale) {
    __shared__ int8_t chi[128];
    const int q = p - 1;
    for (int i = threadIdx.x; i < q; i += blockDim.x) chi[i] = i == 0 ? 0 : -1;
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 1; i < q; ++i) chi[(i * i) % q] = 1;
    __syncthreads();
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= m) return;
    float * row = x + (size_t) blockIdx.x * ldx;
    float col[128];
    for (int i = 0; i < p; ++i) col[i] = row[(size_t) i * m + j];
    for (int a = 0; a < p; ++a) {
        float acc = 0.0f;
        for (int i = 0; i < p; ++i) {
            const int s = i == 0 ? (a == 0 ? 0 : 1) : (a == 0 ? -1 : chi[((a - i) % q + q) % q]);
            const int e = (i == a ? 1 : 0) + s;
            acc = e > 0 ? __fadd_rn(acc, col[i]) : __fsub_rn(acc, col[i]);
        }
        row[(size_t) a * m + j] = __fmul_rn(acc, scale);
    }
}

// The same, spread out: a block holds 64 columns of all P chunks in LDS and each of its 4 waves sums one output
// row over i ascending, so the bits match. Row a's signs come from rq_paley_masks (bit i set: subtract term i), and
// a subtraction is an add of the sign-flipped value (a + (-b) is exactly a - b), so a term is an XOR and an add:
// recomputing chi((a - i) mod q) per term made this VALU-bound (17408 = 68 x 256 at 6 tokens: 26 us). Out of place:
// other blocks still read the rows a block writes. grid (M, m / 64, P / 4), 256 threads; m a multiple of 64.
template <int P>
static __global__ void __launch_bounds__(256) rq_paley_rows_kernel(const float * __restrict__ x, float * __restrict__ y,
                                                                   const uint32_t * __restrict__ masks, int ldx, int m,
                                                                   float scale) {
    static_assert(P % 4 == 0 && P <= 96, "four output rows per block, three mask words per row");
    __shared__ float tile[P][64];
    const int t = threadIdx.x;
    const size_t off = (size_t) blockIdx.x * ldx + blockIdx.y * 64;
    const int jl = t % 64;
    // a constant trip count, so all P / 4 loads are in flight at once
#pragma unroll
    for (int r = 0; r < P / 4; ++r) tile[r * 4 + t / 64][jl] = x[off + (size_t) (r * 4 + t / 64) * m + jl];
    __syncthreads();
    int a = blockIdx.z * 4 + t / 64;
#if defined(GGML_USE_HIP)
    a = __builtin_amdgcn_readfirstlane(a); // wave-uniform, so the masks come in as scalar loads
#endif
    const uint32_t mw[3] = { masks[3 * a], masks[3 * a + 1], masks[3 * a + 2] };
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < P; ++i) {
        const uint32_t flip = ((mw[i / 32] >> (i % 32)) & 1u) << 31;
        acc = acc + __uint_as_float(__float_as_uint(tile[i][jl]) ^ flip);
    }
    y[off + (size_t) a * m + jl] = __fmul_rn(acc, scale);
}

// Sign masks of the Paley matrix above, three words per row (P <= 96), built once per device.
static const uint32_t * rq_paley_masks(int p) {
    static std::mutex mutex;
    static std::map<std::pair<int, int>, uint32_t *> tables;
    const int device = ggml_cuda_get_device();
    std::lock_guard<std::mutex> lock(mutex);
    uint32_t *& d = tables[{ device, p }];
    if (d == nullptr) {
        const int q = p - 1;
        std::vector<int> chi(q, -1);
        chi[0] = 0;
        for (int i = 1; i < q; ++i) chi[(i * i) % q] = 1;
        std::vector<uint32_t> h(3 * p, 0);
        for (int a = 0; a < p; ++a) {
            for (int i = 0; i < p; ++i) {
                const int s = i == 0 ? (a == 0 ? 0 : 1) : (a == 0 ? -1 : chi[((a - i) % q + q) % q]);
                if ((i == a ? 1 : 0) + s <= 0) h[3 * a + i / 32] |= 1u << (i % 32);
            }
        }
        CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    return d;
}

// ---------------------------------------------------------------------------- activation quantization
// one block per token, op for op as rq_quantize_row_a in ggml-quants.c: asymmetric, per group of G (256, 128 or 64)
// a zero point z_g and a scale that is an integer multiple m_g of the token's unit u. aq: codes (q - 8), token-major
// [M, K/8] (KA = 0) or the skinny kernel's step-major rq_act_index layout for Tp tokens (KA > 0); s_a[m] = u;
// mz[g * Tp + m] = m_g | z_g << 8.
static constexpr int RQ_QUANT_THREADS = 256;
static constexpr int RQ_AG_M = 255;       // as in ggml-quants.c
static constexpr int RQ_MAX_BLOCKS = 128; // K <= 32768
static constexpr int RQ_MAX_GROUPS = RQ_MAX_BLOCKS * QK_W4A4 / 64;

// One token's row xr: the per-token unit, the groups' multiples and zero points, and the codes (body of the kernel below).
template <int THREADS>
static __device__ void rq_quantize_row(const float * __restrict__ xr, int K, int G, float clip, int32_t * __restrict__ aq,
                                       float * __restrict__ s_a, int32_t * __restrict__ mz, int Tp, int KA, int m) {
    constexpr int NW = THREADS / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
    const int nb = K / QK_W4A4, L = G / 8; // lanes per group
    __shared__ float s_lo[RQ_MAX_GROUPS], s_hi[RQ_MAX_GROUPS];
    __shared__ float s_wmax[NW];
    // a warp per block of 256, a lane per code word (8 values), L lanes per group
    float wmax = 0.f;
    for (int b = wid; b < nb; b += NW) {
        const float4 * p = reinterpret_cast<const float4 *>(xr + b * QK_W4A4 + 8 * lane);
        const float4 v0 = p[0], v1 = p[1];
        float mn = fminf(fminf(fminf(fminf(v0.x, v0.y), fminf(v0.z, v0.w)), fminf(fminf(v1.x, v1.y), fminf(v1.z, v1.w))), 0.f);
        float mx = fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v0.y), fmaxf(v0.z, v0.w)), fmaxf(fmaxf(v1.x, v1.y), fmaxf(v1.z, v1.w))), 0.f);
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            if (off < L) {
                mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, off, WARP_SIZE));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, WARP_SIZE));
            }
        }
        const float lo = __fmul_rn(mn, clip), hi = __fmul_rn(mx, clip);
        if (lane % L == 0) { s_lo[(b * QK_W4A4 + 8 * lane) / G] = lo; s_hi[(b * QK_W4A4 + 8 * lane) / G] = hi; }
        wmax = fmaxf(wmax, __fsub_rn(hi, lo));
    }
    wmax = warp_reduce_max(wmax);
    if (lane == 0) s_wmax[wid] = wmax;
    __syncthreads();
    float R = s_wmax[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) R = fmaxf(R, s_wmax[i]);
    if (R == 0.f) R = 2.f;
    const float u = __fdiv_rn(R, 15.f * RQ_AG_M);
    if (threadIdx.x == 0) s_a[m] = u;
    for (int b = wid; b < nb; b += NW) {
        const int g = (b * QK_W4A4 + 8 * lane) / G;
        const float lo = s_lo[g], hi = s_hi[g];
        const int mg = (int) fminf(fmaxf(ceilf(__fdiv_rn(__fmul_rn(__fsub_rn(hi, lo), (float) RQ_AG_M), R)), 1.f), (float) RQ_AG_M);
        const float inv = __fdiv_rn(1.f, __fmul_rn((float) mg, u));
        const float zf  = fminf(fmaxf(rintf(__fmul_rn(-lo, inv)), 0.f), 15.f);
        if (lane % L == 0) mz[(size_t) g * Tp + m] = mg | (int) zf << 8;
        const float4 * p = reinterpret_cast<const float4 *>(xr + b * QK_W4A4 + 8 * lane);
        const float4 v0 = p[0], v1 = p[1];
        const float f[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
        uint32_t word = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float q = fminf(fmaxf(__fadd_rn(rintf(__fmul_rn(f[j], inv)), zf), 0.f), 15.f);
            word |= (uint32_t) (((int) q - 8) & 0xF) << (4 * j);
        }
        const int w = b * (QK_W4A4 / 8) + lane;
        aq[KA ? rq_act_index(m, w, Tp, KA) : (size_t) m * (K / 8) + w] = (int32_t) word;
    }
}

// the blocks past the tokens clear the skinny kernel's split-K sums and counters, 16 ints per thread
template <int THREADS>
static __device__ void rq_zero_block(int32_t * __restrict__ zero, int64_t n_zero, int b) {
    const int64_t i0 = ((int64_t) b * THREADS + threadIdx.x) * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) if (i0 + i < n_zero) zero[i0 + i] = 0;
}

static __global__ void __launch_bounds__(RQ_QUANT_THREADS)
rq_quantize_act_kernel(const float * __restrict__ x, int ldx, int K, int G, float clip, int32_t * __restrict__ aq,
                       float * __restrict__ s_a, int32_t * __restrict__ mz, int Tp, int KA,
                       int32_t * __restrict__ zero, int64_t n_zero, int M) {
    const int m = blockIdx.x;
    if (m >= M) {
        rq_zero_block<RQ_QUANT_THREADS>(zero, n_zero, m - M);
        return;
    }
    rq_quantize_row<RQ_QUANT_THREADS>(x + (size_t) m * ldx, K, G, clip, aq, s_a, mz, Tp, KA, m);
}

// Verify widths, W4A4H inputs: the online Hadamard and the quantizer in two parallel launches instead of three.
// rq_hadamard_stats_kernel<P, MW>: grid (M * K / n_rot, P / 4), 256 threads. A block holds a whole rotation block of
// n_rot = P * m (m = 64 * MW) in LDS: Sylvester over all P chunks (the butterflies of rq_fwht_kernel, stage for
// stage), then each wave one output row of Paley as rq_paley_rows_kernel (i ascending, the same sign masks). It writes
// the rotated values to y, and per activation group of 64 the clipped range ends lo / hi as rq_quantize_row computes
// them. Every block of a rotation block redoes the cheap Sylvester part so that the Paley part, the expensive one,
// stays spread over P / 4 blocks per row (one block per token was 4x slower).

// v from lane ^ H (H < 16) within the row of 16 lanes, through DPP: a VALU operand, no LDS round trip
template <int H>
static __device__ __forceinline__ float rq_row_xor(float v, int t) {
    static_assert(H == 1 || H == 2 || H == 4 || H == 8, "within a DPP row");
#if defined(GGML_USE_HIP)
    const int i = __float_as_int(v);
    if constexpr (H == 1) {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, i, 0xB1, 0xF, 0xF, false)); // quad_perm [1, 0, 3, 2]
    } else if constexpr (H == 2) {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, i, 0x4E, 0xF, 0xF, false)); // quad_perm [2, 3, 0, 1]
    } else if constexpr (H == 4) { // rotations by 4 and 12 within the row, one of them is t ^ 4
        const int r4 = __builtin_amdgcn_update_dpp(0, i, 0x124, 0xF, 0xF, false), r12 = __builtin_amdgcn_update_dpp(0, i, 0x12C, 0xF, 0xF, false);
        return __int_as_float((t & 4) ? r4 : r12);
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, i, 0x128, 0xF, 0xF, false)); // row_ror:8
    }
#else
    GGML_UNUSED(t);
    return __shfl_xor_sync(0xffffffff, v, H, 16);
#endif
}

// one butterfly stage H < 16 over all slots: lower lane a + b, upper lane a - b
template <int H, int NP, int Q>
static __device__ __forceinline__ void rq_fwht_row_stage(float (&v)[NP][Q], int t) {
#pragma unroll
    for (int s = 0; s < NP; ++s) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float o = rq_row_xor<H>(v[s][q], t);
            v[s][q] = (t & H) ? __fsub_rn(o, v[s][q]) : __fadd_rn(v[s][q], o);
        }
    }
}

template <int P, int MW>
static __global__ void __launch_bounds__(256)
rq_hadamard_stats_kernel(const float * __restrict__ x, int ldx, float * __restrict__ y, int K, const uint32_t * __restrict__ masks,
                         float scale, float clip, float2 * __restrict__ lohi) {
    static_assert(P % 4 == 0 && P <= 96 && (MW == 1 || MW == 2), "four output rows per block, chunks of 64 or 128");
    constexpr int m = 64 * MW, n_rot = P * m;
    __shared__ float tile[P][m];
    const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    const int rows_per_tok = K / n_rot;
    const int tok = blockIdx.x / rows_per_tok, rb = blockIdx.x % rows_per_tok;
    const float * in = x + (size_t) tok * ldx + (size_t) rb * n_rot;
    // Sylvester: a row of 16 lanes holds a chunk, its lane t the values t + 16 q (q < Q), so stages 1 to 8 cross the
    // row through DPP and stages 16 and up stay in the lane's registers. Slot-major, a wave per chunk, it waited for
    // each chunk's load and six dependent ds_bpermutes (42.7 us vs 13.6 for the Paley part alone, 1 GPU down);
    // stage-major, 28.6 us (two ds_bpermutes in flight); DPP for stages 1 to 8 only, 23.8 us.
    constexpr int Q = m / 16, NP = (P + 15) / 16; // values per lane, passes of 16 chunks over the block
    const int t = threadIdx.x % 16, slot = threadIdx.x / 16;
    float v[NP][Q];
#pragma unroll
    for (int s = 0; s < NP; ++s) {
        const int c = 16 * s + slot;
#pragma unroll
        for (int q = 0; q < Q; ++q) v[s][q] = c < P ? in[c * m + 16 * q + t] : 0.0f;
    }
    rq_fwht_row_stage<1>(v, t);
    rq_fwht_row_stage<2>(v, t);
    rq_fwht_row_stage<4>(v, t);
    rq_fwht_row_stage<8>(v, t);
#pragma unroll
    for (int s = 0; s < NP; ++s) {
#pragma unroll
        for (int hq = 1; hq < Q; hq <<= 1) { // stage 16 hq: values q and q + hq
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                if (q & hq) continue;
                const float a = v[s][q], b = v[s][q + hq];
                v[s][q]      = __fadd_rn(a, b);
                v[s][q + hq] = __fsub_rn(a, b);
            }
        }
        const int c = 16 * s + slot;
        if (c < P) {
#pragma unroll
            for (int q = 0; q < Q; ++q) tile[c][16 * q + t] = v[s][q];
        }
    }
    __syncthreads();
    int a = blockIdx.y * 4 + wave;
#if defined(GGML_USE_HIP)
    a = __builtin_amdgcn_readfirstlane(a); // wave-uniform, so the masks come in as scalar loads
#endif
    const uint32_t mw[3] = { masks[3 * a], masks[3 * a + 1], masks[3 * a + 2] };
    float * out = y + (size_t) tok * K + (size_t) rb * n_rot + (size_t) a * m;
#pragma unroll
    for (int k = 0; k < MW; ++k) {
        const int j = 64 * k + lane;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            const uint32_t flip = ((mw[i / 32] >> (i % 32)) & 1u) << 31;
            acc = acc + __uint_as_float(__float_as_uint(tile[i][j]) ^ flip);
        }
        const float r = __fmul_rn(acc, scale);
        out[j] = r;
        // this wave's 64 values are one activation group (groups of 64 start at multiples of 64 along K)
        float mn = fminf(r, 0.f), mx = fmaxf(r, 0.f);
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, off, 64));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, 64));
        }
        if (lane == 0) {
            const int g = (rb * n_rot + a * m + 64 * k) / 64;
            lohi[(size_t) tok * (K / 64) + g] = make_float2(__fmul_rn(mn, clip), __fmul_rn(mx, clip));
        }
    }
}

// The codes of rq_hadamard_stats_kernel's output: grid (M + zero blocks, ceil(K / 64 / RQ_CODE_GROUPS)), 256 threads. A
// block reduces its token's group ranges to the row range R, then quantizes RQ_CODE_GROUPS groups with
// rq_quantize_row's arithmetic (activation groups of 64 only). Blocks past the tokens clear the split-K sums.
static constexpr int RQ_CODE_GROUPS = 32; // 256 threads, a code word (8 values) per thread
static __global__ void __launch_bounds__(256)
rq_codes_kernel(const float * __restrict__ y, const float2 * __restrict__ lohi, int K, int32_t * __restrict__ aq,
                float * __restrict__ s_a, int32_t * __restrict__ mz, int Tp, int KA, int32_t * __restrict__ zero, int64_t n_zero,
                int M) {
    const int tok = blockIdx.x;
    if (tok >= M) {
        if (blockIdx.y == 0) {
            rq_zero_block<256>(zero, n_zero, tok - M);
        }
        return;
    }
    const int ng = K / 64;
    const float2 * lh = lohi + (size_t) tok * ng;
    float wmax = 0.f;
    for (int g = threadIdx.x; g < ng; g += 256) wmax = fmaxf(wmax, __fsub_rn(lh[g].y, lh[g].x));
    __shared__ float s_w[256 / WARP_SIZE];
    wmax = warp_reduce_max(wmax);
    if (threadIdx.x % WARP_SIZE == 0) s_w[threadIdx.x / WARP_SIZE] = wmax;
    __syncthreads();
    float R = s_w[0];
#pragma unroll
    for (int i = 1; i < 256 / WARP_SIZE; ++i) R = fmaxf(R, s_w[i]);
    if (R == 0.f) R = 2.f;
    const float u = __fdiv_rn(R, 15.f * RQ_AG_M);
    if (blockIdx.y == 0 && threadIdx.x == 0) s_a[tok] = u;
    const int g = blockIdx.y * RQ_CODE_GROUPS + threadIdx.x / 8;
    if (g >= ng) {
        return;
    }
    const float lo = lh[g].x, hi = lh[g].y;
    const int mg = (int) fminf(fmaxf(ceilf(__fdiv_rn(__fmul_rn(__fsub_rn(hi, lo), (float) RQ_AG_M), R)), 1.f), (float) RQ_AG_M);
    const float inv = __fdiv_rn(1.f, __fmul_rn((float) mg, u));
    const float zf  = fminf(fmaxf(rintf(__fmul_rn(-lo, inv)), 0.f), 15.f);
    if (threadIdx.x % 8 == 0) mz[(size_t) g * Tp + tok] = mg | (int) zf << 8;
    const int w = g * 8 + threadIdx.x % 8; // code word: values 8w .. 8w + 7
    const float4 * p = reinterpret_cast<const float4 *>(y + (size_t) tok * K + 8 * w);
    const float4 v0 = p[0], v1 = p[1];
    const float f[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
    uint32_t word = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float q = fminf(fmaxf(__fadd_rn(rintf(__fmul_rn(f[j], inv)), zf), 0.f), 15.f);
        word |= (uint32_t) (((int) q - 8) & 0xF) << (4 * j);
    }
    aq[KA ? rq_act_index(tok, w, Tp, KA) : (size_t) tok * (K / 8) + w] = (int32_t) word;
}

// ---------------------------------------------------------------------------- skinny path (gfx906; emulated dot8 on CUDA)
// The weight is repacked in place on first use: [Wt codes, N*K/2 bytes | s_w, N floats | colsum, K/G x N int16,
// per activation group of G | k_b, K/256 x N int16, the blocks' multipliers], which fits in the N*K/256 blocks of
// 144 B. On that first use tensor->extra (unused for non-split buffers) gets the
// tensor's state: whether it is repacked, and its activation clip and Hadamard length from the block headers.
struct rq_tensor_info {
    bool    repacked;
    float   clip; // block 0 header pad[1]; 0 = unset
    int32_t rot;  // block 0 header pad[0]: W4A4H's Hadamard length (the row, or a block of it)
    bool    wg64; // weight multipliers per 64 inputs (block 0 header sum, RQ_W_MULT_G): activation groups of 64 only
    bool    tc;   // repacked for the INT4 tensor cores (rq-w4a4-tc.cuh) instead of the skinny layout
};

// INT4 tensor cores: NVIDIA Turing, Ampere and Ada, activation groups of 64 (one MMA per group); env RQ_TC=0 keeps the
// dot8-shaped kernels (emulated dot8 on CUDA)
static bool rq_tc_ok(const ggml_tensor * src0) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(src0);
    return false;
#else
    static const bool env_on = !getenv("RQ_TC") || atoi(getenv("RQ_TC")) != 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    GGML_UNUSED(src0); // N is a multiple of RQ_WAVE (ggml_cuda_rq_w4a4_supported), so of the 16-channel tile
    return env_on && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= RQ_TC_MIN_CC && cc < RQ_TC_MAX_CC && ggml_rq_act_group() == 64;
#endif
}

static __global__ void rq_repack_kernel(const uint8_t * __restrict__ src, int32_t * __restrict__ wt, float * __restrict__ sw,
                                        int N, int K) {
    const int Kw = K / 8;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) N * Kw) return;
    const int c = (int) (i / Kw), kw = (int) (i % Kw);
    const uint8_t * row = src + (size_t) c * (K / QK_W4A4) * sizeof(block_w4a4);
    wt[rq_wt_index(c, kw, Kw)] = *reinterpret_cast<const int32_t *>(row + (kw / 32) * sizeof(block_w4a4) + 16 + (kw % 32) * 4);
    if (kw == 0) {
        const block_w4a4 * h = reinterpret_cast<const block_w4a4 *>(row);
        sw[c] = h->d;
    }
}

// colsum[g * N + c]: the sum of row c's codes over activation group g (G inputs; |sum| <= 8 G fits int16)
static __global__ void rq_colsum_kernel(const uint8_t * __restrict__ src, int16_t * __restrict__ colsum, int N, int K, int G) {
    const int ng = K / G;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) N * ng) return;
    const int c = (int) (i % N), g = (int) (i / N);
    const block_w4a4 * b = reinterpret_cast<const block_w4a4 *>(src + ((size_t) c * (K / QK_W4A4) + g * G / QK_W4A4) * sizeof(block_w4a4));
    const int w0 = (g * G % QK_W4A4) / 8;
    int sum = 0;
    for (int w = w0; w < w0 + G / 8; ++w)
#pragma unroll
        for (int j = 0; j < 8; ++j) sum += (int32_t) (((b->qs[w] >> (4 * j)) & 0xF) << 28) >> 28;
    colsum[i] = (int16_t) sum;
}

// wk[g * N + c]: row c's scale multiplier for inputs 64 g .. 64 g + 63 (RQ_W_MULT_G; one per block of 256 in
// files without weight groups of 64), int8 so the [K / 64][N] array still fits the blocks' slack
static __global__ void rq_wk_kernel(const uint8_t * __restrict__ src, int8_t * __restrict__ wk, int N, int K) {
    const int nb = K / QK_W4A4;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) N * (K / 64)) return;
    const int c = (int) (i % N), g = (int) (i / N);
    const block_w4a4 * h = reinterpret_cast<const block_w4a4 *>(src + ((size_t) c * nb + g / 4) * sizeof(block_w4a4));
    wk[i] = (int8_t) RQ_W_MULT_G(h->sum, g % 4);
}

static int rq_skinny_tp(int64_t m) {
    for (int p : {1, 2, 4, 8, 16, 24, 32, 48}) if (m <= p) return p;
    return 64;
}

// launch shape of the skinny kernel for Tp tokens: R channel groups per wave, W waves, G split-K workgroups; sk: the
// stream-K kernel instead (G and kw_split unused)
struct rq_skinny_shape {
    int R, KA, W, G, kw_split, groups;
    bool sk;
};

// N_launch: the channels of the whole launch (more than N when several weights share it, see rq_jobs). K is split
// in whole activation groups of gsteps steps: each group's term is shifted (RQ_TERM_SHIFT), so one wave folds it.
static rq_skinny_shape rq_skinny_plan(int tp, int N, int Kw, int N_launch, int gsteps) {
    int R0 = 1, KA = 1;
    switch (tp) {
        case 1:  R0 = rq_cfg<1>::R;  KA = rq_cfg<1>::KA;  break;
        case 2:  R0 = rq_cfg<2>::R;  KA = rq_cfg<2>::KA;  break;
        case 4:  R0 = rq_cfg<4>::R;  KA = rq_cfg<4>::KA;  break;
        case 8:  R0 = rq_cfg<8>::R;  KA = rq_cfg<8>::KA;  break;
        case 16: R0 = rq_cfg<16>::R; KA = rq_cfg<16>::KA; break;
        case 24: R0 = rq_cfg<24>::R; KA = rq_cfg<24>::KA; break;
        case 32: R0 = rq_cfg<32>::R; KA = rq_cfg<32>::KA; break;
        case 48: R0 = rq_cfg<48>::R; KA = rq_cfg<48>::KA; break;
        default: R0 = rq_cfg<64>::R; KA = rq_cfg<64>::KA; break;
    }
    const int g64 = N / RQ_WAVE;
    const int R = R0 >= 4 && g64 % 4 == 0 ? 4 : R0 >= 2 && g64 % 2 == 0 ? 2 : 1;
    const int groups = g64 / R;
    const int steps = Kw / RQ_KW;
    static const int env_w = getenv("RQ_SKINNY_W") ? atoi(getenv("RQ_SKINNY_W")) : 0;
    static const int env_g = getenv("RQ_SKINNY_G") ? atoi(getenv("RQ_SKINNY_G")) : 0;
    // 8 waves with a 2-way split-K. Alone, a 4-way split was within noise of the best for every 27B shape at the verify
    // widths (rig/results/skinny-sweep-t8.txt), but inside a forward 2-way wins: same on one MI50, and the narrower
    // shapes under tensor parallelism get 12% (TP = 2) and 12% (TP = 4) faster T = 8 forwards
    const int W = env_w > 0 ? std::min(env_w, 8) : 8;
    int G = env_g > 0 ? env_g : 2;
    // a launch with this many channel groups fills the device without splitting K, and skips the split-K fixup
    // (MI50, T = 8: gate+up fused, 5120 x 34816, 179 -> 168 us; single weights are within noise either way)
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    if (env_g == 0 && tp <= 16 && N_launch / RQ_WAVE / R >= 4 * nsm) {
        G = 1;
    }
    // nor does one whose channel groups all fit in one round of resident workgroups (2 per CU at 8 waves) when the
    // split would leave each wave 12 to 20 steps: there it only adds the fixup and the cleared sums. That is mlp.down of
    // one GPU under TP = 2 (MI50 8-token forwards 217.5 -> 222.4 t/s). Longer slices (one GPU's mlp.down) gain from
    // the split, and so did the shorter ones of TP = 4 in the forward (292 -> 288 t/s without it), although alone they
    // time faster unsplit. Env RQ_SKINNY_OLD_G restores the split.
    static const bool old_g = getenv("RQ_SKINNY_OLD_G") != nullptr;
    if (env_g == 0 && !old_g && tp <= 16 && N_launch / RQ_WAVE / R <= 2 * nsm && steps >= 12 * 2 * W && steps < 20 * 2 * W) {
        G = 1;
    }
    const int ngroups = steps / gsteps;
    G = std::max(1, std::min(G, ngroups / W));
    // whole groups per wave can make the longest wave longer than an even split of steps would; split K a little
    // more until it is not (K = 5120 at G 64: 5 workgroups of 2 groups per wave, not 4 with up to 3)
    const int G0 = G, even = (steps + W * G0 - 1) / (W * G0);
    auto wave_steps = [&](int g) { return ((ngroups + g - 1) / g + W - 1) / W * gsteps; };
    while (G > 1 && env_g == 0 && G < 2 * G0 && (G + 1) * W <= ngroups && wave_steps(G) > even) {
        G++;
    }
    const int groups_per_wg = (ngroups + G - 1) / G;
    G = (ngroups + groups_per_wg - 1) / groups_per_wg;
    // stream-K (rq_skinny_sk_kernel), env RQ_SKINNY_SK=1
    static const bool sk = getenv("RQ_SKINNY_SK") != nullptr && atoi(getenv("RQ_SKINNY_SK")) != 0;
    return { R, KA, W, G, groups_per_wg * gsteps * RQ_KW, groups, sk };
}

template <int T, int R>
static void rq_skinny_launch(const rq_skinny_shape & sh, const int32_t * ac, const rq_jobs & jobs, int Kw, cudaStream_t stream) {
    int nblk = 0;
    for (int i = 0; i < jobs.n; ++i) nblk += jobs.j[i].nblk;
    if (sh.sk) {
        // the grid: whole rounds of resident workgroups (from the kernel's occupancy), enough that a share is at most
        // one tile's K. Env RQ_SKINNY_SK_SLOTS overrides the workgroups per round.
        static int slots[9] = {};
        int & sl = slots[sh.W];
        if (sl == 0) {
            int per_cu = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, rq_skinny_sk_kernel<T, R>, RQ_WAVE * sh.W,
                                                                     rq_sk_lds_bytes<T, R>()));
            sl = std::max(1, per_cu) * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
            if (getenv("RQ_SKINNY_SK_SLOTS")) {
                sl = std::max(1, atoi(getenv("RQ_SKINNY_SK_SLOTS")));
            }
        }
        const int nwg = (nblk + sl - 1) / sl * sl;
        const int ngroups = (Kw / RQ_KW) >> jobs.gsh;
        rq_skinny_sk_kernel<T, R><<<nwg, RQ_WAVE * sh.W, rq_sk_lds_bytes<T, R>(), stream>>>(ac, jobs, Kw, ngroups);
        return;
    }
    rq_skinny_kernel<T, R><<<dim3(nblk, sh.G), RQ_WAVE * sh.W, rq_lds_bytes<T, R>(sh.W), stream>>>(ac, jobs, Kw, sh.kw_split);
}

template <int T>
static void rq_skinny_launch_r(const rq_skinny_shape & sh, const int32_t * ac, const rq_jobs & jobs, int Kw, cudaStream_t stream) {
    constexpr int R0 = rq_cfg<T>::R;
    if (sh.R == 4)      rq_skinny_launch<T, R0 >= 4 ? 4 : 1>(sh, ac, jobs, Kw, stream);
    else if (sh.R == 2) rq_skinny_launch<T, R0 >= 2 ? 2 : 1>(sh, ac, jobs, Kw, stream);
    else                rq_skinny_launch<T, 1>(sh, ac, jobs, Kw, stream);
}

static void rq_skinny_dispatch(int tp, const rq_skinny_shape & sh, const int32_t * ac, const rq_jobs & jobs, int Kw, cudaStream_t stream) {
    switch (tp) {
        case 1:  rq_skinny_launch_r<1>(sh, ac, jobs, Kw, stream);  break;
        case 2:  rq_skinny_launch_r<2>(sh, ac, jobs, Kw, stream);  break;
        case 4:  rq_skinny_launch_r<4>(sh, ac, jobs, Kw, stream);  break;
        case 8:  rq_skinny_launch_r<8>(sh, ac, jobs, Kw, stream);  break;
        case 16: rq_skinny_launch_r<16>(sh, ac, jobs, Kw, stream); break;
        case 24: rq_skinny_launch_r<24>(sh, ac, jobs, Kw, stream); break;
        case 32: rq_skinny_launch_r<32>(sh, ac, jobs, Kw, stream); break;
        case 48: rq_skinny_launch_r<48>(sh, ac, jobs, Kw, stream); break;
        default: rq_skinny_launch_r<64>(sh, ac, jobs, Kw, stream); break;
    }
}

// every shape ggml_cuda_rq_w4a4_supported admits (N a multiple of the wave); activation groups have no other path
static bool rq_use_skinny(const ggml_tensor * src0) {
    GGML_UNUSED(src0);
    return true;
}

bool ggml_cuda_rq_w4a4_ready(const ggml_tensor * src0) {
    return src0->extra != nullptr;
}

// ---------------------------------------------------------------------------- dispatch
bool ggml_cuda_rq_w4a4_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    int p, m;
    const bool shapes = (src0->type == GGML_TYPE_W4A4 || src0->type == GGML_TYPE_W4A4H) &&
           src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % QK_W4A4 == 0 && src0->ne[0] <= QK_W4A4 * RQ_MAX_BLOCKS && src0->ne[1] % RQ_WAVE == 0 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src0) &&
           src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && dst->nb[0] == sizeof(float) &&
           (src0->type != GGML_TYPE_W4A4H || (rq_factor(src0->ne[0], p, m) && m <= 12288)); // FWHT chunk fits in 48 KiB
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(cc); GGML_UNUSED(shapes);
    return false;
#elif defined(GGML_USE_HIP)
    return GGML_CUDA_CC_IS_AMD(cc) && shapes;
#else
    return GGML_CUDA_CC_IS_NVIDIA(cc) && shapes;
#endif
}

// The quantized activation of the last one-chunk skinny call, so the next W4A4 matmul of the same input (up after
// gate, k and v after q, the DeltaNet gate after qkv) skips the Hadamard and the quantizer. The quantized layout
// depends on the padded token count only (rq_skinny_plan's KA), not on N. An entry lives for one graph evaluation
// of its context (ggml_cuda_rq_graph_begin), so reused graph tensors never hit stale data.
static constexpr int RQ_CACHE_KW = 4096; // K <= 32768
struct rq_act_cache {
    uint64_t gen = 1, entry_gen = 0;
    const ggml_tensor * src1 = nullptr;
    const void * data = nullptr;
    ggml_type type = GGML_TYPE_COUNT;
    float clip = 0.0f;
    int64_t rot = 0;
    int64_t K = 0, M = 0;
    int32_t * aq = nullptr; // [64 * RQ_CACHE_KW]
    float * sa = nullptr;   // [64]
    int32_t * mul = nullptr; // [RQ_MAX_GROUPS * 64]
};

static rq_act_cache & rq_cache_for(const ggml_backend_cuda_context & ctx) {
    static std::mutex mutex;
    static std::unordered_map<const ggml_backend_cuda_context *, rq_act_cache> caches;
    std::lock_guard<std::mutex> lock(mutex);
    return caches[&ctx];
}

void ggml_cuda_rq_graph_begin(ggml_backend_cuda_context & ctx) {
    rq_cache_for(ctx).gen++;
}

#if !defined(GGML_USE_MUSA)
// first use (never inside a graph capture, see ggml_cuda_rq_w4a4_ready): read the tensor's clip and Hadamard length
static rq_tensor_info * rq_info(const ggml_tensor * src0, cudaStream_t stream) {
    rq_tensor_info * info = (rq_tensor_info *) src0->extra;
    if (info == nullptr) {
        int32_t hdr[3] = { 0, 0, 0 }; // sum, pad[0], pad[1]
        CUDA_CHECK(cudaMemcpyAsync(hdr, (const char *) src0->data + offsetof(block_w4a4, sum), sizeof(hdr),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float tclip;
        memcpy(&tclip, &hdr[2], sizeof(tclip));
        const int32_t rot = src0->type == GGML_TYPE_W4A4H ? (int32_t) ggml_rq_rot_check(hdr[1], src0->ne[0]) : 0;
        const bool wg64 = ((uint32_t) hdr[0] >> 29) == 3u;
        GGML_ASSERT(!wg64 || ggml_rq_act_group() == 64); // the fold takes one weight multiplier per activation group
        info = new rq_tensor_info{ false, tclip, rot, wg64, rq_tc_ok(src0) };
        const_cast<ggml_tensor *>(src0)->extra = info;
    }
    return info;
}

// the skinny layout [Wt codes | s_w | colsum], made in place on first use
static void rq_repack_once(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, rq_tensor_info * info) {
    if (info->repacked) {
        return;
    }
    const int64_t K = src0->ne[0], N = src0->ne[1];
    const int Kw = (int) (K / 8);
    uint8_t * base = (uint8_t *) src0->data;
    const size_t nbytes = ggml_nbytes(src0);
    ggml_cuda_pool_alloc<uint8_t> tmp(ctx.pool(), nbytes);
    CUDA_CHECK(cudaMemcpyAsync(tmp.get(), base, nbytes, cudaMemcpyDeviceToDevice, ctx.stream()));
    const int64_t n_threads = N * Kw;
    if (info->tc) { // [codes in fragment order | s_w | ck]
        rq_tc_repack_kernel<<<(n_threads + 255) / 256, 256, 0, ctx.stream()>>>(tmp.get(), (int32_t *) base, (float *) (base + N * K / 2),
                                                                             (int) N, (int) K);
        const int64_t n_ck = (N / 16) * (K / 64) * 8;
        rq_tc_ck_kernel<<<(n_ck + 255) / 256, 256, 0, ctx.stream()>>>(tmp.get(), (uint32_t *) (base + N * K / 2 + N * 4), (int) N, (int) K);
        CUDA_CHECK(cudaGetLastError());
        info->repacked = true;
        return;
    }
    rq_repack_kernel<<<(n_threads + 255) / 256, 256, 0, ctx.stream()>>>(tmp.get(), (int32_t *) base, (float *) (base + N * K / 2),
                                                                      (int) N, (int) K);
    const int G = ggml_rq_act_group();
    const int64_t n_sums = N * (K / G);
    rq_colsum_kernel<<<(n_sums + 255) / 256, 256, 0, ctx.stream()>>>(tmp.get(), (int16_t *) (base + N * K / 2 + N * 4), (int) N, (int) K, G);
    const int64_t n_wk = N * (K / 64);
    rq_wk_kernel<<<(n_wk + 255) / 256, 256, 0, ctx.stream()>>>(tmp.get(), (int8_t *) (base + N * K / 2 + N * 4 + n_sums * 2), (int) N, (int) K);
    CUDA_CHECK(cudaGetLastError());
    info->repacked = true;
}

// RQ_HOST_PROF=1: host time of the skinny path by section, summed over all threads and printed at exit
struct rq_host_prof {
    enum { CACHE, HADAMARD, ALLOC, PLAN, QUANT, SKINNY, N_SEC };
    const bool on = getenv("RQ_HOST_PROF") != nullptr;
    std::atomic<int64_t> ns[N_SEC] = {}, calls{0};
    ~rq_host_prof() {
        if (!on || calls == 0) {
            return;
        }
        static const char * names[N_SEC] = { "cache lookup", "hadamard launches", "repack check + pool allocs",
                                             "plan + jobs", "quantize launch", "skinny launch" };
        for (int i = 0; i < N_SEC; ++i) {
            fprintf(stderr, "[rq-host-prof] %-28s %8.2f us per call (%lld calls)\n", names[i], ns[i] / 1e3 / calls,
                    (long long) calls.load());
        }
    }
};
static rq_host_prof rq_hprof;
struct rq_host_mark {
    bool on;   // off for a weight's first use (header read, repack), so the sums show the steady state
    int64_t t;
    explicit rq_host_mark(bool warm) : on(rq_hprof.on && warm), t(on ? ggml_time_us() : 0) {}
    void operator()(int sec) {
        if (on) {
            const int64_t now = ggml_time_us();
            rq_hprof.ns[sec] += (now - t) * 1000;
            t = now;
        }
    }
};

// INT4 tensor-core path (rq-w4a4-tc.cuh): quantize the M x K activation (already rotated for W4A4H) once, token-major,
// then one launch per weight. Up to 64 tokens a warp holds one channel tile and every token (decode, verify); wider, a
// block of 8 warps holds 128 channels x 64 tokens. K is split while the grid has fewer warps than the GPU can hold.
template <int MT, int NT, int WN, int WT>
static void rq_tc_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, ggml_tensor * const * dsts, int n,
                         const float * x, int ldx, int64_t K, int64_t M, float clip) {
    cudaStream_t stream = ctx.stream();
    constexpr int BT = 8 * NT * WT, BN = 16 * MT * WN;
    const int64_t Mp = (M + BT - 1) / BT * BT;
    const int Kw = (int) (K / 8), ng = (int) (K / 64);
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    std::vector<int> gz(n), nz(n);
    size_t n_acc = 0, n_cnt = 0;
    for (int i = 0; i < n; ++i) {
        const int64_t blocks = (src0s[i]->ne[1] + BN - 1) / BN * (Mp / BT);
        // split K until the grid could fill every SM with warps (48 on Ampere), keeping 4+ groups per split (one token,
        // 17408 x 5120 on the 3090: 67-72 us for 1 to 8 splits; 512 tokens: splits only cost)
        static const int env_split = getenv("RQ_TC_SPLIT") ? atoi(getenv("RQ_TC_SPLIT")) : 0;
        int s = 1;
        while (env_split == 0 && blocks * s * WN * WT < 48 * nsm && ng / (2 * s) >= 4) s *= 2;
        if (env_split > 0) s = env_split;
        gz[i] = (ng + s - 1) / s;
        nz[i] = (ng + gz[i] - 1) / gz[i];
        if (nz[i] > 1) {
            n_acc += Mp * src0s[i]->ne[1];
            n_cnt += blocks;
        }
    }
    const int G = ggml_rq_act_group();
    ggml_cuda_pool_alloc<int32_t> aq(ctx.pool(), Mp * Kw);
    ggml_cuda_pool_alloc<float>   sa(ctx.pool(), Mp);
    ggml_cuda_pool_alloc<int32_t> mul(ctx.pool(), Mp * (K / G));
    ggml_cuda_pool_alloc<int32_t> acc(ctx.pool(), std::max<size_t>(n_acc + n_cnt, 1)); // split-K sums, then counters
    const int zero_blocks = (int) ((n_acc + n_cnt + RQ_QUANT_THREADS * 16 - 1) / (RQ_QUANT_THREADS * 16));
    rq_quantize_act_kernel<<<M + zero_blocks, RQ_QUANT_THREADS, 0, stream>>>(x, ldx, (int) K, G, clip, aq.get(), sa.get(), mul.get(),
                                                                            (int) Mp, 0, acc.get(), (int64_t) (n_acc + n_cnt), (int) M);
    int32_t * acc_i = acc.get(), * cnt_i = acc.get() + n_acc;
    for (int i = 0; i < n; ++i) {
        const int64_t Ni = src0s[i]->ne[1];
        const uint8_t * base = (const uint8_t *) src0s[i]->data;
        const dim3 grid((Ni + BN - 1) / BN, Mp / BT, nz[i]);
        rq_tc_kernel<MT, NT, WN, WT><<<grid, 32 * WN * WT, 0, stream>>>(
            aq.get(), mul.get(), sa.get(), (const int4 *) base, (const float *) (base + Ni * K / 2),
            (const uint32_t *) (base + Ni * K / 2 + Ni * 4), (float *) dsts[i]->data, (int) (dsts[i]->nb[1] / sizeof(float)),
            (int) Ni, (int) M, (int) Mp, Kw, gz[i], acc_i, cnt_i);
        if (nz[i] > 1) {
            acc_i += Mp * Ni;
            cnt_i += grid.x * grid.y;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

static void rq_mul_mat_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, ggml_tensor * const * dsts, int n,
                          const float * x, int ldx, int64_t K, int64_t M, float clip) {
    for (int i = 0; i < n; ++i) {
        rq_repack_once(ctx, src0s[i], rq_info(src0s[i], ctx.stream()));
    }
    if      (M <= 8)  rq_tc_launch<1, 1, 4, 1>(ctx, src0s, dsts, n, x, ldx, K, M, clip);
    else if (M <= 16) rq_tc_launch<1, 2, 4, 1>(ctx, src0s, dsts, n, x, ldx, K, M, clip);
    else if (M <= 24) rq_tc_launch<1, 3, 4, 1>(ctx, src0s, dsts, n, x, ldx, K, M, clip);
    else if (M <= 32) rq_tc_launch<1, 4, 4, 1>(ctx, src0s, dsts, n, x, ldx, K, M, clip);
    else if (M <= 64) rq_tc_launch<1, 8, 4, 1>(ctx, src0s, dsts, n, x, ldx, K, M, clip);
    else              rq_tc_launch<2, 4, 4, 2>(ctx, src0s, dsts, n, x, ldx, K, M, clip);
}

// dsts[i] = src0s[i] x src1 for n <= 3 weights of one type, K and clip (see ggml_cuda_rq_w4a4_can_fuse): one
// activation transform and quantization, and on the skinny path one launch for all of them.
static void rq_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, ggml_tensor * const * dsts, int n,
                       const ggml_tensor * src1) {
    const ggml_tensor * src0 = src0s[0];
    ggml_tensor * dst = dsts[0];
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    cudaStream_t stream = ctx.stream();
    bool warm = true;
    for (int i = 0; i < n; ++i) {
        warm = warm && src0s[i]->extra != nullptr && ((const rq_tensor_info *) src0s[i]->extra)->repacked;
    }
    rq_host_mark mark(warm);

    const rq_tensor_info * info = rq_info(src0, stream);
    const float clip = ggml_rq_act_clip_for(info->clip, src0->type == GGML_TYPE_W4A4H);
    const int64_t n_rot = info->rot; // W4A4H: Hadamard blocks of n_rot along K

    // tokens per skinny launch (a tile size of rq_skinny_tp; env RQ_CHUNK). 24, not 64: the 64-token tile spills (36
    // VGPRs with ROCm 5.7's compiler, 77 with 7.2.4's) and 24-token launches cost less per token at every prefill
    // shape (MI50 pp512: 219 -> 321 t/s on ROCm 5.7, 176 -> 333 on 7.2.4)
    static const int chunk = getenv("RQ_CHUNK") ? rq_skinny_tp(atoi(getenv("RQ_CHUNK"))) : 24;
    rq_act_cache * cache = nullptr; // set for one-chunk skinny calls, which share their quantized activation
    bool hit = false;
    const bool tc = info->tc;
    if (!tc && rq_use_skinny(src0) && M <= chunk && K / 8 <= RQ_CACHE_KW) {
        cache = &rq_cache_for(ctx);
        hit = cache->entry_gen == cache->gen && cache->src1 == src1 && cache->data == src1->data &&
              cache->type == src0->type && cache->clip == clip && cache->rot == n_rot && cache->K == K && cache->M == M;
        if (cache->aq == nullptr) { // first use, never inside a graph capture (see ggml_cuda_rq_w4a4_ready)
            CUDA_CHECK(cudaMalloc(&cache->aq, (size_t) 64 * RQ_CACHE_KW * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&cache->sa, 64 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&cache->mul, (size_t) RQ_MAX_GROUPS * 64 * sizeof(int32_t)));
        }
    }

    mark(rq_host_prof::CACHE);
    const float * x = (const float *) src1->data;
    int ldx = (int) (src1->nb[1] / sizeof(float));
    ggml_cuda_pool_alloc<float> xh(ctx.pool()), xp(ctx.pool());
    // one-chunk skinny calls rotate and quantize in two launches instead of three (rq_hadamard_stats_kernel, then
    // rq_codes_kernel); env RQ_FUSE_HADAMARD=0 keeps the three
#if defined(GGML_USE_HIP)
    static const bool fuse_env = !(getenv("RQ_FUSE_HADAMARD") && atoi(getenv("RQ_FUSE_HADAMARD")) == 0);
#else
    static const bool fuse_env = false; // the fused kernels assume 64-lane waves
#endif
    int hp = 0, hm = 0;
    const bool fuse_h = fuse_env && src0->type == GGML_TYPE_W4A4H && !hit && cache != nullptr && M <= chunk &&
                        ggml_rq_act_group() == 64 && rq_factor(n_rot, hp, hm) && K % n_rot == 0 &&
                        (hp == 12 || hp == 20 || hp == 68) && (hm == 64 || hm == 128);
    ggml_cuda_pool_alloc<float2> lohi(ctx.pool()); // fused: per activation group of 64, the clipped range ends
    if (fuse_h) {
        xh.alloc(M * K);
        lohi.alloc(M * (K / 64));
        const dim3 hgrid(M * (K / n_rot), hp / 4);
        const uint32_t * masks = rq_paley_masks(hp);
        const float scale = 1.0f / sqrtf((float) n_rot); // same constant as the CPU reference
        auto stats = hp == 12 ? (hm == 64 ? rq_hadamard_stats_kernel<12, 1> : rq_hadamard_stats_kernel<12, 2>) :
                     hp == 20 ? (hm == 64 ? rq_hadamard_stats_kernel<20, 1> : rq_hadamard_stats_kernel<20, 2>) :
                                (hm == 64 ? rq_hadamard_stats_kernel<68, 1> : rq_hadamard_stats_kernel<68, 2>);
        stats<<<hgrid, 256, 0, stream>>>(x, ldx, xh.get(), (int) K, masks, scale, clip, lohi.get());
    } else if (src0->type == GGML_TYPE_W4A4H && !hit) {
        // block-diagonal: the M x K activation is M * (K / n_rot) rows of n_rot, each rotated as the CPU reference does
        int p, m;
        GGML_ASSERT(rq_factor(n_rot, p, m) && m <= 12288);
        const int64_t Mr = M * (K / n_rot);
        xh.alloc(M * K);
        const float scale = 1.0f / sqrtf((float) n_rot); // same constant as the CPU reference
        const int tf = std::min(m / 2, 512);
        rq_fwht_kernel<<<dim3(M, K / m), std::max(tf, 32), m * sizeof(float), stream>>>(x, ldx, xh.get(), (int) K, m, p, scale);
        if (p > 1) {
            if (m % 64 == 0 && (p == 12 || p == 20 || p == 68)) {
                xp.alloc(M * K);
                const dim3 rows(Mr, m / 64, p / 4);
                const uint32_t * masks = rq_paley_masks(p);
                switch (p) {
                    case 12: rq_paley_rows_kernel<12><<<rows, 256, 0, stream>>>(xh.get(), xp.get(), masks, (int) n_rot, m, scale); break;
                    case 20: rq_paley_rows_kernel<20><<<rows, 256, 0, stream>>>(xh.get(), xp.get(), masks, (int) n_rot, m, scale); break;
                    case 68: rq_paley_rows_kernel<68><<<rows, 256, 0, stream>>>(xh.get(), xp.get(), masks, (int) n_rot, m, scale); break;
                }
            } else {
                rq_paley_kernel<<<dim3(Mr, (m + 127) / 128), 128, 0, stream>>>(xh.get(), (int) n_rot, m, p, scale);
            }
        }
        x = xp.get() != nullptr ? xp.get() : xh.get();
        ldx = (int) K;
    }

    mark(rq_host_prof::HADAMARD);
    if (tc) {
        rq_mul_mat_tc(ctx, src0s, dsts, n, x, ldx, K, M, clip);
        return;
    }

    if (rq_use_skinny(src0)) {
        const int Kw = (int) (K / 8);
        for (int i = 0; i < n; ++i) {
            rq_repack_once(ctx, src0s[i], rq_info(src0s[i], stream));
        }
        const int G = ggml_rq_act_group();
        // prefill widths: the tiled GEMM (rq-w4a4-gemm.cuh) over all tokens, quantized in one launch (env RQ_GEMM_MIN,
        // 0 = skinny chunks only)
        static const int gemm_min = getenv("RQ_GEMM_MIN") ? atoi(getenv("RQ_GEMM_MIN")) : 64;
        bool gemm = cache == nullptr && gemm_min > 0 && M >= gemm_min;
        for (int i = 0; i < n; ++i) {
            gemm = gemm && src0s[i]->ne[1] % RQ_GEMM_BN == 0;
        }
        if (gemm) {
            const int64_t Mp = (M + RQ_GEMM_BT - 1) / RQ_GEMM_BT * RQ_GEMM_BT;
            const int gt = G / 64, nk = Kw / RQ_GEMM_KT;
            // split K while the grid is under two workgroups per CU, keeping 16+ tiles per split
            const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
            std::vector<int> kz(n), nz(n);
            size_t n_acc = 0, n_cnt = 0;
            for (int i = 0; i < n; ++i) {
                const int64_t wgs = src0s[i]->ne[1] / RQ_GEMM_BN * (Mp / RQ_GEMM_BT);
                int s = 1;
                while (wgs * s < 2 * nsm && nk / (2 * s) >= 16) s *= 2;
                kz[i] = ((nk + s - 1) / s + gt - 1) / gt * gt;
                nz[i] = (nk + kz[i] - 1) / kz[i];
                if (nz[i] > 1) {
                    n_acc += Mp * src0s[i]->ne[1];
                    n_cnt += wgs;
                }
            }
            ggml_cuda_pool_alloc<int32_t> aq(ctx.pool(), Mp * Kw);
            ggml_cuda_pool_alloc<float>   sa(ctx.pool(), Mp);
            ggml_cuda_pool_alloc<int32_t> mul(ctx.pool(), Mp * (K / G));
            ggml_cuda_pool_alloc<int32_t> acc(ctx.pool(), std::max<size_t>(n_acc + n_cnt, 1)); // split-K sums, then counters
            const int zero_blocks = (int) ((n_acc + n_cnt + RQ_QUANT_THREADS * 16 - 1) / (RQ_QUANT_THREADS * 16));
            rq_quantize_act_kernel<<<M + zero_blocks, RQ_QUANT_THREADS, 0, stream>>>(x, ldx, (int) K, G, clip, aq.get(), sa.get(),
                                                                                    mul.get(), (int) Mp, 0, acc.get(),
                                                                                    (int64_t) (n_acc + n_cnt), (int) M);
            int32_t * acc_i = acc.get(), * cnt_i = acc.get() + n_acc;
            for (int i = 0; i < n; ++i) {
                const int64_t Ni = src0s[i]->ne[1];
                const uint8_t * base = (const uint8_t *) src0s[i]->data;
                const size_t o_sw = Ni * K / 2, o_cs = o_sw + Ni * 4, o_wk = o_cs + Ni * (K / G) * 2;
                const dim3 grid(Ni / RQ_GEMM_BN, Mp / RQ_GEMM_BT, nz[i]);
                auto kernel = gt == 1 ? rq_gemm_kernel<0> : gt == 2 ? rq_gemm_kernel<1> : rq_gemm_kernel<2>;
                kernel<<<grid, RQ_GEMM_THREADS, 0, stream>>>(
                    aq.get(), mul.get(), sa.get(), (const int32_t *) base, (const float *) (base + o_sw),
                    (const int16_t *) (base + o_cs), (const int8_t *) (base + o_wk), (float *) dsts[i]->data,
                    (int) (dsts[i]->nb[1] / sizeof(float)), (int) Ni, (int) M, (int) Mp, Kw, kz[i], acc_i, cnt_i);
                if (nz[i] > 1) {
                    acc_i += Mp * Ni;
                    cnt_i += Ni / RQ_GEMM_BN * (Mp / RQ_GEMM_BT);
                }
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const int tp_max = rq_skinny_tp(std::min<int64_t>(M, chunk));
        ggml_cuda_pool_alloc<int32_t> aq_pool(ctx.pool());
        ggml_cuda_pool_alloc<float>   sa_pool(ctx.pool());
        ggml_cuda_pool_alloc<int32_t> mul_pool(ctx.pool());
        int32_t * aq = cache ? cache->aq : aq_pool.alloc((size_t) tp_max * Kw);
        float   * sa = cache ? cache->sa : sa_pool.alloc(tp_max);
        int32_t * mul = cache ? cache->mul : mul_pool.alloc((size_t) tp_max * (K / G));
        size_t n_acc = 0;
        for (int i = 0; i < n; ++i) {
            n_acc += (size_t) tp_max * src0s[i]->ne[1] + src0s[i]->ne[1] / RQ_WAVE;
        }
        ggml_cuda_pool_alloc<int32_t> acc(ctx.pool(), n_acc); // per weight: split-K sums, then counters
        mark(rq_host_prof::ALLOC);
        if (mark.on) {
            rq_hprof.calls++;
        }
        for (int64_t m0 = 0; m0 < M; m0 += chunk) {
            const int mc = (int) std::min<int64_t>(chunk, M - m0);
            const int tp = rq_skinny_tp(mc);
            int64_t N_launch = 0;
            for (int i = 0; i < n; ++i) {
                N_launch += src0s[i]->ne[1];
            }
            rq_skinny_shape sh = rq_skinny_plan(tp, (int) N, Kw, (int) N_launch, G / 32);
            for (int i = 1; i < n; ++i) {
                const rq_skinny_shape shi = rq_skinny_plan(tp, (int) src0s[i]->ne[1], Kw, (int) N_launch, G / 32);
                if (shi.R < sh.R) { // one R for the launch
                    sh = shi;
                }
            }
            rq_jobs jobs = {};
            jobs.n = n;
            jobs.gsh = G == 64 ? 1 : G == 128 ? 2 : 3;
            // each wave starts its K slice (group * 7) mod its groups in: aligned slices put the waves' weight streams on the
            // same DRAM channels (MI50, T = 8 forward: one GPU 40.4 -> 39.2 ms, TP2 30.2 -> 29.6, TP4 unchanged; some
            // multipliers, 5 among them, are worse on some shapes). Env RQ_SKINNY_ROT (0 = in order).
            static const int rot = getenv("RQ_SKINNY_ROT") ? atoi(getenv("RQ_SKINNY_ROT")) : 7;
            jobs.rot = rot;
            int64_t off = 0;
            for (int i = 0; i < n; ++i) {
                const int64_t Ni = src0s[i]->ne[1];
                const uint8_t * base = (const uint8_t *) src0s[i]->data;
                const int ldd = (int) (dsts[i]->nb[1] / sizeof(float));
                const int groups = (int) (Ni / RQ_WAVE / sh.R);
                int32_t * acc_i = acc.get() + off;
                jobs.j[i] = { (const int32_t *) base, acc_i, (int) Ni, groups,
                              { sa, mul, (const float *) (base + Ni * K / 2), (const int16_t *) (base + Ni * K / 2 + Ni * 4),
                                (const int8_t *) (base + Ni * K / 2 + Ni * 4 + Ni * (K / G) * 2),
                                (float *) dsts[i]->data + m0 * ldd, ldd, mc, acc_i + (size_t) mc * Ni } };
                off += (int64_t) mc * Ni + groups;
            }
            for (int i = n; i < 3; ++i) {
                jobs.j[i] = jobs.j[0];
                jobs.j[i].nblk = 0;
            }
            const int64_t n_zero = sh.G > 1 || sh.sk ? off : 0;
            const int zero_blocks = (int) ((n_zero + RQ_QUANT_THREADS * 16 - 1) / (RQ_QUANT_THREADS * 16));
            mark(rq_host_prof::PLAN);
            if (fuse_h) {
                static_assert(RQ_QUANT_THREADS == 256, "rq_codes_kernel's zero blocks are zero_blocks");
                const dim3 cgrid(mc + zero_blocks, (K / 64 + RQ_CODE_GROUPS - 1) / RQ_CODE_GROUPS);
                rq_codes_kernel<<<cgrid, 256, 0, stream>>>(xh.get(), lohi.get(), (int) K, aq, sa, mul, tp, sh.KA, acc.get(), n_zero, mc);
            } else if (!hit) {
                rq_quantize_act_kernel<<<mc + zero_blocks, RQ_QUANT_THREADS, 0, stream>>>(x + m0 * ldx, ldx, (int) K, G, clip,
                                                                           aq, sa, mul, tp, sh.KA, acc.get(), n_zero, mc);
            } else if (zero_blocks > 0) { // reused activation: only clear the split-K sums and counters
                rq_quantize_act_kernel<<<zero_blocks, RQ_QUANT_THREADS, 0, stream>>>(x, ldx, (int) K, G, clip,
                                                                     aq, sa, mul, tp, sh.KA, acc.get(), n_zero, 0);
            }
            mark(rq_host_prof::QUANT);
            rq_skinny_dispatch(tp, sh, aq, jobs, Kw, stream);
            mark(rq_host_prof::SKINNY);
        }
        if (cache && !hit) {
            cache->entry_gen = cache->gen;
            cache->src1 = src1;
            cache->data = src1->data;
            cache->type = src0->type;
            cache->clip = clip;
            cache->rot = n_rot;
            cache->K = K;
            cache->M = M;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ABORT("ruquant W4A4: no path for this shape");
}
#endif // !GGML_USE_MUSA

void ggml_cuda_mul_mat_rq_w4a4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("ruquant W4A4 is not available on this backend");
#else
    rq_mul_mat(ctx, &src0, &dst, 1, src1);
#endif
}

bool ggml_cuda_rq_w4a4_can_fuse(const ggml_tensor * a, const ggml_tensor * b) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(a); GGML_UNUSED(b);
    return false;
#else
    const ggml_tensor * wa = a->src[0];
    const ggml_tensor * wb = b->src[0];
    const ggml_tensor * x  = a->src[1];
    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT || b->src[1] != x ||
        (wa->type != GGML_TYPE_W4A4 && wa->type != GGML_TYPE_W4A4H) || wb->type != wa->type || wb->ne[0] != wa->ne[0] ||
        x->type != GGML_TYPE_F32 || ggml_nrows(x) != x->ne[1] || x->ne[1] > 16 || // verify widths (fused prefill chunks were slower)
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !rq_use_skinny(wa) || !rq_use_skinny(wb)) {
        return false;
    }
    // both weights seen before: their first use reads the clip with a synchronize, so it stays unfused
    const rq_tensor_info * ia = (const rq_tensor_info *) wa->extra;
    const rq_tensor_info * ib = (const rq_tensor_info *) wb->extra;
    return ia != nullptr && ib != nullptr && ia->clip == ib->clip && ia->rot == ib->rot;
#endif
}

void ggml_cuda_mul_mat_rq_w4a4_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, ggml_tensor * const * dsts,
                                     int n, const ggml_tensor * src1) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx); GGML_UNUSED(src0s); GGML_UNUSED(dsts); GGML_UNUSED(n); GGML_UNUSED(src1);
    GGML_ABORT("ruquant W4A4 is not available on this backend");
#else
    GGML_ASSERT(n >= 1 && n <= 3);
    rq_mul_mat(ctx, src0s, dsts, n, src1);
#endif
}

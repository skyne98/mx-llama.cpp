// ruquant attention over a q4_0 KV cache for decode and speculative-verify widths (1-16 query tokens), head size 256,
// shaped for gfx906: Q.K with v_dot8_i32_i4 straight from the cache's nibbles, P.V with v_dot2_f32_f16.
//
// The cache is llama.cpp's q4_0 with its attention rotation (llama-kv-cache.cpp: Q, K and V go through a Hadamard
// before the cache when it is quantized), so K and V are rotated int4 with a scale per 32 values, and nothing is
// converted to f16 before attention: the f16 tile kernel would convert the whole cache on every call.
//   - Q is quantized per 32 values to int4 (RQ_FA_DOT8=1), or to int8 as two int4 planes, 16 * hi + lo
//     (RQ_FA_DOT8=2, two v_dot8 per 8 products), in q4_0's nibble order. A q4_0 nibble q is q - 8 as a signed int4
//     after flipping its top bit, so a K dword feeds v_dot8 after one xor.
//   - The KV sequence is split over the grid; each workgroup (4 waves) streams 64-token tiles of K and V through LDS.
//     Q.K: lanes are tokens, each wave a quarter of the query rows (the query heads sharing this KV head times the
//     tokens), Q from scalar loads; online softmax per row with wave reductions, P to LDS as f16.
//     P.V: lanes are the 256 output channels; each lane decodes its channel of V for 8 tokens once and uses it for
//     every row (v_dot2 with f32 accumulation).
//   - Partial results go through llama.cpp's flash_attn_combine_results.
// On NVIDIA GPUs with INT4 tensor cores (the #else branch) the same numerics run on mma.sync: m16n8k32 .s4 per q4_0
// block for Q.K, f16 m16n8k16 for P.V.
#include "common.cuh"
#include "fattn-common.cuh"
#include "rq-fattn.cuh"

#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP) // v_dot8, v_dot2 and DPP: gfx906 only, other AMD GPUs keep the stock kernels

#define RQ_FA_D    256
#define RQ_FA_TILE 64
#define RQ_FA_ROWW 37 // LDS words per token row of K / V: 36 (8 q4_0 blocks of 18 bytes) + 1 against bank conflicts

typedef _Float16 rq_fa_h2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ int rq_fa_dot8(int a, int b, int c) {
#if defined(__gfx906__)
    return __builtin_amdgcn_sdot8(a, b, c, false);
#else
#pragma unroll
    for (int j = 0; j < 8; ++j) c += ((a << (28 - 4 * j)) >> 28) * ((b << (28 - 4 * j)) >> 28);
    return c;
#endif
}

static __device__ __forceinline__ float rq_fa_dot2(rq_fa_h2 a, rq_fa_h2 b, float c) {
#if defined(__gfx906__)
    return __builtin_amdgcn_fdot2(a, b, c, false);
#else
    return c + (float) a.x * (float) b.x + (float) a.y * (float) b.y;
#endif
}

// the 4 bytes at byte offset o of a word array (o need not be aligned)
static __device__ __forceinline__ int rq_fa_bytes(const int * w, int o) {
    const int i = o >> 2, s = o & 3;
    const unsigned lo = w[i], hi = w[i + 1];
    return s == 0 ? (int) lo : (int) ((lo >> (8 * s)) | (hi << (32 - 8 * s)));
}

// reductions over the wave without LDS: DPP within each row of 16 lanes, then the four rows' lane 0 through
// v_readlane (a ds_bpermute per step waited out a round trip each, 12 per query row and tile)
template <bool MAX>
static __device__ __forceinline__ float rq_fa_wave_reduce(float v) {
    auto op = [](float a, float b) { return MAX ? fmaxf(a, b) : a + b; };
    const int lane = threadIdx.x % 64;
    v = op(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false))); // ^1
    v = op(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false))); // ^2
    const int r4 = __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x124, 0xF, 0xF, false);           // ror 4
    const int r12 = __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x12C, 0xF, 0xF, false);          // ror 12
    v = op(v, __int_as_float((lane & 4) ? r4 : r12));                                                    // ^4
    v = op(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x128, 0xF, 0xF, false))); // ror 8 = ^8
    const float a = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0));
    const float b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
    const float c = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32));
    const float d = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
    return op(op(a, b), op(c, d));
}

static __device__ __forceinline__ float rq_fa_half(const int * w, int o) { // f16 at an even byte offset
    const unsigned v = (unsigned) w[o >> 2] >> (8 * (o & 3));
    return __half2float(__ushort_as_half((unsigned short) (v & 0xffff)));
}

// Q rows to int4 nibbles in q4_0 order: word j of block b holds elements 4j..4j+3 in its even nibbles and
// 4j+16..4j+19 in its odd ones. Row r of KV head g is query head g * gqa + r / ncols, query token r % ncols; one wave
// per row, 4 values per lane. Qq [row][QB][32], Qd [row][8] (the scale times the softmax scale).
template <int QB>
static __global__ void __launch_bounds__(64) rq_fa_quant_q(const char * __restrict__ Q, size_t nb1, size_t nb2, int ncols, int gqa,
                                                           float scale, int * __restrict__ Qq, float * __restrict__ Qd) {
    const int h = blockIdx.x / ncols, j = blockIdx.x % ncols;
    const int R = gqa * ncols, row = (h / gqa) * R + (h % gqa) * ncols + j;
    const int lane = threadIdx.x;
    const float * x = (const float *) (Q + (size_t) j * nb1 + (size_t) h * nb2) + 4 * lane;
    const float4 v = *(const float4 *) x;
    const float xs[4] = { v.x, v.y, v.z, v.w };
    // the block's value of largest magnitude, with its sign (8 lanes = a block of 32)
    float m = xs[0];
#pragma unroll
    for (int k = 1; k < 4; ++k) m = fabsf(xs[k]) > fabsf(m) ? xs[k] : m;
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) {
        const float mo = __shfl_xor(m, o, 64);
        m = fabsf(mo) > fabsf(m) || (fabsf(mo) == fabsf(m) && mo < m) ? mo : m;
    }
    // int4 as q4_0 rounds: that value to -8, all 16 levels (amax / 7 used 15; NMSE against the CPU went from 5.0-5.6e-4 to
    // under test-backend-ops' 5e-4). int8 as two int4 planes: 119 = 16 * 7 + 7 keeps both planes in [-7, 7]
    const float d = QB == 1 ? m / -8.0f : fabsf(m) / 119.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const float qlo = QB == 1 ? -8.0f : -119.0f, qhi = QB == 1 ? 7.0f : 119.0f;
    unsigned w[QB] = {};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int q = (int) rintf(fminf(fmaxf(xs[k] * id, qlo), qhi));
        const int odd = (lane & 7) >= 4; // elements 16..31 of the block go to the odd nibbles
        const int sh = 4 * (2 * k + odd);
        if (QB == 1) {
            w[0] |= (unsigned) (q & 15) << sh;
        } else {
            const int lo = ((q + 8) & 15) - 8, hi = (q - lo) / 16;
            w[0] |= (unsigned) (hi & 15) << sh;
            w[QB - 1] |= (unsigned) (lo & 15) << sh;
        }
    }
    const int b = lane >> 3, jw = lane & 3;
#pragma unroll
    for (int p = 0; p < QB; ++p) {
        const unsigned both = w[p] | __shfl_xor(w[p], 4, 64);
        if ((lane & 7) < 4) Qq[((size_t) row * QB + p) * 32 + b * 4 + jw] = (int) both;
    }
    if ((lane & 7) == 0) Qd[(size_t) row * 8 + b] = d * scale;
}

// grid (chunks, KV heads), 256 threads. R = gqa * ncols query rows per KV head, at most RMAX.
template <int RMAX, int QB>
static __global__ void __launch_bounds__(256) rq_fa_q4_kernel(
        const int * __restrict__ Qq, const float * __restrict__ Qd, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ parts, float2 * __restrict__ meta, float * __restrict__ dst,
        size_t nbk1, size_t nbk2, size_t nbv1, size_t nbv2, size_t nbm1,
        int R, int ncols, int gqa, int nheads, int n_kv, int chunk) {
    __shared__ int   ks[RQ_FA_TILE * RQ_FA_ROWW];
    __shared__ int   vs[RQ_FA_TILE * RQ_FA_ROWW];
    __shared__ _Float16 ps[RMAX * RQ_FA_TILE];
    __shared__ float ms[RMAX], ls[RMAX], as[RMAX];
    __shared__ float mk[16 * RQ_FA_TILE]; // the tile's mask, per query token (-inf past the cache)

    const int g = blockIdx.y, nchunks = gridDim.x;
    const int t_begin = blockIdx.x * chunk, t_end = min(t_begin + chunk, n_kv);
    const int lane = threadIdx.x % 64;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int c = wave * 64 + lane; // this lane's output channel
    const char * Kg = K + (size_t) g * nbk2;
    const char * Vg = V + (size_t) g * nbv2;

    for (int r = threadIdx.x; r < RMAX; r += 256) {
        ms[r] = -INFINITY;
        ls[r] = 0.0f;
        as[r] = 1.0f;
    }
    // P.V runs over all RMAX rows without branches (a guard per row made each row wait out its own LDS read): the rows
    // past R keep P = 0
    for (int i = threadIdx.x; i < (RMAX - R) * RQ_FA_TILE; i += 256) ps[R * RQ_FA_TILE + i] = (_Float16) 0.0f;
    float acc[RMAX];
#pragma unroll
    for (int r = 0; r < RMAX; ++r) acc[r] = 0.0f;

    for (int t0 = t_begin; t0 < t_end; t0 += RQ_FA_TILE) {
        // K and V rows of the tile into LDS, 36 words each
        for (int i = threadIdx.x; i < RQ_FA_TILE * 36; i += 256) {
            const int tr = i / 36, cw = i % 36;
            const bool in = t0 + tr < t_end;
            ks[tr * RQ_FA_ROWW + cw] = in ? *(const int *) (Kg + (size_t) (t0 + tr) * nbk1 + 4 * cw) : 0;
            vs[tr * RQ_FA_ROWW + cw] = in ? *(const int *) (Vg + (size_t) (t0 + tr) * nbv1 + 4 * cw) : 0;
        }
        // each query row reads its token's mask from LDS: as a global load per row it was a round trip per row
        for (int i = threadIdx.x; i < ncols * RQ_FA_TILE; i += 256) {
            const int col = i / RQ_FA_TILE, tt = i % RQ_FA_TILE;
            const bool in = t0 + tt < t_end;
            mk[i] = !in ? -INFINITY : mask ? __half2float(((const half *) (mask + (size_t) col * nbm1))[t0 + tt]) : 0.0f;
        }
        __syncthreads();

        // Q.K and the online softmax: lane = token t0 + lane, this wave's rows
        {
            const int * kr = ks + lane * RQ_FA_ROWW;
            int kq[32];
            float dk[8];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                dk[b] = rq_fa_half(kr, 18 * b);
#pragma unroll
                for (int j = 0; j < 4; ++j) kq[b * 4 + j] = rq_fa_bytes(kr, 18 * b + 2 + 4 * j) ^ 0x88888888;
            }
#pragma unroll 1
            for (int r = wave; r < R; r += 4) {
                const int * q = Qq + ((size_t) g * R + r) * QB * 32;
                const float * qd = Qd + ((size_t) g * R + r) * 8;
                float s = 0.0f;
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    int sb = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) sb = rq_fa_dot8(q[b * 4 + j], kq[b * 4 + j], sb);
                    if (QB == 2) {
                        int sl = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) sl = rq_fa_dot8(q[32 + b * 4 + j], kq[b * 4 + j], sl);
                        sb = 16 * sb + sl;
                    }
                    s += (float) sb * (dk[b] * qd[b]);
                }
                s += mk[(r % ncols) * RQ_FA_TILE + lane];
                const float mt = rq_fa_wave_reduce<true>(s);
                const float m_old = ms[r];
                const float m_new = fmaxf(m_old, mt);
                const float p = m_new == -INFINITY ? 0.0f : __expf(s - m_new);
                const float lt = rq_fa_wave_reduce<false>(p);
                ps[r * RQ_FA_TILE + lane] = (_Float16) p;
                if (lane == 0) {
                    const float alpha = m_new == -INFINITY ? 1.0f : __expf(m_old - m_new);
                    ms[r] = m_new;
                    ls[r] = ls[r] * alpha + lt;
                    as[r] = alpha;
                }
            }
        }
        __syncthreads();

        // P.V: lane = channel c of block cb, byte cb * 18 + 2 + (c % 16) of a token row, nibble c % 32 / 16
        {
            const int cb = c / 32, cbyte = cb * 18 + 2 + (c % 16), csh = 4 * ((c % 32) / 16);
#pragma unroll
            for (int r = 0; r < RMAX; ++r) acc[r] *= as[r];
#pragma unroll 1
            for (int t = 0; t < RQ_FA_TILE; t += 8) {
                rq_fa_h2 v2[4];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int * vr = vs + (t + k) * RQ_FA_ROWW;
                    const int q = (((unsigned) vr[cbyte >> 2] >> (8 * (cbyte & 3) + csh)) & 15) - 8;
                    const _Float16 val = (_Float16) (rq_fa_half(vr, 18 * cb) * (float) q);
                    if (k & 1) v2[k / 2].y = val; else v2[k / 2].x = val;
                }
                // P of 4 rows read before their dot2s, so that the LDS reads overlap (row by row, each waited out its
                // read; more rows in flight cost VGPRs past 128 at RMAX 48). Rows per wave with 4 channels per lane, a
                // quarter of the P reads, was measured: 64k verify 59.6 -> 59.1 ms, decode 38.5 -> 40.9 ms (each lane
                // decodes V for 4 channels and, at 6 rows, uses each value for 2 rows). LDS bandwidth is not the limit:
                // at 2 waves per SIMD the waves wait on latency 54% of the time (rocprofv3, REPORT 2026-11-06).
#pragma unroll
                for (int r0 = 0; r0 < RMAX; r0 += 4) {
                    rq_fa_h2 pr[4][4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
#pragma unroll
                        for (int k = 0; k < 4; ++k) pr[i][k] = ((const rq_fa_h2 *) (ps + (r0 + i) * RQ_FA_TILE + t))[k];
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
#pragma unroll
                        for (int k = 0; k < 4; ++k) acc[r0 + i] = rq_fa_dot2(pr[i][k], v2[k], acc[r0 + i]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // the chunk's unnormalized sums and (max, sum) per row, or the normalized output if there is one chunk
#pragma unroll
    for (int r = 0; r < RMAX; ++r) {
        if (r >= R) continue;
        const int h = g * gqa + r / ncols, col = r % ncols;
        const size_t jdst = (size_t) col * nheads + h;
        if (nchunks == 1) {
            dst[jdst * RQ_FA_D + c] = ls[r] > 0.0f ? acc[r] / ls[r] : 0.0f;
        } else {
            parts[(jdst * nchunks + blockIdx.x) * RQ_FA_D + c] = acc[r];
            if (threadIdx.x == 0) meta[jdst * nchunks + blockIdx.x] = make_float2(ms[r], ls[r]);
        }
    }
}

static int rq_fa_mode() {
    // on by default with int4 Q: at 8-token ubatches KLD 0.0423 (f16 cache 0.0416, stock q4_0 path 0.0442, int8 Q
    // 0.0425; 16 chunks, +-3%); 0 turns it off, 2 takes int8 Q
    static const int mode = getenv("RQ_FA_DOT8") ? atoi(getenv("RQ_FA_DOT8")) : 1;
    return mode;
}

bool ggml_cuda_rq_fattn_q4_applies(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (rq_fa_mode() == 0 || ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VEGA20) { // v_dot8: gfx906
        return false;
    }
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    return K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 && Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           K->ne[0] == RQ_FA_D && V->ne[0] == RQ_FA_D && Q->ne[0] == RQ_FA_D && Q->ne[1] <= 16 && Q->ne[3] == 1 && K->ne[3] == 1 &&
           Q->ne[2] % K->ne[2] == 0 && (Q->ne[2] / K->ne[2]) * Q->ne[1] <= 96 && K->ne[1] % RQ_FA_TILE == 0 &&
           Q->nb[0] == sizeof(float) && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 &&
           V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 && (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1)) &&
           !sinks && max_bias == 0.0f && softcap == 0.0f;
}

template <int RMAX, int QB>
static void rq_fa_launch(dim3 grid, cudaStream_t s, const int * Qq, const float * Qd, const ggml_tensor * K, const ggml_tensor * V,
                         const ggml_tensor * mask, float * parts, float2 * meta, float * dst, int R, int ncols, int gqa, int nheads,
                         int chunk) {
    rq_fa_q4_kernel<RMAX, QB><<<grid, 256, 0, s>>>(Qq, Qd, (const char *) K->data, (const char *) V->data,
                                                   mask ? (const char *) mask->data : nullptr, parts, meta, dst, K->nb[1], K->nb[2],
                                                   V->nb[1], V->nb[2], mask ? mask->nb[1] : 0, R, ncols, gqa, nheads, (int) K->ne[1],
                                                   chunk);
}

template <int QB>
static void rq_fa_launch_r(int R, dim3 grid, cudaStream_t s, const int * Qq, const float * Qd, const ggml_tensor * K,
                           const ggml_tensor * V, const ggml_tensor * mask, float * parts, float2 * meta, float * dst, int ncols,
                           int gqa, int nheads, int chunk) {
    if (R <= 8)       rq_fa_launch<8,  QB>(grid, s, Qq, Qd, K, V, mask, parts, meta, dst, R, ncols, gqa, nheads, chunk);
    else if (R <= 16) rq_fa_launch<16, QB>(grid, s, Qq, Qd, K, V, mask, parts, meta, dst, R, ncols, gqa, nheads, chunk);
    else if (R <= 32) rq_fa_launch<32, QB>(grid, s, Qq, Qd, K, V, mask, parts, meta, dst, R, ncols, gqa, nheads, chunk);
    else if (R <= 48) rq_fa_launch<48, QB>(grid, s, Qq, Qd, K, V, mask, parts, meta, dst, R, ncols, gqa, nheads, chunk);
    else              rq_fa_launch<96, QB>(grid, s, Qq, Qd, K, V, mask, parts, meta, dst, R, ncols, gqa, nheads, chunk);
}

void ggml_cuda_rq_fattn_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t s = ctx.stream();
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int ncols = (int) Q->ne[1], nheads = (int) Q->ne[2], nkvh = (int) K->ne[2];
    const int gqa = nheads / nkvh, R = gqa * ncols;
    const int QB = rq_fa_mode() == 2 ? 2 : 1;

    ggml_cuda_pool_alloc<int>   Qq(ctx.pool(), (size_t) nheads * ncols * QB * 32);
    ggml_cuda_pool_alloc<float> Qd(ctx.pool(), (size_t) nheads * ncols * 8);
    if (QB == 1) {
        rq_fa_quant_q<1><<<nheads * ncols, 64, 0, s>>>((const char *) Q->data, Q->nb[1], Q->nb[2], ncols, gqa, scale, Qq.get(), Qd.get());
    } else {
        rq_fa_quant_q<2><<<nheads * ncols, 64, 0, s>>>((const char *) Q->data, Q->nb[1], Q->nb[2], ncols, gqa, scale, Qq.get(), Qd.get());
    }

    // the KV sequence over ~2 workgroups per CU, whole tiles each
    const int ntiles = (int) K->ne[1] / RQ_FA_TILE;
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    static const int env_chunks = getenv("RQ_FA_CHUNKS") ? atoi(getenv("RQ_FA_CHUNKS")) : 0;
    int nchunks = env_chunks > 0 ? env_chunks : std::max(1, std::min(ntiles, (2 * nsm + nkvh - 1) / nkvh));
    const int chunk = (ntiles + nchunks - 1) / nchunks * RQ_FA_TILE;
    nchunks = (int) ((K->ne[1] + chunk - 1) / chunk);

    ggml_cuda_pool_alloc<float>  parts(ctx.pool(), nchunks > 1 ? (size_t) nchunks * nheads * ncols * RQ_FA_D : 1);
    ggml_cuda_pool_alloc<float2> meta(ctx.pool(), nchunks > 1 ? (size_t) nchunks * nheads * ncols : 1);
    const dim3 grid(nchunks, nkvh, 1);
    if (QB == 1) {
        rq_fa_launch_r<1>(R, grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), (float *) dst->data, ncols, gqa, nheads, chunk);
    } else {
        rq_fa_launch_r<2>(R, grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), (float *) dst->data, ncols, gqa, nheads, chunk);
    }
    if (nchunks > 1) {
        flash_attn_combine_results<RQ_FA_D><<<dim3(ncols, nheads, 1), RQ_FA_D, nchunks * sizeof(float2), s>>>(
            parts.get(), meta.get(), (float *) dst->data, nchunks);
    }
    CUDA_CHECK(cudaGetLastError());
}

#else // NVIDIA: INT4 tensor cores (Turing, Ampere, Ada), the same numerics

#define RQ_FA_D     256
#define RQ_FA_TILE  64
#define RQ_FA_ROWW  37 // smem words per token row of K / V: 36 (8 q4_0 blocks of 18 bytes) + 1 against bank conflicts
#define RQ_FA_SROW  68 // f32 words per query row of S (64 + 4 against bank conflicts); P is written over it as f16

#define RQ_FA_TC_MIN_CC 750 // Turing: m8n8k32 .s4, m16n8k8 .f16
#define RQ_FA_TC_MAX_CC 900 // Hopper and later have no INT4 MMA

// the 4 bytes at byte offset o of a word array (o need not be aligned)
static __device__ __forceinline__ int rq_fa_bytes(const int * w, int o) {
    const int i = o >> 2, s = o & 3;
    const unsigned lo = w[i], hi = w[i + 1];
    return s == 0 ? (int) lo : (int) ((lo >> (8 * s)) | (hi << (32 - 8 * s)));
}

static __device__ __forceinline__ float rq_fa_half(const int * w, int o) { // f16 at an even byte offset
    const unsigned v = (unsigned) w[o >> 2] >> (8 * (o & 3));
    return __half2float(__ushort_as_half((unsigned short) (v & 0xffff)));
}

// d = a b over one q4_0 block, exact: a0 / a1 the Q words of rows gid / gid + 8, b the K word of token gid
static __device__ __forceinline__ void rq_fa_mma_s4(int (&d)[4], int a0, int a1, int b) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b), "r"(0));
#elif !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %4};"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(a0), "r"(b), "r"(0));
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %4};"
                 : "=r"(d[2]), "=r"(d[3]) : "r"(a1), "r"(b), "r"(0));
#else
    GGML_UNUSED(a0); GGML_UNUSED(a1); GGML_UNUSED(b);
    d[0] = d[1] = d[2] = d[3] = 0;
    NO_DEVICE_CODE;
#endif
}

// d += a b, f16 operands, f32 accumulation: a the P fragment (16 rows x 16 tokens), b0 b1 the V fragment
static __device__ __forceinline__ void rq_fa_mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
    NO_DEVICE_CODE;
#endif
}

// the value of larger magnitude, the negative one on a tie (as the gfx906 kernel)
static __device__ __forceinline__ float rq_fa_absmax(float a, float b) {
    return fabsf(b) > fabsf(a) || (fabsf(b) == fabsf(a) && b < a) ? b : a;
}

// Q rows to int4 nibbles in q4_0 order: word j of block b holds elements 4j..4j+3 in its even nibbles and
// 4j+16..4j+19 in its odd ones. Row r of KV head g is query head g * gqa + r / ncols, query token r % ncols. One warp
// per row, 8 values per lane (lane % 4 = which 8 of block lane / 4). Qq [row][8][4], Qd [row][8] (scale times the
// softmax scale).
static __global__ void __launch_bounds__(32) rq_fa_quant_q(const char * __restrict__ Q, size_t nb1, size_t nb2, int ncols, int gqa,
                                                           float scale, int * __restrict__ Qq, float * __restrict__ Qd) {
    const int h = blockIdx.x / ncols, j = blockIdx.x % ncols;
    const int R = gqa * ncols, row = (h / gqa) * R + (h % gqa) * ncols + j;
    const int lane = threadIdx.x, b = lane / 4, part = lane % 4;
    const float4 * x = (const float4 *) ((const float *) (Q + (size_t) j * nb1 + (size_t) h * nb2) + 8 * lane);
    const float4 v0 = x[0], v1 = x[1];
    const float xs[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
    float m = xs[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = rq_fa_absmax(m, xs[k]);
    m = rq_fa_absmax(m, __shfl_xor_sync(0xffffffff, m, 1, 32));
    m = rq_fa_absmax(m, __shfl_xor_sync(0xffffffff, m, 2, 32));
    const float d = m / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    // elements 8 part + k: parts 0 / 1 are the even nibbles of words 0-1 / 2-3, parts 2 / 3 their odd nibbles
    const int odd = part / 2;
    unsigned w[2] = { 0u, 0u };
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int q = (int) rintf(fminf(fmaxf(xs[k] * id, -8.0f), 7.0f));
        w[k / 4] |= (unsigned) (q & 15) << (4 * (2 * (k % 4) + odd));
    }
    w[0] |= __shfl_xor_sync(0xffffffff, w[0], 2, 32);
    w[1] |= __shfl_xor_sync(0xffffffff, w[1], 2, 32);
    if (part < 2) {
        Qq[((size_t) row * 8 + b) * 4 + 2 * part]     = (int) w[0];
        Qq[((size_t) row * 8 + b) * 4 + 2 * part + 1] = (int) w[1];
    }
    if (part == 0) Qd[(size_t) row * 8 + b] = d * scale;
}

template <int MT>
static constexpr size_t rq_fa_smem_bytes() {
    return sizeof(int) * (2 * RQ_FA_TILE * RQ_FA_ROWW + MT * 8 * 64) +
           sizeof(float) * (16 * MT * 8 + 16 * MT * RQ_FA_SROW + 16 * RQ_FA_TILE + 3 * 16 * MT);
}

// grid (chunks, KV heads), NW warps. R = gqa * ncols query rows per KV head, at most 16 MT.
template <int MT, int NW>
static __global__ void __launch_bounds__(32 * NW) rq_fa_tc_kernel(
        const int * __restrict__ Qq, const float * __restrict__ Qd, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ parts, float2 * __restrict__ meta, float * __restrict__ dst,
        size_t nbk1, size_t nbk2, size_t nbv1, size_t nbv2, size_t nbm1,
        int R, int ncols, int gqa, int nheads, int n_kv, int chunk) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    constexpr int RMAX = 16 * MT, CT = RQ_FA_D / 8 / NW; // channel tiles of 8 per warp in P.V
    extern __shared__ int rq_fa_smem[];
    int   * ks = rq_fa_smem;                       // [TILE][ROWW] raw q4_0 rows
    int   * vs = ks + RQ_FA_TILE * RQ_FA_ROWW;     // [TILE][ROWW]
    int   * qf = vs + RQ_FA_TILE * RQ_FA_ROWW;     // Q fragments [MT][8 blocks][32 lanes][2]
    float * qd = (float *) (qf + MT * 8 * 64);     // [RMAX][8]
    float * sp = qd + RMAX * 8;                    // S [RMAX][SROW], then P (f16) over it
    float * mk = sp + RMAX * RQ_FA_SROW;           // the tile's mask per query token [16][TILE]
    float * ms = mk + 16 * RQ_FA_TILE, * ls = ms + RMAX, * as = ls + RMAX;

    const int g = blockIdx.y, nchunks = gridDim.x;
    const int t_begin = blockIdx.x * chunk, t_end = min(t_begin + chunk, n_kv);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, gid = lane / 4, tig = lane % 4;
    const char * Kg = K + (size_t) g * nbk2;
    const char * Vg = V + (size_t) g * nbv2;

    // Q fragments: (m-tile, block, lane) holds row 16 mt + gid's word tig and row + 8's; rows past R are zero
    for (int i = threadIdx.x; i < MT * 8 * 32; i += 32 * NW) {
        const int mt = i / 256, b = (i / 32) % 8, l = i % 32, r0 = mt * 16 + l / 4, wd = l % 4;
        qf[2 * i]     = r0 < R ? Qq[(((size_t) g * R + r0) * 8 + b) * 4 + wd] : 0;
        qf[2 * i + 1] = r0 + 8 < R ? Qq[(((size_t) g * R + r0 + 8) * 8 + b) * 4 + wd] : 0;
    }
    for (int i = threadIdx.x; i < RMAX * 8; i += 32 * NW) qd[i] = i / 8 < R ? Qd[(size_t) g * R * 8 + i] : 0.0f;
    for (int r = threadIdx.x; r < RMAX; r += 32 * NW) {
        ms[r] = -INFINITY;
        ls[r] = 0.0f;
        as[r] = 1.0f;
    }
    // rows past R keep S = 0 and so P = 0 (two zero halves are a zero float)
    for (int i = threadIdx.x; i < RMAX * RQ_FA_SROW; i += 32 * NW) sp[i] = 0.0f;

    float o[MT][CT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ct = 0; ct < CT; ++ct)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[mt][ct][i] = 0.0f;
    const int c0 = warp * CT * 8; // this warp's first output channel

    for (int t0 = t_begin; t0 < t_end; t0 += RQ_FA_TILE) {
        __syncthreads(); // the previous tile's P.V is done with ks, vs and P
        for (int i = threadIdx.x; i < RQ_FA_TILE * 36; i += 32 * NW) {
            const int tr = i / 36, cw = i % 36;
            const bool in = t0 + tr < t_end;
            ks[tr * RQ_FA_ROWW + cw] = in ? *(const int *) (Kg + (size_t) (t0 + tr) * nbk1 + 4 * cw) : 0;
            vs[tr * RQ_FA_ROWW + cw] = in ? *(const int *) (Vg + (size_t) (t0 + tr) * nbv1 + 4 * cw) : 0;
        }
        for (int i = threadIdx.x; i < ncols * RQ_FA_TILE; i += 32 * NW) {
            const int col = i / RQ_FA_TILE, tt = i % RQ_FA_TILE;
            const bool in = t0 + tt < t_end;
            mk[i] = !in ? -INFINITY : mask ? __half2float(((const half *) (mask + (size_t) col * nbm1))[t0 + tt]) : 0.0f;
        }
        __syncthreads();

        // S = Q K^T: warp w takes the 8-token tiles w, w + NW, ...
        for (int nt = warp; nt < RQ_FA_TILE / 8; nt += NW) {
            const int * kr  = ks + (nt * 8 + gid) * RQ_FA_ROWW;     // B operand: token gid of the tile
            const int * kc0 = ks + (nt * 8 + 2 * tig) * RQ_FA_ROWW; // C columns: tokens 2 tig, 2 tig + 1
            const int * kc1 = kc0 + RQ_FA_ROWW;
            float s[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int i = 0; i < 4; ++i) s[mt][i] = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const int kb = rq_fa_bytes(kr, 18 * b + 2 + 4 * tig) ^ 0x88888888;
                const float dk0 = rq_fa_half(kc0, 18 * b), dk1 = rq_fa_half(kc1, 18 * b);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    int d[4];
                    rq_fa_mma_s4(d, qf[2 * ((mt * 8 + b) * 32 + lane)], qf[2 * ((mt * 8 + b) * 32 + lane) + 1], kb);
                    const float q0 = qd[(mt * 16 + gid) * 8 + b], q1 = qd[(mt * 16 + gid + 8) * 8 + b];
                    s[mt][0] += (float) d[0] * (dk0 * q0);
                    s[mt][1] += (float) d[1] * (dk1 * q0);
                    s[mt][2] += (float) d[2] * (dk0 * q1);
                    s[mt][3] += (float) d[3] * (dk1 * q1);
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = mt * 16 + gid + 8 * (i / 2), t = nt * 8 + 2 * tig + i % 2;
                    if (r < R) sp[r * RQ_FA_SROW + t] = s[mt][i] + mk[(r % ncols) * RQ_FA_TILE + t];
                }
        }
        __syncthreads();

        // online softmax: warp w takes rows w, w + NW, ...; a lane tokens lane and lane + 32
        for (int r = warp; r < R; r += NW) {
            float * row = sp + r * RQ_FA_SROW;
            const float s0 = row[lane], s1 = row[lane + 32];
            const float mt_ = warp_reduce_max<32>(fmaxf(s0, s1));
            const float m_old = ms[r];
            const float m_new = fmaxf(m_old, mt_);
            const float p0 = m_new == -INFINITY ? 0.0f : __expf(s0 - m_new);
            const float p1 = m_new == -INFINITY ? 0.0f : __expf(s1 - m_new);
            const float lt = warp_reduce_sum<32>(p0 + p1);
            __syncwarp();
            half * prow = (half *) row;
            prow[lane]      = __float2half(p0);
            prow[lane + 32] = __float2half(p1);
            if (lane == 0) {
                const float alpha = m_new == -INFINITY ? 1.0f : __expf(m_old - m_new);
                ms[r] = m_new;
                ls[r] = ls[r] * alpha + lt;
                as[r] = alpha;
            }
        }
        __syncthreads();

        // P.V: this warp's channels c0 .. c0 + 8 CT - 1 for every row
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const float a0 = as[mt * 16 + gid], a1 = as[mt * 16 + gid + 8];
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                o[mt][ct][0] *= a0;
                o[mt][ct][1] *= a0;
                o[mt][ct][2] *= a1;
                o[mt][ct][3] *= a1;
            }
        }
        const half * P = (const half *) sp;
#pragma unroll
        for (int kt = 0; kt < RQ_FA_TILE / 16; ++kt) {
            uint32_t pa[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const half * p0 = P + (size_t) (mt * 16 + gid) * (2 * RQ_FA_SROW) + kt * 16 + 2 * tig;
                const half * p1 = p0 + 8 * (2 * RQ_FA_SROW);
                pa[mt][0] = *(const uint32_t *) p0;
                pa[mt][1] = *(const uint32_t *) p1;
                pa[mt][2] = *(const uint32_t *) (p0 + 8);
                pa[mt][3] = *(const uint32_t *) (p1 + 8);
            }
            const int tA = kt * 16 + 2 * tig; // B rows (tokens) tA, tA + 1, tA + 8, tA + 9
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                const int c = c0 + ct * 8 + gid, cb = c / 32, cbyte = cb * 18 + 2 + (c % 16), csh = 4 * ((c % 32) / 16);
                half v[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int * vr = vs + (tA + (k % 2) + 8 * (k / 2)) * RQ_FA_ROWW;
                    const int q = (((unsigned) vr[cbyte >> 2] >> (8 * (cbyte & 3) + csh)) & 15) - 8;
                    v[k] = __float2half(rq_fa_half(vr, 18 * cb) * (float) q);
                }
                const half2 vb0 = __halves2half2(v[0], v[1]), vb1 = __halves2half2(v[2], v[3]);
                const uint32_t b0 = *(const uint32_t *) &vb0, b1 = *(const uint32_t *) &vb1;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) rq_fa_mma_f16(o[mt][ct], pa[mt], b0, b1);
            }
        }
    }

    // the chunk's unnormalized sums and (max, sum) per row, or the normalized output if there is one chunk
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ct = 0; ct < CT; ++ct)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = mt * 16 + gid + 8 * (i / 2), c = c0 + ct * 8 + 2 * tig + i % 2;
                if (r >= R) continue;
                const int h = g * gqa + r / ncols, col = r % ncols;
                const size_t jdst = (size_t) col * nheads + h;
                if (nchunks == 1) {
                    dst[jdst * RQ_FA_D + c] = ls[r] > 0.0f ? o[mt][ct][i] / ls[r] : 0.0f;
                } else {
                    parts[(jdst * nchunks + blockIdx.x) * RQ_FA_D + c] = o[mt][ct][i];
                }
            }
    if (nchunks > 1) {
        for (int r = threadIdx.x; r < R; r += 32 * NW) {
            const int h = g * gqa + r / ncols, col = r % ncols;
            meta[((size_t) col * nheads + h) * nchunks + blockIdx.x] = make_float2(ms[r], ls[r]);
        }
    }
#else
    GGML_UNUSED(Qq); GGML_UNUSED(Qd); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask); GGML_UNUSED(parts); GGML_UNUSED(meta);
    GGML_UNUSED(dst); GGML_UNUSED(nbk1); GGML_UNUSED(nbk2); GGML_UNUSED(nbv1); GGML_UNUSED(nbv2); GGML_UNUSED(nbm1);
    GGML_UNUSED(R); GGML_UNUSED(ncols); GGML_UNUSED(gqa); GGML_UNUSED(nheads); GGML_UNUSED(n_kv); GGML_UNUSED(chunk);
    NO_DEVICE_CODE;
#endif
}

static int rq_fa_mode() {
    // on by default (int4 Q); RQ_FA_DOT8=0 keeps llama.cpp's kernels
    static const int mode = getenv("RQ_FA_DOT8") ? atoi(getenv("RQ_FA_DOT8")) : 1;
    return mode;
}

bool ggml_cuda_rq_fattn_q4_applies(const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (rq_fa_mode() == 0 || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < RQ_FA_TC_MIN_CC || cc >= RQ_FA_TC_MAX_CC) {
        return false;
    }
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    return K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 && Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           K->ne[0] == RQ_FA_D && V->ne[0] == RQ_FA_D && Q->ne[0] == RQ_FA_D && Q->ne[1] <= 16 && Q->ne[3] == 1 && K->ne[3] == 1 &&
           Q->ne[2] % K->ne[2] == 0 && (Q->ne[2] / K->ne[2]) * Q->ne[1] <= 96 && K->ne[1] % RQ_FA_TILE == 0 &&
           Q->nb[0] == sizeof(float) && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 &&
           V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 && (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1)) &&
           !sinks && max_bias == 0.0f && softcap == 0.0f;
#endif
}

template <int MT, int NW>
static void rq_fa_launch(dim3 grid, cudaStream_t s, const int * Qq, const float * Qd, const ggml_tensor * K, const ggml_tensor * V,
                         const ggml_tensor * mask, float * parts, float2 * meta, float * dst, int R, int ncols, int gqa, int nheads,
                         int chunk) {
    constexpr size_t smem = rq_fa_smem_bytes<MT>();
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = ggml_cuda_get_device();
    if (!attr_set[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(rq_fa_tc_kernel<MT, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr_set[dev] = true;
    }
    rq_fa_tc_kernel<MT, NW><<<grid, 32 * NW, smem, s>>>(Qq, Qd, (const char *) K->data, (const char *) V->data,
                                                        mask ? (const char *) mask->data : nullptr, parts, meta, dst, K->nb[1], K->nb[2],
                                                        V->nb[1], V->nb[2], mask ? mask->nb[1] : 0, R, ncols, gqa, nheads,
                                                        (int) K->ne[1], chunk);
}

void ggml_cuda_rq_fattn_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t s = ctx.stream();
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int ncols = (int) Q->ne[1], nheads = (int) Q->ne[2], nkvh = (int) K->ne[2];
    const int gqa = nheads / nkvh, R = gqa * ncols;

    ggml_cuda_pool_alloc<int>   Qq(ctx.pool(), (size_t) nheads * ncols * 32);
    ggml_cuda_pool_alloc<float> Qd(ctx.pool(), (size_t) nheads * ncols * 8);
    rq_fa_quant_q<<<nheads * ncols, 32, 0, s>>>((const char *) Q->data, Q->nb[1], Q->nb[2], ncols, gqa, scale, Qq.get(), Qd.get());

    // the KV sequence over ~2 blocks per SM, whole tiles each
    const int ntiles = (int) K->ne[1] / RQ_FA_TILE;
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    static const int env_chunks = getenv("RQ_FA_CHUNKS") ? atoi(getenv("RQ_FA_CHUNKS")) : 0;
    int nchunks = env_chunks > 0 ? env_chunks : std::max(1, std::min(ntiles, (2 * nsm + nkvh - 1) / nkvh));
    const int chunk = (ntiles + nchunks - 1) / nchunks * RQ_FA_TILE;
    nchunks = (int) ((K->ne[1] + chunk - 1) / chunk);

    ggml_cuda_pool_alloc<float>  parts(ctx.pool(), nchunks > 1 ? (size_t) nchunks * nheads * ncols * RQ_FA_D : 1);
    ggml_cuda_pool_alloc<float2> meta(ctx.pool(), nchunks > 1 ? (size_t) nchunks * nheads * ncols : 1);
    const dim3 grid(nchunks, nkvh, 1);
    float * d = (float *) dst->data;
    // P.V keeps 4 MT x (32 / NW) accumulators per thread: 4 warps up to 32 rows, 8 beyond
    if (R <= 16)      rq_fa_launch<1, 4>(grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), d, R, ncols, gqa, nheads, chunk);
    else if (R <= 32) rq_fa_launch<2, 4>(grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), d, R, ncols, gqa, nheads, chunk);
    else if (R <= 48) rq_fa_launch<3, 8>(grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), d, R, ncols, gqa, nheads, chunk);
    else if (R <= 64) rq_fa_launch<4, 8>(grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), d, R, ncols, gqa, nheads, chunk);
    else              rq_fa_launch<6, 8>(grid, s, Qq.get(), Qd.get(), K, V, mask, parts.get(), meta.get(), d, R, ncols, gqa, nheads, chunk);
    if (nchunks > 1) {
        flash_attn_combine_results<RQ_FA_D><<<dim3(ncols, nheads, 1), RQ_FA_D, nchunks * sizeof(float2), s>>>(
            parts.get(), meta.get(), d, nchunks);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // GGML_USE_HIP

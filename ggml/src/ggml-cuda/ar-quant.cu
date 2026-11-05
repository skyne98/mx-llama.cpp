#include "ar-quant.cuh"
#include "cpy-utils.cuh"

template <typename block> struct ar_q;

template <> struct ar_q<block_q4_0> {
    static constexpr int qk = QK4_0;
    static __device__ void quant(const float * x, block_q4_0 * y) { quantize_f32_q4_0_block(x, y); }
    static __device__ void dequant(const block_q4_0 * b, float * x) {
        const float d = __half2float(b->d);
        for (int j = 0; j < qk/2; ++j) {
            x[j]        = ((b->qs[j] & 0xF) - 8)*d;
            x[j + qk/2] = ((b->qs[j] >>  4) - 8)*d;
        }
    }
};

template <> struct ar_q<block_q4_1> {
    static constexpr int qk = QK4_1;
    // not quantize_f32_q4_1_block: its y->dm.x = d stores garbage scales on ROCm 5.7's __half2
    static __device__ void quant(const float * x, block_q4_1 * y) {
        float vmin = x[0];
        float vmax = x[0];
        for (int j = 1; j < qk; ++j) {
            vmin = fminf(vmin, x[j]);
            vmax = fmaxf(vmax, x[j]);
        }
        const float d  = (vmax - vmin)/15;
        const float id = d ? 1.0f/d : 0.0f;
        y->dm = __floats2half2_rn(d, vmin);
        for (int j = 0; j < qk/2; ++j) {
            const int q0 = min(15, (int) ((x[j]        - vmin)*id + 0.5f));
            const int q1 = min(15, (int) ((x[j + qk/2] - vmin)*id + 0.5f));
            y->qs[j] = q0 | (q1 << 4);
        }
    }
    static __device__ void dequant(const block_q4_1 * b, float * x) {
        const float d = __low2float(b->dm);
        const float m = __high2float(b->dm);
        for (int j = 0; j < qk/2; ++j) {
            x[j]        = (b->qs[j] & 0xF)*d + m;
            x[j + qk/2] = (b->qs[j] >>  4)*d + m;
        }
    }
};

template <> struct ar_q<block_q5_0> {
    static constexpr int qk = QK5_0;
    static __device__ void quant(const float * x, block_q5_0 * y) { quantize_f32_q5_0_block(x, y); }
    static __device__ void dequant(const block_q5_0 * b, float * x) {
        const float d = __half2float(b->d);
        uint32_t qh;
        memcpy(&qh, b->qh, sizeof(qh));
        for (int j = 0; j < qk/2; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            x[j]        = (((b->qs[j] & 0xF) | xh0) - 16)*d;
            x[j + qk/2] = (((b->qs[j] >>  4) | xh1) - 16)*d;
        }
    }
};

template <> struct ar_q<block_q8_0> {
    static constexpr int qk = QK8_0;
    static __device__ void quant(const float * x, block_q8_0 * y) { quantize_f32_q8_0_block(x, y); }
    static __device__ void dequant(const block_q8_0 * b, float * x) {
        const float d = __half2float(b->d);
        for (int j = 0; j < qk; ++j) {
            x[j] = b->qs[j]*d;
        }
    }
};

template <typename block>
static __global__ void k_ar_quantize(const float * __restrict__ x, block * __restrict__ y, const int64_t nb) {
    const int64_t ib = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nb) {
        return;
    }
    ar_q<block>::quant(x + ib*ar_q<block>::qk, y + ib);
}

template <typename block>
static __global__ void k_ar_reduce(float * __restrict__ own, const char * __restrict__ recv, const size_t slot_bytes,
        const int nranks, const int rank, const int64_t nb) {
    constexpr int qk = ar_q<block>::qk;
    const int64_t ib = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nb) {
        return;
    }
    float x[qk];
    for (int j = 0; j < qk; ++j) {
        x[j] = own[ib*qk + j];
    }
    for (int r = 0; r < nranks; ++r) {
        if (r == rank) {
            continue;
        }
        float v[qk];
        ar_q<block>::dequant((const block *) (recv + r*slot_bytes) + ib, v);
        for (int j = 0; j < qk; ++j) {
            x[j] += v[j];
        }
    }
    for (int j = 0; j < qk; ++j) {
        own[ib*qk + j] = x[j];
    }
}

// k-quant super-blocks (Q4_K, Q5_K: 256 values in 8 sub-blocks of 32, each with a 6-bit scale and min, x = d*q - m):
// one thread per sub-block, the 8 threads of a super-block adjacent in the warp. A sub-block's scale and min come
// from its own min and max (the min clamped to <= 0, as in ggml); ggml's CPU quantizer also searches them.
template <typename block> struct ar_k;
template <> struct ar_k<block_q4_K> { static constexpr int nmax = 15; };
template <> struct ar_k<block_q5_K> { static constexpr int nmax = 31; };

static __device__ __forceinline__ void ar_k_scale_min(const uint8_t * q, const int j, int & sc, int & m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
    }
}

template <typename block>
static __device__ void ar_k_dequant(const block * b, const int j, float * x) {
    int sc, m;
    ar_k_scale_min(b->scales, j, sc, m);
    const float d  = __low2float(b->dm)*sc;
    const float mn = __high2float(b->dm)*m;
    const uint8_t * q = b->qs + 32*(j/2);
    const int sh = 4*(j & 1);
    for (int l = 0; l < 32; ++l) {
        int v = (q[l] >> sh) & 0xF;
        if constexpr (ar_k<block>::nmax == 31) {
            v |= ((b->qh[l] >> j) & 1) << 4;
        }
        x[l] = d*v - mn;
    }
}

template <typename block>
static __global__ void k_ar_quantize_k(const float * __restrict__ x, block * __restrict__ y, const int64_t nb) {
    constexpr int nmax = ar_k<block>::nmax;
    const int64_t t  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int     j  = t % 8;
    const bool    ok = t/8 < nb; // whole super-blocks drop out together, the shuffles stay within one
    const int64_t ib = ok ? t/8 : nb - 1;

    float v[32];
    float vmin = 0.0f;
    float vmax = -FLT_MAX;
    for (int l = 0; l < 32; ++l) {
        v[l] = x[ib*QK_K + 32*j + l];
        vmin = fminf(vmin, v[l]);
        vmax = fmaxf(vmax, v[l]);
    }
    const float sc = (vmax - vmin)/nmax;
    const float mn = -vmin;
    float msc = sc;
    float mmn = mn;
    for (int o = 4; o > 0; o >>= 1) {
        msc = fmaxf(msc, __shfl_xor_sync(0xffffffff, msc, o, 8));
        mmn = fmaxf(mmn, __shfl_xor_sync(0xffffffff, mmn, o, 8));
    }
    const int ls = msc > 0.0f ? min(63, __float2int_rn(63.0f*sc/msc)) : 0;
    const int lm = mmn > 0.0f ? min(63, __float2int_rn(63.0f*mn/mmn)) : 0;
    const half2 dm = __floats2half2_rn(msc/63, mmn/63);
    const float d  = __low2float(dm)*ls;
    const float m  = __high2float(dm)*lm;
    const float id = d > 0.0f ? 1.0f/d : 0.0f;

    uint32_t w[8] = {0}; // this sub-block's low nibbles, at its half of each byte
    uint32_t h    = 0;   // Q5_K: bit l = the high bit of value l
    for (int l = 0; l < 32; ++l) {
        const int q = max(0, min(nmax, __float2int_rn((v[l] + m)*id)));
        w[l/4] |= (uint32_t) (q & 0xF) << (8*(l % 4) + 4*(j & 1));
        h      |= (uint32_t) (q >> 4) << l;
    }
    const int ls4 = __shfl_xor_sync(0xffffffff, ls, 4, 8);
    const int lm4 = __shfl_xor_sync(0xffffffff, lm, 4, 8);
    for (int k = 0; k < 8; ++k) {
        w[k] |= __shfl_xor_sync(0xffffffff, w[k], 1, 8);
    }
    uint32_t hs[8];
    for (int k = 0; k < 8; ++k) {
        hs[k] = __shfl_sync(0xffffffff, h, k, 8);
    }
    if (!ok) {
        return;
    }
    block * b = y + ib;
    if (j == 0) {
        b->dm = dm;
    }
    if (j < 4) {
        b->scales[j]     = (ls & 63) | ((ls4 >> 4) << 6);
        b->scales[j + 4] = (lm & 63) | ((lm4 >> 4) << 6);
    } else {
        b->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
    }
    uint32_t * qs = (uint32_t *) (b->qs + 32*(j/2));
    for (int k = 4*(j & 1); k < 4*(j & 1) + 4; ++k) {
        qs[k] = w[k];
    }
    if constexpr (nmax == 31) {
        for (int l = 4*j; l < 4*j + 4; ++l) {
            uint8_t qh = 0;
            for (int k = 0; k < 8; ++k) {
                qh |= ((hs[k] >> l) & 1) << k;
            }
            b->qh[l] = qh;
        }
    }
}

template <typename block>
static __global__ void k_ar_reduce_k(float * __restrict__ own, const char * __restrict__ recv, const size_t slot_bytes,
        const int nranks, const int rank, const int64_t nb) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nb*8) {
        return;
    }
    const int64_t ib = t/8;
    const int     j  = t % 8;
    float x[32];
    for (int l = 0; l < 32; ++l) {
        x[l] = own[ib*QK_K + 32*j + l];
    }
    for (int r = 0; r < nranks; ++r) {
        if (r == rank) {
            continue;
        }
        float v[32];
        ar_k_dequant((const block *) (recv + r*slot_bytes) + ib, j, v);
        for (int l = 0; l < 32; ++l) {
            x[l] += v[l];
        }
    }
    for (int l = 0; l < 32; ++l) {
        own[ib*QK_K + 32*j + l] = x[l];
    }
}

template <typename block>
static void ar_quantize_k(const float * x, void * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = ne/QK_K;
    k_ar_quantize_k<block><<<(8*nb + 255)/256, 256, 0, stream>>>(x, (block *) y, nb);
}

template <typename block>
static void ar_reduce_k(float * own, const void * recv, const size_t slot_bytes, const int nranks, const int rank,
        const int64_t ne, cudaStream_t stream) {
    const int64_t nb = ne/QK_K;
    k_ar_reduce_k<block><<<(8*nb + 255)/256, 256, 0, stream>>>(own, (const char *) recv, slot_bytes, nranks, rank, nb);
}

template <typename block>
static void ar_quantize(const float * x, void * y, const int64_t ne, cudaStream_t stream) {
    const int64_t nb = ne/ar_q<block>::qk;
    k_ar_quantize<block><<<(nb + 255)/256, 256, 0, stream>>>(x, (block *) y, nb);
}

template <typename block>
static void ar_reduce(float * own, const void * recv, const size_t slot_bytes, const int nranks, const int rank,
        const int64_t ne, cudaStream_t stream) {
    const int64_t nb = ne/ar_q<block>::qk;
    k_ar_reduce<block><<<(nb + 255)/256, 256, 0, stream>>>(own, (const char *) recv, slot_bytes, nranks, rank, nb);
}

void ggml_cuda_ar_quantize(ggml_type type, const float * x, void * y, int64_t ne, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q4_0: ar_quantize<block_q4_0>(x, y, ne, stream); break;
        case GGML_TYPE_Q4_1: ar_quantize<block_q4_1>(x, y, ne, stream); break;
        case GGML_TYPE_Q5_0: ar_quantize<block_q5_0>(x, y, ne, stream); break;
        case GGML_TYPE_Q4_K: ar_quantize_k<block_q4_K>(x, y, ne, stream); break;
        case GGML_TYPE_Q5_K: ar_quantize_k<block_q5_K>(x, y, ne, stream); break;
        case GGML_TYPE_Q8_0: ar_quantize<block_q8_0>(x, y, ne, stream); break;
        default: GGML_ABORT("unsupported all-reduce wire type %s", ggml_type_name(type));
    }
}

void ggml_cuda_ar_reduce(ggml_type type, float * own, const void * recv, size_t slot_bytes, int nranks, int rank,
                         int64_t ne, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q4_0: ar_reduce<block_q4_0>(own, recv, slot_bytes, nranks, rank, ne, stream); break;
        case GGML_TYPE_Q4_1: ar_reduce<block_q4_1>(own, recv, slot_bytes, nranks, rank, ne, stream); break;
        case GGML_TYPE_Q5_0: ar_reduce<block_q5_0>(own, recv, slot_bytes, nranks, rank, ne, stream); break;
        case GGML_TYPE_Q4_K: ar_reduce_k<block_q4_K>(own, recv, slot_bytes, nranks, rank, ne, stream); break;
        case GGML_TYPE_Q5_K: ar_reduce_k<block_q5_K>(own, recv, slot_bytes, nranks, rank, ne, stream); break;
        case GGML_TYPE_Q8_0: ar_reduce<block_q8_0>(own, recv, slot_bytes, nranks, rank, ne, stream); break;
        default: GGML_ABORT("unsupported all-reduce wire type %s", ggml_type_name(type));
    }
}

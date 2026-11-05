#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q2_0_ref(const float * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);

// ruquant W4A4 (see block_w4a4 in ggml-common.h). The quantizers take one whole row (k = K).
// ggml_rq_hadamard: x <- x H, H = (H_p kron H_m) / sqrt(n), n = p * m, m a power of 2, H_p a Paley
// Hadamard (p - 1 prime, = 3 mod 4) or 1; with transpose, x <- x H^T. Fixed operation order, so every
// backend gets the same bits.
GGML_API bool ggml_rq_hadamard_factor(int64_t n, int * p, int * m);
GGML_API float ggml_rq_act_clip(bool hadamard);
GGML_API int   ggml_rq_act_group(void); // activation scale group: 64, or RQ_ACT_GROUP = 128 / 256
GGML_API float ggml_rq_act_clip_for(float tensor_clip, bool hadamard); // env, else the tensor's, else 1
GGML_API float ggml_rq_tensor_clip(const void * blocks);               // a W4A4 tensor's clip (block 0 header)
// a W4A4H tensor's Hadamard length (block 0 header pad[0]; the rotation is block-diagonal), checked against a row
// (slice) of k: aborts unless it divides k and factors
GGML_API int64_t ggml_rq_tensor_rot(const void * blocks, int64_t k);
GGML_API int64_t ggml_rq_rot_check(int32_t n, int64_t k);
GGML_API void quantize_row_a4_clip (const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k, float clip);
GGML_API void quantize_row_a4h_clip(const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k, float clip);
GGML_API void quantize_row_a4h_clip_n(const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k, float clip, int64_t n);
GGML_API void ggml_rq_hadamard(float * GGML_RESTRICT x, int64_t n, bool transpose);
GGML_API void quantize_row_w4a4_ref (const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_w4a4h_ref(const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_a4_ref   (const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_a4h_ref  (const float * GGML_RESTRICT x, block_w4a4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_w4a4 (const block_w4a4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_w4a4h(const block_w4a4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_a4   (const block_w4a4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_a4h  (const block_w4a4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_w4a4 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_w4a4h(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif

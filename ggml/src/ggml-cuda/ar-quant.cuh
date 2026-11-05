#pragma once

#include "common.cuh"

// Quantized payloads for prefill-sized all-reduces (GGML_NCCL_AR_WIRE=q4_0 / q4_1 / q5_0 / q8_0 / q4_K / q5_K):
// blocks of 32 values with their own scale, or k-quant super-blocks of 256, in ggml's block layouts, so ggml's
// to_fp32 dequantizes the gathered result.

// x (ne floats) -> y (ne/block size blocks of type)
void ggml_cuda_ar_quantize(ggml_type type, const float * x, void * y, int64_t ne, cudaStream_t stream);

// The reduce-scatter step on rank `rank`: own (ne floats, this rank's exact partial chunk) += the dequantized
// chunks of the other ranks, which sit in recv at r*slot_bytes for rank r. rank -1 adds every slot.
void ggml_cuda_ar_reduce(ggml_type type, float * own, const void * recv, size_t slot_bytes, int nranks, int rank,
                         int64_t ne, cudaStream_t stream);

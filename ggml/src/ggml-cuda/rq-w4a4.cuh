#include "common.cuh"

// ruquant W4A4 / W4A4H matmul: optional online Hadamard, per-token int4 activation quantization,
// int4 x int4 GEMM with int32 accumulation, the spec epilogue. Same bits as the CPU reference.
bool ggml_cuda_rq_w4a4_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// false while the weight still needs its one-time in-place repack (skinny path), which must not be captured in a graph
bool ggml_cuda_rq_w4a4_ready(const ggml_tensor * src0);
void ggml_cuda_mul_mat_rq_w4a4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// b = W_b x can run in one launch with a = W_a x (same x, type, K and clip, skinny widths)
bool ggml_cuda_rq_w4a4_can_fuse(const ggml_tensor * a, const ggml_tensor * b);
// n <= 3 such matmuls (all can_fuse with the first) as one
void ggml_cuda_mul_mat_rq_w4a4_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, ggml_tensor * const * dsts,
                                     int n, const ggml_tensor * src1);
// starts a new graph evaluation of ctx: drops the shared quantized activation of the previous one
void ggml_cuda_rq_graph_begin(ggml_backend_cuda_context & ctx);

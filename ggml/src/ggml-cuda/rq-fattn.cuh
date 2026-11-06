#pragma once
#include "common.cuh"

// ruquant attention over a q4_0 KV cache at decode / verify widths, with v_dot8 (int4 x int4) for Q.K on gfx906
// (rq-fattn.cu). Env RQ_FA_DOT8: 1 int4 Q (default), 2 int8 Q as two int4 planes, 0 off.
bool ggml_cuda_rq_fattn_q4_applies(const ggml_tensor * dst);
void ggml_cuda_rq_fattn_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

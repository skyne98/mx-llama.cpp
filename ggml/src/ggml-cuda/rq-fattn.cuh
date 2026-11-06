#pragma once
#include "common.cuh"

// ruquant attention over a q4_0 KV cache at decode / verify widths with int4 x int4 Q.K (rq-fattn.cu): v_dot8 on
// gfx906, INT4 tensor cores on NVIDIA Turing / Ampere / Ada. Env RQ_FA_DOT8: 1 int4 Q (default), 2 int8 Q as two int4
// planes (gfx906 only), 0 off.
bool ggml_cuda_rq_fattn_q4_applies(const ggml_tensor * dst);
void ggml_cuda_rq_fattn_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifndef MDL_CUDNN_H
#define MDL_CUDNN_H

/**
 * @file
 * @brief C ABI of the cuDNN kernels (library `mdl_cudnn`) of the kernel package `cudnn` (see package.yml.in).
 * @details For tensors in CUDA device memory; compiled models call these symbols where the package's implementations are
 * selected (see docs/kernels.md). bfloat16 (`_bf16`) and float16 (`_f16`; both `uint16_t*` bit patterns). The kernels
 * run on TVM's current stream of the current device (TVMFFIEnvGetStream). Every function returns 0 on success and a
 * non-zero value on failure (e.g. a configuration that cuDNN does not support).
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fused attention `out = softmax(scale * Q @ K^T) @ V` (cuDNN's scaled dot product attention), over two batch
 * dimensions i < b0, j < b1 (e.g. batch and heads).
 * @details Every tensor is a strided view (strides in elements) of [b0, b1, seq, dim]: Q [b0, b1, Sq, D], K [b0, b1, Skv,
 * D], V [b0, b1, Skv, Dv], out [b0, b1, Sq, Dv], e.g. the layout [batch, seq, heads, dim] of the model in place. The
 * score matrix is never stored. The head dimension must be contiguous (stride 1).
 */
#define MDL_ATTENTION_DECLARE(suffix, type)                                                                            \
	int mdl_attention_##suffix(                                                                                        \
			type* out, const type* Q, const type* K, const type* V, int64_t b0, int64_t b1, int64_t Sq, int64_t Skv,   \
			int64_t D, int64_t Dv, int64_t q_s0, int64_t q_s1, int64_t q_sS, int64_t q_sD, int64_t k_s0, int64_t k_s1, \
			int64_t k_sS, int64_t k_sD, int64_t v_s0, int64_t v_s1, int64_t v_sS, int64_t v_sD, int64_t o_s0,          \
			int64_t o_s1, int64_t o_sS, int64_t o_sD, float scale                                                      \
	);
MDL_ATTENTION_DECLARE(bf16, uint16_t)
MDL_ATTENTION_DECLARE(f16, uint16_t)
#undef MDL_ATTENTION_DECLARE

#ifdef __cplusplus
}
#endif

#endif

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

/** @brief The strides (in elements) of a tensor `x` viewed as [b0, b1, seq, dim]. */
#define MDL_ATTENTION_STRIDES(x) int64_t x##_s0, int64_t x##_s1, int64_t x##_sS, int64_t x##_sD

/**
 * @brief Fused attention `out = softmax(scale * Q @ K^T) @ V` (cuDNN's scaled dot product attention), over two batch
 * dimensions i < b0, j < b1 (e.g. batch and heads), and (if @p with_lse) the log-sum-exp of the rows of
 * `scale * Q @ K^T` for the backward pass.
 * @details Every tensor is a strided view (strides in elements) of [b0, b1, seq, dim]: Q [b0, b1, Sq, D], K [b0, b1, Skv,
 * D], V [b0, b1, Skv, Dv], out [b0, b1, Sq, Dv], e.g. the layout [batch, seq, heads, dim] of the model in place. The
 * score matrix is never stored. The head dimension must be contiguous (stride 1). lse is [b0, b1, Sq] float32
 * (contiguous; not written without @p with_lse).
 */
#define MDL_ATTENTION_DECLARE(suffix, type)                                                                            \
	int mdl_attention_fwd_##suffix(                                                                                    \
			type* out, float* lse, const type* Q, const type* K, const type* V, int64_t b0, int64_t b1, int64_t Sq,    \
			int64_t Skv, int64_t D, int64_t Dv, MDL_ATTENTION_STRIDES(q), MDL_ATTENTION_STRIDES(k),                    \
			MDL_ATTENTION_STRIDES(v), MDL_ATTENTION_STRIDES(o), int32_t with_lse, float scale                          \
	);                                                                                                                 \
	/**                                                                                                                \
	 * @brief The gradients dQ, dK, dV of fused attention for the cotangent dO of its result O, from its statistics lse \
	 * (see mdl_attention_fwd); all tensors strided views as for the forward pass (dQ, dK, dV like Q, K, V).            \
	 */                                                                                                                \
	int mdl_attention_bwd_##suffix(                                                                                    \
			type* dq, type* dk, type* dv, const type* Q, const type* K, const type* V, const type* O, const type* dO,  \
			const float* lse, int64_t b0, int64_t b1, int64_t Sq, int64_t Skv, int64_t D, int64_t Dv,                  \
			MDL_ATTENTION_STRIDES(q), MDL_ATTENTION_STRIDES(k), MDL_ATTENTION_STRIDES(v), MDL_ATTENTION_STRIDES(o),    \
			MDL_ATTENTION_STRIDES(do), MDL_ATTENTION_STRIDES(dq), MDL_ATTENTION_STRIDES(dk),                           \
			MDL_ATTENTION_STRIDES(dv), float scale                                                                     \
	);
MDL_ATTENTION_DECLARE(bf16, uint16_t)
MDL_ATTENTION_DECLARE(f16, uint16_t)
#undef MDL_ATTENTION_DECLARE

#ifdef __cplusplus
}
#endif

#endif

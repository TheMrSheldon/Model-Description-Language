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
 * @brief Fused attention `out = softmax(scale * Q @ K^T + bias) @ V` (cuDNN's scaled dot product attention), batched
 * over the leading dimensions, and (if @p with_lse) the log-sum-exp of the rows of the biased scores for the backward
 * pass. bias (or NULL) is a view of the scores' rank whose dimensions are the scores' or 1 (broadcast), e.g. a padding
 * mask (0 at valid positions, the most negative value elsewhere).
 * @details Every tensor is a strided view (strides in elements) of rank @p rank: Q [..., Sq, D], K [..., Skv, D],
 * V [..., Skv, Dv] with their shapes, out [..., Sq, Dv] with Q's batch dimensions (its strides only), e.g. the layout
 * [batch, seq, heads, dim] of the model in place. The score matrix is never stored. The head dimension must be
 * contiguous (stride 1). lse is [..., Sq] float32, contiguous (not written without @p with_lse).
 */
#define MDL_ATTENTION_DECLARE(suffix, type)                                                                            \
	int mdl_attention_fwd_##suffix(                                                                                    \
			type* out, float* lse, const type* Q, const type* K, const type* V, int64_t rank, const int64_t* q_shape,  \
			const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape,        \
			const int64_t* v_strides, const int64_t* o_strides, const type* bias, const int64_t* bias_shape,           \
			const int64_t* bias_strides, int32_t with_lse, float scale                                                 \
	);                                                                                                                 \
	/**                                                                                                                \
	 * @brief The gradients dQ, dK, dV of fused attention for the cotangent dO of its result O, from its statistics lse \
	 * (see mdl_attention_fwd); O and dO have out's shape, dQ, dK, dV the shapes of Q, K, V (strides only).            \
	 */                                                                                                                \
	int mdl_attention_bwd_##suffix(                                                                                    \
			type* dq, type* dk, type* dv, const type* Q, const type* K, const type* V, const type* O, const type* dO,  \
			const float* lse, int64_t rank, const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape,  \
			const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides,      \
			const int64_t* do_strides, const int64_t* dq_strides, const int64_t* dk_strides,                           \
			const int64_t* dv_strides, const type* bias, const int64_t* bias_shape, const int64_t* bias_strides,       \
			float scale                                                                                                \
	);
MDL_ATTENTION_DECLARE(bf16, uint16_t)
MDL_ATTENTION_DECLARE(f16, uint16_t)
#undef MDL_ATTENTION_DECLARE

#ifdef __cplusplus
}
#endif

#endif

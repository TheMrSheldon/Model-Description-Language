#ifndef MDL_RUNTIME_KERNELS_H
#define MDL_RUNTIME_KERNELS_H

/**
 * @file
 * @brief C ABI of the vendor kernels (library `mdl_kernels`) that compiled models call into.
 * @details The TVM backend emits `call_extern` calls to these symbols when it recognizes a matching subgraph (see the
 * backend's pattern registry and legalizers). All tensors are dense and row-major. Every kernel exists for float32
 * (`_f32`, `float*`), bfloat16 (`_bf16`) and float16 (`_f16`; both `uint16_t*` bit patterns); the scalar arguments are
 * float32 in all variants. 16-bit kernels compute in float32 internally where oneDNN supports the type, and otherwise
 * convert to float32 (e.g. float16 on CPUs without float16 arithmetic). Every function returns 0 on success and a
 * non-zero value on failure.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Batched matrix multiplication `dst[b] = alpha * (op(A[b]) @ op(B[b])) + bias`.
 * @param dst   [batch, M, N]
 * @param A     [batch, M, K] (or [batch, K, M] if @p trans_a)
 * @param B     [batchB, K, N] (or [batchB, N, K] if @p trans_b); batchB is either @p batch or 1 (broadcast)
 * @param bias  [N]; only read if @p has_bias is non-zero
 * @param batchB batch size of B (1 or @p batch)
 */
int mdl_matmul_f32(
		float* dst, const float* A, const float* B, const float* bias, int32_t has_bias, int64_t batch,
		int64_t batchB, int64_t M, int64_t K, int64_t N, int32_t trans_a, int32_t trans_b, float alpha
);
int mdl_matmul_bf16(
		uint16_t* dst, const uint16_t* A, const uint16_t* B, const uint16_t* bias, int32_t has_bias, int64_t batch,
		int64_t batchB, int64_t M, int64_t K, int64_t N, int32_t trans_a, int32_t trans_b, float alpha
);
int mdl_matmul_f16(
		uint16_t* dst, const uint16_t* A, const uint16_t* B, const uint16_t* bias, int32_t has_bias, int64_t batch,
		int64_t batchB, int64_t M, int64_t K, int64_t N, int32_t trans_a, int32_t trans_b, float alpha
);

/**
 * @brief Profiling hooks: models compiled with `mdlc --profile` call these around every kernel call. The times are
 * summed per kernel kind and printed to stderr at program exit.
 */
int mdl_profile_begin(const char* kernel);
int mdl_profile_end(const char* kernel);

/** @brief Softmax over the last axis of a [outer, axis] tensor (in-place allowed). */
int mdl_softmax_f32(float* dst, const float* src, int64_t outer, int64_t axis);
int mdl_softmax_bf16(uint16_t* dst, const uint16_t* src, int64_t outer, int64_t axis);
int mdl_softmax_f16(uint16_t* dst, const uint16_t* src, int64_t outer, int64_t axis);

/** @brief Exact (erf-based) GELU over @p n elements (in-place allowed). */
int mdl_gelu_f32(float* dst, const float* src, int64_t n);
int mdl_gelu_bf16(uint16_t* dst, const uint16_t* src, int64_t n);
int mdl_gelu_f16(uint16_t* dst, const uint16_t* src, int64_t n);

/** @brief Layer normalization over the last axis of a [rows, channels] tensor with scale (gamma) and shift (beta). */
int mdl_layer_norm_f32(
		float* dst, const float* src, const float* scale, const float* shift, int64_t rows, int64_t channels, float eps
);
int mdl_layer_norm_bf16(
		uint16_t* dst, const uint16_t* src, const uint16_t* scale, const uint16_t* shift, int64_t rows,
		int64_t channels, float eps
);
int mdl_layer_norm_f16(
		uint16_t* dst, const uint16_t* src, const uint16_t* scale, const uint16_t* shift, int64_t rows,
		int64_t channels, float eps
);

#ifdef __cplusplus
}
#endif

#endif

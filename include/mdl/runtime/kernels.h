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
 * @brief Strided batched matrix multiplication `dst[i, j] = alpha * (A[i, j] @ B[i, j]) + bias` over two batch
 * dimensions i < b0, j < b1.
 * @details Every operand is a strided view (strides in elements): A[i, j][m, k] is at
 * `A + i * a_s0 + j * a_s1 + m * a_sM + k * a_sK`, B[i, j][k, n] at `B + i * b_s0 + j * b_s1 + k * b_sK + n * b_sN`
 * (a batch stride of 0 broadcasts B), dst[i, j][m, n] at `dst + i * d_s0 + j * d_s1 + m * d_sM + n * d_sN`. So
 * transposed and permuted operands (e.g. the heads of attention) are read and written in place.
 * The epilogue computes `act(alpha * A @ B + bias) + addend`.
 * @param bias  [N]; only read if @p has_bias is non-zero
 * @param addend a tensor with the result's layout (strides d_*), e.g. a residual; only read if @p has_addend is non-zero
 * @param activation 0: none, 1: GELU (erf)
 */
#define MDL_MATMUL_DECLARE(suffix, type)                                                                               \
	int mdl_matmul_##suffix(                                                                                           \
			type* dst, const type* A, const type* B, const type* bias, int32_t has_bias, int64_t b0, int64_t b1,      \
			int64_t M, int64_t K, int64_t N, int64_t a_s0, int64_t a_s1, int64_t a_sM, int64_t a_sK, int64_t b_s0,    \
			int64_t b_s1, int64_t b_sK, int64_t b_sN, int64_t d_s0, int64_t d_s1, int64_t d_sM, int64_t d_sN,          \
			const type* addend, int32_t has_addend, int32_t activation, float alpha                                    \
	);
MDL_MATMUL_DECLARE(f32, float)
MDL_MATMUL_DECLARE(bf16, uint16_t)
MDL_MATMUL_DECLARE(f16, uint16_t)
#undef MDL_MATMUL_DECLARE

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

/** @brief Column sums `dst[n] = sum_m src[m, n]` of a [M, N] tensor (e.g. the gradient of a bias). */
int mdl_sum_rows_f32(float* dst, const float* src, int64_t M, int64_t N);
int mdl_sum_rows_bf16(uint16_t* dst, const uint16_t* src, int64_t M, int64_t N);
int mdl_sum_rows_f16(uint16_t* dst, const uint16_t* src, int64_t M, int64_t N);

/** @brief The gradient of exact GELU: `dx = dy * GELU'(x)` over @p n elements. */
int mdl_gelu_backward_f32(float* dx, const float* x, const float* dy, int64_t n);
int mdl_gelu_backward_bf16(uint16_t* dx, const uint16_t* x, const uint16_t* dy, int64_t n);
int mdl_gelu_backward_f16(uint16_t* dx, const uint16_t* x, const uint16_t* dy, int64_t n);

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

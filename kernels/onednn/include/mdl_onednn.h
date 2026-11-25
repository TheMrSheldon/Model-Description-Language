#ifndef MDL_ONEDNN_H
#define MDL_ONEDNN_H

/**
 * @file
 * @brief C ABI of the oneDNN kernels (library `mdl_onednn`) of the kernel package `onednn` (see package.yml.in).
 * @details Compiled models call these symbols where the package's implementations are selected (see docs/dev/spec/kernel-packages.md). All tensors are dense and row-major. Every kernel exists for float32
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
 * @brief Matrix multiplication `C = alpha * A @ B + bias + addend` of strided views (rank, shape, strides in
 * elements), batched over leading dimensions that broadcast like numpy's.
 * @details A [..., M, K], B [..., K, N], and C [..., M, N] are given in the product's logical layout, so permuted
 * operands and results (e.g. the heads of attention, transposed weights) are read and written in place; C's batch
 * dimensions are the broadcast ones of A and B. Batch dimensions are merged where the layout allows it.
 * @param bias [N] (contiguous), or NULL
 * @param addend a tensor at C's positions (C's strides), e.g. a residual, or NULL
 */
#define MDL_MATMUL_DECLARE(suffix, type)                                                                               \
	int mdl_matmul_##suffix(                                                                                           \
			type* out, const type* A, const type* B, const type* bias, const type* addend, int64_t a_rank,             \
			const int64_t* a_shape, const int64_t* a_strides, int64_t b_rank, const int64_t* b_shape,                  \
			const int64_t* b_strides, int64_t c_rank, const int64_t* c_shape, const int64_t* c_strides,                \
			float alpha                                                                                                \
	);
MDL_MATMUL_DECLARE(f32, float)
MDL_MATMUL_DECLARE(bf16, uint16_t)
MDL_MATMUL_DECLARE(f16, uint16_t)
#undef MDL_MATMUL_DECLARE

/**
 * @brief Matrix multiplication with quantized weights (a linear layer): `C = alpha * A @ W^T + bias + addend`
 * with W [N, K] the dequantization of @p q: `W[n, k] = scales[k / group, n] * (q[n, k] - zeros[k / group, n])`.
 * @details A [..., M, K] and C [..., M, N] are strided views as for mdl_matmul.
 * @param q [N, K] int8 values (format 1), or [N, K/2] bytes of two 4-bit values (low nibble: even k), signed (format
 * 2) or unsigned with zero points (format 3); see mdl::builtins::QuantFormat
 * @param scales [K / group, N] in the element type of the kernel
 * @param zeros [K / group, N] int8, or NULL
 */
#define MDL_MATMUL_Q_DECLARE(suffix, type)                                                                             \
	int mdl_matmul_q_##suffix(                                                                                         \
			type* out, const type* A, const int8_t* q, const type* scales, const int8_t* zeros, int32_t format,        \
			int64_t group, int64_t N, const type* bias, const type* addend, int64_t a_rank, const int64_t* a_shape,     \
			const int64_t* a_strides, int64_t c_rank, const int64_t* c_shape, const int64_t* c_strides,                \
			float alpha                                                                                                \
	);
MDL_MATMUL_Q_DECLARE(f32, float)
MDL_MATMUL_Q_DECLARE(bf16, uint16_t)
MDL_MATMUL_Q_DECLARE(f16, uint16_t)
#undef MDL_MATMUL_Q_DECLARE

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

/**
 * @brief The gradient of a layer normalization over the last axis with respect to its input: dx = rsqrt(var) * (g -
 * mean(g) - xhat * mean(g * xhat)) with g = dy * gamma, for [rows, channels] tensors dy and xhat (the normalized
 * input) and var [rows] (the variance plus epsilon).
 */
int mdl_layer_norm_backward_f32(
		float* dx, const float* dy, const float* gamma, const float* xhat, const float* var, int64_t rows,
		int64_t channels
);
int mdl_layer_norm_backward_bf16(
		uint16_t* dx, const uint16_t* dy, const uint16_t* gamma, const uint16_t* xhat, const uint16_t* var,
		int64_t rows, int64_t channels
);
int mdl_layer_norm_backward_f16(
		uint16_t* dx, const uint16_t* dy, const uint16_t* gamma, const uint16_t* xhat, const uint16_t* var,
		int64_t rows, int64_t channels
);
#ifdef __cplusplus
}
#endif

#endif

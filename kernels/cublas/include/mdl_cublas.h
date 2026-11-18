#ifndef MDL_CUBLAS_H
#define MDL_CUBLAS_H

/**
 * @file
 * @brief C ABI of the cuBLAS kernels (library `mdl_cublas`) of the kernel package `cublas` (see package.yml.in).
 * @details The same ABI as the oneDNN package's matrix products (kernels/onednn/include/mdl_onednn.h), for tensors in
 * CUDA device memory: compiled models call these symbols where the package's implementations are selected (see
 * docs/kernels.md). float32 (`_f32`, `float*`), bfloat16 (`_bf16`) and float16 (`_f16`; both `uint16_t*` bit
 * patterns); the scalar arguments are float32 in all variants. The kernels run on TVM's current stream of the current
 * device (TVMFFIEnvGetStream), ordered with the model's other kernels. Every function returns 0 on success and a
 * non-zero value on failure.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Strided batched matrix multiplication `dst[i, j] = act(alpha * (A[i, j] @ B[i, j]) + bias) + addend` over two
 * batch dimensions i < b0, j < b1 (see mdl_matmul_f32 of mdl_onednn.h for the layout of the strided views).
 * @details Each matrix must have a unit stride in one of its two dimensions (row- or column-major; cuBLAS cannot read
 * other layouts): otherwise, the function fails.
 * @param activation 0: none, 1: GELU (erf)
 */
#define MDL_CUBLAS_MATMUL_DECLARE(suffix, type)                                                                        \
	int mdl_cublas_matmul_##suffix(                                                                                    \
			type* dst, const type* A, const type* B, const type* bias, int32_t has_bias, int64_t b0, int64_t b1,       \
			int64_t M, int64_t K, int64_t N, int64_t a_s0, int64_t a_s1, int64_t a_sM, int64_t a_sK, int64_t b_s0,     \
			int64_t b_s1, int64_t b_sK, int64_t b_sN, int64_t d_s0, int64_t d_s1, int64_t d_sM, int64_t d_sN,          \
			const type* addend, int32_t has_addend, int32_t activation, float alpha                                    \
	);
MDL_CUBLAS_MATMUL_DECLARE(f32, float)
MDL_CUBLAS_MATMUL_DECLARE(bf16, uint16_t)
MDL_CUBLAS_MATMUL_DECLARE(f16, uint16_t)
#undef MDL_CUBLAS_MATMUL_DECLARE

#ifdef __cplusplus
}
#endif

#endif

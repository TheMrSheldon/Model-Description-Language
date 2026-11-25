#ifndef MDL_CUBLAS_H
#define MDL_CUBLAS_H

/**
 * @file
 * @brief C ABI of the cuBLAS kernels (library `mdl_cublas`) of the kernel package `cublas` (see package.yml.in).
 * @details The same ABI as the oneDNN package's matrix products (kernels/onednn/include/mdl_onednn.h), for tensors in
 * CUDA device memory: compiled models call these symbols where the package's implementations are selected (see
 * docs/dev/spec/kernel-packages.md). float32 (`_f32`, `float*`), bfloat16 (`_bf16`) and float16 (`_f16`; both `uint16_t*` bit
 * patterns); the scalar arguments are float32 in all variants. The kernels run on TVM's current stream of the current
 * device (TVMFFIEnvGetStream), ordered with the model's other kernels. Every function returns 0 on success and a
 * non-zero value on failure.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Matrix multiplication `C = alpha * A @ B + bias + addend` of strided views (see mdl_matmul_f32 of
 * mdl_onednn.h: the same arguments).
 * @details Each matrix must have a unit stride in one of its two dimensions (row- or column-major; cuBLAS cannot read
 * other layouts): otherwise, the function fails.
 */
#define MDL_CUBLAS_MATMUL_DECLARE(suffix, type)                                                                        \
	int mdl_cublas_matmul_##suffix(                                                                                    \
			type* out, const type* A, const type* B, const type* bias, const type* addend, int64_t a_rank,             \
			const int64_t* a_shape, const int64_t* a_strides, int64_t b_rank, const int64_t* b_shape,                  \
			const int64_t* b_strides, int64_t c_rank, const int64_t* c_shape, const int64_t* c_strides,                \
			float alpha                                                                                                \
	);
MDL_CUBLAS_MATMUL_DECLARE(f32, float)
MDL_CUBLAS_MATMUL_DECLARE(bf16, uint16_t)
MDL_CUBLAS_MATMUL_DECLARE(f16, uint16_t)
#undef MDL_CUBLAS_MATMUL_DECLARE

#ifdef __cplusplus
}
#endif

#endif

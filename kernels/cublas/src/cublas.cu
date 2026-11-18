// The cuBLAS kernels of the kernel package cublas (C ABI: include/mdl_cublas.h).
#include "mdl_cublas.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <optional>
#include <utility>

namespace {

	/** @brief TVM's current stream of the current device (the stream of the model's other kernels). */
	cudaStream_t currentStream() {
		int device = 0;
		cudaGetDevice(&device);
		return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
	}

	/** @brief A cuBLAS handle per thread and device. */
	cublasHandle_t handle() {
		constexpr int kMaxDevices = 64;
		thread_local cublasHandle_t handles[kMaxDevices] = {};
		int device = 0;
		cudaGetDevice(&device);
		if (device < 0 || device >= kMaxDevices)
			return nullptr;
		if (handles[device] == nullptr && cublasCreate(&handles[device]) != CUBLAS_STATUS_SUCCESS)
			return nullptr;
		return handles[device];
	}

	template <typename T>
	struct Types;
	template <>
	struct Types<float> {
		using Device = float;
		static constexpr cudaDataType_t data = CUDA_R_32F;
		static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F; // no TF32 (as PyTorch's default)
	};
	template <>
	struct Types<__nv_bfloat16> {
		using Device = __nv_bfloat16;
		static constexpr cudaDataType_t data = CUDA_R_16BF;
		static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
	};
	template <>
	struct Types<__half> {
		using Device = __half;
		static constexpr cudaDataType_t data = CUDA_R_16F;
		static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
	};

	/**
	 * @brief A strided matrix as a cuBLAS (column-major) operand of @p rows x @p cols, element (r, c) at
	 * `r * stride_r + c * stride_c`: either a column-major matrix (op N) or the transpose of one (op T).
	 */
	struct Operand {
		cublasOperation_t op;
		int64_t ld;
	};
	std::optional<Operand> operand(int64_t rows, int64_t cols, int64_t stride_r, int64_t stride_c) {
		// Column-major (op N): unit row stride (the stride of an extent-1 dimension does not matter), ld >= rows.
		if (rows == 1 || stride_r == 1) {
			const int64_t ld = cols == 1 ? rows : stride_c;
			if (ld >= rows)
				return Operand{CUBLAS_OP_N, std::max<int64_t>(ld, 1)};
		}
		// The transpose of a column-major cols x rows matrix (op T): unit column stride, ld >= cols.
		if (cols == 1 || stride_c == 1) {
			const int64_t ld = rows == 1 ? cols : stride_r;
			if (ld >= cols)
				return Operand{CUBLAS_OP_T, std::max<int64_t>(ld, 1)};
		}
		return std::nullopt;
	}

	__device__ inline float toFloat(float v) { return v; }
	__device__ inline float toFloat(__nv_bfloat16 v) { return __bfloat162float(v); }
	__device__ inline float toFloat(__half v) { return __half2float(v); }
	template <typename T>
	__device__ inline T fromFloat(float v);
	template <>
	__device__ inline float fromFloat<float>(float v) { return v; }
	template <>
	__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
	template <>
	__device__ inline __half fromFloat<__half>(float v) { return __float2half(v); }

	/** @brief dst = act(dst + bias) + addend over the strided result [b0, b1, M, N]. */
	template <typename T>
	__global__ void epilogue(
			T* dst, const T* bias, int has_bias, const T* addend, int has_addend, int activation, int64_t b0, int64_t b1,
			int64_t M, int64_t N, int64_t d_s0, int64_t d_s1, int64_t d_sM, int64_t d_sN
	) {
		const int64_t total = b0 * b1 * M * N;
		for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; idx < total;
			 idx += int64_t(gridDim.x) * blockDim.x) {
			const int64_t n = idx % N, m = idx / N % M, j = idx / (N * M) % b1, i = idx / (N * M * b1);
			const int64_t at = i * d_s0 + j * d_s1 + m * d_sM + n * d_sN;
			float v = toFloat(dst[at]);
			if (has_bias)
				v += toFloat(bias[n]);
			if (activation == 1)
				v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
			if (has_addend)
				v += toFloat(addend[at]);
			dst[at] = fromFloat<T>(v);
		}
	}

	template <typename T>
	int matmul(
			T* dst, const T* A, const T* B, const T* bias, int32_t has_bias, int64_t b0, int64_t b1, int64_t M,
			int64_t K, int64_t N, int64_t a_s0, int64_t a_s1, int64_t a_sM, int64_t a_sK, int64_t b_s0, int64_t b_s1,
			int64_t b_sK, int64_t b_sN, int64_t d_s0, int64_t d_s1, int64_t d_sM, int64_t d_sN, const T* addend,
			int32_t has_addend, int32_t activation, float alpha
	) {
		if (b0 <= 0 || b1 <= 0 || M <= 0 || N <= 0 || K < 0)
			return 0 == b0 * b1 * M * N ? 0 : 1;
		cublasHandle_t h = handle();
		if (h == nullptr)
			return 1;
		const cudaStream_t stream = currentStream();
		if (cublasSetStream(h, stream) != CUBLAS_STATUS_SUCCESS)
			return 2;

		// One strided batched call per outer batch index; the larger batch dimension is the strided one.
		if (b1 == 1 && b0 > 1) {
			std::swap(b0, b1);
			std::swap(a_s0, a_s1);
			std::swap(b_s0, b_s1);
			std::swap(d_s0, d_s1);
		}

		// Row-major D[M, N] = A[M, K] @ B[K, N]. cuBLAS is column-major: with a column-major result D (rows M), compute
		// D = A @ B; with a row-major one, compute its transpose D^T (rows N) = B^T @ A^T.
		const bool dColumnMajor = operand(M, N, d_sM, d_sN).has_value() && operand(M, N, d_sM, d_sN)->op == CUBLAS_OP_N;
		int64_t rows, cols, ldc;
		std::optional<Operand> first, second;
		const T *firstPtr, *secondPtr;
		int64_t firstBatch, secondBatch, firstOuter, secondOuter;
		if (dColumnMajor) {
			rows = M, cols = N, ldc = operand(M, N, d_sM, d_sN)->ld;
			first = operand(M, K, a_sM, a_sK), second = operand(K, N, b_sK, b_sN);
			firstPtr = A, secondPtr = B, firstBatch = a_s1, secondBatch = b_s1, firstOuter = a_s0, secondOuter = b_s0;
		} else {
			const auto d = operand(N, M, d_sN, d_sM);
			if (!d || d->op != CUBLAS_OP_N)
				return 3; // the result has no unit stride
			rows = N, cols = M, ldc = d->ld;
			first = operand(N, K, b_sN, b_sK), second = operand(K, M, a_sK, a_sM);
			firstPtr = B, secondPtr = A, firstBatch = b_s1, secondBatch = a_s1, firstOuter = b_s0, secondOuter = a_s0;
		}
		if (!first || !second)
			return 4; // an operand has no unit stride

		const float one = alpha, zero = 0.0f;
		for (int64_t i = 0; i < b0; ++i) {
			const cublasStatus_t status = cublasGemmStridedBatchedEx(
					h, first->op, second->op, static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(K), &one,
					firstPtr + i * firstOuter, Types<T>::data, static_cast<int>(first->ld), firstBatch,
					secondPtr + i * secondOuter, Types<T>::data, static_cast<int>(second->ld), secondBatch, &zero,
					dst + i * d_s0, Types<T>::data, static_cast<int>(ldc), d_s1, static_cast<int>(b1),
					Types<T>::compute, CUBLAS_GEMM_DEFAULT
			);
			if (status != CUBLAS_STATUS_SUCCESS) {
				std::fprintf(stderr, "mdl_cublas_matmul: cuBLAS error %d\n", static_cast<int>(status));
				return 5;
			}
		}

		if (has_bias || has_addend || activation != 0) {
			const int64_t total = b0 * b1 * M * N;
			const int threads = 256;
			const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
			// (b0/b1 and the strides as given, after the swap: the epilogue only needs a consistent enumeration.)
			epilogue<T><<<blocks, threads, 0, stream>>>(
					dst, bias, has_bias, addend, has_addend, activation, b0, b1, M, N, d_s0, d_s1, d_sM, d_sN
			);
			if (cudaGetLastError() != cudaSuccess)
				return 6;
		}
		return 0;
	}

} // namespace

#define MDL_CUBLAS_MATMUL_DEFINE(suffix, type, device)                                                                 \
	extern "C" int mdl_cublas_matmul_##suffix(                                                                         \
			type* dst, const type* A, const type* B, const type* bias, int32_t has_bias, int64_t b0, int64_t b1,       \
			int64_t M, int64_t K, int64_t N, int64_t a_s0, int64_t a_s1, int64_t a_sM, int64_t a_sK, int64_t b_s0,     \
			int64_t b_s1, int64_t b_sK, int64_t b_sN, int64_t d_s0, int64_t d_s1, int64_t d_sM, int64_t d_sN,          \
			const type* addend, int32_t has_addend, int32_t activation, float alpha                                    \
	) {                                                                                                                \
		return matmul<device>(                                                                                         \
				reinterpret_cast<device*>(dst), reinterpret_cast<const device*>(A),                                    \
				reinterpret_cast<const device*>(B), reinterpret_cast<const device*>(bias), has_bias, b0, b1, M, K, N,  \
				a_s0, a_s1, a_sM, a_sK, b_s0, b_s1, b_sK, b_sN, d_s0, d_s1, d_sM, d_sN,                                \
				reinterpret_cast<const device*>(addend), has_addend, activation, alpha                                 \
		);                                                                                                             \
	}
MDL_CUBLAS_MATMUL_DEFINE(f32, float, float)
MDL_CUBLAS_MATMUL_DEFINE(bf16, uint16_t, __nv_bfloat16)
MDL_CUBLAS_MATMUL_DEFINE(f16, uint16_t, __half)

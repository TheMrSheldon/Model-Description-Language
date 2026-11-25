// The cuBLAS kernels of the kernel package cublas (C ABI: include/mdl_cublas.h).
//
// mdl_cublas_matmul_* computes `dst = act(alpha * A @ B + bias) + addend` with cuBLASLt: the bias is the GEMM's epilogue
// (as in PyTorch's linear layers), the addend (e.g. a residual) is the GEMM's `beta * C` term, and the two batch
// dimensions are one strided batch where their strides allow it (e.g. heads x batch of attention). GELU (exact, erf) is
// a separate elementwise pass: cuBLASLt's GELU epilogue is the tanh approximation. Layouts that cuBLASLt does not
// accept fall back to cublasGemmStridedBatchedEx and a general epilogue kernel.
//
// The algorithm of each configuration is the fastest of cuBLASLt's first candidates, measured once (autotuning;
// MDL_CUBLAS_AUTOTUNE=0: cuBLASLt's first candidate). Two batch dimensions whose strides do not merge (e.g. heads and
// batch of attention, whose matrices are interleaved) are one pointer-array batched GEMM: the pointers are computed on
// the GPU.
#include "mdl_cublas.h"
#include "mdl_strided.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

namespace {

	constexpr int kMaxDevices = 64;
	constexpr size_t kWorkspaceBytes = size_t{32} << 20; ///< cuBLASLt's workspace per device and thread

	int currentDevice() {
		int device = 0;
		cudaGetDevice(&device);
		return device;
	}

	/** @brief TVM's current stream of the current device (the stream of the model's other kernels). */
	cudaStream_t currentStream() { return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, currentDevice())); }

	/** @brief Per thread and device: the cuBLAS and cuBLASLt handles and cuBLASLt's workspace. */
	struct Context {
		cublasHandle_t blas = nullptr;
		cublasLtHandle_t lt = nullptr;
		void* workspace = nullptr;
		void** pointers = nullptr; ///< pointer arrays of batched GEMMs (3 * capacity entries)
		int64_t capacity = 0;
	};
	Context* context() {
		thread_local std::array<Context, kMaxDevices> contexts{};
		const int device = currentDevice();
		if (device < 0 || device >= kMaxDevices)
			return nullptr;
		Context& c = contexts[device];
		if (c.blas == nullptr && cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS)
			return nullptr;
		if (c.lt == nullptr && cublasLtCreate(&c.lt) != CUBLAS_STATUS_SUCCESS)
			return nullptr;
		if (c.workspace == nullptr && cudaMalloc(&c.workspace, kWorkspaceBytes) != cudaSuccess)
			return nullptr;
		return &c;
	}

	template <typename T>
	struct Types;
	template <>
	struct Types<float> {
		static constexpr cudaDataType_t data = CUDA_R_32F;
		static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F; // no TF32 (as PyTorch's default)
	};
	template <>
	struct Types<__nv_bfloat16> {
		static constexpr cudaDataType_t data = CUDA_R_16BF;
		static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
	};
	template <>
	struct Types<__half> {
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

	__device__ inline float gelu(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }

	/** @brief dst = act(dst + bias) + addend over the strided result [b0, b1, M, N] (any layout). */
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
				v = gelu(v);
			if (has_addend)
				v += toFloat(addend[at]);
			dst[at] = fromFloat<T>(v);
		}
	}

	/** @brief dst = gelu(dst) + addend for a contiguous result of @p total elements (the addend in the same layout). */
	template <typename T>
	__global__ void geluContiguous(T* dst, const T* addend, int has_addend, int64_t total) {
		for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
			float v = gelu(toFloat(dst[i]));
			if (has_addend)
				v += toFloat(addend[i]);
			dst[i] = fromFloat<T>(v);
		}
	}

	int blocksFor(int64_t total, int threads) {
		return static_cast<int>(std::clamp<int64_t>((total + threads - 1) / threads, 1, 65535));
	}

	/** @brief The pointers of the matrices of a 2-D batch: base + i * s0 + j * s1 for i < b0, j < b1. */
	template <typename T>
	__global__ void batchPointers(
			const void** a, const void** b, void** d, const T* aBase, const T* bBase, T* dBase, int64_t b0, int64_t b1,
			int64_t a_s0, int64_t a_s1, int64_t b_s0, int64_t b_s1, int64_t d_s0, int64_t d_s1
	) {
		for (int64_t k = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; k < b0 * b1; k += int64_t(gridDim.x) * blockDim.x) {
			const int64_t i = k / b1, j = k % b1;
			a[k] = aBase + i * a_s0 + j * a_s1;
			b[k] = bBase + i * b_s0 + j * b_s1;
			d[k] = dBase + i * d_s0 + j * d_s1;
		}
	}

	bool autotune() {
		static const bool enabled = [] {
			const char* env = std::getenv("MDL_CUBLAS_AUTOTUNE");
			return env == nullptr || std::string(env) != "0";
		}();
		return enabled;
	}

	/** @brief The problem in cuBLAS's (column-major) terms: D (rows x cols) = op(first) @ op(second). */
	template <typename T>
	struct Problem {
		int64_t rows = 0, cols = 0, K = 0, ldd = 0;
		Operand first{}, second{};
		const T *firstPtr = nullptr, *secondPtr = nullptr;
		int64_t firstBatch = 0, secondBatch = 0, firstOuter = 0, secondOuter = 0;
		bool dColumnMajor = false; ///< D is the result itself (rows = M), not its transpose (rows = N)
	};

	using AlgoKey = std::tuple<
			int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
			int64_t, bool, bool>;

	struct Layout {
		cublasLtMatrixLayout_t layout = nullptr;
		Layout(cudaDataType_t type, int64_t rows, int64_t cols, int64_t ld, int64_t batch, int64_t batchStride) {
			cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld);
			if (batch > 1) {
				const int32_t count = static_cast<int32_t>(batch);
				cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count));
				cublasLtMatrixLayoutSetAttribute(
						layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batchStride, sizeof(batchStride)
				);
			}
		}
		Layout(const Layout&) = delete;
		Layout& operator=(const Layout&) = delete;
		~Layout() {
			if (layout != nullptr)
				cublasLtMatrixLayoutDestroy(layout);
		}
	};

	/**
	 * @brief One cuBLASLt call: D = alpha * op(first) @ op(second) [+ bias (per row of D)] [+ C], batched.
	 * @return false if cuBLASLt has no algorithm for it (the caller falls back).
	 */
	template <typename T>
	bool ltMatmul(
			Context& ctx, cudaStream_t stream, const Problem<T>& p, int64_t batch, int64_t firstOffset,
			int64_t secondOffset, T* d, int64_t dBatch, const T* bias, const T* c, float alpha
	) {
		constexpr cudaDataType_t type = Types<T>::data;
		cublasLtMatmulDesc_t desc = nullptr;
		if (cublasLtMatmulDescCreate(&desc, Types<T>::compute, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
			return false;
		const cublasOperation_t opA = p.first.op, opB = p.second.op;
		cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
		cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
		const cublasLtEpilogue_t epilogue = bias != nullptr ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
		cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
		if (bias != nullptr) {
			cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
			const int64_t biasBatchStride = 0; // one bias for all matrices of the batch
			cublasLtMatmulDescSetAttribute(
					desc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &biasBatchStride, sizeof(biasBatchStride)
			);
		}

		// The stored operands: op T reads the transpose of a stored (cols x rows) matrix.
		const T* first = p.firstPtr + firstOffset;
		const T* second = p.secondPtr + secondOffset;
		const int64_t aRows = opA == CUBLAS_OP_N ? p.rows : p.K, aCols = opA == CUBLAS_OP_N ? p.K : p.rows;
		const int64_t bRows = opB == CUBLAS_OP_N ? p.K : p.cols, bCols = opB == CUBLAS_OP_N ? p.cols : p.K;
		const Layout a(type, aRows, aCols, p.first.ld, batch, p.firstBatch);
		const Layout b(type, bRows, bCols, p.second.ld, batch, p.secondBatch);
		const Layout dl(type, p.rows, p.cols, p.ldd, batch, dBatch);

		// The algorithm: cuBLASLt's first heuristic result, cached per configuration (including the pointers'
		// alignment, on which the algorithms depend).
		thread_local std::map<AlgoKey, std::optional<cublasLtMatmulHeuristicResult_t>> algos;
		const auto alignment = static_cast<int64_t>(
				(reinterpret_cast<uintptr_t>(first) | reinterpret_cast<uintptr_t>(second) |
				 reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(c) | reinterpret_cast<uintptr_t>(bias)) &
				255
		);
		const AlgoKey key{
				currentDevice(), static_cast<int>(type), static_cast<int>(opA) * 4 + static_cast<int>(opB), p.rows,
				p.cols, p.K, p.first.ld, p.second.ld, p.ldd, batch, p.firstBatch, p.secondBatch, dBatch, alignment,
				bias != nullptr, c != nullptr
		};
		auto it = algos.find(key);
		if (it == algos.end()) {
			cublasLtMatmulPreference_t preference = nullptr;
			cublasLtMatmulPreferenceCreate(&preference);
			const size_t workspace = kWorkspaceBytes;
			cublasLtMatmulPreferenceSetAttribute(
					preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)
			);
			constexpr int kCandidates = 8;
			cublasLtMatmulHeuristicResult_t results[kCandidates] = {};
			int found = 0;
			const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
					ctx.lt, desc, a.layout, b.layout, dl.layout, dl.layout, preference, autotune() ? kCandidates : 1,
					results, &found
			);
			cublasLtMatmulPreferenceDestroy(preference);
			std::optional<cublasLtMatmulHeuristicResult_t> best;
			if (status == CUBLAS_STATUS_SUCCESS && found > 0)
				best = results[0];
			if (best && found > 1) {
				// Measure the candidates once (into a scratch result: C may be the result itself, e.g. accumulated
				// gradients), keep the fastest.
				const int64_t elements = (batch - 1) * dBatch + p.ldd * p.cols;
				void* scratch = nullptr;
				cudaEvent_t start = nullptr, stop = nullptr;
				if (cudaMalloc(&scratch, elements * sizeof(T)) == cudaSuccess && cudaEventCreate(&start) == cudaSuccess &&
					cudaEventCreate(&stop) == cudaSuccess) {
					float bestMs = 0;
					const float beta = c != nullptr ? 1.0f : 0.0f;
					T* out = static_cast<T*>(scratch);
					for (int r = 0; r < found; ++r) {
						auto run = [&] {
							return cublasLtMatmul(
									ctx.lt, desc, &alpha, first, a.layout, second, b.layout, &beta,
									c != nullptr ? c : out, dl.layout, out, dl.layout, &results[r].algo, ctx.workspace,
									kWorkspaceBytes, stream
							);
						};
						if (run() != CUBLAS_STATUS_SUCCESS) // (also the warm-up)
							continue;
						cudaEventRecord(start, stream);
						constexpr int kRuns = 3;
						for (int k = 0; k < kRuns; ++k)
							run();
						cudaEventRecord(stop, stream);
						cudaEventSynchronize(stop);
						float ms = 0;
						cudaEventElapsedTime(&ms, start, stop);
						if (r == 0 || ms < bestMs) {
							bestMs = ms;
							best = results[r];
						}
					}
				}
				if (start != nullptr)
					cudaEventDestroy(start);
				if (stop != nullptr)
					cudaEventDestroy(stop);
				if (scratch != nullptr)
					cudaFree(scratch);
				cudaGetLastError(); // (a failed candidate does not affect the following calls)
			}
			it = algos.emplace(key, best).first;
		}
		bool ok = false;
		if (it->second) {
			const float beta = c != nullptr ? 1.0f : 0.0f;
			// C has the layout of D (the addend is in the result's layout); with beta = 0 it is not read.
			ok = cublasLtMatmul(
						 ctx.lt, desc, &alpha, first, a.layout, second, b.layout, &beta, c != nullptr ? c : d,
						 dl.layout, d, dl.layout, &it->second->algo, ctx.workspace, kWorkspaceBytes, stream
				 ) == CUBLAS_STATUS_SUCCESS;
		}
		cublasLtMatmulDescDestroy(desc);
		return ok;
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
		Context* ctx = context();
		if (ctx == nullptr)
			return 1;
		const cudaStream_t stream = currentStream();

		// The batch: b1 strided matrices per call, b0 calls. The two batch dimensions merge into one call where b0's
		// strides are b1 times b1's (e.g. heads x batch of attention).
		if (b1 == 1) {
			std::swap(b0, b1);
			std::swap(a_s0, a_s1);
			std::swap(b_s0, b_s1);
			std::swap(d_s0, d_s1);
		}
		if (b0 > 1 && a_s0 == a_s1 * b1 && b_s0 == b_s1 * b1 && d_s0 == d_s1 * b1) {
			b1 *= b0;
			b0 = 1;
		}

		// Row-major D[M, N] = A[M, K] @ B[K, N]. cuBLAS is column-major: with a column-major result D (rows M), compute
		// D = A @ B; with a row-major one, compute its transpose D^T (rows N) = B^T @ A^T.
		Problem<T> p;
		p.K = K;
		const auto dc = operand(M, N, d_sM, d_sN);
		p.dColumnMajor = dc && dc->op == CUBLAS_OP_N;
		std::optional<Operand> first, second;
		if (p.dColumnMajor) {
			p.rows = M, p.cols = N, p.ldd = dc->ld;
			first = operand(M, K, a_sM, a_sK), second = operand(K, N, b_sK, b_sN);
			p.firstPtr = A, p.secondPtr = B, p.firstBatch = a_s1, p.secondBatch = b_s1;
			p.firstOuter = a_s0, p.secondOuter = b_s0;
		} else {
			const auto d = operand(N, M, d_sN, d_sM);
			if (!d || d->op != CUBLAS_OP_N)
				return 3; // the result has no unit stride
			p.rows = N, p.cols = M, p.ldd = d->ld;
			first = operand(N, K, b_sN, b_sK), second = operand(K, M, a_sK, a_sM);
			p.firstPtr = B, p.secondPtr = A, p.firstBatch = b_s1, p.secondBatch = a_s1;
			p.firstOuter = b_s0, p.secondOuter = a_s0;
		}
		if (!first || !second)
			return 4; // an operand has no unit stride
		p.first = *first, p.second = *second;

		// Fused in cuBLASLt: the bias (cuBLASLt's bias is per row of D: fused when D is the transposed result, rows = N)
		// and the addend (beta * C, unless GELU comes between the bias and the addend).
		const bool fuseBias = has_bias && !p.dColumnMajor;
		const bool fuseAddend = has_addend && activation == 0;

		// Two batch dimensions that do not merge, without epilogue: one pointer-array batched GEMM.
		if (b0 > 1 && !has_bias && !has_addend && activation == 0) {
			const int64_t count = b0 * b1;
			if (count > ctx->capacity) {
				if (ctx->pointers != nullptr)
					cudaFree(ctx->pointers);
				ctx->pointers = nullptr;
				ctx->capacity = 0;
				if (cudaMalloc(reinterpret_cast<void**>(&ctx->pointers), 3 * count * sizeof(void*)) == cudaSuccess)
					ctx->capacity = count;
			}
			if (ctx->pointers != nullptr && cublasSetStream(ctx->blas, stream) == CUBLAS_STATUS_SUCCESS) {
				const void** ap = const_cast<const void**>(ctx->pointers);
				const void** bp = const_cast<const void**>(ctx->pointers + count);
				void** dp = ctx->pointers + 2 * count;
				constexpr int threads = 256;
				batchPointers<T><<<blocksFor(count, threads), threads, 0, stream>>>(
						ap, bp, dp, p.firstPtr, p.secondPtr, dst, b0, b1, p.firstOuter, p.firstBatch, p.secondOuter,
						p.secondBatch, d_s0, d_s1
				);
				const float zero = 0.0f;
				const cublasStatus_t status = cublasGemmBatchedEx(
						ctx->blas, p.first.op, p.second.op, static_cast<int>(p.rows), static_cast<int>(p.cols),
						static_cast<int>(K), &alpha, ap, Types<T>::data, static_cast<int>(p.first.ld), bp, Types<T>::data,
						static_cast<int>(p.second.ld), &zero, dp, Types<T>::data, static_cast<int>(p.ldd),
						static_cast<int>(count), Types<T>::compute, CUBLAS_GEMM_DEFAULT
				);
				if (status == CUBLAS_STATUS_SUCCESS)
					return cudaGetLastError() == cudaSuccess ? 0 : 6;
			}
		}

		bool done = true;
		for (int64_t i = 0; i < b0 && done; ++i)
			done = ltMatmul(
					*ctx, stream, p, b1, i * p.firstOuter, i * p.secondOuter, dst + i * d_s0, d_s1,
					fuseBias ? bias : nullptr, fuseAddend ? addend + i * d_s0 : nullptr, alpha
			);
		bool biasDone = fuseBias, addendDone = fuseAddend;
		if (!done) {
			// Fallback: cublasGemmStridedBatchedEx, and everything else in the epilogue kernel.
			if (cublasSetStream(ctx->blas, stream) != CUBLAS_STATUS_SUCCESS)
				return 2;
			const float zero = 0.0f;
			for (int64_t i = 0; i < b0; ++i) {
				const cublasStatus_t status = cublasGemmStridedBatchedEx(
						ctx->blas, p.first.op, p.second.op, static_cast<int>(p.rows), static_cast<int>(p.cols),
						static_cast<int>(K), &alpha, p.firstPtr + i * p.firstOuter, Types<T>::data,
						static_cast<int>(p.first.ld), p.firstBatch, p.secondPtr + i * p.secondOuter, Types<T>::data,
						static_cast<int>(p.second.ld), p.secondBatch, &zero, dst + i * d_s0, Types<T>::data,
						static_cast<int>(p.ldd), d_s1, static_cast<int>(b1), Types<T>::compute, CUBLAS_GEMM_DEFAULT
				);
				if (status != CUBLAS_STATUS_SUCCESS) {
					std::fprintf(stderr, "mdl_cublas_matmul: cuBLAS error %d\n", static_cast<int>(status));
					return 5;
				}
			}
			biasDone = addendDone = false;
		}

		const bool needBias = has_bias && !biasDone, needAddend = has_addend && !addendDone;
		if (!needBias && !needAddend && activation == 0)
			return 0;
		const int64_t total = b0 * b1 * M * N;
		constexpr int threads = 256;
		const bool contiguous = d_sN == 1 && d_sM == N && (b1 == 1 || d_s1 == M * N) && (b0 == 1 || d_s0 == b1 * M * N);
		if (contiguous && !needBias && activation == 1)
			geluContiguous<T><<<blocksFor(total, threads), threads, 0, stream>>>(dst, addend, needAddend, total);
		else
			epilogue<T><<<blocksFor(total, threads), threads, 0, stream>>>(
					dst, bias, needBias, addend, needAddend, activation, b0, b1, M, N, d_s0, d_s1, d_sM, d_sN
			);
		return cudaGetLastError() == cudaSuccess ? 0 : 6;
	}

} // namespace

#define MDL_CUBLAS_MATMUL_DEFINE(suffix, type, device)                                                                 \
	extern "C" int mdl_cublas_matmul_##suffix(                                                                         \
			type* out, const type* A, const type* B, const type* bias, const type* addend, int64_t a_rank,             \
			const int64_t* a_shape, const int64_t* a_strides, int64_t b_rank, const int64_t* b_shape,                  \
			const int64_t* b_strides, int64_t c_rank, const int64_t* c_shape, const int64_t* c_strides,                \
			int32_t activation, float alpha                                                                            \
	) {                                                                                                                \
		using T = device;                                                                                              \
		const auto offset = [](const type* p, int64_t o) {                                                             \
			return p == nullptr ? nullptr : reinterpret_cast<const T*>(p) + o;                                         \
		};                                                                                                             \
		return mdl::kernels::forEachGemm(                                                                              \
				{a_rank, a_shape, a_strides}, {b_rank, b_shape, b_strides}, {c_rank, c_shape, c_strides}, 2,           \
				[&](const mdl::kernels::Gemm& g, int64_t ao, int64_t bo, int64_t co) {                                 \
					return matmul<T>(                                                                                  \
							reinterpret_cast<T*>(out) + co, offset(A, ao), offset(B, bo), offset(bias, 0),             \
							bias != nullptr, g.b0, g.b1, g.M, g.K, g.N, g.a_s0, g.a_s1, g.a_sM, g.a_sK, g.b_s0,        \
							g.b_s1, g.b_sK, g.b_sN, g.c_s0, g.c_s1, g.c_sM, g.c_sN, offset(addend, co),               \
							addend != nullptr, activation, alpha                                                       \
					);                                                                                                 \
				}                                                                                                      \
		);                                                                                                             \
	}
MDL_CUBLAS_MATMUL_DEFINE(f32, float, float)
MDL_CUBLAS_MATMUL_DEFINE(bf16, uint16_t, __nv_bfloat16)
MDL_CUBLAS_MATMUL_DEFINE(f16, uint16_t, __half)

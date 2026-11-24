// The C ABI of the kernel package flash (include/mdl_flash.h) over FlashAttention-2's kernels (their parameters and
// launch functions, src/flash.h of github.com/Dao-AILab/flash-attention; its flash_api.cpp, which needs PyTorch, is
// replaced by this file).
#include "mdl_flash.h"

#include "flash.h"

#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace {
	using FLASH_NAMESPACE::Flash_bwd_params;
	using FLASH_NAMESPACE::Flash_fwd_params;

	constexpr int kMaxDevices = 16;

	/** @brief Per thread and device: the backward pass's float32 workspace (grown on demand, never shrunk). */
	struct Workspace {
		void* data = nullptr;
		size_t bytes = 0;
	};

	int currentDevice() {
		int device = 0;
		cudaGetDevice(&device);
		return device;
	}

	cudaStream_t currentStream() { return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, currentDevice())); }

	/** @brief @p bytes of the current device's workspace, or nullptr. */
	void* workspace(size_t bytes) {
		thread_local std::array<Workspace, kMaxDevices> workspaces{};
		const int device = currentDevice();
		if (device < 0 || device >= kMaxDevices)
			return nullptr;
		Workspace& w = workspaces[device];
		if (bytes > w.bytes) {
			if (w.data != nullptr)
				cudaFree(w.data);
			w = {};
			if (cudaMalloc(&w.data, bytes) != cudaSuccess)
				return nullptr;
			w.bytes = bytes;
		}
		return w.data;
	}

	int64_t roundUp(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

	/**
	 * @brief The view of a tensor [L0, L1, S, D] (rank 4) or [L1, S, D] (rank 3) as FlashAttention's [b, s, h, d]: b =
	 * L0, h = L1, so that its row statistics [b, h, s] are the ABI's [..., S].
	 */
	struct View {
		int64_t b, h, s, d;
		int64_t batchStride, headStride, rowStride;
	};

	bool viewOf(int64_t rank, const int64_t* shape, const int64_t* strides, View& v) {
		if (rank == 4) {
			v = {shape[0], shape[1], shape[2], shape[3], strides[0], strides[1], strides[2]};
		} else if (rank == 3) {
			v = {1, shape[0], shape[1], shape[2], 0, strides[0], strides[1]};
		} else {
			return false;
		}
		return strides[rank - 1] == 1;
	}

	/** @brief Strides (rank 3 or 4) as FlashAttention's (batch, head, row). */
	void stridesOf(int64_t rank, const int64_t* strides, int64_t& batch, int64_t& head, int64_t& row) {
		batch = rank == 4 ? strides[0] : 0;
		head = rank == 4 ? strides[1] : strides[0];
		row = rank == 4 ? strides[2] : strides[1];
	}

	/** @brief The parameters shared by both passes (FlashAttention's set_params_fprop, without dropout and masks). */
	bool forwardParams(
			Flash_fwd_params& params, bool bf16, const void* Q, const void* K, const void* V, void* out, void* lse,
			int64_t rank, const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape,
			const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides,
			float scale
	) {
		View q, k, v;
		if (!viewOf(rank, q_shape, q_strides, q) || !viewOf(rank, k_shape, k_strides, k) ||
			!viewOf(rank, v_shape, v_strides, v) || o_strides[rank - 1] != 1)
			return false;
		// (the same batch and heads; K and V of the same length; one head dimension of a compiled kernel)
		if (k.b != q.b || k.h != q.h || v.b != q.b || v.h != q.h || v.s != k.s || k.d != q.d || v.d != q.d ||
			(q.d != 32 && q.d != 64 && q.d != 128))
			return false;
		params = {};
		params.is_bf16 = bf16;
		params.q_ptr = const_cast<void*>(Q);
		params.k_ptr = const_cast<void*>(K);
		params.v_ptr = const_cast<void*>(V);
		params.q_batch_stride = q.batchStride;
		params.k_batch_stride = k.batchStride;
		params.v_batch_stride = v.batchStride;
		params.q_head_stride = q.headStride;
		params.k_head_stride = k.headStride;
		params.v_head_stride = v.headStride;
		params.q_row_stride = q.rowStride;
		params.k_row_stride = k.rowStride;
		params.v_row_stride = v.rowStride;
		params.o_ptr = out;
		stridesOf(rank, o_strides, params.o_batch_stride, params.o_head_stride, params.o_row_stride);
		params.softmax_lse_ptr = lse;
		params.b = static_cast<int>(q.b);
		params.h = params.h_k = static_cast<int>(q.h);
		params.h_h_k_ratio = 1;
		params.seqlen_q = static_cast<int>(q.s);
		params.seqlen_k = static_cast<int>(k.s);
		params.seqlen_q_rounded = static_cast<int>(roundUp(q.s, 128));
		params.seqlen_k_rounded = static_cast<int>(roundUp(k.s, 128));
		params.d = params.d_rounded = static_cast<int>(q.d);
		params.scale_softmax = scale;
		params.scale_softmax_log2 = scale * static_cast<float>(M_LOG2E);
		params.p_dropout = 1.f; // (the probability to keep)
		params.p_dropout_in_uint8_t = 255;
		params.rp_dropout = 1.f;
		params.scale_softmax_rp_dropout = scale;
		params.window_size_left = params.window_size_right = -1;
		params.is_seqlens_k_cumulative = true;
		params.num_splits = 1;
		return true;
	}

	template <typename T>
	void runForward(Flash_fwd_params& params, cudaStream_t stream) {
		switch (params.d) {
		case 32:
			FLASH_NAMESPACE::run_mha_fwd_<T, 32, false>(params, stream);
			break;
		case 64:
			FLASH_NAMESPACE::run_mha_fwd_<T, 64, false>(params, stream);
			break;
		default:
			FLASH_NAMESPACE::run_mha_fwd_<T, 128, false>(params, stream);
		}
	}

	template <typename T>
	void runBackward(Flash_bwd_params& params, cudaStream_t stream) {
		switch (params.d) {
		case 32:
			FLASH_NAMESPACE::run_mha_bwd_<T, 32, false>(params, stream);
			break;
		case 64:
			FLASH_NAMESPACE::run_mha_bwd_<T, 64, false>(params, stream);
			break;
		default:
			FLASH_NAMESPACE::run_mha_bwd_<T, 128, false>(params, stream);
		}
	}

	template <typename T>
	int forward(
			bool bf16, void* out, float* lse, const void* Q, const void* K, const void* V, int64_t rank,
			const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides,
			const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides, const void* bias,
			const int32_t* lengths, int32_t with_lse, float scale
	) {
		if (bias != nullptr || lengths != nullptr)
			return 1; // (masks: the cuDNN package)
		Flash_fwd_params params;
		if (!forwardParams(
					params, bf16, Q, K, V, out, nullptr, rank, q_shape, q_strides, k_shape, k_strides, v_shape,
					v_strides, o_strides, scale
			))
			return 2;
		// (FlashAttention always writes the statistics: into lse, or a workspace)
		params.softmax_lse_ptr = with_lse != 0
										 ? static_cast<void*>(lse)
										 : workspace(sizeof(float) * static_cast<size_t>(params.b * params.h) *
													 static_cast<size_t>(params.seqlen_q));
		if (params.softmax_lse_ptr == nullptr)
			return 3;
		try {
			runForward<T>(params, currentStream());
		} catch (const mdl_flash::CudaError&) {
			return 4;
		}
		return 0;
	}

	template <typename T>
	int backward(
			bool bf16, void* dq, void* dk, void* dv, const void* Q, const void* K, const void* V, const void* O,
			const void* dO, const float* lse, int64_t rank, const int64_t* q_shape, const int64_t* q_strides,
			const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides,
			const int64_t* o_strides, const int64_t* do_strides, const int64_t* dq_strides,
			const int64_t* dk_strides, const int64_t* dv_strides, const void* bias, const int32_t* lengths,
			float scale
	) {
		if (bias != nullptr || lengths != nullptr)
			return 1;
		Flash_bwd_params params;
		if (!forwardParams(
					params, bf16, Q, K, V, const_cast<void*>(O), const_cast<float*>(lse), rank, q_shape, q_strides,
					k_shape, k_strides, v_shape, v_strides, o_strides, scale
			) ||
			do_strides[rank - 1] != 1 || dq_strides[rank - 1] != 1 || dk_strides[rank - 1] != 1 ||
			dv_strides[rank - 1] != 1)
			return 2;
		params.do_ptr = const_cast<void*>(dO);
		params.dq_ptr = dq;
		params.dk_ptr = dk;
		params.dv_ptr = dv;
		stridesOf(rank, do_strides, params.do_batch_stride, params.do_head_stride, params.do_row_stride);
		stridesOf(rank, dq_strides, params.dq_batch_stride, params.dq_head_stride, params.dq_row_stride);
		stridesOf(rank, dk_strides, params.dk_batch_stride, params.dk_head_stride, params.dk_row_stride);
		stridesOf(rank, dv_strides, params.dv_batch_stride, params.dv_head_stride, params.dv_row_stride);
		// The float32 workspace: dQ accumulated [b, s_q rounded, h, d], then the row sums of dO * O [b, h, s_q rounded].
		const size_t accum = static_cast<size_t>(params.b) * static_cast<size_t>(params.seqlen_q_rounded) *
							 static_cast<size_t>(params.h) * static_cast<size_t>(params.d_rounded);
		const size_t sums =
				static_cast<size_t>(params.b) * static_cast<size_t>(params.h) * static_cast<size_t>(params.seqlen_q_rounded);
		auto* w = static_cast<float*>(workspace(sizeof(float) * (accum + sums)));
		if (w == nullptr)
			return 3;
		params.dq_accum_ptr = w;
		params.dsoftmax_sum = w + accum;
		params.deterministic = false;
		params.dq_accum_split_stride = 0;
		try {
			runBackward<T>(params, currentStream());
		} catch (const mdl_flash::CudaError&) {
			return 4;
		}
		return 0;
	}
} // namespace

#define MDL_FLASH_DEFINE(suffix, T, bf16)                                                                              \
	extern "C" int mdl_flash_attention_fwd_##suffix(                                                                   \
			uint16_t* out, float* lse, const uint16_t* Q, const uint16_t* K, const uint16_t* V, int64_t rank,          \
			const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides,        \
			const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides, const uint16_t* bias,          \
			const int64_t* /*bias_shape*/, const int64_t* /*bias_strides*/, const int32_t* lengths, int32_t with_lse,  \
			float scale                                                                                                \
	) {                                                                                                                \
		return forward<T>(                                                                                             \
				bf16, out, lse, Q, K, V, rank, q_shape, q_strides, k_shape, k_strides, v_shape, v_strides, o_strides,  \
				bias, lengths, with_lse, scale                                                                         \
		);                                                                                                             \
	}                                                                                                                  \
	extern "C" int mdl_flash_attention_bwd_##suffix(                                                                   \
			uint16_t* dq, uint16_t* dk, uint16_t* dv, const uint16_t* Q, const uint16_t* K, const uint16_t* V,         \
			const uint16_t* O, const uint16_t* dO, const float* lse, int64_t rank, const int64_t* q_shape,             \
			const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape,        \
			const int64_t* v_strides, const int64_t* o_strides, const int64_t* do_strides, const int64_t* dq_strides,  \
			const int64_t* dk_strides, const int64_t* dv_strides, const uint16_t* bias, const int64_t* /*bias_shape*/, \
			const int64_t* /*bias_strides*/, const int32_t* lengths, float scale                                       \
	) {                                                                                                                \
		return backward<T>(                                                                                            \
				bf16, dq, dk, dv, Q, K, V, O, dO, lse, rank, q_shape, q_strides, k_shape, k_strides, v_shape,          \
				v_strides, o_strides, do_strides, dq_strides, dk_strides, dv_strides, bias, lengths, scale             \
		);                                                                                                             \
	}
MDL_FLASH_DEFINE(bf16, cutlass::bfloat16_t, true)
MDL_FLASH_DEFINE(f16, cutlass::half_t, false)

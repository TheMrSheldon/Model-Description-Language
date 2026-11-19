// The cuDNN kernels of the kernel package cudnn (C ABI: include/mdl_cudnn.h): fused attention with cuDNN's scaled dot
// product attention, through the cuDNN frontend (an operation graph per configuration, built once and cached).
#include "mdl_cudnn.h"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>

namespace {

	namespace fe = cudnn_frontend;

	constexpr int kMaxDevices = 64;
	constexpr int64_t kQ = 1, kK = 2, kV = 3, kO = 4; ///< the tensors' ids in the graph

	int currentDevice() {
		int device = 0;
		cudaGetDevice(&device);
		return device;
	}

	/** @brief Per thread and device: the cuDNN handle and a workspace (grown on demand). */
	struct Context {
		cudnnHandle_t handle = nullptr;
		void* workspace = nullptr;
		int64_t workspaceBytes = 0;
	};
	Context* context() {
		thread_local std::array<Context, kMaxDevices> contexts{};
		const int device = currentDevice();
		if (device < 0 || device >= kMaxDevices)
			return nullptr;
		Context& c = contexts[device];
		if (c.handle == nullptr && cudnnCreate(&c.handle) != CUDNN_STATUS_SUCCESS)
			return nullptr;
		return &c;
	}

	using Key = std::tuple<int, int, std::array<int64_t, 6>, std::array<int64_t, 16>, float>;

	/** @brief The SDPA graph of a configuration (nullptr if cuDNN does not support it). */
	std::shared_ptr<fe::graph::Graph> graphFor(
			Context& ctx, fe::DataType_t type, const std::array<int64_t, 6>& dims, const std::array<int64_t, 16>& strides,
			float scale
	) {
		thread_local std::map<Key, std::shared_ptr<fe::graph::Graph>> graphs;
		const Key key{currentDevice(), static_cast<int>(type), dims, strides, scale};
		if (auto it = graphs.find(key); it != graphs.end())
			return it->second;
		const auto [b0, b1, Sq, Skv, D, Dv] = dims;
		auto graph = std::make_shared<fe::graph::Graph>();
		graph->set_io_data_type(type)
				.set_intermediate_data_type(fe::DataType_t::FLOAT)
				.set_compute_data_type(fe::DataType_t::FLOAT);
		auto tensor = [&](const char* name, int64_t uid, std::vector<int64_t> dim, size_t s) {
			return graph->tensor(fe::graph::Tensor_attributes()
										 .set_name(name)
										 .set_uid(uid)
										 .set_dim(std::move(dim))
										 .set_stride({strides[s], strides[s + 1], strides[s + 2], strides[s + 3]}));
		};
		auto Q = tensor("Q", kQ, {b0, b1, Sq, D}, 0);
		auto K = tensor("K", kK, {b0, b1, Skv, D}, 4);
		auto V = tensor("V", kV, {b0, b1, Skv, Dv}, 8);
		auto options = fe::graph::SDPA_attributes().set_name("attention").set_attn_scale(scale);
		options.set_is_inference(true); // (no statistics for a backward pass)
		auto [O, stats] = graph->sdpa(Q, K, V, options);
		O->set_output(true)
				.set_uid(kO)
				.set_dim({b0, b1, Sq, Dv})
				.set_stride({strides[12], strides[13], strides[14], strides[15]});
		std::shared_ptr<fe::graph::Graph> result;
		if (graph->validate().is_good() && graph->build_operation_graph(ctx.handle).is_good() &&
			graph->create_execution_plans({fe::HeurMode_t::A}).is_good() && graph->check_support(ctx.handle).is_good() &&
			graph->build_plans(ctx.handle).is_good())
			result = graph;
		else
			std::fprintf(stderr, "mdl_attention: cuDNN does not support this attention configuration\n");
		graphs.emplace(key, result);
		return result;
	}

	int attention(
			fe::DataType_t type, void* out, const void* Q, const void* K, const void* V, int64_t b0, int64_t b1,
			int64_t Sq, int64_t Skv, int64_t D, int64_t Dv, const std::array<int64_t, 16>& strides, float scale
	) {
		if (b0 * b1 * Sq * Dv == 0)
			return 0;
		Context* ctx = context();
		if (ctx == nullptr)
			return 1;
		const auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, currentDevice()));
		if (cudnnSetStream(ctx->handle, stream) != CUDNN_STATUS_SUCCESS)
			return 2;
		auto graph = graphFor(*ctx, type, {b0, b1, Sq, Skv, D, Dv}, strides, scale);
		if (!graph)
			return 3;
		int64_t bytes = 0;
		if (!graph->get_workspace_size(bytes).is_good())
			return 4;
		if (bytes > ctx->workspaceBytes) {
			if (ctx->workspace != nullptr)
				cudaFree(ctx->workspace);
			ctx->workspace = nullptr;
			ctx->workspaceBytes = 0;
			if (cudaMalloc(&ctx->workspace, bytes) != cudaSuccess)
				return 5;
			ctx->workspaceBytes = bytes;
		}
		std::unordered_map<int64_t, void*> pack{
				{kQ, const_cast<void*>(Q)}, {kK, const_cast<void*>(K)}, {kV, const_cast<void*>(V)}, {kO, out}
		};
		if (!graph->execute(ctx->handle, pack, ctx->workspace).is_good())
			return 6;
		return 0;
	}

} // namespace

#define MDL_ATTENTION_DEFINE(suffix, type, cudnnType)                                                                  \
	extern "C" int mdl_attention_##suffix(                                                                             \
			type* out, const type* Q, const type* K, const type* V, int64_t b0, int64_t b1, int64_t Sq, int64_t Skv,   \
			int64_t D, int64_t Dv, int64_t q_s0, int64_t q_s1, int64_t q_sS, int64_t q_sD, int64_t k_s0, int64_t k_s1, \
			int64_t k_sS, int64_t k_sD, int64_t v_s0, int64_t v_s1, int64_t v_sS, int64_t v_sD, int64_t o_s0,          \
			int64_t o_s1, int64_t o_sS, int64_t o_sD, float scale                                                      \
	) {                                                                                                                \
		return attention(                                                                                              \
				cudnnType, out, Q, K, V, b0, b1, Sq, Skv, D, Dv,                                                       \
				{q_s0, q_s1, q_sS, q_sD, k_s0, k_s1, k_sS, k_sD, v_s0, v_s1, v_sS, v_sD, o_s0, o_s1, o_sS, o_sD}, scale \
		);                                                                                                             \
	}
MDL_ATTENTION_DEFINE(bf16, uint16_t, fe::DataType_t::BFLOAT16)
MDL_ATTENTION_DEFINE(f16, uint16_t, fe::DataType_t::HALF)

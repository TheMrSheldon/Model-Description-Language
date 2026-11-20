// The cuDNN kernels of the kernel package cudnn (C ABI: include/mdl_cudnn.h): fused attention, forward and backward,
// with cuDNN's scaled dot product attention through the cuDNN frontend (an operation graph per configuration, built
// once and cached).
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
#include <vector>

namespace {

	namespace fe = cudnn_frontend;

	constexpr int kMaxDevices = 64;
	/** @brief The tensors' ids in the graphs. */
	enum Uid : int64_t { kQ = 1, kK, kV, kO, kStats, kdO, kdQ, kdK, kdV };

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

	/** @brief The dimensions b0, b1, Sq, Skv, D, Dv. */
	using Dims = std::array<int64_t, 6>;
	/** @brief The strides [b0, b1, seq, dim] of the tensors in the order of the function's arguments. */
	using Strides = std::vector<int64_t>;
	enum class Kind { forward, forwardWithStats, backward };
	using Key = std::tuple<int, Kind, int, Dims, Strides, float>;

	/** @brief Builds the graph's execution plan (false if cuDNN does not support the configuration). */
	bool build(fe::graph::Graph& graph, Context& ctx) {
		return graph.validate().is_good() && graph.build_operation_graph(ctx.handle).is_good() &&
			   graph.create_execution_plans({fe::HeurMode_t::A}).is_good() && graph.check_support(ctx.handle).is_good() &&
			   graph.build_plans(ctx.handle).is_good();
	}

	/** @brief The graph of a configuration (nullptr if cuDNN does not support it). */
	std::shared_ptr<fe::graph::Graph>
	graphFor(Context& ctx, Kind kind, fe::DataType_t type, const Dims& dims, const Strides& strides, float scale) {
		thread_local std::map<Key, std::shared_ptr<fe::graph::Graph>> graphs;
		const Key key{currentDevice(), kind, static_cast<int>(type), dims, strides, scale};
		if (auto it = graphs.find(key); it != graphs.end())
			return it->second;
		const auto [b0, b1, Sq, Skv, D, Dv] = dims;
		auto graph = std::make_shared<fe::graph::Graph>();
		graph->set_io_data_type(type)
				.set_intermediate_data_type(fe::DataType_t::FLOAT)
				.set_compute_data_type(fe::DataType_t::FLOAT);
		size_t next = 0; // the next tensor's strides
		auto strided = [&](std::shared_ptr<fe::graph::Tensor_attributes> t, int64_t uid, std::vector<int64_t> dim) {
			const size_t s = 4 * next++;
			t->set_uid(uid).set_dim(std::move(dim)).set_stride({strides[s], strides[s + 1], strides[s + 2], strides[s + 3]});
			return t;
		};
		auto input = [&](const char* name, int64_t uid, std::vector<int64_t> dim) {
			return strided(graph->tensor(fe::graph::Tensor_attributes().set_name(name)), uid, std::move(dim));
		};
		// The statistics: [b0, b1, Sq, 1] float32, contiguous.
		auto statsLayout = [&](fe::graph::Tensor_attributes& t) {
			t.set_uid(kStats).set_dim({b0, b1, Sq, 1}).set_stride({b1 * Sq, Sq, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
		};
		auto Q = input("Q", kQ, {b0, b1, Sq, D});
		auto K = input("K", kK, {b0, b1, Skv, D});
		auto V = input("V", kV, {b0, b1, Skv, Dv});
		if (kind != Kind::backward) {
			auto options = fe::graph::SDPA_attributes().set_name("attention").set_attn_scale(scale);
			options.set_is_inference(kind == Kind::forward);
			auto [O, stats] = graph->sdpa(Q, K, V, options);
			strided(O, kO, {b0, b1, Sq, Dv})->set_output(true);
			if (kind == Kind::forwardWithStats) {
				stats->set_output(true);
				statsLayout(*stats);
			}
		} else {
			auto O = input("O", kO, {b0, b1, Sq, Dv});
			auto dO = input("dO", kdO, {b0, b1, Sq, Dv});
			auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats"));
			statsLayout(*stats);
			auto options = fe::graph::SDPA_backward_attributes().set_name("attention_backward").set_attn_scale(scale);
			auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, options);
			strided(dQ, kdQ, {b0, b1, Sq, D})->set_output(true);
			strided(dK, kdK, {b0, b1, Skv, D})->set_output(true);
			strided(dV, kdV, {b0, b1, Skv, Dv})->set_output(true);
		}
		std::shared_ptr<fe::graph::Graph> result;
		if (build(*graph, ctx))
			result = graph;
		else
			std::fprintf(stderr, "mdl_attention: cuDNN does not support this attention configuration\n");
		graphs.emplace(key, result);
		return result;
	}

	int run(Kind kind, fe::DataType_t type, const Dims& dims, const Strides& strides, float scale,
			std::unordered_map<int64_t, void*> pack) {
		const auto [b0, b1, Sq, Skv, D, Dv] = dims;
		if (b0 * b1 * Sq * Skv * D * Dv == 0)
			return 0; // (\todo zero-filled gradients for empty sequences)
		Context* ctx = context();
		if (ctx == nullptr)
			return 1;
		const auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, currentDevice()));
		if (cudnnSetStream(ctx->handle, stream) != CUDNN_STATUS_SUCCESS)
			return 2;
		auto graph = graphFor(*ctx, kind, type, dims, strides, scale);
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
		if (!graph->execute(ctx->handle, pack, ctx->workspace).is_good())
			return 6;
		return 0;
	}

	void* mut(const void* p) { return const_cast<void*>(p); }

} // namespace

#define MDL_ATTENTION_DEFINE(suffix, type, cudnnType)                                                                  \
	extern "C" int mdl_attention_fwd_##suffix(                                                                         \
			type* out, float* lse, const type* Q, const type* K, const type* V, int64_t b0, int64_t b1, int64_t Sq,    \
			int64_t Skv, int64_t D, int64_t Dv, MDL_ATTENTION_STRIDES(q), MDL_ATTENTION_STRIDES(k),                    \
			MDL_ATTENTION_STRIDES(v), MDL_ATTENTION_STRIDES(o), int32_t with_lse, float scale                          \
	) {                                                                                                                \
		std::unordered_map<int64_t, void*> pack{{kQ, mut(Q)}, {kK, mut(K)}, {kV, mut(V)}, {kO, out}};                  \
		if (with_lse)                                                                                                  \
			pack.emplace(kStats, lse);                                                                                 \
		return run(                                                                                                    \
				with_lse ? Kind::forwardWithStats : Kind::forward, cudnnType, {b0, b1, Sq, Skv, D, Dv},                \
				{q_s0, q_s1, q_sS, q_sD, k_s0, k_s1, k_sS, k_sD, v_s0, v_s1, v_sS, v_sD, o_s0, o_s1, o_sS, o_sD},      \
				scale, std::move(pack)                                                                                 \
		);                                                                                                             \
	}                                                                                                                  \
	extern "C" int mdl_attention_bwd_##suffix(                                                                         \
			type* dq, type* dk, type* dv, const type* Q, const type* K, const type* V, const type* O, const type* dO,  \
			const float* lse, int64_t b0, int64_t b1, int64_t Sq, int64_t Skv, int64_t D, int64_t Dv,                  \
			MDL_ATTENTION_STRIDES(q), MDL_ATTENTION_STRIDES(k), MDL_ATTENTION_STRIDES(v), MDL_ATTENTION_STRIDES(o),    \
			MDL_ATTENTION_STRIDES(do), MDL_ATTENTION_STRIDES(dq), MDL_ATTENTION_STRIDES(dk),                           \
			MDL_ATTENTION_STRIDES(dv), float scale                                                                     \
	) {                                                                                                                \
		return run(                                                                                                    \
				Kind::backward, cudnnType, {b0, b1, Sq, Skv, D, Dv},                                                   \
				{q_s0,	q_s1,  q_sS,  q_sD,	 k_s0,	k_s1,  k_sS,  k_sD,	 v_s0,	v_s1,  v_sS,  v_sD,	 o_s0,	o_s1,  o_sS,   \
				 o_sD,	do_s0, do_s1, do_sS, do_sD, dq_s0, dq_s1, dq_sS, dq_sD, dk_s0, dk_s1, dk_sS, dk_sD, dv_s0, dv_s1, \
				 dv_sS, dv_sD},                                                                                        \
				scale,                                                                                                 \
				{{kQ, mut(Q)},                                                                                         \
				 {kK, mut(K)},                                                                                         \
				 {kV, mut(V)},                                                                                         \
				 {kO, mut(O)},                                                                                         \
				 {kdO, mut(dO)},                                                                                       \
				 {kStats, mut(lse)},                                                                                   \
				 {kdQ, dq},                                                                                            \
				 {kdK, dk},                                                                                            \
				 {kdV, dv}}                                                                                            \
		);                                                                                                             \
	}
MDL_ATTENTION_DEFINE(bf16, uint16_t, fe::DataType_t::BFLOAT16)
MDL_ATTENTION_DEFINE(f16, uint16_t, fe::DataType_t::HALF)

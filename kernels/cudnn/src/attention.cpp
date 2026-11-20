// The cuDNN kernels of the kernel package cudnn (C ABI: include/mdl_cudnn.h): fused attention, forward and backward,
// with cuDNN's scaled dot product attention through the cuDNN frontend (an operation graph per configuration, built
// once and cached).
#include "mdl_cudnn.h"
#include "mdl_strided.hpp"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <array>
#include <cstdint>
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

	template <typename T>
	T* at(T* p, int64_t offset, size_t size) {
		return p == nullptr ? nullptr : reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(p) + offset * size);
	}

	/** @brief A view with @p shape and @p strides (pointers into the caller's arrays). */
	mdl::kernels::View view(int64_t rank, const int64_t* shape, const int64_t* strides) { return {rank, shape, strides}; }

	/** @brief The shape [..., Sq, Dv] of the result (Q's batch dimensions and rows, V's columns). */
	std::vector<int64_t> outShape(int64_t rank, const int64_t* q_shape, const int64_t* v_shape) {
		std::vector<int64_t> s(q_shape, q_shape + rank);
		s.back() = v_shape[rank - 1];
		return s;
	}

	int forward(
			fe::DataType_t type, size_t size, void* out, float* lse, const void* Q, const void* K, const void* V,
			int64_t rank, const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape,
			const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides,
			int32_t with_lse, float scale
	) {
		const auto o_shape = outShape(rank, q_shape, v_shape);
		const int64_t Sq = q_shape[rank - 2], Skv = k_shape[rank - 2], D = q_shape[rank - 1], Dv = v_shape[rank - 1];
		int64_t slice = 0; // (the index of the slice: lse is contiguous)
		return mdl::kernels::forEachBatch2(
				{view(rank, q_shape, q_strides), view(rank, k_shape, k_strides), view(rank, v_shape, v_strides),
				 view(rank, o_shape.data(), o_strides)},
				[&](int64_t b0, int64_t b1, const std::vector<std::vector<int64_t>>& s, const std::vector<int64_t>& o) {
					std::unordered_map<int64_t, void*> pack{
							{kQ, at(mut(Q), o[0], size)}, {kK, at(mut(K), o[1], size)}, {kV, at(mut(V), o[2], size)},
							{kO, at(out, o[3], size)}
					};
					if (with_lse)
						pack.emplace(kStats, lse + slice * b0 * b1 * Sq);
					++slice;
					Strides strides;
					for (const auto& t : s)
						strides.insert(strides.end(), t.begin(), t.end());
					return run(
							with_lse ? Kind::forwardWithStats : Kind::forward, type, {b0, b1, Sq, Skv, D, Dv}, strides,
							scale, std::move(pack)
					);
				}
		);
	}

	int backward(
			fe::DataType_t type, size_t size, void* dq, void* dk, void* dv, const void* Q, const void* K, const void* V,
			const void* O, const void* dO, const float* lse, int64_t rank, const int64_t* q_shape,
			const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape,
			const int64_t* v_strides, const int64_t* o_strides, const int64_t* do_strides, const int64_t* dq_strides,
			const int64_t* dk_strides, const int64_t* dv_strides, float scale
	) {
		const auto o_shape = outShape(rank, q_shape, v_shape);
		const int64_t Sq = q_shape[rank - 2], Skv = k_shape[rank - 2], D = q_shape[rank - 1], Dv = v_shape[rank - 1];
		int64_t slice = 0;
		return mdl::kernels::forEachBatch2(
				{view(rank, q_shape, q_strides), view(rank, k_shape, k_strides), view(rank, v_shape, v_strides),
				 view(rank, o_shape.data(), o_strides), view(rank, o_shape.data(), do_strides),
				 view(rank, q_shape, dq_strides), view(rank, k_shape, dk_strides), view(rank, v_shape, dv_strides)},
				[&](int64_t b0, int64_t b1, const std::vector<std::vector<int64_t>>& s, const std::vector<int64_t>& o) {
					Strides strides;
					for (const auto& t : s)
						strides.insert(strides.end(), t.begin(), t.end());
					const float* stats = lse + slice * b0 * b1 * Sq;
					++slice;
					return run(
							Kind::backward, type, {b0, b1, Sq, Skv, D, Dv}, strides, scale,
							{{kQ, at(mut(Q), o[0], size)},
							 {kK, at(mut(K), o[1], size)},
							 {kV, at(mut(V), o[2], size)},
							 {kO, at(mut(O), o[3], size)},
							 {kdO, at(mut(dO), o[4], size)},
							 {kStats, mut(stats)},
							 {kdQ, at(dq, o[5], size)},
							 {kdK, at(dk, o[6], size)},
							 {kdV, at(dv, o[7], size)}}
					);
				}
		);
	}

} // namespace

#define MDL_ATTENTION_DEFINE(suffix, type, cudnnType)                                                                  \
	extern "C" int mdl_attention_fwd_##suffix(                                                                         \
			type* out, float* lse, const type* Q, const type* K, const type* V, int64_t rank, const int64_t* q_shape,  \
			const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape,        \
			const int64_t* v_strides, const int64_t* o_strides, int32_t with_lse, float scale                          \
	) {                                                                                                                \
		return forward(                                                                                                \
				cudnnType, sizeof(type), out, lse, Q, K, V, rank, q_shape, q_strides, k_shape, k_strides, v_shape,     \
				v_strides, o_strides, with_lse, scale                                                                  \
		);                                                                                                             \
	}                                                                                                                  \
	extern "C" int mdl_attention_bwd_##suffix(                                                                         \
			type* dq, type* dk, type* dv, const type* Q, const type* K, const type* V, const type* O, const type* dO,  \
			const float* lse, int64_t rank, const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape,  \
			const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides,      \
			const int64_t* do_strides, const int64_t* dq_strides, const int64_t* dk_strides,                           \
			const int64_t* dv_strides, float scale                                                                     \
	) {                                                                                                                \
		return backward(                                                                                               \
				cudnnType, sizeof(type), dq, dk, dv, Q, K, V, O, dO, lse, rank, q_shape, q_strides, k_shape,           \
				k_strides, v_shape, v_strides, o_strides, do_strides, dq_strides, dk_strides, dv_strides, scale        \
		);                                                                                                             \
	}
MDL_ATTENTION_DEFINE(bf16, uint16_t, fe::DataType_t::BFLOAT16)
MDL_ATTENTION_DEFINE(f16, uint16_t, fe::DataType_t::HALF)

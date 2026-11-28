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
	enum Uid : int64_t { kQ = 1, kK, kV, kO, kStats, kdO, kdQ, kdK, kdV, kBias, kSeqLen };

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
	/** @brief A bias of the scores: its dims and strides [b0, b1, Sq, Skv] (1: broadcast), empty for none. */
	using BiasLayout = std::vector<int64_t>;
	/**
	 * @brief Padding mode (instead of a bias): the number of valid positions of the queries and keys per element of the
	 * first batch dimension (int32 [b0]), and the strides of the statistics' batch dimensions.
	 */
	struct Padding {
		bool enabled = false;
		std::array<int64_t, 2> statsStrides{};
		auto operator<=>(const Padding&) const = default;
	};
	using Key = std::tuple<int, Kind, int, Dims, Strides, BiasLayout, Padding, float>;

	/** @brief Builds the graph's execution plan (false if cuDNN does not support the configuration). */
	bool build(fe::graph::Graph& graph, Context& ctx) {
		return graph.validate().is_good() && graph.build_operation_graph(ctx.handle).is_good() &&
			   graph.create_execution_plans({fe::HeurMode_t::A}).is_good() &&
			   graph.check_support(ctx.handle).is_good() &&
			   graph.build_plans(ctx.handle, fe::BuildPlanPolicy_t::ALL).is_good(); // (all: measured on first use)
	}

	/** @brief The graph of a configuration (nullptr if cuDNN does not support it). */
	std::shared_ptr<fe::graph::Graph> graphFor(
			Context& ctx, Kind kind, fe::DataType_t type, const Dims& dims, const Strides& strides,
			const BiasLayout& bias, const Padding& padding, float scale
	) {
		thread_local std::map<Key, std::shared_ptr<fe::graph::Graph>> graphs;
		const Key key{currentDevice(), kind, static_cast<int>(type), dims, strides, bias, padding, scale};
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
			t->set_uid(uid)
					.set_dim(std::move(dim))
					.set_stride({strides[s], strides[s + 1], strides[s + 2], strides[s + 3]});
			return t;
		};
		auto input = [&](const char* name, int64_t uid, std::vector<int64_t> dim) {
			return strided(graph->tensor(fe::graph::Tensor_attributes().set_name(name)), uid, std::move(dim));
		};
		// The statistics: [b0, b1, Sq, 1] float32, contiguous (with padding: the given batch strides).
		const auto statsStrides = padding.enabled ? padding.statsStrides : std::array<int64_t, 2>{b1 * Sq, Sq};
		auto statsLayout = [&](fe::graph::Tensor_attributes& t) {
			t.set_uid(kStats)
					.set_dim({b0, b1, Sq, 1})
					.set_stride({statsStrides[0], statsStrides[1], 1, 1})
					.set_data_type(fe::DataType_t::FLOAT);
		};
		std::shared_ptr<fe::graph::Tensor_attributes> lengths;
		if (padding.enabled)
			lengths = graph->tensor(
					fe::graph::Tensor_attributes()
							.set_name("lengths")
							.set_uid(kSeqLen)
							.set_dim({b0, 1, 1, 1})
							.set_stride({1, 1, 1, 1})
							.set_data_type(fe::DataType_t::INT32)
			);
		auto Q = input("Q", kQ, {b0, b1, Sq, D});
		auto K = input("K", kK, {b0, b1, Skv, D});
		auto V = input("V", kV, {b0, b1, Skv, Dv});
		std::shared_ptr<fe::graph::Tensor_attributes> B;
		if (!bias.empty())
			B = graph->tensor(
					fe::graph::Tensor_attributes()
							.set_name("bias")
							.set_uid(kBias)
							.set_dim({bias[0], bias[1], bias[2], bias[3]})
							.set_stride({bias[4], bias[5], bias[6], bias[7]})
			);
		if (kind != Kind::backward) {
			auto options = fe::graph::SDPA_attributes().set_name("attention").set_attn_scale(scale);
			if (B)
				options.set_bias(B);
			if (lengths)
				options.set_padding_mask(true).set_seq_len_q(lengths).set_seq_len_kv(lengths);
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
			if (B)
				options.set_bias(B);
			if (lengths)
				options.set_padding_mask(true).set_seq_len_q(lengths).set_seq_len_kv(lengths);
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

	int run(Kind kind, fe::DataType_t type, const Dims& dims, const Strides& strides, const BiasLayout& bias,
			const Padding& padding, float scale, std::unordered_map<int64_t, void*> pack) {
		const auto [b0, b1, Sq, Skv, D, Dv] = dims;
		if (b0 * b1 * Sq * Skv * D * Dv == 0)
			return 0; // (\todo zero-filled gradients for empty sequences)
		Context* ctx = context();
		if (ctx == nullptr)
			return 1;
		const auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, currentDevice()));
		if (cudnnSetStream(ctx->handle, stream) != CUDNN_STATUS_SUCCESS)
			return 2;
		auto graph = graphFor(*ctx, kind, type, dims, strides, bias, padding, scale);
		if (!graph)
			return 3;
		// The fastest of the heuristics' plans, measured once per configuration (on the first call's tensors: the
		// outputs are written again by the call itself).
		thread_local std::map<const fe::graph::Graph*, int64_t> chosen;
		auto workspaceFor = [&](int64_t plan) -> bool {
			int64_t bytes = 0;
			if (!graph->get_workspace_size_plan_at_index(plan, bytes).is_good())
				return false;
			if (bytes > ctx->workspaceBytes) {
				if (ctx->workspace != nullptr)
					cudaFree(ctx->workspace);
				ctx->workspace = nullptr;
				ctx->workspaceBytes = 0;
				if (cudaMalloc(&ctx->workspace, bytes) != cudaSuccess)
					return false;
				ctx->workspaceBytes = bytes;
			}
			return true;
		};
		auto it = chosen.find(graph.get());
		if (it == chosen.end()) {
			const int64_t count = graph->get_execution_plan_count();
			int64_t best = 0;
			float bestMs = -1;
			cudaEvent_t start = nullptr, stop = nullptr;
			if (count > 1 && cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess) {
				for (int64_t plan = 0; plan < count; ++plan) {
					if (!workspaceFor(plan) ||
						!graph->execute_plan_at_index(ctx->handle, pack, ctx->workspace, plan).is_good())
						continue; // (also the warm-up)
					constexpr int kRuns = 3;
					cudaEventRecord(start, stream);
					for (int r = 0; r < kRuns; ++r)
						graph->execute_plan_at_index(ctx->handle, pack, ctx->workspace, plan);
					cudaEventRecord(stop, stream);
					cudaEventSynchronize(stop);
					float ms = 0;
					cudaEventElapsedTime(&ms, start, stop);
					if (bestMs < 0 || ms < bestMs) {
						bestMs = ms;
						best = plan;
					}
				}
			}
			if (start != nullptr)
				cudaEventDestroy(start);
			if (stop != nullptr)
				cudaEventDestroy(stop);
			it = chosen.emplace(graph.get(), best).first;
		}
		if (!workspaceFor(it->second))
			return 5;
		if (!graph->execute_plan_at_index(ctx->handle, pack, ctx->workspace, it->second).is_good())
			return 6;
		return 0;
	}

	void* mut(const void* p) { return const_cast<void*>(p); }

	template <typename T>
	T* at(T* p, int64_t offset, size_t size) {
		return p == nullptr ? nullptr : reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(p) + offset * size);
	}

	/** @brief A view with @p shape and @p strides (pointers into the caller's arrays). */
	mdl::kernels::View view(int64_t rank, const int64_t* shape, const int64_t* strides) {
		return {rank, shape, strides};
	}

	/** @brief The shape [..., Sq, Dv] of the result (Q's batch dimensions and rows, V's columns). */
	std::vector<int64_t> outShape(int64_t rank, const int64_t* q_shape, const int64_t* v_shape) {
		std::vector<int64_t> s(q_shape, q_shape + rank);
		s.back() = v_shape[rank - 1];
		return s;
	}

	/**
	 * @brief A bias of the scores as a view whose broadcast dimensions (size 1) have stride 0, so that iterating over
	 * the leading dimensions (with Q's sizes) stays at its element.
	 */
	struct Bias {
		const void* data = nullptr;
		std::vector<int64_t> shape, strides;

		Bias(const void* data, int64_t rank, const int64_t* shape, const int64_t* strides) : data(data) {
			if (shape == nullptr || strides == nullptr) // (without a bias; its shape alone: e.g. with lengths)
				return;
			this->shape.assign(shape, shape + rank);
			this->strides.assign(strides, strides + rank);
			for (int64_t i = 0; i < rank; ++i)
				if (this->shape[i] == 1)
					this->strides[i] = 0;
		}
		mdl::kernels::View view(int64_t rank) const { return {rank, shape.data(), strides.data()}; }

		/**
		 * @brief The batch dimension of [b0, b1] that the lengths of a suffix mask index (the bias's batch dimension that
		 * is not broadcast), or -1.
		 */
		int lengthsAxis(int64_t rank) const {
			if (shape.empty() || (rank != 3 && rank != 4))
				return -1;
			if (rank == 3)
				return 1; // (the only batch dimension, after an outer one of size 1)
			const bool first = shape[0] != 1, second = shape[1] != 1;
			return first == second ? -1 : (first ? 0 : 1);
		}

		/** @brief The layout for cuDNN of the slice [b0, b1, Sq, Skv] with the strides @p s of the view. */
		BiasLayout layout(int64_t rank, const std::vector<int64_t>& s) const {
			if (data == nullptr)
				return {};
			auto dim = [&](int64_t i) { return i >= 0 ? shape[i] : int64_t{1}; };
			return {dim(rank - 4), dim(rank - 3), dim(rank - 2), dim(rank - 1), s[0], s[1], s[2], s[3]};
		}
	};

	int
	forward(fe::DataType_t type, size_t size, void* out, float* lse, const void* Q, const void* K, const void* V,
			int64_t rank, const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape,
			const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides,
			const Bias& bias, const int32_t* lengths, int32_t with_lse, float scale) {
		const int axis = lengths != nullptr ? bias.lengthsAxis(rank) : -1;
		if (lengths != nullptr && axis < 0)
			return 7; // (the manifest admits only ranks and biases whose lengths' dimension is known)
		const auto o_shape = outShape(rank, q_shape, v_shape);
		const int64_t Sq = q_shape[rank - 2], Skv = k_shape[rank - 2], D = q_shape[rank - 1], Dv = v_shape[rank - 1];
		std::vector<mdl::kernels::View> views{
				view(rank, q_shape, q_strides), view(rank, k_shape, k_strides), view(rank, v_shape, v_strides),
				view(rank, o_shape.data(), o_strides)
		};
		if (bias.data != nullptr)
			views.push_back(bias.view(rank));
		int64_t slice = 0; // (the index of the slice: lse is contiguous)
		return mdl::kernels::forEachBatch2(
				views,
				[&](int64_t b0, int64_t b1, const std::vector<std::vector<int64_t>>& s, const std::vector<int64_t>& o) {
					std::unordered_map<int64_t, void*> pack{
							{kQ, at(mut(Q), o[0], size)},
							{kK, at(mut(K), o[1], size)},
							{kV, at(mut(V), o[2], size)},
							{kO, at(out, o[3], size)}
					};
					if (bias.data != nullptr)
						pack.emplace(kBias, at(mut(bias.data), o[4], size));
					if (with_lse)
						pack.emplace(kStats, lse + slice * b0 * b1 * Sq);
					++slice;
					std::vector<std::vector<int64_t>> st(s.begin(), s.begin() + 4);
					int64_t c0 = b0, c1 = b1;
					Padding padding;
					if (lengths != nullptr) {
						// Padding mode: the lengths index cuDNN's first batch dimension (swapped if needed).
						padding = {true, {b1 * Sq, Sq}};
						pack.emplace(kSeqLen, mut(lengths));
						if (axis == 1) {
							std::swap(c0, c1);
							for (auto& t : st)
								std::swap(t[0], t[1]);
							std::swap(padding.statsStrides[0], padding.statsStrides[1]);
						}
					}
					Strides strides;
					for (const auto& t : st)
						strides.insert(strides.end(), t.begin(), t.end());
					return run(
							with_lse ? Kind::forwardWithStats : Kind::forward, type, {c0, c1, Sq, Skv, D, Dv}, strides,
							lengths != nullptr ? BiasLayout{} : bias.layout(rank, bias.data != nullptr ? s[4] : s[0]),
							padding, scale, std::move(pack)
					);
				}
		);
	}

	int backward(
			fe::DataType_t type, size_t size, void* dq, void* dk, void* dv, const void* Q, const void* K, const void* V,
			const void* O, const void* dO, const float* lse, int64_t rank, const int64_t* q_shape,
			const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape,
			const int64_t* v_strides, const int64_t* o_strides, const int64_t* do_strides, const int64_t* dq_strides,
			const int64_t* dk_strides, const int64_t* dv_strides, const Bias& bias, const int32_t* lengths, float scale
	) {
		const int axis = lengths != nullptr ? bias.lengthsAxis(rank) : -1;
		if (lengths != nullptr && axis < 0)
			return 7;
		const auto o_shape = outShape(rank, q_shape, v_shape);
		const int64_t Sq = q_shape[rank - 2], Skv = k_shape[rank - 2], D = q_shape[rank - 1], Dv = v_shape[rank - 1];
		std::vector<mdl::kernels::View> views{
				view(rank, q_shape, q_strides),			view(rank, k_shape, k_strides),
				view(rank, v_shape, v_strides),			view(rank, o_shape.data(), o_strides),
				view(rank, o_shape.data(), do_strides), view(rank, q_shape, dq_strides),
				view(rank, k_shape, dk_strides),		view(rank, v_shape, dv_strides)
		};
		if (bias.data != nullptr)
			views.push_back(bias.view(rank));
		int64_t slice = 0;
		return mdl::kernels::forEachBatch2(
				views,
				[&](int64_t b0, int64_t b1, const std::vector<std::vector<int64_t>>& s, const std::vector<int64_t>& o) {
					std::vector<std::vector<int64_t>> st(s.begin(), s.begin() + 8);
					int64_t c0 = b0, c1 = b1;
					Padding padding;
					if (lengths != nullptr) {
						padding = {true, {b1 * Sq, Sq}};
						if (axis == 1) {
							std::swap(c0, c1);
							for (auto& t : st)
								std::swap(t[0], t[1]);
							std::swap(padding.statsStrides[0], padding.statsStrides[1]);
						}
					}
					Strides strides;
					for (const auto& t : st)
						strides.insert(strides.end(), t.begin(), t.end());
					const float* stats = lse + slice * b0 * b1 * Sq;
					++slice;
					std::unordered_map<int64_t, void*> pack{
							{kQ, at(mut(Q), o[0], size)}, {kK, at(mut(K), o[1], size)},	  {kV, at(mut(V), o[2], size)},
							{kO, at(mut(O), o[3], size)}, {kdO, at(mut(dO), o[4], size)}, {kStats, mut(stats)},
							{kdQ, at(dq, o[5], size)},	  {kdK, at(dk, o[6], size)},	  {kdV, at(dv, o[7], size)}
					};
					if (bias.data != nullptr)
						pack.emplace(kBias, at(mut(bias.data), o[8], size));
					if (lengths != nullptr)
						pack.emplace(kSeqLen, mut(lengths));
					return run(
							Kind::backward, type, {c0, c1, Sq, Skv, D, Dv}, strides,
							lengths != nullptr ? BiasLayout{} : bias.layout(rank, bias.data != nullptr ? s[8] : s[0]),
							padding, scale, std::move(pack)
					);
				}
		);
	}

} // namespace

#define MDL_ATTENTION_DEFINE(suffix, type, cudnnType)                                                                  \
	extern "C" int mdl_attention_fwd_##suffix(                                                                         \
			type* out, float* lse, const type* Q, const type* K, const type* V, int64_t rank, const int64_t* q_shape,  \
			const int64_t* q_strides, const int64_t* k_shape, const int64_t* k_strides, const int64_t* v_shape,        \
			const int64_t* v_strides, const int64_t* o_strides, const type* bias, const int64_t* bias_shape,           \
			const int64_t* bias_strides, const int32_t* lengths, int32_t with_lse, float scale                         \
	) {                                                                                                                \
		return forward(                                                                                                \
				cudnnType, sizeof(type), out, lse, Q, K, V, rank, q_shape, q_strides, k_shape, k_strides, v_shape,     \
				v_strides, o_strides, Bias(bias, rank, bias_shape, bias_strides), lengths, with_lse, scale             \
		);                                                                                                             \
	}                                                                                                                  \
	extern "C" int mdl_attention_bwd_##suffix(                                                                         \
			type* dq, type* dk, type* dv, const type* Q, const type* K, const type* V, const type* O, const type* dO,  \
			const float* lse, int64_t rank, const int64_t* q_shape, const int64_t* q_strides, const int64_t* k_shape,  \
			const int64_t* k_strides, const int64_t* v_shape, const int64_t* v_strides, const int64_t* o_strides,      \
			const int64_t* do_strides, const int64_t* dq_strides, const int64_t* dk_strides,                           \
			const int64_t* dv_strides, const type* bias, const int64_t* bias_shape, const int64_t* bias_strides,       \
			const int32_t* lengths, float scale                                                                        \
	) {                                                                                                                \
		return backward(                                                                                               \
				cudnnType, sizeof(type), dq, dk, dv, Q, K, V, O, dO, lse, rank, q_shape, q_strides, k_shape,           \
				k_strides, v_shape, v_strides, o_strides, do_strides, dq_strides, dk_strides, dv_strides,              \
				Bias(bias, rank, bias_shape, bias_strides), lengths, scale                                             \
		);                                                                                                             \
	}
MDL_ATTENTION_DEFINE(bf16, uint16_t, fe::DataType_t::BFLOAT16)
MDL_ATTENTION_DEFINE(f16, uint16_t, fe::DataType_t::HALF)

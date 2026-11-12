/**
 * @file
 * @brief oneDNN implementation of the vendor kernels declared in <mdl/runtime/kernels.h>.
 * @details oneDNN primitives are expensive to create but cheap to execute. Each kernel therefore caches its primitive
 * keyed by the complete problem description (all shapes and flags), so a primitive is created once per distinct shape
 * and reused afterwards. Lookups take a shared lock; only the creation of a new primitive takes an exclusive one.
 */
#include <mdl/runtime/kernels.h>

#include <oneapi/dnnl/dnnl.hpp>

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace {
	using dnnl::memory;
	using dt = dnnl::memory::data_type;

	/** @brief The number of physical cores (from the Linux CPU topology; the logical CPUs elsewhere). */
	int physicalCores() {
		std::set<std::pair<int, int>> cores; // (package, core)
		for (unsigned cpu = 0;; ++cpu) {
			const std::string topology = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
			std::ifstream package{topology + "physical_package_id"}, core{topology + "core_id"};
			int p = 0, c = 0;
			if (!(package >> p) || !(core >> c))
				break;
			cores.emplace(p, c);
		}
		if (!cores.empty())
			return static_cast<int>(cores.size());
		return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
	}

	/**
	 * @brief Unless OMP_NUM_THREADS is set, runs the kernels with one thread per physical core: SMT siblings only add
	 * contention to these compute-bound kernels (and made latencies erratic in measurements). Applied per calling
	 * thread, as OpenMP's thread count is a per-thread setting.
	 */
	void configureThreads() {
		static const int threads = std::getenv("OMP_NUM_THREADS") != nullptr ? 0 : physicalCores();
		thread_local bool configured = false;
		if (!std::exchange(configured, true) && threads > 0)
			omp_set_num_threads(threads);
	}

	dnnl::engine& engine() {
		configureThreads();
		static dnnl::engine e(dnnl::engine::kind::cpu, 0);
		return e;
	}

	dnnl::stream& stream() {
		thread_local dnnl::stream s(engine());
		return s;
	}

	/** @brief A thread-safe cache mapping problem keys onto created primitives. */
	template <typename Key, typename Primitive>
	class PrimitiveCache {
	public:
		template <typename Create>
		const Primitive& get(const Key& key, Create&& create) {
			{
				std::shared_lock lock{mutex};
				if (auto it = cache.find(key); it != cache.end())
					return it->second;
			}
			Primitive primitive = create();
			std::unique_lock lock{mutex};
			return cache.try_emplace(key, std::move(primitive)).first->second;
		}

	private:
		std::shared_mutex mutex;
		std::map<Key, Primitive> cache; // std::map: references stay valid on insertion
	};

	/** @brief Runs @p fn and converts exceptions into a non-zero return code (the C ABI must not throw). */
	template <typename Fn>
	int guarded(Fn&& fn) noexcept {
		try {
			fn();
			return 0;
		} catch (...) {
			return -1;
		}
	}
} // namespace

extern "C" int mdl_matmul_f32(
		float* dst, const float* A, const float* B, const float* bias, int32_t has_bias, int64_t batch,
		int64_t batchB, int64_t M, int64_t K, int64_t N, int32_t trans_b, float alpha
) {
	if (!has_bias)
		bias = nullptr;
	return guarded([&] {
		using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, bool, float>;
		static PrimitiveCache<Key, dnnl::matmul> cache;

		const memory::desc srcMd({batch, M, K}, dt::f32, {M * K, K, 1});
		// B[b] is [K, N] row-major, or [N, K] row-major (viewed as a column-major [K, N]) if transposed.
		const memory::dims wStrides = trans_b ? memory::dims{batchB == 1 ? 0 : N * K, 1, K}
											  : memory::dims{batchB == 1 ? 0 : K * N, N, 1};
		const memory::desc wMd({batchB, K, N}, dt::f32, wStrides);
		const memory::desc biasMd({1, 1, N}, dt::f32, {N, N, 1});
		const memory::desc dstMd({batch, M, N}, dt::f32, {M * N, N, 1});

		const auto& prim = cache.get(Key{batch, batchB, M, K, N, trans_b, bias != nullptr, alpha}, [&] {
			dnnl::primitive_attr attr;
			if (alpha != 1.0f)
				attr.set_scales_mask(DNNL_ARG_SRC, 0);
			auto pd = bias ? dnnl::matmul::primitive_desc(engine(), srcMd, wMd, biasMd, dstMd, attr)
						   : dnnl::matmul::primitive_desc(engine(), srcMd, wMd, dstMd, attr);
			return dnnl::matmul(pd);
		});

		memory src(srcMd, engine(), const_cast<float*>(A));
		memory wei(wMd, engine(), const_cast<float*>(B));
		memory out(dstMd, engine(), dst);
		std::unordered_map<int, memory> args{{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, out}};
		if (bias)
			args.emplace(DNNL_ARG_BIAS, memory(biasMd, engine(), const_cast<float*>(bias)));
		float scale = alpha;
		if (alpha != 1.0f)
			args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, memory({{1}, dt::f32, {1}}, engine(), &scale));
		prim.execute(stream(), args);
		stream().wait();
	});
}

extern "C" int mdl_softmax_f32(float* dst, const float* src, int64_t outer, int64_t axis) {
	return guarded([&] {
		static PrimitiveCache<std::pair<int64_t, int64_t>, dnnl::softmax_forward> cache;
		const memory::desc md({outer, axis}, dt::f32, memory::format_tag::nc);
		const auto& prim = cache.get({outer, axis}, [&] {
			return dnnl::softmax_forward(dnnl::softmax_forward::primitive_desc(
					engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate, md, md, 1
			));
		});
		prim.execute(
				stream(), {{DNNL_ARG_SRC, memory(md, engine(), const_cast<float*>(src))},
						   {DNNL_ARG_DST, memory(md, engine(), dst)}}
		);
		stream().wait();
	});
}

extern "C" int mdl_gelu_f32(float* dst, const float* src, int64_t n) {
	return guarded([&] {
		static PrimitiveCache<int64_t, dnnl::eltwise_forward> cache;
		const memory::desc md({n}, dt::f32, memory::format_tag::a);
		const auto& prim = cache.get(n, [&] {
			return dnnl::eltwise_forward(dnnl::eltwise_forward::primitive_desc(
					engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_gelu_erf, md, md, 0.f, 0.f
			));
		});
		prim.execute(
				stream(), {{DNNL_ARG_SRC, memory(md, engine(), const_cast<float*>(src))},
						   {DNNL_ARG_DST, memory(md, engine(), dst)}}
		);
		stream().wait();
	});
}

extern "C" int mdl_layer_norm_f32(
		float* dst, const float* src, const float* scale, const float* shift, int64_t rows, int64_t channels, float eps
) {
	return guarded([&] {
		using Key = std::tuple<int64_t, int64_t, float>;
		static PrimitiveCache<Key, dnnl::layer_normalization_forward> cache;
		const memory::desc md({rows, channels}, dt::f32, memory::format_tag::nc);
		const memory::desc statMd({rows}, dt::f32, memory::format_tag::a);
		const memory::desc vecMd({channels}, dt::f32, memory::format_tag::a);
		const auto& prim = cache.get(Key{rows, channels, eps}, [&] {
			return dnnl::layer_normalization_forward(dnnl::layer_normalization_forward::primitive_desc(
					engine(), dnnl::prop_kind::forward_inference, md, md, statMd, eps,
					dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift
			));
		});
		prim.execute(
				stream(), {{DNNL_ARG_SRC, memory(md, engine(), const_cast<float*>(src))},
						   {DNNL_ARG_DST, memory(md, engine(), dst)},
						   {DNNL_ARG_SCALE, memory(vecMd, engine(), const_cast<float*>(scale))},
						   {DNNL_ARG_SHIFT, memory(vecMd, engine(), const_cast<float*>(shift))}}
		);
		stream().wait();
	});
}

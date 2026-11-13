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

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <map>
#include <optional>
#include <vector>
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
	// ── element types ─────────────────────────────────────────────────────────────────────────────────────────

#if defined(__x86_64__) || defined(_M_X64)
	/** @brief float16 → float32 with F16C (8 elements per instruction). */
	__attribute__((target("f16c,avx"))) void halfToFloatF16C(const uint16_t* src, float* dst, int64_t n) {
#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < n / 8; ++i)
			_mm256_storeu_ps(dst + 8 * i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8 * i))));
		for (int64_t i = n / 8 * 8; i < n; ++i)
			dst[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(src[i])));
	}

	/** @brief float32 → float16 with F16C (round to nearest even). */
	__attribute__((target("f16c,avx"))) void floatToHalfF16C(const float* src, uint16_t* dst, int64_t n) {
#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < n / 8; ++i)
			_mm_storeu_si128(
					reinterpret_cast<__m128i*>(dst + 8 * i), _mm256_cvtps_ph(_mm256_loadu_ps(src + 8 * i), _MM_FROUND_TO_NEAREST_INT)
			);
		for (int64_t i = n / 8 * 8; i < n; ++i)
			dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(src[i]), _MM_FROUND_TO_NEAREST_INT)));
	}

	bool hasF16C() {
		static const bool supported = __builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx");
		return supported;
	}
#endif

	/**
	 * @brief Converts @p n contiguous elements from @p from to @p to with a oneDNN reorder (cached per size and types);
	 * float16 with F16C where available (oneDNN's float16 reorder is a slow generic one on CPUs without float16
	 * arithmetic).
	 */
	void convert(dt from, const void* src, dt to, void* dst, int64_t n) {
#if defined(__x86_64__) || defined(_M_X64)
		if (hasF16C() && from == dt::f16 && to == dt::f32)
			return halfToFloatF16C(static_cast<const uint16_t*>(src), static_cast<float*>(dst), n);
		if (hasF16C() && from == dt::f32 && to == dt::f16)
			return floatToHalfF16C(static_cast<const float*>(src), static_cast<uint16_t*>(dst), n);
#endif
		using Key = std::tuple<int, int, int64_t>;
		static PrimitiveCache<Key, dnnl::reorder> cache;
		const memory::desc fromMd({n}, from, memory::format_tag::a), toMd({n}, to, memory::format_tag::a);
		const auto& prim = cache.get(Key{static_cast<int>(from), static_cast<int>(to), n}, [&] {
			return dnnl::reorder(dnnl::reorder::primitive_desc(engine(), fromMd, engine(), toMd));
		});
		memory s(fromMd, engine(), const_cast<void*>(src)), d(toMd, engine(), dst);
		prim.execute(stream(), s, d);
		stream().wait();
	}

	/** @brief Float32 copies of 16-bit tensors (per thread, reused between calls). */
	struct Scratch {
		std::vector<std::vector<float>> buffers;

		float* get(size_t index, int64_t n) {
			if (buffers.size() <= index)
				buffers.resize(index + 1);
			if (buffers[index].size() < static_cast<size_t>(n))
				buffers[index].resize(static_cast<size_t>(n));
			return buffers[index].data();
		}
	};

	/** @brief A float32 copy of @p n elements of type @p t at @p src (in scratch slot @p slot). */
	float* widened(Scratch& scratch, size_t slot, dt t, const void* src, int64_t n) {
		float* copy = scratch.get(slot, n);
		convert(t, src, dt::f32, copy, n);
		return copy;
	}

	/**
	 * @brief Whether oneDNN implements @p create for the 16-bit type (else the kernel computes in float32 on
	 * converted copies, e.g. float16 on CPUs without float16 arithmetic); remembered per kernel and type.
	 */
	template <typename Create>
	bool implemented(Create&& create) {
		try {
			create();
			return true;
		} catch (const dnnl::error& e) {
			if (e.status == dnnl_unimplemented)
				return false;
			throw;
		}
	}

	// ── kernels for one element type ──────────────────────────────────────────────────────────────────────────

	void matmul(
			dt t, void* dst, const void* A, const void* B, const void* bias, int64_t batch, int64_t batchB, int64_t M,
			int64_t K, int64_t N, int32_t trans_a, int32_t trans_b, float alpha
	) {
		using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int32_t, bool, float>;
		static PrimitiveCache<Key, std::optional<dnnl::matmul>> cache;

		// A[b] is [M, K] row-major, or [K, M] row-major (viewed as a column-major [M, K]) if transposed.
		const memory::desc srcMd({batch, M, K}, t, trans_a ? memory::dims{M * K, 1, M} : memory::dims{M * K, K, 1});
		// B[b] is [K, N] row-major, or [N, K] row-major (viewed as a column-major [K, N]) if transposed.
		const memory::dims wStrides = trans_b ? memory::dims{batchB == 1 ? 0 : N * K, 1, K}
											  : memory::dims{batchB == 1 ? 0 : K * N, N, 1};
		const memory::desc wMd({batchB, K, N}, t, wStrides);
		const memory::desc biasMd({1, 1, N}, t, {N, N, 1});
		const memory::desc dstMd({batch, M, N}, t, {M * N, N, 1});

		const auto& prim = cache.get(
				Key{static_cast<int>(t), batch, batchB, M, K, N, trans_a, trans_b, bias != nullptr, alpha},
				[&]() -> std::optional<dnnl::matmul> {
					dnnl::primitive_attr attr;
					if (alpha != 1.0f)
						attr.set_scales_mask(DNNL_ARG_SRC, 0);
					auto create = [&] {
						return bias ? dnnl::matmul::primitive_desc(engine(), srcMd, wMd, biasMd, dstMd, attr)
									: dnnl::matmul::primitive_desc(engine(), srcMd, wMd, dstMd, attr);
					};
					if (t != dt::f32 && !implemented(create))
						return std::nullopt;
					return dnnl::matmul(create());
				}
		);
		if (!prim) { // not implemented for the type: in float32 on converted copies
			thread_local Scratch scratch;
			const float* a = widened(scratch, 0, t, A, batch * M * K);
			const float* b = widened(scratch, 1, t, B, batchB * K * N);
			const float* c = bias ? widened(scratch, 2, t, bias, N) : nullptr;
			float* out = scratch.get(3, batch * M * N);
			matmul(dt::f32, out, a, b, c, batch, batchB, M, K, N, trans_a, trans_b, alpha);
			convert(dt::f32, out, t, dst, batch * M * N);
			return;
		}

		memory src(srcMd, engine(), const_cast<void*>(A));
		memory wei(wMd, engine(), const_cast<void*>(B));
		memory out(dstMd, engine(), dst);
		std::unordered_map<int, memory> args{{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, out}};
		if (bias)
			args.emplace(DNNL_ARG_BIAS, memory(biasMd, engine(), const_cast<void*>(bias)));
		float scale = alpha;
		if (alpha != 1.0f)
			args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, memory({{1}, dt::f32, {1}}, engine(), &scale));
		prim->execute(stream(), args);
		stream().wait();
	}

	void softmax(dt t, void* dst, const void* src, int64_t outer, int64_t axis) {
		using Key = std::tuple<int, int64_t, int64_t>;
		static PrimitiveCache<Key, std::optional<dnnl::softmax_forward>> cache;
		const memory::desc md({outer, axis}, t, memory::format_tag::nc);
		const auto& prim = cache.get(Key{static_cast<int>(t), outer, axis}, [&]() -> std::optional<dnnl::softmax_forward> {
			auto create = [&] {
				return dnnl::softmax_forward::primitive_desc(
						engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate, md, md, 1
				);
			};
			if (t != dt::f32 && !implemented(create))
				return std::nullopt;
			return dnnl::softmax_forward(create());
		});
		if (!prim) {
			thread_local Scratch scratch;
			float* x = widened(scratch, 0, t, src, outer * axis);
			softmax(dt::f32, x, x, outer, axis);
			convert(dt::f32, x, t, dst, outer * axis);
			return;
		}
		prim->execute(
				stream(), {{DNNL_ARG_SRC, memory(md, engine(), const_cast<void*>(src))}, {DNNL_ARG_DST, memory(md, engine(), dst)}}
		);
		stream().wait();
	}

	void gelu(dt t, void* dst, const void* src, int64_t n) {
		using Key = std::tuple<int, int64_t>;
		static PrimitiveCache<Key, std::optional<dnnl::eltwise_forward>> cache;
		const memory::desc md({n}, t, memory::format_tag::a);
		const auto& prim = cache.get(Key{static_cast<int>(t), n}, [&]() -> std::optional<dnnl::eltwise_forward> {
			auto create = [&] {
				return dnnl::eltwise_forward::primitive_desc(
						engine(), dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_gelu_erf, md, md, 0.f, 0.f
				);
			};
			if (t != dt::f32 && !implemented(create))
				return std::nullopt;
			return dnnl::eltwise_forward(create());
		});
		if (!prim) {
			thread_local Scratch scratch;
			float* x = widened(scratch, 0, t, src, n);
			gelu(dt::f32, x, x, n);
			convert(dt::f32, x, t, dst, n);
			return;
		}
		prim->execute(
				stream(), {{DNNL_ARG_SRC, memory(md, engine(), const_cast<void*>(src))}, {DNNL_ARG_DST, memory(md, engine(), dst)}}
		);
		stream().wait();
	}

	void layerNorm(
			dt t, void* dst, const void* src, const void* scale, const void* shift, int64_t rows, int64_t channels, float eps
	) {
		using Key = std::tuple<int, int64_t, int64_t, float>;
		static PrimitiveCache<Key, std::optional<dnnl::layer_normalization_forward>> cache;
		const memory::desc md({rows, channels}, t, memory::format_tag::nc);
		const memory::desc statMd({rows}, dt::f32, memory::format_tag::a);
		const memory::desc vecMd({channels}, t, memory::format_tag::a);
		const auto& prim = cache.get(
				Key{static_cast<int>(t), rows, channels, eps}, [&]() -> std::optional<dnnl::layer_normalization_forward> {
					auto create = [&] {
						return dnnl::layer_normalization_forward::primitive_desc(
								engine(), dnnl::prop_kind::forward_inference, md, md, statMd, eps,
								dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift
						);
					};
					if (t != dt::f32 && !implemented(create))
						return std::nullopt;
					return dnnl::layer_normalization_forward(create());
				}
		);
		if (!prim) {
			thread_local Scratch scratch;
			float* x = widened(scratch, 0, t, src, rows * channels);
			const float* g = widened(scratch, 1, t, scale, channels);
			const float* b = widened(scratch, 2, t, shift, channels);
			layerNorm(dt::f32, x, x, g, b, rows, channels, eps);
			convert(dt::f32, x, t, dst, rows * channels);
			return;
		}
		// oneDNN takes float32 scale and shift (also for 16-bit data).
		const float* g = static_cast<const float*>(scale);
		const float* b = static_cast<const float*>(shift);
		thread_local Scratch vectors;
		if (t != dt::f32) {
			g = widened(vectors, 0, t, scale, channels);
			b = widened(vectors, 1, t, shift, channels);
		}
		const memory::desc vec32({channels}, dt::f32, memory::format_tag::a);
		prim->execute(
				stream(), {{DNNL_ARG_SRC, memory(md, engine(), const_cast<void*>(src))},
						   {DNNL_ARG_DST, memory(md, engine(), dst)},
						   {DNNL_ARG_SCALE, memory(vec32, engine(), const_cast<float*>(g))},
						   {DNNL_ARG_SHIFT, memory(vec32, engine(), const_cast<float*>(b))}}
		);
		stream().wait();
	}
} // namespace

// ── C ABI ─────────────────────────────────────────────────────────────────────────────────────────────────────────

#define MDL_MATMUL(suffix, type, dtype)                                                                                \
	extern "C" int mdl_matmul_##suffix(                                                                                \
			type* dst, const type* A, const type* B, const type* bias, int32_t has_bias, int64_t batch,                \
			int64_t batchB, int64_t M, int64_t K, int64_t N, int32_t trans_a, int32_t trans_b, float alpha             \
	) {                                                                                                                \
		return guarded([&] {                                                                                           \
			matmul(dtype, dst, A, B, has_bias ? bias : nullptr, batch, batchB, M, K, N, trans_a, trans_b, alpha);    \
		});                                                                                                            \
	}                                                                                                                  \
	extern "C" int mdl_softmax_##suffix(type* dst, const type* src, int64_t outer, int64_t axis) {                    \
		return guarded([&] { softmax(dtype, dst, src, outer, axis); });                                                \
	}                                                                                                                  \
	extern "C" int mdl_gelu_##suffix(type* dst, const type* src, int64_t n) {                                         \
		return guarded([&] { gelu(dtype, dst, src, n); });                                                             \
	}                                                                                                                  \
	extern "C" int mdl_layer_norm_##suffix(                                                                            \
			type* dst, const type* src, const type* scale, const type* shift, int64_t rows, int64_t channels, float eps \
	) {                                                                                                                \
		return guarded([&] { layerNorm(dtype, dst, src, scale, shift, rows, channels, eps); });                        \
	}

MDL_MATMUL(f32, float, dt::f32)
MDL_MATMUL(bf16, uint16_t, dt::bf16)
MDL_MATMUL(f16, uint16_t, dt::f16)

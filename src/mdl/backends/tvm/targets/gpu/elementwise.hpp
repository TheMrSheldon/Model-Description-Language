#pragma once

/**
 * @file elementwise.hpp
 * @brief A GPU schedule rule for elementwise kernels and copies along the innermost dimension: vector loads and stores.
 */

#include <tvm_ported/s_tir/dlight/gpu/base.hpp>

namespace mdl::backend::tvm::gpu {

	/**
	 * @brief An elementwise kernel (also a copy that permutes the outer dimensions, or broadcasts along them) whose
	 * buffers all have the innermost iterator as their innermost index, e.g. GELU, a residual addition, a conversion,
	 * `x[m, n] + b[n]`, or the transposition of the heads of attention ([heads, batch, seq, dim] to [batch, seq, heads,
	 * dim]): kBytes per thread (e.g. 4 bfloat16 values) as vectors of adjacent elements of the innermost dimension, if
	 * its size is a multiple (provably, also for symbolic sizes). Otherwise DLight's rules schedule it. Our rule, not a
	 * port of TVM's; applied before DLight's Transpose and Fallback.
	 */
	class Elementwise final : public mdl::tvm_ported::dlight::gpu::GPUScheduleRule {
	public:
		static constexpr int64_t kBytes = 8; ///< (TVM's CUDA code generation: at most 4 lanes of 16-bit types)
		static constexpr int64_t kThreads = 256;

		std::optional<::tvm::s_tir::Schedule>
		apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const override;
	};

} // namespace mdl::backend::tvm::gpu

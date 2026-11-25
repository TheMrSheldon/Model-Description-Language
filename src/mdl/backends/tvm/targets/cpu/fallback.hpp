#pragma once

/**
 * @file fallback.hpp
 * @brief The CPU schedule rule for the kernels that DLight's CPU rules decline.
 */

#include <tvm_ported/s_tir/dlight/cpu/base.hpp>

namespace mdl::backend::tvm::cpu {

	/**
	 * @brief Inlines the elementwise blocks into their consumers, then schedules each remaining block: an injective
	 * block is fused, vectorized and parallelized; a block with reductions (sums, maxima, TE matmuls) gets its innermost
	 * spatial loop vectorized inside the reductions and its outer spatial loops parallelized. Our rule, not a port of
	 * TVM's; applied after DLight's CPU Reduction.
	 */
	class Fallback final : public mdl::tvm_ported::dlight::cpu::CPUScheduleRule {
	public:
		static constexpr int64_t kVectorLanes = 16;			 ///< float32 lanes of an AVX-512 register
		static constexpr int64_t kMinParallelWork = 1 << 14; ///< below this many iterations, parallelization costs more

		std::optional<::tvm::s_tir::Schedule>
		apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const override;
	};

} // namespace mdl::backend::tvm::cpu

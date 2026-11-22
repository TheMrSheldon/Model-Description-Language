#pragma once

/**
 * @file elementwise.hpp
 * @brief A GPU schedule rule for elementwise kernels over flat buffers: vector loads and stores.
 */

#include <tvm_ported/s_tir/dlight/gpu/base.hpp>

namespace mdl::backend::tvm::gpu {

	/**
	 * @brief An elementwise kernel whose buffers are all accessed at the same flat index (e.g. GELU, a residual addition,
	 * a conversion; after the backend's flattening of injective kernels, see flatten_injective.hpp): kBytes per thread
	 * (e.g. 4 bfloat16 values), loaded and stored as vectors, if the number of elements is a multiple (provably, also
	 * for symbolic sizes). Otherwise, or with operands at other indices (broadcasts), DLight's rules schedule it. Our
	 * rule, not a port of TVM's; applied before DLight's Fallback.
	 */
	class Elementwise final : public mdl::tvm_ported::dlight::gpu::GPUScheduleRule {
	public:
		static constexpr int64_t kBytes = 8; ///< (TVM's CUDA code generation: at most 4 lanes of 16-bit types)
		static constexpr int64_t kThreads = 256;

		std::optional<::tvm::s_tir::Schedule>
		apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const override;
	};

} // namespace mdl::backend::tvm::gpu

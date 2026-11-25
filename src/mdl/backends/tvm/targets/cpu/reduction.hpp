#pragma once

/**
 * @file reduction.hpp
 * @brief DLight's CPU Reduction for the kernels it can schedule.
 */

#include <tvm_ported/s_tir/dlight/cpu/cpu.hpp>

namespace mdl::backend::tvm::cpu {

	/**
	 * @brief DLight's CPU Reduction (softmax, normalizations), for kernels with one result: it computes every block
	 * at the last block's loops, which TVM's compute_at cannot do for further results (TVM's FuseOps never forms
	 * kernels with several results; our fusion planner does). Others are left to the next rule.
	 */
	class Reduction final : public mdl::tvm_ported::dlight::cpu::CPUScheduleRule {
	public:
		std::optional<::tvm::s_tir::Schedule>
		apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const override;
	};

} // namespace mdl::backend::tvm::cpu

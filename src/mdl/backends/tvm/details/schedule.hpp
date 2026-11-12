#pragma once

/**
 * @file schedule.hpp
 * @brief Default CPU schedule for TVM-generated kernels (TVM 0.24 does not schedule kernels for CPUs by default).
 */

#include <tvm/ir/module.h>

namespace mdl::backend::tvm {

	/**
	 * @brief Parallelizes and vectorizes the TVM-generated kernels (PrimFuncs with a root block) of @p mod.
	 * @details Injective kernels are fused, vectorized, and parallelized; reductions (including TE matmuls) get their
	 * innermost spatial loop vectorized inside the reduction and their outer spatial loops parallelized. Extern kernels
	 * and kernels whose schedule fails are left unchanged.
	 */
	::tvm::IRModule applyDefaultCPUSchedule(::tvm::IRModule mod);

} // namespace mdl::backend::tvm

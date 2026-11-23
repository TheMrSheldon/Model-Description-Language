#pragma once
/**
 * @file fusion.hpp
 * @brief Kernel fusion: which kernel calls become one kernel, and those kernels (target-independent).
 */
#include <tvm/ir/transform.h>

namespace mdl::backend::tvm::fusion {

	/**
	 * @brief Groups the legalized kernel calls (planner.hpp) and replaces each group by a primitive function, which
	 * FuseTIR makes one kernel of (inspired by PyTorch Inductor: decompose, then fuse; library calls stay as they are).
	 * @details Replaces TVM's FuseOps, which fuses producers into consumers only, with at most one reduction per group
	 * and one result: the planner also fuses calls that read the same tensors, several reductions over the same rows
	 * or columns, and groups with several results.
	 */
	::tvm::transform::Pass fuseKernels();

} // namespace mdl::backend::tvm::fusion

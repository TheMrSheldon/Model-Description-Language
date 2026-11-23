#pragma once
/**
 * @file materialize.hpp
 * @brief Planned groups as functions that FuseTIR makes kernels of.
 */
#include "./planner.hpp"

#include <tvm/ir/module.h>

namespace mdl::backend::tvm::fusion {

	/**
	 * @brief Replaces each group of @p groups by a call of a primitive function of its bindings (TVM's grouping, as
	 * FuseOps' and FuseOpsByPattern's: several results become a tuple). A result of a group that the group also uses
	 * itself is computed again at the end of the function, so that every result is written by a trailing epilogue and
	 * every other computation can be inlined (the form the targets' schedule rules expect). Copies of results that
	 * were not fused (duplicateProducers) are merged again (common subexpression elimination outside the groups).
	 */
	::tvm::IRModule materialize(::tvm::IRModule mod, const Partition& groups);

} // namespace mdl::backend::tvm::fusion

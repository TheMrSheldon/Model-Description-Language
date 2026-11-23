#pragma once
/**
 * @file duplicate_producers.hpp
 * @brief Cheap pointwise results with several consumers, computed once per consumer.
 */
#include <tvm/ir/transform.h>

namespace mdl::backend::tvm::fusion {

	/**
	 * @brief A cheap pointwise kernel call whose result only kernel calls of the same block read, each of which could
	 * contain it (pointwise over its domain, or reducing it), is bound once per reader: each copy can then be fused
	 * into its reader instead of being written and read again (e.g. the `erf` that GELU and its gradient share).
	 * @details A group (planner.hpp) contains each binding once; this pass gives the planner the choice. Copies that
	 * are not fused are merged again by common subexpression elimination.
	 */
	::tvm::transform::Pass duplicateProducers();

} // namespace mdl::backend::tvm::fusion

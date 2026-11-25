#pragma once

/**
 * @file merge_sums.hpp
 * @brief Sibling sums (over the same loops) as one block with two results.
 */

#include <tvm/ir/module.h>

namespace mdl::backend::tvm {

	/**
	 * @brief Merges pairs of sums over the same iteration domain that do not depend on each other (e.g. the gradients of
	 * a normalization's scale and shift: `sum(dy)` and `sum(dy * x̂)` over the rows) into one block with two results,
	 * TVM's two-sum reducer. Schedules then compute both in one loop body, reading shared inputs once (as separate blocks,
	 * each has its own body, loads and predicates). The sums may have producers and consumers between them: the root's
	 * statements are reordered by their dependencies. Target-independent; applied before scheduling. (bfloat16 sums need
	 * patches/tvm/0001-bf16-compute-legalize-bind.patch.)
	 */
	::tvm::IRModule mergeSiblingSums(::tvm::IRModule kernels);

} // namespace mdl::backend::tvm

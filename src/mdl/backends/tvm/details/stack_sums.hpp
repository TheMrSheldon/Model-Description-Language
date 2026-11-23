#pragma once
/**
 * @file stack_sums.hpp
 * @brief Sibling column sums: sums to the same shape of tensors that are computed from a common tensor, as one sum.
 */
#include <tvm/ir/transform.h>

namespace mdl::backend::tvm {
	/**
	 * @brief Rewrites `collapse_sum_to(x_i, S)` (i = 1..k; inputs of the same shape that share a tensor: one is the
	 * other's argument, or their elementwise producers have a common argument) to one
	 * `collapse_sum_to(stack(x_1, ..., x_k, axis=-1), S + [k])` and takes the results from it.
	 * @details E.g. the gradients of a normalization's bias and scale, sum(dy) and sum(dy * x̂): one reduction (its
	 * producers fused into it) reads dy once, with the k sums of a column computed by adjacent threads (the stacked axis
	 * is innermost). Target-independent; before legalization.
	 */
	::tvm::transform::Pass stackSiblingSums();
} // namespace mdl::backend::tvm

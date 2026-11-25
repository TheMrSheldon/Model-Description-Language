#pragma once

/**
 * @file flatten_injective.hpp
 * @brief Injective kernels as one flat loop over the elements of their result.
 */

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

namespace mdl::backend::tvm {

	/**
	 * @brief Rewrites injective kernels (elementwise operations, casts, copies and reshapes, broadcasts of trailing
	 * dimensions such as a bias) into one loop over the flat index n of the result: an operand with the result's layout
	 * is read at n, a broadcast operand at n % P (P: the static size of the trailing dimensions it covers).
	 * @details Schedules then need no index arithmetic: fusing and splitting the loops of a multi-dimensional result
	 * with symbolic dimensions (e.g. [batch, seq, 3072]) costs several 64-bit divisions per element, which the
	 * simplifier cannot remove. Kernels of other forms (reductions, transposes, gathers, padded or strided buffers) are
	 * left unchanged. Target-independent; applied before scheduling.
	 */
	::tvm::IRModule flattenInjective(::tvm::IRModule kernels, const ::tvm::Target& target);

} // namespace mdl::backend::tvm

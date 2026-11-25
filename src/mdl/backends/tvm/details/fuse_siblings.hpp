#pragma once
/**
 * @file fuse_siblings.hpp
 * @brief Horizontal fusion: independent elementwise kernels over the same elements that read a common input become one
 * kernel with several results.
 */
#include <tvm/ir/transform.h>

namespace mdl::backend::tvm {
	/**
	 * @brief Merges `call_tir`s of injective kernels (after FuseOps/FuseTIR) that produce results of the same shape, read
	 * at least one common tensor, and do not depend on each other, into one kernel that computes all their results in
	 * one loop: the common inputs are read once instead of once per kernel, and there is one launch.
	 * @details FuseOps fuses only along producer-consumer edges; siblings are typical of gradients, where the backward
	 * pass recomputes a forward value and its derivative from the same saved input (e.g. `gelu(h)` for a weight gradient
	 * and `gelu_backward(h, dy)`), or of several results computed from the same activations. Target-independent; the
	 * merged kernel is one block with one store per result, which the schedules and flattenInjective accept.
	 */
	::tvm::transform::Pass fuseSiblings();
} // namespace mdl::backend::tvm

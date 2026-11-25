#pragma once

/**
 * @file contexts.hpp
 * @brief Context buffers (see mdl::ContextBuffer) after kernel selection: the values a producer saves for its
 * consumers (e.g. the residuals of a forward pass for the backward pass) are those the consumers still read.
 */

#include "./aot.hpp"

#include <tvm/ir/module.h>

#include <string>
#include <unordered_map>

namespace mdl::backend::tvm {

	/**
	 * @brief Removes the packed tensors of context buffers that no consumer reads (anymore): the consumers' parameters
	 * and the producer's outputs, in @p mod and in @p packing (so that the layout of the buffer, which AOT lowering
	 * computes, only holds the others). The computations of the removed outputs become dead code.
	 * @details The frontend saves every value that some computation of the gradient may read; which one is read depends
	 * on the kernels selected for the backward pass (e.g. a fused attention gradient reads the attention's row
	 * statistics, the unfused operations its probabilities). A context whose consumer is not in @p mod is kept.
	 * @return whether a tensor was removed.
	 */
	bool pruneContexts(::tvm::IRModule& mod, std::unordered_map<std::string, EntrySignature>& packing);

	/**
	 * @brief Saves a context tensor in the layout its producer uses it in: if the producer returns a value v that it
	 * otherwise only permutes (p = permute(v)), and the consumers pass the tensor only to attention operations, the
	 * producer returns p and the consumers read permute⁻¹(p). Both permutations then fold into the operations (e.g. the
	 * attention writes its result in the layout of the output projection, and its gradient reads it through strides)
	 * instead of a transposition kernel. The tensor's size is unchanged (the buffer's layout too).
	 * @return whether a context tensor was changed (then fold the permutations: afterPruning).
	 */
	bool permuteContexts(::tvm::IRModule& mod, const std::unordered_map<std::string, EntrySignature>& packing);

} // namespace mdl::backend::tvm

#pragma once
/**
 * @file planner.hpp
 * @brief Which kernel calls of a function become one kernel (inspired by PyTorch Inductor's scheduler).
 */
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm::fusion {

	/** @brief Groups of bindings (by their variables), each computed by one kernel; other bindings stay as they are. */
	using Partition = std::vector<std::vector<::tvm::Var>>;

	/**
	 * @brief Groups the kernel calls of @p f's dataflow blocks: producers with their consumers and calls that read the
	 * same tensors, as long as one kernel computes the group (rules.hpp) and no group depends on itself through
	 * another. Greedily by the memory traffic a merge saves (intermediate results that are no longer written and read,
	 * inputs read once).
	 * @details Pure (it does not change @p f); kernels are those of @p mod (PrimFuncs called by call_tir).
	 */
	Partition plan(const ::tvm::IRModule& mod, const ::tvm::relax::Function& f);

	/** @brief The variables that the blocks after block @p k of @p seq and its result use. */
	std::unordered_set<const void*> usedAfter(const ::tvm::relax::SeqExprNode& seq, size_t k);

} // namespace mdl::backend::tvm::fusion

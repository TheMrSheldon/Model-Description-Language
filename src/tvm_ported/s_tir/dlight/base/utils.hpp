// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/utils.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file utils.hpp
 * @brief Utility methods for generic GPU.
 */

#include "../../../tvm.hpp"

namespace mdl::tvm_ported::dlight {

	/** @brief `s_tir.Schedule(func)`: a (traced) schedule of the module {"main": func} (not in TVM's utils.py). */
	Schedule schedule_of(const PrimFunc& func);

	int64_t get_bytes(const ::tvm::PrimType& dtype);

	/** @brief The extent of a loop: an int if constant (see as_int), else the expression. */
	PrimExpr get_extent(const Schedule& sch, const LoopRV& loop_rv);

	/** @brief Auto vectorize the loop. */
	void auto_vectorize(const Schedule& sch, const LoopRV& loop, int64_t max_vec);

	/** @brief Get the maximum number of threads per block for a given target. */
	int64_t max_threads_per_block(const Target& target);

	std::vector<int64_t> suggest_threads_per_block(
			const Target& target, const std::vector<tirx::For>& loops, int64_t max_threads_for_dynamic_loop = 32
	);

} // namespace mdl::tvm_ported::dlight

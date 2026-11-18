// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/common_schedules.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
#pragma once

/**
 * @file common_schedules.hpp
 * @brief Common schedule strategies for TIR.
 */

#include "../analysis/common_analysis.hpp"

namespace mdl::tvm_ported::dlight {

	/** @brief Try to inline as many blocks as possible, and return the remaining blocks. */
	std::vector<analysis::SBlockInfo> try_inline(const Schedule& sch, std::vector<analysis::SBlockInfo> blocks);

	/** @brief Try to inline contiguous spatial blocks in a schedule; returns the remaining blocks. */
	std::optional<std::vector<analysis::SBlockInfo>>
	try_inline_contiguous_spatial(const Schedule& sch, const std::optional<std::vector<analysis::SBlockInfo>>& block_infos);

} // namespace mdl::tvm_ported::dlight

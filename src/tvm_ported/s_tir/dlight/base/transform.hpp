// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/transform.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file transform.hpp
 * @brief Apply ScheduleRules onto an IRModule to generate default schedules without tuning.
 */

#include "./schedule_rule.hpp"

#include <tvm/ir/transform.h>

#include <memory>

namespace mdl::tvm_ported::dlight {

	/**
	 * @brief A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module: the first rule that
	 * applies schedules a function (marked `tirx.is_scheduled`); functions no rule applies to stay unchanged.
	 */
	::tvm::transform::Pass ApplyDefaultSchedule(std::vector<std::shared_ptr<const ScheduleRule>> rules);

} // namespace mdl::tvm_ported::dlight

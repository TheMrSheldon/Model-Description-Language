// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/cpu/base.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file base.hpp
 * @brief Base schedule rule for CPU operators.
 */

#include "../base/schedule_rule.hpp"

namespace mdl::tvm_ported::dlight::cpu {

	/** @brief The Schedule Rule specific to CPU targets, will return std::nullopt if the target is not CPU. */
	class CPUScheduleRule : public ScheduleRule {
	public:
		bool is_target_available(const Target& target) const override {
			return ScheduleRule::is_target_available(target) && target->kind->name == "llvm";
		}
	};

} // namespace mdl::tvm_ported::dlight::cpu

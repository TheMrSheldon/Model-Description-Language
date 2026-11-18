// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/schedule_rule.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file schedule_rule.hpp
 * @brief A lightweight wrapper on an arbitrary function that can be used to schedule a TIR PrimFunc.
 */

#include "../../../tvm.hpp"

namespace mdl::tvm_ported::dlight {

	/**
	 * @brief A thin wrapper on an arbitrary function that can be used to schedule a TIR PrimFunc.
	 * @details Given a PrimFunc, a target, and a tunable flag, apply returns a Schedule, or std::nullopt if the rule is
	 * not applicable to the PrimFunc. (TVM may also return several schedules when tunable; the port is not tunable.)
	 */
	class ScheduleRule {
	public:
		virtual ~ScheduleRule() = default;

		/** @brief Apply the ScheduleRule to the given PrimFunc. */
		virtual std::optional<Schedule> apply(const PrimFunc& func, const Target& target, bool tunable) const = 0;

		/** @brief Check whether the rule is available for the given target. */
		virtual bool is_target_available(const Target& /*target*/) const { return true; }
	};

} // namespace mdl::tvm_ported::dlight

#pragma once

/**
 * @file base.hpp
 * @brief Base schedule rule for GPU operators.
 * Port of https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/base.py
 * (see ../README.md).
 */

#include "../base/schedule_rule.hpp"

namespace mdl::dlight::gpu {

	/** @brief The Schedule Rule specific to GPU targets, will return std::nullopt if the target is not GPU. */
	class GPUScheduleRule : public ScheduleRule {
	public:
		bool is_target_available(const Target& target) const override {
			if (!ScheduleRule::is_target_available(target))
				return false;
			for (const auto& key : target->keys)
				if (key == "gpu")
					return true;
			return false;
		}
	};

} // namespace mdl::dlight::gpu

// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/cpu/__init__.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file cpu.hpp
 * @brief CPU-generic schedule rules.
 */

#include "./base.hpp"

namespace mdl::tvm_ported::dlight::cpu {

	/** @brief cpu/reduction.py: CPU reduction rule for softmax, layer norm, RMS norm, and similar operators. */
	class Reduction final : public CPUScheduleRule {
	public:
		std::optional<Schedule> apply(const PrimFunc& func, const Target& target, bool tunable) const override;
	};

} // namespace mdl::tvm_ported::dlight::cpu

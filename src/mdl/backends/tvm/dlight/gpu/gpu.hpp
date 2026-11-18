#pragma once

/**
 * @file gpu.hpp
 * @brief GPU-generic schedule rules.
 * Port of https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/__init__.py
 * (see ../README.md; GEMV, LowBatchGEMV and RMSNorm are not ported yet).
 */

#include "./base.hpp"

namespace mdl::dlight::gpu {

#define MDL_DLIGHT_RULE(Name)                                                                                          \
	class Name final : public GPUScheduleRule {                                                                        \
	public:                                                                                                            \
		std::optional<Schedule> apply(const PrimFunc& func, const Target& target, bool tunable) const override;       \
	};

	/** @brief gpu/fallback.py: a fallback schedule rule for all GPU operators. */
	MDL_DLIGHT_RULE(Fallback)
	/** @brief gpu/general_reduction.py: general reduction rule for operators including softmax, layer norm, RMS norm. */
	MDL_DLIGHT_RULE(GeneralReduction)
	/** @brief gpu/reduction.py: a rule for reductions (`Out[...] += In[...]`, with an optional elementwise epilogue). */
	MDL_DLIGHT_RULE(Reduction)
	/** @brief gpu/transpose.py: schedule rule for transpose. */
	MDL_DLIGHT_RULE(Transpose)
	/** @brief gpu/matmul.py: the schedule rule for matmul-like computation (without tensorization). */
	MDL_DLIGHT_RULE(Matmul)

#undef MDL_DLIGHT_RULE

} // namespace mdl::dlight::gpu

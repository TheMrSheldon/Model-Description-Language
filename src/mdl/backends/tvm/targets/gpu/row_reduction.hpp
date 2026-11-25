#pragma once

/**
 * @file row_reduction.hpp
 * @brief A GPU schedule rule for reductions over rows: a warp per row.
 */

#include <tvm_ported/s_tir/dlight/analysis/common_analysis.hpp>
#include <tvm_ported/s_tir/dlight/gpu/base.hpp>

#include <vector>

namespace mdl::backend::tvm::gpu {

	/**
	 * @brief Reductions over the innermost dimensions (sums, maxima; softmax and normalizations: reductions followed by
	 * an elementwise epilogue over the same rows) with a warp per row: the 32 threads of a warp reduce a row with warp
	 * shuffles; for a single reduction, kRowsPerBlock rows share a block (several reductions: one row per block).
	 * @details For rows of a symbolic length (e.g. the sequence length of attention) or of at most kMaxRowLength
	 * elements, and not too few rows (kMinRows). DLight's rules choose their threads by the static row length: for a
	 * symbolic one, the maximum (1024 threads for a row of perhaps 128 elements). Our rule, not a port of TVM's
	 * (derived from DLight's GeneralReduction); applied before DLight's reduction rules.
	 */
	class RowReduction final : public mdl::tvm_ported::dlight::gpu::GPUScheduleRule {
	public:
		static constexpr int64_t kWarp = 32;
		static constexpr int64_t kRowsPerBlock = 4;
		static constexpr int64_t kMaxRowLength = 4096;
		static constexpr int64_t kMinRows = 64;

		std::optional<::tvm::s_tir::Schedule>
		apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const override;

	private:
		static void schedule(
				const ::tvm::s_tir::Schedule& sch,
				const std::vector<mdl::tvm_ported::dlight::analysis::SBlockInfo>& infos, size_t numLeadingS,
				size_t numTrailingR, size_t numReductions
		);
	};

} // namespace mdl::backend::tvm::gpu

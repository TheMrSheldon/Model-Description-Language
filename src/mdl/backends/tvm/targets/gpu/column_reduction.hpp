#pragma once

/**
 * @file column_reduction.hpp
 * @brief A GPU schedule rule for reductions over columns: sums over the leading dimensions of a tensor whose last
 * dimension is contiguous (e.g. the gradients of biases and of normalization parameters).
 */

#include <tvm_ported/s_tir/dlight/analysis/common_analysis.hpp>
#include <tvm_ported/s_tir/dlight/gpu/base.hpp>

namespace mdl::backend::tvm::gpu {

	/**
	 * @brief Reductions over the leading dimensions of their inputs, whose innermost (contiguous) dimension is not
	 * reduced, e.g. `out[n] = sum over m of x[m, n]` (several over the same columns, e.g. the gradients of a
	 * normalization's bias and scale, which read dy once), optionally followed by elementwise epilogues per column (e.g.
	 * a conversion, or the accumulation into a gradient): in two passes, the reductions' loops merged. The rows are split into chunks of kChunk (rfactor): the partial sums
	 * [chunks, columns] have a thread per (chunk, column), with the threads of a warp on adjacent columns (coalesced
	 * reads), and each thread sums its chunk; a second kernel sums the chunks per column in a register of the
	 * reduction's type (e.g. float32) and applies the epilogue.
	 * @details DLight's reduction rules expect the reduced dimension innermost and read such columns uncoalesced, with
	 * too few blocks. Our rule, not a port of TVM's; applied after RowReduction, before DLight's reduction rules.
	 */
	class ColumnReduction final : public mdl::tvm_ported::dlight::gpu::GPUScheduleRule {
	public:
		static constexpr int64_t kWarp = 32;
		static constexpr int64_t kChunk = 64;		  ///< rows per partial sum
		static constexpr int64_t kChunksPerBlock = 8; ///< threadIdx.y of the first pass
		static constexpr int64_t kVector = 4;		  ///< adjacent columns per thread of the first pass (vector loads)
		static constexpr int64_t kThreads = 32;		  ///< threads of the second pass (few: blocks over many SMs)
		static constexpr int64_t kUnroll = 32;		  ///< chunks per iteration of the second pass
		static constexpr int64_t kMinRows = 256;	  ///< (fewer: DLight's rules)
		static constexpr size_t kMaxReductions = 8;	  ///< in one kernel

		std::optional<::tvm::s_tir::Schedule>
		apply(const ::tvm::tirx::PrimFunc& func, const ::tvm::Target& target, bool tunable) const override;
	};

} // namespace mdl::backend::tvm::gpu

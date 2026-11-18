// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/analysis/common_analysis.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
#pragma once

/**
 * @file common_analysis.hpp
 * @brief Analysis on TIR blocks, loops and functions.
 */

#include "../../../tvm.hpp"

#include <string>
#include <unordered_set>

namespace mdl::tvm_ported::dlight::analysis {

	/** @brief Information about a loop/iter var. */
	struct IterInfo {
		char kind; ///< 'S', 'R' or 'O'
		::tvm::Var var;
		PrimExpr dom;
		LoopRV loop_rv;
	};

	/** @brief Information about a TIR block. */
	struct SBlockInfo {
		std::string name;
		std::vector<IterInfo> iters;
		SBlockRV block_rv;
		bool reduction_block = false;

		/** @brief The iteration domain of the block. */
		std::vector<PrimExpr> dom() const;
		/** @brief The iteration domain kind of the block, for example, SSSS, SSSR. */
		std::string dom_kind() const;
		/** @brief Whether the block is injective, i.e. all its iteration domains are injective. */
		bool is_injective() const;
		/** @brief Whether the SBlock is elementwise, i.e. trivial mapping between read/write region. */
		bool is_elementwise(const Schedule& sch) const;
		/** @brief Get the loops of the block. */
		std::vector<LoopRV> get_loops() const;
		/** @brief Whether the block is a reduction workload. */
		bool is_reduction() const { return reduction_block; }
	};

	/** @brief Normalize the primfunc to normal form; std::nullopt if it is not normalizable. */
	std::optional<std::vector<SBlockInfo>> normalize_prim_func(const Schedule& sch);

	SBlockInfo get_sblock_info(const Schedule& sch, const SBlockRV& block);

	/** @brief Get the maximum number of threads per block for a given target. */
	int64_t get_max_threads_per_block(const Target& target);

	/** @brief Get the maximum shared memory per block for a given target. */
	int64_t get_max_shared_memory_per_block(const Target& target);

	SBlockRV get_root_block(const Schedule& sch, const std::string& func_name = "main");

	/** @brief Variables (by node) as Python's `set[tirx.Var]`. */
	using VarSet = std::unordered_set<const void*>;

	/** @brief Collect the block iter variables used in the access region of a buffer region. */
	VarSet collect_block_iter_vars_used_in_access_region(const s_tir::SBlock& block, const Array<Range>& region);

	/** @brief Collect the variables used in the PrimExpr. */
	VarSet collect_vars_used_in_prim_expr(const PrimExpr& expr);

	/** @brief Detect the dominant read indices in the block. */
	PrimExpr detect_dominant_read(const s_tir::SBlock& block);

	/** @brief Check if the epilogue block is a broadcast pattern. */
	bool is_broadcast_epilogue(const Schedule& sch, const SBlockRV& block, const SBlockRV& epilogue);

} // namespace mdl::tvm_ported::dlight::analysis

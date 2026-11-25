#pragma once
/**
 * @file rules.hpp
 * @brief Which groups of kernels one kernel can compute: the contract between the fusion planner and the targets'
 * schedule rules.
 * @details A group is one of the kinds of Signature, with the group's domains:
 * - pointwise over D: every result has the domain D;
 * - reductions over the same rows (or columns), several of them, with pointwise computations before them and (rows)
 *   after them: results per row or of the full domain (rows), or per column (columns);
 * - an anchor (one) with pointwise computations after it: results of the anchor's domain.
 * Reshapes are free views on their own: a group computes something else too, and a reshape is not one of its results. Members whose results are only
 * used inside the group may have any domain (they are inlined: broadcasts, per-row
 * values). The targets schedule each kind (e.g. a warp per row on GPUs); limits of a target (row lengths, threads) are
 * not part of the contract: a rule declines, and the target's fallback schedules the group.
 */
#include "./signature.hpp"

#include <optional>
#include <vector>

namespace mdl::backend::tvm::fusion {

	/** @brief A member of a candidate group. */
	struct Member {
		const Signature* signature;
		bool escapes;		 ///< its result is used outside the group (a result of the group's kernel)
		bool afterReduction; ///< it depends on a reduction (or the anchor) of the group
	};

	/** @brief The group's signature, if one kernel computes the members; else std::nullopt. */
	std::optional<Signature> groupSignature(const std::vector<Member>& members);

} // namespace mdl::backend::tvm::fusion

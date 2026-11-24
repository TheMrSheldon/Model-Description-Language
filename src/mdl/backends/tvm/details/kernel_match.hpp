#pragma once

/**
 * @file kernel_match.hpp
 * @brief Matching kernel packages against the Relax module at the matching boundary (see docs/dev/spec/kernel-packages.md): candidate
 * regions, the verdicts of their constraints, and the selection of non-overlapping implementations.
 */

#include <mdl/kernels.hpp>

#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>

#include <ostream>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	/** @brief The verdict of one constraint of a candidate. */
	struct Verdict {
		enum class Result { proven, refuted, unknown };
		/**
		 * @brief How an unknown verdict could be resolved at runtime: shape arithmetic over symbolic dimensions is
		 * checked by a runtime guard; numerics, artifacts, element types cannot be.
		 */
		enum class Class { notGuardable, shapeArithmetic, pointer };
		std::string what; ///< e.g. `where $x.shape[-1] % 16 == 0`
		Result result;
		std::string detail; ///< e.g. the simplified condition, or why it is unknown
		Class guardClass = Class::notGuardable;
		::tvm::PrimExpr guard; ///< the condition to check at runtime (shape arithmetic only)
	};

	/** @brief An implementation that matches at one place of a function. */
	struct Candidate {
		const kernels::Package* package;
		const kernels::Implementation* implementation;
		std::string function;			  ///< the Relax function
		std::vector<::tvm::Var> replaced; ///< the bindings the region replaces (root last)
		std::vector<std::pair<std::string, ::tvm::Expr>> captures;
		std::vector<std::pair<std::string, ::tvm::Expr>> calls; ///< `out` and the captured operators (attributes)
		std::string artifact;									///< the artifact that would be used (empty: none)
		std::vector<Verdict> verdicts;
		bool eligible = false;
		bool guarded = false; ///< eligible only if its runtime guard holds (the original region is the fallback)
		enum class Outcome { selected, overlapped, ineligible } outcome = Outcome::ineligible;
		std::string reason; ///< why it is ineligible or overlapped
	};

	/** @brief A problem of a pattern (e.g. an unknown operator), reported once per implementation. */
	struct PatternProblem {
		const kernels::Package* package;
		std::string pattern;
		std::string message;
	};

	struct MatchResult {
		std::vector<Candidate> candidates; ///< in the order of the functions and their bindings
		std::vector<PatternProblem> problems;
	};

	/**
	 * @brief All candidates of the packages' implementations in @p mod (collected before anything is selected), with
	 * their verdicts, then the selection: eligible candidates by priority, then the number of replaced bindings, then
	 * package and implementation names; a candidate is selected unless it shares a replaced binding with a selected one.
	 */
	MatchResult matchKernels(
			const ::tvm::IRModule& mod, const std::vector<kernels::Package>& packages, const kernels::Platform& target
	);

	/**
	 * @brief Replaces every selected region of @p result by a call of its implementation (a PrimFunc that `call_extern`s
	 * the symbol with the evaluated arguments, through `call_tir`); the region's other bindings become dead code. A
	 * guarded region becomes `if guard: kernel else: original region`; @p mod must not have dataflow blocks then (see
	 * needsNonDataflow).
	 */
	::tvm::IRModule applyKernels(::tvm::IRModule mod, const MatchResult& result);

	/**
	 * @brief Writes the result of matching (`mdl-kernels check`): the selected implementations, and with @p verbose every
	 * candidate with the verdicts of its constraints (`--explain`).
	 */
	/** @brief The attribute of the fallback functions of runtime guards (private Relax functions). */
	inline constexpr const char* kGuardFallbackAttr = "mdl.guard_fallback";

	/**
	 * @brief Inlines the fallbacks of runtime guards into their `If` branches (after fusion: FuseOps does not look into
	 * branches, so the fallbacks are fused as functions of their own).
	 */
	::tvm::transform::Pass inlineGuardFallbacks();

	/** @brief Whether applyKernels emits a runtime guard (an `If`, which cannot be inside a dataflow block). */
	bool needsNonDataflow(const MatchResult& result);

	void explain(std::ostream& out, const MatchResult& result, const kernels::Discovery& discovery, bool verbose);

} // namespace mdl::backend::tvm

#pragma once

/**
 * @file kernel_match.hpp
 * @brief Matching kernel packages against the Relax module at the matching boundary (see docs/kernels.md): candidate
 * regions, the verdicts of their constraints, and the selection of non-overlapping implementations.
 */

#include <mdl/kernels.hpp>

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include <ostream>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	/** @brief The verdict of one constraint of a candidate. */
	struct Verdict {
		enum class Result { proven, refuted, unknown };
		/**
		 * @brief How an unknown verdict could be resolved at runtime (runtime guards come in a later phase): shape
		 * arithmetic over symbolic dimensions can be checked; numerics, artifacts, element types cannot.
		 */
		enum class Class { notGuardable, shapeArithmetic, pointer };
		std::string what; ///< e.g. `where $x.shape[-1] % 16 == 0`
		Result result;
		std::string detail; ///< e.g. the simplified condition, or why it is unknown
		Class guardClass = Class::notGuardable;
	};

	/** @brief An implementation that matches at one place of a function. */
	struct Candidate {
		const kernels::Package* package;
		const kernels::Implementation* implementation;
		std::string function;			  ///< the Relax function
		std::vector<::tvm::Var> replaced; ///< the bindings the region replaces (root last)
		std::vector<std::pair<std::string, ::tvm::Expr>> captures;
		std::string artifact; ///< the artifact that would be used (empty: none)
		std::vector<Verdict> verdicts;
		bool eligible = false;
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
	 * @brief Writes the result of matching (`mdl-kernels check`): the selected implementations, and with @p verbose every
	 * candidate with the verdicts of its constraints (`--explain`).
	 */
	void explain(std::ostream& out, const MatchResult& result, const kernels::Discovery& discovery, bool verbose);

} // namespace mdl::backend::tvm

#pragma once

/**
 * @file pattern_registry.hpp
 * @brief Backend-owned pattern rewrites on Relax: recognize subgraphs that the backend can implement better (e.g., the
 * softmax decomposition of the standard library) and replace them by single operators.
 * @details Patterns are an optimization of this backend only; the computegraph has no notion of them. Every rewriter
 * verifies the semantics of the match (constants, reduction axes, shapes) and leaves the expression unchanged
 * otherwise, so a pattern can never change the result of a model.
 */

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>

#include <functional>
#include <string_view>
#include <vector>

namespace mdl::backend::tvm {

	/**
	 * @brief Rewriter callback type used by RewriteCall.
	 * @details Receives the matched call expression, a map from each DFPattern to the sub-expression it captured (often
	 * a variable that is bound to the matched call), and the bindings of the function to look through such variables.
	 * Returns the replacement expression. To veto a match (e.g. after a semantic check), return `call` unchanged.
	 */
	using RewriterFn = std::function<::tvm::relax::Expr(
			::tvm::relax::Expr, ::tvm::ffi::Map<::tvm::relax::DFPattern, ::tvm::relax::Expr>,
			const ::tvm::ffi::Map<::tvm::relax::Var, ::tvm::relax::Expr>& /* bindings of the function */
	)>;

	/** @brief One recognizable decomposition and its replacement. */
	struct PatternEntry {
		std::string_view name;			///< logged on match (diagnostic only)
		::tvm::relax::DFPattern pattern; ///< root DFPattern to match
		RewriterFn rewriter;			 ///< called on every successful match
	};

	/** @brief Applies the patterns (in order) to every Relax function of the module. */
	::tvm::IRModule applyPatterns(::tvm::IRModule mod, const std::vector<PatternEntry>& entries);

	/**
	 * @brief The patterns of the backend, in the order in which they must be applied.
	 * @details softmax, GELU, layer norm (→ relax.nn ops) and matmul variants (→ mdl.matmul with fused transposition,
	 * bias, and scale).
	 */
	std::vector<PatternEntry> defaultPatterns();

} // namespace mdl::backend::tvm

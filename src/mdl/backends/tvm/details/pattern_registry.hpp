#pragma once

/**
 * @file pattern_registry.hpp
 * @brief Relax dataflow-pattern registry for high-level op recognition.
 *
 * Usage
 * -----
 * 1. Define a factory function that returns a `PatternEntry` (see patterns/ sub-headers).
 * 2. Add the entry to `defaultPatterns()` below.
 * 3. Call `applyPatterns(mod, defaultPatterns())` before handing the IRModule to the
 *    Python compilation backend.
 *
 * Adding a new pattern
 * --------------------
 * Create `details/patterns/<name>.hpp` with a single `inline PatternEntry make<Name>()`.
 * Include it here and push `patterns::make<Name>()` into `defaultPatterns()`.
 * No other files need to change.
 */

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>

#include <string_view>
#include <vector>

namespace mdl::backend::tvm {

/**
 * @brief Rewriter callback type used by RewriteCall.
 *
 * Receives the matched call expression and a map from each DFPattern to the
 * sub-expression it captured.  Returns the replacement expression.  To veto a
 * match (e.g. after a semantic sanity check), return `call` unchanged.
 */
using RewriterFn = ::tvm::ffi::TypedFunction<
    ::tvm::relax::Expr(::tvm::relax::Expr, ::tvm::ffi::Map<::tvm::relax::DFPattern, ::tvm::relax::Expr>)>;

/**
 * @brief One recognisable primitive decomposition and its high-level replacement.
 */
struct PatternEntry {
    std::string_view name;              ///< logged on match (diagnostic only)
    ::tvm::relax::DFPattern pattern;    ///< root DFPattern to match
    RewriterFn                rewriter; ///< called on every successful match
};

/**
 * @brief Apply a list of patterns to every Relax Function in the module.
 *
 * Each entry is applied to every function independently.  Failures are caught
 * and logged rather than aborting compilation.
 *
 * @param mod     The IRModule to transform (value-type, returns modified copy).
 * @param entries Span of patterns to apply.
 * @return The (possibly rewritten) IRModule.
 */
::tvm::IRModule applyPatterns(::tvm::IRModule mod, const std::vector<PatternEntry>& entries);

} // namespace mdl::backend::tvm

// ── default pattern set ───────────────────────────────────────────────────────
// Include concrete pattern headers here and register them in defaultPatterns().

#include "./patterns/softmax.hpp"
#include "./patterns/gelu.hpp"
#include "./patterns/layer_norm.hpp"
#include "./patterns/attention.hpp"

namespace mdl::backend::tvm {

/**
 * @brief Return the default set of patterns applied to every compilation.
 *
 * Order matters: softmax must run before attention so that the softmax
 * decomposition inside SDPA is already replaced with relax.nn.softmax
 * when the attention pattern tries to match it.
 */
inline std::vector<PatternEntry> defaultPatterns() {
    return {
        patterns::makeSoftmax(),
        patterns::makeGelu(),
        patterns::makeLayerNorm(),
        // Attention rewrite is done in Python (_rewrite_attention_python) because
        // TVM's C++ RewriteCall always produces a new AST, making same_as() useless
        // as a change detector, and compound expressions from the rewriter are not
        // reliably inserted into ANF form.  Python dpl.rewrite_call handles this.
        // patterns::makeAttention(),
    };
}

} // namespace mdl::backend::tvm

#pragma once

/**
 * @file softmax.hpp
 * @brief Detects numerically-stable softmax decomposition and replaces it with
 *        relax.nn.softmax.
 *
 * Matched pattern (reduces over the last axis):
 *   power = exp(x - max(x, axis=-1, keepdims=True))
 *   out   = power / sum(power, axis=-1, keepdims=True)
 *
 * This is the canonical decomposition emitted for the MDL user-defined
 * `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
 * Replacing it with a single kernel avoids the redundant exp computation and
 * enables TVM to fuse the four sub-operations into one loop.
 */

#include "../pattern_registry.hpp"
#include "../relax_builtins.hpp"  // callRelaxOp

#include <tvm/relax/dataflow_pattern.h>

namespace mdl::backend::tvm::patterns {

inline PatternEntry makeSoftmax() {
    using namespace ::tvm::relax;

    // x — the input tensor (any shape/rank)
    auto x = Wildcard();

    // max(x, axis=-1, keepdims=True)
    // axis/keepdims are *attrs*, not positional args — match with 1 arg only.
    auto max_x  = IsOp("relax.max")(x);

    // x - max(x)  [numerically stabilised shift]
    auto shifted = IsOp("relax.subtract")(x, max_x);

    // exp(x - max(x))
    auto exp_val = IsOp("relax.exp")(shifted);

    // sum(exp_val, axis=-1, keepdims=True) — 1 positional arg
    auto sum_exp = IsOp("relax.sum")(exp_val);

    // exp_val / sum_exp
    DFPattern pat = IsOp("relax.divide")(exp_val, sum_exp);

    RewriterFn rewriter = [x](Expr call, ::tvm::ffi::Map<DFPattern, Expr> m) -> Expr {
        return callRelaxOp("relax.op.nn.softmax", m.at(x),
                           ::tvm::Integer(-1));
    };

    return {"softmax", pat, std::move(rewriter)};
}

} // namespace mdl::backend::tvm::patterns

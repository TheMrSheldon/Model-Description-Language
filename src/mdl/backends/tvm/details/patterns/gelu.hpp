#pragma once

/**
 * @file gelu.hpp
 * @brief Detects the BERT-style GELU decomposition and replaces it with
 *        relax.nn.gelu.
 *
 * Matched pattern:
 *   out = (x * 0.5) * (1 + erf(x / sqrt(2)))
 *
 * The multiplicative constants (0.5, sqrt(2)) and the addend (1) are matched
 * as wildcards so they are not sensitive to the exact constant representation.
 *
 * relax.nn.gelu maps to a fused LLVM vectorised kernel that computes erf via
 * a Chebyshev polynomial approximation, faster than 3 separate element-wise
 * kernels.
 */

#include "../pattern_registry.hpp"
#include "../relax_builtins.hpp"  // callRelaxOp

#include <tvm/relax/dataflow_pattern.h>

namespace mdl::backend::tvm::patterns {

inline PatternEntry makeGelu() {
    using namespace ::tvm::relax;

    auto x = Wildcard();

    // erf(x / sqrt(2))  — divisor is any constant
    auto erf_arg = IsOp("relax.divide")(x, Wildcard());
    auto erf_val = IsOp("relax.erf")(erf_arg);

    // 1 + erf(...)  — the addend (1) is any constant
    auto add1 = IsOp("relax.add")(Wildcard(), erf_val);

    // x * 0.5  — factor (0.5) is any constant
    auto x_half = IsOp("relax.multiply")(x, Wildcard());

    // (x * 0.5) * (1 + erf(...))
    DFPattern pat = IsOp("relax.multiply")(x_half, add1);

    RewriterFn rewriter = [x](Expr call, ::tvm::ffi::Map<DFPattern, Expr> m) -> Expr {
        return callRelaxOp("relax.op.nn.gelu", m.at(x));
    };

    return {"gelu", pat, std::move(rewriter)};
}

} // namespace mdl::backend::tvm::patterns

#pragma once

/**
 * @file layer_norm.hpp
 * @brief Detects the mean/variance/rsqrt layer-normalisation decomposition and
 *        replaces it with relax.nn.layer_norm.
 *
 * Matched pattern (normalising over the last axis, epsilon = any constant):
 *
 *   mean_x    = reshape(sum(x) / N)                       [keepdims form]
 *   x_c       = x - mean_x
 *   mean_xc2  = reshape(sum(x_c * x_c) / N)              [variance]
 *   sigma     = rsqrt(mean_xc2 + eps)
 *   norm      = x_c * sigma
 *   out       = norm * gamma + beta
 *
 * The reshape after sum() is the keepdims-emulation produced by MDL's MEAN
 * builtin which emits sum/N (keepdims=false) + explicit reshape.
 *
 * Note: epsilon is matched as a wildcard, so the same pattern works regardless
 * of the exact epsilon value (1e-12 for BERT, 1e-5 for other architectures).
 * The replacement always uses 1e-12; if per-instance epsilon is needed, extend
 * the rewriter to extract it from the matched add-constant.
 */

#include "../pattern_registry.hpp"
#include "../relax_builtins.hpp"  // callRelaxOp

#include <tvm/relax/dataflow_pattern.h>

namespace mdl::backend::tvm::patterns {

inline PatternEntry makeLayerNorm() {
    using namespace ::tvm::relax;

    auto x     = Wildcard();
    auto gamma = Wildcard();
    auto beta  = Wildcard();

    // mean(x) = sum(x) / N  then reshape to keepdims shape
    // axis/keepdims are *attrs* not positional args — 1 arg only.
    auto sum_x     = IsOp("relax.sum")(x);
    auto mean_x    = IsOp("relax.divide")(sum_x, Wildcard());
    auto mean_x_r  = IsOp("relax.reshape")(mean_x, Wildcard());

    // x - mean(x)
    auto x_c = IsOp("relax.subtract")(x, mean_x_r);

    // variance: sum(x_c²) / N  then reshape to keepdims shape
    auto xc_sq      = IsOp("relax.multiply")(x_c, x_c);
    auto sum_xcsq   = IsOp("relax.sum")(xc_sq);
    auto mean_xcsq  = IsOp("relax.divide")(sum_xcsq, Wildcard());
    auto mean_xcsq_r = IsOp("relax.reshape")(mean_xcsq, Wildcard());

    // rsqrt(variance + eps)
    auto var_eps = IsOp("relax.add")(mean_xcsq_r, Wildcard());
    auto sigma   = IsOp("relax.rsqrt")(var_eps);

    // (x - mean) * sigma * gamma + beta
    auto norm   = IsOp("relax.multiply")(x_c, sigma);
    auto scaled = IsOp("relax.multiply")(norm, gamma);
    DFPattern pat = IsOp("relax.add")(scaled, beta);

    RewriterFn rewriter = [x, gamma, beta](Expr call, ::tvm::ffi::Map<DFPattern, Expr> m) -> Expr {
        // axes=[-1], epsilon=1e-12, center=true, scale=true
        ::tvm::ffi::Array<::tvm::Integer> axes{::tvm::Integer(-1)};
        double eps = 1e-12;
        return callRelaxOp("relax.op.nn.layer_norm",
                           m.at(x), m.at(gamma), m.at(beta),
                           axes, eps, true, true);
    };

    return {"layer_norm", pat, std::move(rewriter)};
}

} // namespace mdl::backend::tvm::patterns

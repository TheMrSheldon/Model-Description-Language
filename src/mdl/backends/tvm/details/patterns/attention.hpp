#pragma once

/**
 * @file attention.hpp
 * @brief Detects the scaled dot-product attention (SDPA) decomposition and
 *        replaces it with relax.nn.attention for TVM's fused BSNH kernel.
 *
 * Matched pattern — produced by MDL's transformer attention:
 *
 *   Q_proj [S, H*D]  →  reshape [S,H,D]  →  permute_dims [H,S,D]  ─── Q_heads
 *   K_proj [S, H*D]  →  reshape [S,H,D]  →  permute_dims [H,S,D]  ─── K_heads
 *                                              permute_dims [H,D,S]  ─── K_T
 *   V_proj [S, H*D]  →  reshape [S,H,D]  →  permute_dims [H,S,D]  ─── V_heads
 *
 *   scores    = matmul(Q_heads, K_T)            [H, S, S]
 *   scaled    = divide(scores, sqrt_scale)       [H, S, S]
 *   attn      = relax.nn.softmax(scaled)         [H, S, S]
 *   ctx       = matmul(attn, V_heads)            [H, S, D]
 *   ctx_perm  = permute_dims(ctx, [1,0,2])       [S, H, D]
 *   out       = reshape(ctx_perm, [S, H*D])      ← TOP-LEVEL MATCH
 *
 * Replacement:
 *   Q_bsnh = reshape(Q_proj, [1, S, H, D])       no-copy BSNH view
 *   K_bsnh = reshape(K_proj, [1, S, H, D])
 *   V_bsnh = reshape(V_proj, [1, S, H, D])
 *   attn_bsnh = relax.nn.attention(Q_bsnh, K_bsnh, V_bsnh)   [1, S, H, D]
 *   out    = reshape(attn_bsnh, [S, H*D])
 *
 * TVM legalizes relax.nn.attention to a single TIR PrimFunc (_te_attention) that
 * contains the transpose, batched matmul, scale, softmax, and second batched
 * matmul — all in one fused kernel.  This eliminates:
 *   - 4 explicit head-transpose kernels (~500 µs at SEQ=512)
 *   - 3 separate kernel launches (2× batched matmul, 1× softmax)
 *
 * The scale passed to relax.nn.attention is None, so TVM uses the default
 * 1/sqrt(head_dim), which matches standard transformer scaling.
 */

#include "../pattern_registry.hpp"
#include "../relax_builtins.hpp"

#include <tvm/relax/dataflow_pattern.h>

namespace mdl::backend::tvm::patterns {

inline PatternEntry makeAttention() {
    using namespace ::tvm::relax;

    // ── Wildcards for the linear projection outputs [S, H*D] ─────────────────
    auto q_proj = Wildcard();
    auto k_proj = Wildcard();
    auto v_proj = Wildcard();

    // ── Head split: reshape [S,H*D] → [S,H,D], then permute → [H,S,D] ───────
    auto q_heads = IsOp("relax.permute_dims")(IsOp("relax.reshape")(q_proj, Wildcard()));
    auto k_heads = IsOp("relax.permute_dims")(IsOp("relax.reshape")(k_proj, Wildcard()));
    auto v_heads = IsOp("relax.permute_dims")(IsOp("relax.reshape")(v_proj, Wildcard()));

    // ── Attention core ────────────────────────────────────────────────────────
    auto k_T    = IsOp("relax.permute_dims")(k_heads);     // [H,S,D] → [H,D,S]
    auto scores = IsOp("relax.matmul")(q_heads, k_T);      // [H,S,S]
    auto scaled = IsOp("relax.divide")(scores, Wildcard()); // 1/sqrt(dk)
    auto attn   = IsOp("relax.nn.softmax")(scaled);        // [H,S,S]
    auto ctx    = IsOp("relax.matmul")(attn, v_heads);     // [H,S,D]

    // ── Output reshape: permute [H,S,D]→[S,H,D], reshape →[S,H*D] ───────────
    DFPattern pat = IsOp("relax.reshape")(IsOp("relax.permute_dims")(ctx), Wildcard());

    RewriterFn rewriter = [q_proj, k_proj, v_proj, k_heads, v_heads](
        Expr call, ::tvm::ffi::Map<DFPattern, Expr> m) -> Expr {

        // Extract H, D from k_heads shape [H, S_k, D].
        // S is read separately from each projection to preserve the per-parameter
        // symbolic dims (q→B5, k→B6, v→B7); using the wrong S causes a TVM
        // struct-info mismatch when the symbolic dims have different names.
        auto kh_shape = tensorShape(m.at(k_heads));
        auto vh_shape = tensorShape(m.at(v_heads));
        if (kh_shape.size() != 3 || vh_shape.size() != 3) return call;
        ::tvm::PrimExpr H   = kh_shape[0];
        ::tvm::PrimExpr D_k = kh_shape[2];  // key/query head dim
        ::tvm::PrimExpr D_v = vh_shape[2];  // value head dim

        // Read each projection's own sequence-length dim.
        auto q_shape = tensorShape(m.at(q_proj));
        auto k_shape = tensorShape(m.at(k_proj));
        auto v_shape = tensorShape(m.at(v_proj));
        if (q_shape.size() < 1 || k_shape.size() < 1 || v_shape.size() < 1) return call;
        ::tvm::PrimExpr S_q = q_shape[0];
        ::tvm::PrimExpr S_k = k_shape[0];
        ::tvm::PrimExpr S_v = v_shape[0];

        // Reshape each projection from [S, H*D] to [1, S, H, D] (BSNH, batch=1).
        auto q_bsnh = callRelaxOp("relax.op.reshape", m.at(q_proj),
                                  ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>{::tvm::Integer(1), S_q, H, D_k}));
        auto k_bsnh = callRelaxOp("relax.op.reshape", m.at(k_proj),
                                  ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>{::tvm::Integer(1), S_k, H, D_k}));
        auto v_bsnh = callRelaxOp("relax.op.reshape", m.at(v_proj),
                                  ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>{::tvm::Integer(1), S_v, H, D_v}));

        // relax.nn.attention — scale=None → TVM uses default 1/sqrt(D_k)
        auto attn_out = callRelaxOp(
            "relax.op.nn.attention",
            q_bsnh, k_bsnh, v_bsnh,
            ::tvm::ffi::Optional<Expr>(),                   // bias   = None
            ::tvm::ffi::Optional<::tvm::FloatImm>(),        // scale  = None
            ::tvm::ffi::Optional<::tvm::ffi::String>(),     // causal_mask = None
            ::tvm::ffi::Optional<::tvm::IntImm>());         // window_size = None

        // Reshape output [1, S_q, H, D_v] → [S_q, H*D_v]
        ::tvm::ffi::Array<::tvm::PrimExpr> flat_dims{S_q, H * D_v};
        return callRelaxOp("relax.op.reshape", attn_out, ShapeExpr(flat_dims));
    };

    return {"attention", pat, std::move(rewriter)};
}

} // namespace mdl::backend::tvm::patterns

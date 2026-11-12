#pragma once

#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

using RelaxTranslateFunc = std::function<::tvm::relax::Expr(
    ::tvm::relax::BlockBuilder&, std::vector<::tvm::relax::Expr>)>;

// ── helpers ─────────────────────────────────────────────────────────────────

/** Call a registered Relax op via the global function registry. */
template <typename... Args>
static inline ::tvm::relax::Expr callRelaxOp(const char* name, Args&&... args) {
    auto fn = *::tvm::ffi::Function::GetGlobal(name);
    return fn(std::forward<Args>(args)...).template cast<::tvm::relax::Expr>();
}

/** Extract PrimExpr from a relax::PrimValue expression. */
static inline ::tvm::PrimExpr extractPrim(const ::tvm::relax::Expr& e) {
    const auto* pv = e.as<::tvm::relax::PrimValueNode>();
    assert(pv && "expected PrimValue for scalar shape dim");
    return pv->value;
}

/**
 * Get input tensor's shape as a vector of PrimExpr.
 * The tensor must have TensorStructInfo with a known ShapeExpr.
 */
static inline std::vector<::tvm::PrimExpr> tensorShape(const ::tvm::relax::Expr& e) {
    const auto* sinfo = e->struct_info_.as<::tvm::relax::TensorStructInfoNode>();
    if (!sinfo) return {};
    auto maybe = sinfo->GetShape();
    if (!maybe.defined()) return {};
    const auto& arr = maybe.value();
    return {arr.begin(), arr.end()};
}

/** Get tensor ndim (or -1 if unknown). */
static inline int tensorNdim(const ::tvm::relax::Expr& e) {
    const auto* sinfo = e->struct_info_.as<::tvm::relax::TensorStructInfoNode>();
    if (!sinfo || sinfo->IsUnknownNdim()) return -1;
    return sinfo->ndim;
}

/** Reduce over last axis. keepdims controls whether axis is retained. */
static inline ::tvm::relax::Expr reduceLastAxis(
    const char* opName, const ::tvm::relax::Expr& x, bool keepdims)
{
    ::tvm::ffi::Array<::tvm::Integer> axis_arr{::tvm::Integer(-1)};
    ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>> axis = axis_arr;
    return callRelaxOp(opName, x, axis, keepdims);
}

/** Create a 0-D float32 Constant from a PrimExpr (must be IntImm or FloatImm). */
static inline ::tvm::relax::Expr makeFloatConst(float val) {
    DLDevice dev{kDLCPU, 0};
    auto data = ::tvm::runtime::Tensor::Empty(
        ::tvm::ffi::Shape{}, ::tvm::DataType::Float(32), dev);
    data.CopyFromBytes(&val, sizeof(float));
    return ::tvm::relax::Constant(data);
}

/** The size of the last dimension as a float divisor (a constant if static, else the symbolic size). */
static inline ::tvm::relax::Expr lastDimDivisor(const ::tvm::relax::Expr& x) {
    auto shape = tensorShape(x);
    assert(!shape.empty() && "reduction over a tensor without dimensions");
    if (const auto* imm = shape.back().template as<::tvm::IntImmNode>())
        return makeFloatConst(static_cast<float>(imm->value));
    return ::tvm::relax::PrimValue(::tvm::cast(::tvm::DataType::Float(32), shape.back()));
}

/** Emit a value into the BlockBuilder and return the bound Var. */
static inline ::tvm::relax::Var emitAs(
    ::tvm::relax::BlockBuilder& bb,
    ::tvm::relax::Expr e,
    const char* hint = "v")
{
    return bb->Emit(std::move(e), hint);
}

/**
 * The axes of a PERMUTE permutation (see mdl::builtins::FunctionIDs::PERMUTE) for a tensor of the given rank, or an
 * empty array if the permutation is invalid.
 */
static inline ::tvm::ffi::Array<::tvm::Integer> permuteAxes(int rank, const ::tvm::relax::ShapeExprNode* se) {
    int permSize = static_cast<int>(se->values.size());
    std::vector<int64_t> perm;
    for (const auto& v : se->values)
        perm.push_back(v.template as<::tvm::IntImmNode>()->value);

    ::tvm::ffi::Array<::tvm::Integer> axes;
    const bool hasEllipsis =
        std::find(perm.begin(), perm.end(), mdl::builtins::kEllipsisPermMarker) != perm.end();
    if (hasEllipsis) {
        // The ellipsis group starts after the dims listed with non-negative indices and covers all dims that
        // are not listed.
        const int64_t start = std::count_if(perm.begin(), perm.end(), [](int64_t p) { return p >= 0; });
        const int64_t groupRank = rank - (permSize - 1);
        for (int64_t p : perm) {
            if (p == mdl::builtins::kEllipsisPermMarker) {
                for (int64_t d = 0; d < groupRank; ++d)
                    axes.push_back(::tvm::Integer(start + d));
            } else {
                axes.push_back(::tvm::Integer(p < 0 ? p + rank : p));
            }
        }
    } else {
        // A shorter permutation applies to the trailing dims.
        int offset = (rank >= 0) ? (rank - permSize) : 0;
        for (int i = 0; i < offset; ++i)
            axes.push_back(::tvm::Integer(i));
        for (int64_t p : perm)
            axes.push_back(::tvm::Integer(p < 0 && rank > 0 ? p + rank : p));
    }
    // Validate: must be a permutation of all dims.
    if (rank >= 0 && static_cast<int>(axes.size()) != rank)
        return {};
    std::vector<bool> seen(std::max(rank, 0), false);
    for (const auto& ax : axes) {
        int a = static_cast<int>(ax->value);
        if (a < 0 || a >= rank || seen[a])
            return {};
        seen[a] = true;
    }
    return axes;
}

// ── op dispatch table ────────────────────────────────────────────────────────

// The size is deduced from the initializers so that the static_assert below catches missing translations.
inline const auto relaxFunctions = std::to_array<RelaxTranslateFunc>({
    /** TENSOR_ADD **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return callRelaxOp("relax.op.add", ts[0], ts[1]);
    },
    /** TENSOR_SUB **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return callRelaxOp("relax.op.subtract", ts[0], ts[1]);
    },
    /** TENSOR_DOT **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        // relax.matmul has NumPy semantics (batched, broadcasting, 1-D operands as vectors).
        ::tvm::ffi::Optional<::tvm::DataType> out_dtype;
        return callRelaxOp("relax.op.matmul", ts[0], ts[1], out_dtype);
    },
    /** TENSOR_PROD **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return callRelaxOp("relax.op.multiply", ts[0], ts[1]);
    },
    /** TENSOR_DIV **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return callRelaxOp("relax.op.divide", ts[0], ts[1]);
    },

    // ── statistical ─────────────────────────────────────────────────────────
    /** MAX — keepdims=true for stable softmax broadcasting **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return reduceLastAxis("relax.op.max", ts[0], /*keepdims=*/true);
    },
    /** MIN **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return reduceLastAxis("relax.op.min", ts[0], /*keepdims=*/true);
    },
    /** MEAN — implemented as sum/N to avoid TVM's symbolic intermediate reduction buffer
     *  (relax.op.mean generates a B1-sized intermediate that LLVM codegen can't stack-alloc). **/
    [](auto& bb, auto ts) -> ::tvm::relax::Expr {
        const auto& x = ts[0];
        auto s = emitAs(bb, reduceLastAxis("relax.op.sum", x, /*keepdims=*/false), "sum");
        return callRelaxOp("relax.op.divide", s, lastDimDivisor(x));
    },
    /** SUM — keepdims=true **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return reduceLastAxis("relax.op.sum", ts[0], /*keepdims=*/true);
    },

    // ── unary arithmetic ────────────────────────────────────────────────────
    /** COS **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.cos", ts[0]); },
    /** SIN **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.sin", ts[0]); },
    /** TAN **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.tan", ts[0]); },
    /** EXP **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.exp", ts[0]); },
    /** ERF **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.erf", ts[0]); },
    /** SQRT **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.sqrt", ts[0]); },
    /** RSQRT **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.rsqrt", ts[0]); },
    /** LOG **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.log", ts[0]); },
    /** TANH **/[](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.tanh", ts[0]); },

    // ── VAR = E[x²] − E[x]² ─────────────────────────────────────────────────
    /** VAR — uses sum/N instead of relax.op.mean to avoid symbolic intermediate buffer **/
    [](auto& bb, auto ts) -> ::tvm::relax::Expr {
        const auto& x = ts[0];
        auto n_const = lastDimDivisor(x);

        auto x2  = emitAs(bb, callRelaxOp("relax.op.multiply", x, x), "sq");
        auto sx  = emitAs(bb, reduceLastAxis("relax.op.sum", x,  /*keepdims=*/false), "sum");
        auto sx2 = emitAs(bb, reduceLastAxis("relax.op.sum", x2, /*keepdims=*/false), "sum_sq");
        auto ex  = emitAs(bb, callRelaxOp("relax.op.divide", sx,  n_const), "mean");
        auto ex2 = emitAs(bb, callRelaxOp("relax.op.divide", sx2, n_const), "mean_sq");
        auto ex_sq = emitAs(bb, callRelaxOp("relax.op.multiply", ex, ex), "mean2");
        return callRelaxOp("relax.op.subtract", ex2, ex_sq);
    },

    // ── einops ──────────────────────────────────────────────────────────────
    /**
     * RESHAPE — ts[0] tensor, ts[1..n] shape dims as PrimValues.
     *
     * Sentinels (from the MDL compiler):
     *   -1  = passthrough: copy the next leading dim of the input tensor.
     *   -2  = infer: let TVM's reshape compute this as (total / others).
     *         TVM uses -1 as its own "infer" sentinel.
     */
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        const auto& tensor = ts[0];
        auto inShape = tensorShape(tensor);

        ::tvm::ffi::Array<::tvm::PrimExpr> outDims;
        outDims.reserve(ts.size() - 1);

        int passthroughIdx = 0;
        for (size_t i = 1; i < ts.size(); ++i) {
            // Ellipsis group {before, after}: the input dims [before, rank - after).
            if (const auto* group = ts[i].template as<::tvm::relax::ShapeExprNode>()) {
                const int64_t before = group->values[0].template as<::tvm::IntImmNode>()->value;
                const int64_t after = group->values[1].template as<::tvm::IntImmNode>()->value;
                for (int64_t d = before; d < static_cast<int64_t>(inShape.size()) - after; ++d)
                    outDims.push_back(::tvm::cast(::tvm::DataType::Int(64), inShape[d]));
                continue;
            }
            ::tvm::PrimExpr dim = extractPrim(ts[i]);
            if (const auto* imm = dim.as<::tvm::IntImmNode>()) {
                if (imm->value == -1) {
                    // passthrough: use next input dim
                    if (passthroughIdx < static_cast<int>(inShape.size()))
                        dim = ::tvm::cast(::tvm::DataType::Int(64), inShape[passthroughIdx]);
                    ++passthroughIdx;
                } else if (imm->value == -2) {
                    // infer: pass TVM's -1 sentinel
                    dim = ::tvm::IntImm(::tvm::DataType::Int(64), -1);
                } else {
                    dim = ::tvm::cast(::tvm::DataType::Int(64), dim);
                }
            } else {
                // symbolic (e.g. hyperparameter): cast to int64 for ShapeExpr
                dim = ::tvm::cast(::tvm::DataType::Int(64), dim);
            }
            outDims.push_back(dim);
        }
        ::tvm::ffi::Array<::tvm::PrimExpr> shape_arr = outDims;
        return callRelaxOp("relax.op.reshape", tensor, shape_arr);
    },

    /**
     * PERMUTE — ts[0] tensor, ts[1] ShapeExpr encoding the permutation (see FunctionIDs::PERMUTE).
     */
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        const auto& tensor = ts[0];
        const auto* se = ts[1].template as<::tvm::relax::ShapeExprNode>();
        assert(se && "PERMUTE: expected ShapeExpr for permutation");
        auto axes = permuteAxes(tensorNdim(tensor), se);
        if (axes.empty())
            return tensor; // identity fallback
        ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>> opt_axes = axes;
        return callRelaxOp("relax.op.permute_dims", tensor, opt_axes);
    },

    /**
     * EXPAND (broadcast_to) — ts[0] tensor, ts[1..n] target dims as PrimValues.
     *
     * Sentinel -1: keep the input dim at that position.
     */
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        const auto& tensor = ts[0];
        auto inShape = tensorShape(tensor);

        ::tvm::ffi::Array<::tvm::PrimExpr> outDims;
        outDims.reserve(ts.size() - 1);
        for (size_t i = 1; i < ts.size(); ++i) {
            ::tvm::PrimExpr dim = extractPrim(ts[i]);
            if (const auto* imm = dim.as<::tvm::IntImmNode>(); imm && imm->value == -1) {
                int idx = static_cast<int>(i - 1);
                if (idx < static_cast<int>(inShape.size()))
                    dim = ::tvm::cast(::tvm::DataType::Int(64), inShape[idx]);
            } else {
                dim = ::tvm::cast(::tvm::DataType::Int(64), dim);
            }
            outDims.push_back(dim);
        }
        ::tvm::relax::Expr shape_expr = ::tvm::relax::ShapeExpr(outDims);
        return callRelaxOp("relax.op.broadcast_to", tensor, shape_expr);
    },

    // ── adjoints (gradient pass) ────────────────────────────────────────────
    /** COLLAPSE_SUM_LIKE — sum ts[0] to the shape of ts[1] **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        ::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[1]));
        return callRelaxOp("relax.op.collapse_sum_to", ts[0], ::tvm::relax::ShapeExpr(shape));
    },
    /** BROADCAST_LIKE — broadcast ts[0] to the shape of ts[1] **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        ::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[1]));
        return callRelaxOp("relax.op.broadcast_to", ts[0], ::tvm::relax::ShapeExpr(shape));
    },
    /** RESHAPE_LIKE — reshape ts[0] to the shape of ts[1] **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        ::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[1]));
        return callRelaxOp("relax.op.reshape", ts[0], shape);
    },
    /** PERMUTE_INVERSE — permute ts[0] by the inverse of the permutation ts[1] **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        const auto* se = ts[1].template as<::tvm::relax::ShapeExprNode>();
        assert(se && "PERMUTE_INVERSE: expected ShapeExpr for permutation");
        auto axes = permuteAxes(tensorNdim(ts[0]), se);
        if (axes.empty())
            return ts[0];
        std::vector<int64_t> inverse(axes.size());
        for (size_t i = 0; i < axes.size(); ++i)
            inverse[axes[i]->value] = static_cast<int64_t>(i);
        ::tvm::ffi::Array<::tvm::Integer> inv;
        for (int64_t a : inverse)
            inv.push_back(::tvm::Integer(a));
        ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>> opt_axes = inv;
        return callRelaxOp("relax.op.permute_dims", ts[0], opt_axes);
    },
    /** EQUAL_MASK — 1 where ts[0] == ts[1], else 0 (float32) **/
    [](auto& bb, auto ts) -> ::tvm::relax::Expr {
        auto eq = emitAs(bb, callRelaxOp("relax.op.equal", ts[0], ts[1]), "eq");
        return callRelaxOp("relax.op.astype", eq, ::tvm::DataType::Float(32));
    },
    /** STOP_GRADIENT — identity (only meaningful to the gradient pass) **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return ts[0]; },
});

// Every builtin needs a translation (the table is positional).
static_assert(std::tuple_size_v<decltype(relaxFunctions)> == std::to_underlying(mdl::builtins::FunctionIDs::COUNT));

} // namespace mdl::backend::tvm

#pragma once

#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/tensor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <stdexcept>
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

/** The element type of a tensor expression (float32 if unknown). */
static inline ::tvm::DataType dtypeOf(const ::tvm::relax::Expr& e) {
    const auto* sinfo = e->struct_info_.as<::tvm::relax::TensorStructInfoNode>();
    return sinfo != nullptr && !sinfo->IsUnknownDtype() ? sinfo->dtype : ::tvm::DataType::Float(32);
}

/** Create a 0-D float Constant of @p dtype (float32, float16, or bfloat16). */
static inline ::tvm::relax::Expr makeFloatConst(float val, ::tvm::DataType dtype = ::tvm::DataType::Float(32)) {
    DLDevice dev{kDLCPU, 0};
    auto data = ::tvm::runtime::Tensor::Empty(::tvm::ffi::Shape{}, dtype, dev);
    if (dtype == ::tvm::DataType::Float(16)) {
        const _Float16 half = static_cast<_Float16>(val);
        data.CopyFromBytes(&half, sizeof(half));
    } else if (dtype == ::tvm::DataType::BFloat(16)) {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        const uint16_t bf16 = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16); // round to nearest even
        data.CopyFromBytes(&bf16, sizeof(bf16));
    } else {
        data.CopyFromBytes(&val, sizeof(float));
    }
    return ::tvm::relax::Constant(data);
}

/** The size of the last dimension as a float divisor (a constant if static, else the symbolic size), in x's type. */
static inline ::tvm::relax::Expr lastDimDivisor(const ::tvm::relax::Expr& x) {
    auto shape = tensorShape(x);
    assert(!shape.empty() && "reduction over a tensor without dimensions");
    if (const auto* imm = shape.back().template as<::tvm::IntImmNode>())
        return makeFloatConst(static_cast<float>(imm->value), dtypeOf(x));
    return ::tvm::relax::PrimValue(::tvm::cast(dtypeOf(x), shape.back()));
}

static inline ::tvm::PrimExpr shapeProduct(const std::vector<::tvm::PrimExpr>& shape, size_t from = 0) {
    ::tvm::PrimExpr p = ::tvm::IntImm(::tvm::DataType::Int(64), 1);
    for (size_t i = from; i < shape.size(); ++i)
        p = p * ::tvm::cast(::tvm::DataType::Int(64), shape[i]);
    return ::tvm::arith::Analyzer().Simplify(p);
}

/**
 * `out = zeros like `like` ([V, ...])`, then `out[ids[n]] += g[n]` for every index n (the adjoint of `take`, e.g. the
 * gradient of an embedding table). A hand-written kernel (TVM's scatter is only available in Python): `g` and `ids`
 * are flattened to [N, R] and [N]; the columns are split into chunks of 16 that are processed in parallel, so threads
 * never write the same element (repeated indices accumulate correctly). Indices are clipped like `take(mode="clip")`.
 */
static inline ::tvm::relax::Expr scatterAddRows(
    ::tvm::relax::BlockBuilder& bb, const ::tvm::relax::Expr& g, const ::tvm::relax::Expr& ids,
    const ::tvm::relax::Expr& like)
{
    namespace tirx = ::tvm::tirx;
    using ::tvm::IntImm;
    const auto i64 = ::tvm::DataType::Int(64);
    const auto likeShape = tensorShape(like);
    const ::tvm::PrimExpr V = likeShape.at(0), R = shapeProduct(likeShape, 1), N = shapeProduct(tensorShape(ids));
    auto flat = [&](const ::tvm::relax::Expr& x, ::tvm::ffi::Array<::tvm::PrimExpr> shape, const char* hint) {
        return bb->Emit(callRelaxOp("relax.op.reshape", x, ::tvm::relax::ShapeExpr(shape)), hint);
    };
    // The kernel adds in float32 (also for 16-bit gradients; the translation converts the result).
    auto g32 = dtypeOf(g) == ::tvm::DataType::Float(32) ? g : bb->Emit(callRelaxOp("relax.op.astype", g, ::tvm::DataType::Float(32)), "g32");
    auto g2 = flat(g32, {N, R}, "g_rows");
    auto ids1 = flat(ids, {N}, "ids_flat");
    const auto idsType = ids->struct_info_.as<::tvm::relax::TensorStructInfoNode>()->dtype;

    // The kernel's dimensions are its own variables (bound from the buffers).
    tirx::Var n("n", i64), r("r", i64), v("v", i64);
    auto G = tirx::decl_buffer({n, r}, ::tvm::DataType::Float(32), "g");
    auto I = tirx::decl_buffer({n}, idsType, "ids");
    auto O = tirx::decl_buffer({v, r}, ::tvm::DataType::Float(32), "out");
    tirx::Var hg("g_handle", ::tvm::DataType::Handle()), hi("ids_handle", ::tvm::DataType::Handle()),
        ho("out_handle", ::tvm::DataType::Handle());

    constexpr int64_t kChunk = 16;
    tirx::Var row("row", i64), col("col", i64), chunk("chunk", i64), lane("lane", i64), k("k", i64);
    auto zero = tirx::For(row, IntImm(i64, 0), v, tirx::ForKind::kParallel,
        tirx::For(col, IntImm(i64, 0), r, tirx::ForKind::kSerial,
            tirx::BufferStore(O, ::tvm::FloatImm(::tvm::DataType::Float(32), 0.0), {row, col})));
    const ::tvm::PrimExpr chunkSize = IntImm(i64, kChunk);
    const ::tvm::PrimExpr c = chunk * chunkSize + lane;
    const ::tvm::PrimExpr index = ::tvm::max(::tvm::min(::tvm::cast(i64, tirx::BufferLoad(I, {k})), v - 1), IntImm(i64, 0));
    auto accumulate = tirx::For(chunk, IntImm(i64, 0), ::tvm::floordiv(r + IntImm(i64, kChunk - 1), chunkSize), tirx::ForKind::kParallel,
        tirx::For(k, IntImm(i64, 0), n, tirx::ForKind::kSerial,
            tirx::For(lane, IntImm(i64, 0), IntImm(i64, kChunk), tirx::ForKind::kSerial,
                tirx::IfThenElse(c < r,
                    tirx::BufferStore(O, tirx::BufferLoad(O, {index, c}) + tirx::BufferLoad(G, {k, c}), {index, c})))));
    auto body = tirx::SBlockRealize({}, ::tvm::Bool(true),
        tirx::SBlock({}, {}, {}, "root", tirx::SeqStmt(::tvm::ffi::Array<tirx::Stmt>{zero, accumulate})));
    ::tvm::ffi::Map<tirx::Var, tirx::Buffer> buffers{{hg, G}, {hi, I}, {ho, O}};
    auto func = tirx::PrimFunc({hg, hi, ho}, body, ::tvm::VoidType(), buffers);
    func = ::tvm::WithAttr(std::move(func), "op_pattern", ::tvm::Integer(8)); // opaque: not fused or rescheduled
    auto gv = bb->AddFunction(func, "scatter_add_rows");

    static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
    auto out = bb->Emit(::tvm::relax::Call(callTir, {gv, ::tvm::relax::Tuple({g2, ids1})}, ::tvm::Attrs(),
                                           {::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr({V, R}), ::tvm::DataType::Float(32))}),
                        "scatter");
    return callRelaxOp("relax.op.reshape", out, ::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>(likeShape.begin(), likeShape.end())));
}

/**
 * `out = zeros like `like``, then `out[..., index, ...] = g` along `axis` (the adjoint of selecting an index): a
 * hand-written kernel over `like` flattened to [P, A, Q] (P: the dimensions before the axis, A: the axis, Q: after it).
 * A negative index counts from the end of the axis.
 */
static inline ::tvm::relax::Expr unselectLike(
    ::tvm::relax::BlockBuilder& bb, const ::tvm::relax::Expr& g, int64_t axis, int64_t index,
    const ::tvm::relax::Expr& like)
{
    namespace tirx = ::tvm::tirx;
    using ::tvm::IntImm;
    const auto i64 = ::tvm::DataType::Int(64);
    const auto likeShape = tensorShape(like);
    const int64_t rank = static_cast<int64_t>(likeShape.size());
    const int64_t a = axis < 0 ? axis + rank : axis;
    const ::tvm::PrimExpr P = shapeProduct(std::vector<::tvm::PrimExpr>(likeShape.begin(), likeShape.begin() + a));
    const ::tvm::PrimExpr A = likeShape.at(a);
    const ::tvm::PrimExpr Q = shapeProduct(likeShape, a + 1);
    // A float32 kernel (the translation converts the result to the gradient's type).
    auto g32 = dtypeOf(g) == ::tvm::DataType::Float(32) ? g : bb->Emit(callRelaxOp("relax.op.astype", g, ::tvm::DataType::Float(32)), "g32");
    auto g2 = bb->Emit(callRelaxOp("relax.op.reshape", g32, ::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>{P, Q})), "g_flat");

    tirx::Var p("p", i64), q("q", i64), n("n", i64);
    auto G = tirx::decl_buffer({p, q}, ::tvm::DataType::Float(32), "g");
    auto O = tirx::decl_buffer({p, n, q}, ::tvm::DataType::Float(32), "out");
    tirx::Var hg("g_handle", ::tvm::DataType::Handle()), ho("out_handle", ::tvm::DataType::Handle());
    tirx::Var i("i", i64), j("j", i64), k("k", i64);
    const ::tvm::PrimExpr at = index < 0 ? n + IntImm(i64, index) : ::tvm::PrimExpr(IntImm(i64, index));
    auto value = tirx::Select(j == at, tirx::BufferLoad(G, {i, k}), ::tvm::FloatImm(::tvm::DataType::Float(32), 0.0));
    auto loops = tirx::For(i, IntImm(i64, 0), p, tirx::ForKind::kParallel,
        tirx::For(j, IntImm(i64, 0), n, tirx::ForKind::kSerial,
            tirx::For(k, IntImm(i64, 0), q, tirx::ForKind::kSerial, tirx::BufferStore(O, value, {i, j, k}))));
    auto body = tirx::SBlockRealize({}, ::tvm::Bool(true), tirx::SBlock({}, {}, {}, "root", loops));
    ::tvm::ffi::Map<tirx::Var, tirx::Buffer> buffers{{hg, G}, {ho, O}};
    auto func = ::tvm::WithAttr(tirx::PrimFunc({hg, ho}, body, ::tvm::VoidType(), buffers), "op_pattern", ::tvm::Integer(8));
    auto gv = bb->AddFunction(func, "unselect_like");
    static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
    auto out = bb->Emit(::tvm::relax::Call(callTir, {gv, ::tvm::relax::Tuple({g2})}, ::tvm::Attrs(),
                                           {::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr({P, A, Q}), ::tvm::DataType::Float(32))}),
                        "unselected");
    return callRelaxOp("relax.op.reshape", out, ::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>(likeShape.begin(), likeShape.end())));
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
        bool identity = true;
        for (size_t i = 0; i < axes.size(); ++i)
            identity &= axes[i]->value == static_cast<int64_t>(i);
        if (identity)
            return tensor; // also for an invalid permutation (empty axes); identities would hide patterns
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
    /** SUM_ALL — sum over all axes (0-d result) **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>> all;
        return callRelaxOp("relax.op.sum", ts[0], all, false);
    },
    /** MEAN_ALL — sum over all axes divided by the number of elements **/
    [](auto& bb, auto ts) -> ::tvm::relax::Expr {
        ::tvm::ffi::Optional<::tvm::ffi::Array<::tvm::Integer>> all;
        auto s = emitAs(bb, callRelaxOp("relax.op.sum", ts[0], all, false), "sum");
        ::tvm::PrimExpr count = ::tvm::IntImm(::tvm::DataType::Int(64), 1);
        for (const auto& d : tensorShape(ts[0]))
            count = count * ::tvm::cast(::tvm::DataType::Int(64), d);
        if (const auto* imm = ::tvm::arith::Analyzer().Simplify(count).template as<::tvm::IntImmNode>())
            return callRelaxOp("relax.op.divide", s, makeFloatConst(static_cast<float>(imm->value), dtypeOf(ts[0])));
        // Symbolic number of elements: counted as a tensor (a PrimValue divisor would leave the dimension variables
        // unbound in the kernel).
        ::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[0]));
        auto ones = emitAs(bb, callRelaxOp("relax.op.broadcast_to", makeFloatConst(1.0f, dtypeOf(ts[0])), ::tvm::relax::ShapeExpr(shape)), "ones");
        auto n = emitAs(bb, callRelaxOp("relax.op.sum", ones, all, false), "count");
        return callRelaxOp("relax.op.divide", s, n);
    },
    /** TAKE — rows of ts[0] at the indices ts[1] (out-of-range indices are clipped) **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        return callRelaxOp("relax.op.take", ts[0], ts[1], ::tvm::ffi::Optional<int64_t>(0), ::tvm::ffi::String("clip"));
    },
    /** SCATTER_ADD_ROWS — the adjoint of TAKE **/
    [](auto& bb, auto ts) -> ::tvm::relax::Expr { return scatterAddRows(bb, ts[0], ts[1], ts[2]); },
    /** ONE_HOT_LIKE — ts[0] integer labels; the number of classes is the last dimension of ts[1] **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        const auto like = tensorShape(ts[1]);
        const auto* imm = like.empty() ? nullptr : like.back().template as<::tvm::IntImmNode>();
        if (imm == nullptr)
            throw std::runtime_error("one_hot_like: the number of classes (last dimension) must be static");
        auto value = [](double x) { return ::tvm::relax::PrimValue(::tvm::FloatImm(::tvm::DataType::Float(32), x)); };
        return callRelaxOp("relax.op.one_hot", ts[0], value(1.0), value(0.0), static_cast<int>(imm->value), -1);
    },
    /** WHERE — ts[1] where ts[0], else ts[2] **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return callRelaxOp("relax.op.where", ts[0], ts[1], ts[2]); },
    /** MASK_OF, UNMASK — resolved by the masking pass (placed after WHERE) **/
    [](auto& /*bb*/, auto /*ts*/) -> ::tvm::relax::Expr {
        throw std::runtime_error("`.mask` of a tensor that is not masked in the compiled function (see --mask)");
    },
    [](auto& /*bb*/, auto /*ts*/) -> ::tvm::relax::Expr {
        throw std::runtime_error("`.data` of a tensor that is not masked in the compiled function (see --mask)");
    },
    /** SELECT — ts[0] at the index ts[2] along the axis ts[1] (negative indices count from the end) **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
        const int64_t axis = extractPrim(ts[1]).template as<::tvm::IntImmNode>()->value;
        const int64_t index = extractPrim(ts[2]).template as<::tvm::IntImmNode>()->value;
        const int64_t rank = tensorNdim(ts[0]);
        int32_t i = static_cast<int32_t>(index);
        DLDevice dev{kDLCPU, 0};
        auto data = ::tvm::runtime::Tensor::Empty(::tvm::ffi::Shape{}, ::tvm::DataType::Int(32), dev);
        data.CopyFromBytes(&i, sizeof(i));
        return callRelaxOp("relax.op.take", ts[0], ::tvm::relax::Constant(data),
                           ::tvm::ffi::Optional<int64_t>(axis < 0 ? axis + rank : axis),
                           ::tvm::ffi::String(index < 0 ? "wrap" : "clip"));
    },
    /** UNSELECT_LIKE — the adjoint of SELECT **/
    [](auto& bb, auto ts) -> ::tvm::relax::Expr {
        const int64_t axis = extractPrim(ts[1]).template as<::tvm::IntImmNode>()->value;
        const int64_t index = extractPrim(ts[2]).template as<::tvm::IntImmNode>()->value;
        return unselectLike(bb, ts[0], axis, index, ts[3]);
    },
    /** CAST — converted to the node's element type by the translation (see translateRelax) **/
    [](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return ts[0]; },
});

// Every builtin needs a translation (the table is positional).
static_assert(std::tuple_size_v<decltype(relaxFunctions)> == std::to_underlying(mdl::builtins::FunctionIDs::COUNT));

} // namespace mdl::backend::tvm

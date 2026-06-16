#pragma once

/**
 * @file legalizers.hpp
 * @brief C++ LegalizeOps handlers for MDL's CBLAS + oneDNN kernel dispatch.
 *
 * Each function matches the FLegalize signature:
 *   Expr handler(BlockBuilder bb, Call call)
 *
 * Pattern-independent boilerplate is hidden in file-local helpers.
 * The TIR PrimFuncs built here call extern C functions compiled into the
 * model .so by _DNNL_OPS_SRC (tvm.cpp).
 *
 * Handler map (returned by mdlLegalizeMap()):
 *   relax.matmul         → CBLAS sgemm / batch_strided
 *   relax.divide         → fold 1/sqrt(dk) into CBLAS alpha (else default)
 *   relax.add            → fold bias into mdl_dnnl_matmul_bias_f32 (else default)
 *   relax.nn.softmax     → mdl_dnnl_softmax_f32
 *   relax.nn.gelu        → mdl_dnnl_gelu_f32
 *   relax.nn.layer_norm  → mdl_dnnl_layer_norm_f32
 *   relax.nn.attention   → mdl_sdpa_bsnh_f32
 */

#include <tvm/ir/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <cmath>
#include <string>
#include <vector>

namespace mdl::backend::tvm::legalizers {

// ── local helpers ─────────────────────────────────────────────────────────────

namespace {

using ::tvm::ffi::Array;
using ::tvm::ffi::Map;
using ::tvm::ffi::Optional;
using ::tvm::ffi::String;
using ::tvm::relax::BlockBuilder;
using ::tvm::relax::Call;
using ::tvm::relax::Expr;
using ::tvm::relax::ShapeExpr;
using ::tvm::relax::TensorStructInfo;
using ::tvm::relax::Tuple;
using ::tvm::relax::Var;
using ::tvm::tirx::Buffer;
using ::tvm::PrimExpr;
using ::tvm::IntImm;

static const auto& call_tir_op = ::tvm::ir::Op::Get("relax.call_tir");

// Return the shape values of a relax tensor expression.
Array<PrimExpr> tensorDims(const Expr& e) {
    auto sinfo = ::tvm::Downcast<TensorStructInfo>(e->struct_info_);
    return ::tvm::Downcast<ShapeExpr>(sinfo->shape.value())->values;
}

TensorStructInfo tensorSinfo(const Expr& e) {
    return ::tvm::Downcast<TensorStructInfo>(e->struct_info_);
}

// Build a 1-arg extern call:  call_extern("int32", name, out, in, args...)
::tvm::tirx::Stmt externCallStmt(
    const std::string& fn_name,
    const std::vector<PrimExpr>& args)
{
    auto i32 = ::tvm::DataType::Int(32);
    Array<PrimExpr> full_args;
    full_args.push_back(::tvm::tirx::StringImm(fn_name));
    for (const auto& a : args) full_args.push_back(a);
    auto expr = ::tvm::tirx::Call(i32, ::tvm::tirx::builtin::call_extern(), full_args);
    return ::tvm::tirx::Evaluate(expr);
}

// Build a simple 1-input, 1-output TIR PrimFunc that calls an extern.
// body_args receives: [out_buf.access_ptr(write), in_buf.access_ptr(read), extra...]
::tvm::tirx::PrimFunc build1to1ExternFunc(
    const Array<PrimExpr>& shape,
    ::tvm::DataType dtype,
    const std::string& fn_name,
    const std::string& func_label,
    const std::vector<PrimExpr>& extra_args)
{
    auto in_v  = ::tvm::tirx::Var("in",  ::tvm::DataType::Handle());
    auto out_v = ::tvm::tirx::Var("out", ::tvm::DataType::Handle());
    auto in_b  = ::tvm::tirx::decl_buffer(shape, dtype, "in");
    auto out_b = ::tvm::tirx::decl_buffer(shape, dtype, "out");
    in_b.CopyOnWrite()->data  = in_v;
    out_b.CopyOnWrite()->data = out_v;

    std::vector<PrimExpr> call_args = {
        out_b.access_ptr(2),   // write
        in_b.access_ptr(1),    // read
    };
    for (const auto& a : extra_args) call_args.push_back(a);

    auto body = externCallStmt(fn_name, call_args);
    Map<::tvm::tirx::Var, Buffer> buf_map = {{in_v, in_b}, {out_v, out_b}};
    return ::tvm::tirx::PrimFunc({in_v, out_v}, body, ::tvm::VoidType(), buf_map);
}

// Emit call_tir(gv, {inputs...}, out_sinfo) from a BlockBuilder handler.
Expr emitExternTIR(
    const BlockBuilder& bb,
    const ::tvm::tirx::PrimFunc& pf,
    const std::string& hint,
    const Array<Expr>& inputs,
    const TensorStructInfo& out_sinfo)
{
    auto gv  = bb->AddFunction(pf, hint);
    auto tup = ::tvm::relax::Tuple(inputs);
    return ::tvm::relax::Call(call_tir_op, {gv, tup}, ::tvm::Attrs(), {out_sinfo});
}

// Build a multi-buffer extern PrimFunc (arbitrary input/output layout).
struct BufSpec {
    ::tvm::tirx::Var  var;
    Buffer            buf;
    int               access;  // 1=read 2=write
};

::tvm::tirx::PrimFunc buildExternFunc(
    const std::vector<BufSpec>& specs,
    const std::string& fn_name,
    const std::vector<PrimExpr>& extra_args)
{
    Array<::tvm::tirx::Var>         params;
    Map<::tvm::tirx::Var, Buffer>   buf_map;
    std::vector<PrimExpr>           call_args;

    for (const auto& s : specs) {
        params.push_back(s.var);
        buf_map.Set(s.var, s.buf);
        call_args.push_back(s.buf.access_ptr(s.access));
    }
    for (const auto& a : extra_args) call_args.push_back(a);

    auto body = externCallStmt(fn_name, call_args);
    return ::tvm::tirx::PrimFunc(params, body, ::tvm::VoidType(), buf_map);
}

// Make a named handle Var + buffer pair.
BufSpec makeSpec(const std::string& name, const Array<PrimExpr>& shape,
                 ::tvm::DataType dtype, int access) {
    BufSpec s;
    s.var    = ::tvm::tirx::Var(name, ::tvm::DataType::Handle());
    s.buf    = ::tvm::tirx::decl_buffer(shape, dtype, name);
    s.access = access;
    s.buf.CopyOnWrite()->data = s.var;
    return s;
}

// Retrieve the default FLegalize function registered for an op (used as fallback).
Optional<::tvm::relax::FLegalize> defaultLegalize(const Call& call) {
    static const auto& legalize_map =
        ::tvm::ir::Op::GetAttrMap<::tvm::relax::FLegalize>("FLegalize");
    if (auto op = call->op.as<::tvm::ir::OpNode>()) {
        auto ref = ::tvm::ffi::GetRef<::tvm::ir::Op>(op);
        if (legalize_map.count(ref))
            return legalize_map[ref];
    }
    return std::nullopt;
}

// Invoke the default legalization (element-wise ops like divide/add).
Expr fallbackLegalize(const BlockBuilder& bb, const Call& call) {
    if (auto fn = defaultLegalize(call)) return (*fn)(bb, call);
    return call;  // no default → leave unchanged
}

} // anonymous namespace

// ── Softmax ──────────────────────────────────────────────────────────────────

inline Expr legalizeSoftmaxDnnl(BlockBuilder bb, Call call) {
    auto x     = call->args[0];
    auto shape = tensorDims(x);
    auto sinfo = tensorSinfo(x);
    auto f32   = ::tvm::DataType::Float(32);
    auto i64   = ::tvm::DataType::Int(64);

    size_t ndim = shape.size();
    PrimExpr cols = shape[ndim - 1];
    PrimExpr rows = ::tvm::IntImm(i64, 1);
    for (size_t i = 0; i + 1 < ndim; ++i)
        rows = rows * shape[i];

    auto pf = build1to1ExternFunc(shape, f32, "mdl_dnnl_softmax_f32", "_dnnl_softmax",
                                  {rows, cols});
    return emitExternTIR(bb, pf, "_dnnl_softmax", {x}, sinfo);
}

// ── GELU ─────────────────────────────────────────────────────────────────────

inline Expr legalizeGeluDnnl(BlockBuilder bb, Call call) {
    auto x     = call->args[0];
    auto shape = tensorDims(x);
    auto sinfo = tensorSinfo(x);
    auto f32   = ::tvm::DataType::Float(32);
    auto i64   = ::tvm::DataType::Int(64);

    PrimExpr n = ::tvm::IntImm(i64, 1);
    for (const auto& d : shape) n = n * d;

    auto pf = build1to1ExternFunc(shape, f32, "mdl_dnnl_gelu_f32", "_dnnl_gelu", {n});
    return emitExternTIR(bb, pf, "_dnnl_gelu", {x}, sinfo);
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

inline Expr legalizeLayerNormDnnl(BlockBuilder bb, Call call) {
    // relax.nn.layer_norm(data, gamma, beta, axes, epsilon, center, scale)
    auto data  = call->args[0];
    auto gamma = call->args[1];
    auto beta  = call->args[2];
    auto dshape = tensorDims(data);
    auto sinfo  = tensorSinfo(data);
    auto f32    = ::tvm::DataType::Float(32);
    auto i64    = ::tvm::DataType::Int(64);

    // axes attr is a single integer (last axis).
    // rows = product of all dims except last, cols = last dim.
    size_t ndim = dshape.size();
    PrimExpr cols = dshape[ndim - 1];
    PrimExpr rows = ::tvm::IntImm(i64, 1);
    for (size_t i = 0; i + 1 < ndim; ++i) rows = rows * dshape[i];

    auto gshape = tensorDims(gamma);  // [cols]
    float eps = 1e-5f;
    // Extract epsilon from attrs if available
    if (auto* attrs = call->attrs.as<::tvm::AttrsNode>()) {
        // Attrs differ per op; access epsilon via common path if possible
        (void)attrs;
    }

    // mdl_dnnl_layer_norm_f32(out, in, gamma, beta, rows, cols, eps)
    auto out_s   = makeSpec("out",   dshape,  f32, 2);
    auto in_s    = makeSpec("in",    dshape,  f32, 1);
    auto gamma_s = makeSpec("gamma", gshape,  f32, 1);
    auto beta_s  = makeSpec("beta",  gshape,  f32, 1);

    auto pf = buildExternFunc({out_s, in_s, gamma_s, beta_s},
                              "mdl_dnnl_layer_norm_f32",
                              {rows, cols, ::tvm::FloatImm(::tvm::DataType::Float(32), eps)});
    return emitExternTIR(bb, pf, "_dnnl_layer_norm", {data, gamma, beta}, sinfo);
}

// ── Matmul → CBLAS ───────────────────────────────────────────────────────────

namespace {

// Build TIR for: cblas_sgemm(CblasRowMajor, NoTrans, NoTrans, M, N, K, 1, A, K, B, N, 0, C, N)
::tvm::tirx::PrimFunc buildCblasMatmulFunc(const Array<PrimExpr>& ashape,
                                            const Array<PrimExpr>& bshape,
                                            bool trans_b) {
    auto f32  = ::tvm::DataType::Float(32);
    auto i32  = ::tvm::DataType::Int(32);

    PrimExpr M = ashape[0], K = ashape[1];
    PrimExpr N = trans_b ? bshape[0] : bshape[1];
    Array<PrimExpr> cshape = {M, N};

    auto A = makeSpec("A", ashape, f32, 1);
    auto B = makeSpec("B", bshape, f32, 1);
    auto C = makeSpec("C", cshape, f32, 2);

    // cblas constants
    auto CblasRowMajor = ::tvm::IntImm(i32, 101);
    auto CblasNoTrans  = ::tvm::IntImm(i32, 111);
    auto CblasTrans    = ::tvm::IntImm(i32, 112);
    auto one  = ::tvm::FloatImm(f32, 1.0f);
    auto zero = ::tvm::FloatImm(f32, 0.0f);

    PrimExpr ldb = trans_b ? K : N;

    auto pf = buildExternFunc(
        {A, B, C},
        "cblas_sgemm",
        {
            CblasRowMajor, CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            ::tvm::tirx::Cast(i32, M), ::tvm::tirx::Cast(i32, N), ::tvm::tirx::Cast(i32, K),
            one,
            A.buf.access_ptr(1), ::tvm::tirx::Cast(i32, K),
            B.buf.access_ptr(1), ::tvm::tirx::Cast(i32, ldb),
            zero,
            C.buf.access_ptr(2), ::tvm::tirx::Cast(i32, N),
        });
    return pf;
}

// Build TIR for cblas_sgemm_batch_strided (3D batched).
::tvm::tirx::PrimFunc buildCblasBatmulFunc(const Array<PrimExpr>& ashape,
                                            const Array<PrimExpr>& bshape,
                                            float alpha_val = 1.0f) {
    auto f32  = ::tvm::DataType::Float(32);
    auto i32  = ::tvm::DataType::Int(32);
    auto i64  = ::tvm::DataType::Int(64);

    PrimExpr batch = ashape[0], M = ashape[1], K = ashape[2];
    PrimExpr N = bshape[2];
    Array<PrimExpr> cshape = {batch, M, N};

    auto A = makeSpec("A", ashape, f32, 1);
    auto B = makeSpec("B", bshape, f32, 1);
    auto C = makeSpec("C", cshape, f32, 2);

    auto CblasRowMajor = ::tvm::IntImm(i32, 101);
    auto CblasNoTrans  = ::tvm::IntImm(i32, 111);
    auto alpha  = ::tvm::FloatImm(f32, alpha_val);
    auto zero   = ::tvm::FloatImm(f32, 0.0f);
    auto one_i  = ::tvm::IntImm(i32, 1);  // group_count

    auto pf = buildExternFunc(
        {A, B, C},
        "cblas_sgemm_batch_strided",
        {
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            ::tvm::tirx::Cast(i32, M), ::tvm::tirx::Cast(i32, N), ::tvm::tirx::Cast(i32, K),
            alpha,
            A.buf.access_ptr(1), ::tvm::tirx::Cast(i32, K),
            M * K,   // stride_a
            B.buf.access_ptr(1), ::tvm::tirx::Cast(i32, N),
            K * N,   // stride_b
            zero,
            C.buf.access_ptr(2), ::tvm::tirx::Cast(i32, N),
            M * N,   // stride_c
            ::tvm::tirx::Cast(i32, batch), // batch_count
        });
    return pf;
}

// Build TIR for mdl_dnnl_matmul_bias_f32.
::tvm::tirx::PrimFunc buildMatmulBiasFunc(const Array<PrimExpr>& ashape,
                                           const Array<PrimExpr>& bshape,
                                           bool trans_b) {
    auto f32 = ::tvm::DataType::Float(32);
    auto i32 = ::tvm::DataType::Int(32);

    PrimExpr M = ashape[0], K = ashape[1];
    PrimExpr N = trans_b ? bshape[0] : bshape[1];
    Array<PrimExpr> cshape = {M, N};
    Array<PrimExpr> biasSh = {N};

    auto A    = makeSpec("A",    ashape, f32, 1);
    auto B    = makeSpec("B",    bshape, f32, 1);
    auto bias = makeSpec("bias", biasSh, f32, 1);
    auto C    = makeSpec("C",    cshape, f32, 2);

    return buildExternFunc(
        {C, A, B, bias},
        "mdl_dnnl_matmul_bias_f32",
        {
            ::tvm::tirx::Cast(i32, M), ::tvm::tirx::Cast(i32, K), ::tvm::tirx::Cast(i32, N),
            ::tvm::IntImm(i32, trans_b ? 1 : 0),
        });
}

// Return true if expr is a call_tir node whose GV name starts with prefix.
bool isTirCallWithPrefix(const Expr& e, const std::string& prefix) {
    const auto* c = e.as<::tvm::relax::CallNode>();
    if (!c) return false;
    if (!c->op.same_as(call_tir_op)) return false;
    if (c->args.empty()) return false;
    const auto* gv = c->args[0].as<::tvm::ir::GlobalVarNode>();
    if (!gv) return false;
    return gv->name_hint.rfind(prefix, 0) == 0;
}

// Check if B is a transposed weight: a 2D call_tir with swapped shape dims.
Optional<Expr> detectTransposeB(const BlockBuilder& bb, const Var& B) {
    auto binding = bb->LookupBinding(B);
    if (!binding.defined()) return std::nullopt;
    const auto* c = binding.value().as<::tvm::relax::CallNode>();
    if (!c || !c->op.same_as(call_tir_op)) return std::nullopt;
    if (c->args.size() < 2) return std::nullopt;
    const auto* tup = c->args[1].as<::tvm::relax::TupleNode>();
    if (!tup || tup->fields.size() != 1) return std::nullopt;
    auto B_in = tup->fields[0];

    auto in_dims  = tensorDims(B_in);
    auto out_dims = tensorDims(B);
    if (in_dims.size() != 2 || out_dims.size() != 2) return std::nullopt;

    const auto* i0 = in_dims[0].as<::tvm::IntImmNode>();
    const auto* i1 = in_dims[1].as<::tvm::IntImmNode>();
    const auto* o0 = out_dims[0].as<::tvm::IntImmNode>();
    const auto* o1 = out_dims[1].as<::tvm::IntImmNode>();
    if (!i0 || !i1 || !o0 || !o1) return std::nullopt;
    if (i0->value == o1->value && i1->value == o0->value)
        return B_in;
    return std::nullopt;
}

} // anonymous namespace

inline Expr legalizeMatmulCblas(BlockBuilder bb, Call call) {
    auto A = call->args[0];
    auto B = call->args[1];
    auto f32 = ::tvm::DataType::Float(32);

    auto ashape = tensorDims(A);
    auto bshape = tensorDims(B);

    // 3D batched: A[batch,M,K] @ B[batch,K,N] → cblas_sgemm_batch_strided
    if (ashape.size() == 3 && bshape.size() == 3) {
        auto pf    = buildCblasBatmulFunc(ashape, bshape);
        Array<PrimExpr> cshape = {ashape[0], ashape[1], bshape[2]};
        auto csinfo = TensorStructInfo(ShapeExpr(cshape), f32);
        return emitExternTIR(bb, pf, "_cblas_batmul", {A, B}, csinfo);
    }

    // 2D: check for pre-transposed weight
    if (auto B_var = B.as<Var>()) {
        if (auto B_orig = detectTransposeB(bb, B_var.value())) {
            auto bshape_orig = tensorDims(B_orig.value());
            auto pf    = buildCblasMatmulFunc(ashape, bshape_orig, /*trans_b=*/true);
            PrimExpr N = bshape_orig[0];  // pre-transpose: [N, K]
            Array<PrimExpr> cshape = {ashape[0], N};
            auto csinfo = TensorStructInfo(ShapeExpr(cshape), f32);
            return emitExternTIR(bb, pf, "_cblas_matmul_transb", {A, B_orig.value()}, csinfo);
        }
    }

    // 2D: plain A @ B
    auto pf    = buildCblasMatmulFunc(ashape, bshape, /*trans_b=*/false);
    PrimExpr N = bshape[1];
    Array<PrimExpr> cshape = {ashape[0], N};
    auto csinfo = TensorStructInfo(ShapeExpr(cshape), f32);
    return emitExternTIR(bb, pf, "_cblas_matmul", {A, B}, csinfo);
}

// ── Divide: fold attention scale into batmul alpha ────────────────────────────

inline Expr legalizeDivideFoldScale(BlockBuilder bb, Call call) {
    auto dividend = call->args[0];
    auto divisor  = call->args[1];

    // Only fold if divisor is a scalar constant and dividend comes from batmul.
    const auto* div_const = divisor.as<::tvm::relax::ConstantNode>();
    if (!div_const) return fallbackLegalize(bb, call);

    auto dv = dividend.as<Var>();
    if (!dv) return fallbackLegalize(bb, call);

    auto binding = bb->LookupBinding(dv.value());
    if (!binding.defined()) return fallbackLegalize(bb, call);
    if (!isTirCallWithPrefix(binding.value(), "_cblas_batmul"))
        return fallbackLegalize(bb, call);

    // Extract scalar value from the constant.
    auto* cn = div_const->data.as<::tvm::runtime::NDArray::ContainerType>();
    if (!cn) return fallbackLegalize(bb, call);

    // Get divisor scalar from constant NDArray
    float scale_val = 0.0f;
    {
        auto arr = div_const->data;
        if (arr->dtype.code != kDLFloat || arr->dtype.bits != 32) return fallbackLegalize(bb, call);
        if (arr->ndim != 0 && arr.Size() != 1) return fallbackLegalize(bb, call);
        scale_val = static_cast<const float*>(arr->data)[0];
    }
    if (scale_val == 0.0f) return fallbackLegalize(bb, call);

    float alpha = 1.0f / scale_val;

    // Extract A, B from the batmul binding
    const auto* batmul_call = binding.value().as<::tvm::relax::CallNode>();
    if (!batmul_call || batmul_call->args.size() < 2) return fallbackLegalize(bb, call);
    const auto* args_tup = batmul_call->args[1].as<::tvm::relax::TupleNode>();
    if (!args_tup || args_tup->fields.size() != 2) return fallbackLegalize(bb, call);

    auto A = args_tup->fields[0];
    auto B = args_tup->fields[1];
    auto ashape = tensorDims(A);
    auto bshape = tensorDims(B);
    if (ashape.size() != 3 || bshape.size() != 3) return fallbackLegalize(bb, call);

    auto pf = buildCblasBatmulFunc(ashape, bshape, alpha);
    Array<PrimExpr> cshape = {ashape[0], ashape[1], bshape[2]};
    auto csinfo = TensorStructInfo(ShapeExpr(cshape), ::tvm::DataType::Float(32));
    return emitExternTIR(bb, pf, "_cblas_batmul_scaled", {A, B}, csinfo);
}

// ── Add: fold bias into matmul ────────────────────────────────────────────────

inline Expr legalizeAddFoldBias(BlockBuilder bb, Call call) {
    auto lhs = call->args[0];
    auto rhs = call->args[1];

    // Identify which side is 1-D bias
    Expr matmul_expr, bias;
    auto is1d = [](const Expr& e) {
        const auto* s = e->struct_info_.as<::tvm::relax::TensorStructInfoNode>();
        return s && s->shape.defined() &&
               ::tvm::Downcast<ShapeExpr>(s->shape.value())->values.size() == 1;
    };
    if (is1d(rhs))      { matmul_expr = lhs; bias = rhs; }
    else if (is1d(lhs)) { matmul_expr = rhs; bias = lhs; }
    else return fallbackLegalize(bb, call);

    auto mv = matmul_expr.as<Var>();
    if (!mv) return fallbackLegalize(bb, call);
    auto binding = bb->LookupBinding(mv.value());
    if (!binding.defined()) return fallbackLegalize(bb, call);
    if (!isTirCallWithPrefix(binding.value(), "_cblas_matmul"))
        return fallbackLegalize(bb, call);

    // Recover A, B from the matmul binding
    const auto* mc = binding.value().as<::tvm::relax::CallNode>();
    if (!mc || mc->args.size() < 2) return fallbackLegalize(bb, call);
    const auto* tup = mc->args[1].as<::tvm::relax::TupleNode>();
    if (!tup || tup->fields.size() != 2) return fallbackLegalize(bb, call);
    auto A = tup->fields[0];
    auto B = tup->fields[1];

    auto ashape = tensorDims(A);
    auto bshape = tensorDims(B);
    if (ashape.size() != 2 || bshape.size() != 2) return fallbackLegalize(bb, call);

    // Determine trans_b from shape
    const auto* ka = ashape[1].as<::tvm::IntImmNode>();
    const auto* b0 = bshape[0].as<::tvm::IntImmNode>();
    const auto* b1 = bshape[1].as<::tvm::IntImmNode>();
    if (!ka || !b0 || !b1) return fallbackLegalize(bb, call);

    bool trans_b;
    if (b1->value == ka->value)      trans_b = true;   // B is [N,K]
    else if (b0->value == ka->value) trans_b = false;  // B is [K,N]
    else return fallbackLegalize(bb, call);

    auto pf = buildMatmulBiasFunc(ashape, bshape, trans_b);
    PrimExpr M = ashape[0];
    PrimExpr N = trans_b ? bshape[0] : bshape[1];
    Array<PrimExpr> cshape = {M, N};
    auto csinfo = TensorStructInfo(ShapeExpr(cshape), ::tvm::DataType::Float(32));
    return emitExternTIR(bb, pf, "_cblas_matmul_biased", {A, B, bias}, csinfo);
}

// ── Attention → SDPA ─────────────────────────────────────────────────────────

inline Expr legalizeAttentionSdpa(BlockBuilder bb, Call call) {
    // relax.nn.attention(Q[1,S,H,D], K[1,S,H,D], V[1,S,H,D], ...)
    // → mdl_sdpa_bsnh_f32(out, Q, K, V, S, H, D)
    auto Q = call->args[0];
    auto K = call->args[1];
    auto V = call->args[2];

    auto qshape = tensorDims(Q);  // [1, S, H, D]
    auto vshape = tensorDims(V);
    if (qshape.size() != 4) return call;  // unsupported shape

    PrimExpr S   = qshape[1];
    PrimExpr H   = qshape[2];
    PrimExpr D_k = qshape[3];
    PrimExpr D_v = vshape[3];

    auto f32 = ::tvm::DataType::Float(32);
    auto i64 = ::tvm::DataType::Int(64);

    // Output: [1, S, H, D_v]
    Array<PrimExpr> out_shape = {qshape[0], S, H, D_v};
    auto out_sinfo = TensorStructInfo(ShapeExpr(out_shape), f32);

    auto out_s = makeSpec("out", out_shape, f32, 2);
    auto q_s   = makeSpec("Q",   qshape,    f32, 1);
    auto k_s   = makeSpec("K",   {::tvm::IntImm(i64,1), S, H, D_k}, f32, 1);
    auto v_s   = makeSpec("V",   {::tvm::IntImm(i64,1), S, H, D_v}, f32, 1);

    auto pf = buildExternFunc({out_s, q_s, k_s, v_s}, "mdl_sdpa_bsnh_f32", {S, H, D_k});
    return emitExternTIR(bb, pf, "_sdpa_bsnh", {Q, K, V}, out_sinfo);
}

// ── Handler map ───────────────────────────────────────────────────────────────

/**
 * @brief Return the LegalizeOps custom handler map for MDL's CBLAS + oneDNN pipeline.
 *
 * Pass this to tvm::relax::transform::LegalizeOps(mdlLegalizeMap()).
 */
inline ::tvm::ffi::Map<::tvm::String, ::tvm::ffi::Function> mdlLegalizeMap() {
    using ::tvm::ffi::Function;
    using ::tvm::TypedPackedFunc;

    auto wrap = [](auto fn) -> Function {
        return TypedPackedFunc<Expr(BlockBuilder, Call)>(fn);
    };

    return {
        {"relax.matmul",        wrap(legalizeMatmulCblas)},
        {"relax.divide",        wrap(legalizeDivideFoldScale)},
        {"relax.add",           wrap(legalizeAddFoldBias)},
        {"relax.nn.softmax",    wrap(legalizeSoftmaxDnnl)},
        {"relax.nn.gelu",       wrap(legalizeGeluDnnl)},
        {"relax.nn.layer_norm", wrap(legalizeLayerNormDnnl)},
        {"relax.nn.attention",  wrap(legalizeAttentionSdpa)},
    };
}

} // namespace mdl::backend::tvm::legalizers

#include "./tvm.hpp"

#include "../../argvals.hpp"
#include "../../logging.hpp"
#include "./details/relax_builtins.hpp"
#include "./details/utils.hpp"

#include <mdl/compiler.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/utils.hpp>

#include <tvm/node/serialization.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/node/cast.h>

#include <any>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <sstream>
#include <unordered_map>

// ── type helpers ──────────────────────────────────────────────────────────────

static ::tvm::runtime::DataType toTVMType(mdl::PrimitiveType p) {
    switch (p) {
    case mdl::PrimitiveType::boolean:  return ::tvm::runtime::DataType::Bool();
    case mdl::PrimitiveType::floating: return ::tvm::runtime::DataType::Float(32);
    case mdl::PrimitiveType::uint:     return ::tvm::runtime::DataType::UInt(32);
    case mdl::PrimitiveType::integer:  return ::tvm::runtime::DataType::Int(32);
    }
    assert(false);
}

static ::tvm::runtime::DataType toUnderlyingTVMType(mdl::DataType dtype) {
    return std::visit(
        mdl::overloaded{
            [](const mdl::PrimitiveType& p) { return toTVMType(p); },
            [](const mdl::TensorType& t)    { return toTVMType(t.type); },
            [](const mdl::AggregateType&)   { assert(false); return ::tvm::runtime::DataType{}; },
            [](const mdl::ModuleType&)      { std::abort(); return ::tvm::runtime::DataType{}; },
            [](const mdl::BoundModuleType&) { std::abort(); return ::tvm::runtime::DataType{}; },
        },
        dtype
    );
}

/**
 * Recursively evaluate a compile-time-evaluable Node to a TVM PrimExpr.
 * HyperparameterNodes are looked up in argVals; arithmetic Invocations are folded.
 */
static ::tvm::PrimExpr nodeToPrimExpr(const mdl::Node& node, const mdl::ArgVals& argVals) {
    using FIDs = mdl::builtins::FunctionIDs;
    return std::visit(
        mdl::overloaded{
            [](const mdl::ConstNode& cn) -> ::tvm::PrimExpr {
                return std::visit(
                    mdl::overloaded{
                        [](int v)    -> ::tvm::PrimExpr { return ::tvm::IntImm(::tvm::DataType::Int(32), v); },
                        [](double v) -> ::tvm::PrimExpr { return ::tvm::FloatImm(::tvm::DataType::Float(64), v); },
                        [](const std::vector<int>&) -> ::tvm::PrimExpr { return ::tvm::IntImm(::tvm::DataType::Int(32), 0); },
                    },
                    cn.value
                );
            },
            [&argVals](const mdl::HyperparameterNode& hn) -> ::tvm::PrimExpr {
                if (auto val = argVals.lookup(hn.name))
                    return ::tvm::IntImm(::tvm::DataType::Int(32), *val);
                return ::tvm::tir::Var(hn.name.empty() ? "dim" : hn.name, ::tvm::DataType::Int(32));
            },
            [&argVals](const mdl::InvocationNode& in) -> ::tvm::PrimExpr {
                if (in.arguments.size() == 2) {
                    auto lhs = nodeToPrimExpr(in.arguments[0].get(), argVals);
                    auto rhs = nodeToPrimExpr(in.arguments[1].get(), argVals);
                    switch (in.id) {
                    case FIDs::TENSOR_ADD:  return lhs + rhs;
                    case FIDs::TENSOR_SUB:  return lhs - rhs;
                    case FIDs::TENSOR_PROD: return lhs * rhs;
                    case FIDs::TENSOR_DIV:  return ::tvm::truncdiv(lhs, rhs);
                    default: break;
                    }
                }
                return ::tvm::tir::Var("dim", ::tvm::DataType::Int(32));
            },
            [](const auto&) -> ::tvm::PrimExpr { return ::tvm::tir::Var("dim", ::tvm::DataType::Int(32)); },
        },
        node
    );
}

static int ellipsisDimCounter = 0;

static ::tvm::ffi::Array<::tvm::PrimExpr> getShape(mdl::DataType dtype, const mdl::ArgVals& argVals) {
    return std::visit(
        mdl::overloaded{
            [](const mdl::PrimitiveType&) { return ::tvm::ffi::Array<::tvm::PrimExpr>{1}; },
            [&argVals](const mdl::TensorType& ttype) {
                ::tvm::ffi::Array<::tvm::PrimExpr> shape;
                for (const auto i : ttype.shape) {
                    auto expr = std::visit(
                        mdl::overloaded{
                            [](mdl::Elipsis) -> ::tvm::PrimExpr {
                                return ::tvm::tir::Var(
                                    "B" + std::to_string(ellipsisDimCounter++),
                                    ::tvm::DataType::Int(64)
                                );
                            },
                            [](unsigned u) -> ::tvm::PrimExpr {
                                return ::tvm::IntImm(::tvm::DataType::Int(64), u);
                            },
                            [&argVals](const mdl::Node* node) {
                                return ::tvm::cast(::tvm::DataType::Int(64),
                                                   nodeToPrimExpr(*node, argVals));
                            },
                        },
                        i
                    );
                    shape.emplace_back(expr);
                }
                return shape;
            },
            [](const mdl::AggregateType&)   { assert(false); return ::tvm::ffi::Array<::tvm::PrimExpr>{}; },
            [](const mdl::ModuleType&)      { std::abort(); return ::tvm::ffi::Array<::tvm::PrimExpr>{}; },
            [](const mdl::BoundModuleType&) { std::abort(); return ::tvm::ffi::Array<::tvm::PrimExpr>{}; },
        },
        dtype
    );
}

// ── Relax translation ─────────────────────────────────────────────────────────

// Cache: MDL Node → its Relax Expr (Var or PrimValue or ShapeExpr)
using RelaxCache = std::unordered_map<const mdl::Node*, ::tvm::relax::Expr>;

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb,
    RelaxCache& cache,
    std::vector<::tvm::relax::Var>& params,
    const mdl::Node& node,
    const mdl::ArgVals& argVals
);

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb, RelaxCache&,
    std::vector<::tvm::relax::Var>& params,
    const mdl::ConstNode& node, const mdl::ArgVals&)
{
    return std::visit(
        mdl::overloaded{
            [](int v) -> ::tvm::relax::Expr {
                // Scalar int: used as shape dim (RESHAPE/PERMUTE/EXPAND sentinels)
                return ::tvm::relax::PrimValue(::tvm::IntImm(::tvm::DataType::Int(32), v));
            },
            [](double v) -> ::tvm::relax::Expr {
                // Scalar float: used in arithmetic (e.g. 1e-12 epsilon, 0.5 scale)
                float fval = static_cast<float>(v);
                DLDevice dev = {kDLCPU, 0};
                auto data = ::tvm::runtime::Tensor::Empty(
                    ::tvm::ffi::Shape{},
                    ::tvm::DataType::Float(32),
                    dev
                );
                data.CopyFromBytes(&fval, sizeof(float));
                return ::tvm::relax::Constant(data);
            },
            [](const std::vector<int>& v) -> ::tvm::relax::Expr {
                // Int vector: used as permutation axes in PERMUTE
                ::tvm::ffi::Array<::tvm::PrimExpr> vals;
                for (int x : v)
                    vals.push_back(::tvm::IntImm(::tvm::DataType::Int(32), x));
                return ::tvm::relax::ShapeExpr(vals);
            },
        },
        node.value
    );
}

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb, RelaxCache&,
    std::vector<::tvm::relax::Var>& /*params*/,
    const mdl::ErrorNode& /*node*/, const mdl::ArgVals&)
{
    std::abort(); // compile-time error: should not reach here
}

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb, RelaxCache&,
    std::vector<::tvm::relax::Var>& params,
    const mdl::ParameterNode& node, const mdl::ArgVals& argVals)
{
    // Create a Relax Var with full tensor StructInfo
    auto shape  = getShape(mdl::getDataType(node), argVals);
    auto dtype  = toUnderlyingTVMType(mdl::getDataType(node));
    auto sinfo  = ::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr(shape), dtype);
    auto var    = ::tvm::relax::Var(mdl::getName(node), sinfo);
    params.push_back(var);
    return var;
}

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb, RelaxCache&,
    std::vector<::tvm::relax::Var>& /*params*/,
    const mdl::HyperparameterNode& node, const mdl::ArgVals& argVals)
{
    if (auto val = argVals.lookup(node.name))
        return ::tvm::relax::PrimValue(::tvm::IntImm(::tvm::DataType::Int(32), *val));
    // Symbolic hyperparameter
    return ::tvm::relax::PrimValue(
        ::tvm::tir::Var(node.name.empty() ? "dim" : node.name, ::tvm::DataType::Int(32))
    );
}

// Promote a PrimValue(int) to a float32 0-D Constant for use in arithmetic ops.
static ::tvm::relax::Expr promoteToFloatTensor(const ::tvm::relax::Expr& e) {
    const auto* pv = e.as<::tvm::relax::PrimValueNode>();
    if (!pv) return e;
    float fval = 0.0f;
    if (const auto* imm = pv->value.as<::tvm::IntImmNode>())
        fval = static_cast<float>(imm->value);
    else if (const auto* fimm = pv->value.as<::tvm::FloatImmNode>())
        fval = static_cast<float>(fimm->value);
    DLDevice dev = {kDLCPU, 0};
    auto data = ::tvm::runtime::Tensor::Empty(
        ::tvm::ffi::Shape{}, ::tvm::DataType::Float(32), dev);
    data.CopyFromBytes(&fval, sizeof(float));
    return ::tvm::relax::Constant(data);
}

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb, RelaxCache& cache,
    std::vector<::tvm::relax::Var>& params,
    const mdl::InvocationNode& node, const mdl::ArgVals& argVals)
{
    // Recursively translate all arguments
    std::vector<::tvm::relax::Expr> translatedArgs;
    translatedArgs.reserve(node.arguments.size());
    for (const auto& argRef : node.arguments)
        translatedArgs.push_back(translateRelax(bb, cache, params, argRef.get(), argVals));

    // RESHAPE/PERMUTE/EXPAND pass raw PrimValues as shape sentinels.
    // All other ops operate on tensors — promote any scalar PrimValues to float32 constants.
    using FIDs = mdl::builtins::FunctionIDs;
    bool isShapeOp = (node.id == FIDs::RESHAPE || node.id == FIDs::PERMUTE || node.id == FIDs::EXPAND);
    if (!isShapeOp) {
        for (auto& arg : translatedArgs)
            arg = promoteToFloatTensor(arg);
    }

    auto& fn = mdl::backend::tvm::relaxFunctions[std::to_underlying(node.id)];
    assert(fn != nullptr);
    auto result = fn(bb, translatedArgs);

    // Emit non-leaf expressions into the dataflow block
    if (!result.as<::tvm::relax::VarNode>() &&
        !result.as<::tvm::relax::PrimValueNode>() &&
        !result.as<::tvm::relax::ShapeExprNode>() &&
        !result.as<::tvm::relax::ConstantNode>())
    {
        return bb->Emit(result, "v");
    }
    return result;
}

static ::tvm::relax::Expr translateRelax(
    ::tvm::relax::BlockBuilder& bb, RelaxCache& cache,
    std::vector<::tvm::relax::Var>& params,
    const mdl::Node& node, const mdl::ArgVals& argVals)
{
    if (auto it = cache.find(&node); it != cache.end())
        return it->second;
    auto result = std::visit(
        [&](const auto& n) { return translateRelax(bb, cache, params, n, argVals); },
        node
    );
    cache[&node] = result;
    return result;
}

// ── per-function Relax IR builder ─────────────────────────────────────────────

static void translateFuncRelax(
    ::tvm::relax::BlockBuilder& bb,
    const mdl::FunctionDecl& func,
    const mdl::ArgVals& argVals)
{
    auto logger = mdl::log::getLogger("compiler");
    logger->info("Translating {} (Relax)", func.name);

    RelaxCache cache;
    std::vector<::tvm::relax::Var> paramVars;

    // Set up BeginScope with an empty params list first; params get populated during
    // translation (ParameterNodes push themselves via translateRelax → ParameterNode).
    // We actually need params before BeginScope, so we do a pre-DFS to discover them.

    // Pre-DFS: collect ParameterNodes in visitation order
    std::vector<const mdl::Node*> toVisit;
    std::unordered_map<const mdl::Node*, bool> visited;
    std::function<void(const mdl::Node&)> preTraverse = [&](const mdl::Node& node) {
        if (visited[&node]) return;
        visited[&node] = true;
        if (std::holds_alternative<mdl::InvocationNode>(node)) {
            for (const auto& argRef : std::get<mdl::InvocationNode>(node).arguments)
                preTraverse(argRef.get());
        } else if (std::holds_alternative<mdl::ParameterNode>(node)) {
            toVisit.push_back(&node);
        }
    };
    preTraverse(func.output);

    // Create Relax Var for each parameter in DFS order
    ::tvm::ffi::Array<::tvm::relax::Var> bbParams;
    for (const auto* nodePtr : toVisit) {
        const auto& pn = std::get<mdl::ParameterNode>(*nodePtr);
        auto shape = getShape(mdl::getDataType(pn), argVals);
        auto dtype = toUnderlyingTVMType(mdl::getDataType(pn));
        auto sinfo = ::tvm::relax::TensorStructInfo(::tvm::relax::ShapeExpr(shape), dtype);
        auto var   = ::tvm::relax::Var(mdl::getName(pn), sinfo);
        bbParams.push_back(var);
        cache[nodePtr] = var;
        logger->info("  param: {} {}", mdl::getName(pn),
            std::to_string(mdl::getDataType(pn)));
    }

    // Log param shapes
    for (size_t i = 0; i < bbParams.size(); ++i) {
        const auto* sinfo = bbParams[i]->struct_info_.as<::tvm::relax::TensorStructInfoNode>();
        std::string shape_str = "[";
        if (sinfo) {
            auto maybe = sinfo->GetShape();
            if (maybe.defined()) {
                for (size_t j = 0; j < maybe.value().size(); ++j) {
                    if (j) shape_str += ", ";
                    std::ostringstream ss; ss << maybe.value()[j];
                    shape_str += ss.str();
                }
            }
        }
        shape_str += "]";
        logger->info("  arg[{}] = {} {}", i,
            std::string(bbParams[i]->name_hint()), shape_str);
    }

    bb->BeginScope(bbParams);
    bb->BeginDataflowBlock();

    // Translate the output DAG
    // The paramVars vector is unused here; cache already has the params
    std::vector<::tvm::relax::Var> unusedParams; // not used — params already in cache
    auto result = translateRelax(bb, cache, unusedParams, func.output, argVals);

    // Emit as output binding
    // result might already be a Var or might need emitting
    ::tvm::relax::Var outputVar;
    if (auto* v = result.as<::tvm::relax::VarNode>()) {
        outputVar = bb->EmitOutput(result, "output");
    } else {
        auto emitted = bb->Emit(result, "out");
        outputVar = bb->EmitOutput(emitted, "output");
    }

    auto block = bb->EndBlock();
    bb->EndScope();

    // Build the Relax Function
    auto body = ::tvm::relax::SeqExpr(
        ::tvm::ffi::Array<::tvm::relax::BindingBlock>{block},
        outputVar
    );

    // Downcast Optional<ObjectRef> → StructInfo for the return type
    auto retSI = ::tvm::Downcast<::tvm::relax::StructInfo>(
        outputVar->struct_info_.value()
    );
    ::tvm::relax::Function fn(bbParams, body, retSI, /*is_pure=*/true);

    // Attach global symbol attribute
    fn = ::tvm::WithAttr(fn, ::tvm::attr::kGlobalSymbol, ::tvm::ffi::String{func.name});

    bb->AddFunction(fn, func.name);
    logger->info("  → added Relax function '{}'", func.name);
}

// ── main entry point ──────────────────────────────────────────────────────────

void mdl::backends::tvm::compileToFile(
    const mdl::CompilationUnit& unit,
    const std::filesystem::path& output,
    const mdl::ArgVals& argvals,
    const mdl::CompileOptions& opts)
{
    auto logger = mdl::log::getLogger("compiler");

    // ── 1. Build Relax IRModule ───────────────────────────────────────────────
    auto bb = ::tvm::relax::BlockBuilder::Create(::tvm::ffi::Optional<::tvm::IRModule>{});

    for (const auto& [name, member] : unit.definitions) {
        if (const auto* func = std::get_if<mdl::FunctionDecl>(&member)) {
            // Skip standalone functions — they have dynamic shapes (ellipsis dims)
            // and are inlined into module compute graphs anyway.
            (void)func;
            continue;
        } else if (const auto* module = std::get_if<mdl::ModuleType>(&member)) {
            // Skip template modules whose scalar type args aren't in argVals.
            // Struct-typed typeargs (InvalidDataType) are OK — their leaf fields are in argVals.
            bool hasUnresolvable = false;
            for (const auto& [tname, hn] : module->typeargs) {
                auto res = argvals.lookup(hn.get().name);
                if (!res.has_value() &&
                    res.error() == mdl::ArgVals::ErrorCode::NotFound)
                {
                    hasUnresolvable = true;
                    break;
                }
            }
            if (hasUnresolvable) {
                logger->info("Skipping module '{}' (unresolvable type args)", name);
                continue;
            }
            for (const auto& [funcName, funcDecl] : module->functions) {
                if (!funcDecl.templateParameters.empty()) continue;
                mdl::FunctionDecl mangledDecl = funcDecl;
                mangledDecl.name = mdl::mangledName(name, funcName);
                translateFuncRelax(bb, mangledDecl, argvals);
            }
        }
    }

    auto irMod = bb->Finalize();
    logger->info("Relax IRModule built ({} functions)", irMod->functions.size());

    // ── 2. Determine CPU target string ────────────────────────────────────────
    std::string cpuName = "generic";
    if (auto fn = ::tvm::ffi::Function::GetGlobal("tvm.codegen.llvm.GetHostCPUName"))
        cpuName = (*fn)().cast<std::string>();
    std::string targetStr = "llvm -mcpu=" + cpuName;

    // ── 3. Serialize IRModule to JSON ─────────────────────────────────────────
    std::string jsonPath = output.string() + ".irmod.json";
    {
        std::string json = ::tvm::SaveJSON(irMod);
        std::ofstream f(jsonPath);
        f << json;
    }
    logger->info("IRModule serialized to {}", jsonPath);

    // ── 4. Delegate compilation to Python (relax.build handles LegalizeOps) ───
    //    The Python helper script is written inline to avoid external file deps.
    std::string helperPath = output.string() + ".build_helper.py";
    {
        std::ofstream f(helperPath);
        f << R"PYHELPER(
import sys, os

json_path   = sys.argv[1]
output_path = sys.argv[2]
target_str  = sys.argv[3] if len(sys.argv) > 3 else "llvm"
opt_level   = sys.argv[4] if len(sys.argv) > 4 else "default"

# Find TVM Python package in common build-tree locations
_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_here, "../out/build/conf-dev/_deps/tvm-src/python"),
    "/workspaces/model-description-language/out/build/conf-dev/_deps/tvm-src/python",
    os.environ.get("TVM_PYTHON_PATH", ""),
]
_lib_candidates = [
    "/workspaces/model-description-language/out/build/conf-dev/_deps/tvm-build",
]
for c in _candidates:
    if c and os.path.isdir(c):
        sys.path.insert(0, c)
        # Propagate to subprocess workers (e.g. MetaSchedule popen_worker)
        os.environ["PYTHONPATH"] = c + ":" + os.environ.get("PYTHONPATH", "")
        break
for c in _lib_candidates:
    if c and os.path.isdir(c):
        os.environ["LD_LIBRARY_PATH"] = c + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        break

import tvm
import tvm.relax
import tvm.relax.transform as RT

with open(json_path) as f:
    mod = tvm.ir.load_json(f.read())

target = tvm.target.Target(target_str)

# Step 1a: Bind all symbolic dimension variables to 8.
# The whole IR has dynamic leading dims (B0..B7) for sequence length.
# Binding them statically eliminates runtime shape computation, allows
# StaticPlanBlockMemory to pre-plan all allocations, and unlocks vectorization
# for loops that would otherwise have symbolic extents.
sym_vars = {}
for gv in mod.get_global_vars():
    fn = mod[gv]
    if not isinstance(fn, tvm.relax.Function):
        continue
    for p in fn.params:
        si = p.struct_info
        if hasattr(si, 'shape') and si.shape is not None:
            for dim in si.shape.values:
                if isinstance(dim, tvm.tir.Var) and dim.name not in sym_vars:
                    sym_vars[dim.name] = 8

if sym_vars:
    print(f"[mdl] Binding symbolic vars to 8: {list(sym_vars.keys())}", flush=True)
    mod = RT.BindSymbolicVars(sym_vars)(mod)

def default_pipeline(mod, target, target_str):
    """Default pipeline: CBLAS for 2-D matmul + tile+vectorize rest, FuseOps, VM lowering."""
    # CBLAS dispatch via tir.call_extern — generates a direct LLVM call that survives
    # the full Relax+TIR lowering pipeline without requiring LowerTVMBuiltin.
    # libopenblas (linked into libtvm.so) provides cblas_sgemm at runtime.
    #
    # Two cases:
    #   (a) matmul(A, B) where B is already in the right orientation → NoTrans/NoTrans
    #   (b) matmul(A, permute_dims(B_orig, [1,0])) → eliminate the explicit transpose
    #       by calling cblas_sgemm with CblasTrans on B_orig directly.
    _CblasRowMajor = 101
    _CblasColMajor = 102
    _CblasNoTrans  = 111
    _CblasTrans    = 112

    def _cblas_matmul_te(A, B):
        """C = A[M,K] @ B[K,N] — row-major, no transpose (ColMajor swap trick)."""
        M, K, N = A.shape[0], A.shape[1], B.shape[1]
        return tvm.te.extern(
            (M, N), [A, B],
            lambda ins, outs: tvm.tir.Evaluate(tvm.tir.call_extern(
                "int32", "cblas_sgemm",
                _CblasColMajor, _CblasNoTrans, _CblasNoTrans,
                N, M, K,
                tvm.tir.const(1.0, "float32"),
                ins[1].access_ptr("r"), N,
                ins[0].access_ptr("r"), K,
                tvm.tir.const(0.0, "float32"),
                outs[0].access_ptr("w"), N,
            )),
            name="C", dtype="float32",
        )

    def _cblas_matmul_transb_te(A, B):
        """C = A[M,K] @ B[N,K].T — avoids materialising the explicit transpose.
        Uses CblasRowMajor with CblasTrans so BLAS handles the transposition internally."""
        M, K, N = A.shape[0], A.shape[1], B.shape[0]
        return tvm.te.extern(
            (M, N), [A, B],
            lambda ins, outs: tvm.tir.Evaluate(tvm.tir.call_extern(
                "int32", "cblas_sgemm",
                _CblasRowMajor, _CblasNoTrans, _CblasTrans,
                M, N, K,
                tvm.tir.const(1.0, "float32"),
                ins[0].access_ptr("r"), K,   # A [M, K], lda = K
                ins[1].access_ptr("r"), K,   # B [N, K], ldb = K (transposed by BLAS)
                tvm.tir.const(0.0, "float32"),
                outs[0].access_ptr("w"), N,
            )),
            name="C", dtype="float32",
        )

    _call_tir_op = tvm.ir.Op.get("relax.call_tir")

    def _is_2d_transpose(b_defn, B):
        """Return the original (pre-transpose) tensor if b_defn is a call_tir to a
        2-D transpose (output shape is the row/col-swap of input shape).  Else None."""
        if not (isinstance(b_defn, tvm.relax.Call) and
                isinstance(b_defn.op, tvm.ir.Op) and
                b_defn.op == _call_tir_op):
            return None
        # call_tir(gv, Tuple([input, ...]), sinfo_args)
        args_tuple = b_defn.args[1]
        if not (isinstance(args_tuple, tvm.relax.Tuple) and len(args_tuple.fields) == 1):
            return None
        B_input = args_tuple.fields[0]
        # Shape of the transpose input vs output (what matmul sees as B)
        try:
            in_shape = B_input.struct_info.shape.values
            out_shape = B.struct_info.shape.values
        except Exception:
            return None
        if (len(in_shape) == 2 and len(out_shape) == 2 and
                int(in_shape[0]) == int(out_shape[1]) and
                int(in_shape[1]) == int(out_shape[0])):
            return B_input
        return None

    def _legalize_matmul_cblas(bb, call):
        A, B = call.args
        # LegalizeOps processes permute_dims before matmul, so by the time we see
        # matmul(A, B), B already points to a call_tir(T_transpose, [W], sinfo).
        # Detect this pattern by checking whether B's defining call_tir is a 2-D
        # transpose (swapped shape), and if so use CblasTrans to avoid the copy.
        if isinstance(B, tvm.relax.Var):
            b_defn = bb.lookup_binding(B)
            B_orig = _is_2d_transpose(b_defn, B)
            if B_orig is not None:
                return bb.call_te(_cblas_matmul_transb_te, A, B_orig)
        return bb.call_te(_cblas_matmul_te, A, B)

    with tvm.target.Target(target_str):
        mod = tvm.transform.Sequential([
            RT.LegalizeOps({"relax.matmul": _legalize_matmul_cblas}),
            RT.DeadCodeElimination(),  # remove now-unused permute_dims bindings
        ])(mod)
    print("[mdl] LegalizeOps done (CBLAS dispatch + transpose fusion)", flush=True)

    # Schedule individual TIR PrimFuncs for cache-friendly access.
    # Reorder (i, j, k) → (i, j_outer, k, j_inner) with tile=16:
    #  - j_inner is vectorizable (sequential B access, one cache line per step)
    #  - C accumulator tile stays in SIMD registers across all k iterations
    def try_schedule_reduction_func(func):
        try:
            sch = tvm.tir.Schedule(func)
            for bname in ["matmul", "C", "T_matmul_NN", "T_batch_matmul_NT",
                          "einsum", "T_einsum", "compute"]:
                try:
                    block = sch.get_block(bname)
                except Exception:
                    continue
                loops = sch.get_loops(block)
                if len(loops) < 3:
                    continue
                stmt = sch.get(block)
                ivs = stmt.iter_vars
                spatial = [i for i, iv in enumerate(ivs) if iv.iter_type == 0]
                reduc   = [i for i, iv in enumerate(ivs) if iv.iter_type == 2]
                if len(spatial) < 2 or not reduc:
                    continue
                TILE_J = 16
                last_s_idx = spatial[-1]
                j_loop = loops[last_s_idx]
                j_ext = sch.get(j_loop).extent
                if isinstance(j_ext, tvm.tir.IntImm) and int(j_ext.value) % TILE_J == 0:
                    jo, ji = sch.split(j_loop, factors=[None, TILE_J])
                    loops2 = sch.get_loops(block)
                    outer_s = [loops2[i] for i in spatial[:-1]]
                    new_red = [loops2[i + 1] for i in reduc]
                    new_order = outer_s + [jo] + new_red + [ji]
                    sch.reorder(*new_order)
                    try:
                        sch.vectorize(sch.get_loops(block)[-1])
                    except Exception:
                        pass
                else:
                    new_order = ([loops[i] for i in spatial[:-1]] +
                                 [loops[i] for i in reduc] +
                                 [loops[spatial[-1]]])
                    sch.reorder(*new_order)
                    inner = sch.get_loops(block)[-1]
                    if isinstance(sch.get(inner).extent, tvm.tir.IntImm):
                        try:
                            sch.vectorize(inner)
                        except Exception:
                            pass
                print(f"[mdl]   tiled/vectorized block '{bname}'", flush=True)
            return sch.mod["main"]
        except Exception:
            return func

    n_scheduled = 0
    for gv in list(mod.get_global_vars()):
        fn = mod[gv]
        if not isinstance(fn, tvm.tir.PrimFunc):
            continue
        opt = try_schedule_reduction_func(fn)
        if opt is not fn:
            mod[gv] = opt
            n_scheduled += 1
    print(f"[mdl] Scheduled {n_scheduled} PrimFuncs", flush=True)

    # FuseOps+FuseTIR reduces VM dispatch overhead.
    # Safe now that all shapes are static (no symbolic-sized intermediates).
    with tvm.target.Target(target_str):
        mod = tvm.transform.Sequential([
            RT.AnnotateTIROpPattern(),
            RT.FoldConstant(),
            RT.FuseOps(),
            RT.FuseTIR(),
        ])(mod)
    print("[mdl] FuseOps+FuseTIR done", flush=True)

    # Standard Relax VM lowering
    with tvm.target.Target(target_str):
        mod = tvm.transform.Sequential([
            RT.RewriteDataflowReshape(),
            RT.ToNonDataflow(),
            RT.RemovePurityChecking(),
            RT.CallTIRRewrite(),
            RT.StaticPlanBlockMemory(),
            RT.LowerAllocTensor(),
            RT.KillAfterLastUse(),
            RT.LowerRuntimeBuiltin(),
            RT.ComputePrimValue(),
            RT.VMShapeLower(),
            RT.AttachGlobalSymbol(),
        ])(mod)
    return mod

def metaschedule_pipeline(mod, target, target_str):
    """MetaSchedule pipeline: auto-tune all Relax operators, then compile."""
    import tvm.meta_schedule as ms
    import multiprocessing
    work_dir = output_path + ".ms_workdir"
    os.makedirs(work_dir, exist_ok=True)

    # MetaSchedule requires num-cores. -libs=cblas enables BLAS dispatch for matmul blocks,
    # which the MultiLevelTilingWithIntrin schedule rule will pick up automatically — this
    # keeps the pattern visible to the compiler (no opaque extern call) while letting the
    # runtime link to OpenBLAS.
    num_cores = multiprocessing.cpu_count()
    ms_target_str = target_str + f" -num-cores {num_cores} -libs=cblas"
    ms_target = tvm.target.Target(ms_target_str)

    print(f"[mdl] MetaSchedule tuning (work_dir={work_dir}, cores={num_cores})", flush=True)
    print(f"[mdl] This may take several minutes...", flush=True)

    # Enable TVM's built-in MetaSchedule progress logging
    import logging
    logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)
    logging.basicConfig(format="[mdl/ms] %(message)s", level=logging.INFO)

    # tune_relax requires TIR PrimFuncs — must legalize first so it can extract tasks.
    _CblasRowMajor = 101
    _CblasColMajor = 102
    _CblasNoTrans  = 111
    _CblasTrans    = 112

    def _cblas_matmul_te_ms(A, B):
        M, K, N = A.shape[0], A.shape[1], B.shape[1]
        return tvm.te.extern(
            (M, N), [A, B],
            lambda ins, outs: tvm.tir.Evaluate(tvm.tir.call_extern(
                "int32", "cblas_sgemm",
                _CblasColMajor, _CblasNoTrans, _CblasNoTrans,
                N, M, K,
                tvm.tir.const(1.0, "float32"),
                ins[1].access_ptr("r"), N,
                ins[0].access_ptr("r"), K,
                tvm.tir.const(0.0, "float32"),
                outs[0].access_ptr("w"), N,
            )),
            name="C", dtype="float32",
        )

    def _cblas_matmul_transb_te_ms(A, B):
        M, K, N = A.shape[0], A.shape[1], B.shape[0]
        return tvm.te.extern(
            (M, N), [A, B],
            lambda ins, outs: tvm.tir.Evaluate(tvm.tir.call_extern(
                "int32", "cblas_sgemm",
                _CblasRowMajor, _CblasNoTrans, _CblasTrans,
                M, N, K,
                tvm.tir.const(1.0, "float32"),
                ins[0].access_ptr("r"), K,
                ins[1].access_ptr("r"), K,
                tvm.tir.const(0.0, "float32"),
                outs[0].access_ptr("w"), N,
            )),
            name="C", dtype="float32",
        )

    _call_tir_op_ms = tvm.ir.Op.get("relax.call_tir")

    def _is_2d_transpose_ms(b_defn, B):
        if not (isinstance(b_defn, tvm.relax.Call) and
                isinstance(b_defn.op, tvm.ir.Op) and b_defn.op == _call_tir_op_ms):
            return None
        args_tuple = b_defn.args[1]
        if not (isinstance(args_tuple, tvm.relax.Tuple) and len(args_tuple.fields) == 1):
            return None
        B_input = args_tuple.fields[0]
        try:
            in_s = B_input.struct_info.shape.values
            out_s = B.struct_info.shape.values
        except Exception:
            return None
        if (len(in_s) == 2 and len(out_s) == 2 and
                int(in_s[0]) == int(out_s[1]) and int(in_s[1]) == int(out_s[0])):
            return B_input
        return None

    def _legalize_matmul_cblas_ms(bb, call):
        A, B = call.args
        if isinstance(B, tvm.relax.Var):
            b_defn = bb.lookup_binding(B)
            B_orig = _is_2d_transpose_ms(b_defn, B)
            if B_orig is not None:
                return bb.call_te(_cblas_matmul_transb_te_ms, A, B_orig)
        return bb.call_te(_cblas_matmul_te_ms, A, B)

    with tvm.target.Target(ms_target_str):
        mod_legalized = tvm.transform.Sequential([
            RT.LegalizeOps({"relax.matmul": _legalize_matmul_cblas_ms}),
            RT.DeadCodeElimination(),
        ])(mod)

    # Pick cost model: XGBoost is best but requires the package; fall back to random.
    try:
        import xgboost  # noqa: F401
        cost_model = "xgb"
    except ImportError:
        cost_model = "random"
        print("[mdl] xgboost not found, using random cost model (install xgboost for better tuning)", flush=True)

    # params={} because weights are loaded at runtime, not baked in at compile time.
    # num_trials_per_iter=16 prints the task summary table (with GFLOPS + trial count)
    # more often than the default 64 — useful for tracking progress.
    max_trials_global = 512
    database = ms.relax_integration.tune_relax(
        mod=mod_legalized,
        params={},
        target=ms_target_str,
        work_dir=work_dir,
        max_trials_global=max_trials_global,
        num_trials_per_iter=16,
        cost_model=cost_model,
    )

    print("[mdl] MetaSchedule tuning done, compiling...", flush=True)

    # compile_relax applies the tuned schedules and lowers to VM bytecode
    mod = ms.relax_integration.compile_relax(database, mod_legalized, ms_target_str, params={})

    print("[mdl] MetaSchedule compile done", flush=True)
    return mod, ms_target

# Dispatch based on optimization level
if opt_level in ("metaschedule", "*"):
    try:
        ex, ms_target = metaschedule_pipeline(mod, target, target_str)
        ex.export_library(output_path)
        print(f"[mdl] Relax compiled (MetaSchedule) -> {output_path}", flush=True)
    except Exception as e:
        import traceback
        print(f"[mdl] MetaSchedule failed ({e}), falling back to default", flush=True)
        traceback.print_exc()
        # Re-load the module since metaschedule_pipeline may have partially mutated it
        with open(json_path) as f:
            mod = tvm.ir.load_json(f.read())
        # Re-bind
        sym_vars2 = {}
        for gv in mod.get_global_vars():
            fn = mod[gv]
            if not isinstance(fn, tvm.relax.Function):
                continue
            for p in fn.params:
                si = p.struct_info
                if hasattr(si, 'shape') and si.shape is not None:
                    for dim in si.shape.values:
                        if isinstance(dim, tvm.tir.Var) and dim.name not in sym_vars2:
                            sym_vars2[dim.name] = 8
        if sym_vars2:
            mod = RT.BindSymbolicVars(sym_vars2)(mod)
        mod = default_pipeline(mod, target, target_str)
        ex = tvm.relax.build(mod, target=target, relax_pipeline=None)
        ex.export_library(output_path)
        print(f"[mdl] Relax compiled (fallback) -> {output_path}", flush=True)
else:
    mod = default_pipeline(mod, target, target_str)
    ex = tvm.relax.build(mod, target=target, relax_pipeline=None)
    ex.export_library(output_path)
    print(f"[mdl] Relax compiled -> {output_path}", flush=True)
)PYHELPER";
    }

    std::string cmd = "python3 " + helperPath
                    + " " + jsonPath
                    + " " + output.string()
                    + " \"" + targetStr + "\""
                    + " \"" + (opts.optimization.empty() ? "default" : opts.optimization) + "\"";
    logger->info("Running: {}", cmd);
    int rc = std::system(cmd.c_str());
    if (rc != 0)
        throw std::runtime_error("Python relax.build() failed (exit " + std::to_string(rc) + ")");

    // ── 5. Clean up temp files ─────────────────────────────────────────────────
    // std::filesystem::remove(jsonPath);   // keep for debug
    // std::filesystem::remove(helperPath); // keep for debug
    logger->info("Done → {}", output.string());
}

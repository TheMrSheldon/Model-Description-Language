#include "./pipeline.hpp"
#include "./legalizers.hpp"
#include "./pattern_registry.hpp"
#include "../../../logging.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/var.h>
#include <tvm/ir/transform.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace mdl::backend::tvm {

namespace RT = ::tvm::relax::transform;

// ── helpers ───────────────────────────────────────────────────────────────────

static ::tvm::ffi::Function requireGlobal(const std::string& name) {
    auto fn = ::tvm::ffi::Function::GetGlobal(name);
    if (!fn) throw std::runtime_error("TVM global not found: " + name);
    return *fn;
}

static ::tvm::IRModule runPass(::tvm::IRModule mod,
                                ::tvm::transform::Pass pass,
                                const ::tvm::Target& target)
{
    ::tvm::transform::PassContext ctx = ::tvm::transform::PassContext::Create();
    tvm::With<::tvm::transform::PassContext> guard(ctx);
    tvm::With<::tvm::Target>                 tguard(target);
    return pass(mod);
}

static ::tvm::IRModule runSeq(::tvm::IRModule mod,
                               std::initializer_list<::tvm::transform::Pass> passes,
                               const ::tvm::Target& target)
{
    ::tvm::ffi::Array<::tvm::transform::Pass> arr(passes);
    return runPass(std::move(mod), ::tvm::transform::Sequential(arr), target);
}

// Call a globally registered zero-arg pass factory (e.g. relax.transform.LowerAllocTensor).
static ::tvm::IRModule callGlobalPass(::tvm::IRModule mod, const std::string& global_name) {
    auto pass = requireGlobal(global_name)().cast<::tvm::transform::Pass>();
    ::tvm::transform::PassContext ctx = ::tvm::transform::PassContext::Create();
    tvm::With<::tvm::transform::PassContext> guard(ctx);
    return pass(mod);
}

// ── bind symbolic dims (dynamic shapes → constants for LegalizeOps) ───────────

static ::tvm::IRModule bindSymbolicVars(::tvm::IRModule mod) {
    ::tvm::ffi::Map<::tvm::ffi::Variant<::tvm::tirx::Var, ::tvm::String>, ::tvm::PrimExpr> sym;
    for (const auto& gv : mod->GetGlobalVars()) {
        const auto* fn = mod->Lookup(gv).as<::tvm::relax::FunctionNode>();
        if (!fn) continue;
        for (const auto& p : fn->params) {
            auto sinfo = ::tvm::relax::GetStructInfo(p);
            if (const auto* ts = sinfo.as<::tvm::relax::TensorStructInfoNode>()) {
                if (!ts->shape.defined()) continue;
                if (const auto* se = ts->shape.value().as<::tvm::relax::ShapeExprNode>()) {
                    for (const auto& d : se->values) {
                        if (const auto* v = d.as<::tvm::tirx::VarNode>()) {
                            if (!sym.count(v->name_hint))
                                sym.Set(v->name_hint,
                                        ::tvm::IntImm(::tvm::DataType::Int(64), 8));
                        }
                    }
                }
            }
        }
    }
    if (sym.empty()) return mod;
    return RT::BindSymbolicVars(sym)(mod);
}

// ── main pipeline ─────────────────────────────────────────────────────────────

::tvm::IRModule runRelaxPipeline(::tvm::IRModule irMod, const ::tvm::Target& target)
{
    auto logger = mdl::log::getLogger("compiler");

    // Bind unresolved symbolic dims with a placeholder value so LegalizeOps
    // can see concrete shapes for the custom handlers.
    irMod = bindSymbolicVars(irMod);

    // After C++ pattern rewrites the module may contain compound expressions
    // (e.g. attention rewriter returns reshape(nn.attention(...))).
    // Normalize() flattens these into proper ANF DataflowVar bindings.
    irMod = runPass(irMod, RT::Normalize(), target);

    // Pre-legalization: remove dead code, constant-fold sqrt(64) → 8 so the
    // divide-fold handler can recognise the attention scale as a Constant.
    irMod = runSeq(irMod, {
        RT::DeadCodeElimination(),
        RT::FoldConstant(),
    }, target);
    logger->info("[mdl] Pre-legalization done");

    // LegalizeOps: CBLAS matmul, oneDNN softmax/gelu/layer_norm/attention,
    // bias fold, and scale fold all happen here via C++ TIR builders.
    irMod = runSeq(irMod, {
        RT::LegalizeOps(legalizers::mdlLegalizeMap()),
        RT::DeadCodeElimination(),
    }, target);
    logger->info("[mdl] LegalizeOps done (CBLAS + oneDNN)");

    // Op fusion.
    irMod = runSeq(irMod, {
        RT::AnnotateTIROpPattern(),
        RT::FoldConstant(),
        RT::FuseOps(),
        RT::FuseTIR(),
    }, target);
    logger->info("[mdl] FuseOps + FuseTIR done");

    // VM lowering: in-header passes.
    irMod = runSeq(irMod, {
        RT::RewriteDataflowReshape(),
        RT::ToNonDataflow(),
        RT::RemovePurityChecking(),
        RT::CallTIRRewrite(),
        RT::StaticPlanBlockMemory(),
    }, target);

    // VM lowering: passes only available via FFI registry.
    for (const char* name : {
            "relax.transform.LowerAllocTensor",
            "relax.transform.KillAfterLastUse",
            "relax.transform.LowerRuntimeBuiltin",
            "relax.transform.ComputePrimValue",
            "relax.transform.VMShapeLower",
        }) {
        irMod = callGlobalPass(irMod, name);
    }

    irMod = runPass(irMod, RT::AttachGlobalSymbol(), target);
    logger->info("[mdl] VM lowering done");

    return irMod;
}

// ── public entry point ────────────────────────────────────────────────────────

::tvm::Target parseTarget(const std::string& targetStr) {
    // targetStr: "llvm -mcpu=xxx"  →  {"kind":"llvm","mcpu":"xxx","fast-math":true}
    std::istringstream ss(targetStr);
    std::string kind;
    ss >> kind;
    std::ostringstream js;
    js << "{\"kind\":\"" << kind << "\"";
    std::string tok;
    while (ss >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string key = tok.substr(0, eq);
        while (!key.empty() && key.front() == '-') key = key.substr(1);
        js << ",\"" << key << "\":\"" << tok.substr(eq + 1) << "\"";
    }
    js << ",\"fast-math\":true}";
    return ::tvm::Target(js.str());
}

} // namespace mdl::backend::tvm

#include "./pattern_registry.hpp"
#include "../../../logging.hpp"

#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>

namespace mdl::backend::tvm {

::tvm::IRModule applyPatterns(::tvm::IRModule mod,
                              const std::vector<PatternEntry>& entries)
{
    auto logger = mdl::log::getLogger("compiler");

    for (const auto& entry : entries) {
        int nRewrites = 0;
        for (const auto& gv : mod->GetGlobalVars()) {
            auto fn_obj = mod->Lookup(gv);
            const auto* fn = fn_obj.as<::tvm::relax::FunctionNode>();
            if (!fn) continue;

            ::tvm::relax::Function func = ::tvm::Downcast<::tvm::relax::Function>(fn_obj);
            try {
                auto new_fn = ::tvm::relax::RewriteCall(entry.pattern, entry.rewriter, func);
                if (!new_fn.same_as(func)) {
                    mod->Update(gv, new_fn);
                    ++nRewrites;
                    logger->info("[mdl] {} pattern: rewrote '{}'",
                                 std::string(entry.name),
                                 std::string(gv->name_hint));
                }
            } catch (const std::exception& e) {
                logger->warn("[mdl] {} pattern rewrite failed for '{}': {}",
                             std::string(entry.name),
                             std::string(gv->name_hint),
                             e.what());
            }
        }
        logger->info("[mdl] {} pattern: {} function(s) rewritten",
                     std::string(entry.name), nRewrites);
    }

    return mod;
}

} // namespace mdl::backend::tvm

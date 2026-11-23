#include "./pattern_registry.hpp"
#include "../../../logging.hpp"

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>

#include <unordered_map>

namespace mdl::backend::tvm {

	namespace {
		thread_local const std::unordered_map<const void*, size_t>* currentUses = nullptr;

		std::unordered_map<const void*, size_t> usesIn(const ::tvm::relax::Function& f) {
			std::unordered_map<const void*, size_t> uses;
			auto count = [&](const ::tvm::relax::Expr& e) {
				for (const auto& v : ::tvm::relax::FreeVars(e))
					++uses[v.get()];
			};
			if (const auto* seq = f->body.as<::tvm::relax::SeqExprNode>()) {
				for (const auto& block : seq->blocks)
					for (const auto& b : block->bindings)
						if (const auto* vb = b.as<::tvm::relax::VarBindingNode>())
							count(vb->value);
				count(seq->body);
				// (a bound value: the uses of its variable; the matcher reports matched calls, not their variables)
				for (const auto& block : seq->blocks)
					for (const auto& b : block->bindings)
						if (const auto* vb = b.as<::tvm::relax::VarBindingNode>()) {
							auto it = uses.find(vb->var.get());
							uses[vb->value.get()] = it == uses.end() ? 0 : it->second;
						}
			}
			return uses;
		}
	} // namespace

	size_t patternUses(const ::tvm::relax::Expr& e) {
		if (currentUses == nullptr)
			return 0;
		auto it = currentUses->find(e.get());
		return it == currentUses->end() ? 0 : it->second;
	}

	::tvm::IRModule applyPatterns(::tvm::IRModule mod, const std::vector<PatternEntry>& entries) {
		auto logger = mdl::log::getLogger("compiler");

		for (const auto& entry : entries) {
			int nRewrites = 0;
			for (const auto& gv : mod->GetGlobalVars()) {
				auto fn_obj = mod->Lookup(gv);
				const auto* fn = fn_obj.as<::tvm::relax::FunctionNode>();
				if (!fn)
					continue;

				auto func = fn_obj.as_or_throw<::tvm::relax::Function>();
				try {
					auto bindings = ::tvm::relax::AnalyzeVar2Value(func);
					const auto uses = usesIn(func);
					currentUses = &uses;
					auto rewriter = ::tvm::ffi::TypedFunction<
							::tvm::relax::
									Expr(::tvm::relax::Expr,
										 ::tvm::ffi::Map<::tvm::relax::DFPattern, ::tvm::relax::Expr>)>(
							[&](::tvm::relax::Expr call, ::tvm::ffi::Map<::tvm::relax::DFPattern, ::tvm::relax::Expr> m
							) { return entry.rewriter(call, m, bindings); }
					);
					auto new_fn = ::tvm::relax::RewriteCall(entry.pattern, rewriter, func);
					currentUses = nullptr;
					if (!new_fn.same_as(func)) {
						mod->Update(gv, new_fn);
						++nRewrites;
						logger->info(
								"[mdl] {} pattern: rewrote '{}'", std::string(entry.name), std::string(gv->name_hint)
						);
					}
				} catch (const std::exception& e) {
					currentUses = nullptr;
					logger->warn(
							"[mdl] {} pattern rewrite failed for '{}': {}", std::string(entry.name),
							std::string(gv->name_hint), e.what()
					);
				}
			}
			logger->info("[mdl] {} pattern: {} function(s) rewritten", std::string(entry.name), nRewrites);
		}

		return mod;
	}

} // namespace mdl::backend::tvm

#include "./pattern_registry.hpp"
#include "../../../logging.hpp"

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace mdl::backend::tvm {

	namespace {
		thread_local const std::unordered_map<const void*, size_t>* currentUses = nullptr;
		thread_local std::unordered_map<const void*, const void*> currentVarOf; ///< bound value -> its variable

		std::unordered_map<const void*, size_t> usesIn(const ::tvm::relax::Function& f) {
			std::unordered_map<const void*, size_t> uses;
			currentVarOf.clear();
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
							currentVarOf[vb->value.get()] = vb->var.get();
						}
			}
			return uses;
		}
	} // namespace

	bool sharesIntermediates(
			const ::tvm::relax::DFPattern& root, const ::tvm::ffi::Map<::tvm::relax::DFPattern, ::tvm::relax::Expr>& m
	) {
		// The uses of each intermediate (a matched operation other than the root) inside the match: once per matched
		// operation that takes it (the uses are counted per binding). By expression: a pattern may contain the same
		// subpattern twice (e.g. `exp` in softmax), as two pattern nodes that match one expression.
		auto key = [&](const ::tvm::relax::Expr& e) -> const void* { // (a bound value: its variable)
			auto it = currentVarOf.find(e.get());
			return it == currentVarOf.end() ? e.get() : it->second;
		};
		std::unordered_map<const void*, std::unordered_set<const void*>> parentsOf;
		for (const auto& [p, e] : m) {
			const auto* call = p.as<::tvm::relax::CallPatternNode>();
			if (call == nullptr)
				continue;
			// (alternatives `a | b`: the matched one)
			std::function<void(const ::tvm::relax::DFPattern&)> arg = [&](const ::tvm::relax::DFPattern& a) {
				if (const auto* alt = a.as<::tvm::relax::OrPatternNode>()) {
					arg(alt->left);
					arg(alt->right);
				} else if (a.as<::tvm::relax::CallPatternNode>() != nullptr && m.count(a)) {
					parentsOf[key(m.at(a))].insert(key(e));
				}
			};
			for (const auto& a : call->args)
				arg(a);
		}
		for (const auto& [p, e] : m)
			if (p.as<::tvm::relax::CallPatternNode>() != nullptr && !p.same_as(root) &&
				patternUses(e) > parentsOf[key(e)].size())
				return true;
		return false;
	}

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
							[&](::tvm::relax::Expr call,
								::tvm::ffi::Map<::tvm::relax::DFPattern, ::tvm::relax::Expr> m) {
								return sharesIntermediates(entry.pattern, m) ? call : entry.rewriter(call, m, bindings);
							}
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

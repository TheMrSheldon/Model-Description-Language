#include "./contexts.hpp"

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>

#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	namespace {
		namespace relax = ::tvm::relax;
		using ::tvm::Expr;
		using ::tvm::ffi::Array;

		/** @brief The packed positions (indices into @p flags) in order. */
		std::vector<size_t> packedPositions(const std::vector<bool>& flags) {
			std::vector<size_t> r;
			for (size_t i = 0; i < flags.size(); ++i)
				if (flags[i])
					r.push_back(i);
			return r;
		}

		std::unordered_set<const ::tvm::VarNode*> usedVars(const relax::Function& f) {
			std::unordered_set<const ::tvm::VarNode*> used;
			relax::PostOrderVisit(f->body, [&](const Expr& e) {
				if (const auto* v = e.as<::tvm::VarNode>())
					used.insert(v);
			});
			return used;
		}

		template <typename T>
		void erase(std::vector<T>& xs, const std::vector<size_t>& sortedPositions) {
			for (size_t i = sortedPositions.size(); i-- > 0;)
				xs.erase(xs.begin() + static_cast<std::ptrdiff_t>(sortedPositions[i]));
		}

		/** @brief @p f without the parameters at @p positions (unused). */
		relax::Function withoutParams(const relax::Function& f, const std::vector<size_t>& positions) {
			std::vector<::tvm::Var> params(f->params.begin(), f->params.end());
			erase(params, positions);
			return relax::Function(Array<::tvm::Var>(params.begin(), params.end()), f->body, f->ret_ty, f->is_pure, f->attrs);
		}

		/** @brief @p f without the results at @p positions (of its returned tuple). */
		relax::Function withoutResults(const relax::Function& f, const std::vector<size_t>& positions) {
			const auto* seq = f->body.as<relax::SeqExprNode>();
			const auto* out = seq ? seq->body.as<::tvm::VarNode>() : nullptr;
			if (out == nullptr)
				throw std::runtime_error("context pruning: the producer does not return a variable");
			Array<relax::BindingBlock> blocks;
			::tvm::Var result;
			for (const auto& block : seq->blocks) {
				Array<relax::Binding> bindings;
				for (const auto& binding : block->bindings) {
					const auto* vb = binding.as<relax::VarBindingNode>();
					const auto* tuple = vb && vb->var.get() == out ? vb->value.as<::tvm::TupleNode>() : nullptr;
					if (tuple == nullptr) {
						bindings.push_back(binding);
						continue;
					}
					std::vector<Expr> fields(tuple->fields.begin(), tuple->fields.end());
					erase(fields, positions);
					const Expr value = fields.size() == 1 ? fields.front() : Expr(::tvm::Tuple(Array<Expr>(fields.begin(), fields.end())));
					result = ::tvm::Var(std::string(vb->var->name), relax::GetType(value));
					bindings.push_back(relax::VarBinding(result, value));
				}
				blocks.push_back(block.as<relax::DataflowBlockNode>() ? relax::BindingBlock(relax::DataflowBlock(bindings))
																	  : relax::BindingBlock(bindings));
			}
			if (!result.defined())
				throw std::runtime_error("context pruning: the producer does not return a tuple");
			return relax::Function(f->params, relax::SeqExpr(blocks, result), relax::GetType(result), f->is_pure, f->attrs);
		}
	} // namespace

	bool pruneContexts(::tvm::IRModule& mod, std::unordered_map<std::string, EntrySignature>& packing) {
		struct Context {
			std::string producer;
			std::vector<std::string> consumers;
		};
		std::map<std::string, Context> contexts;
		for (const auto& [symbol, sig] : packing) {
			if (!sig.context || sig.contextName.empty())
				continue;
			auto& c = contexts[sig.contextName];
			if (!sig.sizeSymbol.empty())
				c.producer = symbol;
			else
				c.consumers.push_back(symbol);
		}
		auto function = [&](const std::string& symbol) -> std::optional<relax::Function> {
			if (!mod->ContainGlobalVar(symbol))
				return std::nullopt;
			auto f = mod->Lookup(mod->GetGlobalVar(symbol)).as<relax::Function>();
			return f ? std::optional<relax::Function>(*f) : std::nullopt;
		};

		// One context: whether a tensor was removed.
		auto prune = [&](const Context& c) {
			auto producer = function(c.producer);
			if (!producer || c.consumers.empty())
				return false;
			const size_t numPacked = packedPositions(packing.at(c.producer).outputs).size();
			std::vector<bool> read(numPacked, false);
			std::vector<relax::Function> consumers;
			for (const auto& symbol : c.consumers) {
				auto f = function(symbol);
				if (!f)
					return false; // (a consumer that is not compiled here may read everything)
				consumers.push_back(*f);
				const auto used = usedVars(*f);
				const auto positions = packedPositions(packing.at(symbol).params);
				for (size_t k = 0; k < positions.size() && k < numPacked; ++k)
					read[k] = read[k] || used.contains((*f)->params[positions[k]].get());
			}
			std::vector<size_t> unread;
			for (size_t k = 0; k < numPacked; ++k)
				if (!read[k])
					unread.push_back(k);
			if (unread.empty())
				return false;
			// The positions of the unread tensors among a signature's parameters or outputs.
			auto select = [&](const std::vector<bool>& flags) {
				const auto positions = packedPositions(flags);
				std::vector<size_t> r;
				for (size_t k : unread)
					r.push_back(positions.at(k));
				return r;
			};
			for (size_t i = 0; i < consumers.size(); ++i) {
				auto& sig = packing.at(c.consumers[i]);
				const auto positions = select(sig.params);
				mod->Update(mod->GetGlobalVar(c.consumers[i]), withoutParams(consumers[i], positions));
				erase(sig.params, positions);
			}
			auto& sig = packing.at(c.producer);
			const auto positions = select(sig.outputs);
			mod->Update(mod->GetGlobalVar(c.producer), withoutResults(*producer, positions));
			erase(sig.outputs, positions);
			erase(sig.accumulate, positions);
			return true;
		};
		bool changed = false;
		for (const auto& [name, c] : contexts)
			changed = prune(c) || changed;
		return changed;
	}

} // namespace mdl::backend::tvm

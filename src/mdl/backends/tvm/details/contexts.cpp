#include "./contexts.hpp"
#include "./tvm_compat.hpp"

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/block_builder.h>
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
				if (sig.aliases.size() == sig.params.size())
					erase(sig.aliases, positions);
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


	namespace {
		/** @brief The bindings of @p f that use @p v (by their values), and whether the function's result uses it. */
		std::pair<std::vector<const relax::VarBindingNode*>, bool> usesOf(const relax::Function& f, const ::tvm::VarNode* v) {
			std::vector<const relax::VarBindingNode*> uses;
			const auto* seq = f->body.as<relax::SeqExprNode>();
			if (seq == nullptr)
				return {uses, true};
			for (const auto& block : seq->blocks)
				for (const auto& b : block->bindings)
					if (const auto* vb = b.as<relax::VarBindingNode>())
						for (const auto& u : relax::FreeVars(vb->value))
							if (u.get() == v) {
								uses.push_back(vb);
								break;
							}
			bool result = false;
			for (const auto& u : relax::FreeVars(seq->body))
				result |= u.get() == v;
			return {uses, result};
		}

		std::optional<std::vector<int64_t>> permuteAxes(const Expr& e) {
			static const auto& permute = ::tvm::Op::Get("relax.permute_dims");
			const auto* call = e.as<::tvm::CallNode>();
			const auto* attrs = call && call->op.same_as(permute) ? call->attrs.as<relax::PermuteDimsAttrs>() : nullptr;
			if (attrs == nullptr || !attrs->axes.has_value())
				return std::nullopt;
			std::vector<int64_t> axes;
			for (const auto& a : attrs->axes.value())
				axes.push_back(a);
			return axes;
		}
	} // namespace

	bool permuteContexts(::tvm::IRModule& mod, const std::unordered_map<std::string, EntrySignature>& packing) {
		static const auto& attention = ::tvm::Op::Get("mdl.attention");
		static const auto& attentionGrad = ::tvm::Op::Get("mdl.attention_grad");
		static const auto permuteDims = ::tvm::ffi::Function::GetGlobalRequired("relax.op.permute_dims");
		auto function = [&](const std::string& symbol) -> std::optional<relax::Function> {
			if (!mod->ContainGlobalVar(symbol))
				return std::nullopt;
			auto f = mod->Lookup(mod->GetGlobalVar(symbol)).as<relax::Function>();
			return f ? std::optional<relax::Function>(*f) : std::nullopt;
		};
		std::map<std::string, std::pair<std::string, std::vector<std::string>>> contexts; // name -> producer, consumers
		for (const auto& [symbol, sig] : packing) {
			if (!sig.context || sig.contextName.empty())
				continue;
			auto& c = contexts[sig.contextName];
			if (sig.sizeSymbol.empty())
				c.second.push_back(symbol);
			else
				c.first = symbol;
		}
		bool changed = false;
		for (const auto& [name, c] : contexts) {
			auto producer = function(c.first);
			if (!producer || c.second.empty())
				continue;
			std::vector<relax::Function> consumers;
			for (const auto& symbol : c.second)
				if (auto f = function(symbol))
					consumers.push_back(*f);
			if (consumers.size() != c.second.size())
				continue; // (a consumer that is not compiled here reads the layout as declared)
			const auto* seq = (*producer)->body.as<relax::SeqExprNode>();
			const auto* out = seq ? seq->body.as<::tvm::VarNode>() : nullptr;
			const relax::TupleNode* tuple = nullptr;
			for (const auto& block : seq ? seq->blocks : Array<relax::BindingBlock>())
				for (const auto& b : block->bindings)
					if (const auto* vb = b.as<relax::VarBindingNode>(); vb && vb->var.get() == out)
						tuple = vb->value.as<relax::TupleNode>();
			if (tuple == nullptr)
				continue;
			const auto outPositions = packedPositions(packing.at(c.first).outputs);
			std::map<size_t, std::pair<Expr, std::vector<int64_t>>> rewrites; // field -> (permuted value, axes)
			for (size_t k = 0; k < outPositions.size(); ++k) {
				const size_t field = outPositions[k];
				const auto* v = field < tuple->fields.size() ? tuple->fields[field].as<::tvm::VarNode>() : nullptr;
				if (v == nullptr)
					continue;
				// The producer: the tuple and one permutation.
				auto [uses, inResult] = usesOf(*producer, v);
				const relax::VarBindingNode* perm = nullptr;
				bool ok = !inResult;
				for (const auto* u : uses) {
					if (u->var.get() == out)
						continue;
					if (perm == nullptr && permuteAxes(u->value))
						perm = u;
					else
						ok = false;
				}
				if (!ok || perm == nullptr)
					continue;
				// The consumers: only attention operations.
				for (size_t i = 0; i < consumers.size() && ok; ++i) {
					const auto positions = packedPositions(packing.at(c.second[i]).params);
					if (k >= positions.size())
						ok = false;
					else {
						auto [cuses, cresult] = usesOf(consumers[i], consumers[i]->params[positions[k]].get());
						ok = !cresult;
						for (const auto* u : cuses) {
							const auto* call = u->value.as<::tvm::CallNode>();
							ok &= call != nullptr && (call->op.same_as(attention) || call->op.same_as(attentionGrad) ||
													  permuteAxes(u->value).has_value());
						}
					}
				}
				if (ok)
					rewrites.emplace(field, std::make_pair(Expr(perm->var), *permuteAxes(perm->value)));
			}
			if (rewrites.empty())
				continue;
			// The producer returns the permuted values.
			{
				Array<relax::BindingBlock> blocks;
				::tvm::Var result;
				for (const auto& block : seq->blocks) {
					Array<relax::Binding> bindings;
					for (const auto& b : block->bindings) {
						const auto* vb = b.as<relax::VarBindingNode>();
						if (vb == nullptr || vb->var.get() != out) {
							bindings.push_back(b);
							continue;
						}
						Array<Expr> fields = tuple->fields;
						for (const auto& [field, r] : rewrites)
							fields.Set(field, r.first);
						const Expr value = relax::Tuple(fields);
						result = ::tvm::Var(std::string(vb->var->name), relax::GetType(value));
						bindings.push_back(relax::VarBinding(result, value));
					}
					blocks.push_back(block.as<relax::DataflowBlockNode>() ? relax::BindingBlock(relax::DataflowBlock(bindings))
																		  : relax::BindingBlock(bindings));
				}
				mod->Update(
						mod->GetGlobalVar(c.first),
						relax::Function((*producer)->params, relax::SeqExpr(blocks, result), relax::GetType(result),
										(*producer)->is_pure, (*producer)->attrs)
				);
			}
			// The consumers read them through the inverse permutations.
			for (size_t i = 0; i < consumers.size(); ++i) {
				const auto& f = consumers[i];
				const auto positions = packedPositions(packing.at(c.second[i]).params);
				auto params = f->params;
				Array<relax::Binding> unpermute;
				std::unordered_map<const void*, std::pair<::tvm::Var, std::vector<int64_t>>> permutedParams; // q -> (q', inverse)
				auto bb = relax::BlockBuilder::Create(mod);
				for (size_t k = 0; k < outPositions.size(); ++k) {
					auto it = rewrites.find(outPositions[k]);
					if (it == rewrites.end())
						continue;
					const ::tvm::Var& q = f->params[positions[k]];
					const auto* info = tensorInfo(q);
					const auto& axes = it->second.second;
					Array<::tvm::PrimExpr> shape;
					const auto original = info->GetShape().value();
					for (int64_t a : axes)
						shape.push_back(original[a]);
					const ::tvm::Var permuted(std::string(q->name), relax::TensorType(relax::ShapeExpr(shape), info->dtype));
					params.Set(positions[k], permuted);
					std::vector<int64_t> inv(axes.size());
					Array<int64_t> inverse;
					for (size_t d = 0; d < axes.size(); ++d)
						inv[axes[d]] = static_cast<int64_t>(d);
					for (int64_t a : inv)
						inverse.push_back(a);
					unpermute.push_back(relax::VarBinding(q, bb->Normalize(permuteDims(permuted, inverse).cast<Expr>())));
					permutedParams.emplace(q.get(), std::make_pair(permuted, inv));
				}
				// Permutations of them compose with the inverse (e.g. the same permutation as in the producer: the saved
				// tensor itself).
				const auto* fseq = f->body.as<relax::SeqExprNode>();
				Array<relax::BindingBlock> blocks{relax::BindingBlock(unpermute)};
				for (const auto& block : fseq->blocks) {
					Array<relax::Binding> bindings;
					for (const auto& b : block->bindings) {
						const auto* vb = b.as<relax::VarBindingNode>();
						const auto* call = vb ? vb->value.as<::tvm::CallNode>() : nullptr;
						auto axes = call ? permuteAxes(vb->value) : std::nullopt;
						auto it = axes ? permutedParams.find(call->args[0].get()) : permutedParams.end();
						if (it == permutedParams.end()) {
							bindings.push_back(b);
							continue;
						}
						const auto& [permuted, inverse] = it->second;
						Array<int64_t> composed;
						bool identity = true;
						for (size_t d = 0; d < axes->size(); ++d) {
							const int64_t a = inverse[(*axes)[d]];
							identity &= a == static_cast<int64_t>(d);
							composed.push_back(a);
						}
						const Expr value = identity ? Expr(permuted) : bb->Normalize(permuteDims(permuted, composed).cast<Expr>());
						bindings.push_back(relax::VarBinding(vb->var, value));
					}
					blocks.push_back(block.as<relax::DataflowBlockNode>() ? relax::BindingBlock(relax::DataflowBlock(bindings))
																		  : relax::BindingBlock(bindings));
				}
				mod->Update(
						mod->GetGlobalVar(c.second[i]),
						relax::Function(params, relax::SeqExpr(blocks, fseq->body), f->ret_ty, f->is_pure, f->attrs)
				);
			}
			changed = true;
		}
		return changed;
	}

} // namespace mdl::backend::tvm

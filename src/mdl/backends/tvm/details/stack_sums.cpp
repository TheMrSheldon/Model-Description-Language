#include "./stack_sums.hpp"

#include "../../../logging.hpp"
#include "./tvm_compat.hpp"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/tensor.h>

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	namespace {
		namespace ffi = ::tvm::ffi;
		namespace relax = ::tvm::relax;
		using ::tvm::Expr;

		constexpr size_t kMaxGroup = 4;

		/** @brief Elementwise operators whose result FuseOps fuses into a following reduction. */
		bool elementwise(const ::tvm::CallNode* call) {
			static const std::unordered_set<std::string> ops{
					"relax.add", "relax.subtract", "relax.multiply", "relax.divide", "relax.negative", "relax.astype",
			};
			const auto* op = call ? call->op.as<::tvm::OpNode>() : nullptr;
			return op != nullptr && ops.contains(std::string(op->name));
		}

		class Stacker {
		public:
			explicit Stacker(::tvm::IRModule mod) : mod(std::move(mod)) {}

			::tvm::IRModule run() {
				std::vector<std::pair<::tvm::GlobalVar, relax::Function>> updates;
				for (const auto& [gv, f] : mod->functions)
					if (const auto* fn = f.as<relax::FunctionNode>();
						fn && !fn->HasNonzeroAttr(relax::attr::kPrimitive))
						if (auto changed = rewrite(ffi::GetRef<relax::Function>(fn)))
							updates.emplace_back(gv, *changed);
				for (const auto& [gv, f] : updates)
					mod->Update(gv, f);
				mdl::log::getLogger("compiler")->info("Stacked {} group(s) of sibling sums", groups);
				return mod;
			}

		private:
			::tvm::IRModule mod;
			size_t groups = 0;

			std::optional<relax::Function> rewrite(const relax::Function& f) {
				const auto* seq = f->body.as<relax::SeqExprNode>();
				if (seq == nullptr)
					return std::nullopt;
				std::unordered_map<const void*, Expr> valueOf;
				for (const auto& block : seq->blocks)
					for (const auto& b : block->bindings)
						if (const auto* vb = b.as<relax::VarBindingNode>())
							valueOf.emplace(vb->var.get(), vb->value);
				bool changed = false;
				ffi::Array<relax::BindingBlock> blocks;
				for (const auto& block : seq->blocks) {
					auto r = rewrite(block, valueOf);
					changed |= r.has_value();
					blocks.push_back(r.value_or(block));
				}
				if (!changed)
					return std::nullopt;
				return relax::Function(f->params, relax::SeqExpr(blocks, seq->body), f->ret_ty, f->is_pure, f->attrs);
			}

			std::optional<relax::BindingBlock>
			rewrite(const relax::BindingBlock& block, const std::unordered_map<const void*, Expr>& valueOf) {
				static const auto& collapse = ::tvm::Op::Get("relax.collapse_sum_to");
				const auto& bindings = block->bindings;
				// The sums and the tensors their inputs are computed from (the input, and the arguments of its elementwise
				// producer).
				struct Sum {
					size_t index;
					Expr input, shape;
					std::unordered_set<const void*> sources;
				};
				std::vector<Sum> sums;
				for (size_t i = 0; i < bindings.size(); ++i) {
					const auto* vb = bindings[i].as<relax::VarBindingNode>();
					const auto* call = vb ? vb->value.as<::tvm::CallNode>() : nullptr;
					if (call == nullptr || !call->op.same_as(collapse) || !call->args[0].as<::tvm::VarNode>() ||
						!knownShape(call->args[0]) || !call->args[1].as<relax::ShapeExprNode>())
						continue;
					Sum s{i, call->args[0], call->args[1], {call->args[0].get()}};
					if (auto it = valueOf.find(call->args[0].get()); it != valueOf.end())
						if (const auto* producer = it->second.as<::tvm::CallNode>(); elementwise(producer))
							for (const auto& a : producer->args)
								if (a.as<::tvm::VarNode>())
									s.sources.insert(a.get());
					sums.push_back(std::move(s));
				}
				// Uses of each variable, by binding.
				auto uses = [&](size_t j, const void* v) {
					const auto* vb = bindings[j].as<relax::VarBindingNode>();
					for (const auto& u : relax::FreeVars(vb->value))
						if (u.get() == v)
							return true;
					return false;
				};
				std::vector<bool> taken(sums.size(), false);
				std::map<size_t, std::vector<size_t>> groupAt; // last binding -> sums
				for (size_t a = 0; a < sums.size(); ++a) {
					if (taken[a])
						continue;
					std::vector<size_t> group{a};
					const auto shapeA = *knownShape(sums[a].input);
					const auto dtypeA = tensorInfo(sums[a].input)->dtype;
					for (size_t b = a + 1; b < sums.size() && group.size() < kMaxGroup; ++b) {
						if (taken[b] || !ffi::StructuralEqual()(*knownShape(sums[b].input), shapeA) ||
							!ffi::StructuralEqual()(sums[b].shape, sums[a].shape) ||
							tensorInfo(sums[b].input)->dtype != dtypeA)
							continue;
						bool shares = false;
						for (const void* s : sums[b].sources)
							shares |= sums[a].sources.contains(s);
						if (!shares)
							continue;
						// (the results are computed at the last sum; the bindings in between that use them follow it, so
						// the later sums' inputs must not depend on them)
						std::unordered_set<const void*> delayed;
						for (size_t m : group)
							delayed.insert(bindings[sums[m].index]->var.get());
						bool ok = true;
						for (size_t j = sums[a].index + 1; j < sums[b].index; ++j)
							for (const void* v : std::vector<const void*>(delayed.begin(), delayed.end()))
								if (uses(j, v)) {
									delayed.insert(bindings[j]->var.get());
									break;
								}
						ok = !delayed.contains(sums[b].input.get());
						if (ok)
							group.push_back(b);
					}
					if (group.size() < 2)
						continue;
					for (size_t m : group)
						taken[m] = true;
					groupAt.emplace(sums[group.back()].index, std::move(group));
				}
				if (groupAt.empty())
					return std::nullopt;

				static const auto stack = ffi::Function::GetGlobalRequired("relax.op.stack");
				static const auto collapseSumTo = ffi::Function::GetGlobalRequired("relax.op.collapse_sum_to");
				static const auto take = ffi::Function::GetGlobalRequired("relax.op.take");
				std::unordered_set<size_t> removed;
				for (const auto& [last, group] : groupAt)
					for (size_t m : group)
						removed.insert(sums[m].index);
				auto bb = relax::BlockBuilder::Create(mod);
				ffi::Array<relax::Binding> result;
				// Bindings that use a result of a group before its last sum follow the group.
				std::unordered_set<const void*> pending; // results of groups not computed yet, and their uses
				std::vector<relax::Binding> delayedBindings;
				std::unordered_set<const void*> emitted; // pending variables that are computed now
				for (size_t i = 0; i < bindings.size(); ++i) {
					auto it = groupAt.find(i);
					if (it == groupAt.end()) {
						if (removed.contains(i)) {
							pending.insert(bindings[i]->var.get());
							continue;
						}
						bool waits = false;
						for (const auto& u : relax::FreeVars(bindings[i].as_or_throw<relax::VarBinding>()->value))
							waits |= pending.contains(u.get());
						if (waits) {
							pending.insert(bindings[i]->var.get());
							delayedBindings.push_back(bindings[i]);
						} else {
							result.push_back(bindings[i]);
						}
						continue;
					}
					for (size_t m : it->second)
						if (pending.contains(sums[m].input.get()))
							return std::nullopt; // (cannot happen: checked when grouping)
					const auto& group = it->second;
					ffi::Array<Expr> inputs;
					for (size_t m : group)
						inputs.push_back(sums[m].input);
					auto shape = sums[group.front()].shape.as<relax::ShapeExprNode>()->values;
					shape.push_back(::tvm::IntImm(::tvm::PrimType::Int(64), static_cast<int64_t>(group.size())));
					const auto name = std::string(bindings[i]->var->name);
					const bool dataflow = block.as<relax::DataflowBlockNode>() != nullptr;
					auto var = [&](const std::string& n, const Expr& value) {
						return dataflow ? ::tvm::Var(relax::DataflowVar(n, relax::GetType(value)))
										: ::tvm::Var(n, relax::GetType(value));
					};
					const Expr stackedValue =
							bb->Normalize(stack(relax::Tuple(inputs), ffi::Optional<int64_t>(-1)).cast<Expr>());
					const ::tvm::Var stacked = var(name + "_stacked", stackedValue);
					result.push_back(relax::VarBinding(stacked, stackedValue));
					const Expr summed = bb->Normalize(collapseSumTo(stacked, relax::ShapeExpr(shape)).cast<Expr>());
					const ::tvm::Var sum = var(name + "_sums", summed);
					result.push_back(relax::VarBinding(sum, summed));
					for (size_t k = 0; k < group.size(); ++k) {
						auto index = ::tvm::runtime::Tensor::Empty(
								ffi::Shape{}, ::tvm::PrimType::Int(32)->dtype, DLDevice{kDLCPU, 0}
						);
						const int32_t value = static_cast<int32_t>(k); // (AOT: int32 constants)
						index.CopyFromBytes(&value, sizeof(value));
						const Expr taken = bb->Normalize(take(sum, relax::MakeTensorConst(index),
															  ffi::Optional<int64_t>(-1), ffi::String("fast"))
																 .cast<Expr>());
						result.push_back(relax::VarBinding(bindings[sums[group[k]].index]->var, taken));
					}
					++groups;
					// (delayed bindings of this group: their other pending inputs are computed by now unless they wait for
					// a later group)
					std::vector<relax::Binding> still;
					for (const auto& b : delayedBindings) {
						bool waits = false;
						for (const auto& u : relax::FreeVars(b.as_or_throw<relax::VarBinding>()->value))
							waits |= pending.contains(u.get()) && !emitted.contains(u.get());
						if (waits) {
							still.push_back(b);
						} else {
							result.push_back(b);
							emitted.insert(b->var.get());
						}
					}
					delayedBindings = std::move(still);
					for (size_t m : group)
						emitted.insert(bindings[sums[m].index]->var.get());
				}
				for (const auto& b : delayedBindings)
					result.push_back(b);
				return block.as<relax::DataflowBlockNode>() ? relax::BindingBlock(relax::DataflowBlock(result))
															: relax::BindingBlock(result);
			}
		};
	} // namespace

	::tvm::transform::Pass stackSiblingSums() {
		return ::tvm::transform::CreateModulePass(
				[](::tvm::IRModule mod, ::tvm::transform::PassContext) { return Stacker(std::move(mod)).run(); }, 0,
				"mdl.StackSiblingSums", {}
		);
	}

} // namespace mdl::backend::tvm

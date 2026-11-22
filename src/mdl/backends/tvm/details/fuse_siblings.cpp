#include "./fuse_siblings.hpp"

#include "../../../logging.hpp"
#include "./tvm_compat.hpp"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <exception>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	namespace {
		namespace ffi = ::tvm::ffi;
		namespace tirx = ::tvm::tirx;
		namespace s_tir = ::tvm::s_tir;
		namespace relax = ::tvm::relax;
		using ::tvm::Expr;
		using ::tvm::PrimExpr;

		constexpr size_t kMaxGroup = 8;

		/**
		 * @brief An injective kernel as one perfect loop nest around one block that writes its single result at the
		 * block's iterators (the producers of fused kernels inlined).
		 */
		struct Injective {
			tirx::PrimFunc func;			  ///< (inlined)
			std::vector<tirx::For> loops;	  ///< outermost first
			s_tir::SBlockRealize realize;	  ///< the block
			s_tir::SBlock root;
			s_tir::SBlockRealize rootRealize;
		};

		std::optional<Injective> injective(const tirx::PrimFunc& original, size_t numInputs) {
			if (original->params.size() != numInputs + 1)
				return std::nullopt; // (scalar parameters or several results)
			const auto* realize = original->body.as<s_tir::SBlockRealizeNode>();
			if (realize == nullptr)
				return std::nullopt;
			s_tir::Schedule sch = s_tir::Schedule::Concrete(
					::tvm::IRModule({{::tvm::GlobalVar("main"), original}}), /*seed=*/0, /*debug_mask=*/0,
					s_tir::ScheduleErrorRenderLevel::kNone
			);
			const s_tir::SBlockRV rootRV = sch->GetSBlock(realize->block->name_hint, ffi::String("main"));
			for (bool progress = true; progress;) {
				progress = false;
				const auto children = sch->GetChildBlocks(rootRV);
				if (children.size() <= 1)
					break;
				for (const auto& block : children) {
					try {
						sch->ComputeInline(block);
						progress = true;
						break;
					} catch (const std::exception&) {
					}
				}
			}
			if (sch->GetChildBlocks(rootRV).size() != 1)
				return std::nullopt;
			Injective k;
			k.func = sch->mod()->Lookup("main").as_or_throw<tirx::PrimFunc>();
			k.rootRealize = ffi::GetRef<s_tir::SBlockRealize>(k.func->body.as<s_tir::SBlockRealizeNode>());
			k.root = k.rootRealize->block;
			if (!k.root->alloc_buffers.empty() || !k.root->match_buffers.empty())
				return std::nullopt;
			tirx::Stmt s = k.root->body;
			while (const auto* f = s.as<tirx::ForNode>()) {
				if (f->kind != tirx::ForKind::kSerial || !::tvm::prim::is_zero(f->min))
					return std::nullopt;
				k.loops.push_back(ffi::GetRef<tirx::For>(f));
				s = f->body;
			}
			const auto* inner = s.as<s_tir::SBlockRealizeNode>();
			if (inner == nullptr)
				return std::nullopt;
			k.realize = ffi::GetRef<s_tir::SBlockRealize>(inner);
			const auto& block = inner->block;
			if (block->init.has_value() || !block->alloc_buffers.empty() || !block->match_buffers.empty() ||
				block->iter_vars.empty() || block->iter_vars.size() != k.loops.size() || !::tvm::prim::is_one(inner->predicate))
				return std::nullopt;
			const auto* store = block->body.as<tirx::BufferStoreNode>();
			const auto& out = k.func->params.back();
			if (store == nullptr || store->buffer.get() != out.get() || store->indices.size() != block->iter_vars.size())
				return std::nullopt;
			const auto outBuffer = out.as_or_throw<tirx::BufferVar>();
			::tvm::sym::Analyzer analyzer;
			for (size_t d = 0; d < block->iter_vars.size(); ++d) {
				const auto& iv = block->iter_vars[d];
				if (iv->iter_type != tirx::kDataPar || !::tvm::prim::is_zero(iv->dom->min) ||
					!inner->iter_values[d].same_as(k.loops[d]->loop_var) || !store->indices[d].same_as(iv->var) ||
					!analyzer->CanProveEqual(iv->dom->extent, k.loops[d]->extent) ||
					!analyzer->CanProveEqual(iv->dom->extent, outBuffer->shape[d]))
					return std::nullopt;
			}
			for (const auto& r : block->reads)
				for (const auto& range : r->region)
					if (!::tvm::prim::is_one(range->extent))
						return std::nullopt;
			return k;
		}

		/** @brief @p buffer with the shape @p shape (a new buffer over new memory, bound by a parameter). */
		tirx::BufferVar reshaped(const tirx::BufferVar& buffer, const ffi::Array<PrimExpr>& shape) {
			return tirx::BufferVar(
					buffer.name(),
					tirx::BufferType(
							buffer->storage_scope, buffer->dtype, shape, buffer->strides, buffer->elem_offset,
							buffer->data_alignment, buffer->offset_factor
					)
			);
		}

		/**
		 * @brief One kernel computing the results of @p kernels (same iteration space) in the loops of the first, with
		 * the arguments @p args of each kernel's call: an argument passed to several kernels is one parameter, and the
		 * symbolic dimensions of the later kernels are those of the first (they have results of the same shape), so
		 * that every buffer is written in terms of the same variables.
		 * @return The kernel and its arguments.
		 */
		std::pair<tirx::PrimFunc, ffi::Array<Expr>>
		merge(const std::vector<Injective>& kernels, const std::vector<ffi::Array<Expr>>& args) {
			const Injective& base = kernels.front();
			const auto& baseBlock = base.realize->block;
			const auto baseOut = base.func->params.back().as_or_throw<tirx::BufferVar>();
			ffi::Array<tirx::Stmt> stores;
			ffi::Array<::tvm::TensorRegion> reads, writes;
			ffi::Array<tirx::Var> inputs, outputs;
			ffi::Array<Expr> mergedArgs;
			std::unordered_map<const void*, tirx::BufferVar> paramOfArg; // argument -> parameter
			for (size_t m = 0; m < kernels.size(); ++m) {
				const Injective& k = kernels[m];
				const auto& block = k.realize->block;
				std::unordered_map<const void*, PrimExpr> vars;
				for (size_t d = 0; d < block->iter_vars.size(); ++d)
					vars.emplace(block->iter_vars[d]->var.get(), baseBlock->iter_vars[d]->var);
				const auto out = k.func->params.back().as_or_throw<tirx::BufferVar>();
				if (m > 0)
					for (size_t d = 0; d < out->shape.size(); ++d)
						if (out->shape[d].as<::tvm::VarNode>() != nullptr)
							vars.emplace(out->shape[d].get(), baseOut->shape[d]);
				auto subst = [&](const PrimExpr& e) {
					return tirx::SubstituteWithDataTypeLegalization(e, [&](const ::tvm::Var& v) -> ffi::Optional<PrimExpr> {
						auto it = vars.find(v.get());
						return it == vars.end() ? ffi::Optional<PrimExpr>() : ffi::Optional<PrimExpr>(it->second);
					});
				};
				// The parameters: shared, reshaped (the first kernel's variables) or unchanged.
				std::unordered_map<const void*, tirx::BufferVar> buffers; // the kernel's parameter -> the merged one
				for (size_t i = 0; i < k.func->params.size(); ++i) {
					const auto buffer = k.func->params[i].as_or_throw<tirx::BufferVar>();
					const bool output = i + 1 == k.func->params.size();
					if (!output) {
						const Expr& arg = args[m][i];
						if (auto it = paramOfArg.find(arg.get()); it != paramOfArg.end() && arg.as<::tvm::VarNode>()) {
							buffers.emplace(buffer.get(), it->second);
							continue;
						}
					}
					ffi::Array<PrimExpr> shape;
					bool changed = false;
					for (const auto& d : buffer->shape) {
						shape.push_back(subst(d));
						changed |= !shape.back().same_as(d);
					}
					const auto param = changed ? reshaped(buffer, shape) : buffer;
					buffers.emplace(buffer.get(), param);
					if (output) {
						outputs.push_back(param);
					} else {
						inputs.push_back(param);
						mergedArgs.push_back(args[m][i]);
						paramOfArg.emplace(args[m][i].get(), param);
					}
				}
				auto bufferOf = [&](const tirx::BufferVar& b) {
					auto it = buffers.find(b.get());
					return it == buffers.end() ? b : it->second;
				};
				auto rewrite = [&](const PrimExpr& e) {
					const PrimExpr loads = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
												   e,
												   [&](const ::tvm::TensorLoad& load) -> ffi::UnchangedOr<ffi::Any> {
													   auto buffer = load->source.as<tirx::BufferVar>();
													   if (!buffer)
														   return ffi::Unchanged();
													   const auto to = bufferOf(*buffer);
													   if (to.same_as(*buffer))
														   return ffi::Unchanged();
													   return ffi::Any(tirx::BufferLoad(to, load->indices));
												   }
	                                       )
												   .cast<PrimExpr>();
					return subst(loads);
				};
				auto points = [&](const ffi::Array<::tvm::TensorRegion>& regions, ffi::Array<::tvm::TensorRegion>& into) {
					for (const auto& r : regions) {
						ffi::Array<PrimExpr> indices;
						for (const auto& range : r->region)
							indices.push_back(subst(range->min));
						into.push_back(tirx::BufferRegionFromPoint(bufferOf(r->source.as_or_throw<tirx::BufferVar>()), indices));
					}
				};
				points(block->reads, reads);
				points(block->writes, writes);
				const auto* store = block->body.as<tirx::BufferStoreNode>();
				ffi::Array<PrimExpr> indices;
				for (const auto& i : store->indices)
					indices.push_back(subst(i));
				stores.push_back(tirx::BufferStore(bufferOf(store->buffer), rewrite(store->value), indices));
			}
			auto block = baseBlock;
			{
				auto* b = block.CopyOnWrite();
				b->body = tirx::SeqStmt(stores);
				b->reads = reads;
				b->writes = writes;
			}
			auto realize = base.realize;
			realize.CopyOnWrite()->block = block;
			tirx::Stmt body = realize;
			for (size_t i = base.loops.size(); i-- > 0;) {
				auto loop = base.loops[i];
				loop.CopyOnWrite()->body = body;
				body = loop;
			}
			auto root = base.root;
			root.CopyOnWrite()->body = body;
			auto rootRealize = base.rootRealize;
			rootRealize.CopyOnWrite()->block = root;
			ffi::Array<tirx::Var> params = inputs;
			for (const auto& o : outputs)
				params.push_back(o);
			return {tirx::PrimFunc(params, rootRealize, base.func->ret_type, base.func->attrs), mergedArgs};
		}

		/** @brief The variables among the arguments of a call_tir. */
		std::vector<const void*> argumentVars(const ::tvm::Call& call) {
			std::vector<const void*> vars;
			for (const auto& a : call->args[1].as_or_throw<relax::Tuple>()->fields)
				if (a.as<::tvm::VarNode>() != nullptr)
					vars.push_back(a.get());
			return vars;
		}

		class SiblingFuser {
		public:
			explicit SiblingFuser(::tvm::IRModule mod) : mod(mod), builder(relax::BlockBuilder::Create(mod)) {}

			::tvm::IRModule run() {
				std::vector<std::pair<::tvm::GlobalVar, relax::Function>> functions, updates;
				for (const auto& [gv, f] : mod->functions) // (copied: the builder adds to the module)
					if (const auto* fn = f.as<relax::FunctionNode>())
						functions.emplace_back(gv, ffi::GetRef<relax::Function>(fn));
				for (const auto& [gv, f] : functions)
					if (auto changed = rewrite(f))
						updates.emplace_back(gv, *changed);
				mod = builder->GetContextIRModule(); // (with the merged kernels)
				for (const auto& [gv, f] : updates)
					mod->Update(gv, f);
				mdl::log::getLogger("compiler")->info("Fused {} group(s) of sibling kernels", groups);
				return mod;
			}

		private:
			::tvm::IRModule mod;
			relax::BlockBuilder builder;
			std::map<const void*, std::optional<Injective>> cache; // by callee
			size_t groups = 0;

			struct Site {
				::tvm::Call call;
				const Injective* kernel;
				std::string name;
			};

			std::optional<Site> site(const relax::Binding& binding) {
				static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
				const auto* vb = binding.as<relax::VarBindingNode>();
				const auto* call = vb ? vb->value.as<::tvm::CallNode>() : nullptr;
				if (call == nullptr || !call->op.same_as(callTir) || call->args.size() != 2)
					return std::nullopt;
				const auto* gv = call->args[0].as<::tvm::GlobalVarNode>();
				const auto* args = call->args[1].as<relax::TupleNode>();
				if (gv == nullptr || args == nullptr || tensorInfo(vb->var) == nullptr || !knownShape(vb->var))
					return std::nullopt;
				auto it = cache.find(gv);
				if (it == cache.end()) {
					std::optional<Injective> k;
					if (const auto* prim = mod->Lookup(ffi::GetRef<::tvm::GlobalVar>(gv)).as<tirx::PrimFuncNode>();
						prim != nullptr && !prim->GetAttr<::tvm::IntImm>("tirx.is_scheduled").has_value()) {
						try {
							k = injective(ffi::GetRef<tirx::PrimFunc>(prim), args->fields.size());
						} catch (const std::exception&) {
						}
					}
					it = cache.emplace(gv, std::move(k)).first;
				}
				if (!it->second)
					return std::nullopt;
				return Site{ffi::GetRef<::tvm::Call>(call), &*it->second, std::string(gv->name_hint)};
			}

			std::optional<relax::Function> rewrite(const relax::Function& f) {
				const auto* seq = f->body.as<relax::SeqExprNode>();
				if (seq == nullptr)
					return std::nullopt;
				bool changed = false;
				ffi::Array<relax::BindingBlock> blocks;
				for (const auto& block : seq->blocks) {
					auto rewritten = rewrite(block);
					changed |= rewritten.has_value();
					blocks.push_back(rewritten.value_or(block));
				}
				if (!changed)
					return std::nullopt;
				return relax::Function(f->params, relax::SeqExpr(blocks, seq->body), f->ret_ty, f->is_pure, f->attrs);
			}

			std::optional<relax::BindingBlock> rewrite(const relax::BindingBlock& block) {
				const auto& bindings = block->bindings;
				std::unordered_map<const void*, size_t> definedAt;
				for (size_t i = 0; i < bindings.size(); ++i)
					definedAt.emplace(bindings[i]->var.get(), i);
				std::vector<std::optional<Site>> sites;
				for (const auto& b : bindings)
					sites.push_back(site(b));

				std::vector<bool> taken(bindings.size(), false);
				std::map<size_t, std::vector<size_t>> groupAt; // first member -> members
				for (size_t i = 0; i < bindings.size(); ++i) {
					if (!sites[i] || taken[i])
						continue;
					const auto shape = *knownShape(bindings[i]->var);
					std::vector<size_t> group{i};
					std::unordered_set<const void*> inputs;
					for (const void* v : argumentVars(sites[i]->call))
						inputs.insert(v);
					for (size_t j = i + 1; j < bindings.size() && group.size() < kMaxGroup; ++j) {
						if (!sites[j] || taken[j] ||
							!ffi::StructuralEqual()(*knownShape(bindings[j]->var), shape))
							continue;
						bool shares = false, available = true;
						for (const void* v : argumentVars(sites[j]->call)) {
							shares |= inputs.contains(v);
							auto it = definedAt.find(v);
							available &= it == definedAt.end() || it->second < i; // (computed before the group)
						}
						if (!shares || !available)
							continue;
						group.push_back(j);
						for (const void* v : argumentVars(sites[j]->call))
							inputs.insert(v);
					}
					if (group.size() < 2)
						continue;
					for (size_t m : group)
						taken[m] = true;
					groupAt.emplace(i, std::move(group));
				}
				if (groupAt.empty())
					return std::nullopt;

				static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
				std::vector<bool> moved(bindings.size(), false);
				ffi::Array<relax::Binding> result;
				for (size_t i = 0; i < bindings.size(); ++i) {
					auto it = groupAt.find(i);
					if (it == groupAt.end()) {
						if (!moved[i])
							result.push_back(bindings[i]);
						continue;
					}
					std::vector<Injective> kernels;
					std::vector<ffi::Array<Expr>> args;
					ffi::Array<::tvm::Type> outs;
					std::string name = "fused_siblings";
					for (size_t m : it->second) {
						kernels.push_back(*sites[m]->kernel);
						args.push_back(sites[m]->call->args[1].as_or_throw<relax::Tuple>()->fields);
						outs.push_back(relax::GetType(bindings[m]->var));
						if (name.size() < 64)
							name += "_" + sites[m]->name;
						moved[m] = true;
					}
					auto [merged, mergedArgs] = merge(kernels, args);
					const ::tvm::GlobalVar gv = builder->AddFunction(merged, name); // (unique, typed)
					const std::string unique = gv->name_hint;
					const ::tvm::TupleType tupleType(outs);
					const Expr call = ::tvm::Call(
							::tvm::Type::Missing(), callTir, {gv, relax::Tuple(mergedArgs)}, ::tvm::Attrs(), {tupleType}
					);
					const bool dataflow = block.as<relax::DataflowBlockNode>() != nullptr;
					const ::tvm::Var tuple = dataflow ? ::tvm::Var(relax::DataflowVar(unique, tupleType))
													  : ::tvm::Var(unique, tupleType);
					result.push_back(relax::VarBinding(tuple, call));
					for (size_t k = 0; k < it->second.size(); ++k)
						result.push_back(relax::VarBinding(
								bindings[it->second[k]]->var, ::tvm::TupleGetItem(tuple, static_cast<int>(k))
						));
					++groups;
				}
				return block.as<relax::DataflowBlockNode>() ? relax::BindingBlock(relax::DataflowBlock(result))
															: relax::BindingBlock(result);
			}

		};
	} // namespace

	::tvm::transform::Pass fuseSiblings() {
		return ::tvm::transform::CreateModulePass(
				[](::tvm::IRModule mod, ::tvm::transform::PassContext) { return SiblingFuser(std::move(mod)).run(); }, 0,
				"mdl.FuseSiblings", {}
		);
	}

} // namespace mdl::backend::tvm

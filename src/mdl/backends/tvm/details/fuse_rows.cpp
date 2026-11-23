#include "./fuse_rows.hpp"

#include "../../../logging.hpp"
#include "./tvm_compat.hpp"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/attrs.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>
#include <tvm/sym/analyzer.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
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
		using ::tvm::PrimExpr;
		using Shape = ffi::Array<PrimExpr>;

		constexpr int kInjective = 2;  // (TVM's OpPatternKind)
		constexpr int kCommReduce = 3;
		constexpr size_t kMaxGroup = 16;

		/** @brief @p shape without its unit dimensions. */
		Shape withoutUnits(const Shape& shape) {
			Shape result;
			for (const auto& d : shape)
				if (!::tvm::prim::is_one(d))
					result.push_back(d);
			return result;
		}

		bool equal(const Shape& a, const Shape& b) { return ffi::StructuralEqual()(a, b); }

		Shape concat(const Shape& a, const Shape& b) {
			Shape result = a;
			for (const auto& d : b)
				result.push_back(d);
			return result;
		}

		/** @brief A fused call: of a PrimFunc (call_tir) or of a primitive Relax function (a group of FuseOps). */
		struct Site {
			int pattern;			   ///< the maximum OpPatternKind of its kernels
			Shape shape;			   ///< of its result
			std::vector<Expr> args;	   ///< tensor arguments
			std::vector<Shape> shapes; ///< of the arguments (empty: unknown)
			bool unitReshape = false;  ///< a reshape that only inserts or removes unit dimensions
		};

		/**
		 * @brief The kernel of a reshape from @p in to @p out that only inserts or removes unit dimensions: out[i...] =
		 * in[the same indices of the other dimensions] (TVM's reshape divides by symbolic sizes; this one is
		 * elementwise, so that it inlines into its consumers).
		 */
		::tvm::tirx::PrimFunc unitReshape(const Shape& in, const Shape& out, ::tvm::PrimType dtype) {
			namespace tirx = ::tvm::tirx;
			namespace s_tir = ::tvm::s_tir;
			const auto i64 = ::tvm::PrimType::Int(64);
			std::vector<std::pair<PrimExpr, ::tvm::PrimVar>> dims; // symbolic dimensions -> the kernel's variables
			auto dim = [&](const PrimExpr& d) -> PrimExpr {
				if (d.as<::tvm::IntImmNode>())
					return d;
				for (const auto& [e, v] : dims)
					if (ffi::StructuralEqual()(e, d))
						return v;
				const ::tvm::PrimVar v(d.as<::tvm::VarNode>() ? std::string(d.as<::tvm::VarNode>()->name) : "n", d.ty());
				dims.emplace_back(d, v);
				return v;
			};
			Shape inShape, outShape;
			for (const auto& d : in)
				inShape.push_back(dim(d));
			for (const auto& d : out)
				outShape.push_back(dim(d));
			const auto A = tirx::decl_buffer(inShape, dtype, "x");
			const auto B = tirx::decl_buffer(outShape, dtype, "y");
			ffi::Array<tirx::IterVar> ivs;
			ffi::Array<PrimExpr> loopVars, outIdx, inIdx;
			std::vector<::tvm::PrimVar> loops;
			std::vector<PrimExpr> nonUnit;
			for (size_t d = 0; d < outShape.size(); ++d) {
				const ::tvm::PrimVar l("i" + std::to_string(d), i64), v("v" + std::to_string(d), i64);
				loops.push_back(l);
				loopVars.push_back(l);
				ivs.push_back(tirx::IterVar(::tvm::Range::FromMinExtent(::tvm::IntImm(i64, 0), ::tvm::prim::cast(i64, outShape[d])), v, tirx::kDataPar));
				outIdx.push_back(v);
				if (!::tvm::prim::is_one(outShape[d]))
					nonUnit.push_back(v);
			}
			size_t k = 0;
			for (const auto& d : inShape)
				inIdx.push_back(::tvm::prim::is_one(d) ? PrimExpr(::tvm::IntImm(i64, 0)) : nonUnit.at(k++));
			const s_tir::SBlock block(
					ivs, {tirx::BufferRegionFromPoint(A, inIdx)}, {tirx::BufferRegionFromPoint(B, outIdx)}, "T_reshape",
					tirx::BufferStore(B, tirx::BufferLoad(A, inIdx), outIdx)
			);
			tirx::Stmt body = s_tir::SBlockRealize(loopVars, ::tvm::IntImm::Bool(true), block);
			for (size_t d = outShape.size(); d-- > 0;)
				body = tirx::For(loops[d], ::tvm::IntImm(i64, 0), ::tvm::prim::cast(i64, outShape[d]), tirx::ForKind::kSerial, body);
			body = s_tir::SBlockRealize({}, ::tvm::IntImm::Bool(true), s_tir::SBlock({}, {}, {}, "root", body));
			auto func = tirx::PrimFunc({A, B}, body, ::tvm::VoidType());
			return ::tvm::WithAttr(std::move(func), "op_pattern", ::tvm::IntImm(::tvm::PrimType::Int(32), kInjective));
		}

		class RowFuser {
		public:
			enum class Mode { rows, inlineShared };
			RowFuser(::tvm::IRModule mod, Mode mode) : mod(mod), builder(relax::BlockBuilder::Create(mod)), mode(mode) {}

			::tvm::IRModule run() {
				std::vector<std::pair<::tvm::GlobalVar, relax::Function>> functions, updates;
				for (const auto& [gv, f] : mod->functions)
					if (const auto* fn = f.as<relax::FunctionNode>(); fn && !fn->HasNonzeroAttr(relax::attr::kPrimitive))
						functions.emplace_back(gv, ffi::GetRef<relax::Function>(fn));
				for (const auto& [gv, f] : functions)
					if (auto changed = rewrite(f))
						updates.emplace_back(gv, *changed);
				mod = builder->GetContextIRModule();
				for (const auto& [gv, f] : updates)
					mod->Update(gv, f);
				if (mode == Mode::rows)
					mdl::log::getLogger("compiler")->info("Fused {} group(s) of row kernels", groups);
				else
					mdl::log::getLogger("compiler")->info("Recomputed {} shared elementwise value(s) in their consumers", groups);
				return mod;
			}

		private:
			::tvm::IRModule mod;
			relax::BlockBuilder builder;
			Mode mode;
			size_t groups = 0;

			/** @brief A function of the module, also one added by this pass. */
			::tvm::BaseFunc lookup(const ::tvm::GlobalVar& gv) const { return builder->GetContextIRModule()->Lookup(gv); }

			static int patternOf(const ::tvm::BaseFunc& f) {
				if (f->attrs.defined())
					if (auto v = f->attrs->dict.Get("op_pattern")) {
						if (auto i = v->try_cast<int64_t>())
							return static_cast<int>(*i);
						if (auto imm = v->try_cast<::tvm::IntImm>())
							return static_cast<int>(intValue(imm.value().get()));
					}
				return 8; // (opaque)
			}

			std::optional<Site> site(const relax::Binding& binding) const {
				static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
				const auto* vb = binding.as<relax::VarBindingNode>();
				const auto* call = vb ? vb->value.as<::tvm::CallNode>() : nullptr;
				const auto shape = vb ? knownShape(vb->var) : std::nullopt;
				if (call == nullptr || !shape)
					return std::nullopt;
				Site s;
				s.shape = *shape;
				if (call->op.same_as(callTir)) {
					const auto* gv = call->args[0].as<::tvm::GlobalVarNode>();
					if (gv == nullptr || call->args.size() != 2)
						return std::nullopt;
					s.pattern = patternOf(lookup(ffi::GetRef<::tvm::GlobalVar>(gv)));
					for (const auto& a : call->args[1].as_or_throw<relax::Tuple>()->fields)
						s.args.push_back(a);
				} else if (const auto* gv = call->op.as<::tvm::GlobalVarNode>()) {
					const auto fn = lookup(ffi::GetRef<::tvm::GlobalVar>(gv)).as<relax::Function>();
					if (!fn || !(*fn)->HasNonzeroAttr(relax::attr::kPrimitive))
						return std::nullopt;
					const auto& f = *fn;
					s.pattern = 0;
					for (const auto& block : f->body.as_or_throw<relax::SeqExpr>()->blocks)
						for (const auto& b : block->bindings) {
							const auto* inner = b.as<relax::VarBindingNode>();
							const auto* c = inner ? inner->value.as<::tvm::CallNode>() : nullptr;
							if (c == nullptr || !c->op.same_as(callTir)) {
								if (inner && inner->value.as<::tvm::TupleGetItemNode>() == nullptr &&
									inner->value.as<relax::TupleNode>() == nullptr)
									return std::nullopt;
								continue;
							}
							s.pattern = std::max(
									s.pattern, patternOf(lookup(c->args[0].as_or_throw<::tvm::GlobalVar>()))
							);
						}
					for (const auto& a : call->args)
						s.args.push_back(a);
				} else {
					return std::nullopt;
				}
				for (const auto& a : s.args)
					s.shapes.push_back(knownShape(a).value_or(Shape()));
				if (s.pattern > kInjective && s.args.size() == 1 && call->op.same_as(callTir) && !s.shapes[0].empty() &&
					equal(withoutUnits(s.shapes[0]), withoutUnits(s.shape)) && tensorInfo(s.args[0]) &&
					tensorInfo(s.args[0])->dtype == tensorInfo(vb->var)->dtype) {
					s.pattern = kInjective; // (replaced by unitReshape in a group)
					s.unitReshape = true;
				}
				return s;
			}

			/** @brief The trailing dimensions L reduced by @p s (rows: its result's shape), if it reduces rows. */
			static std::optional<Shape> reducedRow(const Site& s) {
				if (s.pattern != kCommReduce || s.shape.empty())
					return std::nullopt;
				for (const auto& shape : s.shapes) {
					if (shape.size() <= s.shape.size())
						continue;
					if (!equal(Shape(shape.begin(), shape.begin() + static_cast<std::ptrdiff_t>(s.shape.size())), s.shape))
						continue;
					return Shape(shape.begin() + static_cast<std::ptrdiff_t>(s.shape.size()), shape.end());
				}
				return std::nullopt;
			}

			std::optional<relax::Function> rewrite(const relax::Function& f) {
				const auto* seq = f->body.as<relax::SeqExprNode>();
				if (seq == nullptr)
					return std::nullopt;
				// The uses of every variable in the function.
				std::unordered_map<const void*, size_t> uses;
				auto count = [&](const Expr& e) {
					for (const auto& v : relax::FreeVars(e))
						++uses[v.get()];
				};
				for (const auto& block : seq->blocks)
					for (const auto& b : block->bindings)
						if (const auto* vb = b.as<relax::VarBindingNode>())
							count(vb->value);
				count(seq->body);

				bool changed = false;
				ffi::Array<relax::BindingBlock> blocks;
				for (const auto& block : seq->blocks) {
					auto rewritten = block.as<relax::DataflowBlockNode>() ? rewrite(block, uses) : std::nullopt;
					changed |= rewritten.has_value();
					blocks.push_back(rewritten.value_or(block));
				}
				if (!changed)
					return std::nullopt;
				return relax::Function(f->params, relax::SeqExpr(blocks, seq->body), f->ret_ty, f->is_pure, f->attrs);
			}

			/**
			 * @brief An elementwise value whose uses are all elementwise computations of its shape: computed in each of
			 * them (FuseOps does not duplicate a producer with several consumers; e.g. the `erf` that GELU and its
			 * gradient share would be written and read again). The consumers usually become one kernel then (sibling
			 * fusion), where common subexpression elimination computes the value once.
			 */
			std::optional<relax::BindingBlock>
			inlineShared(const relax::BindingBlock& block, const std::unordered_map<const void*, size_t>& uses) {
				const auto& bindings = block->bindings;
				std::vector<std::optional<Site>> sites;
				for (const auto& b : bindings)
					sites.push_back(site(b));
				std::map<size_t, size_t> producerOf; // consumer -> producer
				std::unordered_set<size_t> producers;
				for (size_t i = 0; i < bindings.size(); ++i) {
					if (!sites[i] || sites[i]->pattern > kInjective || reducedRow(*sites[i]))
						continue;
					const void* v = bindings[i]->var.get();
					auto u = uses.find(v);
					if (u == uses.end() || u->second < 2)
						continue;
					std::vector<size_t> consumers;
					for (size_t j = i + 1; j < bindings.size(); ++j)
						if (usesVar(bindings[j], v))
							consumers.push_back(j);
					bool ok = consumers.size() == u->second;
					for (size_t j : consumers)
						ok &= sites[j] && sites[j]->pattern <= kInjective && !reducedRow(*sites[j]) &&
							  !producerOf.contains(j) && !producers.contains(j) && equal(sites[j]->shape, sites[i]->shape);
					if (!ok)
						continue;
					producers.insert(i);
					for (size_t j : consumers)
						producerOf.emplace(j, i);
				}
				if (producers.empty())
					return std::nullopt;
				ffi::Array<relax::Binding> result;
				for (size_t i = 0; i < bindings.size(); ++i) {
					if (producers.contains(i))
						continue;
					auto it = producerOf.find(i);
					if (it == producerOf.end()) {
						result.push_back(bindings[i]);
						continue;
					}
					for (const auto& b : fuse(bindings, {it->second, i}, {}))
						result.push_back(b);
				}
				groups += producers.size();
				return relax::BindingBlock(relax::DataflowBlock(result));
			}

			std::optional<relax::BindingBlock>
			rewrite(const relax::BindingBlock& block, const std::unordered_map<const void*, size_t>& uses) {
				if (mode == Mode::inlineShared)
					return inlineShared(block, uses);
				const auto& bindings = block->bindings;
				std::vector<std::optional<Site>> sites;
				for (const auto& b : bindings)
					sites.push_back(site(b));

				std::vector<bool> taken(bindings.size(), false);
				std::map<size_t, std::vector<size_t>> groupAt; // last member -> members
				std::map<size_t, std::vector<size_t>> escapingAt; // last member -> further results (indices into members)
				for (size_t i = 0; i < bindings.size(); ++i) {
					if (!sites[i] || taken[i])
						continue;
					const auto L = reducedRow(*sites[i]);
					if (!L)
						continue;
					const Shape rows = sites[i]->shape, rowsU = withoutUnits(rows), fullU = withoutUnits(concat(rows, *L));
					std::vector<size_t> group{i};
					std::unordered_set<const void*> outputs{bindings[i]->var.get()}, inputs;
					for (const auto& a : sites[i]->args)
						inputs.insert(a.get());
					std::vector<bool> full{false}; // (per member: an elementwise computation of shape [rows..., L...])
					for (size_t j = i + 1; j < bindings.size() && group.size() < kMaxGroup; ++j) {
						if (!sites[j] || taken[j])
							continue;
						const Site& s = *sites[j];
						bool readsGroup = false, shares = false;
						for (const auto& a : s.args) {
							readsGroup |= outputs.contains(a.get());
							shares |= a.as<::tvm::VarNode>() != nullptr && inputs.contains(a.get());
						}
						bool add = false, isFull = false;
						if (const auto l = reducedRow(s); l && equal(s.shape, rows) && equal(*l, *L)) {
							add = readsGroup || shares; // (a sibling reduction, or one of a group result)
						} else if (s.pattern <= kInjective && readsGroup) {
							const Shape u = withoutUnits(s.shape);
							if (equal(u, rowsU)) {
								add = true; // (per row)
							} else if (equal(u, fullU)) {
								add = isFull = true;
							}
						}
						if (!add)
							continue;
						group.push_back(j);
						full.push_back(isFull);
						outputs.insert(bindings[j]->var.get());
						for (const auto& a : s.args)
							inputs.insert(a.get());
					}
					// The longest prefix that ends with an elementwise epilogue (the group's result). Other results used
					// outside the prefix are further results of the group if they are elementwise (computed again at its
					// end, so that all results are written by its epilogues; reductions stay internal).
					const std::unordered_set<size_t> members(group.begin(), group.end());
					auto escapingUpTo = [&](size_t end) -> std::optional<std::vector<size_t>> {
						std::unordered_map<const void*, size_t> inGroup;
						for (size_t k = 0; k <= end; ++k)
							for (const auto& a : sites[group[k]]->args)
								++inGroup[a.get()];
						std::vector<size_t> escaping;
						for (size_t k = 0; k < end; ++k) {
							const void* v = bindings[group[k]]->var.get();
							auto it = uses.find(v);
							if (it != uses.end() && it->second == inGroup[v])
								continue;
							if (sites[group[k]]->pattern > kInjective || reducedRow(*sites[group[k]]))
								return std::nullopt;
							// (its other uses follow the group, which is computed in place of its last member)
							for (size_t j = group[k] + 1; j < group[end]; ++j)
								if (!members.contains(j) && usesVar(bindings[j], v))
									return std::nullopt;
							escaping.push_back(k);
						}
						return escaping;
					};
					size_t end = group.size();
					std::optional<std::vector<size_t>> escaping;
					while (end-- > 0)
						if (full[end] && (escaping = escapingUpTo(end)))
							break;
					if (end >= group.size() || end < 2)
						continue; // (one reduction and its epilogue: FuseOps fuses them already)
					group.resize(end + 1);
					for (size_t m : group)
						taken[m] = true;
					// (the per-row results first: the full-shape epilogue is the last block)
					std::stable_sort(escaping->begin(), escaping->end(), [&](size_t a, size_t b) { return !full[a] && full[b]; });
					escapingAt.emplace(group.back(), std::move(*escaping));
					groupAt.emplace(group.back(), std::move(group));
				}
				if (groupAt.empty())
					return std::nullopt;

				ffi::Array<relax::Binding> result;
				for (size_t i = 0; i < bindings.size(); ++i) {
					auto it = groupAt.find(i);
					if (it == groupAt.end()) {
						if (!taken[i])
							result.push_back(bindings[i]);
						continue;
					}
					for (const auto& b : fuse(bindings, it->second, escapingAt.at(i)))
						result.push_back(b);
					++groups;
				}
				return relax::BindingBlock(relax::DataflowBlock(result));
			}

			/** @brief One primitive function of the members, called in place of the last one. */
			std::vector<relax::Binding> fuse(
					const ffi::Array<relax::Binding>& bindings, const std::vector<size_t>& members,
					const std::vector<size_t>& escaping
			) {
				std::unordered_set<const void*> internal;
				for (size_t m : members)
					internal.insert(bindings[m]->var.get());
				// The parameters: the external tensor arguments.
				ffi::Array<::tvm::Var> params;
				ffi::Array<Expr> args;
				ffi::Map<::tvm::Var, Expr> remap; // main's variables -> the function's
				for (size_t m : members) {
					for (const auto& a : FreeVarsOf(bindings[m])) {
						if (internal.contains(a.get()) || remap.count(a))
							continue;
						::tvm::Var p(std::string(a->name), relax::GetType(a));
						params.push_back(p);
						args.push_back(a);
						remap.Set(a, p);
					}
				}
				// Tensor constants are parameters as well (FuseTIR: not in primitive functions).
				std::unordered_map<const void*, Expr> constants;
				auto argument = [&](const Expr& e) -> Expr {
					if (!tensorConstant(e))
						return relax::Bind(e, remap);
					auto it = constants.find(e.get());
					if (it == constants.end()) {
						::tvm::Var p("const", relax::GetType(e));
						params.push_back(p);
						args.push_back(e);
						it = constants.emplace(e.get(), p).first;
					}
					return it->second;
				};
				auto bb = relax::BlockBuilder::Create(std::nullopt);
				bb->BeginDataflowBlock();
				Expr last;
				std::vector<Expr> defining(members.size()); // per member: the computation of its result
				ffi::Array<Expr> results;
				for (size_t k = 0; k < members.size(); ++k) {
					if (k + 1 == members.size())
						for (size_t e : escaping) { // (computed again: written by an epilogue)
							const auto* c = defining[e].as<::tvm::CallNode>();
							results.push_back(bb->EmitOutput(
									::tvm::Call(c->ty, c->op, c->args, c->attrs, c->ty_args),
									std::string(bindings[members[e]]->var->name)
							));
						}
					const auto* vb = bindings[members[k]].as<relax::VarBindingNode>();
					const auto* call = vb->value.as<::tvm::CallNode>();
					Expr value;
					if (const auto* gv = call->op.as<::tvm::GlobalVarNode>()) {
						// A group of FuseOps: its bindings, with fresh variables (a function may be called twice).
						auto callee = relax::CopyWithNewVars(lookup(ffi::GetRef<::tvm::GlobalVar>(gv)).as_or_throw<relax::Function>());
						ffi::Map<::tvm::Var, Expr> inner;
						for (size_t p = 0; p < callee->params.size(); ++p)
							inner.Set(callee->params[p], argument(call->args[p]));
						// (also the callee's symbolic variables: FuseOps gives each group its own)
						::tvm::sym::Analyzer analyzer;
						inner = relax::InferSymbolicVarMap(inner, analyzer);
						const auto calleeSeq = callee->body.as_or_throw<relax::SeqExpr>();
						for (const auto& block : calleeSeq->blocks)
							for (const auto& b : block->bindings) {
								const auto* ib = b.as<relax::VarBindingNode>();
								const Expr bound = relax::Bind(ib->value, inner);
								if (ib->var.same_as(calleeSeq->body))
									defining[k] = bound;
								inner.Set(ib->var, bb->Emit(bound, std::string(ib->var->name)));
							}
						value = relax::Bind(calleeSeq->body, inner);
					} else if (site(bindings[members[k]])->unitReshape) {
						static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
						const Expr arg = argument(call->args[1].as_or_throw<relax::Tuple>()->fields[0]);
						const auto kernel = builder->AddFunction(
								unitReshape(*knownShape(arg), *knownShape(vb->var), tensorInfo(vb->var)->dtype.value()), "reshape_units"
						);
						value = ::tvm::Call(
								::tvm::Type::Missing(), callTir, {kernel, relax::Tuple({arg})}, ::tvm::Attrs(),
								{relax::GetType(vb->var)}
						);
					} else {
						ffi::Array<Expr> fields;
						for (const auto& a : call->args[1].as_or_throw<relax::Tuple>()->fields)
							fields.push_back(argument(a));
						value = ::tvm::Call(call->ty, call->op, {call->args[0], relax::Tuple(fields)}, call->attrs, call->ty_args);
					}
					if (!defining[k].defined())
						defining[k] = value;
					if (k + 1 < members.size()) {
						remap.Set(vb->var, bb->Emit(value, std::string(vb->var->name)));
					} else {
						last = bb->EmitOutput(value, std::string(vb->var->name));
					}
				}
				results.push_back(last);
				const auto block = bb->EndBlock();
				const Expr body = bb->Normalize(relax::SeqExpr({block}, results.size() == 1 ? last : Expr(relax::Tuple(results))));
				ffi::Map<ffi::String, ffi::Any> attrs;
				attrs.Set(relax::attr::kPrimitive, true);
				const relax::Function function(params, body, std::nullopt, true, ::tvm::DictAttrs(attrs));
				const auto gv = builder->AddFunction(function, "fused_rows");
				const auto& lastBinding = bindings[members.back()];
				if (escaping.empty())
					return {relax::VarBinding(
							lastBinding->var, ::tvm::Call(relax::GetType(lastBinding->var), gv, args, ::tvm::Attrs(), {})
					)};
				ffi::Array<::tvm::Type> types;
				std::vector<::tvm::Var> vars;
				for (size_t e : escaping)
					vars.push_back(bindings[members[e]]->var);
				vars.push_back(lastBinding->var);
				for (const auto& v : vars)
					types.push_back(relax::GetType(v));
				const ::tvm::TupleType tupleType(types);
				const ::tvm::Var tuple(relax::DataflowVar("fused_rows", tupleType));
				std::vector<relax::Binding> out{relax::VarBinding(tuple, ::tvm::Call(tupleType, gv, args, ::tvm::Attrs(), {}))};
				for (size_t k = 0; k < vars.size(); ++k)
					out.push_back(relax::VarBinding(vars[k], ::tvm::TupleGetItem(tuple, static_cast<int>(k))));
				return out;
			}

			static bool usesVar(const relax::Binding& b, const void* v) {
				for (const auto& u : FreeVarsOf(b))
					if (u.get() == v)
						return true;
				return false;
			}

			static ffi::Array<::tvm::Var> FreeVarsOf(const relax::Binding& b) {
				return relax::FreeVars(b.as_or_throw<relax::VarBinding>()->value);
			}
		};
	} // namespace

	::tvm::transform::Pass fuseRows() {
		return ::tvm::transform::CreateModulePass(
				[](::tvm::IRModule mod, ::tvm::transform::PassContext) {
					mod = RowFuser(std::move(mod), RowFuser::Mode::inlineShared).run();
					return RowFuser(std::move(mod), RowFuser::Mode::rows).run();
				}, 0,
				"mdl.FuseRows", {}
		);
	}

} // namespace mdl::backend::tvm

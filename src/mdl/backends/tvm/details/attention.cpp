#include "./attention.hpp"
#include "./tvm_compat.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ir/op.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op.h>

#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mdl::backend::tvm {

	namespace {
		using ::tvm::Call;
		using ::tvm::PrimExpr;
		using ::tvm::ffi::Array;
		using ::tvm::relax::BlockBuilder;
		using ::tvm::relax::Expr;
		using ::tvm::relax::TensorType;
		namespace RT = ::tvm::relax::transform;

		const ::tvm::Op& attentionOp() {
			static const auto& op = ::tvm::Op::Get("mdl.attention");
			return op;
		}
		const ::tvm::Op& attentionGradOp() {
			static const auto& op = ::tvm::Op::Get("mdl.attention_grad");
			return op;
		}

		template <typename... Args>
		Expr relaxOp(const char* name, Args&&... args) {
			return ::tvm::ffi::Function::GetGlobalRequired(name)(std::forward<Args>(args)...).template cast<Expr>();
		}

		Array<PrimExpr> shapeOf(const Expr& e) {
			if (auto shape = knownShape(e))
				return *shape;
			throw std::runtime_error("mdl backend: expected a tensor with known shape");
		}

		Expr intArg(int64_t v) { return ::tvm::IntImm(::tvm::PrimType::Int(64), v); }

		/** @brief A tensor operand read through a permutation: X = permute(x, perm). */
		struct Operand {
			Expr x;
			Perm perm;
			Array<PrimExpr> shape() const { return permuted(shapeOf(x), perm); }
			/** @brief The strides (in elements) of X's dimensions. */
			Array<PrimExpr> strides() const { return permuted(contiguousStrides(shapeOf(x)), perm); }
		};

		/**
		 * @brief The arguments of mdl.attention (forward) or mdl.attention_grad (see attention.hpp): the operands, the
		 * permutations of the results (forward: out; backward: dq, dk, dv), and the scale.
		 */
		struct AttentionArgs {
			bool backward = false;
			Operand q, k, v, o, dy; ///< (o, dy: backward only)
			Expr lse, probs;		///< (backward only)
			Expr bias;				///< added to the scores (broadcast), or PrimValue(0) for none
			Expr lengths;			///< if the bias is a suffix mask: its lengths, or PrimValue(0)
			double scale = 1.0;
			std::vector<Perm> outPerms;
			bool withLse = true; ///< (forward only)

			explicit AttentionArgs(const Call& call) : backward(call->op.same_as(attentionGradOp())) {
				const auto& a = call->args;
				if (!backward) {
					q = {a[0], permOf(a[4])};
					k = {a[1], permOf(a[5])};
					v = {a[2], permOf(a[6])};
					scale = *scalarConstant(a[3]);
					outPerms = {permOf(a[7])};
					withLse = intValue(a[8]) != 0;
					bias = a[9];
					lengths = a[10];
				} else {
					q = {a[0], permOf(a[8])};
					k = {a[1], permOf(a[9])};
					v = {a[2], permOf(a[10])};
					o = {a[3], permOf(a[11])};
					lse = a[4];
					probs = a[5];
					dy = {a[6], permOf(a[12])};
					scale = *scalarConstant(a[7]);
					outPerms = {permOf(a[13]), permOf(a[14]), permOf(a[15])};
					bias = a[16];
					lengths = a[17];
				}
			}

			bool hasBias() const { return tensorInfo(bias) != nullptr; }

			/** @brief The tensor operands (with their argument positions and permutation positions). */
			std::vector<Operand*> operands() {
				return backward ? std::vector<Operand*>{&q, &k, &v, &o, &dy} : std::vector<Operand*>{&q, &k, &v};
			}

			Call make() const {
				const Expr s = ::tvm::FloatImm(::tvm::PrimType::Float(64), scale);
				if (!backward)
					return Call(
							::tvm::Type::Missing(), attentionOp(),
							{q.x, k.x, v.x, s, permExpr(q.perm), permExpr(k.perm), permExpr(v.perm),
							 permExpr(outPerms[0]), intArg(withLse ? 1 : 0), bias, lengths}
					);
				return Call(
						::tvm::Type::Missing(), attentionGradOp(),
						{q.x, k.x, v.x, o.x, lse, probs, dy.x, s, permExpr(q.perm), permExpr(k.perm), permExpr(v.perm),
						 permExpr(o.perm), permExpr(dy.perm), permExpr(outPerms[0]), permExpr(outPerms[1]),
						 permExpr(outPerms[2]), bias, lengths}
				);
			}

			/** @brief The logical shape of the forward's result O [..., Sq, Dv]. */
			Array<PrimExpr> outShape() const {
				auto s = q.shape();
				auto vs = v.shape();
				Array<PrimExpr> r(s.begin(), s.end() - 1);
				r.push_back(vs[vs.size() - 1]);
				return r;
			}

			/** @brief The result types (stored layouts). */
			Array<::tvm::Type> resultTypes() const {
				const auto dtype = tensorInfo(q.x)->dtype.value();
				auto tensor = [&](const Array<PrimExpr>& shape, const Perm& perm, ::tvm::PrimType t) -> ::tvm::Type {
					return TensorType(::tvm::relax::ShapeExpr(permuted(shape, perm)), t);
				};
				if (!backward) {
					auto s = q.shape();
					Array<PrimExpr> rows(s.begin(), s.end() - 1);
					return {tensor(outShape(), outPerms[0], dtype), tensor(rows, {}, ::tvm::PrimType::Float(32))};
				}
				return {tensor(q.shape(), outPerms[0], dtype), tensor(k.shape(), outPerms[1], dtype),
						tensor(v.shape(), outPerms[2], dtype)};
			}
		};

		::tvm::Type inferAttention(const Call& call, const BlockBuilder&) {
			return ::tvm::TupleType(AttentionArgs{call}.resultTypes());
		}

		bool isAttention(const ::tvm::CallNode* call) {
			return call != nullptr && (call->op.same_as(attentionOp()) || call->op.same_as(attentionGradOp()));
		}

		/** @brief Where a variable is used: the consuming call and argument (call nullptr: another expression). */
		struct Use {
			const ::tvm::CallNode* call;
			size_t arg;
		};

		struct Analysis {
			std::unordered_map<const ::tvm::VarNode*, std::vector<Use>> uses;
			std::unordered_map<const ::tvm::VarNode*, Expr> values;

			explicit Analysis(const ::tvm::relax::Function& f) {
				auto other = [&](const Expr& e) {
					::tvm::relax::PostOrderVisit(e, [&](const Expr& x) {
						if (const auto* v = x.as<::tvm::VarNode>())
							uses[v].push_back({nullptr, 0});
					});
				};
				const auto* seq = f->body.as<::tvm::relax::SeqExprNode>();
				if (seq == nullptr)
					return;
				for (const auto& block : seq->blocks)
					for (const auto& binding : block->bindings) {
						const auto* vb = binding.as<::tvm::relax::VarBindingNode>();
						if (vb == nullptr)
							continue;
						values[vb->var.get()] = vb->value;
						if (const auto* call = vb->value.as<::tvm::CallNode>()) {
							for (size_t i = 0; i < call->args.size(); ++i) {
								if (const auto* v = call->args[i].as<::tvm::VarNode>())
									uses[v].push_back({call, i});
								else
									other(call->args[i]);
							}
						} else {
							other(vb->value);
						}
					}
				other(seq->body);
			}

			const ::tvm::CallNode* boundCall(const Expr& e, const ::tvm::Op& op) const {
				const auto* var = e.as<::tvm::VarNode>();
				auto it = var ? values.find(var) : values.end();
				const auto* call = it == values.end() ? nullptr : it->second.as<::tvm::CallNode>();
				return call != nullptr && call->op.same_as(op) ? call : nullptr;
			}
		};

		/** @brief Whether argument @p arg of an attention call is a tensor operand read through a permutation. */
		bool isOperandArg(const ::tvm::CallNode* call, size_t arg) {
			if (!isAttention(call))
				return false;
			return call->op.same_as(attentionGradOp()) ? (arg <= 3 || arg == 6) : arg <= 2;
		}

		std::optional<Perm> axesOf(const ::tvm::CallNode* permute) {
			const auto* attrs = permute->attrs.as<::tvm::relax::PermuteDimsAttrs>();
			if (attrs == nullptr || !attrs->axes.defined())
				return std::nullopt;
			Perm axes;
			for (const auto& a : attrs->axes.value())
				axes.push_back(a);
			return axes;
		}

		class FoldPermutes : public ::tvm::relax::ExprMutator {
		public:
			explicit FoldPermutes(const ::tvm::relax::Function& f) : analysis(f) {
				static const auto& permute = ::tvm::Op::Get("relax.permute_dims");
				// The results to write permuted: `g = t[i]` used by one permutation (and attention operands).
				for (const auto& [var, value] : analysis.values) {
					const auto* get = value.as<::tvm::TupleGetItemNode>();
					const auto* tuple = get ? get->tuple.as<::tvm::VarNode>() : nullptr;
					auto producer = tuple ? analysis.values.find(tuple) : analysis.values.end();
					const auto* call =
							producer == analysis.values.end() ? nullptr : producer->second.as<::tvm::CallNode>();
					if (!isAttention(call))
						continue;
					if (call->op.same_as(attentionOp()) && get->index == 1)
						continue; // (lse is not permuted)
					std::optional<Perm> axes;
					const ::tvm::CallNode* permuteCall = nullptr;
					bool ok = true;
					for (const Use& use : analysis.uses[var]) {
						if (use.call != nullptr && use.call->op.same_as(permute) && !permuteCall) {
							permuteCall = use.call;
							axes = axesOf(use.call);
						} else if (!isOperandArg(use.call, use.arg)) {
							ok = false;
						}
					}
					if (ok && axes)
						folded[var] = {tuple, static_cast<size_t>(get->index), *axes, permuteCall};
				}
				for (const auto& [var, f] : folded)
					outputFolds[f.tuple].emplace_back(f.index, f.axes);
			}

			using ExprMutator::VisitExpr_;

			Expr VisitExpr_(const ::tvm::CallNode* op) final {
				static const auto& permute = ::tvm::Op::Get("relax.permute_dims");
				if (op->op.same_as(permute)) {
					const auto* arg = op->args[0].as<::tvm::VarNode>();
					auto it = arg ? folded.find(arg) : folded.end();
					if (it != folded.end() && it->second.permute == op)
						return VisitExpr(op->args[0]); // (the result is written permuted)
				}
				if (!isAttention(op))
					return ExprMutator::VisitExpr_(op);
				const auto original = ::tvm::ffi::GetRef<Call>(op);
				AttentionArgs args{original};
				auto operands = args.operands();
				for (size_t i = 0; i < operands.size(); ++i) {
					Operand& x = *operands[i];
					if (const ::tvm::CallNode* p = analysis.boundCall(x.x, permute)) {
						if (auto axes = axesOf(p)) {
							x.x = p->args[0];
							x.perm = compose(x.perm, *axes);
						}
					}
					// A result that is now written permuted: read it through the inverse.
					const auto* var = x.x.as<::tvm::VarNode>();
					if (auto it = var ? folded.find(var) : folded.end(); it != folded.end())
						x.perm = compose(x.perm, inverse(it->second.axes));
					x.x = VisitExpr(x.x);
				}
				if (args.backward) {
					args.lse = VisitExpr(args.lse);
					args.probs = VisitExpr(args.probs);
				}
				args.bias = VisitExpr(args.bias);
				args.lengths = VisitExpr(args.lengths);
				if (const auto it = outputFolds.find(current); it != outputFolds.end())
					for (const auto& [index, axes] : it->second)
						args.outPerms[index] = compose(axes, args.outPerms[index]);
				if (!args.backward) {
					bool lseUsed = false;
					for (const auto& [var, value] : analysis.values)
						if (const auto* get = value.as<::tvm::TupleGetItemNode>())
							lseUsed |= get->index == 1 && get->tuple.as<::tvm::VarNode>() == current &&
									   !analysis.uses[var].empty();
					args.withLse = lseUsed;
				}
				return args.make();
			}

			void VisitBinding_(const ::tvm::relax::VarBindingNode* binding) override {
				current = binding->var.get();
				ExprMutator::VisitBinding_(binding);
				current = nullptr;
			}

		private:
			struct Folded {
				const ::tvm::VarNode* tuple;
				size_t index;
				Perm axes;
				const ::tvm::CallNode* permute;
			};
			Analysis analysis;
			std::unordered_map<const ::tvm::VarNode*, Folded> folded; ///< by the variable `t[i]`
			std::unordered_map<const ::tvm::VarNode*, std::vector<std::pair<size_t, Perm>>> outputFolds;
			const ::tvm::VarNode* current = nullptr;
		};

		/** @brief The unfused operations of the attention calls. */
		class Expand : public ::tvm::relax::ExprMutator {
		public:
			using ExprMutator::VisitExpr_;

			Expr VisitExpr_(const ::tvm::CallNode* op) final {
				auto call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();
				if (!isAttention(call.get()))
					return call;
				const AttentionArgs a{call};
				const size_t rank = a.q.shape().size();
				const Perm swap = swapLastTwo(rank);
				auto emit = [&](const Expr& e, const char* name) { return builder_->Emit(e, name); };
				const ::tvm::ffi::Optional<Array<int64_t>> last = Array<int64_t>{-1};
				if (!a.backward) {
					Expr scores =
							emit(makeMatmul(a.q.x, a.k.x, a.q.perm, compose(swap, a.k.perm), {}, a.scale), "scores");
					if (a.hasBias())
						scores = emit(relaxOp("relax.op.add", scores, a.bias), "biased");
					const Expr p = emit(relaxOp("relax.op.nn.softmax", scores, -1), "probs");
					const Expr out = emit(makeMatmul(p, a.v.x, {}, a.v.perm, a.outPerms[0], 1.0), "attn");
					Expr s32 = scores;
					if (tensorInfo(scores)->dtype.value() != ::tvm::PrimType::Float(32))
						s32 = emit(relaxOp("relax.op.astype", scores, ::tvm::PrimType::Float(32)->dtype), "s32");
					const Expr m = emit(relaxOp("relax.op.max", s32, last, true), "m");
					const Expr e = emit(relaxOp("relax.op.exp", relaxOp("relax.op.subtract", s32, m)), "e");
					const Expr lse =
							emit(relaxOp("relax.op.add", relaxOp("relax.op.max", s32, last, false),
										 relaxOp("relax.op.log", relaxOp("relax.op.sum", e, last, false))),
								 "lse");
					return ::tvm::relax::Tuple({out, lse});
				}
				const Expr& p = a.probs; // (saved by the forward pass)
				// dV = P^T dO, dP = dO V^T, dS = P * (dP - rowsum(dP * P)), dQ = scale dS K, dK = scale dS^T Q.
				const Expr dv = emit(makeMatmul(p, a.dy.x, swap, a.dy.perm, a.outPerms[2], 1.0), "dv");
				const Expr dp = emit(makeMatmul(a.dy.x, a.v.x, a.dy.perm, compose(swap, a.v.perm), {}, 1.0), "dp");
				const Expr rowsum =
						emit(relaxOp("relax.op.sum", relaxOp("relax.op.multiply", dp, p), last, true), "rowsum");
				const Expr ds = emit(relaxOp("relax.op.multiply", p, relaxOp("relax.op.subtract", dp, rowsum)), "ds");
				const Expr dq = emit(makeMatmul(ds, a.k.x, {}, a.k.perm, a.outPerms[0], a.scale), "dq");
				const Expr dk = emit(makeMatmul(ds, a.q.x, swap, a.q.perm, a.outPerms[1], a.scale), "dk");
				return ::tvm::relax::Tuple({dq, dk, dv});
			}
		};

		::tvm::transform::Pass
		functionPass(std::function<::tvm::relax::Function(const ::tvm::relax::Function&)> f, const char* name) {
			return RT::CreateFunctionPass(
					[f](::tvm::relax::Function func, ::tvm::IRModule, ::tvm::transform::PassContext) {
						return f(func);
					},
					0, name, {}
			);
		}

	} // namespace

	// ── operators ─────────────────────────────────────────────────────────────

	Expr makeAttention(Expr q, Expr k, Expr v, double scale, std::optional<Expr> bias, std::optional<Expr> lengths) {
		const Expr none = permExpr({});
		return Call(
				::tvm::Type::Missing(), attentionOp(),
				{std::move(q), std::move(k), std::move(v), ::tvm::FloatImm(::tvm::PrimType::Float(64), scale), none,
				 none, none, none, intArg(1), bias ? *bias : intArg(0), lengths ? *lengths : intArg(0)}
		);
	}

	Expr makeAttentionGrad(
			Expr q, Expr k, Expr v, Expr out, Expr lse, Expr probs, Expr dy, double scale, std::optional<Expr> bias,
			std::optional<Expr> lengths
	) {
		const Expr none = permExpr({});
		return Call(
				::tvm::Type::Missing(), attentionGradOp(),
				{std::move(q), std::move(k), std::move(v), std::move(out), std::move(lse), std::move(probs),
				 std::move(dy), ::tvm::FloatImm(::tvm::PrimType::Float(64), scale), none, none, none, none, none, none,
				 none, none, bias ? *bias : intArg(0), lengths ? *lengths : intArg(0)}
		);
	}

	TVM_REGISTER_OP("mdl.attention")
			.set_num_inputs(11)
			.add_argument("q", "Tensor", "Q = permute(q, perm_q): [..., Sq, D].")
			.add_argument("k", "Tensor", "K = permute(k, perm_k): [..., Skv, D].")
			.add_argument("v", "Tensor", "V = permute(v, perm_v): [..., Skv, Dv].")
			.add_argument("scale", "PrimValue", "The factor of the scores.")
			.add_argument("perm_q", "ShapeExpr", "The permutation of q (empty: none).")
			.add_argument("perm_k", "ShapeExpr", "The permutation of k (empty: none).")
			.add_argument("perm_v", "ShapeExpr", "The permutation of v (empty: none).")
			.add_argument("perm_out", "ShapeExpr", "The permutation of the result (empty: none).")
			.add_argument("with_lse", "PrimValue", "Whether the row statistics are used.")
			.add_argument(
					"bias", "Tensor | PrimValue", "Added to the scores [..., Sq, Skv] (broadcast), or PrimValue(0)."
			)
			.add_argument(
					"lengths", "Tensor | PrimValue",
					"If the bias is the mask of a suffix mask (of the queries' and keys' positions): its lengths "
					"[batch], "
					"else PrimValue(0)."
			)
			.set_attr<::tvm::relax::FInferType>("FInferType", inferAttention)
			.set_attr<bool>("FPurity", true);

	TVM_REGISTER_OP("mdl.attention_grad")
			.set_num_inputs(18)
			.add_argument("q", "Tensor", "As for mdl.attention.")
			.add_argument("k", "Tensor", "As for mdl.attention.")
			.add_argument("v", "Tensor", "As for mdl.attention.")
			.add_argument("out", "Tensor", "The result of mdl.attention: O = permute(out, perm_out).")
			.add_argument("lse", "Tensor", "The row statistics of mdl.attention [..., Sq] (float32).")
			.add_argument("probs", "Tensor", "The probabilities softmax(scale * Q @ K^T) [..., Sq, Skv].")
			.add_argument("dy", "Tensor", "The cotangent of the result: dY = permute(dy, perm_dy).")
			.add_argument("scale", "PrimValue", "The factor of the scores.")
			.add_argument("perm_q", "ShapeExpr", "")
			.add_argument("perm_k", "ShapeExpr", "")
			.add_argument("perm_v", "ShapeExpr", "")
			.add_argument("perm_out", "ShapeExpr", "")
			.add_argument("perm_dy", "ShapeExpr", "")
			.add_argument("perm_dq", "ShapeExpr", "The permutation of the result dq (empty: none).")
			.add_argument("perm_dk", "ShapeExpr", "The permutation of the result dk (empty: none).")
			.add_argument("perm_dv", "ShapeExpr", "The permutation of the result dv (empty: none).")
			.add_argument("bias", "Tensor | PrimValue", "As for mdl.attention.")
			.add_argument("lengths", "Tensor | PrimValue", "As for mdl.attention.")
			.set_attr<::tvm::relax::FInferType>("FInferType", inferAttention)
			.set_attr<bool>("FPurity", true);

	// ── passes ────────────────────────────────────────────────────────────────

	::tvm::transform::Pass foldPermutesIntoAttention() {
		return functionPass(
				[](const ::tvm::relax::Function& f) {
					return FoldPermutes(f).VisitExpr(f).as_or_throw<::tvm::relax::Function>();
				},
				"mdl.FoldPermutesIntoAttention"
		);
	}

	::tvm::transform::Pass expandAttention() {
		return functionPass(
				[](const ::tvm::relax::Function& f) {
					return Expand().VisitExpr(f).as_or_throw<::tvm::relax::Function>();
				},
				"mdl.ExpandAttention"
		);
	}

	bool hasAttention(const ::tvm::IRModule& mod) {
		bool found = false;
		for (const auto& [gv, func] : mod->functions)
			if (const auto* f = func.as<::tvm::relax::FunctionNode>())
				::tvm::relax::PostOrderVisit(::tvm::ffi::GetRef<::tvm::relax::Function>(f), [&](const Expr& e) {
					found |= isAttention(e.as<::tvm::CallNode>());
				});
		return found;
	}

} // namespace mdl::backend::tvm

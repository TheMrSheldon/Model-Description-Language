#include "./legalize.hpp"
#include "./extern_call.hpp"
#include "./tvm_compat.hpp"

#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/ir/op.h>
#include <tvm/relax/attrs/datatype.h>
#include <tvm/relax/attrs/index.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>
// nn/layer_norm.h uses helpers of reduction.h without including it.
#include <tvm/topi/nn/layer_norm.h>
#include <tvm/topi/nn/softmax.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mdl::backend::tvm {

	using ::tvm::Call;
	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	using ::tvm::ffi::Function;
	using ::tvm::ffi::Map;
	using ::tvm::ffi::String;
	using ::tvm::relax::BlockBuilder;
	using ::tvm::relax::Expr;
	using ::tvm::relax::ShapeExpr;
	using ::tvm::relax::TensorType;
	using ::tvm::relax::TensorTypeNode;
	namespace te = ::tvm::te;
	namespace tirx = ::tvm::tirx;
	namespace topi = ::tvm::topi;

	// ── helpers ──────────────────────────────────────────────────────────────────

	namespace {
		Array<PrimExpr> shapeOf(const Expr& e) {
			if (auto shape = knownShape(e))
				return *shape;
			throw std::runtime_error("mdl backend: expected a tensor with known shape");
		}

		double floatValue(const Expr& e) {
			if (const auto* f = e.as<::tvm::FloatImmNode>())
				return f->value;
			if (const auto* i = e.as<::tvm::IntImmNode>())
				return static_cast<double>(static_cast<int64_t>(i->value));
			throw std::runtime_error("mdl backend: expected a scalar");
		}

		PrimExpr product(const Array<PrimExpr>& dims, size_t begin, size_t end) {
			PrimExpr result = ::tvm::IntImm(::tvm::PrimType::Int(64), 1);
			for (size_t i = begin; i < end; ++i)
				result = result * ::tvm::prim::cast(::tvm::PrimType::Int(64), dims[i]);
			return result;
		}

		/**
		 * @brief C++ counterpart of Python's `BlockBuilder.call_te`: lowers a TE computation over the call's arguments
		 * to a PrimFunc and emits a `call_tir` to it.
		 */
		class TECall {
		public:
			TECall(BlockBuilder bb, std::string name) : bb(std::move(bb)), name(std::move(name)) {}

			/** @brief A placeholder for a tensor argument. */
			te::Tensor tensor(const Expr& arg, const std::string& argName = "x") {
				const auto* info = tensorInfo(arg);
				auto placeholder = te::placeholder(shapeOf(arg), info->dtype.value(), argName);
				shapes.push_back(placeholder->shape);
				inputs.push_back(arg);
				argList.push_back(placeholder);
				return placeholder;
			}

			/** @brief The value of a scalar argument (its symbolic variables become kernel arguments). */
			PrimExpr scalar(const Expr& arg, ::tvm::PrimType dtype) {
				auto value = scalarValue(arg);
				if (!value)
					throw std::runtime_error("mdl backend: expected a tensor or scalar operand");
				scalars.push_back(*value);
				return ::tvm::prim::cast(dtype, *value);
			}

			Expr finish(te::Tensor out) {
				// Relax's int64 extents (e.g. TOPI's kept dimensions of reductions are int32): the kernels that FuseTIR
				// fuses agree on the buffers of the tensors they pass (an identity stage, inlined).
				if (std::any_of(out->shape.begin(), out->shape.end(), [](const PrimExpr& d) {
						return d.ty() != ::tvm::PrimType::Int(64);
					})) {
					Array<PrimExpr> shape;
					for (const auto& d : out->shape)
						shape.push_back(::tvm::prim::cast(::tvm::PrimType::Int(64), d));
					const te::Tensor narrow = out;
					out = te::compute(
							shape,
							[&](const Array<::tvm::PrimVar>& i) { return narrow(Array<PrimExpr>(i.begin(), i.end())); },
							narrow->op->name + "_int64"
					);
				}
				shapes.push_back(out->shape);
				auto vars = unboundVars(shapes, scalars);
				for (const auto& v : vars)
					argList.push_back(v);
				argList.push_back(out);
				static const auto createPrimFunc = Function::GetGlobalRequired("te.CreatePrimFunc");
				auto func = createPrimFunc(argList, nullptr).cast<tirx::PrimFunc>();
				auto gv = bb->AddFunction(func, name);
				return callTir(gv, inputs, TensorType(ShapeExpr(out->shape), out->dtype), vars);
			}

		private:
			BlockBuilder bb;
			std::string name;
			Array<Expr> inputs;
			Array<::tvm::ffi::ObjectRef> argList;
			std::vector<Array<PrimExpr>> shapes;
			std::vector<PrimExpr> scalars;
		};

		Function wrap(std::function<Expr(const BlockBuilder&, const Call&)> fn) {
			return ::tvm::ffi::TypedFunction<Expr(BlockBuilder, Call)>(
					[fn = std::move(fn)](BlockBuilder bb, Call call) { return fn(bb, call); }
			);
		}

		// ── elementwise / reductions / layout (TOPI) ────────────────────────────────

		using BinaryTensor = te::Tensor (*)(const te::Tensor&, const te::Tensor&, std::string, std::string);
		using BinaryLeft = te::Tensor (*)(const PrimExpr&, const te::Tensor&, std::string, std::string);
		using BinaryRight = te::Tensor (*)(const te::Tensor&, const PrimExpr&, std::string, std::string);

		template <BinaryTensor TT, BinaryLeft ST, BinaryRight TS>
		Expr legalizeBinary(const BlockBuilder& bb, const Call& call, const std::string& name) {
			TECall te{bb, name};
			const auto* lhsInfo = tensorInfo(call->args[0]);
			const auto* rhsInfo = tensorInfo(call->args[1]);
			if (lhsInfo && rhsInfo)
				return te.finish(TT(
						te.tensor(call->args[0], "lhs"), te.tensor(call->args[1], "rhs"), "T_" + name, topi::kBroadcast
				));
			if (lhsInfo)
				return te.finish(
						TS(te.tensor(call->args[0], "lhs"), te.scalar(call->args[1], lhsInfo->dtype.value()),
						   "T_" + name, topi::kElementWise)
				);
			if (rhsInfo)
				return te.finish(
						ST(te.scalar(call->args[0], rhsInfo->dtype.value()), te.tensor(call->args[1], "rhs"),
						   "T_" + name, topi::kElementWise)
				);
			return call; // scalar arithmetic is handled by ComputePrimValue
		}

		using Unary = te::Tensor (*)(const te::Tensor&, std::string, std::string);

		Function unary(Unary fn, std::string name) {
			return wrap([fn, name](const BlockBuilder& bb, const Call& call) {
				TECall te{bb, name};
				return te.finish(fn(te.tensor(call->args[0]), "T_" + name, topi::kElementWise));
			});
		}

		template <te::Tensor (*Reduce)(const te::Tensor&, const ::tvm::ffi::Optional<Array<int64_t>>&, bool, bool)>
		Function reduction(std::string name) {
			return wrap([name](const BlockBuilder& bb, const Call& call) {
				const auto* attrs = call->attrs.as<::tvm::relax::StatisticalAttrs>();
				TECall te{bb, name};
				return te.finish(Reduce(te.tensor(call->args[0]), attrs->axis, attrs->keepdims, false));
			});
		}

		/** @brief TE matmul with NumPy semantics plus optional transposition of b, scaling, and bias. */
		te::Tensor matmulCompute(
				const te::Tensor& a, const te::Tensor& b, std::optional<te::Tensor> bias, bool transB, double alpha,
				bool transA
		) {
			const size_t ra = a->shape.size(), rb = b->shape.size();
			if (ra < 2 || rb < 2)
				throw std::runtime_error("mdl backend: matmul of 1-D tensors must be handled by the default path");
			PrimExpr K = transA ? a->shape[ra - 2] : a->shape[ra - 1];
			PrimExpr N = transB ? b->shape[rb - 2] : b->shape[rb - 1];
			// Output: broadcast batch dims of a and b, then [M, N].
			const size_t batchRank = std::max(ra, rb) - 2;
			Array<PrimExpr> outShape;
			for (size_t i = 0; i < batchRank; ++i) {
				const int ia = static_cast<int>(i) - static_cast<int>(batchRank - (ra - 2));
				const int ib = static_cast<int>(i) - static_cast<int>(batchRank - (rb - 2));
				PrimExpr da = ia >= 0 ? a->shape[ia] : PrimExpr(1);
				PrimExpr db = ib >= 0 ? b->shape[ib] : PrimExpr(1);
				const auto* imm = da.as<::tvm::IntImmNode>();
				outShape.push_back(imm && imm->value == 1 ? db : da);
			}
			outShape.push_back(transA ? a->shape[ra - 1] : a->shape[ra - 2]);
			outShape.push_back(N);
			auto k = te::reduce_axis(::tvm::Range(0, K), "k");
			const auto dtype = a->dtype;
			const auto f32 = ::tvm::PrimType::Float(32);
			const bool wide =
					dtype.bits() == 16 && (dtype.MatchesCode(kDLFloat) || dtype.MatchesElementType(kDLBfloat, 16));
			te::Tensor product = te::compute(
					outShape,
					[&](const Array<::tvm::PrimVar>& idx) {
						auto batchIndex = [&](const te::Tensor& t, size_t rank) {
							Array<PrimExpr> index;
							for (size_t i = 0; i + 2 < rank; ++i) {
								PrimExpr dim = t->shape[i];
								const auto* imm = dim.as<::tvm::IntImmNode>();
								PrimExpr v = idx[i + batchRank - (rank - 2)];
								index.push_back(imm && imm->value == 1 ? PrimExpr(0) : v);
							}
							return index;
						};
						auto ai = batchIndex(a, ra);
						if (transA) {
							ai.push_back(k);
							ai.push_back(idx[batchRank]);
						} else {
							ai.push_back(idx[batchRank]);
							ai.push_back(k);
						}
						auto bi = batchIndex(b, rb);
						if (transB) {
							bi.push_back(idx[batchRank + 1]);
							bi.push_back(k);
						} else {
							bi.push_back(k);
							bi.push_back(idx[batchRank + 1]);
						}
						// 16-bit floats accumulate in float32.
						if (wide)
							return ::tvm::prim::sum(::tvm::prim::cast(f32, a(ai)) * ::tvm::prim::cast(f32, b(bi)), {k});
						return ::tvm::prim::sum(a(ai) * b(bi), {k});
					},
					"T_matmul", "opaque"
			);
			if (alpha == 1.0 && !bias && !wide)
				return product;
			// Epilogue as a separate stage (a reduction must be the top-level expression of a compute).
			return te::compute(
					outShape,
					[&](const Array<::tvm::PrimVar>& idx) {
						PrimExpr v = product(Array<PrimExpr>{idx.begin(), idx.end()});
						const auto accumulator = wide ? f32 : dtype;
						if (alpha != 1.0)
							v = v * ::tvm::prim::MakeConst(accumulator, alpha);
						if (bias)
							v = v + ::tvm::prim::cast(accumulator, (*bias)(Array<PrimExpr>{idx[batchRank + 1]}));
						return wide ? ::tvm::prim::cast(dtype, v) : v;
					},
					"T_matmul_epilogue", topi::kElementWise
			);
		}

		/** @brief A quantized operand: the arguments of mdl.dequant besides the quantized values (see QuantFormat). */
		struct Quantized {
			Expr scales;
			std::optional<Expr> zeros;
			mdl::builtins::QuantFormat format;
			int64_t group;

			bool packed() const { return format != mdl::builtins::QuantFormat::int8; }
		};

		/** @brief The dequantization of q (see mdl.dequant): W [N, K] in the scales' type. */
		te::Tensor dequantCompute(TECall& te, const Expr& qArg, const Quantized& quant) {
			using mdl::builtins::QuantFormat;
			auto q = te.tensor(qArg, "q");
			auto scales = te.tensor(quant.scales, "scales");
			std::optional<te::Tensor> z;
			if (quant.zeros)
				z = te.tensor(*quant.zeros, "zeros");
			const auto f32 = ::tvm::PrimType::Float(32), i32 = ::tvm::PrimType::Int(32);
			const int64_t group = quant.group;
			const auto format = quant.format;
			Array<PrimExpr> shape{q->shape[0], quant.packed() ? q->shape[1] * 2 : q->shape[1]};
			return te::compute(
					shape,
					[&](const Array<::tvm::PrimVar>& i) {
						const PrimExpr n = i[0], k = i[1];
						const PrimExpr g = ::tvm::floordiv(k, ::tvm::IntImm(k.ty(), group));
						PrimExpr value;
						if (format == QuantFormat::int8) {
							value = ::tvm::prim::cast(i32, q(Array<PrimExpr>{n, k}));
						} else {
							const PrimExpr two = ::tvm::IntImm(k.ty(), 2);
							const PrimExpr byte =
									::tvm::prim::cast(i32, q(Array<PrimExpr>{n, ::tvm::floordiv(k, two)})) & 0xFF;
							const PrimExpr nibble = ::tvm::prim::Select(
									::tvm::floormod(k, two) == ::tvm::IntImm(k.ty(), 0), byte & 0xF, (byte >> 4) & 0xF
							);
							value = format == QuantFormat::int4
											? ::tvm::prim::Select(nibble >= 8, nibble - 16, nibble) // signed
											: nibble;
						}
						if (z)
							value = value - ::tvm::prim::cast(i32, (*z)(Array<PrimExpr>{g, n}));
						const PrimExpr scale = ::tvm::prim::cast(f32, scales(Array<PrimExpr>{g, n}));
						return ::tvm::prim::cast(scales->dtype, ::tvm::prim::cast(f32, value) * scale);
					},
					"T_dequant", topi::kElementWise
			);
		}

		/** @brief The quantization of a mdl.matmul or mdl.dequant call from its arguments [first, first + 3). */
		std::optional<Quantized> quantizedOf(const Call& call, size_t first) {
			const Perm info = permOf(call->args[first + 2]); // [format, group] or empty
			if (info.size() != 2)
				return std::nullopt;
			return Quantized{
					call->args[first],
					tensorInfo(call->args[first + 1]) ? std::optional<Expr>{call->args[first + 1]} : std::nullopt,
					static_cast<mdl::builtins::QuantFormat>(info[0]), info[1]
			};
		}

		/**
		 * @brief mdl.matmul's arguments: (a, b, bias or PrimValue(0), perm_a, perm_b, perm_out, alpha, addend or
		 * PrimValue(0), scales, zeros, quant). The operands are A = permute(a, perm_a) and B = permute(b, perm_b); the
		 * result is permute(alpha * A @ B + bias, perm_out) + addend (the epilogue: e.g. a residual connection). The
		 * epilogues are those that GEMM libraries compute natively (bias, scale, addend); others (e.g. activations) are
		 * generated kernels, fused with their neighbors. If quant is [format, group], b holds quantized values and B is permute(dequant(b,
		 * scales, zeros), perm_b) (see mdl.dequant); otherwise scales and zeros are PrimValue(0) and quant is empty.
		 */
		struct MatmulArgs {
			Expr a, b;
			std::optional<Expr> bias;
			Perm permA, permB, permOut;
			double alpha = 1.0;
			std::optional<Expr> addend;
			std::optional<Quantized> quant;

			MatmulArgs(Expr a, Expr b) : a(std::move(a)), b(std::move(b)) {}

			explicit MatmulArgs(const Call& call)
					: a(call->args[0]), b(call->args[1]),
					  bias(tensorInfo(call->args[2]) ? std::optional<Expr>{call->args[2]} : std::nullopt),
					  permA(permOf(call->args[3])), permB(permOf(call->args[4])), permOut(permOf(call->args[5])),
					  alpha(floatValue(call->args[6])),
					  addend(tensorInfo(call->args[7]) ? std::optional<Expr>{call->args[7]} : std::nullopt),
					  quant(quantizedOf(call, 8)) {}

			/** @brief The shape of B before perm_b (the dequantized shape [N, K] if b is quantized). */
			Array<PrimExpr> bShape() const {
				auto shape = shapeOf(b);
				if (quant && quant->packed())
					return Array<PrimExpr>{shape[0], shape[1] * 2};
				return shape;
			}

			/** @brief The shape of A @ B (before perm_out): the broadcast batch dimensions, then [M, N]. */
			Array<PrimExpr> productShape() const {
				auto la = permuted(shapeOf(a), permA), lb = permuted(bShape(), permB);
				const size_t ra = la.size(), rb = lb.size(), batchRank = std::max(ra, rb) - 2;
				Array<PrimExpr> out;
				for (size_t i = 0; i < batchRank; ++i) {
					const int ia = static_cast<int>(i) - static_cast<int>(batchRank - (ra - 2));
					const int ib = static_cast<int>(i) - static_cast<int>(batchRank - (rb - 2));
					PrimExpr da = ia >= 0 ? la[ia] : PrimExpr(1);
					PrimExpr db = ib >= 0 ? lb[ib] : PrimExpr(1);
					const auto* imm = da.as<::tvm::IntImmNode>();
					out.push_back(imm && imm->value == 1 ? db : da);
				}
				out.push_back(la[ra - 2]);
				out.push_back(lb[rb - 1]);
				return out;
			}
		};

		/** @brief @p m with a quantized b replaced by its dequantization (emitted as a kernel of its own). */
		MatmulArgs dequantized(const BlockBuilder& bb, MatmulArgs m) {
			if (!m.quant)
				return m;
			TECall te{bb, "dequant"};
			auto w = dequantCompute(te, m.b, *m.quant);
			m.b = bb->Emit(te.finish(w));
			m.quant.reset();
			return m;
		}

		Expr matmulTE(const BlockBuilder& bb, const MatmulArgs& quantized) {
			const MatmulArgs m = dequantized(bb, quantized);
			TECall te{bb, "matmul"};
			auto ta = te.tensor(m.a, "A");
			auto tb = te.tensor(m.b, "B");
			auto axes = [](const Perm& p) { return Array<int64_t>(p.begin(), p.end()); };
			if (!m.permA.empty() && !isIdentity(m.permA))
				ta = topi::transpose(ta, axes(m.permA));
			if (!m.permB.empty() && !isIdentity(m.permB))
				tb = topi::transpose(tb, axes(m.permB));
			std::optional<te::Tensor> tbias;
			if (m.bias)
				tbias = te.tensor(*m.bias, "bias");
			te::Tensor out = matmulCompute(ta, tb, tbias, false, m.alpha, false);
			if (!m.permOut.empty() && !isIdentity(m.permOut))
				out = topi::transpose(out, axes(m.permOut));
			if (m.addend)
				out = topi::add(out, te.tensor(*m.addend, "addend"));
			return te.finish(out);
		}

		// ── mdl.matmul operator ───────────────────────────────────────────────────

		::tvm::Type inferMatmul(const Call& call, const BlockBuilder& ctx) {
			MatmulArgs args{call};
			const auto* ai = tensorInfo(args.a);
			return TensorType(ShapeExpr(permuted(args.productShape(), args.permOut)), ai->dtype.value(), ai->vdevice);
		}

		Expr makeCall(const MatmulArgs& m) {
			static const auto& op = ::tvm::Op::Get("mdl.matmul");
			Expr bias = m.bias ? *m.bias : Expr(::tvm::IntImm(::tvm::PrimType::Int(64), 0));
			Expr addend = m.addend ? *m.addend : Expr(::tvm::IntImm(::tvm::PrimType::Int(64), 0));
			const Expr none = ::tvm::IntImm(::tvm::PrimType::Int(64), 0);
			Expr scales = m.quant ? m.quant->scales : none;
			Expr zeros = m.quant && m.quant->zeros ? *m.quant->zeros : none;
			Expr quant = permExpr(m.quant ? Perm{static_cast<int64_t>(m.quant->format), m.quant->group} : Perm{});
			return Call(
					::tvm::Type::Missing(), op,
					{m.a, m.b, bias, permExpr(m.permA), permExpr(m.permB), permExpr(m.permOut),
					 ::tvm::FloatImm(::tvm::PrimType::Float(64), m.alpha), addend, scales, zeros, quant}
			);
		}

		/**
		 * @brief Folds permutations and epilogues into mdl.matmul (see MatmulArgs): `permute(x) @ y` reads x through
		 * strides; if the product has no other use, `permute(a @ b)` writes the result in the permuted layout, and
		 * `a @ b + r` is computed by the matrix product's epilogue.
		 */
		class FoldPermutes : public ::tvm::relax::ExprMutator {
		public:
			explicit FoldPermutes(const ::tvm::relax::Function& f) {
				struct Counter : ::tvm::relax::ExprVisitor {
					std::unordered_map<const ::tvm::VarNode*, int>* uses = nullptr;
					void VisitExpr_(const ::tvm::VarNode* op) final { ++(*uses)[op]; }
					void VisitExpr_(const ::tvm::relax::DataflowVarNode* op) final { ++(*uses)[op]; }
				} counter;
				counter.uses = &uses;
				counter.VisitExpr(f);
			}

			using ExprMutator::VisitExpr_;

			Expr VisitExpr_(const ::tvm::CallNode* op) final {
				static const auto& matmul = ::tvm::Op::Get("mdl.matmul");
				static const auto& permute = ::tvm::Op::Get("relax.permute_dims");
				auto call = (ExprMutator::VisitExpr_(op)).as_or_throw<Call>();
				if (call->op.same_as(matmul)) {
					MatmulArgs m{call};
					bool changed = false;
					auto fold = [&](Expr& x, Perm& perm) {
						const ::tvm::CallNode* p = boundCall(x, permute);
						if (p == nullptr)
							return;
						auto axes = axesOf(p);
						if (!axes)
							return;
						x = p->args[0];
						perm = compose(perm, *axes);
						changed = true;
					};
					fold(m.a, m.permA);
					fold(m.b, m.permB);
					// A dequantized operand: the kernel reads the quantized values (the dequantization becomes dead
					// if no other operation uses it).
					static const auto& dequant = ::tvm::Op::Get("mdl.dequant");
					if (const ::tvm::CallNode* d = m.quant ? nullptr : boundCall(m.b, dequant)) {
						const Call dq = ::tvm::ffi::GetRef<Call>(d);
						m.b = dq->args[0];
						m.quant = Quantized{
								dq->args[1], tensorInfo(dq->args[2]) ? std::optional<Expr>{dq->args[2]} : std::nullopt,
								static_cast<mdl::builtins::QuantFormat>(intValue(dq->args[3])), intValue(dq->args[4])
						};
						changed = true;
					}
					return changed ? makeCall(m) : call;
				}
				static const auto& add = ::tvm::Op::Get("relax.add");
				// The product's single use: an addition of a tensor of the result's shape (e.g. a residual connection).
				if (call->op.same_as(add)) {
					for (size_t i = 0; i < 2; ++i) {
						if (!singleUse(op->args[i]))
							continue;
						const ::tvm::CallNode* matmulOp = boundCall(call->args[i], matmul);
						const Expr& other = call->args[1 - i];
						const auto* product =
								matmulOp ? ::tvm::relax::GetTypeAs<TensorTypeNode>(call->args[i]) : nullptr;
						const auto* addend = ::tvm::relax::GetTypeAs<TensorTypeNode>(other);
						if (product == nullptr || addend == nullptr || addend->dtype != product->dtype ||
							!product->shape.defined() || !addend->shape.defined() ||
							!::tvm::ffi::StructuralEqual()(product->shape, addend->shape))
							continue;
						MatmulArgs m{::tvm::ffi::GetRef<Call>(matmulOp)};
						if (m.addend)
							continue;
						m.addend = other;
						return makeCall(m);
					}
					return call;
				}
				const auto* arg = op->args.empty() ? nullptr : op->args[0].as<::tvm::VarNode>();
				if (call->op.same_as(permute) && arg != nullptr && uses[arg] == 1) {
					const ::tvm::CallNode* matmulOp = boundCall(call->args[0], matmul);
					auto axes = axesOf(call.get());
					if (matmulOp == nullptr || !axes)
						return call;
					MatmulArgs m{::tvm::ffi::GetRef<Call>(matmulOp)};
					if (m.addend) // (the addend has the unpermuted result's layout)
						return call;
					m.permOut = compose(*axes, m.permOut);
					return makeCall(m);
				}
				return call;
			}

		private:
			std::unordered_map<const ::tvm::VarNode*, int> uses;

			bool singleUse(const Expr& e) {
				const auto* var = e.as<::tvm::VarNode>();
				return var != nullptr && uses[var] == 1;
			}

			/** @brief The call of @p op bound to the variable @p e (in the rewritten function), if any. */
			const ::tvm::CallNode* boundCall(const Expr& e, const ::tvm::Op& op) {
				const auto* var = e.as<::tvm::VarNode>();
				if (var == nullptr)
					return nullptr;
				auto value = builder_->LookupBinding(::tvm::ffi::GetRef<::tvm::Var>(var));
				const auto* call = value ? value.value().as<::tvm::CallNode>() : nullptr;
				return call != nullptr && call->op.same_as(op) ? call : nullptr;
			}

			static std::optional<Perm> axesOf(const ::tvm::CallNode* permute) {
				const auto* attrs = permute->attrs.as<::tvm::relax::PermuteDimsAttrs>();
				if (attrs == nullptr || !attrs->axes.defined())
					return std::nullopt;
				Perm axes;
				for (const auto& a : attrs->axes.value())
					axes.push_back(a);
				return axes;
			}
		};
	} // namespace

	Expr makeMatmul(Expr a, Expr b, std::optional<Expr> bias, bool transB, double alpha, bool transA) {
		auto rank = [](const Expr& e) { return static_cast<size_t>(::tvm::relax::GetTypeAs<TensorTypeNode>(e)->ndim); };
		MatmulArgs m{a, b};
		m.bias = std::move(bias);
		if (transA)
			m.permA = swapLastTwo(rank(a));
		if (transB)
			m.permB = swapLastTwo(rank(b));
		m.alpha = alpha;
		return makeCall(m);
	}

	Expr makeMatmul(Expr a, Expr b, Perm permA, Perm permB, Perm permOut, double alpha) {
		MatmulArgs m{std::move(a), std::move(b)};
		m.permA = std::move(permA);
		m.permB = std::move(permB);
		m.permOut = std::move(permOut);
		m.alpha = alpha;
		return makeCall(m);
	}

	Expr makeDequant(Expr q, Expr scales, Expr zeros, int64_t format, int64_t group) {
		static const auto& op = ::tvm::Op::Get("mdl.dequant");
		return Call(
				::tvm::Type::Missing(), op,
				{std::move(q), std::move(scales), std::move(zeros), ::tvm::IntImm(::tvm::PrimType::Int(64), format),
				 ::tvm::IntImm(::tvm::PrimType::Int(64), group)}
		);
	}

	TVM_REGISTER_OP("mdl.dequant")
			.set_num_inputs(5)
			.add_argument("q", "Tensor", "The quantized values [N, K] int8 (int4: [N, K/2], two per byte).")
			.add_argument("scales", "Tensor", "The scales [K/group, N] (float).")
			.add_argument("zeros", "Tensor | PrimValue", "The zero points [K/group, N] int8, or PrimValue(0).")
			.add_argument("format", "PrimValue", "The format (builtins::QuantFormat).")
			.add_argument("group", "PrimValue", "The number of values of K per scale.")
			.set_attr<::tvm::relax::FInferType>(
					"FInferType",
					[](const Call& call, const BlockBuilder&) -> ::tvm::Type {
						auto q = shapeOf(call->args[0]);
						const int64_t format = intValue(call->args[3]);
						const bool packed = format != static_cast<int64_t>(mdl::builtins::QuantFormat::int8);
						Array<PrimExpr> shape{q[0], packed ? q[1] * 2 : q[1]};
						return TensorType(ShapeExpr(shape), tensorInfo(call->args[1])->dtype.value());
					}
			)
			.set_attr<bool>("FPurity", true);

	Expr makeSuffixMask(Expr lengths, Expr like) {
		static const auto& op = ::tvm::Op::Get("mdl.suffix_mask");
		return Call(::tvm::Type::Missing(), op, {std::move(lengths), std::move(like)});
	}

	namespace {
		/** @brief The shape of mdl.suffix_mask: the lengths' dimensions and the next dimension of the reference. */
		Array<PrimExpr> suffixMaskShape(const Call& call) {
			const auto lengths = shapeOf(call->args[0]), like = shapeOf(call->args[1]);
			if (like.size() <= lengths.size())
				throw std::runtime_error("mdl.suffix_mask: the reference has no dimension after the lengths' dimensions"
				);
			return Array<PrimExpr>(like.begin(), like.begin() + static_cast<std::ptrdiff_t>(lengths.size() + 1));
		}
	} // namespace

	TVM_REGISTER_OP("mdl.suffix_mask")
			.set_num_inputs(2)
			.add_argument("lengths", "Tensor", "The number of valid elements [batch...] (integer).")
			.add_argument("like", "Tensor", "A tensor [batch..., S, ...] (its dimension S: the positions; not read).")
			.set_attr<::tvm::relax::FInferType>(
					"FInferType",
					[](const Call& call, const BlockBuilder&) -> ::tvm::Type {
						return TensorType(ShapeExpr(suffixMaskShape(call)), ::tvm::PrimType::Bool());
					}
			)
			.set_attr<bool>("FPurity", true);

	Expr makeGeluBackward(Expr x, Expr dy) {
		static const auto& op = ::tvm::Op::Get("mdl.gelu_backward");
		return Call(::tvm::Type::Missing(), op, {std::move(x), std::move(dy)});
	}

	TVM_REGISTER_OP("mdl.gelu_backward")
			.set_num_inputs(2)
			.add_argument("x", "Tensor", "The input of GELU.")
			.add_argument("dy", "Tensor", "The cotangent of GELU's result (same shape).")
			.set_attr<::tvm::relax::FInferType>(
					"FInferType",
					[](const Call& call, const BlockBuilder&) -> ::tvm::Type {
						return ::tvm::relax::GetType(call->args[1]);
					}
			)
			.set_attr<bool>("FPurity", true);

	::tvm::transform::Pass foldPermutesIntoMatmul() {
		return ::tvm::relax::transform::CreateFunctionPass(
				[](::tvm::relax::Function f, ::tvm::IRModule, ::tvm::transform::PassContext) {
					return (FoldPermutes(f).VisitExpr(f)).as_or_throw<::tvm::relax::Function>();
				},
				0, "mdl.FoldPermutesIntoMatmul", {}
		);
	}

	TVM_REGISTER_OP("mdl.matmul")
			.set_num_inputs(11)
			.add_argument("a", "Tensor", "The left operand A = permute(a, perm_a): [..., M, K].")
			.add_argument("b", "Tensor", "The right operand B = permute(b, perm_b): [K, N], possibly batched.")
			.add_argument("bias", "Tensor | PrimValue", "The bias [N] or PrimValue(0) for no bias.")
			.add_argument("perm_a", "ShapeExpr", "The permutation of a (empty: none).")
			.add_argument("perm_b", "ShapeExpr", "The permutation of b (empty: none).")
			.add_argument("perm_out", "ShapeExpr", "The permutation of the result (empty: none).")
			.add_argument("alpha", "PrimValue", "Scale applied to the product.")
			.add_argument("addend", "Tensor | PrimValue", "A tensor of the result's shape added last, or PrimValue(0).")
			.add_argument(
					"scales", "Tensor | PrimValue", "If b is quantized: the scales of mdl.dequant, else PrimValue(0)."
			)
			.add_argument("zeros", "Tensor | PrimValue", "If b is quantized: the zero points or PrimValue(0).")
			.add_argument("quant", "ShapeExpr", "If b is quantized: [format, group] of mdl.dequant, else empty.")
			.set_attr<::tvm::relax::FInferType>("FInferType", inferMatmul)
			.set_attr<bool>("FPurity", true);

	// ── handler map ──────────────────────────────────────────────────────────────

	Function legalizeHandler(std::function<Expr(const BlockBuilder&, const Call&)> fn) { return wrap(std::move(fn)); }

	Expr callPrimFunc(
			const BlockBuilder& bb, tirx::PrimFunc func, const std::string& name, Array<Expr> inputs,
			const TensorType& out
	) {
		static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
		auto gv = bb->AddFunction(func, name);
		return Call(::tvm::Type::Missing(), callTir, {gv, ::tvm::relax::Tuple(inputs)}, ::tvm::Attrs(), {out});
	}

	namespace {
		/**
		 * @brief The CPU kernel of mdl.scatter_add_rows (TVM's scatter is only available in Python): the columns are
		 * split into chunks of 16 that are processed in parallel, so threads never write the same element (repeated
		 * indices accumulate correctly).
		 */
		tirx::PrimFunc scatterAddRowsKernel(::tvm::PrimType idsType) {
			using ::tvm::IntImm;
			const auto i64 = ::tvm::PrimType::Int(64);
			auto rootBlock = [](tirx::Stmt body) -> tirx::Stmt {
				return ::tvm::s_tir::SBlockRealize(
						{}, IntImm::Bool(true), ::tvm::s_tir::SBlock("root", std::move(body))
				);
			};
			// The kernel's dimensions are its own variables (bound from the buffers).
			::tvm::PrimVar n("n", i64), r("r", i64), v("v", i64);
			auto G = tirx::decl_buffer({n, r}, ::tvm::PrimType::Float(32), "g");
			auto I = tirx::decl_buffer({n}, idsType, "ids");
			auto O = tirx::decl_buffer({v, r}, ::tvm::PrimType::Float(32), "out");

			constexpr int64_t kChunk = 16;
			::tvm::PrimVar row("row", i64), col("col", i64), chunk("chunk", i64), lane("lane", i64), k("k", i64);
			auto zero = tirx::For(
					row, IntImm(i64, 0), v, tirx::ForKind::kParallel,
					tirx::For(
							col, IntImm(i64, 0), r, tirx::ForKind::kSerial,
							tirx::BufferStore(O, ::tvm::FloatImm(::tvm::PrimType::Float(32), 0.0), {row, col})
					)
			);
			const ::tvm::PrimExpr chunkSize = IntImm(i64, kChunk);
			const ::tvm::PrimExpr c = chunk * chunkSize + lane;
			const ::tvm::PrimExpr index =
					::tvm::max(::tvm::min(::tvm::prim::cast(i64, tirx::BufferLoad(I, {k})), v - 1), IntImm(i64, 0));
			auto accumulate = tirx::For(
					chunk, IntImm(i64, 0), ::tvm::floordiv(r + IntImm(i64, kChunk - 1), chunkSize),
					tirx::ForKind::kParallel,
					tirx::For(
							k, IntImm(i64, 0), n, tirx::ForKind::kSerial,
							tirx::For(
									lane, IntImm(i64, 0), IntImm(i64, kChunk), tirx::ForKind::kSerial,
									tirx::IfThenElse(
											c < r,
											tirx::BufferStore(
													O, tirx::BufferLoad(O, {index, c}) + tirx::BufferLoad(G, {k, c}),
													{index, c}
											)
									)
							)
					)
			);
			auto body = rootBlock(tirx::SeqStmt(::tvm::ffi::Array<tirx::Stmt>{zero, accumulate}));
			auto func = tirx::PrimFunc({G, I, O}, body, ::tvm::VoidType());
			func = ::tvm::WithAttr(std::move(func), "tirx.is_scheduled", ::tvm::IntImm::Bool(true)); // (not fused)
			return func;
		}
	} // namespace

	Expr makeScatterAddRows(Expr g, Expr ids, PrimExpr rows) {
		static const auto& op = ::tvm::Op::Get("mdl.scatter_add_rows");
		return Call(::tvm::Type::Missing(), op, {std::move(g), std::move(ids), std::move(rows)});
	}

	TVM_REGISTER_OP("mdl.scatter_add_rows")
			.set_num_inputs(3)
			.add_argument("g", "Tensor", "The rows to add [N, R] (float32).")
			.add_argument("ids", "Tensor", "The row of each [N] (integer; clipped to [0, V)).")
			.add_argument("rows", "PrimValue", "V, the number of rows of the result.")
			.set_attr<::tvm::relax::FInferType>(
					"FInferType",
					[](const Call& call, const BlockBuilder&) -> ::tvm::Type {
						const auto g = shapeOf(call->args[0]);
						return TensorType(ShapeExpr({*scalarValue(call->args[2]), g[1]}), ::tvm::PrimType::Float(32));
					}
			)
			.set_attr<bool>("FPurity", true);

	Map<String, Function> legalizeMap(const mdl::CompileOptions& opts) {
		const bool fast = opts.fastMath;

		Map<String, Function> map;
		map.Set("relax.add", wrap([](auto& bb, auto& call) {
					return legalizeBinary<topi::add, topi::add, topi::add>(bb, call, "add");
				}));
		map.Set("relax.subtract", wrap([](auto& bb, auto& call) {
					return legalizeBinary<topi::subtract, topi::subtract, topi::subtract>(bb, call, "subtract");
				}));
		map.Set("relax.multiply", wrap([](auto& bb, auto& call) {
					return legalizeBinary<topi::multiply, topi::multiply, topi::multiply>(bb, call, "multiply");
				}));
		map.Set("relax.divide", wrap([](auto& bb, auto& call) {
					return legalizeBinary<topi::divide, topi::divide, topi::divide>(bb, call, "divide");
				}));

		map.Set("relax.exp", unary(fast ? static_cast<Unary>(topi::fast_exp) : topi::exp, "exp"));
		map.Set("relax.erf", unary(fast ? static_cast<Unary>(topi::fast_erf) : topi::erf, "erf"));
		map.Set("relax.tanh", unary(fast ? static_cast<Unary>(topi::fast_tanh) : topi::tanh, "tanh"));
		map.Set("relax.sqrt", unary(topi::sqrt, "sqrt"));
		map.Set("relax.rsqrt", unary(topi::rsqrt, "rsqrt"));
		map.Set("relax.log", unary(topi::log, "log"));
		map.Set("relax.cos", unary(topi::cos, "cos"));
		map.Set("relax.sin", unary(topi::sin, "sin"));
		map.Set("relax.tan", unary(topi::tan, "tan"));

		map.Set("relax.sum", reduction<topi::sum>("sum"));
		map.Set("relax.max", reduction<topi::max>("max"));
		map.Set("relax.min", reduction<topi::min>("min"));

		map.Set("relax.reshape", wrap([](const BlockBuilder& bb, const Call& call) {
					TECall te{bb, "reshape"};
					auto x = te.tensor(call->args[0]);
					const auto shape = (call->args[1]).as_or_throw<ShapeExpr>()->values;
					// Only unit dimensions inserted or removed: the same indices (topi::reshape divides by the
					// symbolic extents, which keeps the reshape from being inlined into a fused kernel).
					auto nonUnit = [](const Array<PrimExpr>& dims) {
						Array<PrimExpr> r;
						for (const auto& d : dims)
							if (!::tvm::prim::is_one(d))
								r.push_back(d);
						return r;
					};
					const auto from = nonUnit(x->shape), to = nonUnit(shape);
					::tvm::sym::Analyzer analyzer;
					bool units = from.size() == to.size();
					for (size_t i = 0; units && i < from.size(); ++i)
						units = analyzer->CanProveEqual(from[i], to[i]);
					if (!units)
						return te.finish(topi::reshape(x, shape));
					return te.finish(te::compute(
							shape,
							[&](const Array<::tvm::PrimVar>& i) {
								Array<PrimExpr> index;
								size_t k = 0;
								auto next = [&]() -> PrimExpr {
									while (::tvm::prim::is_one(shape[k]))
										++k;
									return i[k++];
								};
								for (const auto& d : x->shape)
									index.push_back(
											::tvm::prim::is_one(d) ? PrimExpr(::tvm::IntImm(d.ty(), 0)) : next()
									);
								return x(index);
							},
							"T_reshape"
					));
				}));
		map.Set("relax.permute_dims", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::PermuteDimsAttrs>();
					TECall te{bb, "transpose"};
					return te.finish(topi::transpose(te.tensor(call->args[0]), attrs->axes));
				}));
		map.Set("relax.broadcast_to", wrap([](const BlockBuilder& bb, const Call& call) {
					TECall te{bb, "broadcast_to"};
					auto x = te.tensor(call->args[0]);
					return te.finish(topi::broadcast_to(x, (call->args[1]).as_or_throw<ShapeExpr>()->values));
				}));

		map.Set("relax.concat", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::ConcatAttrs>();
					TECall te{bb, "concat"};
					Array<te::Tensor> parts;
					for (const auto& field : (call->args[0]).as_or_throw<::tvm::relax::Tuple>()->fields)
						parts.push_back(te.tensor(field));
					const int axis = attrs->axis ? static_cast<int>(attrs->axis.value()) : 0;
					return te.finish(topi::concatenate(parts, axis));
				}));
		map.Set("relax.stack", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::StackAttrs>();
					TECall te{bb, "stack"};
					Array<te::Tensor> parts;
					for (const auto& field : (call->args[0]).as_or_throw<::tvm::relax::Tuple>()->fields)
						parts.push_back(te.tensor(field));
					const int rank = static_cast<int>(parts[0]->shape.size());
					int axis = attrs->axis ? static_cast<int>(attrs->axis.value()) : 0;
					if (axis < 0)
						axis += rank + 1;
					// (not topi::stack: its new axis has an int32 extent, the Relax type an int64 one)
					Array<PrimExpr> shape;
					for (int d = 0; d < rank; ++d) {
						if (d == axis)
							shape.push_back(::tvm::IntImm(::tvm::PrimType::Int(64), static_cast<int64_t>(parts.size()))
							);
						shape.push_back(parts[0]->shape[d]);
					}
					if (axis == rank)
						shape.push_back(::tvm::IntImm(::tvm::PrimType::Int(64), static_cast<int64_t>(parts.size())));
					return te.finish(te::compute(
							shape,
							[&](const Array<::tvm::PrimVar>& i) {
								Array<PrimExpr> index;
								for (int d = 0; d <= rank; ++d)
									if (d != axis)
										index.push_back(i[d]);
								PrimExpr value = parts.back()(index);
								for (size_t k = parts.size() - 1; k-- > 0;)
									value = ::tvm::prim::Select(
											i[axis] == ::tvm::IntImm(i[axis].ty(), static_cast<int64_t>(k)),
											parts[k](index), value
									);
								return value;
							},
							"T_stack"
					));
				}));
		map.Set("relax.collapse_sum_to", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					// (Column sums by a kernel: kernels/onednn, sum-rows.)
					const auto& x = call->args[0];
					auto target = (call->args[1]).as_or_throw<ShapeExpr>()->values;
					TECall te{bb, "collapse_sum"};
					return te.finish(topi::collapse_sum(te.tensor(x), target));
				}));
		map.Set("relax.equal", wrap([](const BlockBuilder& bb, const Call& call) {
					TECall te{bb, "equal"};
					auto a = te.tensor(call->args[0], "lhs");
					auto b = te.tensor(call->args[1], "rhs");
					return te.finish(topi::equal(a, b));
				}));
		map.Set("relax.astype", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::AstypeAttrs>();
					TECall te{bb, "cast"};
					return te.finish(topi::cast(te.tensor(call->args[0]), attrs->dtype));
				}));

		map.Set("relax.matmul", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					const Expr &a = call->args[0], &b = call->args[1];
					if (shapeOf(a).size() < 2 || shapeOf(b).size() < 2) {
						// Vector operands: unsqueeze to matrices, multiply, squeeze the result.
						auto as = shapeOf(a), bs = shapeOf(b);
						TECall te{bb, "matvec"};
						auto ta = te.tensor(a, "A"), tb = te.tensor(b, "B");
						auto ma = as.size() == 1 ? topi::reshape(ta, {1, as[0]}) : ta;
						auto mb = bs.size() == 1 ? topi::reshape(tb, {bs[0], 1}) : tb;
						auto product = matmulCompute(ma, mb, std::nullopt, false, 1.0, false);
						Array<PrimExpr> outShape;
						for (size_t i = 0; i < product->shape.size(); ++i) {
							const bool squeezed = (as.size() == 1 && i == product->shape.size() - 2) ||
												  (bs.size() == 1 && i == product->shape.size() - 1);
							if (!squeezed)
								outShape.push_back(product->shape[i]);
						}
						return te.finish(topi::reshape(product, outShape));
					}
					return matmulTE(bb, MatmulArgs{a, b});
				}));
		map.Set("mdl.matmul",
				wrap([](const BlockBuilder& bb, const Call& call) -> Expr { return matmulTE(bb, MatmulArgs{call}); }));

		map.Set("relax.nn.softmax", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto* attrs = call->attrs.as<::tvm::relax::SoftmaxAttrs>();
					const auto& x = call->args[0];
					TECall te{bb, "softmax"};
					return te.finish(topi::nn::softmax(te.tensor(x), attrs->axis));
				}));
		map.Set("relax.nn.gelu", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto& x = call->args[0];
					TECall te{bb, "gelu"};
					auto t = te.tensor(x);
					return te.finish(te::compute(
							t->shape,
							[&](const Array<::tvm::PrimVar>& i) {
								PrimExpr v = t(Array<PrimExpr>{i.begin(), i.end()});
								auto c = [&](double d) { return ::tvm::prim::MakeConst(t->dtype, d); };
								return v * c(0.5) * (c(1.0) + ::tvm::prim::erf(v * c(M_SQRT1_2)));
							},
							"T_gelu", topi::kElementWise
					));
				}));
		map.Set("mdl.dequant", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					// W[n, k] = scales[k / group, n] * (q[n, k] - zeros[k / group, n]), in float32 (see QuantFormat).
					const Quantized quant{
							call->args[1],
							tensorInfo(call->args[2]) ? std::optional<Expr>{call->args[2]} : std::nullopt,
							static_cast<mdl::builtins::QuantFormat>(intValue(call->args[3])), intValue(call->args[4])
					};
					TECall te{bb, "dequant"};
					auto w = dequantCompute(te, call->args[0], quant);
					return te.finish(w);
				}));
		map.Set("mdl.suffix_mask", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					// out[b..., s] = s < lengths[b...] (the reference only gives the number of positions)
					TECall te{bb, "suffix_mask"};
					auto lengths = te.tensor(call->args[0], "lengths");
					const auto i64 = ::tvm::PrimType::Int(64);
					return te.finish(te::compute(
							suffixMaskShape(call),
							[&](const Array<::tvm::PrimVar>& i) {
								Array<PrimExpr> b(i.begin(), i.end() - 1);
								return ::tvm::prim::cast(i64, i[i.size() - 1]) < ::tvm::prim::cast(i64, lengths(b));
							},
							"T_suffix_mask", topi::kElementWise
					));
				}));
		map.Set("mdl.gelu_backward", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto &x = call->args[0], &dy = call->args[1];
					// dy * GELU'(x) = dy * (Φ(x) + x φ(x)), in float32.
					TECall te{bb, "gelu_backward"};
					auto tx = te.tensor(x, "x");
					auto tdy = te.tensor(dy, "dy");
					return te.finish(te::compute(
							tdy->shape,
							[&](const Array<::tvm::PrimVar>& i) {
								const auto f32 = ::tvm::PrimType::Float(32);
								Array<PrimExpr> index{i.begin(), i.end()};
								PrimExpr v = ::tvm::prim::cast(f32, tx(index));
								auto c = [&](double d) { return ::tvm::prim::MakeConst(f32, d); };
								PrimExpr grad = c(0.5) * (c(1.0) + ::tvm::prim::erf(v * c(M_SQRT1_2))) +
												v * ::tvm::prim::exp(c(-0.5) * v * v) * c(0.3989422804014327);
								return ::tvm::prim::cast(tdy->dtype, ::tvm::prim::cast(f32, tdy(index)) * grad);
							},
							"T_gelu_backward", topi::kElementWise
					));
				}));
		map.Set("relax.take", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::TakeAttrs>();
					TECall te{bb, "take"};
					auto table = te.tensor(call->args[0], "table");
					auto ids = te.tensor(call->args[1], "ids");
					const int axis = attrs->axis ? static_cast<int>(attrs->axis.value()) : 0;
					return te.finish(topi::take(table, ids, /*batch_dims=*/0, axis, std::string(attrs->mode)));
				}));
		map.Set("relax.where", wrap([](const BlockBuilder& bb, const Call& call) {
					TECall te{bb, "where"};
					auto c = te.tensor(call->args[0], "cond");
					auto a = te.tensor(call->args[1], "a");
					auto b = te.tensor(call->args[2], "b");
					return te.finish(topi::where(c, a, b));
				}));
		map.Set("relax.one_hot", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::OneHotAttrs>();
					const PrimExpr on = scalarValue(call->args[1]).value(), off = scalarValue(call->args[2]).value();
					TECall te{bb, "one_hot"};
					auto labels = te.tensor(call->args[0], "labels");
					// The output shape as Relax inferred it (int64 dims; TOPI would use an int32 depth).
					return te.finish(topi::one_hot(labels, on, off, attrs->depth, attrs->axis, on.ty(), shapeOf(call)));
				}));

		map.Set("relax.nn.layer_norm", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto* attrs = call->attrs.as<::tvm::relax::LayerNormAttrs>();
					const auto &x = call->args[0], &gamma = call->args[1], &beta = call->args[2];
					TECall te{bb, "layer_norm"};
					auto tx = te.tensor(x, "x");
					auto tg = te.tensor(gamma, "gamma");
					auto tb = te.tensor(beta, "beta");
					// TOPI's layer norm supports float32 and float16 only: bfloat16 is normalized in float32 (when
					// no kernel package replaces it, e.g. on GPUs).
					if (tx->dtype.MatchesElementType(kDLBfloat, 16)) {
						const auto f32 = ::tvm::PrimType::Float(32);
						auto y = topi::nn::layer_norm(
								topi::cast(tx, f32), topi::cast(tg, f32), topi::cast(tb, f32), attrs->axes,
								attrs->epsilon
						);
						return te.finish(topi::cast(y, tx->dtype));
					}
					return te.finish(topi::nn::layer_norm(tx, tg, tb, attrs->axes, attrs->epsilon));
				}));
		map.Set("mdl.scatter_add_rows", wrap([](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto out = ::tvm::relax::GetTypeAs<TensorTypeNode>(call);
					return callPrimFunc(
							bb, scatterAddRowsKernel(tensorInfo(call->args[1])->dtype.value()), "scatter_add_rows",
							{call->args[0], call->args[1]}, ::tvm::ffi::GetRef<TensorType>(out)
					);
				}));
		return map;
	}

} // namespace mdl::backend::tvm

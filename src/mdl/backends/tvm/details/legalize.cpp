#include "./legalize.hpp"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/relax/attrs/datatype.h>
#include <tvm/relax/attrs/index.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mdl::backend::tvm {

	using ::tvm::Integer;
	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	using ::tvm::ffi::Function;
	using ::tvm::ffi::Map;
	using ::tvm::ffi::String;
	using ::tvm::relax::BlockBuilder;
	using ::tvm::relax::Call;
	using ::tvm::relax::Expr;
	using ::tvm::relax::ShapeExpr;
	using ::tvm::relax::TensorStructInfo;
	using ::tvm::relax::TensorStructInfoNode;
	namespace te = ::tvm::te;
	namespace tirx = ::tvm::tirx;
	namespace topi = ::tvm::topi;

	// ── helpers ──────────────────────────────────────────────────────────────────

	namespace {
		const TensorStructInfoNode* tensorInfo(const Expr& e) {
			return ::tvm::relax::GetStructInfoAs<TensorStructInfoNode>(e);
		}

		Array<PrimExpr> shapeOf(const Expr& e) {
			const auto* info = tensorInfo(e);
			if (info == nullptr || !info->GetShape().defined())
				throw std::runtime_error("mdl backend: expected a tensor with known shape");
			return info->GetShape().value();
		}

		int64_t intValue(const Expr& e) {
			const auto* pv = e.as<::tvm::relax::PrimValueNode>();
			const auto* imm = pv ? pv->value.as<::tvm::IntImmNode>() : nullptr;
			if (imm == nullptr)
				throw std::runtime_error("mdl backend: expected an integer PrimValue");
			return imm->value;
		}

		double floatValue(const Expr& e) {
			const auto* pv = e.as<::tvm::relax::PrimValueNode>();
			if (pv != nullptr) {
				if (const auto* f = pv->value.as<::tvm::FloatImmNode>())
					return f->value;
				if (const auto* i = pv->value.as<::tvm::IntImmNode>())
					return static_cast<double>(i->value);
			}
			throw std::runtime_error("mdl backend: expected a scalar PrimValue");
		}

		PrimExpr product(const Array<PrimExpr>& dims, size_t begin, size_t end) {
			PrimExpr result = ::tvm::IntImm(::tvm::DataType::Int(64), 1);
			for (size_t i = begin; i < end; ++i)
				result = result * ::tvm::cast(::tvm::DataType::Int(64), dims[i]);
			return result;
		}

		bool isLastAxis(int64_t axis, size_t ndim) { return axis == -1 || axis == static_cast<int64_t>(ndim) - 1; }

		/**
		 * @brief The symbolic variables in @p shapes that no shape determines directly (as a whole dimension), e.g. `B0`
		 * and `B1` of a flattened dimension `B0 * B1`. A kernel cannot recover them from its buffers, so they are passed
		 * as extra arguments (`tir_vars` of `call_tir`).
		 */
		Array<tirx::Var> unboundVars(const std::vector<Array<PrimExpr>>& shapes) {
			std::vector<const tirx::VarNode*> direct;
			for (const auto& shape : shapes)
				for (const auto& d : shape)
					if (const auto* v = d.as<tirx::VarNode>())
						direct.push_back(v);
			Array<tirx::Var> unbound;
			for (const auto& shape : shapes)
				for (const auto& d : shape)
					for (const auto& v : tirx::UndefinedVars(d))
						if (std::find(direct.begin(), direct.end(), v.get()) == direct.end() &&
							std::find_if(unbound.begin(), unbound.end(), [&](const tirx::Var& u) { return u.same_as(v); }) ==
									unbound.end())
							unbound.push_back(v);
			return unbound;
		}

		/** @brief `call_tir(gv, inputs, out)`, with the unbound symbolic variables as `tir_vars`. */
		Expr callTir(const ::tvm::GlobalVar& gv, const Array<Expr>& inputs, const TensorStructInfo& out,
					 const Array<tirx::Var>& vars) {
			static const auto& op = ::tvm::Op::Get("relax.call_tir");
			if (vars.empty())
				return Call(op, {gv, ::tvm::relax::Tuple(inputs)}, ::tvm::Attrs(), {out});
			Array<PrimExpr> values(vars.begin(), vars.end());
			return Call(op, {gv, ::tvm::relax::Tuple(inputs), ShapeExpr(values)}, ::tvm::Attrs(), {out});
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
				auto placeholder = te::placeholder(shapeOf(arg), info->dtype, argName);
				shapes.push_back(placeholder->shape);
				inputs.push_back(arg);
				argList.push_back(placeholder);
				return placeholder;
			}

			/** @brief The value of a scalar (PrimValue) argument. */
			static PrimExpr scalar(const Expr& arg, ::tvm::DataType dtype) {
				const auto* pv = arg.as<::tvm::relax::PrimValueNode>();
				if (pv == nullptr)
					throw std::runtime_error("mdl backend: expected a tensor or PrimValue operand");
				return ::tvm::cast(dtype, pv->value);
			}

			Expr finish(const te::Tensor& out) {
				argList.push_back(out);
				shapes.push_back(out->shape);
				auto vars = unboundVars(shapes);
				for (const auto& v : vars)
					argList.push_back(v);
				static const auto createPrimFunc = Function::GetGlobalRequired("te.CreatePrimFunc");
				auto func = createPrimFunc(argList, nullptr).cast<tirx::PrimFunc>();
				auto gv = bb->AddFunction(func, name);
				return callTir(gv, inputs, TensorStructInfo(ShapeExpr(out->shape), out->dtype), vars);
			}

		private:
			BlockBuilder bb;
			std::string name;
			Array<Expr> inputs;
			Array<::tvm::ffi::ObjectRef> argList;
			std::vector<Array<PrimExpr>> shapes;
		};

		/** @brief Builds a PrimFunc that calls an extern kernel and emits a `call_tir` to it. */
		class ExternCall {
		public:
			ExternCall(BlockBuilder bb, std::string kernel) : bb(std::move(bb)), kernel(std::move(kernel)) {}

			/** @brief Adds a tensor argument; returns the pointer expression to pass to the kernel. */
			PrimExpr input(const Expr& arg, const std::string& argName) {
				inputs.push_back(arg);
				return buffer(shapeOf(arg), tensorInfo(arg)->dtype, argName, /*write=*/false);
			}

			PrimExpr output(const Array<PrimExpr>& shape, ::tvm::DataType dtype) {
				outInfo = TensorStructInfo(ShapeExpr(shape), dtype);
				outputPtr = buffer(shape, dtype, "out", /*write=*/true);
				outputHandle = params.back();
				return outputPtr;
			}

			Expr finish(std::vector<PrimExpr> kernelArgs) {
				// The output buffer is the last parameter (destination passing style); it was added before the inputs'
				// pointers were computed only if output() was called first, so reorder explicitly.
				std::vector<tirx::Var> ordered;
				for (const auto& p : params)
					if (!p.same_as(outputHandle))
						ordered.push_back(p);
				ordered.push_back(outputHandle);
				auto vars = unboundVars(shapes);
				ordered.insert(ordered.end(), vars.begin(), vars.end());
				Array<PrimExpr> callArgs{tirx::StringImm(kernel)};
				for (auto& arg : kernelArgs)
					callArgs.push_back(std::move(arg));
				auto call = tirx::Call(::tvm::DataType::Int(32), tirx::builtin::call_extern(), callArgs);
				auto check = tirx::AssertStmt(
						call == 0, tirx::StringImm("RuntimeError"),
						{tirx::StringImm("mdl kernel " + kernel + " failed")}
				);
				// Wrap the call in a root block like any other (TE-generated) PrimFunc; Relax analyses expect it.
				auto body = tirx::SBlockRealize(
						{}, ::tvm::Bool(true), tirx::SBlock({}, {}, {}, "root", check)
				);
				auto func = tirx::PrimFunc(Array<tirx::Var>(ordered.begin(), ordered.end()), body, ::tvm::VoidType(), bufferMap);
				// Opaque (kOpaque = 8): extern calls must not be fused with other kernels.
				func = ::tvm::WithAttr(std::move(func), "op_pattern", ::tvm::Integer(8));
				auto gv = bb->AddFunction(func, kernel);
				return callTir(gv, inputs, outInfo.value(), vars);
			}

		private:
			BlockBuilder bb;
			std::string kernel;
			Array<Expr> inputs;
			std::vector<tirx::Var> params;
			Map<tirx::Var, tirx::Buffer> bufferMap;
			std::vector<Array<PrimExpr>> shapes;
			std::optional<TensorStructInfo> outInfo;
			PrimExpr outputPtr;
			tirx::Var outputHandle;

			PrimExpr buffer(const Array<PrimExpr>& shape, ::tvm::DataType dtype, const std::string& n, bool write) {
				// The parameter is the DLTensor handle; the buffer has its own data pointer (bound by MakePackedAPI).
				auto handle = tirx::Var(n + "_handle", ::tvm::DataType::Handle());
				auto buf = tirx::decl_buffer(shape, dtype, n);
				shapes.push_back(shape);
				params.push_back(handle);
				bufferMap.Set(handle, buf);
				return buf.access_ptr(write ? 2 : 1);
			}
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
				return te.finish(TT(te.tensor(call->args[0], "lhs"), te.tensor(call->args[1], "rhs"), "T_" + name,
									topi::kBroadcast));
			if (lhsInfo)
				return te.finish(TS(te.tensor(call->args[0], "lhs"), TECall::scalar(call->args[1], lhsInfo->dtype),
									"T_" + name, topi::kElementWise));
			if (rhsInfo)
				return te.finish(ST(TECall::scalar(call->args[0], rhsInfo->dtype), te.tensor(call->args[1], "rhs"),
									"T_" + name, topi::kElementWise));
			return call; // scalar arithmetic is handled by ComputePrimValue
		}

		using Unary = te::Tensor (*)(const te::Tensor&, std::string, std::string);

		Function unary(Unary fn, std::string name) {
			return wrap([fn, name](const BlockBuilder& bb, const Call& call) {
				TECall te{bb, name};
				return te.finish(fn(te.tensor(call->args[0]), "T_" + name, topi::kElementWise));
			});
		}

		template <te::Tensor (*Reduce)(const te::Tensor&, const ::tvm::ffi::Optional<Array<Integer>>&, bool, bool)>
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
			te::Tensor product = te::compute(
					outShape,
					[&](const Array<tirx::Var>& idx) {
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
						return ::tvm::sum(a(ai) * b(bi), {k});
					},
					"T_matmul", "opaque"
			);
			if (alpha == 1.0 && !bias)
				return product;
			// Epilogue as a separate stage (a reduction must be the top-level expression of a compute).
			return te::compute(
					outShape,
					[&](const Array<tirx::Var>& idx) {
						PrimExpr v = product(Array<PrimExpr>{idx.begin(), idx.end()});
						if (alpha != 1.0)
							v = v * ::tvm::tirx::make_const(dtype, alpha);
						if (bias)
							v = v + (*bias)(Array<PrimExpr>{idx[batchRank + 1]});
						return v;
					},
					"T_matmul_epilogue", topi::kElementWise
			);
		}

		// ── vendor kernels (mdl_kernels) ───────────────────────────────────────────

		/**
		 * @brief Lowers `alpha * (a @ op(b)) + bias` to mdl_matmul_f32 if the shapes are supported (a `[..., M, K]`
		 * with 2-D b, or equal-rank batched a and b where b's batch is 1 or equal to a's batch).
		 */
		std::optional<Expr> matmulKernel(
				const BlockBuilder& bb, const Expr& a, const Expr& b, std::optional<Expr> bias, bool transB,
				double alpha, bool transA
		) {
			const auto* ai = tensorInfo(a);
			const auto* bi = tensorInfo(b);
			if (!ai || !bi || ai->dtype != ::tvm::DataType::Float(32) || bi->dtype != ::tvm::DataType::Float(32))
				return std::nullopt;
			auto as = shapeOf(a), bs = shapeOf(b);
			const size_t ra = as.size(), rb = bs.size();
			if (ra < 2 || rb < 2)
				return std::nullopt;
			PrimExpr M, K = transA ? as[ra - 2] : as[ra - 1], N = transB ? bs[rb - 2] : bs[rb - 1];
			PrimExpr batch, batchB;
			Array<PrimExpr> outShape;
			if (rb == 2 && transA) {
				if (ra != 2)
					return std::nullopt;
				M = as[1];
				batch = batchB = ::tvm::IntImm(::tvm::DataType::Int(64), 1);
				outShape.push_back(M);
			} else if (rb == 2) {
				M = product(as, 0, ra - 1); // fold all leading dims of a into M
				batch = batchB = ::tvm::IntImm(::tvm::DataType::Int(64), 1);
				for (size_t i = 0; i + 1 < ra; ++i)
					outShape.push_back(as[i]);
			} else if (ra == rb) {
				::tvm::arith::Analyzer analyzer;
				bool bBroadcast = true, sameBatch = true;
				for (size_t i = 0; i + 2 < ra; ++i) {
					bBroadcast &= analyzer.CanProveEqual(bs[i], 1);
					sameBatch &= analyzer.CanProveEqual(as[i], bs[i]);
					outShape.push_back(as[i]);
				}
				if (!bBroadcast && !sameBatch)
					return std::nullopt;
				M = transA ? as[ra - 1] : as[ra - 2];
				batch = product(as, 0, ra - 2);
				batchB = bBroadcast ? PrimExpr(::tvm::IntImm(::tvm::DataType::Int(64), 1)) : batch;
				outShape.push_back(M);
			} else {
				return std::nullopt;
			}
			outShape.push_back(N);

			ExternCall ext{bb, "mdl_matmul_f32"};
			auto pa = ext.input(a, "A");
			auto pb = ext.input(b, "B");
			PrimExpr pbias = bias ? ext.input(*bias, "bias") : pa; // ignored if has_bias == 0
			auto pout = ext.output(outShape, ::tvm::DataType::Float(32));
			auto i64 = [](PrimExpr e) { return ::tvm::cast(::tvm::DataType::Int(64), e); };
			return ext.finish(
					{pout, pa, pb, pbias, ::tvm::IntImm(::tvm::DataType::Int(32), bias ? 1 : 0), i64(batch),
					 i64(batchB), i64(M), i64(K), i64(N), ::tvm::IntImm(::tvm::DataType::Int(32), transA ? 1 : 0),
					 ::tvm::IntImm(::tvm::DataType::Int(32), transB ? 1 : 0),
					 ::tvm::FloatImm(::tvm::DataType::Float(32), alpha)}
			);
		}

		Expr matmulTE(
				const BlockBuilder& bb, const Expr& a, const Expr& b, std::optional<Expr> bias, bool transB,
				double alpha, bool transA
		) {
			TECall te{bb, "matmul"};
			auto ta = te.tensor(a, "A");
			auto tb = te.tensor(b, "B");
			std::optional<te::Tensor> tbias;
			if (bias)
				tbias = te.tensor(*bias, "bias");
			return te.finish(matmulCompute(ta, tb, tbias, transB, alpha, transA));
		}

		/** @brief Splits mdl.matmul's arguments: (a, b, bias or PrimValue(0), trans_b, alpha, trans_a). */
		struct MatmulArgs {
			Expr a, b;
			std::optional<Expr> bias;
			bool transB;
			double alpha;
			bool transA;

			explicit MatmulArgs(const Call& call)
					: a(call->args[0]), b(call->args[1]),
					  bias(tensorInfo(call->args[2]) ? std::optional<Expr>{call->args[2]} : std::nullopt),
					  transB(intValue(call->args[3]) != 0), alpha(floatValue(call->args[4])),
					  transA(intValue(call->args[5]) != 0) {}
		};

		// ── mdl.matmul operator ───────────────────────────────────────────────────

		::tvm::relax::StructInfo inferMatmul(const Call& call, const BlockBuilder& ctx) {
			MatmulArgs args{call};
			const auto* ai = tensorInfo(args.a);
			auto as = shapeOf(args.a), bs = shapeOf(args.b);
			Array<PrimExpr> out;
			for (size_t i = 0; i + 2 < as.size(); ++i)
				out.push_back(as[i]);
			out.push_back(args.transA ? as[as.size() - 1] : as[as.size() - 2]);
			out.push_back(args.transB ? bs[bs.size() - 2] : bs[bs.size() - 1]);
			return TensorStructInfo(ShapeExpr(out), ai->dtype, ai->vdevice);
		}
	} // namespace

	Expr makeMatmul(Expr a, Expr b, std::optional<Expr> bias, bool transB, double alpha, bool transA) {
		static const auto& op = ::tvm::Op::Get("mdl.matmul");
		Expr biasArg = bias ? *bias : Expr(::tvm::relax::PrimValue::Int64(0));
		return Call(
				op, {a, b, biasArg, ::tvm::relax::PrimValue::Int64(transB ? 1 : 0),
					 ::tvm::relax::PrimValue(::tvm::FloatImm(::tvm::DataType::Float(64), alpha)),
					 ::tvm::relax::PrimValue::Int64(transA ? 1 : 0)}
		);
	}

	TVM_REGISTER_OP("mdl.matmul")
			.set_num_inputs(6)
			.add_argument("a", "Tensor", "The left operand [..., M, K].")
			.add_argument("b", "Tensor", "The right operand [K, N] (or [N, K] if transposed), possibly batched.")
			.add_argument("bias", "Tensor | PrimValue", "The bias [N] or PrimValue(0) for no bias.")
			.add_argument("trans_b", "PrimValue", "Whether b is transposed (last two dims).")
			.add_argument("alpha", "PrimValue", "Scale applied to the product.")
			.add_argument("trans_a", "PrimValue", "Whether a is transposed (last two dims).")
			.set_attr<::tvm::relax::FInferStructInfo>("FInferStructInfo", inferMatmul)
			.set_attr<::tvm::Bool>("FPurity", ::tvm::Bool(true));

	// ── handler map ──────────────────────────────────────────────────────────────

	Map<String, Function> legalizeMap(const mdl::CompileOptions& opts) {
		const bool vendor = opts.kernels == mdl::Kernels::OneDnn;
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
					return te.finish(topi::reshape(x, ::tvm::Downcast<ShapeExpr>(call->args[1])->values));
				}));
		map.Set("relax.permute_dims", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::PermuteDimsAttrs>();
					TECall te{bb, "transpose"};
					return te.finish(topi::transpose(te.tensor(call->args[0]), attrs->axes));
				}));
		map.Set("relax.broadcast_to", wrap([](const BlockBuilder& bb, const Call& call) {
					TECall te{bb, "broadcast_to"};
					auto x = te.tensor(call->args[0]);
					return te.finish(topi::broadcast_to(x, ::tvm::Downcast<ShapeExpr>(call->args[1])->values));
				}));

		map.Set("relax.collapse_sum_to", wrap([](const BlockBuilder& bb, const Call& call) {
					TECall te{bb, "collapse_sum"};
					auto x = te.tensor(call->args[0]);
					return te.finish(topi::collapse_sum(x, ::tvm::Downcast<ShapeExpr>(call->args[1])->values));
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

		map.Set("relax.matmul", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const Expr &a = call->args[0], &b = call->args[1];
					if (vendor)
						if (auto e = matmulKernel(bb, a, b, std::nullopt, false, 1.0, false))
							return *e;
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
					return matmulTE(bb, a, b, std::nullopt, false, 1.0, false);
				}));
		map.Set("mdl.matmul", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					MatmulArgs args{call};
					if (vendor)
						if (auto e = matmulKernel(bb, args.a, args.b, args.bias, args.transB, args.alpha, args.transA))
							return *e;
					return matmulTE(bb, args.a, args.b, args.bias, args.transB, args.alpha, args.transA);
				}));

		map.Set("relax.nn.softmax", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto* attrs = call->attrs.as<::tvm::relax::SoftmaxAttrs>();
					const auto& x = call->args[0];
					auto shape = shapeOf(x);
					if (vendor && isLastAxis(attrs->axis, shape.size()) &&
						tensorInfo(x)->dtype == ::tvm::DataType::Float(32)) {
						ExternCall ext{bb, "mdl_softmax_f32"};
						auto px = ext.input(x, "x");
						auto pout = ext.output(shape, ::tvm::DataType::Float(32));
						return ext.finish({pout, px, product(shape, 0, shape.size() - 1), product(shape, shape.size() - 1, shape.size())});
					}
					TECall te{bb, "softmax"};
					return te.finish(topi::nn::softmax(te.tensor(x), attrs->axis));
				}));
		map.Set("relax.nn.gelu", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto& x = call->args[0];
					auto shape = shapeOf(x);
					if (vendor && tensorInfo(x)->dtype == ::tvm::DataType::Float(32)) {
						ExternCall ext{bb, "mdl_gelu_f32"};
						auto px = ext.input(x, "x");
						auto pout = ext.output(shape, ::tvm::DataType::Float(32));
						return ext.finish({pout, px, product(shape, 0, shape.size())});
					}
					TECall te{bb, "gelu"};
					auto t = te.tensor(x);
					return te.finish(te::compute(
							t->shape,
							[&](const Array<tirx::Var>& i) {
								PrimExpr v = t(Array<PrimExpr>{i.begin(), i.end()});
								auto c = [&](double d) { return ::tvm::tirx::make_const(t->dtype, d); };
								return v * c(0.5) * (c(1.0) + ::tvm::erf(v * c(M_SQRT1_2)));
							},
							"T_gelu", topi::kElementWise
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
		map.Set("relax.one_hot", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::OneHotAttrs>();
					const auto* on = call->args[1].as<::tvm::relax::PrimValueNode>();
					const auto* off = call->args[2].as<::tvm::relax::PrimValueNode>();
					TECall te{bb, "one_hot"};
					auto labels = te.tensor(call->args[0], "labels");
					return te.finish(topi::one_hot(labels, on->value, off->value, attrs->depth, attrs->axis, on->value.dtype()));
				}));

		map.Set("relax.nn.layer_norm", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto* attrs = call->attrs.as<::tvm::relax::LayerNormAttrs>();
					const auto &x = call->args[0], &gamma = call->args[1], &beta = call->args[2];
					auto shape = shapeOf(x);
					const bool lastAxis = attrs->axes.size() == 1 && isLastAxis(attrs->axes[0]->value, shape.size());
					if (vendor && lastAxis && attrs->center && attrs->scale &&
						tensorInfo(x)->dtype == ::tvm::DataType::Float(32)) {
						ExternCall ext{bb, "mdl_layer_norm_f32"};
						auto px = ext.input(x, "x");
						auto pg = ext.input(gamma, "gamma");
						auto pb = ext.input(beta, "beta");
						auto pout = ext.output(shape, ::tvm::DataType::Float(32));
						return ext.finish(
								{pout, px, pg, pb, product(shape, 0, shape.size() - 1),
								 product(shape, shape.size() - 1, shape.size()),
								 ::tvm::FloatImm(::tvm::DataType::Float(32), attrs->epsilon)}
						);
					}
					TECall te{bb, "layer_norm"};
					auto tx = te.tensor(x, "x");
					auto tg = te.tensor(gamma, "gamma");
					auto tb = te.tensor(beta, "beta");
					return te.finish(topi::nn::layer_norm(tx, tg, tb, attrs->axes, attrs->epsilon));
				}));
		return map;
	}

} // namespace mdl::backend::tvm

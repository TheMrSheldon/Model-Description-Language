#include "./legalize.hpp"

#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/relax/attrs/datatype.h>
#include <tvm/relax/attrs/index.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
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
#include <numeric>
#include <unordered_map>
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
		Array<tirx::Var> unboundVars(const std::vector<Array<PrimExpr>>& shapes, const std::vector<PrimExpr>& scalars = {}) {
			std::vector<const tirx::VarNode*> direct;
			for (const auto& shape : shapes)
				for (const auto& d : shape)
					if (const auto* v = d.as<tirx::VarNode>())
						direct.push_back(v);
			std::vector<Array<PrimExpr>> all = shapes;
			all.emplace_back(scalars.begin(), scalars.end()); // e.g. a runtime dimension as a divisor
			Array<tirx::Var> unbound;
			for (const auto& shape : all)
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

			/** @brief The value of a scalar (PrimValue) argument (its symbolic variables become kernel arguments). */
			PrimExpr scalar(const Expr& arg, ::tvm::DataType dtype) {
				const auto* pv = arg.as<::tvm::relax::PrimValueNode>();
				if (pv == nullptr)
					throw std::runtime_error("mdl backend: expected a tensor or PrimValue operand");
				scalars.push_back(pv->value);
				return ::tvm::cast(dtype, pv->value);
			}

			Expr finish(const te::Tensor& out) {
				argList.push_back(out);
				shapes.push_back(out->shape);
				auto vars = unboundVars(shapes, scalars);
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
			std::vector<PrimExpr> scalars;
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
				return te.finish(TS(te.tensor(call->args[0], "lhs"), te.scalar(call->args[1], lhsInfo->dtype),
									"T_" + name, topi::kElementWise));
			if (rhsInfo)
				return te.finish(ST(te.scalar(call->args[0], rhsInfo->dtype), te.tensor(call->args[1], "rhs"),
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
			const auto f32 = ::tvm::DataType::Float(32);
			const bool wide = dtype.bits() == 16 && (dtype.is_float() || dtype.is_bfloat16());
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
						// 16-bit floats accumulate in float32.
						if (wide)
							return ::tvm::sum(::tvm::cast(f32, a(ai)) * ::tvm::cast(f32, b(bi)), {k});
						return ::tvm::sum(a(ai) * b(bi), {k});
					},
					"T_matmul", "opaque"
			);
			if (alpha == 1.0 && !bias && !wide)
				return product;
			// Epilogue as a separate stage (a reduction must be the top-level expression of a compute).
			return te::compute(
					outShape,
					[&](const Array<tirx::Var>& idx) {
						PrimExpr v = product(Array<PrimExpr>{idx.begin(), idx.end()});
						const auto accumulator = wide ? f32 : dtype;
						if (alpha != 1.0)
							v = v * ::tvm::tirx::make_const(accumulator, alpha);
						if (bias)
							v = v + ::tvm::cast(accumulator, (*bias)(Array<PrimExpr>{idx[batchRank + 1]}));
						return wide ? ::tvm::cast(dtype, v) : v;
					},
					"T_matmul_epilogue", topi::kElementWise
			);
		}

		// ── vendor kernels (mdl_kernels) ───────────────────────────────────────────

		/** @brief The suffix of the vendor kernels for the element type @p dtype (`mdl_matmul_bf16`), if any. */
		std::optional<std::string> kernelSuffix(::tvm::DataType dtype) {
			if (dtype == ::tvm::DataType::Float(32))
				return "f32";
			if (dtype == ::tvm::DataType::BFloat(16))
				return "bf16";
			if (dtype == ::tvm::DataType::Float(16))
				return "f16";
			return std::nullopt;
		}

		/** @brief A permutation of the dimensions of a tensor (empty: none). */
		using Perm = std::vector<int64_t>;

		Perm permOf(const Expr& e) {
			Perm perm;
			if (const auto* shape = e.as<::tvm::relax::ShapeExprNode>())
				for (const auto& v : shape->values)
					perm.push_back(v.as<::tvm::IntImmNode>()->value);
			return perm;
		}

		Expr permExpr(const Perm& perm) {
			Array<PrimExpr> values;
			for (int64_t p : perm)
				values.push_back(::tvm::IntImm(::tvm::DataType::Int(64), p));
			return ShapeExpr(values);
		}

		/** @brief permute(xs, perm): entry i is xs[perm[i]] (like relax.permute_dims). */
		Array<PrimExpr> permuted(const Array<PrimExpr>& xs, const Perm& perm) {
			if (perm.empty())
				return xs;
			Array<PrimExpr> out;
			for (int64_t p : perm)
				out.push_back(xs[p]);
			return out;
		}

		/** @brief permute(permute(x, inner), outer) = permute(x, compose(outer, inner)). */
		Perm compose(const Perm& outer, const Perm& inner) {
			if (outer.empty())
				return inner;
			if (inner.empty())
				return outer;
			Perm r;
			for (int64_t p : outer)
				r.push_back(inner[p]);
			return r;
		}

		Perm swapLastTwo(size_t rank) {
			Perm p(rank);
			std::iota(p.begin(), p.end(), 0);
			std::swap(p[rank - 1], p[rank - 2]);
			return p;
		}

		bool isIdentity(const Perm& p) {
			for (size_t i = 0; i < p.size(); ++i)
				if (p[i] != static_cast<int64_t>(i))
					return false;
			return true;
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
			const auto f32 = ::tvm::DataType::Float(32), i32 = ::tvm::DataType::Int(32);
			const int64_t group = quant.group;
			const auto format = quant.format;
			Array<PrimExpr> shape{q->shape[0], quant.packed() ? q->shape[1] * 2 : q->shape[1]};
			return te::compute(
					shape,
					[&](const Array<tirx::Var>& i) {
						const PrimExpr n = i[0], k = i[1];
						const PrimExpr g = ::tvm::floordiv(k, ::tvm::IntImm(k.dtype(), group));
						PrimExpr value;
						if (format == QuantFormat::int8) {
							value = ::tvm::cast(i32, q(Array<PrimExpr>{n, k}));
						} else {
							const PrimExpr two = ::tvm::IntImm(k.dtype(), 2);
							const PrimExpr byte = ::tvm::cast(i32, q(Array<PrimExpr>{n, ::tvm::floordiv(k, two)})) & 0xFF;
							const PrimExpr nibble = ::tvm::tirx::Select(
									::tvm::floormod(k, two) == ::tvm::IntImm(k.dtype(), 0), byte & 0xF, (byte >> 4) & 0xF
							);
							value = format == QuantFormat::int4 ? ::tvm::tirx::Select(nibble >= 8, nibble - 16, nibble) // signed
																: nibble;
						}
						if (z)
							value = value - ::tvm::cast(i32, (*z)(Array<PrimExpr>{g, n}));
						const PrimExpr scale = ::tvm::cast(f32, scales(Array<PrimExpr>{g, n}));
						return ::tvm::cast(scales->dtype, ::tvm::cast(f32, value) * scale);
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
		 * @brief mdl.matmul's arguments: (a, b, bias or PrimValue(0), perm_a, perm_b, perm_out, alpha, activation,
		 * addend or PrimValue(0), scales, zeros, quant). The operands are A = permute(a, perm_a) and B = permute(b,
		 * perm_b); the result is permute(act(alpha * A @ B + bias), perm_out) + addend (the epilogue: e.g. GELU, a
		 * residual connection). If quant is [format, group], b holds quantized values and B is permute(dequant(b,
		 * scales, zeros), perm_b) (see mdl.dequant); otherwise scales and zeros are PrimValue(0) and quant is empty.
		 */
		struct MatmulArgs {
			Expr a, b;
			std::optional<Expr> bias;
			Perm permA, permB, permOut;
			double alpha = 1.0;
			int64_t activation = 0; ///< 0: none, 1: GELU (erf)
			std::optional<Expr> addend;
			std::optional<Quantized> quant;

			MatmulArgs(Expr a, Expr b) : a(std::move(a)), b(std::move(b)) {}

			explicit MatmulArgs(const Call& call)
					: a(call->args[0]), b(call->args[1]),
					  bias(tensorInfo(call->args[2]) ? std::optional<Expr>{call->args[2]} : std::nullopt),
					  permA(permOf(call->args[3])), permB(permOf(call->args[4])), permOut(permOf(call->args[5])),
					  alpha(floatValue(call->args[6])), activation(intValue(call->args[7])),
					  addend(tensorInfo(call->args[8]) ? std::optional<Expr>{call->args[8]} : std::nullopt),
					  quant(quantizedOf(call, 9)) {}

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

		/** @brief Row-major strides of a contiguous tensor of shape @p shape. */
		Array<PrimExpr> contiguousStrides(const Array<PrimExpr>& shape) {
			std::vector<PrimExpr> strides(shape.size());
			PrimExpr s = ::tvm::IntImm(::tvm::DataType::Int(64), 1);
			for (size_t i = shape.size(); i-- > 0;) {
				strides[i] = s;
				s = s * ::tvm::cast(::tvm::DataType::Int(64), shape[i]);
			}
			return Array<PrimExpr>(strides.begin(), strides.end());
		}

		/**
		 * @brief Lowers mdl.matmul to mdl_matmul_<type> on strided views of the operands and the result: the
		 * permutations become strides, so permuted operands (e.g. the heads of attention) are read and written in
		 * place. Supports at most two batch dimensions after merging (b's batch dimensions may be broadcast).
		 */
		std::optional<Expr> matmulKernel(const BlockBuilder& bb, const MatmulArgs& m) {
			const auto* ai = tensorInfo(m.a);
			const auto* bi = tensorInfo(m.quant ? m.quant->scales : m.b);
			if (!ai || !bi || ai->dtype != bi->dtype || !kernelSuffix(ai->dtype))
				return std::nullopt;
			const auto as = shapeOf(m.a), bs = m.bShape();
			const auto la = permuted(as, m.permA), lb = permuted(bs, m.permB);
			const auto sa = permuted(contiguousStrides(as), m.permA), sb = permuted(contiguousStrides(bs), m.permB);
			const size_t ra = la.size(), rb = lb.size();
			if (ra < 2 || rb < 2 || ra < rb || (rb > 2 && ra != rb))
				return std::nullopt;
			if (m.addend && (tensorInfo(*m.addend)->dtype != ai->dtype ||
							 !::tvm::ffi::StructuralEqual()(shapeOf(*m.addend), permuted(m.productShape(), m.permOut))))
				return std::nullopt;
			// The result: the product's dimension j is the result's dimension at position p with perm_out[p] == j.
			const auto lo = m.productShape();
			const auto resultShape = permuted(lo, m.permOut);
			const auto rs = contiguousStrides(resultShape);
			std::vector<PrimExpr> so(lo.size());
			for (size_t p = 0; p < lo.size(); ++p)
				so[m.permOut.empty() ? p : static_cast<size_t>(m.permOut[p])] = rs[p];

			const PrimExpr zero = ::tvm::IntImm(::tvm::DataType::Int(64), 0);
			::tvm::arith::Analyzer analyzer;
			auto equal = [&](const PrimExpr& x, const PrimExpr& y) { return analyzer.CanProveEqual(x, y); };
			struct Dim {
				PrimExpr size, a, b, out; // the size and the strides in A, B, and the result
			};
			std::vector<Dim> batch;
			for (size_t i = 0; i + 2 < ra; ++i) {
				PrimExpr bStride = zero; // b broadcast over this dimension (2-D b, or size 1)
				if (rb > 2 && !equal(lb[i], 1))
					bStride = sb[i];
				batch.push_back({la[i], sa[i], bStride, so[i]});
			}
			PrimExpr M = la[ra - 2], K = la[ra - 1], N = lb[rb - 1];
			PrimExpr aM = sa[ra - 2], aK = sa[ra - 1], bK = sb[rb - 2], bN = sb[rb - 1];
			PrimExpr oM = so[lo.size() - 2], oN = so[lo.size() - 1];
			// Batch dimensions over which b does not vary and that continue M in a and the result belong to M (one
			// larger GEMM, e.g. a linear layer over [batch, seq]).
			while (!batch.empty() && equal(batch.back().b, zero) && equal(batch.back().a, aM * M) &&
				   equal(batch.back().out, oM * M)) {
				M = M * batch.back().size;
				batch.pop_back();
			}
			// Adjacent batch dimensions that are one dimension in all operands.
			for (size_t i = batch.size(); i-- > 1;) {
				const Dim outer = batch[i - 1], inner = batch[i];
				if (equal(outer.a, inner.a * inner.size) && equal(outer.out, inner.out * inner.size) &&
					(equal(outer.b, inner.b * inner.size) || (equal(outer.b, zero) && equal(inner.b, zero)))) {
					batch[i - 1] = {outer.size * inner.size, inner.a, inner.b, inner.out};
					batch.erase(batch.begin() + static_cast<std::ptrdiff_t>(i));
				}
			}
			if (m.quant) {
				// The quantized weights W [N, K] as B = W^T (a linear layer): a single GEMM with B[k, n] at n * K + k.
				if (!batch.empty() || rb != 2 || !equal(bK, 1) || !equal(bN, K))
					return std::nullopt;
				ExternCall ext{bb, "mdl_matmul_q_" + *kernelSuffix(ai->dtype)};
				auto pa = ext.input(m.a, "A");
				auto pq = ext.input(m.b, "q");
				auto pscales = ext.input(m.quant->scales, "scales");
				PrimExpr pzeros = m.quant->zeros ? ext.input(*m.quant->zeros, "zeros") : pq; // ignored without zeros
				PrimExpr pbias = m.bias ? ext.input(*m.bias, "bias") : pa;
				PrimExpr paddend = m.addend ? ext.input(*m.addend, "addend") : pa;
				auto pout = ext.output(resultShape, ai->dtype);
				auto i64 = [](PrimExpr e) { return ::tvm::cast(::tvm::DataType::Int(64), e); };
				auto i32 = [](int64_t v) { return ::tvm::IntImm(::tvm::DataType::Int(32), v); };
				return ext.finish(
						{pout, pa, pq, pscales, pzeros, i32(m.quant->zeros ? 1 : 0), i32(static_cast<int>(m.quant->format)),
						 i64(::tvm::IntImm(::tvm::DataType::Int(64), m.quant->group)), pbias, i32(m.bias ? 1 : 0), i64(M),
						 i64(K), i64(N), i64(aM), i64(aK), i64(oM), i64(oN), paddend, i32(m.addend ? 1 : 0),
						 i32(m.activation), ::tvm::FloatImm(::tvm::DataType::Float(32), m.alpha)}
				);
			}
			if (batch.size() > 2)
				return std::nullopt;
			while (batch.size() < 2) { // outer dimensions of size 1 (with the strides of a contiguous outer dimension)
				const Dim next = batch.empty() ? Dim{M, aM, zero, oM} : batch.front();
				batch.insert(
						batch.begin(),
						Dim{::tvm::IntImm(::tvm::DataType::Int(64), 1), next.a * next.size, next.b * next.size, next.out * next.size}
				);
			}

			ExternCall ext{bb, "mdl_matmul_" + *kernelSuffix(ai->dtype)};
			auto pa = ext.input(m.a, "A");
			auto pb = ext.input(m.b, "B");
			PrimExpr pbias = m.bias ? ext.input(*m.bias, "bias") : pa;		 // ignored if has_bias == 0
			PrimExpr paddend = m.addend ? ext.input(*m.addend, "addend") : pa; // ignored if has_addend == 0
			auto pout = ext.output(resultShape, ai->dtype);
			auto i64 = [](PrimExpr e) { return ::tvm::cast(::tvm::DataType::Int(64), e); };
			auto i32 = [](int64_t v) { return ::tvm::IntImm(::tvm::DataType::Int(32), v); };
			return ext.finish(
					{pout, pa, pb, pbias, i32(m.bias ? 1 : 0), i64(batch[0].size), i64(batch[1].size), i64(M), i64(K),
					 i64(N), i64(batch[0].a), i64(batch[1].a), i64(aM), i64(aK), i64(batch[0].b), i64(batch[1].b), i64(bK),
					 i64(bN), i64(batch[0].out), i64(batch[1].out), i64(oM), i64(oN), paddend, i32(m.addend ? 1 : 0),
					 i32(m.activation), ::tvm::FloatImm(::tvm::DataType::Float(32), m.alpha)}
			);
		}

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
			auto axes = [](const Perm& p) {
				Array<Integer> result;
				for (int64_t x : p)
					result.push_back(Integer(static_cast<int>(x)));
				return result;
			};
			if (!m.permA.empty() && !isIdentity(m.permA))
				ta = topi::transpose(ta, axes(m.permA));
			if (!m.permB.empty() && !isIdentity(m.permB))
				tb = topi::transpose(tb, axes(m.permB));
			std::optional<te::Tensor> tbias;
			if (m.bias)
				tbias = te.tensor(*m.bias, "bias");
			te::Tensor out = matmulCompute(ta, tb, tbias, false, m.alpha, false);
			if (m.activation == 1)
				out = te::compute(
						out->shape,
						[&](const Array<tirx::Var>& i) {
							PrimExpr v = ::tvm::cast(::tvm::DataType::Float(32), out(Array<PrimExpr>{i.begin(), i.end()}));
							auto c = [](double d) { return ::tvm::tirx::make_const(::tvm::DataType::Float(32), d); };
							return ::tvm::cast(out->dtype, v * c(0.5) * (c(1.0) + ::tvm::erf(v * c(M_SQRT1_2))));
						},
						"T_gelu", topi::kElementWise
				);
			if (!m.permOut.empty() && !isIdentity(m.permOut))
				out = topi::transpose(out, axes(m.permOut));
			if (m.addend)
				out = topi::add(out, te.tensor(*m.addend, "addend"));
			return te.finish(out);
		}

		// ── mdl.matmul operator ───────────────────────────────────────────────────

		::tvm::relax::StructInfo inferMatmul(const Call& call, const BlockBuilder& ctx) {
			MatmulArgs args{call};
			const auto* ai = tensorInfo(args.a);
			return TensorStructInfo(ShapeExpr(permuted(args.productShape(), args.permOut)), ai->dtype, ai->vdevice);
		}

		Expr makeCall(const MatmulArgs& m) {
			static const auto& op = ::tvm::Op::Get("mdl.matmul");
			Expr bias = m.bias ? *m.bias : Expr(::tvm::relax::PrimValue::Int64(0));
			Expr addend = m.addend ? *m.addend : Expr(::tvm::relax::PrimValue::Int64(0));
			const Expr none = ::tvm::relax::PrimValue::Int64(0);
			Expr scales = m.quant ? m.quant->scales : none;
			Expr zeros = m.quant && m.quant->zeros ? *m.quant->zeros : none;
			Expr quant = permExpr(
					m.quant ? Perm{static_cast<int64_t>(m.quant->format), m.quant->group} : Perm{}
			);
			return Call(
					op, {m.a, m.b, bias, permExpr(m.permA), permExpr(m.permB), permExpr(m.permOut),
						 ::tvm::relax::PrimValue(::tvm::FloatImm(::tvm::DataType::Float(64), m.alpha)),
						 ::tvm::relax::PrimValue::Int64(m.activation), addend, scales, zeros, quant}
			);
		}

		/**
		 * @brief Folds permutations and epilogues into mdl.matmul (see MatmulArgs): `permute(x) @ y` reads x through
		 * strides; if the product has no other use, `permute(a @ b)` writes the result in the permuted layout,
		 * `gelu(a @ b)` and `a @ b + r` are computed by the matrix product's epilogue.
		 */
		class FoldPermutes : public ::tvm::relax::ExprMutator {
		public:
			explicit FoldPermutes(const ::tvm::relax::Function& f) {
				struct Counter : ::tvm::relax::ExprVisitor {
					std::unordered_map<const ::tvm::relax::VarNode*, int>* uses = nullptr;
					void VisitExpr_(const ::tvm::relax::VarNode* op) final { ++(*uses)[op]; }
					void VisitExpr_(const ::tvm::relax::DataflowVarNode* op) final { ++(*uses)[op]; }
				} counter;
				counter.uses = &uses;
				counter.VisitExpr(f);
			}

			using ExprMutator::VisitExpr_;

			Expr VisitExpr_(const ::tvm::relax::CallNode* op) final {
				static const auto& matmul = ::tvm::Op::Get("mdl.matmul");
				static const auto& permute = ::tvm::Op::Get("relax.permute_dims");
				auto call = ::tvm::Downcast<Call>(ExprMutator::VisitExpr_(op));
				if (call->op.same_as(matmul)) {
					MatmulArgs m{call};
					bool changed = false;
					auto fold = [&](Expr& x, Perm& perm) {
						const ::tvm::relax::CallNode* p = boundCall(x, permute);
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
					if (const ::tvm::relax::CallNode* d = m.quant ? nullptr : boundCall(m.b, dequant)) {
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
				static const auto& gelu = ::tvm::Op::Get("relax.nn.gelu");
				static const auto& add = ::tvm::Op::Get("relax.add");
				// The product's single use: an activation (before a permutation or an addend is applied).
				if (call->op.same_as(gelu) && singleUse(op->args[0])) {
					const ::tvm::relax::CallNode* matmulOp = boundCall(call->args[0], matmul);
					if (matmulOp == nullptr)
						return call;
					MatmulArgs m{::tvm::ffi::GetRef<Call>(matmulOp)};
					if (m.activation != 0 || m.addend || (!m.permOut.empty() && !isIdentity(m.permOut)))
						return call;
					m.activation = 1;
					return makeCall(m);
				}
				// The product's single use: an addition of a tensor of the result's shape (e.g. a residual connection).
				if (call->op.same_as(add)) {
					for (size_t i = 0; i < 2; ++i) {
						if (!singleUse(op->args[i]))
							continue;
						const ::tvm::relax::CallNode* matmulOp = boundCall(call->args[i], matmul);
						const Expr& other = call->args[1 - i];
						const auto* product = matmulOp ? ::tvm::relax::GetStructInfoAs<TensorStructInfoNode>(call->args[i]) : nullptr;
						const auto* addend = ::tvm::relax::GetStructInfoAs<TensorStructInfoNode>(other);
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
				const auto* arg = op->args.empty() ? nullptr : op->args[0].as<::tvm::relax::VarNode>();
				if (call->op.same_as(permute) && arg != nullptr && uses[arg] == 1) {
					const ::tvm::relax::CallNode* matmulOp = boundCall(call->args[0], matmul);
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
			std::unordered_map<const ::tvm::relax::VarNode*, int> uses;

			bool singleUse(const Expr& e) {
				const auto* var = e.as<::tvm::relax::VarNode>();
				return var != nullptr && uses[var] == 1;
			}

			/** @brief The call of @p op bound to the variable @p e (in the rewritten function), if any. */
			const ::tvm::relax::CallNode* boundCall(const Expr& e, const ::tvm::Op& op) {
				const auto* var = e.as<::tvm::relax::VarNode>();
				if (var == nullptr)
					return nullptr;
				auto value = builder_->LookupBinding(::tvm::ffi::GetRef<::tvm::relax::Var>(var));
				const auto* call = value ? value.value().as<::tvm::relax::CallNode>() : nullptr;
				return call != nullptr && call->op.same_as(op) ? call : nullptr;
			}

			static std::optional<Perm> axesOf(const ::tvm::relax::CallNode* permute) {
				const auto* attrs = permute->attrs.as<::tvm::relax::PermuteDimsAttrs>();
				if (attrs == nullptr || !attrs->axes.defined())
					return std::nullopt;
				Perm axes;
				for (const auto& a : attrs->axes.value())
					axes.push_back(a->value);
				return axes;
			}
		};
	} // namespace

	Expr makeMatmul(Expr a, Expr b, std::optional<Expr> bias, bool transB, double alpha, bool transA) {
		auto rank = [](const Expr& e) { return static_cast<size_t>(::tvm::relax::GetStructInfoAs<TensorStructInfoNode>(e)->ndim); };
		MatmulArgs m{a, b};
		m.bias = std::move(bias);
		if (transA)
			m.permA = swapLastTwo(rank(a));
		if (transB)
			m.permB = swapLastTwo(rank(b));
		m.alpha = alpha;
		return makeCall(m);
	}

	Expr makeDequant(Expr q, Expr scales, Expr zeros, int64_t format, int64_t group) {
		static const auto& op = ::tvm::Op::Get("mdl.dequant");
		return Call(op, {std::move(q), std::move(scales), std::move(zeros), ::tvm::relax::PrimValue::Int64(format),
						 ::tvm::relax::PrimValue::Int64(group)});
	}

	TVM_REGISTER_OP("mdl.dequant")
			.set_num_inputs(5)
			.add_argument("q", "Tensor", "The quantized values [N, K] int8 (int4: [N, K/2], two per byte).")
			.add_argument("scales", "Tensor", "The scales [K/group, N] (float).")
			.add_argument("zeros", "Tensor | PrimValue", "The zero points [K/group, N] int8, or PrimValue(0).")
			.add_argument("format", "PrimValue", "The format (builtins::QuantFormat).")
			.add_argument("group", "PrimValue", "The number of values of K per scale.")
			.set_attr<::tvm::relax::FInferStructInfo>(
					"FInferStructInfo",
					[](const Call& call, const BlockBuilder&) -> ::tvm::relax::StructInfo {
						auto q = shapeOf(call->args[0]);
						const int64_t format = intValue(call->args[3]);
						const bool packed = format != static_cast<int64_t>(mdl::builtins::QuantFormat::int8);
						Array<PrimExpr> shape{q[0], packed ? q[1] * 2 : q[1]};
						return TensorStructInfo(ShapeExpr(shape), tensorInfo(call->args[1])->dtype);
					}
			)
			.set_attr<::tvm::Bool>("FPurity", ::tvm::Bool(true));

	Expr makeGeluBackward(Expr x, Expr dy) {
		static const auto& op = ::tvm::Op::Get("mdl.gelu_backward");
		return Call(op, {std::move(x), std::move(dy)});
	}

	TVM_REGISTER_OP("mdl.gelu_backward")
			.set_num_inputs(2)
			.add_argument("x", "Tensor", "The input of GELU.")
			.add_argument("dy", "Tensor", "The cotangent of GELU's result (same shape).")
			.set_attr<::tvm::relax::FInferStructInfo>(
					"FInferStructInfo",
					[](const Call& call, const BlockBuilder&) -> ::tvm::relax::StructInfo {
						return ::tvm::relax::GetStructInfo(call->args[1]);
					}
			)
			.set_attr<::tvm::Bool>("FPurity", ::tvm::Bool(true));

	::tvm::transform::Pass foldPermutesIntoMatmul() {
		return ::tvm::relax::transform::CreateFunctionPass(
				[](::tvm::relax::Function f, ::tvm::IRModule, ::tvm::transform::PassContext) {
					return ::tvm::Downcast<::tvm::relax::Function>(FoldPermutes(f).VisitExpr(f));
				},
				0, "mdl.FoldPermutesIntoMatmul", {}
		);
	}

	TVM_REGISTER_OP("mdl.matmul")
			.set_num_inputs(12)
			.add_argument("a", "Tensor", "The left operand A = permute(a, perm_a): [..., M, K].")
			.add_argument("b", "Tensor", "The right operand B = permute(b, perm_b): [K, N], possibly batched.")
			.add_argument("bias", "Tensor | PrimValue", "The bias [N] or PrimValue(0) for no bias.")
			.add_argument("perm_a", "ShapeExpr", "The permutation of a (empty: none).")
			.add_argument("perm_b", "ShapeExpr", "The permutation of b (empty: none).")
			.add_argument("perm_out", "ShapeExpr", "The permutation of the result (empty: none).")
			.add_argument("alpha", "PrimValue", "Scale applied to the product.")
			.add_argument("activation", "PrimValue", "The activation of the epilogue: 0 none, 1 GELU (erf).")
			.add_argument("addend", "Tensor | PrimValue", "A tensor of the result's shape added last, or PrimValue(0).")
			.add_argument("scales", "Tensor | PrimValue", "If b is quantized: the scales of mdl.dequant, else PrimValue(0).")
			.add_argument("zeros", "Tensor | PrimValue", "If b is quantized: the zero points or PrimValue(0).")
			.add_argument("quant", "ShapeExpr", "If b is quantized: [format, group] of mdl.dequant, else empty.")
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

		map.Set("relax.concat", wrap([](const BlockBuilder& bb, const Call& call) {
					const auto* attrs = call->attrs.as<::tvm::relax::ConcatAttrs>();
					TECall te{bb, "concat"};
					Array<te::Tensor> parts;
					for (const auto& field : ::tvm::Downcast<::tvm::relax::Tuple>(call->args[0])->fields)
						parts.push_back(te.tensor(field));
					const int axis = attrs->axis ? static_cast<int>(attrs->axis.value()) : 0;
					return te.finish(topi::concatenate(parts, axis));
				}));
		map.Set("relax.collapse_sum_to", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					// Sums over leading dimensions (e.g. the gradient of a bias): a column sum of [M, N] by the kernel.
					const auto& x = call->args[0];
					const auto dtype = tensorInfo(x)->dtype;
					auto shape = shapeOf(x);
					auto target = ::tvm::Downcast<ShapeExpr>(call->args[1])->values;
					::tvm::arith::Analyzer analyzer;
					size_t lead = 0; // leading 1s of the target
					while (lead < target.size() && analyzer.CanProveEqual(target[lead], 1))
						++lead;
					const size_t t = target.size() - lead;
					// (A sum of a product, e.g. the gradient of LayerNorm's scale, fuses with the product instead.)
					bool ofProduct = false;
					if (const auto* var = x.as<::tvm::relax::VarNode>())
						if (auto value = bb->LookupBinding(::tvm::ffi::GetRef<::tvm::relax::Var>(var)))
							if (const auto* producer = value.value().as<::tvm::relax::CallNode>()) {
								static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
								const auto* kernel = producer->op.same_as(callTir) && !producer->args.empty()
															 ? producer->args[0].as<::tvm::GlobalVarNode>()
															 : nullptr;
								ofProduct = producer->op.same_as(::tvm::Op::Get("relax.multiply")) ||
											(kernel != nullptr && std::string(kernel->name_hint).starts_with("multiply"));
							}
					bool suffix = vendor && !ofProduct && kernelSuffix(dtype) && t > 0 && t < shape.size();
					for (size_t i = 0; suffix && i < t; ++i)
						suffix = analyzer.CanProveEqual(target[lead + i], shape[shape.size() - t + i]);
					if (suffix) {
						ExternCall ext{bb, "mdl_sum_rows_" + *kernelSuffix(dtype)};
						auto px = ext.input(x, "x");
						auto pout = ext.output(target, dtype);
						return ext.finish(
								{pout, px, product(shape, 0, shape.size() - t), product(shape, shape.size() - t, shape.size())}
						);
					}
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

		map.Set("relax.matmul", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const Expr &a = call->args[0], &b = call->args[1];
					if (vendor)
						if (auto e = matmulKernel(bb, MatmulArgs{a, b}))
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
					return matmulTE(bb, MatmulArgs{a, b});
				}));
		map.Set("mdl.matmul", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					MatmulArgs args{call};
					if (vendor) {
						if (auto e = matmulKernel(bb, args))
							return *e;
						if (args.quant) // a layout the quantized kernel does not cover: dequantize first
							if (auto e = matmulKernel(bb, args = dequantized(bb, args)))
								return *e;
					}
					return matmulTE(bb, args);
				}));

		map.Set("relax.nn.softmax", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto* attrs = call->attrs.as<::tvm::relax::SoftmaxAttrs>();
					const auto& x = call->args[0];
					auto shape = shapeOf(x);
					const auto dtype = tensorInfo(x)->dtype;
					if (vendor && isLastAxis(attrs->axis, shape.size()) && kernelSuffix(dtype)) {
						ExternCall ext{bb, "mdl_softmax_" + *kernelSuffix(dtype)};
						auto px = ext.input(x, "x");
						auto pout = ext.output(shape, dtype);
						return ext.finish({pout, px, product(shape, 0, shape.size() - 1), product(shape, shape.size() - 1, shape.size())});
					}
					TECall te{bb, "softmax"};
					return te.finish(topi::nn::softmax(te.tensor(x), attrs->axis));
				}));
		map.Set("relax.nn.gelu", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto& x = call->args[0];
					auto shape = shapeOf(x);
					const auto dtype = tensorInfo(x)->dtype;
					if (vendor && kernelSuffix(dtype)) {
						ExternCall ext{bb, "mdl_gelu_" + *kernelSuffix(dtype)};
						auto px = ext.input(x, "x");
						auto pout = ext.output(shape, dtype);
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
		map.Set("mdl.gelu_backward", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto &x = call->args[0], &dy = call->args[1];
					auto shape = shapeOf(dy);
					const auto dtype = tensorInfo(dy)->dtype;
					if (vendor && kernelSuffix(dtype) && tensorInfo(x)->dtype == dtype &&
						::tvm::ffi::StructuralEqual()(shapeOf(x), shape)) {
						ExternCall ext{bb, "mdl_gelu_backward_" + *kernelSuffix(dtype)};
						auto px = ext.input(x, "x");
						auto pdy = ext.input(dy, "dy");
						auto pout = ext.output(shape, dtype);
						return ext.finish({pout, px, pdy, product(shape, 0, shape.size())});
					}
					// dy * GELU'(x) = dy * (Φ(x) + x φ(x)), in float32.
					TECall te{bb, "gelu_backward"};
					auto tx = te.tensor(x, "x");
					auto tdy = te.tensor(dy, "dy");
					return te.finish(te::compute(
							tdy->shape,
							[&](const Array<tirx::Var>& i) {
								const auto f32 = ::tvm::DataType::Float(32);
								Array<PrimExpr> index{i.begin(), i.end()};
								PrimExpr v = ::tvm::cast(f32, tx(index));
								auto c = [&](double d) { return ::tvm::tirx::make_const(f32, d); };
								PrimExpr grad = c(0.5) * (c(1.0) + ::tvm::erf(v * c(M_SQRT1_2))) +
												v * ::tvm::exp(c(-0.5) * v * v) * c(0.3989422804014327);
								return ::tvm::cast(tdy->dtype, ::tvm::cast(f32, tdy(index)) * grad);
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
					const auto* on = call->args[1].as<::tvm::relax::PrimValueNode>();
					const auto* off = call->args[2].as<::tvm::relax::PrimValueNode>();
					TECall te{bb, "one_hot"};
					auto labels = te.tensor(call->args[0], "labels");
					// The output shape as Relax inferred it (int64 dims; TOPI would use an int32 depth).
					return te.finish(topi::one_hot(
							labels, on->value, off->value, attrs->depth, attrs->axis, on->value.dtype(), shapeOf(call)
					));
				}));

		map.Set("relax.nn.layer_norm", wrap([vendor](const BlockBuilder& bb, const Call& call) -> Expr {
					const auto* attrs = call->attrs.as<::tvm::relax::LayerNormAttrs>();
					const auto &x = call->args[0], &gamma = call->args[1], &beta = call->args[2];
					auto shape = shapeOf(x);
					const bool lastAxis = attrs->axes.size() == 1 && isLastAxis(attrs->axes[0]->value, shape.size());
					const auto dtype = tensorInfo(x)->dtype;
					if (vendor && lastAxis && attrs->center && attrs->scale && kernelSuffix(dtype) &&
						tensorInfo(gamma)->dtype == dtype && tensorInfo(beta)->dtype == dtype) {
						ExternCall ext{bb, "mdl_layer_norm_" + *kernelSuffix(dtype)};
						auto px = ext.input(x, "x");
						auto pg = ext.input(gamma, "gamma");
						auto pb = ext.input(beta, "beta");
						auto pout = ext.output(shape, dtype);
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

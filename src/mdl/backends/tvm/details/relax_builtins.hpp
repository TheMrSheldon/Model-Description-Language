#pragma once

#include <mdl/computegraphv2/builtins.hpp>

#include "./attention.hpp"
#include "./legalize.hpp"
#include "./tvm_compat.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/tensor.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	using RelaxTranslateFunc =
			std::function<::tvm::relax::Expr(::tvm::relax::BlockBuilder&, std::vector<::tvm::relax::Expr>)>;

	// ── helpers ─────────────────────────────────────────────────────────────────

	/** @brief The scale of attention (a compile-time constant). */
	static inline double attentionScale(const ::tvm::relax::Expr& e) {
		auto value = scalarConstant(e);
		if (!value)
			throw std::runtime_error("attention: the scale is not a compile-time constant");
		return *value;
	}

	/** Call a registered Relax op via the global function registry. */
	template <typename... Args>
	static inline ::tvm::relax::Expr callRelaxOp(const char* name, Args&&... args) {
		auto fn = *::tvm::ffi::Function::GetGlobal(name);
		return fn(std::forward<Args>(args)...).template cast<::tvm::relax::Expr>();
	}

	/** The PrimExpr of a scalar operand (e.g. a shape dimension). */
	static inline ::tvm::PrimExpr extractPrim(const ::tvm::relax::Expr& e) {
		auto value = scalarValue(e);
		assert(value && "expected a scalar for a shape dimension");
		return *value;
	}

	/**
 * Get input tensor's shape as a vector of PrimExpr.
 * The tensor must have TensorType with a known ShapeExpr.
 */
	static inline std::vector<::tvm::PrimExpr> tensorShape(const ::tvm::relax::Expr& e) {
		auto shape = knownShape(e);
		return shape ? std::vector<::tvm::PrimExpr>(shape->begin(), shape->end()) : std::vector<::tvm::PrimExpr>{};
	}

	/** Get tensor ndim (or -1 if unknown). */
	static inline int tensorNdim(const ::tvm::relax::Expr& e) {
		const auto* sinfo = e->ty.as<::tvm::relax::TensorTypeNode>();
		if (!sinfo || sinfo->IsUnknownNdim())
			return -1;
		return sinfo->ndim;
	}

	/** Reduce over last axis. keepdims controls whether axis is retained. */
	static inline ::tvm::relax::Expr reduceLastAxis(const char* opName, const ::tvm::relax::Expr& x, bool keepdims) {
		::tvm::ffi::Optional<::tvm::ffi::Array<int64_t>> axis = ::tvm::ffi::Array<int64_t>{-1};
		return callRelaxOp(opName, x, axis, keepdims);
	}

	/** The element type of a tensor expression (float32 if unknown). */
	static inline ::tvm::PrimType dtypeOf(const ::tvm::relax::Expr& e) {
		const auto* sinfo = e->ty.as<::tvm::relax::TensorTypeNode>();
		return sinfo != nullptr && !sinfo->IsUnknownDtype() ? sinfo->dtype.value() : ::tvm::PrimType::Float(32);
	}

	/** Create a 0-D float Constant of @p dtype (float32, float16, or bfloat16). */
	static inline ::tvm::relax::Expr makeFloatConst(float val, ::tvm::PrimType dtype = ::tvm::PrimType::Float(32)) {
		DLDevice dev{kDLCPU, 0};
		auto data = ::tvm::runtime::Tensor::Empty(::tvm::ffi::Shape{}, dtype->dtype, dev);
		if (dtype == ::tvm::PrimType::Float(16)) {
			const _Float16 half = static_cast<_Float16>(val);
			data.CopyFromBytes(&half, sizeof(half));
		} else if (dtype == ::tvm::PrimType::BFloat(16)) {
			uint32_t bits;
			std::memcpy(&bits, &val, sizeof(bits));
			const uint16_t bf16 =
					static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16); // round to nearest even
			data.CopyFromBytes(&bf16, sizeof(bf16));
		} else {
			data.CopyFromBytes(&val, sizeof(float));
		}
		return ::tvm::relax::MakeTensorConst(data);
	}

	/** The size of the last dimension as a float divisor (a constant if static, else the symbolic size), in x's type. */
	static inline ::tvm::relax::Expr lastDimDivisor(const ::tvm::relax::Expr& x) {
		auto shape = tensorShape(x);
		assert(!shape.empty() && "reduction over a tensor without dimensions");
		if (const auto* imm = shape.back().template as<::tvm::IntImmNode>())
			return makeFloatConst(static_cast<float>(static_cast<int64_t>(imm->value)), dtypeOf(x));
		return ::tvm::prim::cast(dtypeOf(x), shape.back());
	}

	/** @brief @p body in a root block, like the PrimFuncs that TVM generates (Relax analyses expect one). */
	static inline ::tvm::tirx::Stmt rootBlock(::tvm::tirx::Stmt body) {
		return ::tvm::s_tir::SBlockRealize(
				{}, ::tvm::IntImm::Bool(true), ::tvm::s_tir::SBlock("root", std::move(body))
		);
	}

	static inline ::tvm::PrimExpr shapeProduct(const std::vector<::tvm::PrimExpr>& shape, size_t from = 0) {
		::tvm::PrimExpr p = ::tvm::IntImm(::tvm::PrimType::Int(64), 1);
		for (size_t i = from; i < shape.size(); ++i)
			p = p * ::tvm::prim::cast(::tvm::PrimType::Int(64), shape[i]);
		return ::tvm::sym::Analyzer()->Simplify(p);
	}

	/**
 * `out = zeros like `like` ([V, ...])`, then `out[ids[n]] += g[n]` for every index n (the adjoint of `take`, e.g. the
 * gradient of an embedding table): the backend operator mdl.scatter_add_rows over `g` and `ids` flattened to [N, R]
 * and [N], in float32 (also for 16-bit gradients; the translation converts the result).
 */
	static inline ::tvm::relax::Expr scatterAddRows(
			::tvm::relax::BlockBuilder& bb, const ::tvm::relax::Expr& g, const ::tvm::relax::Expr& ids,
			const ::tvm::relax::Expr& like
	) {
		const auto likeShape = tensorShape(like);
		const ::tvm::PrimExpr V = likeShape.at(0), R = shapeProduct(likeShape, 1), N = shapeProduct(tensorShape(ids));
		auto flat = [&](const ::tvm::relax::Expr& x, ::tvm::ffi::Array<::tvm::PrimExpr> shape, const char* hint) {
			return bb->Emit(callRelaxOp("relax.op.reshape", x, ::tvm::relax::ShapeExpr(shape)), hint);
		};
		auto g32 = dtypeOf(g) == ::tvm::PrimType::Float(32)
						   ? g
						   : bb->Emit(callRelaxOp("relax.op.astype", g, ::tvm::PrimType::Float(32)->dtype), "g32");
		auto out = bb->Emit(makeScatterAddRows(flat(g32, {N, R}, "g_rows"), flat(ids, {N}, "ids_flat"), V), "scatter");
		return callRelaxOp(
				"relax.op.reshape", out,
				::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>(likeShape.begin(), likeShape.end()))
		);
	}

	/**
 * `out = zeros like `like``, then `out[..., index, ...] = g` along `axis` (the adjoint of selecting an index): a
 * hand-written kernel over `like` flattened to [P, A, Q] (P: the dimensions before the axis, A: the axis, Q: after it).
 * A negative index counts from the end of the axis.
 */
	static inline ::tvm::relax::Expr unselectLike(
			::tvm::relax::BlockBuilder& bb, const ::tvm::relax::Expr& g, int64_t axis, int64_t index,
			const ::tvm::relax::Expr& like
	) {
		namespace tirx = ::tvm::tirx;
		using ::tvm::IntImm;
		const auto i64 = ::tvm::PrimType::Int(64);
		const auto likeShape = tensorShape(like);
		const int64_t rank = static_cast<int64_t>(likeShape.size());
		const int64_t a = axis < 0 ? axis + rank : axis;
		const ::tvm::PrimExpr P = shapeProduct(std::vector<::tvm::PrimExpr>(likeShape.begin(), likeShape.begin() + a));
		const ::tvm::PrimExpr A = likeShape.at(a);
		const ::tvm::PrimExpr Q = shapeProduct(likeShape, a + 1);
		// A float32 kernel (the translation converts the result to the gradient's type).
		auto g32 = dtypeOf(g) == ::tvm::PrimType::Float(32)
						   ? g
						   : bb->Emit(callRelaxOp("relax.op.astype", g, ::tvm::PrimType::Float(32)->dtype), "g32");
		auto g2 = bb->Emit(
				callRelaxOp("relax.op.reshape", g32, ::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>{P, Q})),
				"g_flat"
		);

		::tvm::PrimVar p("p", i64), q("q", i64), n("n", i64);
		auto G = tirx::decl_buffer({p, q}, ::tvm::PrimType::Float(32), "g");
		auto O = tirx::decl_buffer({p, n, q}, ::tvm::PrimType::Float(32), "out");
		::tvm::PrimVar i("i", i64), j("j", i64), k("k", i64);
		const ::tvm::PrimExpr at = index < 0 ? n + IntImm(i64, index) : ::tvm::PrimExpr(IntImm(i64, index));
		auto value = ::tvm::prim::Select(
				j == at, tirx::BufferLoad(G, {i, k}), ::tvm::FloatImm(::tvm::PrimType::Float(32), 0.0)
		);
		auto loops = tirx::For(
				i, IntImm(i64, 0), p, tirx::ForKind::kParallel,
				tirx::For(
						j, IntImm(i64, 0), n, tirx::ForKind::kSerial,
						tirx::For(k, IntImm(i64, 0), q, tirx::ForKind::kSerial, tirx::BufferStore(O, value, {i, j, k}))
				)
		);
		auto func = ::tvm::WithAttr(tirx::PrimFunc({G, O}, rootBlock(loops), ::tvm::VoidType()), "op_pattern", 8);
		auto gv = bb->AddFunction(func, "unselect_like");
		static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
		auto out = bb->Emit(
				::tvm::Call(
						::tvm::Type::Missing(), callTir, {gv, ::tvm::relax::Tuple({g2})}, ::tvm::Attrs(),
						{::tvm::relax::TensorType(::tvm::relax::ShapeExpr({P, A, Q}), ::tvm::PrimType::Float(32))}
				),
				"unselected"
		);
		return callRelaxOp(
				"relax.op.reshape", out,
				::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>(likeShape.begin(), likeShape.end()))
		);
	}

	/** Emit a value into the BlockBuilder and return the bound Var. */
	static inline ::tvm::Var emitAs(::tvm::relax::BlockBuilder& bb, ::tvm::relax::Expr e, const char* hint = "v") {
		return bb->Emit(std::move(e), hint);
	}

	/**
	 * @brief The result @p i of attention for the operands (q, k, v, scale) @p ts: out and lse of mdl.attention, or the
	 * probabilities softmax(scale * q @ k^T) as the unfused operations (the same as mdl.attention's expansion).
	 */
	static inline ::tvm::relax::Expr
	attentionResult(::tvm::relax::BlockBuilder& bb, const std::vector<::tvm::relax::Expr>& ts, int i) {
		const double scale = attentionScale(ts[3]);
		const auto bias = ts.size() > 4 ? std::optional<::tvm::relax::Expr>{ts[4]} : std::nullopt;
		if (i == 2) {
			const size_t rank = static_cast<size_t>(tensorNdim(ts[0]));
			auto scores = emitAs(bb, makeMatmul(ts[0], ts[1], {}, swapLastTwo(rank), {}, scale), "scores");
			if (bias)
				scores = emitAs(bb, callRelaxOp("relax.op.add", scores, *bias), "biased");
			return callRelaxOp("relax.op.nn.softmax", scores, -1);
		}
		const auto lengths = ts.size() > 5 ? std::optional<::tvm::relax::Expr>{ts[5]} : std::nullopt;
		auto call = emitAs(bb, makeAttention(ts[0], ts[1], ts[2], scale, bias, lengths), "attention");
		return ::tvm::relax::TupleGetItem(call, i);
	}

	/**
 * The axes of a PERMUTE permutation (see mdl::builtins::FunctionIDs::PERMUTE) for a tensor of the given rank, or an
 * empty array if the permutation is invalid.
 */
	static inline ::tvm::ffi::Array<int64_t> permuteAxes(int rank, const ::tvm::relax::ShapeExprNode* se) {
		int permSize = static_cast<int>(se->values.size());
		std::vector<int64_t> perm;
		for (const auto& v : se->values)
			perm.push_back(static_cast<int64_t>(intValue(v.template as<::tvm::IntImmNode>())));

		::tvm::ffi::Array<int64_t> axes;
		const bool hasEllipsis = std::find(perm.begin(), perm.end(), mdl::builtins::kEllipsisPermMarker) != perm.end();
		if (hasEllipsis) {
			// The ellipsis group starts after the dims listed with non-negative indices and covers all dims that
			// are not listed.
			const int64_t start = std::count_if(perm.begin(), perm.end(), [](int64_t p) { return p >= 0; });
			const int64_t groupRank = rank - (permSize - 1);
			for (int64_t p : perm) {
				if (p == mdl::builtins::kEllipsisPermMarker) {
					for (int64_t d = 0; d < groupRank; ++d)
						axes.push_back(static_cast<int64_t>(start + d));
				} else {
					axes.push_back(static_cast<int64_t>(p < 0 ? p + rank : p));
				}
			}
		} else {
			// A shorter permutation applies to the trailing dims.
			int offset = (rank >= 0) ? (rank - permSize) : 0;
			for (int i = 0; i < offset; ++i)
				axes.push_back(static_cast<int64_t>(i));
			for (int64_t p : perm)
				axes.push_back(static_cast<int64_t>(p < 0 && rank > 0 ? p + rank : p));
		}
		// Validate: must be a permutation of all dims.
		if (rank >= 0 && static_cast<int>(axes.size()) != rank)
			return {};
		std::vector<bool> seen(std::max(rank, 0), false);
		for (const auto& ax : axes) {
			int a = static_cast<int>(ax);
			if (a < 0 || a >= rank || seen[a])
				return {};
			seen[a] = true;
		}
		return axes;
	}

	// ── op dispatch table ────────────────────────────────────────────────────────

	// The size is deduced from the initializers so that the static_assert below catches missing translations.
	inline const auto relaxFunctions = std::to_array<RelaxTranslateFunc>({
			/** TENSOR_ADD **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return callRelaxOp("relax.op.add", ts[0], ts[1]); },
			/** TENSOR_SUB **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return callRelaxOp("relax.op.subtract", ts[0], ts[1]); },
			/** TENSOR_DOT **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				// relax.matmul has NumPy semantics (batched, broadcasting, 1-D operands as vectors).
				::tvm::ffi::Optional<::tvm::PrimType> out_dtype;
				return callRelaxOp("relax.op.matmul", ts[0], ts[1], out_dtype);
			},
			/** TENSOR_PROD **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return callRelaxOp("relax.op.multiply", ts[0], ts[1]); },
			/** TENSOR_DIV **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return callRelaxOp("relax.op.divide", ts[0], ts[1]); },

			// ── statistical ─────────────────────────────────────────────────────────
			/** MAX — over the last axis (removed, like MEAN and SUM) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				return reduceLastAxis("relax.op.max", ts[0], /*keepdims=*/false);
			},
			/** MIN **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				return reduceLastAxis("relax.op.min", ts[0], /*keepdims=*/false);
			},
			/** MEAN — implemented as sum/N to avoid TVM's symbolic intermediate reduction buffer
     *  (relax.op.mean generates a B1-sized intermediate that LLVM codegen can't stack-alloc). **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr {
				const auto& x = ts[0];
				auto s = emitAs(bb, reduceLastAxis("relax.op.sum", x, /*keepdims=*/false), "sum");
				auto mean = callRelaxOp("relax.op.divide", s, lastDimDivisor(x));
				if (tensorShape(x).size() == 1) // (a symbolic divisor broadcasts a 0-d sum to [1])
					return callRelaxOp(
							"relax.op.reshape", emitAs(bb, mean, "mean"),
							::tvm::relax::ShapeExpr(::tvm::ffi::Array<::tvm::PrimExpr>{})
					);
				return mean;
			},
			/** SUM — over the last axis (removed) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				return reduceLastAxis("relax.op.sum", ts[0], /*keepdims=*/false);
			},

			// ── unary arithmetic ────────────────────────────────────────────────────
			/** COS **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.cos", ts[0]); },
			/** SIN **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.sin", ts[0]); },
			/** TAN **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.tan", ts[0]); },
			/** EXP **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.exp", ts[0]); },
			/** ERF **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.erf", ts[0]); },
			/** SQRT **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.sqrt", ts[0]); },
			/** RSQRT **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.rsqrt", ts[0]); },
			/** LOG **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.log", ts[0]); },
			/** TANH **/ [](auto& /*bb*/, auto ts) { return callRelaxOp("relax.op.tanh", ts[0]); },

			// ── VAR = sum((x − mean(x))²) / (N − dof) over the last axis (two passes: numerically stable) ──
			/** VAR — uses sum/N instead of relax.op.mean to avoid symbolic intermediate buffer **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr {
				const auto& x = ts[0];
				auto n = lastDimDivisor(x);
				auto sx = emitAs(bb, reduceLastAxis("relax.op.sum", x, /*keepdims=*/true), "sum");
				auto mean = emitAs(bb, callRelaxOp("relax.op.divide", sx, n), "mean");
				auto centered = emitAs(bb, callRelaxOp("relax.op.subtract", x, mean), "centered");
				auto squares = emitAs(bb, callRelaxOp("relax.op.multiply", centered, centered), "sq");
				auto ss = emitAs(bb, reduceLastAxis("relax.op.sum", squares, /*keepdims=*/false), "sum_sq");
				// (The degrees of freedom: a compile-time constant.)
				const auto dof = ts.size() > 1 && ts[1].defined() ? scalarConstant(ts[1]) : std::optional<double>{};
				if (!dof)
					throw std::runtime_error("var: the degrees of freedom must be a compile-time constant");
				auto divisor = emitAs(
						bb, callRelaxOp("relax.op.subtract", n, makeFloatConst(static_cast<float>(*dof), dtypeOf(x))),
						"dof"
				);
				return callRelaxOp("relax.op.divide", ss, divisor);
			},

			// ── einops ──────────────────────────────────────────────────────────────
			/**
     * RESHAPE — ts[0] tensor, ts[1..n] shape dims as PrimValues.
     *
     * Sentinels (from the MDL compiler):
     *   -1  = passthrough: copy the next leading dim of the input tensor.
     *   -2  = infer: let TVM's reshape compute this as (total / others).
     *         TVM uses -1 as its own "infer" sentinel.
     */
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const auto& tensor = ts[0];
				auto inShape = tensorShape(tensor);

				::tvm::ffi::Array<::tvm::PrimExpr> outDims;
				outDims.reserve(ts.size() - 1);

				int passthroughIdx = 0;
				for (size_t i = 1; i < ts.size(); ++i) {
					// Ellipsis group {before, after}: the input dims [before, rank - after).
					if (const auto* group = ts[i].template as<::tvm::relax::ShapeExprNode>()) {
						const int64_t before = intValue(group->values[0].template as<::tvm::IntImmNode>());
						const int64_t after = intValue(group->values[1].template as<::tvm::IntImmNode>());
						for (int64_t d = before; d < static_cast<int64_t>(inShape.size()) - after; ++d)
							outDims.push_back(::tvm::prim::cast(::tvm::PrimType::Int(64), inShape[d]));
						continue;
					}
					::tvm::PrimExpr dim = extractPrim(ts[i]);
					if (const auto* imm = dim.as<::tvm::IntImmNode>()) {
						if (imm->value == -1) {
							// passthrough: use next input dim
							if (passthroughIdx < static_cast<int>(inShape.size()))
								dim = ::tvm::prim::cast(::tvm::PrimType::Int(64), inShape[passthroughIdx]);
							++passthroughIdx;
						} else if (imm->value == -2) {
							// infer: pass TVM's -1 sentinel
							dim = ::tvm::IntImm(::tvm::PrimType::Int(64), -1);
						} else {
							dim = ::tvm::prim::cast(::tvm::PrimType::Int(64), dim);
						}
					} else {
						// symbolic (e.g. hyperparameter): cast to int64 for ShapeExpr
						dim = ::tvm::prim::cast(::tvm::PrimType::Int(64), dim);
					}
					outDims.push_back(dim);
				}
				::tvm::ffi::Array<::tvm::PrimExpr> shape_arr = outDims;
				return callRelaxOp("relax.op.reshape", tensor, shape_arr);
			},

			/**
     * PERMUTE — ts[0] tensor, ts[1] ShapeExpr encoding the permutation (see FunctionIDs::PERMUTE).
     */
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const auto& tensor = ts[0];
				const auto* se = ts[1].template as<::tvm::relax::ShapeExprNode>();
				assert(se && "PERMUTE: expected ShapeExpr for permutation");
				auto axes = permuteAxes(tensorNdim(tensor), se);
				bool identity = true;
				for (size_t i = 0; i < axes.size(); ++i)
					identity &= axes[i] == static_cast<int64_t>(i);
				if (identity)
					return tensor; // also for an invalid permutation (empty axes); identities would hide patterns
				::tvm::ffi::Optional<::tvm::ffi::Array<int64_t>> opt_axes = axes;
				return callRelaxOp("relax.op.permute_dims", tensor, opt_axes);
			},

			/**
     * EXPAND (broadcast_to) — ts[0] tensor, ts[1..n] target dims as PrimValues.
     *
     * Sentinel -1: keep the input dim at that position.
     */
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const auto& tensor = ts[0];
				auto inShape = tensorShape(tensor);

				::tvm::ffi::Array<::tvm::PrimExpr> outDims;
				outDims.reserve(ts.size() - 1);
				for (size_t i = 1; i < ts.size(); ++i) {
					::tvm::PrimExpr dim = extractPrim(ts[i]);
					if (const auto* imm = dim.as<::tvm::IntImmNode>(); imm && imm->value == -1) {
						int idx = static_cast<int>(i - 1);
						if (idx < static_cast<int>(inShape.size()))
							dim = ::tvm::prim::cast(::tvm::PrimType::Int(64), inShape[idx]);
					} else {
						dim = ::tvm::prim::cast(::tvm::PrimType::Int(64), dim);
					}
					outDims.push_back(dim);
				}
				::tvm::relax::Expr shape_expr = ::tvm::relax::ShapeExpr(outDims);
				return callRelaxOp("relax.op.broadcast_to", tensor, shape_expr);
			},

			// ── adjoints (gradient pass) ────────────────────────────────────────────
			/** COLLAPSE_SUM_LIKE — sum ts[0] to the shape of ts[1] **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[1]));
				return callRelaxOp("relax.op.collapse_sum_to", ts[0], ::tvm::relax::ShapeExpr(shape));
			},
			/** BROADCAST_LIKE — broadcast ts[0] to the shape of ts[1] **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[1]));
				return callRelaxOp("relax.op.broadcast_to", ts[0], ::tvm::relax::ShapeExpr(shape));
			},
			/** RESHAPE_LIKE — reshape ts[0] to the shape of ts[1] **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				::tvm::ffi::Array<::tvm::PrimExpr> shape(tensorShape(ts[1]));
				return callRelaxOp("relax.op.reshape", ts[0], shape);
			},
			/** PERMUTE_INVERSE — permute ts[0] by the inverse of the permutation ts[1] **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const auto* se = ts[1].template as<::tvm::relax::ShapeExprNode>();
				assert(se && "PERMUTE_INVERSE: expected ShapeExpr for permutation");
				auto axes = permuteAxes(tensorNdim(ts[0]), se);
				if (axes.empty())
					return ts[0];
				std::vector<int64_t> inverse(axes.size());
				for (size_t i = 0; i < axes.size(); ++i)
					inverse[axes[i]] = static_cast<int64_t>(i);
				::tvm::ffi::Array<int64_t> inv;
				for (int64_t a : inverse)
					inv.push_back(static_cast<int64_t>(a));
				::tvm::ffi::Optional<::tvm::ffi::Array<int64_t>> opt_axes = inv;
				return callRelaxOp("relax.op.permute_dims", ts[0], opt_axes);
			},
			/** EQUAL_MASK — 1 where ts[0] == ts[1], else 0 (float32) **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr {
				auto eq = emitAs(bb, callRelaxOp("relax.op.equal", ts[0], ts[1]), "eq");
				return callRelaxOp("relax.op.astype", eq, ::tvm::PrimType::Float(32)->dtype);
			},
			/** STOP_GRADIENT — identity (only meaningful to the gradient pass) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return ts[0]; },
			/** TAKE — rows of ts[0] at the indices ts[1] (out-of-range indices are clipped) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				return callRelaxOp(
						"relax.op.take", ts[0], ts[1], ::tvm::ffi::Optional<int64_t>(0), ::tvm::ffi::String("clip")
				);
			},
			/** SCATTER_ADD_ROWS — the adjoint of TAKE **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr { return scatterAddRows(bb, ts[0], ts[1], ts[2]); },
			/** ONE_HOT_LIKE — ts[0] integer labels; the number of classes is the last dimension of ts[1] **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const auto like = tensorShape(ts[1]);
				const auto* imm = like.empty() ? nullptr : like.back().template as<::tvm::IntImmNode>();
				if (imm == nullptr)
					throw std::runtime_error("one_hot_like: the number of classes (last dimension) must be static");
				auto value = [](double x) { return ::tvm::FloatImm(::tvm::PrimType::Float(32), x); };
				return callRelaxOp(
						"relax.op.one_hot", ts[0], value(1.0), value(0.0),
						static_cast<int>(static_cast<int64_t>(imm->value)), -1
				);
			},
			/** WHERE — ts[1] where ts[0], else ts[2] **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				return callRelaxOp("relax.op.where", ts[0], ts[1], ts[2]);
			},
			/** MASK_OF, UNMASK — resolved by the masking pass (placed after WHERE) **/
			[](auto& /*bb*/, auto /*ts*/) -> ::tvm::relax::Expr {
				throw std::runtime_error("`.mask` of a tensor that is not masked in the compiled function (see --mask)"
				);
			},
			[](auto& /*bb*/, auto /*ts*/) -> ::tvm::relax::Expr {
				throw std::runtime_error("`.data` of a tensor that is not masked in the compiled function (see --mask)"
				);
			},
			/** SELECT — ts[0] at the index ts[2] along the axis ts[1] (negative indices count from the end) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const int64_t axis = intValue(extractPrim(ts[1]).template as<::tvm::IntImmNode>());
				const int64_t index = intValue(extractPrim(ts[2]).template as<::tvm::IntImmNode>());
				const int64_t rank = tensorNdim(ts[0]);
				int32_t i = static_cast<int32_t>(index);
				DLDevice dev{kDLCPU, 0};
				auto data = ::tvm::runtime::Tensor::Empty(::tvm::ffi::Shape{}, ::tvm::PrimType::Int(32)->dtype, dev);
				data.CopyFromBytes(&i, sizeof(i));
				return callRelaxOp(
						"relax.op.take", ts[0], ::tvm::relax::MakeTensorConst(data),
						::tvm::ffi::Optional<int64_t>(axis < 0 ? axis + rank : axis),
						::tvm::ffi::String(index < 0 ? "wrap" : "clip")
				);
			},
			/** UNSELECT_LIKE — the adjoint of SELECT **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr {
				const int64_t axis = intValue(extractPrim(ts[1]).template as<::tvm::IntImmNode>());
				const int64_t index = intValue(extractPrim(ts[2]).template as<::tvm::IntImmNode>());
				return unselectLike(bb, ts[0], axis, index, ts[3]);
			},
			/** CAST — converted to the node's element type by the translation (see translateRelax) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return ts[0]; },
			/** STACK — each tensor as [1, ...], concatenated along the first dimension (see below) **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr {
				::tvm::ffi::Array<::tvm::relax::Expr> parts;
				for (const auto& t : ts) {
					::tvm::ffi::Array<::tvm::PrimExpr> shape{::tvm::IntImm(::tvm::PrimType::Int(64), 1)};
					for (const auto& d : tensorShape(t))
						shape.push_back(d);
					parts.push_back(
							emitAs(bb, callRelaxOp("relax.op.reshape", t, ::tvm::relax::ShapeExpr(shape)), "slot")
					);
				}
				return callRelaxOp("relax.op.concat", ::tvm::relax::Tuple(parts), ::tvm::ffi::Optional<int64_t>(0));
			},
			/** DEQUANT — the backend operator mdl.dequant (format and group stay integer PrimValues) **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr {
				const int64_t format = intValue(extractPrim(ts[3]).template as<::tvm::IntImmNode>());
				const int64_t group = intValue(extractPrim(ts[4]).template as<::tvm::IntImmNode>());
				return makeDequant(ts[0], ts[1], ts[2], format, group);
			},
			/** ATTENTION, ATTENTION_LSE, ATTENTION_PROBS — the results of mdl.attention (one call: merged by CSE) **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr { return attentionResult(bb, ts, 0); },
			[](auto& bb, auto ts) -> ::tvm::relax::Expr { return attentionResult(bb, ts, 1); },
			[](auto& bb, auto ts) -> ::tvm::relax::Expr { return attentionResult(bb, ts, 2); },
			/** ATTENTION_GRAD — a result of mdl.attention_grad (q, k, v, scale, out, lse, probs, dy, which[, bias]) **/
			[](auto& bb, auto ts) -> ::tvm::relax::Expr {
				const auto which = scalarConstant(ts[8]);
				if (!which || *which < 0 || *which > 2)
					throw std::runtime_error("attention gradient: expected the operand 0, 1, or 2");
				const auto bias = ts.size() > 9 ? std::optional<::tvm::relax::Expr>{ts[9]} : std::nullopt;
				const auto lengths = ts.size() > 10 ? std::optional<::tvm::relax::Expr>{ts[10]} : std::nullopt;
				auto grad = makeAttentionGrad(
						ts[0], ts[1], ts[2], ts[4], ts[5], ts[6], ts[7], attentionScale(ts[3]), bias, lengths
				);
				return ::tvm::relax::TupleGetItem(emitAs(bb, grad, "attention_grad"), static_cast<int>(*which));
			},
			/** SUFFIX_MASK — the backend operator mdl.suffix_mask **/
			[](auto& /*bb*/, auto ts) -> ::tvm::relax::Expr { return makeSuffixMask(ts[0], ts[1]); },
	});

	// Every builtin needs a translation (the table is positional).
	static_assert(std::tuple_size_v<decltype(relaxFunctions)> == std::to_underlying(mdl::builtins::FunctionIDs::COUNT));

} // namespace mdl::backend::tvm

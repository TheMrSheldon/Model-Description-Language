#include <mdl/computegraphv2/builtins.hpp>

#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/einsum.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <any>
#include <array>
#include <cassert>
#include <functional>
#include <vector>

namespace mdl::backend::tvm {

	/**
	 * @brief Translation function for a builtin operation.
	 * @details All arguments are tvm::te::Tensor (translated from compute graph nodes).
	 */
	using TranslateFunc = std::function<::tvm::te::Tensor(std::vector<std::any>)>;

	/** Type-checked cast helper — mismatched type is an internal compiler error. **/
	template <typename T>
	inline const T& tvmArg(const std::any& a) {
		assert(a.type() == typeid(T));
		return *std::any_cast<T>(&a);
	}

	/** Cast tensor to float32 if it is not already a float type.
	 *  Needed for math builtins like sqrt/exp when called with integer hyperparameters. **/
	inline ::tvm::te::Tensor toFloat(const ::tvm::te::Tensor& t) {
		if (t->dtype.is_float())
			return t;
		return ::tvm::topi::cast(t, ::tvm::DataType::Float(32));
	}

	/** Extract the body PrimExpr from a single-element compute tensor.
	 *  Works for ConstNode scalars and HyperparameterNodes. **/
	inline ::tvm::PrimExpr bodyOf(const ::tvm::te::Tensor& t) {
		const auto* cop = t->op.template as<::tvm::te::ComputeOpNode>();
		assert(cop && !cop->body.empty());
		return cop->body[t->value_index];
	}

	/** Evaluate element @p i of a compute tensor by substituting the loop variable. **/
	inline int extractInt(const ::tvm::te::Tensor& t, int i) {
		const auto* cop = t->op.template as<::tvm::te::ComputeOpNode>();
		assert(cop && cop->axis.size() == 1);
		::tvm::tir::Var axisVar = cop->axis[0]->var;
		::tvm::PrimExpr val = ::tvm::IntImm(::tvm::DataType::Int(32), i);
		auto expr = ::tvm::arith::Analyzer().Simplify(
				::tvm::tir::Substitute(cop->body[t->value_index], [&](const ::tvm::tir::Var& v) -> ::tvm::ffi::Optional<::tvm::PrimExpr> {
					if (v.same_as(axisVar)) return val;
					return ::tvm::ffi::Optional<::tvm::PrimExpr>{};
				})
		);
		const auto* imm = expr.template as<::tvm::IntImmNode>();
		assert(imm);
		return static_cast<int>(imm->value);
	}

	// Wrap a TVM Buffer as a DLTensor argument for tvm_call_packed.
	// Replicates what Python's tvm.tir.call_packed(fname, buf) generates:
	// the dtype of the FloatImm constant encodes the element type (its value is unused).
	inline ::tvm::PrimExpr bufAsArray(const ::tvm::tir::Buffer& buf) {
		::tvm::ffi::Array<::tvm::PrimExpr> dims(buf->shape.begin(), buf->shape.end());
		auto shape = ::tvm::tir::Call(
			::tvm::DataType::Handle(), ::tvm::tir::builtin::tvm_stack_make_shape(), dims
		);
		return ::tvm::tir::Call(
			::tvm::DataType::Handle(), ::tvm::tir::builtin::tvm_stack_make_array(),
			{
				buf->data,
				shape,
				::tvm::IntImm(::tvm::DataType::Int(32), 0),
				::tvm::IntImm(::tvm::DataType::Int(32), (int)buf->shape.size()),
				::tvm::FloatImm(buf->dtype, 0.0),
				::tvm::IntImm(::tvm::DataType::Int(32), 0),
			}
		);
	}

	// Build a CBLAS ExternOp for a batch matmul of rank-3 tensors.
	// Computes C[b,i,k] = sum_j A[b,i,j] * B[b,j,k] via tvm.contrib.cblas.batch_matmul.
	inline ::tvm::te::Tensor cblasGemm(
			const ::tvm::te::Tensor& A, const ::tvm::te::Tensor& B,
			const ::tvm::ffi::Array<::tvm::PrimExpr>& outShape,
			const std::string& fname, bool transa, bool transb)
	{
		auto bufA = ::tvm::tir::decl_buffer(A->shape, A->dtype, "cblas_A");
		auto bufB = ::tvm::tir::decl_buffer(B->shape, B->dtype, "cblas_B");
		auto bufC = ::tvm::tir::decl_buffer(outShape, A->dtype, "cblas_C");
		auto body = ::tvm::tir::Evaluate(::tvm::tir::Call(
			::tvm::DataType::Int(32), ::tvm::tir::builtin::tvm_call_packed(),
			{
				::tvm::tir::StringImm(fname),
				bufAsArray(bufA), bufAsArray(bufB), bufAsArray(bufC),
				::tvm::Bool(transa), ::tvm::Bool(transb),
			}
		));
		return ::tvm::te::ExternOp(fname, "", {}, {A, B}, {bufA, bufB}, {bufC}, body).output(0);
	}

	std::array<TranslateFunc, std::to_underlying(mdl::builtins::FunctionIDs::COUNT)> functions{
			/** Tensor Operators **/
			/*[TENSOR_ADD] =*/
			[](auto ts) {
				return ::tvm::topi::add(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_SUB] =*/
			[](auto ts) {
				return ::tvm::topi::subtract(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_DOT] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				const auto& A = tvmArg<::tvm::te::Tensor>(ts[0]);
				const auto& B = tvmArg<::tvm::te::Tensor>(ts[1]);
				int rA = static_cast<int>(A->shape.size());
				int rB = static_cast<int>(B->shape.size());
				if (rA == 2 && rB == 2)
					return cblasGemm(A, B, {A->shape[0], B->shape[1]},
						"tvm.contrib.cblas.matmul", false, false);
				if (rA == 3 && rB == 3)
					return cblasGemm(A, B, {A->shape[0], A->shape[1], B->shape[2]},
						"tvm.contrib.cblas.batch_matmul", false, false);
				// General: leading dims are shared batch dims; last two of A contract with last two of B.
				// Build einsum string: batch letters + (i,j) for A's matrix dims, (j,k) for B's.
				int nBatch = std::max(rA, rB) - 2;
				std::string lhsSub, rhsSub, outSub;
				for (int b = 0; b < nBatch; ++b) {
					char c = static_cast<char>('a' + b);
					lhsSub += c; rhsSub += c; outSub += c;
				}
				lhsSub += "ij"; rhsSub += "jk"; outSub += "ik";
				return ::tvm::topi::einsum(lhsSub + "," + rhsSub + "->" + outSub, {A, B});
			},
			/*[TENSOR_PROD]=*/
			[](auto ts) {
				return ::tvm::topi::multiply(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/*[TENSOR_DIV] =*/
			[](auto ts) {
				return ::tvm::topi::divide(tvmArg<::tvm::te::Tensor>(ts[0]), tvmArg<::tvm::te::Tensor>(ts[1]));
			},
			/** Statistic Functions **/
			/*[MAX] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				const auto& t = tvmArg<::tvm::te::Tensor>(ts[0]);
				int ndim = static_cast<int>(t->shape.size());
				if (ndim == 0) return t;
				// keepdims=true: last dim becomes 1, enabling direct broadcast in
				// stable softmax (x - max(x)) without an extra reshape.
				::tvm::ffi::Array<::tvm::Integer> axes{::tvm::Integer(ndim - 1)};
				return ::tvm::topi::max(t, axes, /*keepdims=*/true);
			},
			/*[MIN] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				const auto& t = tvmArg<::tvm::te::Tensor>(ts[0]);
				int ndim = static_cast<int>(t->shape.size());
				if (ndim == 0) return t;
				::tvm::ffi::Array<::tvm::Integer> axes{::tvm::Integer(ndim - 1)};
				return ::tvm::topi::min(t, axes, /*keepdims=*/true);
			},
			/*[MEAN] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// Reduce over last axis: float[...] = mean(float[..., N])
				// If the input is 1-D the output would be 0-D; use [1] instead to
				// stay compatible with TVM's lowering pipeline (0-D allocations crash).
				const auto& t = tvmArg<::tvm::te::Tensor>(ts[0]);
				int ndim = static_cast<int>(t->shape.size());
				int batchDims = ndim - 1;
				::tvm::ffi::Array<::tvm::PrimExpr> outShape;
				for (int i = 0; i < batchDims; ++i)
					outShape.push_back(t->shape[i]);
				if (outShape.empty())
					outShape.push_back(::tvm::PrimExpr(1)); // keep output at least 1-D
				auto lastDim = t->shape[ndim - 1];
				auto k = ::tvm::te::reduce_axis(::tvm::Range(0, lastDim), "k");
				auto dtype = t->dtype;
				return ::tvm::te::compute(
						outShape,
						[t, k, lastDim, dtype, batchDims](::tvm::ffi::Array<::tvm::tir::Var> idx) -> ::tvm::PrimExpr {
							// idx.size() == max(batchDims, 1); real batch indices are the last batchDims entries.
							::tvm::ffi::Array<::tvm::PrimExpr> fullIdx;
							int start = static_cast<int>(idx.size()) - batchDims;
							for (int i = start; i < static_cast<int>(idx.size()); ++i)
								fullIdx.push_back(::tvm::PrimExpr(idx[i]));
							fullIdx.push_back(k->var);
							return ::tvm::sum(t(fullIdx) / ::tvm::cast(dtype, lastDim), {k});
						},
						"mean"
				);
			},
			/*[SUM] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				const auto& t = tvmArg<::tvm::te::Tensor>(ts[0]);
				int ndim = static_cast<int>(t->shape.size());
				if (ndim == 0) return t;
				// keepdims=true: last dim becomes 1, enabling direct broadcast in
				// stable softmax (power / sum(power)) without an extra reshape.
				::tvm::ffi::Array<::tvm::Integer> axes{::tvm::Integer(ndim - 1)};
				return ::tvm::topi::sum(t, axes, /*keepdims=*/true);
			},
			/** Arithmetic Functions **/
			/*[COS] =*/[](auto ts) { return ::tvm::topi::cos(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[SIN] =*/[](auto ts) { return ::tvm::topi::sin(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[TAN] =*/[](auto ts) { return ::tvm::topi::tan(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[EXP] =*/[](auto ts) { return ::tvm::topi::exp(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[ERF] =*/[](auto ts) { return ::tvm::topi::erf(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[SQRT] =*/[](auto ts) { return ::tvm::topi::sqrt(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[RSQRT] =*/[](auto ts) { return ::tvm::topi::rsqrt(toFloat(tvmArg<::tvm::te::Tensor>(ts[0]))); },
			/*[VAR] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// var(x) ≈ E[x²] - E[x]²  (population variance, dof correction ignored)
				// Reduces over last axis to match MEAN semantics.
				// If the input is 1-D the output would be 0-D; use [1] instead to
				// stay compatible with TVM's lowering pipeline (0-D allocations crash).
				const auto& t = tvmArg<::tvm::te::Tensor>(ts[0]);
				int ndim = static_cast<int>(t->shape.size());
				int batchDims = ndim - 1;
				::tvm::ffi::Array<::tvm::PrimExpr> outShape;
				for (int i = 0; i < batchDims; ++i)
					outShape.push_back(t->shape[i]);
				if (outShape.empty())
					outShape.push_back(::tvm::PrimExpr(1)); // keep output at least 1-D
				auto lastDim = t->shape[ndim - 1];
				auto dtype = t->dtype;
				auto k1 = ::tvm::te::reduce_axis(::tvm::Range(0, lastDim), "k1");
				auto meanT = ::tvm::te::compute(outShape,
						[t, k1, lastDim, dtype, batchDims](::tvm::ffi::Array<::tvm::tir::Var> idx) {
							::tvm::ffi::Array<::tvm::PrimExpr> fi;
							int start = static_cast<int>(idx.size()) - batchDims;
							for (int i = start; i < static_cast<int>(idx.size()); ++i)
								fi.push_back(::tvm::PrimExpr(idx[i]));
							fi.push_back(k1->var);
							return ::tvm::sum(t(fi) / ::tvm::cast(dtype, lastDim), {k1});
						}, "var_mean");
				auto k2 = ::tvm::te::reduce_axis(::tvm::Range(0, lastDim), "k2");
				auto meanSqT = ::tvm::te::compute(outShape,
						[t, k2, lastDim, dtype, batchDims](::tvm::ffi::Array<::tvm::tir::Var> idx) {
							::tvm::ffi::Array<::tvm::PrimExpr> fi;
							int start = static_cast<int>(idx.size()) - batchDims;
							for (int i = start; i < static_cast<int>(idx.size()); ++i)
								fi.push_back(::tvm::PrimExpr(idx[i]));
							fi.push_back(k2->var);
							auto x = t(fi);
							return ::tvm::sum(x * x / ::tvm::cast(dtype, lastDim), {k2});
						}, "var_mean_sq");
				return ::tvm::topi::subtract(meanSqT, ::tvm::topi::multiply(meanT, meanT));
			},
			/** Einops **/
			/*[RESHAPE] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// ts[0]: tensor; ts[1..]: single-element tensors encoding shape dims.
				// Sentinels:
				//   -1: Ellipsis/batch passthrough — replace with the next leading input dim.
				//   -2: Unknown group dim — infer as total_input / product_of_other_output_dims.
				//       We compute this explicitly rather than relying on topi::reshape's -1
				//       inference, which breaks when batch dims are symbolic.
				const auto& tensor = tvmArg<::tvm::te::Tensor>(ts[0]);

				// Pass 1: resolve all dims; record position of -2 infer sentinel.
				std::vector<::tvm::PrimExpr> dims;
				dims.reserve(ts.size() - 1);
				int passthroughIdx = 0;
				int inferPos = -1;
				for (size_t i = 1; i < ts.size(); ++i) {
					auto dim = bodyOf(tvmArg<::tvm::te::Tensor>(ts[i]));
					if (const auto* imm = dim.template as<::tvm::IntImmNode>()) {
						if (imm->value == -1) {
							// Batch-dim passthrough.
							if (passthroughIdx < static_cast<int>(tensor->shape.size()))
								dim = tensor->shape[passthroughIdx++];
						} else if (imm->value == -2) {
							// Infer position; use 1 as placeholder.
							inferPos = static_cast<int>(dims.size());
							dim = ::tvm::IntImm(::tvm::DataType::Int(32), 1);
						}
					}
					dims.push_back(dim);
				}

				// Pass 2: if there is an infer dim, compute it as
				//   product_of_unconsumed_input_dims / product_of_non_passthrough_non_infer_output_dims.
				// Using only the dims AFTER the passthrough region avoids symbolic batch
				// variables that TVM cannot simplify away.
				if (inferPos >= 0) {
					::tvm::PrimExpr remaining{::tvm::IntImm(::tvm::DataType::Int(32), 1)};
					for (int j = passthroughIdx; j < static_cast<int>(tensor->shape.size()); ++j)
						remaining = remaining * ::tvm::cast(::tvm::DataType::Int(32), tensor->shape[j]);
					::tvm::PrimExpr others{::tvm::IntImm(::tvm::DataType::Int(32), 1)};
					for (int j = passthroughIdx; j < static_cast<int>(dims.size()); ++j)
						if (j != inferPos)
							others = others * ::tvm::cast(::tvm::DataType::Int(32), dims[j]);
					dims[inferPos] = ::tvm::indexdiv(remaining, others);
				}

				::tvm::ffi::Array<::tvm::PrimExpr> shape(dims.begin(), dims.end());
				return ::tvm::topi::reshape(tensor, shape);
			},
			/*[PERMUTE] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// ts[0]: tensor; ts[1]: 1D compute tensor encoding the permutation via select chain.
				// Negative indices are resolved relative to the actual tensor rank.
				// When perm.size() < rank (trailing permutation), the leading rank-perm.size()
				// axes are mapped to themselves and the trailing axes follow the given permutation.
				const auto& tensor = tvmArg<::tvm::te::Tensor>(ts[0]);
				const auto& permTensor = tvmArg<::tvm::te::Tensor>(ts[1]);
				int rank = static_cast<int>(tensor->shape.size());
				int permSize = static_cast<int>(
						::tvm::Downcast<::tvm::IntImm>(permTensor->shape[0])->value
				);
				int offset = rank - permSize;
				::tvm::ffi::Array<::tvm::Integer> axes;
				for (int i = 0; i < offset; ++i)
					axes.push_back(::tvm::Integer(i));
				for (int i = 0; i < permSize; ++i) {
					int idx = extractInt(permTensor, i);
					if (idx < 0)
						idx += rank;
					axes.push_back(::tvm::Integer(idx));
				}
				// Validate: wrong size or any out-of-range/duplicate axis → identity (no-op).
				// This handles the case where a permutation was computed for a higher logical rank
				// than the physical tensor rank (e.g. template dims merged into a single axis).
				bool valid = (static_cast<int>(axes.size()) == rank);
				if (valid) {
					std::vector<bool> used(rank, false);
					for (const auto& ax : axes) {
						int a = static_cast<int>(ax->value);
						if (a < 0 || a >= rank || used[a]) { valid = false; break; }
						used[a] = true;
					}
				}
				if (!valid)
					return tensor;
				return ::tvm::topi::transpose(tensor, axes);
			},
			/*[EXPAND] =*/
			[](auto ts) -> ::tvm::te::Tensor {
				// ts[0]: tensor; ts[1..]: single-element tensors encoding target shape dims.
				// -1 sentinels mean "keep the input dim at this position unchanged".
				// Replace with the actual input dimension so TVM never sees a negative extent.
				const auto& tensor = tvmArg<::tvm::te::Tensor>(ts[0]);
				::tvm::ffi::Array<::tvm::PrimExpr> shape;
				for (size_t i = 1; i < ts.size(); ++i) {
					auto dim = bodyOf(tvmArg<::tvm::te::Tensor>(ts[i]));
					if (const auto* imm = dim.template as<::tvm::IntImmNode>(); imm && imm->value == -1) {
						int idx = static_cast<int>(i - 1);
						if (idx < static_cast<int>(tensor->shape.size()))
							dim = tensor->shape[idx];
					}
					shape.push_back(dim);
				}
				return ::tvm::topi::broadcast_to(tensor, shape);
			},
	};

} // namespace mdl::backend::tvm

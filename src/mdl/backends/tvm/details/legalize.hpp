#pragma once

/**
 * @file legalize.hpp
 * @brief Legalization of all Relax operators that the MDL backend emits (Relax → TIR), implemented in C++.
 * @details TVM registers the default legalizations of Relax operators from Python. To compile without Python, the
 * backend provides its own handlers for exactly the operators it emits (the translation of the builtins and the
 * outputs of the backend's pattern rewrites), as TVM-generated kernels (TOPI). Optimized kernels come from kernel
 * packages (kernel_match.hpp), which are selected before legalization.
 */

#include <mdl/compiler.hpp>

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>

#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace mdl::backend::tvm {

	/**
	 * @brief Creates a call of the backend operator `mdl.matmul`: `alpha * (op(a) @ op(b)) + bias` with
	 * `op(x) = x^T` (transposing the last two dimensions) if @p transA / @p transB.
	 * @details Shapes: a `[..., M, K]` with b `[K, N]` (or `[N, K]`), or a and b batched with the same rank (b's batch
	 * dimensions may be 1). bias is `[N]`. A transposed a (`[..., K, M]`) with a 2-D b must be 2-D.
	 */
	::tvm::relax::Expr makeMatmul(
			::tvm::relax::Expr a, ::tvm::relax::Expr b, std::optional<::tvm::relax::Expr> bias, bool transB,
			double alpha, bool transA = false
	);

	/**
	 * @brief Creates a call of the backend operator `mdl.dequant`: the float weight of a quantized one (see
	 * builtins::QuantFormat; @p zeros is PrimValue(0) for symmetric formats).
	 */
	::tvm::relax::Expr makeDequant(
			::tvm::relax::Expr q, ::tvm::relax::Expr scales, ::tvm::relax::Expr zeros, int64_t format, int64_t group
	);

	/** @brief Creates a call of the backend operator `mdl.gelu_backward`: dy * GELU'(x) (exact GELU). */
	::tvm::relax::Expr makeGeluBackward(::tvm::relax::Expr x, ::tvm::relax::Expr dy);

	/**
	 * @brief Folds permutations into mdl.matmul: permuted operands are read through strides, and a permuted product
	 * (without other uses) is written in the permuted layout (e.g. the heads of attention), instead of transposing.
	 */
	::tvm::transform::Pass foldPermutesIntoMatmul();

	/**
	 * @brief The arguments of an extern kernel call, computed by an adapter (see kernel packages, `adapter:`): the
	 * tensors it reads (in the order of their buffers), and the kernel's arguments in order.
	 */
	struct ExternPlan {
		struct Arg {
			enum class Kind { input, output, scalar } kind;
			size_t input = 0;	   ///< the index into inputs
			::tvm::PrimExpr value; ///< a scalar
		};
		std::vector<::tvm::relax::Expr> inputs;
		std::vector<Arg> args;
		::tvm::ffi::Array<::tvm::PrimExpr> outShape;
		::tvm::PrimType outDtype = ::tvm::PrimType::Void();

		Arg input(size_t i) const { return {.kind = Arg::Kind::input, .input = i}; }
		Arg add(const ::tvm::relax::Expr& e) {
			inputs.push_back(e);
			return input(inputs.size() - 1);
		}
		static Arg output() { return {.kind = Arg::Kind::output}; }
		static Arg scalar(::tvm::PrimExpr v) { return {.kind = Arg::Kind::scalar, .value = std::move(v)}; }
	};

	/**
	 * @brief The adapters `matmul-strided` (@p quantized false: mdl_matmul_<type>) and `matmul-quantized`
	 * (mdl_matmul_q_<type>) for a call of mdl.matmul or relax.matmul, or why the kernel does not cover it.
	 */
	std::expected<ExternPlan, std::string> matmulAdapter(const ::tvm::Call& call, bool quantized);

	/**
	 * @brief The adapter `attention-strided` (mdl_attention_<type>) for attention `softmax(scale * Q @ K^T) @ V`: @p out
	 * is the product with V, @p scores the product Q @ K^T (both mdl.matmul), or why the kernel does not cover it.
	 */
	std::expected<ExternPlan, std::string> attentionAdapter(const ::tvm::Call& out, const ::tvm::Call& scores);

	/** @brief The handlers for tvm::relax::transform::LegalizeOps (operator name → handler). */
	::tvm::ffi::Map<::tvm::ffi::String, ::tvm::ffi::Function> legalizeMap(const mdl::CompileOptions& opts);

} // namespace mdl::backend::tvm

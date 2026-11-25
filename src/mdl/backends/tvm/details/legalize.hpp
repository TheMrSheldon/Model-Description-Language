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
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/function.h>

#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "./permutations.hpp"

namespace mdl::backend::tvm {

	/**
	 * @brief A call of `mdl.matmul`: `permute(alpha * permute(a, permA) @ permute(b, permB), permOut)` (see the
	 * legalizer's MatmulArgs).
	 */
	::tvm::relax::Expr
	makeMatmul(::tvm::relax::Expr a, ::tvm::relax::Expr b, Perm permA, Perm permB, Perm permOut, double alpha);

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

	/**
	 * @brief Creates a call of the backend operator `mdl.scatter_add_rows`: zeros [rows, R] (float32), then
	 * `out[ids[n]] += g[n]` for every n (g: [N, R] float32, ids: [N]; indices clipped). Targets may override its kernel
	 * (see Target::overrideLegalizations).
	 */
	::tvm::relax::Expr makeScatterAddRows(::tvm::relax::Expr g, ::tvm::relax::Expr ids, ::tvm::PrimExpr rows);

	/** @brief A handler for LegalizeOps (see legalizeMap). */
	::tvm::ffi::Function
	legalizeHandler(std::function<::tvm::relax::Expr(const ::tvm::relax::BlockBuilder&, const ::tvm::Call&)> fn);

	/** @brief `call_tir` of @p func (added to the module as @p name) with @p inputs and the result @p out. */
	::tvm::relax::Expr callPrimFunc(
			const ::tvm::relax::BlockBuilder& bb, ::tvm::tirx::PrimFunc func, const std::string& name,
			::tvm::ffi::Array<::tvm::relax::Expr> inputs, const ::tvm::relax::TensorType& out
	);

	/**
	 * @brief Creates a call of the backend operator `mdl.suffix_mask`: bool [batch..., S], true at the positions
	 * s < lengths[batch...] (lengths: [batch...] integer; S: the next dimension of @p like, which is not read).
	 */
	::tvm::relax::Expr makeSuffixMask(::tvm::relax::Expr lengths, ::tvm::relax::Expr like);

	/** @brief Creates a call of the backend operator `mdl.gelu_backward`: dy * GELU'(x) (exact GELU). */
	::tvm::relax::Expr makeGeluBackward(::tvm::relax::Expr x, ::tvm::relax::Expr dy);

	/**
	 * @brief Folds permutations into mdl.matmul: permuted operands are read through strides, and a permuted product
	 * (without other uses) is written in the permuted layout (e.g. the heads of attention), instead of transposing.
	 */
	::tvm::transform::Pass foldPermutesIntoMatmul();

	/** @brief The handlers for tvm::relax::transform::LegalizeOps (operator name → handler). */
	::tvm::ffi::Map<::tvm::ffi::String, ::tvm::ffi::Function> legalizeMap(const mdl::CompileOptions& opts);

} // namespace mdl::backend::tvm

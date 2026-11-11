#pragma once

/**
 * @file legalize.hpp
 * @brief Legalization of all Relax operators that the MDL backend emits (Relax → TIR), implemented in C++.
 * @details TVM registers the default legalizations of Relax operators from Python. To compile without Python, the
 * backend provides its own handlers for exactly the operators it emits (the translation of the builtins and the
 * outputs of the backend's pattern rewrites). Depending on mdl::Kernels, recognized operators (matmul, softmax, GELU,
 * layer norm) are lowered to calls of the vendor kernels in `mdl_kernels` or to TVM-generated kernels (TOPI).
 */

#include <mdl/compiler.hpp>

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/relax/expr.h>

#include <optional>

namespace mdl::backend::tvm {

	/**
	 * @brief Creates a call of the backend operator `mdl.matmul`: `alpha * (a @ op(b)) + bias` with `op(b) = b^T` if
	 * @p transB (transposing the last two dimensions).
	 * @details Shapes: a `[..., M, K]` with b `[K, N]` (or `[N, K]`), or a and b batched with the same rank (b's batch
	 * dimensions may be 1). bias is `[N]`.
	 */
	::tvm::relax::Expr makeMatmul(
			::tvm::relax::Expr a, ::tvm::relax::Expr b, std::optional<::tvm::relax::Expr> bias, bool transB,
			double alpha
	);

	/** @brief The handlers for tvm::relax::transform::LegalizeOps (operator name → handler). */
	::tvm::ffi::Map<::tvm::ffi::String, ::tvm::ffi::Function> legalizeMap(const mdl::CompileOptions& opts);

} // namespace mdl::backend::tvm

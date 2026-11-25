#pragma once

/**
 * @file permutations.hpp
 * @brief Permutations of tensor dimensions, as the backend's operators carry them (e.g. mdl.matmul, mdl.attention):
 * operands read and results written through permutations instead of transposed copies.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>

#include <cstdint>
#include <vector>

namespace mdl::backend::tvm {

	/** @brief A permutation of the dimensions of a tensor (empty: none). */
	using Perm = std::vector<int64_t>;
	/** @brief The permutation of a ShapeExpr argument. */
	Perm permOf(const ::tvm::relax::Expr& e);
	/** @brief A permutation as a ShapeExpr argument. */
	::tvm::relax::Expr permExpr(const Perm& perm);
	/** @brief permute(xs, perm): entry i is xs[perm[i]] (like relax.permute_dims). */
	::tvm::ffi::Array<::tvm::PrimExpr> permuted(const ::tvm::ffi::Array<::tvm::PrimExpr>& xs, const Perm& perm);
	/** @brief permute(permute(x, inner), outer) = permute(x, compose(outer, inner)). */
	Perm compose(const Perm& outer, const Perm& inner);
	/** @brief The inverse permutation (empty for empty). */
	Perm inverse(const Perm& perm);
	Perm swapLastTwo(size_t rank);
	bool isIdentity(const Perm& p);
	/** @brief Row-major strides of a contiguous tensor of shape @p shape. */
	::tvm::ffi::Array<::tvm::PrimExpr> contiguousStrides(const ::tvm::ffi::Array<::tvm::PrimExpr>& shape);

} // namespace mdl::backend::tvm

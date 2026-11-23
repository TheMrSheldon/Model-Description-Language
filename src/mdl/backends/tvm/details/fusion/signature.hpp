#pragma once
/**
 * @file signature.hpp
 * @brief The loop signature of a kernel call: what kind of loop nest computes its result, over which domain.
 */
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include <optional>

namespace mdl::backend::tvm::fusion {

	/** @brief A domain: the non-unit extents of a shape, outermost first (unit dimensions do not change the loops). */
	using Domain = ::tvm::ffi::Array<::tvm::PrimExpr>;

	/** @brief Whether two domains are provably equal. */
	bool sameDomain(const Domain& a, const Domain& b);

	/**
	 * @brief How a kernel's loops compute its result (from the blocks of its PrimFunc; the domains from the Relax types
	 * of the call, so that the domains of different kernels compare).
	 * - Pointwise: spatial loops only (elementwise, broadcasts, casts, reshapes, transposes, gathers): `out`.
	 * - Reduction: reductions over the `full` input domain that keep the dimensions `space`; rows: they reduce the
	 *   innermost dimensions (`full` = space × reduced), columns: the outermost ones (`full` = reduced × space). The
	 *   result is per kept index (`out` = space) or, after an epilogue over the rows, of the full domain.
	 * - Anchor: a reduction whose operands are read along different dimensions (a matrix product without a library
	 *   kernel, pooling), which only pointwise consumers join.
	 * - Opaque: never fused (library calls, kernels marked `op_pattern` 8, several results, unanalyzable loops).
	 */
	struct Signature {
		enum class Kind { pointwise, reduction, anchor, opaque };
		enum class Layout { rows, columns };
		Kind kind = Kind::opaque;
		Layout layout = Layout::rows;
		Domain out;		   ///< the result's domain
		Domain space;	   ///< reduction: the kept dimensions (else `out`)
		Domain full;	   ///< reduction: the reduced input's domain (else `out`)
		bool view = false; ///< a reshape: free on its own (a view of its input), worth nothing in a group by itself
	};

	/** @brief The signature of `var = call_tir(kernel, args)` (opaque for any other binding). */
	Signature signatureOf(const ::tvm::IRModule& mod, const ::tvm::Var& var, const ::tvm::Expr& value);

} // namespace mdl::backend::tvm::fusion

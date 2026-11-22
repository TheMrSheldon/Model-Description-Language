#pragma once
/**
 * @file fuse_rows.hpp
 * @brief Row fusion: reductions over the trailing dimensions of the same rows, the per-row values computed from them,
 * and the elementwise epilogue that uses them, as one kernel.
 */
#include <tvm/ir/transform.h>

namespace mdl::backend::tvm {
	/**
	 * @brief Groups the fused calls (after FuseOps, before FuseTIR) that work on the same rows: a reduction over the
	 * trailing dimensions L of tensors of shape [rows..., L...] (result [rows...]), further such reductions that read
	 * the same inputs or the results of the group, injective per-row computations on the results (shape [rows...], up
	 * to unit dimensions), and one elementwise epilogue of shape [rows..., L...] that reads them. FuseTIR then makes
	 * one kernel of the group, which the targets schedule as reductions followed by an epilogue (e.g. a warp per row
	 * on GPUs), instead of one kernel per reduction that reads the rows again.
	 * @details FuseOps puts at most one reduction into a group, so e.g. the gradient of a normalization, dx = rstd *
	 * (g - mean(g) - x̂ * mean(g * x̂)), takes five kernels. Only intermediate results that are not used outside the
	 * group are fused (the epilogue's result is the group's result). Target-independent.
	 */
	::tvm::transform::Pass fuseRows();
} // namespace mdl::backend::tvm

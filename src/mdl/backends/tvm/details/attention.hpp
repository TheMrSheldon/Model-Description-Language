#pragma once

/**
 * @file attention.hpp
 * @brief The backend operators of the attention builtins (see passes/attention.hpp): `mdl.attention` (the forward
 * pass and the row statistics) and `mdl.attention_grad` (the gradients w.r.t. q, k, and v).
 * @details Both return tuples, so that a fused kernel computes all results of one attention at once. Their tensor
 * operands are read through permutations and their results written through permutations (folded from the
 * surrounding `permute_dims`, e.g. the heads of multi-head attention), so that strided kernels need no copies.
 * Calls that no kernel package covers are expanded into the unfused operations (mdl.matmul and softmax) by
 * expandAttention, which are then matched again (e.g. by matrix product kernels).
 */

#include "./legalize.hpp"

#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>

#include <expected>
#include <string>

namespace mdl::backend::tvm {

	/**
	 * @brief `mdl.attention(q, k, v, scale, perm_q, perm_k, perm_v, perm_out, with_lse)`: the tuple (out, lse) with
	 * Q = permute(q, perm_q) [..., Sq, D], K = permute(k, perm_k) [..., Skv, D], V = permute(v, perm_v) [..., Skv, Dv],
	 * out = permute(softmax(scale * Q @ K^T) @ V, perm_out), and lse [..., Sq] (float32) the log-sum-exp of the rows
	 * of scale * Q @ K^T. with_lse: whether lse is used (otherwise its contents are unspecified).
	 */
	::tvm::relax::Expr makeAttention(::tvm::relax::Expr q, ::tvm::relax::Expr k, ::tvm::relax::Expr v, double scale);

	/**
	 * @brief `mdl.attention_grad(q, k, v, out, lse, probs, dy, scale, perm_q, perm_k, perm_v, perm_out, perm_dy,
	 * perm_dq, perm_dk, perm_dv)`: the tuple (dq, dk, dv) of the gradients of mdl.attention for the cotangent dy of its
	 * result (out, lse: its results; probs: softmax(scale * Q @ K^T); O = permute(out, perm_out) and dY = permute(dy,
	 * perm_dy) like the inputs; the gradients are written as permute(dQ, perm_dq) etc.). A fused kernel reads out and
	 * lse, the unfused operations (expandAttention) probs; the backend saves only what is read (see pruneContexts).
	 */
	::tvm::relax::Expr makeAttentionGrad(
			::tvm::relax::Expr q, ::tvm::relax::Expr k, ::tvm::relax::Expr v, ::tvm::relax::Expr out,
			::tvm::relax::Expr lse, ::tvm::relax::Expr probs, ::tvm::relax::Expr dy, double scale
	);

	/**
	 * @brief Folds permutations into the attention operators: permuted operands are read through strides; a result
	 * whose uses are a single permutation (and attention operators, which then read it through the inverse) is
	 * written in the permuted layout. Also records whether lse is used.
	 */
	::tvm::transform::Pass foldPermutesIntoAttention();

	/** @brief Expands the attention operators into the unfused operations (after kernel matching). */
	::tvm::transform::Pass expandAttention();

	/** @brief Whether @p mod calls an attention operator. */
	bool hasAttention(const ::tvm::IRModule& mod);

	/**
	 * @brief The adapter `attention-forward` (mdl_attention_fwd_<type>, kernels/cudnn/include/mdl_cudnn.h) for a call
	 * of mdl.attention, or why the kernel does not cover it.
	 */
	std::expected<ExternPlan, std::string> attentionForwardAdapter(const ::tvm::Call& call);

	/**
	 * @brief The adapter `attention-backward` (mdl_attention_bwd_<type>) for a call of mdl.attention_grad, or why the
	 * kernel does not cover it.
	 */
	std::expected<ExternPlan, std::string> attentionBackwardAdapter(const ::tvm::Call& call);

} // namespace mdl::backend::tvm

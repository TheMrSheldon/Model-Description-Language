#ifndef MDL_PASSES_ATTENTION_HPP
#define MDL_PASSES_ATTENTION_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"

namespace mdl::attention {

	/**
	 * @brief Forms the attention builtins: every `softmax(scale * q @ k^T) @ v` in @p func (with the std softmax
	 * inlined: `exp(x - max(x))` normalized by its sum; the scale a scalar factor or divisor of the scores, or none)
	 * becomes `ATTENTION(q, k, v, scale)`.
	 * @details Target-independent: a backend computes ATTENTION as a fused kernel where it has one and otherwise as the
	 * original operations. Apply it before differentiation: the gradient of ATTENTION is ATTENTION_GRAD (from the
	 * forward's result and the row statistics ATTENTION_LSE), which a fused kernel computes without the score matrix.
	 * Masked attention (a bias added to the scores) keeps its operations.
	 * @return @p func with the attention builtins (the same function if it has no attention).
	 */
	FunctionDecl fuse(Computegraph& graph, const FunctionDecl& func);

} // namespace mdl::attention

#endif

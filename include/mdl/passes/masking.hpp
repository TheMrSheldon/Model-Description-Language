#ifndef MDL_PASSES_MASKING_HPP
#define MDL_PASSES_MASKING_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"
#include "./shapes.hpp"

#include <expected>
#include <string>

namespace mdl::masking {

	/** @brief A masked function and its mask input. */
	struct Masked {
		FunctionDecl function;
		const Node* mask;
	};

	/**
	 * @brief Makes @p func a masked function: it takes an additional input `mask: bool[...]` over the dimensions of
	 * `...`, and the elements at positions where the mask is false in its last dimension (e.g. the padding of shorter
	 * sequences in [batch, seq]) do not influence the valid elements of the result.
	 * @details Semantics like torch.masked: the values are dense, the mask says which ones are valid. Every dimension
	 * whose size is the last dimension of `...` (shape inference identifies them, see mdl::shapes) is masked with the
	 * mask's row of the batch dimension(s), e.g. both sequence dimensions of attention scores [heads, batch, seq, seq].
	 * The pass inserts masking where masked elements would mix with valid ones:
	 *  - reductions over a masked dimension: `max`/`min` ignore masked elements, `sum` adds zeros, `mean` divides by the
	 *    number of valid elements (likewise for `sum_all`/`mean_all`);
	 *  - contractions (`@`) over a masked dimension: masked elements are zero;
	 *  - the outputs: zero at masked positions.
	 * Elementwise operations need nothing: their masked elements stay masked. A function that does not use the mask has
	 * no masking (no cost).
	 *
	 * Apply it before differentiation: the gradient of the masked function is then the gradient of the masked
	 * computation. Applying it again to the gradient function (sanitize) additionally zeroes masked positions in the
	 * reductions of the backward pass (where the gradient is zero anyway), so that non-finite values at masked positions
	 * cannot reach the weight gradients.
	 *
	 * @param shapes the inferred shapes of @p func (see mdl::shapes::infer).
	 * @param mask the mask input to use (e.g. of the function that was differentiated); nullptr to add a new input
	 * `mask` (the last parameter of the result).
	 * @return the masked function (same name) and its mask input, or an error (e.g. if a masked dimension cannot be
	 * aligned with the mask).
	 */
	std::expected<Masked, std::string> apply(
			Computegraph& graph, const FunctionDecl& func, const shapes::Shapes& shapes, const Node* mask = nullptr
	);

} // namespace mdl::masking

#endif

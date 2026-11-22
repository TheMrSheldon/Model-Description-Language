#ifndef MDL_PASSES_MASKING_HPP
#define MDL_PASSES_MASKING_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"
#include "./shapes.hpp"

#include <expected>
#include <string>

namespace mdl::masking {

	/**
	 * @brief How a masked function takes its mask: element masks (any element may be masked) as `mask: bool[...]`,
	 * suffix masks (the valid elements first, e.g. padded sequences) as `lengths: int[...]` over the dimensions of `...`
	 * but the last (the number of valid elements in the last one). The semantics are the same; a suffix mask's structure
	 * is known to the compiler (see builtins::FunctionIDs::SUFFIX_MASK), e.g. for kernels that skip the masked suffix.
	 */
	enum class MaskKind { element, suffix };

	/** @brief A masked function and its mask input (`mask`, or `lengths` for a suffix mask). */
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
	 * The pass zeroes the masked positions of the floating-point inputs (so that all values at masked positions are
	 * finite, whatever the caller passed there) and inserts masking where masked elements would mix with valid ones:
	 *  - reductions over a masked dimension: `max`/`min` ignore masked elements, `sum` adds zeros, `mean` divides by the
	 *    number of valid elements. Likewise for a reduction over a flattening that merges the masked dimension with
	 *    others (e.g. `mean_flat`, in any order of the merged dimensions); other operations between the flattening and
	 *    the reduction are an error (the masked elements cannot be told apart after them);
	 *  - contractions (`@`) over a masked dimension: masked elements are zero;
	 *  - the outputs: zero at masked positions.
	 * Elementwise operations need nothing: their masked elements stay masked. A function that does not use the mask has
	 * no masking (no cost).
	 *
	 * Only values of masked type are masked: values computed from `x.data` and `x.mask` alone are plain (e.g.
	 * `x.data + (x.mask - 1) * 3.4e38` before a softmax, see the std softmax of masked tensors), so their masked elements
	 * are what the code made them.
	 *
	 * Apply it before differentiation: the gradient of the masked function is then the gradient of the masked
	 * computation. Its backward pass needs no masking: the cotangents are zero at masked positions (the outputs are
	 * zeroed there) and the values there are finite, so masked positions contribute exact zeros to the gradients.
	 *
	 * @param shapes the inferred shapes of @p func (see mdl::shapes::infer).
	 * @param kind how the mask is given (see MaskKind): an element mask `mask: bool[...]`, or a suffix mask
	 * `lengths: int[...]` (the leading dimensions of `...`; its boolean mask: SUFFIX_MASK of the lengths and the first
	 * parameter of `...`).
	 * @return the masked function (same name; the mask is its last parameter) and its mask input, or an error
	 * (e.g. if a masked dimension cannot be aligned with the mask).
	 */
	std::expected<Masked, std::string> apply(
			Computegraph& graph, const FunctionDecl& func, const shapes::Shapes& shapes, MaskKind kind = MaskKind::element
	);

} // namespace mdl::masking

#endif

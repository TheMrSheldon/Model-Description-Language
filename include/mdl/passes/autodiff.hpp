#ifndef MDL_PASSES_AUTODIFF_HPP
#define MDL_PASSES_AUTODIFF_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"
#include "../utils/pathpattern.hpp"

#include <expected>
#include <string>
#include <vector>

namespace mdl::autodiff {

	struct Options {
		/** @brief The weights to differentiate with respect to (by path); all weights if empty. */
		PathPatterns trainable;
		/** @brief Whether to also compute the gradients with respect to the function's inputs. */
		bool inputs = false;
	};

	/**
	 * @brief Builds the vector-Jacobian product (reverse-mode gradient) of @p func in the same graph.
	 * @details The returned function takes the inputs of @p func followed by one cotangent `d_<output>` per output and
	 * returns the outputs of @p func followed by the gradient of each selected weight (in the order of
	 * @p weightOrder, named `d_<path>`) and, if requested, the gradient of each input (`d_<input>`). Selected weights
	 * that do not influence the outputs get a zero gradient. The gradient is expressed with builtins only, so the result
	 * can be differentiated again (higher-order derivatives).
	 *
	 * @param weightOrder the weights of the module in canonical order (see mdl::weightPaths).
	 * @return the gradient function, or an error if a builtin without gradient rule is involved.
	 */
	std::expected<FunctionDecl, std::string> vjp(
			Computegraph& graph, const FunctionDecl& func, const std::vector<std::string>& weightOrder,
			const Options& options = {}
	);

} // namespace mdl::autodiff

#endif

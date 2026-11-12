#ifndef MDL_PASSES_AUTODIFF_HPP
#define MDL_PASSES_AUTODIFF_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"
#include "../utils/pathpattern.hpp"
#include "./shapes.hpp"

#include <expected>
#include <string>
#include <vector>

namespace mdl::autodiff {

	struct Options {
		/** @brief The weights to differentiate with respect to (by path); all weights if empty. */
		PathPatterns trainable;
		/** @brief Whether to also compute the gradients with respect to the function's inputs. */
		bool inputs = false;
		/**
		 * @brief Exact shapes (optional; for the `...` rank the function is compiled with). Without them, the pass
		 * decides from the declared types whether a gradient must be summed over broadcast dimensions, which is only
		 * reliable if the types are exact.
		 */
		shapes::Oracle* shapes = nullptr;
	};

	/**
	 * @brief The number of cotangent inputs of the gradient of @p func (see vjp): 0 if @p func is an objective (has a
	 * scalar output), else one per output.
	 */
	size_t numCotangents(const FunctionDecl& func);

	/**
	 * @brief Builds the vector-Jacobian product (reverse-mode gradient) of @p func in the same graph.
	 * @details The returned function takes the inputs of @p func followed by one cotangent `d_<output>` per output and
	 * returns the outputs of @p func followed by the gradient of each selected weight (in the order of
	 * @p weightOrder, named `d_<path>`) and, if requested, the gradient of each input (`d_<input>`). Selected weights
	 * that do not influence the outputs get a zero gradient. The gradient is expressed with builtins only, so the result
	 * can be differentiated again (higher-order derivatives).
	 *
	 * If @p func has a scalar output, it is an objective: the gradient is that of its scalar outputs (other outputs are
	 * auxiliary) and there are no cotangent inputs (see numCotangents).
	 *
	 * @param weightOrder the weights of the module in canonical order (see mdl::weightPaths).
	 * @return the gradient function, or an error if a builtin without gradient rule is involved.
	 */
	std::expected<FunctionDecl, std::string> vjp(
			Computegraph& graph, const FunctionDecl& func, const std::vector<std::string>& weightOrder,
			const Options& options = {}
	);

	/** @brief A gradient function split into a forward pass that saves residuals and a backward pass. */
	struct Split {
		/**
		 * @brief `train(inputs...) -> (outputs..., residuals...)`: the forward pass, which also returns the values the
		 * backward pass needs (the last `numResiduals` outputs).
		 */
		FunctionDecl train;
		/** @brief `backward(residuals..., cotangents...) -> gradients...` (the first `numResiduals` parameters). */
		FunctionDecl backward;
		size_t numResiduals = 0;
	};

	/**
	 * @brief Splits a gradient function (see vjp) into a forward and a backward pass.
	 * @details The residuals are the values computed from the inputs that the backward pass needs. Values that only
	 * depend on weights and constants, and cheap reshaping operations of residuals, are recomputed in the backward
	 * pass instead of being saved.
	 *
	 * @param numOutputs the number of outputs of the differentiated function (the first outputs of @p vjp; its last
	 * @p numOutputs parameters are the cotangents).
	 * @param checkpoint module instances (gitignore-style patterns over their paths, e.g. `encoder.layers.*`) whose
	 * computations are recomputed in the backward pass instead of being saved (gradient checkpointing): only their
	 * inputs are saved. nullptr for none.
	 */
	std::expected<Split, std::string> split(
			Computegraph& graph, const FunctionDecl& vjp, size_t numOutputs, const PathPatterns* checkpoint = nullptr
	);

} // namespace mdl::autodiff

#endif

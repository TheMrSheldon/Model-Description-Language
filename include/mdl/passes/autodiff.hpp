#ifndef MDL_PASSES_AUTODIFF_HPP
#define MDL_PASSES_AUTODIFF_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"
#include "../utils/pathpattern.hpp"
#include "./shapes.hpp"

#include <expected>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace mdl::autodiff {

	struct Options {
		/** @brief The weights to differentiate with respect to (by path); all weights if empty. */
		PathPatterns trainable;
		/** @brief Weights that are never differentiated, whatever @p trainable selects (e.g. quantized weights). */
		std::set<std::string> frozen = {};
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
	std::expected<FunctionDecl, std::string>
	vjp(Computegraph& graph, const FunctionDecl& func, const std::vector<std::string>& weightOrder,
		const Options& options = {});

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

	struct SplitOptions {
		/**
		 * @brief Module instances (gitignore-style patterns over their paths, e.g. `encoder.layers.*`) whose
		 * computations are recomputed in the backward pass instead of being saved (gradient checkpointing): only their
		 * inputs are saved. nullptr for none.
		 */
		const PathPatterns* checkpoint = nullptr;
		/** @brief Exact shapes, for the sizes of the values (optional; else from the declared types). */
		shapes::Oracle* shapes = nullptr;
		/**
		 * @brief The bytes the residuals may take at most, at `typicalDim` (internal; not exposed to users yet):
		 * above it, values are recomputed instead of saved wherever possible (see docs/technical/optimization.md).
		 */
		double memoryBudget = std::numeric_limits<double>::infinity();
		/** @brief The size assumed for dimensions known only at runtime (e.g. batch, sequence), to compare sizes. */
		int64_t typicalDim = 128;
	};

	/**
	 * @brief Splits a gradient function (see vjp) into a forward and a backward pass.
	 * @details The residuals are the values computed from the inputs that the backward pass needs, chosen by a minimum
	 * cut that minimizes their bytes (as PyTorch's AOTAutograd partitioner does): the backward pass recomputes the
	 * other values from them. Matrix products, attention and other library operations are never recomputed; values
	 * that a non-fusible operation of the backward pass reads are saved, unless that exceeds the memory budget. Values that only depend on weights and constants are recomputed.
	 * See docs/technical/optimization.md.
	 *
	 * @param numOutputs the number of outputs of the differentiated function (the first outputs of @p vjp; its last
	 * @p numOutputs parameters are the cotangents).
	 */
	std::expected<Split, std::string>
	split(Computegraph& graph, const FunctionDecl& vjp, size_t numOutputs, const SplitOptions& options = {});

} // namespace mdl::autodiff

#endif

#ifndef MDL_COMPUTEGRAPH_BUILTINS_HPP
#define MDL_COMPUTEGRAPH_BUILTINS_HPP

#include <cstddef>
#include <limits>

namespace mdl::builtins {
	/**
     * @brief Enumerates all builtin functions. These functions make up the basic operations in the computegraph.
     * @details The compilation backend must provide an "implementation" for each of these functions by translating them into their internal representation.
     */
	enum class FunctionIDs : std::size_t {
		/** Tensor Operators **/
		TENSOR_ADD,
		TENSOR_SUB,
		TENSOR_DOT,
		TENSOR_PROD,
		TENSOR_DIV,
		/** Statistic Functions **/
		MAX,
		MIN,
		MEAN,
		SUM,
		/** Arithmetic Functions **/
		COS,
		SIN,
		TAN,
		EXP,
		ERF,
		SQRT,
		RSQRT,
		LOG,
		TANH,
		VAR,
		/** Einops **/
		/**
		 * Reshapes a tensor: arguments are the tensor and one entry per output dimension: a size, -1 (copy the next
		 * leading input dimension), -2 (infer), or a vector constant `{before, after}` that denotes an ellipsis group,
		 * i.e., the input dimensions `[before, rank - after)`.
		 */
		RESHAPE,
		/**
		 * Permutes a tensor: arguments are the tensor and a vector constant with one entry per output dimension:
		 * non-negative entries index the input dimensions before an ellipsis, negative entries count from the end,
		 * and kEllipsisPermMarker stands for the (possibly multi-dimensional) ellipsis group, which covers all input
		 * dimensions that are not listed. Without an ellipsis, a shorter permutation applies to the trailing dims.
		 */
		PERMUTE,
		EXPAND,

		COUNT
	};

	/** @brief Marks the position of the ellipsis group in a PERMUTE permutation (see FunctionIDs::PERMUTE). */
	inline constexpr int kEllipsisPermMarker = std::numeric_limits<int>::min();
} // namespace mdl::builtins

#endif
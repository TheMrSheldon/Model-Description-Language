#ifndef MDL_COMPUTEGRAPH_BUILTINS_HPP
#define MDL_COMPUTEGRAPH_BUILTINS_HPP

#include <cstddef>

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
		RESHAPE,
		PERMUTE, /**< Permutes a tensor based on the given permutation **/
		EXPAND,

		COUNT
	};
} // namespace mdl::builtins

#endif
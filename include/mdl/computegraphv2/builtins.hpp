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
		/** Adjoints (used by the gradient pass; each is differentiable again) **/
		COLLAPSE_SUM_LIKE, /**< (g, ref): sums g over the dimensions in which ref was broadcast (result: ref's shape) */
		BROADCAST_LIKE,	   /**< (g, ref): broadcasts g to ref's shape */
		RESHAPE_LIKE,	   /**< (g, ref): reshapes g to ref's shape */
		PERMUTE_INVERSE,   /**< (g, perm): permutes g by the inverse of the PERMUTE permutation perm */
		EQUAL_MASK,		   /**< (a, b): 1 where a == b (broadcasting), 0 elsewhere */
		/** Differentiation **/
		STOP_GRADIENT, /**< (x): x; no gradient flows through it (std `nograd`) */
		/** Full reductions (e.g. objectives) **/
		SUM_ALL,  /**< (x): the sum of all elements (a scalar) */
		MEAN_ALL, /**< (x): the mean of all elements (a scalar) */
		/** Indexing (integer tensors) **/
		TAKE,			  /**< (table [V, ...], ids int[...]): the rows table[ids] ([..., ...]), e.g. embeddings */
		SCATTER_ADD_ROWS, /**< (g [..., ...], ids int[...], like [V, ...]): zeros like `like`, plus g's rows added at ids
							 (the adjoint of TAKE) */
		ONE_HOT_LIKE,	  /**< (labels int[...], like [..., C]): 1.0 at labels, 0.0 elsewhere (like's shape; C classes) */
		/** Selection **/
		WHERE, /**< (cond bool[...], a, b): a where cond, b elsewhere (all three broadcast) */
		/** Masked tensors (resolved by the masking pass; backends never see them) **/
		MASK_OF, /**< (x masked): `x.mask`, 1.0 at x's valid elements and 0.0 at masked ones (broadcastable to x) */
		UNMASK,	 /**< (x masked): `x.data`, x's values with zeros at the masked elements (an unmasked tensor) */
		/** Selection along an axis **/
		SELECT,		   /**< (x, axis, index): x at `index` along `axis` (the axis is removed; negative: from the end) */
		UNSELECT_LIKE, /**< (g, axis, index, like): zeros like `like`, g at `index` along `axis` (the adjoint) */
		/** Conversion of the element type **/
		CAST, /**< (x, type): x converted to the element type `type` (a constant: the value of mdl::PrimitiveType) */
		/** Stacking **/
		STACK, /**< (x_0, ..., x_{n-1}): the tensors (of equal shapes) stacked along a new first dimension [n, ...] */
		/** Quantization **/
		DEQUANT, /**< (q, scales, zeros, format, group): the float weight [N, K] of a quantized one (see QuantFormat) */

		COUNT
	};

	/**
	 * @brief The formats of quantized weights (the constant `format` of FunctionIDs::DEQUANT). A weight W [N, K] (e.g.
	 * of a linear layer: N outputs, K inputs) is stored as q [N, K] int8 (int4: [N, K/2], two values per byte, the
	 * even k in the low nibble), scales [K/group, N] in the float type of W, and (asymmetric) zeros [K/group, N] int8:
	 * W[n, k] = scales[k/group, n] * (q[n, k] - zeros[k/group, n]) (the layout of GPTQ checkpoints and of oneDNN).
	 */
	enum class QuantFormat : int {
		int8 = 1,		 ///< signed 8 bits, symmetric (no zeros)
		int4 = 2,		 ///< signed 4 bits (-8..7), symmetric (no zeros)
		int4Asym = 3,	 ///< unsigned 4 bits (0..15) with zeros
	};

	/** @brief Marks the position of the ellipsis group in a PERMUTE permutation (see FunctionIDs::PERMUTE). */
	inline constexpr int kEllipsisPermMarker = std::numeric_limits<int>::min();
} // namespace mdl::builtins

#endif
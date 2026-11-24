/**
 * @file
 * @brief Helpers for gradients, in particular for custom `gradient` functions of modules.
 */

/**
 * @brief Sums \p x over the dimensions in which a tensor of the shape of \p like would be broadcast to the shape of
 * \p x (the adjoint of broadcasting); the result has the shape of \p like.
 * @details Typically used for weight gradients in custom `gradient` functions, e.g. `sum_like(dy, bias)` sums the
 * output cotangent over all batch dimensions.
 * @ingroup Autodiff
 */
public float[...] sum_like(x: float[...], like: float[...]) => __builtin_sum_like(x, like);

/**
 * @brief Broadcasts \p x to the shape of \p like.
 * @ingroup Autodiff
 */
public float[...] broadcast_like(x: float[...], like: float[...]) => __builtin_broadcast_like(x, like);

/**
 * @brief Returns \p x unchanged but stops gradients: no gradient flows from the result back to \p x.
 * @ingroup Autodiff
 */
public float[...] nograd(x: float[...]) => __builtin_stop_gradient(x);

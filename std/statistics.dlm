/**
 * @file
 * @brief Reductions over the last dimension.
 * \todo support selecting the reduced dimension(s).
 */

/**
 * @brief Sum over the last dimension.
 * @ingroup Statistics
 */
template<N: uint>
float[...] sum(x: float[..., N]) => __builtin_sum(x);

/**
 * @brief Maximum over the last dimension.
 * @ingroup Statistics
 */
template<N: uint>
float[...] max(x: float[..., N]) => __builtin_max(x);

/**
 * @brief Minimum over the last dimension.
 * @ingroup Statistics
 */
template<N: uint>
float[...] min(x: float[..., N]) => __builtin_min(x);

/**
 * @brief Mean over the last dimension.
 * @ingroup Statistics
 */
template<N: uint>
float[...] mean(x: float[..., N]) => __builtin_mean(x);

/**
 * @brief Variance over the last dimension.
 * @ingroup Statistics
 * \todo `dof` is not yet forwarded (default values are not yet supported).
 */
float[...] var(x: float[...], dof: uint) => __builtin_var(x, dof);

/**
 * @brief Sum of all elements (a scalar), e.g. for objectives.
 * @ingroup Statistics
 */
float sum_all(x: float[...]) => __builtin_sum_all(x);

/**
 * @brief Mean of all elements (a scalar), e.g. for objectives.
 * @ingroup Statistics
 */
float mean_all(x: float[...]) => __builtin_mean_all(x);

/**
 * @brief The element at `index` along the axis `axis` (the axis is removed; negative values count from the end),
 * e.g. `select(h, -2, 0)` is the first position of a sequence h: [..., seq, dim].
 * @ingroup Statistics
 */
float[...] select(x: float[...], axis: int, index: int) => __builtin_select(x, axis, index);

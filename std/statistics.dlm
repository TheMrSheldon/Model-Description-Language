/**
 * @file
 * @brief Reductions over the last dimension.
 * \todo support selecting the reduced dimension(s).
 */

/**
 * @brief Sum over the last dimension.
 * @ingroup Statistics
 */
public template<N: uint>
float[...] sum(x: float[..., N]) => __builtin_sum(x);

/**
 * @brief Maximum over the last dimension.
 * @ingroup Statistics
 */
public template<N: uint>
float[...] max(x: float[..., N]) => __builtin_max(x);

/**
 * @brief Minimum over the last dimension.
 * @ingroup Statistics
 */
public template<N: uint>
float[...] min(x: float[..., N]) => __builtin_min(x);

/**
 * @brief Mean over the last dimension.
 * @ingroup Statistics
 */
public template<N: uint>
float[...] mean(x: float[..., N]) => __builtin_mean(x);

/**
 * @brief Variance over the last dimension: `sum((x - mean(x))^2) / (N - dof)`.
 * @param dof the delta degrees of freedom: 0 for the population variance, 1 for the unbiased sample variance.
 * @ingroup Statistics
 */
public template<N: uint>
float[...] var(x: float[..., N], dof: uint) => __builtin_var(x, dof);

/**
 * @brief Sum of all elements (a scalar), e.g. for objectives: the sum of the flattened tensor.
 * @ingroup Statistics
 */
public float sum_flat(x: float[...]) => sum(x[... -> (...)]);

/**
 * @brief Mean of all elements (a scalar), e.g. for objectives: the mean of the flattened tensor (with a mask, of the
 * valid elements).
 * @ingroup Statistics
 */
public float mean_flat(x: float[...]) => mean(x[... -> (...)]);

/**
 * @file
 * @brief Elementwise arithmetic functions.
 */

/**
 * @brief Computes the elementwise \p x -th power of e.
 * @details \f[(x_i)_{1\leq i \leq n} \mapsto (e^{x_i})_{1\leq i \leq n}.\f]
 * @ingroup Arithmetic
 *
 * @param x
 * @return the \p x -th power of e for  each component of \p x .
 */
float[...] exp(x: float[...]) => __builtin_exp(x);

/**
 * @brief Natural logarithm, \f(\log_e(\,\cdot\,)\f)
 * @details \f[(x_i)_{1\leq i \leq n} \mapsto (\log_e x_i)_{1\leq i \leq n}.\f]
 * @ingroup Arithmetic
 *
 * @param x
 * @return the natural logarithm of \p x .
 */
float[...] log(x: float[...]) => __builtin_log(x);

/**
 * @brief Computes the elementwise square root of \p x .
 * @details \f[(x_i)_{1\leq i \leq n} \mapsto (\sqrt{x_i})_{1\leq i \leq n}.\f]
 * @ingroup Arithmetic
 *
 * @param x
 * @return the square root of each component of \p x .
 */
float[...] sqrt(x: float[...]) => __builtin_sqrt(x);

/**
 * @brief Computes the reciprocal square root for each element in \p x.
 * @details \f[(x_i)_{1\leq i \leq n} \mapsto \left(\frac{1}{\sqrt{x_i}}\right)_{1\leq i \leq n}.\f]
 * @ingroup Arithmetic
 *
 * @param x
 * @return the reciprocal square root of each component of \p x .
 */
float[...] rsqrt(x: float[...]) => __builtin_rsqrt(x);

/**
 * @brief (Gauss) Error Function
 * @ingroup Arithmetic
 *
 * @param x
 * @return the error function of each component of \p x .
 */
float[...] erf(x: float[...]) => __builtin_erf(x);

/**
 * @brief Hyperbolic tangent
 * @ingroup Arithmetic
 *
 * @param x
 * @return the hyperbolic tangent of each component of \p x .
 */
float[...] tanh(x: float[...]) => __builtin_tanh(x);

/** @brief Elementwise cosine. @ingroup Arithmetic */
float[...] cos(x: float[...]) => __builtin_cos(x);

/** @brief Elementwise sine. @ingroup Arithmetic */
float[...] sin(x: float[...]) => __builtin_sin(x);

/** @brief Elementwise tangent. @ingroup Arithmetic */
float[...] tan(x: float[...]) => __builtin_tan(x);

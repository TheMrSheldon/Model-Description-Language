/**
 * @file
 * @brief Activation functions.
 * @details These are ordinary functions composed of builtins; backends may recognize them (pattern matching) and
 * replace them by optimized implementations where the target provides one.
 */
import std.arithmetic;
import std.statistics;

/**
 * @brief Computes the softmax function over the last dimension.
 * @details Let \f(x \in \mathbb R^n\f) be a vector, the softmax is defined as
 * \f[
 * 	\text{softmax}\colon \mathbb R^n \to \mathbb R^n
 * 	~\text{via}~ x\mapsto \left(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\right)_{1\leq i\leq n}.
 * \f]
 *
 * @param x the input tensor.
 * @return The softmax of \p x .
 */
float[...] softmax(x: float[...]) {
    auto power = exp(x - max(x));
    return power / sum(power);
}

/**
 * @brief Gaussian Error Linear Unit (exact, erf-based).
 * @details \f[\text{GELU}(x) := \frac x2\left(1 + \text{erf}\left(\frac x{\sqrt 2}\right)\right)\f]
 */
float[...] gelu(x: float[...]) => x * 0.5 * (1 + erf(x / sqrt(2)));

/**
 * @brief Logistic sigmoid, \f(\sigma(x) = 1 / (1 + e^{-x})\f).
 */
float[...] sigmoid(x: float[...]) => 1 / (1 + exp(-x));

/**
 * @brief Sigmoid Linear Unit (a.k.a. swish), \f(x\sigma(x)\f).
 */
float[...] silu(x: float[...]) => x * sigmoid(x);

/**
 * @file
 * @brief Activation functions.
 * @details These are ordinary functions composed of builtins; backends may recognize them (pattern matching) and
 * replace them by optimized implementations where the target provides one.
 */
from .arithmetic import *;
from .statistics import *;

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
public float[...] softmax(x: float[...]) {
    auto power = exp(x - max(x)[... -> ... 1]);
    return power / sum(power)[... -> ... 1];
}

/**
 * @brief Closed-form gradient of softmax (used by the gradient pass): only needs the result \f(y\f), which the
 * backward pass of attention saves anyway.
 */
float[...] softmax_gradient(x: float[...], dy: float[...]) {
    auto y = softmax(x);  // the forward pass's node (common subexpression)
    return y * (dy - sum(dy * y)[... -> ... 1]);
}

/**
 * @brief Softmax of a masked tensor: the masked elements get the probability 0 (they do not take part).
 * @details Adds a large negative bias at the masked elements (like an additive attention mask), so that this is an
 * ordinary softmax: backends can use their fused softmax, and the gradient pass uses softmax_gradient.
 */
public float[...] softmax(x: masked float[...]) => softmax(x.data + (x.mask - 1) * 3.4028234e38);

/**
 * @brief Gaussian Error Linear Unit (exact, erf-based).
 * @details \f[\text{GELU}(x) := \frac x2\left(1 + \text{erf}\left(\frac x{\sqrt 2}\right)\right)\f]
 */
public float[...] gelu(x: float[...]) => x * 0.5 * (1 + erf(x / sqrt(2)));

/**
 * @brief Closed-form gradient of GELU (used by the gradient pass):
 * \f(\text{GELU}'(x) = \frac12\left(1 + \text{erf}\left(\frac x{\sqrt 2}\right)\right) + x\,\varphi(x)\f) with the
 * standard normal density \f(\varphi\f); computed from \f(x\f) alone.
 */
float[...] gelu_gradient(x: float[...], dy: float[...]) =>
    dy * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 * x * x) * 0.3989422804014327);

/**
 * @brief Logistic sigmoid, \f(\sigma(x) = 1 / (1 + e^{-x})\f).
 */
public float[...] sigmoid(x: float[...]) => 1 / (1 + exp(-x));

/**
 * @brief Sigmoid Linear Unit (a.k.a. swish), \f(x\sigma(x)\f).
 */
public float[...] silu(x: float[...]) => x * sigmoid(x);

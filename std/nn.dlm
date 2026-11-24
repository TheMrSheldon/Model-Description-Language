/**
 * @file
 * @brief Basic neural network layers.
 */
from .arithmetic import *;
from .statistics import *;
from .autodiff import *;

/**
 * @brief Implements layer normalization as proposed by \cite layernorm.
 * \todo make epsilon configurable (currently 1e-12 as in BERT).
 */
public module LayerNorm[dim: uint] {
    public beta: float[dim]
    public gamma: float[dim]

    public float[..., dim] forward(x: float[..., dim]) {
        auto mu = mean(x)[... -> ... 1];
        auto x_c = x - mu;
        auto sigma = rsqrt(mean(x_c * x_c)[... -> ... 1] + 1e-12);
        return x_c * sigma * gamma + beta;
    }

    /**
     * @brief Closed-form gradient (used by the gradient pass instead of differentiating forward).
     * @return the cotangent of \p x , then the gradients of beta and gamma.
     */
    (float[..., dim], float[dim], float[dim]) gradient(x: float[..., dim], dy: float[..., dim]) {
        // Recomputed like in forward; common subexpression elimination shares these nodes with the forward pass.
        auto mu = mean(x)[... -> ... 1];
        auto x_c = x - mu;
        auto sigma = rsqrt(mean(x_c * x_c)[... -> ... 1] + 1e-12);
        auto xhat = x_c * sigma;
        auto g = dy * gamma;
        auto dx = sigma * (g - mean(g)[... -> ... 1] - xhat * mean(g * xhat)[... -> ... 1]);
        return dx, sum_like(dy, beta), sum_like(dy * xhat, gamma);
    }
}

/**
 * @brief Applies a learnable linear transformation with optional bias.
 * @details Given a vector \f(x\f) computes \f(xW^\top + b\f), or \f(xW^\top\f) without bias (`.bias=false`; then
 * the instance has no weight `b`).
 */
public module Linear[indim: uint, outdim: uint, bias: bool = true] {
    public W: float[outdim, indim]
    @conditional(bias)
    public b: float[outdim]

    public float[..., outdim] forward(x: float[..., indim]) => bias ? x @ W.T() + b : x @ W.T();
}

/**
 * @brief A lookup table of `num` vectors of size `dim`, e.g. token embeddings: `ids` selects the rows.
 * @details Out-of-range ids are clipped to the valid range. The gradient of the table adds the gradients of the
 * selected rows (repeated ids accumulate); rows that are not selected get a zero gradient.
 */
public module Embedding[num: uint, dim: uint] {
    public table: float[num, dim]

    public float[..., dim] forward(ids: int[...]) => table[ids];
}

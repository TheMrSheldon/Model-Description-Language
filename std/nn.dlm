/**
 * @file
 * @brief Basic neural network layers.
 */
import std.arithmetic;
import std.statistics;

/**
 * @brief Implements layer normalization as proposed by \cite layernorm.
 * \todo make epsilon configurable (currently 1e-12 as in BERT).
 */
module LayerNorm[dim: uint] {
    beta: float[dim]
    gamma: float[dim]

    float[..., dim] forward(x: float[..., dim]) {
        auto mu = mean(x)[... -> ... 1];
        auto x_c = x - mu;
        auto sigma = rsqrt(mean(x_c * x_c)[... -> ... 1] + 1e-12);
        return x_c * sigma * gamma + beta;
    }
}

/**
 * @brief Applies a learnable linear transformation with optional bias.
 * @details Given a vector \f(x\f) computes \f(xW^\top + b\f).
 * \todo honor `bias` once switch expressions are implemented (the bias is currently always added).
 */
module Linear[indim: uint, outdim: uint, bias: bool] {
    W: float[outdim, indim]
    b: float[outdim]

    float[..., outdim] forward(x: float[..., indim]) => x @ W.T() + b;
}

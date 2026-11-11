import std;

/**
 * @brief A single fully connected layer: computes x W^T + b.
 */
module FFN[indim: uint, outdim: uint] {
    W: float[outdim, indim]
    b: float[outdim]

    float[..., outdim] forward(x: float[..., indim]) => x @ W.T() + b;
}

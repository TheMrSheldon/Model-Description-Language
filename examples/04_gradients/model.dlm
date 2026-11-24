from std.nn import LayerNorm, Linear;
from std.losses import mse;

/**
 * @brief A small transformer-style block that exercises most gradient rules: linear layers (matmul, transposed weights,
 * bias broadcasting), multi-head attention (einops reshapes/permutations, batched matmuls, softmax with max/exp/sum),
 * layer norm (mean, rsqrt), and GELU (erf).
 */
module Block[dim: uint, hidden: uint, heads: uint] {
    q: Linear[dim, dim]
    k: Linear[dim, dim]
    v: Linear[dim, dim]
    norm: LayerNorm[dim]
    up: Linear[dim, hidden]
    down: Linear[hidden, dim]

    public float[..., dim] forward(x: float[..., dim]) {
        auto qh = q(x)[... (h d) -> h ... d | h=heads];
        auto kh = k(x)[... (h d) -> h ... d | h=heads];
        auto vh = v(x)[... (h d) -> h ... d | h=heads];
        auto attn = softmax(qh @ kh.T() / sqrt(dim / heads)) @ vh;
        auto h1 = norm(attn[h ... d -> ... (h d)] + x);
        return h1 + down(gelu(up(h1)));
    }

    /** An objective: compiled with `--grad loss`, it returns the loss and the gradients in one call. */
    public float loss(x: float[..., dim], target: float[..., dim]) => mse(forward(x), target);
}

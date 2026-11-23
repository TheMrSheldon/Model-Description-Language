import std;

/** The sizes of an MLP: layer i maps sizes[i] to sizes[i + 1]; biases[i] says whether it has a bias. */
struct MlpConf {
    sizes: array[uint; 4]
    biases: array[bool; 3]
}

/**
 * @brief A multi-layer perceptron with GELU between the layers: array hyperparameters (sizes, biases), a default
 * (`normalize`), named hyperparameters (`.bias=...`), a conditional member, and a compile-time conditional.
 */
module MLP[conf: MlpConf, normalize: bool = false] {
    layers: [Linear[conf.sizes[i], conf.sizes[i + 1], .bias=conf.biases[i]] for i in 0..3]
    @conditional(normalize)
    norm: LayerNorm[conf.sizes[3]]

    float[..., conf.sizes[3]] forward(x: float[..., conf.sizes[0]]) {
        auto h = x;
        for (i in 0..2) {
            h = gelu(layers[i](h));
        }
        h = layers[2](h);
        return normalize ? norm(h) : h;
    }
}

import std;

/** @brief A two-layer MLP whose weight matrices are quantized by the builds of model.yml (the model is unchanged). */
module MLP[dims: array[uint; 3]] {
    layers: [Linear[dims[i], dims[i + 1]] for i in 0..2]

    float[..., dims[2]] forward(x: float[..., dims[0]]) => layers[1](gelu(layers[0](x)));
}

// TODO: implement imports
// import std;

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
 * @brief Sum
 * @details
 * @ingroup Statistics
 * 
 * @param x 
 * @return 
 */
template<N: uint>
float[...] sum(x: float[..., N]) => __builtin_sum(x);

/**
 * @brief Maximum
 * @details
 * @ingroup Statistics
 * 
 * @param x 
 * @return 
 */
template<N: uint>
float[...] max(x: float[..., N]) => __builtin_max(x);

/**
 * @brief (Gauss) Error Function
 * @details
 * @ingroup Statistics
 * 
 * @param x 
 * @return 
 */
float[...] erf(x: float[...]) => __builtin_erf(x);

/**
 * @brief Computes the softmax function of the input vector.
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
 * @brief Recitifed Linear Unit
 * @details
 * \f[\text{ReLU}(x) := \max(0, x)\f]
 * 
 * @param x 
 * @return 
 */
// float[...] relu(x: float[...]) => max(x, 0);

/**
 * @brief Computes the reciprocal square root for each element in \p x.
 * @details \f[(x_i)_{1\leq i \leq n} \mapsto \left(\frac{1}{\sqrt{x_i}}\right)_{1\leq i \leq n}.\f]
 * @ingroup Arithmetic
 * 
 * @param x 
 * @return 
 */
float[...] rsqrt(x: float[...]) => __builtin_rsqrt(x);

/**
 * @brief Variance
 * @details
 * @ingroup Statistics
 * 
 * @param x 
 * @param dof 
 * @return 
 */
float[...] var(x: float[...], dof: uint = 1) => __builtin_var(x, dof);

/**
 * @brief Implements layer normalization as proposed by \cite layernorm.
 */
module LayerNorm[dim: uint] {
    beta: float[dim]
    gamma: float[dim]

    template<N: uint>
    float[..., dim, N] forward(x: float[..., dim, N]) {
        auto numerator = mean(input)[... -> ... 1];
        auto denominator = rsqrt(var(x[... a b -> ... b a]) + 1e-5);
        return (numerator * denominator) * gamma + beta;
    }
}

/**
 * @brief Applies a learnable linear transformation with optional bias.
 * @details Given a vector \f(x\f) computes \f(xW^\top + b\f).
 */
module Linear[indim: uint, outdim: uint, bias: bool /*= false*/] {
    W: float[outdim, indim]
    b: float[outdim]

    float[..., outdim] forward(x: float[..., indim]) =>
        bias?
            no => x @ W.T
            yes => x @ W.T + b;
}

struct TransformerConf {
    dims: struct {
        model: uint
        key: uint
        value: uint
        inner: uint
    }
    numEncoders: uint
    numAttnHeads: uint
}

module TransformerEncoder[conf: TransformerConf] {
    // Multi-Head Attention
    weightQuery: Linear[conf.dims.model, conf.numAttnHeads*conf.dims.key]
    weightKey: Linear[conf.dims.model, conf.numAttnHeads*conf.dims.key]
    weightValue: Linear[conf.dims.model, conf.numAttnHeads*conf.dims.value]
    weightOut: Linear[conf.numAttnHeads * conf.dims.value, conf.dims.model]
    weightIntermed: Linear[conf.dims.model, conf.dims.inner]
    mhaNorm: Linear[conf.dims.model]
    // FFN
    weightIntermedOut: Linear[conf.dims.inner, conf.dims.model]
    ffnNorm: LayerNorm[conf.dims.model]

    float[...] bertGELU(x: float[...]) =>
        x * 0.5 * (1 + erf(x / sqrt(2)));

    float[..., conf.numAttnHeads * conf.dims.value] sdpa(query: float[..., conf.numAttnHeads * conf.dims.key], key: float[..., conf.numAttnHeads * conf.dims.key], value: float[..., conf.numAttnHeads * conf.dims.value]) {
        auto facq = query[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto fack = key[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto facv = value[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto prod = facq @ fack.T;
        auto tmp = softmax(prod / sqrt(conf.dims.key), -1) @ facv;
        return tmp[h ... d -> ... (h d)];
    }
    
    float[..., conf.dims.model] mha(query: float[..., conf.numAttnHeads * conf.dims.key], key: float[..., conf.numAttnHeads * conf.dims.key], value: float[..., conf.numAttnHeads * conf.dims.value]) =>
        weightOut(sdpa(query, key, value));

    float[..., conf.dims.model] forward(x: float[..., conf.dims.model]) {
        auto attn = mhaNorm(mha(weightQuery(x), weightKey(x), weightValue(x)) + x);
        auto intermed = bertGELU(weightIntermed(attn));
        return ffnNorm(weightIntermedOut(intermed) + attn);
    }
}

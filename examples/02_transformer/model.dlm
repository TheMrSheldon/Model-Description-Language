import std;

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
    weightQuery: Linear[conf.dims.model, conf.numAttnHeads*conf.dims.key, true]
    weightKey: Linear[conf.dims.model, conf.numAttnHeads*conf.dims.key, true]
    weightValue: Linear[conf.dims.model, conf.numAttnHeads*conf.dims.value, true]
    weightOut: Linear[conf.numAttnHeads * conf.dims.value, conf.dims.model, true]
    weightIntermed: Linear[conf.dims.model, conf.dims.inner, true]
    mhaNorm: LayerNorm[conf.dims.model]
    // FFN
    weightIntermedOut: Linear[conf.dims.inner, conf.dims.model, true]
    ffnNorm: LayerNorm[conf.dims.model]

    float[..., conf.numAttnHeads * conf.dims.value] sdpa(query: float[..., conf.numAttnHeads * conf.dims.key], key: float[..., conf.numAttnHeads * conf.dims.key], value: float[..., conf.numAttnHeads * conf.dims.value]) {
        auto facq = query[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto fack = key[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto facv = value[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto prod = facq @ fack.T();
        auto tmp = softmax(prod / sqrt(conf.dims.key)/*, -1*/) @ facv;
        return tmp[h ... d -> ... (h d)];
    }
    
    float[..., conf.dims.model] mha(query: float[..., conf.numAttnHeads * conf.dims.key], key: float[..., conf.numAttnHeads * conf.dims.key], value: float[..., conf.numAttnHeads * conf.dims.value]) =>
        weightOut(sdpa(query, key, value));

    float[..., conf.dims.model] forward(x: float[..., conf.dims.model]) {
        auto attn = mhaNorm(mha(weightQuery(x), weightKey(x), weightValue(x)) + x);
        auto intermed = gelu(weightIntermed(attn));
        return ffnNorm(weightIntermedOut(intermed) + attn);
    }
}

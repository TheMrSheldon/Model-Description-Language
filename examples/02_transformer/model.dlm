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

/**
 * @brief A stack of `conf.numEncoders` encoder layers (e.g. the encoder of BERT).
 */
module BertEncoder[conf: TransformerConf] {
    layers: [TransformerEncoder[conf] for i in 0..conf.numEncoders]

    float[..., conf.dims.model] forward(x: float[..., conf.dims.model]) {
        auto h = x;
        for (i in 0..conf.numEncoders) {
            h = layers[i](h);
        }
        return h;
    }
}

/** @brief Sizes of BERT's embedding tables. */
struct EmbeddingConf {
    vocab: uint      // number of tokens
    positions: uint  // maximum sequence length
    types: uint      // number of segment types
}

/**
 * @brief BERT's input embeddings: the sum of the token, position, and segment embeddings, normalized.
 */
module BertEmbeddings[conf: TransformerConf, emb: EmbeddingConf] {
    word: Embedding[emb.vocab, conf.dims.model]
    position: Embedding[emb.positions, conf.dims.model]
    tokenType: Embedding[emb.types, conf.dims.model]
    norm: LayerNorm[conf.dims.model]

    float[..., conf.dims.model] forward(ids: int[...], types: int[...], positions: int[...]) =>
        norm(word(ids) + position(positions) + tokenType(types));
}

/**
 * @brief The BERT model (like Hugging Face's `BertModel` without pooler): token, segment, and position ids to the
 * last hidden states. With `--ellipsis-rank 2`, the ids are [batch, seq] and the output is [batch, seq, model].
 */
module Bert[conf: TransformerConf, emb: EmbeddingConf] {
    embeddings: BertEmbeddings[conf, emb]
    encoder: BertEncoder[conf]

    float[..., conf.dims.model] forward(ids: int[...], types: int[...], positions: int[...]) =>
        encoder(embeddings(ids, types, positions));
}

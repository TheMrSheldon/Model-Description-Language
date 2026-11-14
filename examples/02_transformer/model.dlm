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
    weightQuery: Linear[conf.dims.model, conf.numAttnHeads * conf.dims.key]
    weightKey: Linear[conf.dims.model, conf.numAttnHeads * conf.dims.key]
    weightValue: Linear[conf.dims.model, conf.numAttnHeads * conf.dims.value]
    weightOut: Linear[conf.numAttnHeads * conf.dims.value, conf.dims.model]
    weightIntermed: Linear[conf.dims.model, conf.dims.inner]
    mhaNorm: LayerNorm[conf.dims.model]
    // FFN
    weightIntermedOut: Linear[conf.dims.inner, conf.dims.model]
    ffnNorm: LayerNorm[conf.dims.model]

    float[..., conf.numAttnHeads * conf.dims.value] sdpa(query: float[..., conf.numAttnHeads * conf.dims.key],
            key: float[..., conf.numAttnHeads * conf.dims.key],
            value: float[..., conf.numAttnHeads * conf.dims.value]) {
        auto facq = query[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto fack = key[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto facv = value[... (h d) -> h ... d | h=conf.numAttnHeads];
        auto prod = facq @ fack.T();
        auto tmp = softmax(prod / sqrt(conf.dims.key) /*, -1*/) @ facv;
        return tmp[h ... d -> ... (h d)];
    }

    float[..., conf.dims.model] mha(query: float[..., conf.numAttnHeads * conf.dims.key],
            key: float[..., conf.numAttnHeads * conf.dims.key],
            value: float[..., conf.numAttnHeads * conf.dims.value]) =>
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

    /** @brief The output of every layer (e.g. for features from intermediate layers): [layers, ..., model]. */
    float[conf.numEncoders, ..., conf.dims.model] hiddenStates(x: float[..., conf.dims.model]) {
        float[conf.numEncoders, ..., conf.dims.model] hs;
        auto h = x;
        for (i in 0..conf.numEncoders) {
            h = layers[i](h);
            hs[i] = h;
        }
        return hs;
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
 * @brief BERT's pooler: the first position of each sequence ([CLS]), transformed (the input of classification heads).
 * For h: [batch, seq, model], the result is [batch, model].
 */
module BertPooler[conf: TransformerConf] {
    dense: Linear[conf.dims.model, conf.dims.model]

    float[..., conf.dims.model] forward(h: float[..., conf.dims.model]) => tanh(dense(h[..., 0, :]));
}

/**
 * @brief The BERT model (like Hugging Face's `BertModel`): token, segment, and position ids to the last hidden states
 * (`forward`) and to the pooled output (`pooled`). With `--ellipsis-rank 2`, the ids are [batch, seq], the hidden states
 * [batch, seq, model], and the pooled output [batch, model].
 */
module Bert[conf: TransformerConf, emb: EmbeddingConf] {
    embeddings: BertEmbeddings[conf, emb]
    encoder: BertEncoder[conf]
    pooler: BertPooler[conf]

    float[..., conf.dims.model] forward(ids: int[...], types: int[...], positions: int[...]) =>
        encoder(embeddings(ids, types, positions));

    float[..., conf.dims.model] pooled(ids: int[...], types: int[...], positions: int[...]) =>
        pooler(forward(ids, types, positions));
}

/**
 * @brief Sequence classification (like Hugging Face's `BertForSequenceClassification` in eval mode): the logits of
 * `classes` classes from the pooled output. For ids: [batch, seq], the logits are [batch, classes].
 */
module BertForSequenceClassification[conf: TransformerConf, emb: EmbeddingConf, classes: uint] {
    bert: Bert[conf, emb]
    classifier: Linear[conf.dims.model, classes]

    float[..., classes] logits(ids: int[...], types: int[...], positions: int[...]) =>
        classifier(bert.pooled(ids, types, positions));

    /**
     * The training objective: the mean cross-entropy of the logits and the class labels (one per sequence; compile
     * with `--ellipsis-rank labels=1` so that labels are [batch] while the ids are [batch, seq]).
     */
    float loss(ids: int[...], types: int[...], positions: int[...], labels: int[...]) =>
        cross_entropy(logits(ids, types, positions), labels);
}

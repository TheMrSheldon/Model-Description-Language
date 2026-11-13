import std;

/**
 * @brief A bag-of-words text classifier: embeds the tokens, averages them over the sequence, and classifies.
 * @details `loss` is the training objective (compile with `--grad loss`): it takes integer token ids and labels.
 */
module Classifier[vocab: uint, dim: uint, classes: uint, seq: uint] {
    embed: Embedding[vocab, dim]
    hidden: Linear[dim, dim]
    out: Linear[dim, classes]

    float[..., classes] forward(ids: int[..., seq]) {
        auto tokens = embed(ids);                        // [..., seq, dim]
        auto pooled = mean(tokens[... s d -> ... d s]);  // [..., dim]
        return out(tanh(hidden(pooled)));
    }

    float loss(ids: int[..., seq], labels: int[...]) => cross_entropy(forward(ids), labels);
}

/**
 * @file
 * @brief Objectives. A function that returns a scalar is differentiated directly (`mdlc --grad`): the compiled
 * function returns the objective and the gradients of the trainable weights.
 */
import std.arithmetic;
import std.statistics;

/**
 * @brief Mean squared error over all elements.
 * @ingroup Losses
 */
float mse(prediction: float[...], target: float[...]) => mean_all((prediction - target) * (prediction - target));

/**
 * @brief One-hot encoding of the integer `labels` with the shape of `like` (the number of classes is its last
 * dimension): 1.0 at the labels, 0.0 elsewhere.
 * @ingroup Losses
 */
float[...] one_hot_like(labels: int[...], like: float[...]) => __builtin_one_hot_like(labels, like);

/**
 * @brief `log(sum(exp(x)))` over the last dimension, computed stably (the maximum is subtracted first).
 * @ingroup Losses
 */
float[...] logsumexp(x: float[...]) {
    auto m = max(x);
    return m + log(sum(exp(x - m)));
}

/**
 * @brief Mean cross-entropy of the classification `logits` (unnormalized log-probabilities over the last dimension,
 * the classes) and the integer class `labels` (one per row of the logits, in 0..classes-1).
 * @ingroup Losses
 */
float cross_entropy(logits: float[...], labels: int[...]) =>
    mean_all(logsumexp(logits) - sum(one_hot_like(labels, logits) * logits));

/**
 * @file
 * @brief Objectives. A function that returns a scalar is differentiated directly (`mdlc --grad`): the compiled
 * function returns the objective and the gradients of the trainable weights.
 */
import std.statistics;

/**
 * @brief Mean squared error over all elements.
 * @ingroup Losses
 */
float mse(prediction: float[...], target: float[...]) => mean_all((prediction - target) * (prediction - target));

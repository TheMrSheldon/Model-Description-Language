/**
 * @file
 * @brief The prelude: the names that every file outside the standard library imports implicitly (with the lowest
 * precedence: local definitions and explicit imports take precedence).
 * @details The elementwise functions, the reductions, and the activations. The modules (`std.nn`), losses, and autodiff
 * helpers are imported explicitly, e.g. `from std.nn import Linear;`.
 */
public from .arithmetic import *;
public from .statistics import *;
public from .activations import *;

\page foundations_autodiff Automatic differentiation
\tableofcontents

Training a neural network means minimizing a loss function \f(L(\theta)\f) over the network's parameters
\f(\theta\f), usually with a variant of gradient descent:
\f[\theta \leftarrow \theta - \eta \nabla_\theta L(\theta).\f]
Each step needs the gradient \f(\nabla_\theta L\f) at the current parameters, for millions (or billions) of parameters.
Deriving it by hand is error-prone, and symbolic differentiation (as in a computer algebra system) produces expressions
that grow quickly. *Automatic differentiation* (autodiff) instead evaluates the derivative of a program at a point,
exactly (up to floating-point rounding) and at a small constant multiple of the program's own cost. This page follows
the survey by Baydin et al. \cite baydin2018autodiff; the standard reference is the book by Griewank and Walther
\cite griewank2008derivatives.

# The chain rule on a computation graph
Every program that computes a number from its inputs is a composition of elementary operations (additions,
multiplications, \f(\exp\f), matrix products, ...) whose derivatives are known. Consider
\f[f(x_1, x_2) = x_1 x_2 + \sin(x_1).\f]
Naming the intermediate values gives a sequence of elementary steps, which we can also draw as a graph:
\f{align}{
	v_1 &= x_1 x_2, \\
	v_2 &= \sin(x_1), \\
	y &= v_1 + v_2.
\f}
The chain rule tells us how to combine the derivatives of the steps into the derivative of the whole. There are two
orders in which to apply it.

# Forward mode
Forward mode carries, next to each value \f(v\f), its derivative \f(\dot v = \partial v / \partial x_i\f) with respect
to one chosen input. Every elementary operation is extended to propagate it, e.g. \f(\dot v_1 = \dot x_1 x_2 + x_1
\dot x_2\f) and \f(\dot v_2 = \cos(x_1)\dot x_1\f). An elegant way to see this are *dual numbers*
\f(a + b\varepsilon\f) with \f(\varepsilon^2 = 0\f): then
\f[(u + \dot u\varepsilon)(v + \dot v\varepsilon) = uv + (u\dot v + \dot u v)\varepsilon,\f]
so computing with dual numbers *is* forward mode, the product rule falls out of the arithmetic. Seeding
\f(\dot x_1 = 1, \dot x_2 = 0\f) gives \f(\partial y / \partial x_1\f) in a single pass. The catch: one pass yields the
derivative with respect to *one* input. For \f(f\colon \mathbb{R}^n \to \mathbb{R}\f), the gradient needs \f(n\f)
passes — hopeless for a network with millions of parameters.

# Reverse mode
Reverse mode turns this around. It first evaluates the program (the *forward pass*) and keeps the intermediate values;
then it propagates the *adjoints* \f(\bar v = \partial y / \partial v\f) from the output back to the inputs (the
*backward pass*). Starting with \f(\bar y = 1\f):
\f{align}{
	\bar v_1 &= \bar y, \qquad \bar v_2 = \bar y, \\
	\bar x_1 &= \bar v_1 x_2 + \bar v_2 \cos(x_1), \qquad \bar x_2 = \bar v_1 x_1.
\f}
A value that is used several times (here \f(x_1\f)) sums the adjoints of its uses. One backward pass yields the
derivatives with respect to *all* inputs, at a cost of a small constant times the forward pass. Since a loss is a
single number computed from many parameters, reverse mode is exactly what training needs; in neural networks it is
known as *backpropagation* \cite rumelhart1986backprop. The price is memory: the intermediate values of the forward
pass must be kept until the backward pass has used them.

# Vector-Jacobian products
For tensors, the elementary steps are functions \f(g\colon \mathbb{R}^n \to \mathbb{R}^m\f) with a Jacobian
\f(J_g \in \mathbb{R}^{m\times n}\f). The backward pass never forms this (often huge) matrix. It only needs the
*vector-Jacobian product* (VJP) \f(\bar x = \bar y^\top J_g\f), which can usually be computed directly. For a matrix
product \f(Y = XW\f), for example,
\f[\bar X = \bar Y W^\top, \qquad \bar W = X^\top \bar Y,\f]
which are again matrix products. An autodiff system therefore only needs one VJP rule per elementary operation, and the
chain rule does the rest.

# In MDL
MDL differentiates at compile time: `mdlc --grad forward` generates, next to `forward`, a function that computes the
forward pass and the gradients with respect to the inputs and the trainable weights, by reverse mode on the computation
graph. Because the whole graph is known to the compiler, the backward pass is optimized together with the forward pass
(e.g. shared subexpressions are computed once), and only the intermediate values that the backward pass actually needs
are kept. A module can also provide a closed-form `gradient` method instead (e.g. `LayerNorm` in \ref std_nn), which
the gradient pass then uses rather than differentiating `forward`. The helpers for writing such gradients are in
\ref std_autodiff, and \ref example_gradients shows how gradients are compiled and called.

\todo Link to the technical description of the gradient pass once it is written.

<div class="page-sequence">Previous: \ref foundations_tensors</div>

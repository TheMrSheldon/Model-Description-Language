\page foundations_tensors Tensors and shapes
\tableofcontents

# What is a tensor?
In deep learning, a *tensor* is simply a multi-dimensional array of numbers: a scalar has zero dimensions (also called
*axes*), a vector one, a matrix two, and a batch of images (batch, channels, height, width) four. The list of the
sizes of all axes is the tensor's *shape*; a tensor of shape \f([2, 3]\f) is a \f(2\times 3\f) matrix. (In mathematics
and physics, tensors are multilinear maps with a transformation behavior; deep learning only borrows the name.)

Almost every operation in a neural network is an operation on whole tensors rather than on single numbers. Writing a
layer as \f(y = xW^\top + b\f) instead of as a loop over the elements is not only shorter: it lets the compiler (or
library) choose an efficient implementation, e.g. a tuned matrix multiplication.

# Shapes as types
Most mistakes in deep learning code are shape mistakes: an axis that was transposed, a dimension that does not match,
a reduction over the wrong axis. Libraries like PyTorch find them at runtime, when the offending line is executed. MDL
treats shapes as part of the type instead:

<div class="mdl">

```
float[..., outdim] forward(x: float[..., indim]) => x @ W.T() + b;
```
</div>

Here, `x` may have any number of leading axes (`...`, e.g. batch and sequence), but its last axis must have the size
`indim`; the result keeps the leading axes and ends in `outdim`. These sizes are symbolic: the compiler checks the
function once for *all* values of `indim` and `outdim`, and an instantiation with mismatching sizes is a compile
error, not a crash during training.

# Broadcasting
Adding the bias \f(b \in \mathbb{R}^{n}\f) to \f(xW^\top \in \mathbb{R}^{B\times n}\f) is mathematically sloppy but
convenient: the vector is implicitly repeated along the batch axis. This is called *broadcasting* and follows the NumPy
rules: shapes are aligned at their last axis, and an axis of size 1 (or a missing one) is stretched to the size of the
other operand.

\todo Reductions and `keepdims`, einops rearrangements \cite rogozhnikov2022einops (see \ref spec_einops), memory
layout and strides.

<div class="page-sequence">Next: \ref foundations_autodiff</div>

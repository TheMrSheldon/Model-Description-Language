\page example_gradients Gradients
\tableofcontents

Training needs the gradients of a loss with respect to the weights (see \ref foundations_autodiff). MDL computes them at
compile time: the compiler generates the backward pass next to the forward pass. This example compiles the gradients of
a small transformer block and checks them against finite differences. The complete project is in `examples/04_gradients`.

# The model
The block combines multi-head attention \cite vaswani2017attention, layer normalization \cite layernorm and a
feed-forward network with GELU \cite hendrycks2016gelu, using the modules of \ref std_nn "std.nn":

<div class="mdl">

\snippet 04_gradients/model.dlm forward
</div>

\todo A short paragraph explaining the attention lines (the einops reshapes into heads, the scaled dot product, and the
merge of the heads), or a link to a foundations page on attention once it exists.

Besides `forward`, the module has a second public function, an *objective*: a function that returns a scalar loss.

<div class="mdl">

\snippet 04_gradients/model.dlm loss
</div>

# Compiling the gradients
Gradients are a build option, so the model itself contains no backward code. `--grad forward` adds the backward pass of
`forward`, `--grad loss` the gradients of the objective, and `--input-grads` also the gradient with respect to the input:

<div class="lang-cmake">

\snippet 04_gradients/CMakeLists.txt build
</div>

For `forward`, the compiled model then has two functions:

- `forward(x, ...)` computes the output and saves what the backward pass needs (the *residuals*) in a context buffer;
- `forward_backward(d_output, ...)` takes the gradient of the output and computes the gradients of the weights and of
  the input. The gradients are *accumulated*: a second call adds to them, as needed for gradient accumulation over
  several batches.

For an objective, a single call returns the loss and its gradients, since the gradient of a scalar loss needs no input
from the caller.

\todo A sentence on when to use which: the split `forward`/`forward_backward` when the gradient of the output comes
from outside (e.g. a loss computed in PyTorch), an objective when the whole loss is part of the model.

# Calling them
With the split functions, the program allocates the context, runs the forward pass, and then the backward pass with
the gradient of the output `dy`:

<div class="lang-cpp">

\snippet 04_gradients/main.cpp backward
</div>

The objective returns the loss and its gradients in one call:

<div class="lang-cpp">

\snippet 04_gradients/main.cpp loss
</div>

The program compares both with central finite differences, i.e., with
\f(\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \varepsilon e_i) - L(\theta - \varepsilon e_i)}{2\varepsilon}\f).

# The fused variant
With `--grad-fused`, the forward and the backward pass are compiled into a single function `forward(x, d_output)` that
returns the output and the gradients at once. It needs no context buffer, but the gradient of the output must be known
before the call:

<div class="lang-cmake">

\snippet 04_gradients/CMakeLists.txt fused
</div>

\todo A sentence on the trade-off (memory for the residuals vs. needing `d_output` in advance), and on gradient
checkpointing (`checkpoint:` in \ref build_config) for large models.

<div class="page-sequence">Previous: \ref example_ffn</div>

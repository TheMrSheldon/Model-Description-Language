\page example_ffn A first model: a feed-forward layer
\tableofcontents

This example is the smallest complete MDL project: a model with one layer, the values of its hyperparameters, and a C++
program that calls the compiled model. The complete project is in `examples/01_ffn`.

# The model
A fully connected layer computes \f(y = xW^\top + b\f) for an input \f(x\f) with `indim` features, a weight matrix
\f(W\f) and a bias \f(b\f). In MDL, it is a module whose hyperparameters are the two sizes:

<div class="mdl">

\include 01_ffn/model.dlm
</div>

- The hyperparameters in `[...]` are known when the model is compiled; the shapes of the weights and of the input are
  written in terms of them.
- `W` and `b` are the module's weights. The compiled model does not own them: the caller passes them in.
- `forward` is `public`, so it is compiled into a function that the program can call. Its input `float[..., indim]` may
  have any number of leading axes (e.g. a batch); the result keeps them (see \ref foundations_tensors).

# The configuration
The values of the hyperparameters are not part of the model. They are set when compiling, in a configuration file:

<div class="lang-yaml">

\include 01_ffn/model.yml
</div>

The same model can thus be compiled for different sizes without changing its source (see \ref build_config).

# Building
The CMake function `mdl_add_model` compiles the model into a library and generates the header `model.hpp` with a C++
interface to it (see \ref using_cmake):

<div class="lang-cmake">

\include 01_ffn/CMakeLists.txt
</div>

# Calling the model
The program creates the input and the weights as tensors of the runtime library:

<div class="lang-cpp">

\snippet 01_ffn/main.cpp inputs
</div>

The generated header has a struct `Weights` with one member per weight, and one function per `public` function of the
module. The output is passed in as well, so the compiled code allocates nothing:

<div class="lang-cpp">

\snippet 01_ffn/main.cpp call
</div>

The rest of `main.cpp` compares the result with a reference implementation.

\todo One or two sentences on what to try next (e.g. change `outdim` in model.yml and rebuild, or replace the layer
by `Linear` from \ref std_nn), and a pointer to \ref example_gradients.

<div class="page-sequence">Next: \ref example_gradients</div>

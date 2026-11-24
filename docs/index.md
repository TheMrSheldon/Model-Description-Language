# Model Description Language {#mainpage}

MDL is a language for describing neural networks, and `mdlc` is its compiler: it turns a model into a linkable object
file plus bindings (a C++ header, Python modules). The model is written once, at the level of the mathematics,
while shapes, gradients, kernels and the calling convention are the compiler's business:

<div class="mdl">

```
module FFN[indim: uint, outdim: uint] {
    W: float[outdim, indim]
    b: float[outdim]

    public float[..., outdim] forward(x: float[..., indim]) => gelu(x @ W.T() + b);
}
```
</div>

# Getting started
1. \ref installation "Install" the tooling, or build MDL from source.
2. Write and compile \ref example_ffn "a first model", and call it from C++.
3. Compile \ref example_gradients "its gradients" to train it.

The \ref std_api "standard library" has the common layers and functions, and the \ref glossary explains the terms.

# The documentation
The documentation is split into three parts that build on each other but can be read independently:

- \ref user_docs is for those who *use* MDL: installing it, compiling and calling models, the editor tooling,
  examples, and the API of the standard library.
- \ref foundations explains the deep learning background that MDL builds on (tensors, automatic
  differentiation, layers, training), so these pages also work as a learning resource on their own.
- \ref dev_docs is for those who want to know how MDL works or want to work *on* it: the specifications
  (the language, einops expressions, kernel packages) and technical descriptions of the compiler.

The C++ interface of the compiler and the runtime is listed under *C++ API* in the navigation.

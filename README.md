<h1 align="center">Model Description Language</h1>

<p align="center">
   <img width=120px src="./res/icon-solo.png">
</p>

---

MDL is a language for describing neural networks and a compiler (`mdlc`) that turns a model into a linkable object
file plus bindings (C++ header, Python).

```dlm
import std;

module FFN[indim: uint, outdim: uint] {
    W: float[outdim, indim]
    b: float[outdim]

    float[..., outdim] forward(x: float[..., indim]) => x @ W.T() + b;
}
```

## Compile
```bash
mdlc model.dlm --arg-vals model.yml -o model.o --binding cpp=model.hpp
```
- `--arg-vals` sets the hyperparameters (here `FFN: {indim: 8, outdim: 3}`).
- `--binding TYPE[=PATH]` generates bindings: `cpp` (C++ header), `python`, `python-torch`, `python-tensorflow`, `json` (metadata).
- `--kernels packages|none` selects whether optimized kernels of kernel packages (e.g. oneDNN, see `docs/kernels.md`) are used, or TVM-generated kernels only.
- `--ellipsis-rank N` sets how many dimensions `...` stands for in compiled signatures (e.g. 2 for `[batch, seq, ...]`).
  `--ellipsis-rank NAME=K` gives the input `NAME` only the leading `K` of them (e.g. `labels=1`: labels `[batch]` next
  to token ids `[batch, seq]`). Outputs get their shapes from shape inference (e.g. pooled `[batch, dim]`).
- `--float f32|bf16|f16` sets the type that `float` stands for (see [Precision](#precision-and-quantization)).
- `-i DIR` adds import search paths; the standard library (`std/`) is always available.

The language is described in [docs/language-spec.md](docs/language-spec.md).

### Build configuration
The argument-values file can also hold the build settings, with named builds for e.g. inference and training; options
on the command line take precedence (see [docs/build-config.md](docs/build-config.md)):
```yaml
args:
  FFN: { indim: 8, outdim: 3 }
build:
  ellipsis-rank: 2
builds:
  train: { grad: [forward], trainable: ["W"] }
```
```bash
mdlc model.dlm -a model.yml --build train -o model_train.o --binding cpp=model_train.hpp
```

The object file exports one function per member function (`__tvm_ffi_<Module>__<function>`, TVM FFI calling
convention: inputs, weights, preallocated outputs). Link it against `mdl_runtime` (TVM runtime) and the libraries of
the kernel packages it uses (`mdlc --kernel-report`, `mdl-kernels link-args`; e.g. `mdl_onednn` for oneDNN).

## Gradients
```bash
mdlc model.dlm -a model.yml -o model_train.o --binding cpp=model_train.hpp --grad forward --trainable 'layers.**' --trainable '!layers.0.**'
```
`--grad f` makes a *gradient build*: the function `f` is replaced by its gradient (a transformation of the
computegraph; the backend only sees ordinary functions). Compile without `--grad` for inference. `--trainable` takes
gitignore-style patterns over weight paths (all weights by default); `--input-grads` adds gradients with respect to the
inputs. Gradients are added to the caller's buffers (accumulation; zero them before the first step, or use
`--grad-overwrite`).

**The objective is part of the model.** If `f` returns a scalar, it is differentiated directly:
`loss(x, target) -> loss, d_<weight>...` in one call.
```dlm
float loss(x: float[..., 768], target: float[..., 768]) => mse(forward(x), target);
float loss(ids: int[..., seq], labels: int[...]) => cross_entropy(forward(ids), labels);  // integer inputs
```
`int` tensors (token ids, labels) are passed as int32 (`mdl::runtime::IntTensor`); they get no gradients. `std` provides
`Embedding` (a lookup table; its gradient adds the selected rows), `cross_entropy`, `one_hot_like`, `mse`, `logsumexp`,
and `sum_flat`/`mean_flat`. See [examples/05_classification](examples/05_classification/).

**The caller computes the objective** (e.g. the trainer) and provides the cotangent `dL/d_output`:
- `forward(x) -> output` also saves the values the backward pass needs in a context buffer `ctx`,
- `forward_backward(d_output) -> d_<weight>...` reads them and computes the gradients.

`ctx` is an opaque byte buffer whose layout depends on the runtime dimensions; `forward_ctx_size(dims of ...)` returns
the bytes needed (`forward_ctx_alloc`/`forward_ctx_free` allocate it, 64-byte aligned). The size grows with the
dimensions, so a buffer allocated once for the largest batch/sequence is reused for all steps; `forward_backward` must
be called with the same dimensions as the `forward` that wrote it.
```cpp
auto ctx = Model::forward_ctx_alloc(maxBatch, maxSeq);
Model::forward(x, weights, y, ctx);          // then compute dL/dy
Model::forward_backward(dy, weights, d_W1, d_b1, /* ... */ ctx);
Model::forward_ctx_free(ctx);
```
Only values that depend on the inputs are saved; values computed from weights alone, reshapes, and elementwise
functions of other saved values are recomputed. With `--grad-fused`, `forward(x, d_output) -> output, d_<weight>...`
does both in one call instead (nothing is saved, but `d_output` must be known before the call).

`--binding json` writes metadata for runtimes that load the library dynamically (e.g. a trainer): the weights with
shapes and whether they are trainable, and every function with its C symbol, arguments, which outputs are gradients
of which weight, and its context buffer.

A module can provide its own gradient as an ordinary member function named `gradient` (for `forward`; other member
functions use `<name>_gradient`). It takes the inputs followed by one cotangent per output and returns one cotangent
per input followed by one gradient per weight (in declaration order). If it is not defined, the gradient is derived
automatically. Intermediates are simply recomputed: identical subexpressions are shared with the forward pass.
```dlm
module Scale[dim: uint] {
    s: float[dim]
    float[..., dim] forward(x: float[..., dim]) => x * s;
    (float[..., dim], float[dim]) gradient(x: float[..., dim], dy: float[..., dim]) {
        return dy * s, sum_like(dy * x, s);
    }
}
```
`std.autodiff` provides `sum_like(x, like)` (sums over broadcast dimensions), `broadcast_like` and `nograd(x)`.

## Masked tensors (padding)
```bash
mdlc model.dlm -a model.yml --ellipsis-rank 2 --mask forward -o model.o --binding cpp=model.hpp
```
`--mask f` makes `f` a masked function with the semantics of `torch.masked`: it takes an additional input
`mask: bool[...]` (true: valid; `mdl::runtime::MaskTensor`), and the elements at masked positions of the last
dimension of `...` (e.g. the padding of shorter sequences in `[batch, seq]`) do not influence the valid results. The
model itself is unchanged: the compiler inserts the masking where masked elements would mix with valid ones
(reductions such as softmax, `sum`/`mean`, and contractions over the sequence; e.g. attention ignores padded keys), and
the outputs are zero at masked positions. Functions without `--mask` have no masking and no cost. Masking composes with
`--grad` (the gradient of the masked computation; padding, even NaN, cannot reach the weight gradients). The full BERT
matches Hugging Face's `BertModel` with `attention_mask` on the valid positions.

In the language, `masked float[...]` declares a masked tensor. A function with masked parameters is compiled as a masked
function, and a masked argument selects a `masked` overload, also in nested calls (functions are elaborated again for
masked arguments), where `x.data` (zeros at masked elements) and `x.mask` (1.0 valid, 0.0 masked) are available:
```dlm
float[...] average(x: float[...]) => mean(x);
float[...] average(x: masked float[...]) => mean(x.data) / mean(x.mask);
```
See [examples/06_masking](examples/06_masking/).

Tensors are indexed like in NumPy: `h[..., 0, :]` is the first position of each sequence of `h: [..., seq, dim]`
(integers select and remove an axis, `:` keeps one, `...` stands for the others). The BERT example
([examples/02_transformer](examples/02_transformer/)) contains the full model with pooler, sequence classification,
and its training objective, all checked against Hugging Face (with padding).

## Precision and quantization
Models are written with `float`; `--float bf16` (or `f16`) compiles the same model for 16-bit activations and weights,
while explicit `float32` tensors stay float32. Conversions between float types are explicit (`float32(x)`); implicit
ones can be allowed as warnings (`-Wfloat-widening`, `-Wfloat-narrowing`).

`--quantize` stores weight matrices as int8 or int4 (grouped scales, optionally zero points); the model is unchanged:
```bash
mdlc model.dlm -a model.yml --quantize 'int4:128=encoder.**' -o model.o --binding cpp=model.hpp
```
A weight `W` becomes `W.qweight`, `W.scales` (and `W.zeros`) in the ABI (the GPTQ/AWQ layout); quantized weights are
frozen. With the oneDNN kernels, linear layers read the compressed weights directly, which speeds up calls with few
tokens (BERT-base with 8 tokens: 10.2 → 5.4 ms in float32). `mdl::runtime::loadWeights` can quantize a float
checkpoint while loading. See [docs/build-config.md](docs/build-config.md#quantization) and
[examples/09_quantization](examples/09_quantization/).

## Editor tooling
`mdl-format` formats `.dlm` files (like clang-format; style in `.mdl-format`), and `mdls` is a language server with
diagnostics, hover (types and documentation), completion, go to definition, semantic highlighting (einops axes
included), and formatting. Both build without the compiler's dependencies (`cmake --preset conf-tooling`), see
[docs/tooling.md](docs/tooling.md).

## CMake
```cmake
include(MDL)
mdl_add_model(ffn SOURCE model.dlm CONFIG model.yml)   # model.o + model.hpp
target_link_libraries(myexe PRIVATE ffn)
```
See [docs/using-cmake.md](docs/using-cmake.md) and the [examples](examples/).

## Use from C++
```cpp
#include "model.hpp"
#include <mdl/runtime.hpp>

mdl::runtime::Tensor x({2, 8}), W({3, 8}), b({3}), out({2, 3});
mdl::models::FFN::Weights weights;
weights.W = W;
weights.b = b;
mdl::models::FFN::forward(x, weights, out);

// The hyperparameters the model was compiled with, as a constexpr struct:
static_assert(mdl::models::FFN::hyperparameters.outdim == 3);
```

## Roadmap
Open work is listed in [TODO.md](TODO.md).

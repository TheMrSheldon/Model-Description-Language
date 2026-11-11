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
- `--binding TYPE[=PATH]` generates bindings: `cpp` (C++ header), `python`, `python-torch`, `python-tensorflow`.
- `--kernels onednn|tvm` selects the kernels for recognized operations (oneDNN via `mdl_kernels`, or TVM-generated only).
- `-i DIR` adds import search paths; the standard library (`std/`) is always available.

The object file exports one function per member function (`__tvm_ffi_<Module>__<function>`, TVM FFI calling
convention: inputs, weights, preallocated outputs). Link it against `mdl_runtime` (TVM runtime) and, with the oneDNN
kernels, `mdl_kernels`.

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
```

## Roadmap
See the issues/plan: performance work on the TVM backend, loops (`for` with loop-carried variables), gradients as a
computegraph transform (trainable weights selected by gitignore-style patterns), ragged tensors, and low-bit
quantization.

# TODO:
- Torch/AOTInductor backend(?)

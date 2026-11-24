# Using MDL inside a CMake Project {#using_cmake}

`cmake/MDL.cmake` provides `mdl_add_model()`, which compiles a model into an object file and generates a C++ header
for it:

<div class="lang-cmake">

```cmake
include(MDL)  # with cmake/ on CMAKE_MODULE_PATH

# Creates the library target `transformer` from model.dlm (hyperparameters in model.yml).
mdl_add_model(transformer SOURCE model.dlm CONFIG model.yml)

add_executable(myexe main.cpp)
target_link_libraries(myexe PRIVATE transformer)  # adds the include path of model.hpp and the runtime libraries
```
</div>

<div class="lang-cpp">

```cpp
#include "model.hpp"  // generated: mdl::models::<Module>::Weights and one function per exported member function

mdl::models::TransformerEncoder::Weights weights;
*weights.find("weightQuery.W") = &myTensor;  // or weights.weightQuery.W = &myTensor;
mdl::models::TransformerEncoder::forward(&x, weights, &out);
```
</div>

Options:

| Option | Meaning |
|---|---|
| `SOURCE <file>` | The model (required). |
| `CONFIG <file>` | Hyperparameter values (`mdlc --arg-vals`). |
| `SHARED` | Build a shared library (e.g., to load it from Python via `tvm_ffi.load_module`). |
| `KERNELS packages\|none` | Use the kernel packages, including oneDNN (default), or TVM-generated kernels only. |
| `KERNEL_PACKAGES <name>...` | Kernel packages the model may use; they are linked (see [kernel packages](../dev/spec/kernel-packages.md)). |
| `IMPORT_DIRS <dir>...` | Additional search paths for `import`. |
| `BINDINGS <type>...` | Additional bindings (e.g. `python-torch`), generated next to the object file. |
| `OPTIONS <option>...` | Additional `mdlc` options. |

Imported files (including the standard library) are tracked through a depfile, so the model is recompiled when any of
them changes.

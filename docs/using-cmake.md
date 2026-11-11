# Using MDL inside a CMake Project

`cmake/MDL.cmake` provides `mdl_add_model()`, which compiles a model into an object file and generates a C++ header
for it:

```cmake
include(MDL)  # with cmake/ on CMAKE_MODULE_PATH

# Creates the library target `transformer` from model.dlm (hyperparameters in model.yml).
mdl_add_model(transformer SOURCE model.dlm CONFIG model.yml)

add_executable(myexe main.cpp)
target_link_libraries(myexe PRIVATE transformer)  # adds the include path of model.hpp and the runtime libraries
```

```cpp
#include "model.hpp"  // generated: mdl::models::<Module>::Weights and one function per exported member function

mdl::models::TransformerEncoder::Weights weights;
*weights.find("weightQuery.W") = &myTensor;  // or weights.weightQuery.W = &myTensor;
mdl::models::TransformerEncoder::forward(&x, weights, &out);
```

Options:

| Option | Meaning |
|---|---|
| `SOURCE <file>` | The model (required). |
| `CONFIG <file>` | Hyperparameter values (`mdlc --arg-vals`). |
| `SHARED` | Build a shared library (e.g., to load it from Python via `tvm_ffi.load_module`). |
| `KERNELS onednn\|tvm` | Use the oneDNN kernels of `mdl_kernels` (default when available) or TVM-generated kernels only. |
| `IMPORT_DIRS <dir>...` | Additional search paths for `import`. |
| `BINDINGS <type>...` | Additional bindings (e.g. `python-torch`), generated next to the object file. |
| `OPTIONS <option>...` | Additional `mdlc` options. |

Imported files (including the standard library) are tracked through a depfile, so the model is recompiled when any of
them changes.

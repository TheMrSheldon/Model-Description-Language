# TVM's Python, ported to C++

mdlc does not depend on Python, but some parts of TVM that the CUDA target needs exist only in TVM's Python package.
This directory translates them to C++, file by file, mirroring `python/tvm/`:

- **DLight** (`s_tir/dlight/`): the default GPU schedules.
- **CUDA tensor intrinsics** (`s_tir/tensor_intrin/cuda.cpp`): the WMMA intrinsics.
- **CUDA C code generation callbacks** (`backend/cuda/codegen/`, `support/nvcc.cpp`): the CUDA header, the intrinsic
  codegen registry, and compiling the generated CUDA with NVRTC.

The port is pinned to TVM `7b5acb32bd271342a24130f4ef439ff6c2e52c6c` (v0.27.0.post1, the TVM version of
`src/CMakeLists.txt`). All sources are at
<https://github.com/apache/tvm/tree/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm>.

## Conventions

The goal is a port that can be diffed against TVM and updated when TVM changes:

- **Files:** one C++ file per Python file, at the same relative path (`s_tir/dlight/gpu/matmul.py` →
  `s_tir/dlight/gpu/matmul.cpp`, with a header for declarations). The **first line** of every translated file is
  `// Translated from <URL of the Python file at the pinned commit>`.
- **Names:** TVM's names stay as they are: `normalize_prim_func`, `try_inline`, `Matmul::apply`, `get_wmma_load_intrin`,
  local variable names. Only the syntax is C++.
- **Structure:** statements keep TVM's order. Python constructs are translated mechanically:
  - `None` returns become `std::nullopt`.
  - `try/except` around schedule primitives becomes `catch (const std::exception&)`.
  - `s_tir.Schedule(func)` becomes `schedule_of(func)` (a traced schedule of `{"main": func}`).
  - Python's `int | Expr` domains are `PrimExpr`, with `as_int` where TVM checks for an `int`.
- **TVMScript:** functions written as `@T.prim_func` are built with TVM's C++ IR builder, which TVMScript itself drives
  (`tvm/tirx/script/builder/ir.h`): one builder call per TVMScript statement.
- **Generated text:** `backend/cuda/codegen/header.cpp` copies the header text of `header.py` verbatim (each Python string
  literal becomes a C++ raw string, under the same conditions).
- **TVM's own C++:** where the Python calls a C++ function registered with TVM, the port calls that C++ function
  directly. If the function is not in TVM's public headers, it is declared `extern` next to its use, with the TVM source
  file that defines it (TVM exports these symbols: `HIDE_PRIVATE_SYMBOLS OFF`).
- **Registration:** Python registers functions with TVM when a module is imported (`register_global_func`,
  `TensorIntrin.register`). Here `globals.cpp` registers them explicitly (`register_cuda_globals`). It is not a
  translation, and neither is `tvm.hpp` (the TVM names the port uses).
- **Independence:** this directory is separate from the rest of mdlc and depends on nothing of it: its own CMake target
  `mdl_tvm_ported` only uses TVM and NVRTC, and mdlc includes it like an external library (`<tvm_ported/...>`). Should
  TVM provide these parts in C++, the directory can be replaced by TVM's. The CUDA target
  (`src/mdl/backends/tvm/targets/cuda.cpp`) uses it like TVM's pipelines do:
  `ApplyDefaultSchedule(Matmul, Reduction, GeneralReduction, Transpose, Fallback)`.

## Status

| TVM file | Port | Notes |
|---|---|---|
| `s_tir/dlight/base/schedule_rule.py` | `s_tir/dlight/base/schedule_rule.hpp` | `apply` returns one schedule (non-tunable). `from_callable` is not ported (no callers). |
| `s_tir/dlight/base/transform.py` | `s_tir/dlight/base/transform.{hpp,cpp}` | `ApplyDefaultSchedule` |
| `s_tir/dlight/base/utils.py` | `s_tir/dlight/base/utils.{hpp,cpp}` | Adds `schedule_of`, the C++ counterpart of `s_tir.Schedule(func)`. |
| `s_tir/dlight/base/common_schedules.py` | `s_tir/dlight/base/common_schedules.{hpp,cpp}` | |
| `s_tir/dlight/analysis/common_analysis.py` | `s_tir/dlight/analysis/common_analysis.{hpp,cpp}` | `BufferInfo`, `is_layout_transform` and `is_data_pad` are not ported: only unported rules use them. |
| `s_tir/dlight/gpu/__init__.py`, `gpu/base.py` | `s_tir/dlight/gpu/gpu.hpp`, `gpu/base.hpp` | |
| `s_tir/dlight/gpu/fallback.py` | `s_tir/dlight/gpu/fallback.cpp` | |
| `s_tir/dlight/gpu/general_reduction.py` | `s_tir/dlight/gpu/general_reduction.cpp` | |
| `s_tir/dlight/gpu/reduction.py` | `s_tir/dlight/gpu/reduction.cpp` | |
| `s_tir/dlight/gpu/transpose.py` | `s_tir/dlight/gpu/transpose.cpp` | |
| `s_tir/dlight/gpu/matmul.py` | `s_tir/dlight/gpu/matmul.cpp` | `Matmul` and `MatmulTensorization` (float16 WMMA tensor cores). Not ported: `MatmulInt8Tensorization`, `MetalMatmul`, and the OpenCL-only `sch_outer_reduction`. |
| `s_tir/dlight/gpu/gemv.py`, `low_batch_gemv.py`, `rmsnorm.py`, `analysis/gemv.py` | — | Not ported yet: they serve matrix-vector products (LLM decoding). |
| `s_tir/tensor_intrin/cuda.py` | `s_tir/tensor_intrin/cuda.{hpp,cpp}` | The WMMA intrinsics and `get_wmma_intrin_group`. Not ported: the ldmatrix/MMA intrinsics. |
| `backend/cuda/codegen/header.py` | `backend/cuda/codegen/header.{hpp,cpp}` | Complete. |
| `backend/cuda/codegen/registry.py` | `backend/cuda/codegen/registry.{hpp,cpp}` | The registry is empty: the codegens that TVM registers in `backend/cuda/cpp/*.py` and `ptx/*.py` (TIRX device intrinsics: TMA, async copies, ...) are not ported. |
| `support/nvcc.py` | `support/nvcc.{hpp,cpp}` | The NVRTC path of `compile_cuda` (cubin, ptx) and `tvm_callback_cuda_compile`. Not ported: the nvcc subprocess path, NVSHMEM, and the version/feature queries. |

## Updating to a new TVM version

1. Diff the Python sources between the pinned commit and the new one:
   `git diff OLD NEW -- python/tvm/s_tir/dlight python/tvm/s_tir/tensor_intrin/cuda.py python/tvm/backend/cuda/codegen python/tvm/support/nvcc.py`.
2. Apply the same changes to the corresponding C++ files.
3. Update the pinned commit in this README and in every file's first line.

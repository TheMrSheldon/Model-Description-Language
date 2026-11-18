# DLight in C++

A C++ port of TVM's DLight, the default (untuned) schedules for GPUs. DLight is Python-only in TVM. mdlc does not
depend on Python, so the rules are translated here, file by file.

The port is pinned to TVM `7b5acb32bd271342a24130f4ef439ff6c2e52c6c` (v0.27.0.post1, the TVM version of
`src/CMakeLists.txt`). All sources are at
<https://github.com/apache/tvm/tree/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight>.

## Conventions

The goal is a port that can be diffed against TVM and updated when TVM changes:

- **Files:** one C++ file per Python file, at the same relative path (`gpu/matmul.py` → `gpu/matmul.cpp`). Every file
  names its source URL in its header.
- **Names:** TVM's names stay as they are: `normalize_prim_func`, `try_inline`, `Matmul::apply`, `get_index_map`,
  local variable names. Only the syntax is C++.
- **Structure:** statements keep TVM's order. Python constructs are translated mechanically:
  - `None` returns become `std::nullopt`.
  - `try/except` around schedule primitives becomes `catch (const std::exception&)`.
  - `s_tir.Schedule(func)` becomes `schedule_of(func)` (a traced schedule of `{"main": func}`).
  - Python's `int | Expr` domains are `PrimExpr`, with `as_int` where TVM checks for an `int`.
- **TVM's own C++ analyses:** where the Python calls a C++ function registered with TVM (`NormalizePrimFunc`), the
  port calls that C++ function directly. If the function is not in TVM's public headers, it is declared `extern` next
  to its use, with the TVM source file that defines it (TVM exports these symbols: `HIDE_PRIVATE_SYMBOLS OFF`).
- **Independence:** nothing here depends on the rest of mdlc. The library `mdl_dlight` only uses TVM. The CUDA target
  (`../targets/cuda.cpp`) applies it like TVM's pipelines do:
  `ApplyDefaultSchedule(Matmul, Reduction, GeneralReduction, Transpose, Fallback)`.

## Status

| TVM file | Port | Notes |
|---|---|---|
| `base/schedule_rule.py` | `base/schedule_rule.hpp` | `apply` returns one schedule (non-tunable). `from_callable` is not ported (no callers). |
| `base/transform.py` | `base/transform.{hpp,cpp}` | `ApplyDefaultSchedule` |
| `base/utils.py` | `base/utils.{hpp,cpp}` | Adds `schedule_of`, the C++ counterpart of `s_tir.Schedule(func)`. |
| `base/common_schedules.py` | `base/common_schedules.{hpp,cpp}` | |
| `analysis/common_analysis.py` | `analysis/common_analysis.{hpp,cpp}` | `BufferInfo`, `is_layout_transform` and `is_data_pad` are not ported: only the unported rules use them. |
| `gpu/base.py` | `gpu/base.hpp` | |
| `gpu/fallback.py` | `gpu/fallback.cpp` | |
| `gpu/general_reduction.py` | `gpu/general_reduction.cpp` | |
| `gpu/reduction.py` | `gpu/reduction.cpp` | |
| `gpu/transpose.py` | `gpu/transpose.cpp` | |
| `gpu/matmul.py` | `gpu/matmul.cpp` | `Matmul` (SIMT tiling), without its tensor-core branches. `MatmulTensorization`, `MatmulInt8Tensorization` and `MetalMatmul` need the tensor intrinsics of `python/tvm/s_tir/tensor_intrin/cuda.py`, which are also Python-only (TVMScript). The OpenCL-only `sch_outer_reduction` is not ported. |
| `gpu/gemv.py`, `gpu/low_batch_gemv.py`, `gpu/rmsnorm.py`, `analysis/gemv.py` | — | Not ported yet: they serve matrix-vector products (LLM decoding). |

## Updating to a new TVM version

1. Diff the Python sources between the pinned commit and the new one:
   `git diff OLD NEW -- python/tvm/s_tir/dlight`.
2. Apply the same changes to the corresponding C++ files.
3. Update the pinned commit in this README and in every file header.

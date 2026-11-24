\page dev_runtime Runtime: models, sessions, and CUDA graphs
\tableofcontents

How compiled models are called: what state a call needs (code, library handles, workspace, streams, captured CUDA
graphs), who owns it, and how the bindings expose it.

**Status:** implemented: the caller's workspace, sessions, and CUDA graphs (see "Implementation order"). A C API and
a `Model` class are deferred (TODO.md).

# Why

- **CUDA graphs** remove the host overhead of a step: about 3 ms per BERT-base training step on an A100, against
  about 1 ms for PyTorch. A graph replays the device pointers it was captured with, so its workspace must stay
  allocated and must not be used by anything else. TVM's workspace pool cannot guarantee that: when a later
  allocation needs a larger block, it frees a block and allocates a new one
  (`src/runtime/workspace_pool.cc`).
- **No allocation per call:** a preallocated workspace also helps the CPU.
- **Concurrency:** several threads should be able to use one model at once without duplicating what can be shared.

# Model and session

The split follows TensorRT's engine and execution context.

| Object | Owns | Sharing |
|---|---|---|
| **Model** | the loaded code (system library); target handles and selections that do not depend on the caller (e.g. cuBLAS/cuDNN handles, selected plans); later prepacked weights | immutable after creation; one per process, used by any number of threads |
| **Session** | the workspace arena (grown on demand, never shrunk while graphs use it); on CUDA its stream and its captured graphs | one per thread (or per stream); not thread-safe |

Several threads therefore share one model and each has its own session. Only the per-call state is duplicated.

## Compiled interface

The compiled object exports TVM FFI functions, and these are the interface that every binding builds on (see
`mdl::ExportedFunction`):
- **Each function `<symbol>`** takes the caller's workspace as its last argument: a 1-D uint8 tensor on the tensors'
  device, aligned to 64 bytes. Its intermediate buffers are views into the workspace, so a call allocates nothing.
- **`<symbol>_workspace_size`** (`mdl::workspaceSizeSymbol`) takes the same arguments without the workspace and
  returns the bytes needed. It launches nothing; for a consumer whose dimensions come from the context header, it
  copies the 64-byte header to the host, as the function itself does.

Model and session are objects of the bindings, not compiled code: the model object holds no C++ state, so a C API
around them would only wrap the bindings' code. A C API (`mdl_<prefix>_model_create`, `..._session_create`, ...)
becomes worthwhile once the model owns state (handles, plans), together with that state.

## Bindings

- **C++ (implemented):**
  - Each module struct has a nested `Session` class, movable but not copyable, that owns the workspace and grows it
    on demand.
  - The functions are members of `Session`. The module's static functions call them on a session per thread
    (`defaultSession()`), so existing code keeps working.
  - The workspace is not taken from TVM's workspace pool, which keeps one pool per thread: sessions can be moved
    between threads and outlive the pool at program exit. Host memory comes from `aligned_alloc`; device memory is a
    TVM tensor object (`runtime.TVMTensorAllocWithScope`), freed when its last reference goes.
  - The model is the module struct itself (the system library is loaded once); a `Model` class comes with the state
    it would own.
- **Python (implemented: `python`, `python-torch`, `python-tensorflow`):**
  - The generated class plays the model's role (library and weights) and keeps one workspace per thread
    (`threading.local`): the per-thread default session. An explicit `Session` class is not generated yet.
  - With PyTorch, the workspace is a `torch.empty` tensor, so it goes through PyTorch's allocator. Otherwise it is a
    tensor of the TVM runtime.
  - The class loads the functions through the system library (`ffi.SystemLib(prefix)`), as the C++ bindings do; the
    functions' calls into the TVM runtime (e.g. copying the context header) crashed when they were called through
    `tvm_ffi.load_module`.
- **Later:** the session's stream should be the framework's current stream where it has one (PyTorch:
  `torch.cuda.current_stream()`), so that the model is ordered with the framework's work.

Context buffers (the values that `forward_train` saves for `forward_backward`) stay as they are: the caller
allocates them per step, separate from the session.

# CUDA graphs

These live in the session and are on by default; `MDL_CUDA_GRAPHS=0` disables them.

**Where:** the bindings are target-independent, so the device-specific part is a library of the device's runtime:
- `mdl_session_cuda` (`src/mdl/runtime/sessions/cuda.cpp`, linked by `mdl_target_cuda`) registers the TVM FFI
  functions `mdl.session.2.create(device_id)`, `call(session, function, args...)` and `free(session)` (2 is
  `kDLCUDA`).
- A binding's session (C++ `Session`, Python's per-thread state) looks them up for the tensors' device and calls
  through them. Where none are registered (the CPU), it calls directly.
- A CUDA session owns a stream, created blocking so that it is ordered with the legacy default stream, and sets it as
  the TVM FFI environment's stream during its calls. TVM's kernels and the kernel packages launch on that stream.

- **Key:** the function plus everything a graph replays: the data pointers, shapes and strides of the tensor
  arguments, and the integer arguments.
- **First call of a key:** runs normally. It is a warm-up: cuBLASLt autotuning, cuDNN plan building and workspace
  growth cannot be captured.
- **Second call:** captured on the session's stream, instantiated, then launched. Capture uses
  `cudaStreamCaptureModeThreadLocal`: a synchronizing call fails instead of running once. A relaxed capture would
  run such a call (e.g. a copy of device data to the host) once, and every replay would reuse the value it read.
- **Later calls:** replay the graph (`cudaGraphLaunch`). A graph whose pointers differ only in the arguments could
  be updated with `cudaGraphExecUpdate` instead of recaptured (later).
- **Failure:** if capture fails (an operation that cannot be captured, e.g. a synchronization in a kernel
  package, or the context header that a consumer copies to the host), the key is marked as uncapturable and runs
  directly from then on. A call that fails for other reasons fails again when it runs directly, and its error is
  raised.
- **Bounds:** a session keeps at most 32 keys (graphs, warm-ups, failed captures); the least recently used one is
  dropped. Shapes that change on every call (e.g. sequence lengths) therefore do not accumulate graphs.
- **Memory:** a graph keeps the session's arena at the size it was captured with. Several shapes share the arena,
  because calls of one session are ordered on its stream. Different sessions never share it.

# Other targets

The C API is target-independent; only what a session holds differs.

| Target | Session holds | Graph equivalent |
|---|---|---|
| CPU (LLVM) | the arena; the thread-pool setting | none (host code calls the kernels directly) |
| CUDA | the arena, a stream, CUDA graphs | CUDA graphs |
| Vulkan / Metal (later) | the arena, a queue | recorded command buffers / indirect command buffers |

# Implementation order

1. **Done.** The caller's workspace and sessions (C++: `Session`; Python: one workspace per thread); the static
   functions use a session per thread.
   - Measured on BERT-base: GPU A100 bf16 train 64x128 min 31.55 ms (unchanged), CPU f32 train min 1229 ms (1203
     before; within the run-to-run spread of 5 iterations).
   - Each call also calls the size function: one more FFI call.
2. **Done.** CUDA graphs in the session, with a stream per session.
   - Measured, A100 bf16 BERT-base training step: 64x128 30.85 ms vs 31.67 ms without graphs, 16x512 33.44 vs 34.13
     ms.
   - Kernel launches by the host per profile: about 18,000 → about 1,000, plus one graph launch per call.
   - `cuda_bert_*` checks the result of a replay (`bert_check.cpp` calls three times).
   - Not yet: the framework's current stream in the Python bindings; updating a graph whose pointers changed
     (`cudaGraphExecUpdate`) instead of a new capture.

This replaces the TODO item "caller-provided workspace".

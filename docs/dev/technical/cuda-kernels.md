\page dev_cuda_kernels CUDA kernels
\tableofcontents

Technical notes on the kernels a model compiled with `--target 'cuda -arch=sm_XX'` runs, and how they are produced.
For the kernel package format (manifests, patterns, expressions) see \ref spec_kernel_packages.

A compiled model's GPU work comes from two sources:

1. **Generated kernels**: the fusion planner groups the remaining operations and TVM's FuseTIR makes one PrimFunc per
   group (see \ref dev_optimization). The CUDA target schedules them and compiles them to CUDA C, then
   to a cubin with NVRTC at compile time. The cubin is embedded in the model's
   object file. Loading a model needs no compiler.
2. **Kernel packages**: library calls (cuBLASLt, cuDNN) that the backend selects by pattern matching before fusion.
   From TVM's point of view these are `call_extern`s in host code.

Every kernel runs on TVM's current stream (`TVMFFIEnvGetStream`). During a call, a session sets that stream to its own
stream and replays repeated calls as CUDA graphs. Temporary buffers are offsets in the session's workspace. See
\ref dev_runtime.

Kernel packages must therefore launch on `TVMFFIEnvGetStream` and must not synchronize, or calls that use them are not
captured. They also must not allocate after their first call with a configuration: the first call is not captured.

# Pipeline overview (CUDA-relevant parts)

| Stage | Where | What |
|---|---|---|
| Attention fusion | `src/mdl/passes/attention.cpp` | softmax(QKᵀ·s + bias)·V → `ATTENTION*` builtins (target-independent) |
| Kernel selection | `details/kernel_match.cpp`, `tvm.cpp` | Relax ops → package calls, in rounds; context pruning afterwards |
| Legalization | `details/legalize.cpp`, `Target::overrideLegalizations` | remaining ops → TIR; the CUDA target overrides `mdl.scatter_add_rows` |
| Fusion planner | `details/fusion/` | which kernel calls become one kernel: producers and consumers, readers of the same tensors, reductions over the same rows or columns, several results (target-independent; before FuseTIR) |
| Merging and flattening | `details/flatten_injective.cpp` | the results of a kernel → one loop nest; injective kernels → one flat loop (target-independent) |
| Scheduling | `targets/cuda.cpp` `CUDATarget::schedule` | schedule rules, first match wins (below) |
| Lowering/emit | `CUDATarget::blockLowering/beforeSplit/emit` | TVM's default GPU TIR passes; CUDA C → NVRTC cubin → one `.o` |

Set `--dump-dir` to also write the generated CUDA source (`<model>.cu`).

# Generated kernels: schedule rules

`CUDATarget::schedule` applies the rules below in this order, and the first rule that accepts a PrimFunc schedules
it. If no rule fits a kernel, compilation fails with a diagnostic that names the kernel and prints its TIR. PrimFuncs
without loops, i.e. package calls, are marked as scheduled and skipped.

| # | Rule | Origin | Takes |
|---|---|---|---|
| 1 | `Matmul` | DLight (ported) | contractions that no package handles (tensor cores for f16/bf16) |
| 2 | `RowReduction` | ours | reductions over the innermost dimension(s), optionally followed by an epilogue |
| 3 | `ColumnReduction` | ours | sums over leading dimensions, innermost dimension kept |
| 4 | `Reduction`, `GeneralReduction` | DLight | other reductions |
| 5 | `Elementwise` | ours | elementwise ops and innermost-preserving copies, vectorized |
| 6 | `Transpose` | DLight | transposes through shared memory |
| 7 | `Fallback` | DLight | anything else: flat loop, one thread per element |

Our rules live in `src/mdl/backends/tvm/targets/gpu/`. They exist because DLight picks threads from static sizes,
which breaks down for symbolic dimensions such as batch and sequence, and because DLight reads column reductions
uncoalesced.

## RowReduction (`row_reduction.{hpp,cpp}`)

Sums, maxima, softmax and normalizations (LayerNorm forward and backward, log-softmax): one or more reductions over
the innermost dimension(s), followed by elementwise epilogues over the same rows. The fusion planner produces such kernels.

- One **warp per row**. The 32 lanes stride over the row and combine their partials with warp shuffles.
- A single reduction packs `kRowsPerBlock = 4` rows into a block. With several reductions (e.g. mean, then
  variance), each row gets its own block.
- **Several epilogues** (several results, e.g. a normalization's output and its statistics saved for the gradient):
  their loop nests are merged into those of the *anchor*, the last epilogue over the full rows (`Merge`). Full-row
  epilogues share all loops; per-row results share only the row loops and are written outside the column loop. Per-row
  results may come in any order after the reductions.
- **Per-row blocks between the reductions** (e.g. a mean that is a result and that the variance's reduction reads) are
  computed at the row loop like the reductions; every lane of the row computes the same value.
- The rule applies to rows of symbolic length, or of static length up to `kMaxRowLength = 4096`, and needs at least
  `kMinRows = 64` rows.

DLight's GeneralReduction would use 1024 threads for a symbolic row of perhaps 128 elements.

## ColumnReduction (`column_reduction.{hpp,cpp}`)

`out[n] = Σ_m x[m, n]`, several of them over the same columns (e.g. a normalization's `Σdy` and `Σ(dy·x̂)`, which
then read `dy` once), each optionally followed by epilogues per column (a cast, or accumulation into a gradient).
Bias and LayerNorm-parameter gradients have this form. The rule runs **two kernels** (rfactor):

1. **First pass:** the rows are split into chunks of `kChunk = 64`. Each thread sums one chunk for `kVector = 4`
   adjacent columns, using vector loads, in registers, and stores its partial sums once. A warp covers adjacent
   columns, so reads are coalesced. Blocks are `8 × 32` (`kChunksPerBlock × warp`). The output is partial sums of
   shape `[chunks, columns]` in the reduction type, e.g. f32.
2. **Second pass:** each thread sums the chunks of one column in a register and applies that reduction's epilogues.
   With few columns (e.g. 768 bias gradients) this pass is latency-bound, so it uses blocks of `kThreads = 32` (spread
   over many SMs) and unrolls the chunks by `kUnroll = 32`, which keeps their loads in flight together. Splitting a
   column's chunks over several threads would need its epilogue (often `gradient += sum`, possibly in place) on one of
   them; TVM's schedule cannot guard that, so it isn't done (see the open items below).

A reduction block may have two results (sibling sums merged before scheduling, see \ref dev_optimization): both passes then
keep a register cache per result, created before any is moved into the loops. Each reduction is scheduled with its
epilogues in its own loop nest; then the nests of all reductions are merged level
by level (`Merge`), so that both passes are one kernel each. The rule needs at least `kMinRows = 256` rows; with
fewer, DLight's rules take the kernel. The second pass's write-back is not predicated, so its thread count must
divide a static number of columns: 32 if it does, or the number of columns itself if it is at most 32. Otherwise
DLight takes the kernel.

## Elementwise (`elementwise.{hpp,cpp}`)

Elementwise ops such as GELU, residual additions, casts, and `x[m, n] + b[n]`. It also takes copies that permute or
broadcast only the *outer* dimensions, e.g. the attention head transpose `[heads, batch, seq, dim] → [batch, seq,
heads, dim]`.

- The condition: every buffer's innermost index is the innermost loop iterator.
- Each thread handles `kBytes = 8` bytes as one vector, i.e. 4 bf16 or f16 values, or 2 f32 values. 256 threads per
  block.
- The innermost size must provably be a multiple of the vector width, including for symbolic sizes. Otherwise
  DLight's rules take the kernel.
- The limit is 4 lanes because TVM's CUDA code generation cannot emit 8-lane 16-bit vectors (`make_ushort4` with 8
  arguments).

## flattenInjective (`details/flatten_injective.{hpp,cpp}`)

This pass runs before scheduling and is not CUDA-specific. It rewrites injective kernels (elementwise ops, casts,
reshape copies, broadcasts of trailing dimensions) into one loop over the result's flat index `n`:

- Operands with the result's layout are read at `n`.
- Trailing broadcasts are read at `n % P`.

A kernel with several results over the same domain first becomes one block with a store per result (one loop nest,
also when it cannot be flattened). Blocks with several stores are flattened as well, as long as every store writes at the
iterators.

Without this pass, splitting a fused loop over `[batch, seq, 3072]` with symbolic sizes costs several 64-bit
divisions per element. Kernels with reductions, transposes, gathers or strided buffers are left unchanged.

## Target-specific legalization: `mdl.scatter_add_rows` (`gpu/scatter_add_rows.{hpp,cpp}`)

This is the gradient of an embedding lookup (gather). The CPU legalization is a sequential loop that cannot be
scheduled on a GPU. The CUDA version is one PrimFunc, already scheduled, with two thread-bound loop nests that become
two launches:

1. a zero fill;
2. one thread per element of the added rows, using `atomicAdd`.

Repeated indices accumulate in arbitrary order, so the result is not bitwise deterministic.

# Kernel packages

Packages are built from `kernels/<name>/` and selected by the manifest's `pattern` and constraints. CPU artifacts
such as oneDNN are never selected for a CUDA target, and vice versa. Matching uses the artifact's `device-arch`.
Strided-view helpers shared with the CPU packages are in `kernels/common/include/mdl_strided.hpp` (`forEachGemm`,
`forEachBatch2`): they turn rank-generic strided views into batched GEMMs at call time.

## cuBLAS: `kernels/cublas` (`mdl_cublas_matmul_{f32,bf16,f16}`)

- **Pattern:** `matmul` (`mdl.matmul`, with permutations, an optional bias, an optional addend, alpha and
  quantization captures).
- **ABI:** the output, A, B, the bias and the addend as pointers; the bias and addend are `NULL` when absent. Each
  operand is a strided view `(ndim, shape[], strides[])`, with the matched permutations applied to the shapes and
  strides. A permuted operand, e.g. `xᵀ` in a weight gradient, therefore costs no copy.
- **Implementation:** cuBLASLt `cublasLtMatmul`, computing in f32. Transposes and leading dimensions are derived from
  the strides. Batched views use a strided-batch layout where possible, otherwise `forEachGemm` loops.
- **Epilogue:** `CUBLASLT_EPILOGUE_BIAS` when there is a bias.
- **Addend:** passed as C with β = 1, which accumulates. This is how the in-place gradient accumulation works: the
  graph's `add(previous, result)` matches `addend`, and C aliases the output.
- **Algorithm choice:** on the first call for a problem shape, the heuristic candidates are each timed 3 times into a
  scratch buffer, never into the output, because C may alias it. The fastest is cached.
- **Per thread and device:** the cuBLAS and cuBLASLt handles and a 32 MiB cuBLASLt workspace.
- **Epilogues:** only those `mdl.matmul` has (bias, scale, addend). Activations such as GELU are generated kernels,
  which the fusion planner fuses with their neighbors (cuBLASLt's GELU is only the tanh approximation anyway). The
  bias-gradient epilogue (BGRADB) is not used (see the open items below).

## cuDNN: `kernels/cudnn` (`mdl_attention_{fwd,bwd}_{bf16,f16}`)

Fused scaled-dot-product attention through the cuDNN frontend (`src/attention.cpp`).

- **Patterns:** the Relax tuple ops `mdl.attention` (11 arguments) and `mdl.attention_grad` (18 arguments). The
  frontend's `ATTENTION`, `ATTENTION_LSE` and `ATTENTION_GRAD` builtins become these ops. Permutes around Q, K, V and
  O are folded into the views (`FoldPermutes`), so head transposes are strides.
- **Forward:** O, plus the log-sum-exp statistics when a backward pass needs them (training). The statistics are what
  context pruning keeps, instead of the S×S probabilities.
- **Backward:** dQ, dK and dV from Q, K, V, O, dO and the statistics.
- **Bias:** optional additive bias (element masks, as `-inf` fill). The pointer may be `NULL`; its shape is always
  passed. The bias has no gradient (`NoGradientRule`).
- **Lengths (suffix masks, `--mask f=suffix`):** int32 per-sequence lengths turn on cuDNN's padding mode
  (`set_padding_mask`, `set_seq_len_q/kv`). No mask bias is materialized. Batch dimensions are swapped where cuDNN
  requires it.
- **Plan selection:** all execution plans that the heuristic (mode A) offers are built. On the first call of a
  configuration each is timed three times on the call's tensors, and the fastest is kept, as the cuBLAS package does
  for algorithms. On the A100 slice the first plan was already the fastest.
- **Caching:** one operation graph per configuration (device, kind, type, dims, strides, bias, padding, scale), built
  once (`HeurMode A`, then check support, then build plans) and cached per thread. If cuDNN does not support a
  configuration, the call returns an error.
- **f32:** not covered. f32 attention is expanded into matmul + softmax kernels, the expansion fallback for unmatched
  ops.

## FlashAttention-2: `kernels/flash` (`mdl_flash_attention_{fwd,bwd}_{bf16,f16}`)

FlashAttention-2 (github.com/Dao-AILab/flash-attention, BSD-3-Clause, v2.7.4.post1) behind the cuDNN package's ABI.

- **Patterns and priority:** the same patterns as the cuDNN package, priority 300 against cuDNN's 200, so it is chosen
  wherever it applies:
  - no bias and no lengths: masked attention stays on cuDNN;
  - rank 3 or 4;
  - head dimensions 32, 64 or 128 (V's the same as Q's);
  - the head dimension contiguous.
- **Mixing packages:** both packages write the row statistics in the same layout (`[..., Sq]`, float32), so a forward
  pass of one and a backward pass of the other fit together.
- **Sources:** fetched by CPM with FlashAttention's CUTLASS submodule.
  - Only the instantiations we call are compiled: forward and backward, bf16 and f16, head dimensions 32/64/128, not
    causal. That is 12 compilation units, about 3 minutes with `-j8`.
  - `FLASHATTENTION_DISABLE_{DROPOUT,ALIBI,SOFTCAP,LOCAL,UNEVEN_K}` remove the features we don't use.
  - Built for sm_80, and newer GPUs through PTX.
  - `MDL_ENABLE_FLASH=OFF` skips the package.
- **Wrapper (`src/flash.cu`):** replaces FlashAttention's `flash_api.cpp`, which needs PyTorch. It fills FlashAttention's
  parameters from the strided views: the logical `[L0, L1, S, D]` becomes FlashAttention's batch `L0` and heads `L1`.
  Then FlashAttention's row statistics `[b, h, s]` are the ABI's layout.
- **PyTorch includes:** `shim/` replaces the three PyTorch headers that FlashAttention's kernels include. These are
  the dropout generator state (unused without dropout) and the CUDA error macros, which throw an error that the
  wrapper turns into a non-zero return value.
- **Backward workspace:** dQ is accumulated in float32 (`[b, Sq rounded to 128, h, d]`), plus the row sums of
  dO·O. Both live in a per-thread, per-device buffer that grows on demand, so CUDA graph captures (after the warm-up)
  allocate nothing.
- **Measured** (A100 bf16 BERT-base training): attention 3.58 → 3.13 ms per step (PyTorch's FlashAttention: 2.99 ms).
  The step takes 29.27 ms against `torch.compile` + SDPA's 30.18 ms at 64×128, and 31.37 ms against 32.29 ms at
  16×512.

# Adding a CUDA kernel

- **A new library call:** add an implementation to a package manifest (see \ref spec_kernel_packages), with a
  strided-view ABI, `NULL` for absent optional operands, and TVM's stream.
- **A new generated-kernel shape:** add a `GPUScheduleRule` under `targets/gpu/` and insert it into the rule list in
  `CUDATarget::schedule`, before the DLight rule it should pre-empt. Rules must cope with symbolic dimensions and
  return `std::nullopt` for anything they don't handle.
- **An op that cannot be legalized generically:** register a handler in `CUDATarget::overrideLegalizations`.

# Open items

\todo The cuBLASLt bias-gradient epilogue (BGRADB) was slower than separate column sums on an A100 slice
(cuBLAS 12.9: CUTLASS `bgradb` kernels) and is not used; measure it again on other GPUs and versions.

\todo ColumnReduction's second pass: the partial sums of a column over several threads (a cross-thread
reduction), with the epilogue on one of them.

\todo The merged sibling sums of a LayerNorm backward pass take 46 µs where a hand-written kernel takes 17 µs: the
reductions' initialization inside the loop (`DecomposeReduction` splits the kernel) and 64-bit index arithmetic.

\todo The Python bindings do not use the framework's current stream yet (see \ref dev_runtime).

\todo Unpadding of suffix-masked regions.

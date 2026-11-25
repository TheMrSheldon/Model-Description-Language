\page dev_optimization Optimization patterns
\tableofcontents

This page describes the general optimizations mdlc applies between the compute graph and the generated code. None of
them is specific to a model, and almost all of them are target-independent. Target-specific parts are noted.
For the CUDA kernels themselves see \ref dev_cuda_kernels; for kernel packages see
\ref spec_kernel_packages.

The guiding rule is to recognize *structure* rather than specific models or layers. Each pass looks for a general
shape in the graph (an elementwise chain, a reduction, operations reading the same tensor) and is safe whenever that
shape occurs.

# Overview

| Stage | Pattern | Where |
|---|---|---|
| Graph (frontend) | Attention fusion | `src/mdl/passes/attention.cpp` |
| Graph (frontend) | Masking: element vs suffix masks | `src/mdl/passes/masking.cpp` |
| Graph (frontend) | Gradients: what to save (minimum cut), what to recompute | `src/mdl/passes/autodiff.cpp` |
| Relax | Canonicalization, CSE, permutation folding | `details/pipeline.cpp`, `details/permutations.cpp` |
| Relax | Kernel selection (library calls) in rounds | `details/kernel_match.cpp`, `tvm.cpp` |
| Relax | Context pruning, context layouts | `details/contexts.cpp` |
| Relax | In-place accumulation | `tvm.cpp`, `details/aot.cpp` |
| Relax | Fusion planner: which kernel calls become one kernel | `details/fusion/` (then TVM FuseTIR) |
| TIR | Merging results, flattening injective kernels | `details/flatten_injective.cpp` |
| TIR | Sibling sums in one block | `details/merge_sums.cpp` |
| TIR | Schedule rules (per target) | `targets/` |
| AOT | Caller's workspace, kernel CSE | `details/aot.cpp`, `details/pipeline.cpp` |
| Runtime | Sessions, CUDA graphs | bindings, `src/mdl/runtime/sessions/` (\ref dev_runtime) |

# Graph level

## Attention fusion

`softmax(Q·Kᵀ·scale + bias) · V` is rewritten to an `ATTENTION` builtin.

- **Scale:** the pass collects the constant factors around the product.
- **Bias:** optional. A masked softmax's fill is also recognized as a mask bias.
- **Lengths:** optional; they come from a suffix mask.

The pass runs to a fixed point, *before* autodiff. Its gradient is therefore `ATTENTION_GRAD`, which needs only the
log-sum-exp statistics (`ATTENTION_LSE`), not the S×S probabilities.

If no kernel package implements the fused op for a target, the backend expands it back into matmul + softmax. The
rewrite is therefore never a pessimization.

## Element masks and suffix masks

The masking pass turns a masked function into an unmasked one with explicit fills. The kind is chosen at compile time
(`--mask f=element|suffix`); the model code is the same for both.

- **Element mask:** arbitrary booleans per element. The most general form.
- **Suffix mask:** per-sequence lengths. The structure is known: everything after the length is masked. This allows
  cheaper forms, such as cuDNN's padding mode instead of a materialized `-inf` bias, and later unpadding.

## Gradients: save or recompute

The forward pass of a training build stores *residuals* in a context buffer, and `forward_backward` reads them.
`autodiff::split` chooses them with a minimum cut, as PyTorch's AOTAutograd partitioner does (`autodiff.cpp`):

- **The graph:** the primal values: they depend on the inputs but not on the cotangents. Values that depend only on
  weights and constants are always recomputed and are not part of the cut. Each value has a capacity, the cost of
  saving it. The backward pass must be able to compute every value it reads from the saved values; it recomputes
  the rest.
- **Cost of saving a value:**
  - Its bytes, if the forward pass writes it to memory anyway. These are the inputs and outputs, the results of
    matrix products, attention and other library operations, and the arguments of non-elementwise forward
    operations, also through reshapes and permutations.
  - A GEMM's epilogue, a bias or scale of weights and constants that the backends compute with the product, counts
    as written in the product's place.
  - Twice its bytes otherwise, because the forward pass must also write it.
  - The sizes come from the exact shapes. Dimensions known only at runtime are taken as `typicalDim` (128); since all
    sizes scale with them, this only orders the costs.
- **Never recomputed:** matrix products, attention, embeddings and other library operations.
- **Saved by preference:**
  - A reduction whose result is at least 4x smaller than its input, e.g. the statistics of a normalization:
    recomputing it would read the whole input again.
  - A value of the forward pass that a non-fusible operation of the backward pass reads, e.g. `gelu(h)` for the
    weight gradient of the next matrix product. Recomputing it would write it again; saving it costs only memory.
  - A view is never preferred, since the backends fold views into the operation (e.g. the permuted heads that
    attention reads). Its source is preferred instead. A view also costs slightly more to save than its source, so
    that on a tie its source is saved (e.g. the per-row mean, not its reshape for broadcasting).
- **Ties:** among cuts of the same cost, the one closest to the backward pass is taken: the latest values. For
  example, a matrix product with its bias is saved rather than without it, which would move the bias out of the
  GEMM's epilogue.
- **Memory budget:** `SplitOptions::memoryBudget`, in bytes at `typicalDim`; internal, not exposed to users. If the
  preferred residuals exceed it, the cut is taken without preferences: everything that can be recomputed is.
  \todo The compiler could set the budget from the model, the batch size and the target's memory.
- **Boolean masks** are small and simply part of the cut.
- **Checkpointing:** module calls selected for checkpointing are always recomputed, including their matrix products;
  their inputs can be saved.

Measured (A100 bf16 BERT-base training):
- GELU backward: 149 → 117 µs (PyTorch: 105 µs).
- LayerNorm statistics saved (with RowReduction's per-row results, see \ref dev_cuda_kernels): forward LayerNorm 43 → 22
  µs, and the backward pass no longer recomputes them. That is about 0.1 ms per step of GPU time.
- Step: 30.85 → 30.35 ms at 64x128, 33.44 → 32.9 ms at 16x512.
- The context grows by `gelu(h)`, about 600 MB at 64x128 (12 layers), and by the statistics, a few MB.

Recomputation can create *siblings*: a recomputed value and its derivative, both computed from the same saved input.
The fusion planner puts these into one kernel (below), so recomputing costs neither memory nor an extra memory pass.

# Relax level

## Canonicalization, CSE and permutation folding

After translation, the module is normalized: constants folded, common subexpressions eliminated, scalar constants
inlined. Permutations (transposes) next to matmuls and attention are folded into the operation's view arguments:

- `mdl.matmul` carries `perm_a`, `perm_b` and `perm_out`.
- The attention ops take strided views of Q, K, V and O.

A transpose is therefore usually just strides, not a copy.

## Kernel selection in rounds

Kernel packages declare patterns (`package.yml`) over the backend's ops. Matching runs in rounds. Each round:

1. picks the best matches (by priority and constraints);
2. replaces them with extern calls;
3. re-canonicalizes, which may expose new matches.

An epilogue is part of the pattern, so the library computes it in the same call:

- a bias, through cuBLASLt's bias epilogue;
- an addend (C with β = 1);
- scaling;
- dequantization.

Operations that no kernel package matches are left for TVM to generate code for.

## Context pruning

After selection, the context buffer between forward and backward is pruned to the values the backward pass still
reads. For example, once attention and its gradient are library calls, only the statistics remain, not the
probabilities. Unused context outputs are removed from the producer and unused parameters from the consumer
(`pruneContexts`), followed by dead-code elimination (`afterPruning`).

## Context layouts

`permuteContexts` runs once before kernel selection. A tensor is saved in the layout its producer uses it in when:

- the forward pass returns a value `v` that it otherwise only permutes (`p = permute(v)`);
- the consumers use it only in attention operations or permutations.

The producer then returns `p`, and the consumers read `permute⁻¹(p)`. A consumer's own permutations compose with the
inverse. The same permutation as the producer's cancels, so that use reads the saved tensor itself.

Example: attention writes O head-first. The output projection reads it as `[batch, seq, heads, dim]`, the backward
pass passes it to the attention gradient, and the weight gradient reads it transposed. Saving the transposed O lets
cuDNN write it in that layout through strides, and the attention gradient reads it through strides. The transpose
kernel disappears in both passes. The tensor's size, and so the buffer layout, is unchanged.

## In-place gradient accumulation

Gradient outputs are accumulated: `grad = previous + result`. The graph says this with an ordinary `add(previous, result)`.
The hidden `previous` parameter aliases the output buffer (`EntrySignature::aliases`), so:

- a matmul whose result is added matches the `addend` of the matmul pattern, and cuBLAS or oneDNN writes
  `C += A·B` directly into the gradient;
- an elementwise producer fuses the add into its kernel.

There is no separate gradient buffer and no copy, and no special case in the backend: it is pattern matching on the
graph.

## Fusion planner

The planner (`details/fusion/`, inspired by PyTorch Inductor's scheduler) decides which legalized kernel calls
become one kernel. It replaces TVM's FuseOps. FuseOps fuses only producers into consumers, puts at most one reduction
into a group, never makes kernels with several results, and never duplicates a producer.

Its parts:

| Part | What it does |
|---|---|
| `signature` | The **loop signature** of a kernel call, from its PrimFunc's blocks (the same analysis as the schedule rules use) with domains from the Relax types: pointwise over D; reductions over rows or columns (kept dimensions, reduced input domain); an anchor (a reduction whose operands are read along different dimensions, e.g. a matrix product without a library kernel); or opaque (library calls, kernels marked `op_pattern` 8). Reshapes are pointwise **views**. |
| `rules` | The contract with the targets' schedule rules: which groups one kernel computes. Pointwise groups store results of one domain. Reduction groups combine reductions over the same rows or columns with pointwise computations: rows store per-row results and full-row epilogues, columns per-column results, and nothing of the full domain follows a column reduction. An anchor takes pointwise consumers. Internal members may have any domain (they are inlined). Views never form a group alone and are never a group's result (they are free views on their own). |
| `planner` | Per dataflow block: candidates are producer→consumer edges and consecutive readers of the same tensor. They are scored by the bytes a merge saves (intermediates that are no longer written and read, inputs read once) and merged greedily (union-find) while the rules hold and no group would depend on itself through another. |
| `duplicate_producers` | Before planning: a cheap pointwise value whose readers could all contain it (e.g. the `erf` that GELU and its gradient share) is computed once per reader, along shared chains. Copies that are not fused are merged again. |
| `materialize` | Groups → primitive functions, with TVM's grouping (`MakeGroupedFunctions`, as FuseOps' and FuseOpsByPattern's; the only use of TVM internals). A result that its group also uses is computed again at the end, so that every result is written by a trailing epilogue (the form the schedule rules expect). |

FuseTIR then makes one PrimFunc per group. Target limits (row lengths, threads) are not part of the rules: a target's
rule declines, and its fallback schedules the kernel.

Results in BERT-base bf16 training on an A100: one kernel each for LayerNorm forward and backward, GELU and its
gradient, the bias-gradient sums with their accumulation, and a normalization's `Σdy` and `Σ(dy·x̂)` (reading `dy`
once); 600 kernel launches per step (FuseOps and the earlier special passes: 816).

## Generated kernels against the oneDNN kernels (CPU)

The oneDNN package still has hand-written kernels for operations that the fusion planner and the CPU schedule rules
also generate: softmax, GELU and its gradient, layer normalization and its gradient, and row sums. A kernel is only
removed once the generated version is measured at parity. Measured on 16 Zen 2 cores (BERT-base, float32, batch 8x128,
median of 20 runs; the same configuration varies between 310 and 328 ms for inference and 1181 and 1221 ms for
training from run to run), each without one kernel:

| Without | Inference | Training |
|---|---|---|
| (all kernels) | 310 ms | 1181 ms |
| GELU | 337 ms | 1251 ms |
| GELU gradient | 316 ms | 1203 ms |
| softmax | 325 ms | 1200 ms |
| layer normalization | 323 ms | 1202 ms |
| layer normalization gradient | 315 ms | 1209 ms |
| row sums | 323 ms | 1219 ms |

Only GELU is clearly slower when generated (6-8 %); the others are within the run-to-run variation, so they are not
conclusively at parity either. None is removed yet.

\todo Repeat the measurements interleaved (several alternating runs) for the kernels within the variation; then
remove those at parity, and improve the CPU schedules for GELU (a vectorized `erf`).

## Patterns must not duplicate work

`applyPatterns` does not rewrite a match whose intermediate values have users outside the match
(`sharesIntermediates`). Intermediate values are matched operations other than the root. The replacing operation
would compute the shared value again. This holds for every pattern:

- **LayerNorm:** in training, the mean, `x − mean`, `rstd` and `x̂` are saved for the backward pass.
- **GELU:** its gradient shares `erf(x/√2)` with it, which the graph's common-subexpression elimination already
  shares.

In such cases the decomposed form stays, and the fusion planner turns it into as few kernels as before. Inference, where
nothing shares the intermediates, keeps the single operations.

# TIR level

## Merging results and flattening injective kernels

`flattenInjective` (TIR, before scheduling, target-independent):

- A kernel with several results over the same domain becomes one block with a store per result, in one loop nest. If
  it cannot be flattened (e.g. transposes), it stays that one loop nest.
- Injective kernels become one loop over the result's flat index: operands with the result's layout are read at `n`,
  trailing broadcasts at `n % P`. This removes the 64-bit divisions of split loops over symbolic dimensions.
- Reshapes that only insert or remove unit dimensions are legalized as direct index maps (not TOPI's reshape, which
  divides by symbolic extents), so that they inline into fused kernels.

## Sibling sums in one block

`mergeSiblingSums` (TIR, before scheduling, target-independent) merges pairs of sums over the same iteration domain that
do not depend on each other, e.g. the gradients of a normalization's scale and shift: `sum(dy · x̂)` and `sum(dy)` over
the rows.

- **Why:** as separate blocks, each sum has its own loop body, loads and predicates in the scheduled kernel, even with
  its loops merged. A shared input is then loaded twice.
- **Form:** one block with TVM's two-sum reducer. The new values are bound first, then stored: `v0 = acc0 + x; v1 =
  acc1 + y; acc0 = v0; acc1 = v1`, which is how TVM recognizes a reduction with several results (for `rfactor` and
  cross-thread reductions).
- **Order:** the root's statements are reordered by their buffer dependencies, so that the merged sums come after the
  producers of both and before the consumers of both. A pair is merged only if neither depends on the other, also
  through the statements between them.
- **Pairs only:** TVM's reducers have at most two results.
- **bfloat16:** needs `patches/tvm/0001-bf16-compute-legalize-bind.patch`. TVM v0.27's bfloat16 legalization does not
  legalize the values of bound variables, which multi-result reductions use.
- **Measured** (A100 bf16 BERT-base training, LayerNorm parameter gradients): first pass 61 → 46 µs, second pass 14 →
  9.5 µs. A hand-written kernel needs 17 µs; the rest is the reductions' initialization inside the loop and 64-bit
  index arithmetic (see the open items in \ref dev_cuda_kernels).

## Schedule rules

Each target schedules the kernels with an ordered list of rules, for the kinds of groups the planner forms (the
contract in `fusion/rules.hpp`). Our rules are written for symbolic dimensions (batch, sequence):

- **CUDA:** warp-per-row reductions (several reductions and epilogues), two-pass column reductions, vectorized
  elementwise kernels, then DLight's rules; see \ref dev_cuda_kernels.
- **CPU:**
  1. DLight's CPU `Reduction` (softmax, normalizations), ported. It applies only to kernels with one result: it
     computes every block at the last block's loops, and TVM's `compute_at` cannot move a result block
     (`targets/cpu/reduction.cpp` checks this).
  2. Our `cpu::Fallback` (`targets/cpu/fallback.cpp`): inlines the elementwise blocks, then vectorizes and
     parallelizes each remaining block.

# AOT level

- **Workspace:** all temporaries of a call are offsets in the caller's workspace (the last argument), sized from the
  symbolic dimensions by the function `<symbol>_workspace_size`. Offsets are bound as variables, so the size
  computation stays cheap. Sessions own the workspace; on CUDA they also replay calls as CUDA graphs. See
  \ref dev_runtime.
- **Kernel CSE:** common subexpressions are eliminated inside the generated kernels (TIR CSE). It runs on the kernels
  only, not on the host code with its arena bindings.

# Adding an optimization

- Express it as a structural property of the graph (elementwise, reduction over an axis, a shared input, a known mask
  structure), not as a model-specific rewrite.
- Prefer target-independent passes. Put target-specific choices in the target (schedule rules, legalization
  overrides) or in kernel packages (patterns).
- Fusion: extend the planner's contract (`fusion/rules`) and the targets' schedule rules, not a new fusion pass.
- A rewrite must be safe whenever it applies. If a fused form may be unsupported somewhere, keep a fallback, like the
  attention expansion.
- Verify with the test suite (`ctest`, including the planner's unit tests in `tests/mdl/backends/tvm_fusion.cpp`) and
  with the benchmarks against PyTorch (`examples/02_transformer`, `scripts/benchmark_torch.py`).

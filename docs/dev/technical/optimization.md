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
| Graph (frontend) | Gradients: what to save, what to recompute | `src/mdl/passes/autodiff.cpp` |
| Relax | Canonicalization, CSE, permutation folding | `details/pipeline.cpp`, `details/permutations.cpp` |
| Relax | Kernel selection (library calls) in rounds | `details/kernel_match.cpp`, `tvm.cpp` |
| Relax | Context pruning, context layouts | `details/contexts.cpp` |
| Relax | In-place accumulation | `tvm.cpp`, `details/aot.cpp` |
| Relax | Fusion planner: which kernel calls become one kernel | `details/fusion/` (then TVM FuseTIR) |
| TIR | Merging results, flattening injective kernels | `details/flatten_injective.cpp` |
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
  cheaper forms, such as cuDNN's padding mode instead of a materialized `-inf` bias, and later unpadding (see TODO).

## Gradients: save or recompute

The forward pass of a training build stores *residuals* in a context buffer, and `forward_backward` reads them. Which
values become residuals is decided in `autodiff.cpp`:

- **Recomputed in the backward pass:** results of purely elementwise chains. The backward pass instead saves the
  chain's input and recomputes the chain where it fuses into the kernel that uses it. An elementwise result is never
  smaller than its arguments, so this never costs memory. It also keeps the forward pass free of extra
  materializations, so a backend may still replace that computation with a fused kernel (e.g. a matmul epilogue).
- **Saved:** values that the forward pass materializes anyway, because a non-elementwise operation such as a matmul
  reads them.
- **Boolean masks:** saved; they are small.
- **Checkpointing:** module calls selected for checkpointing are recomputed from their saved inputs.

Recomputation can create *siblings*: a recomputed value and its derivative, both computed from the same saved input.
Sibling fusion turns these into one kernel (below), so recomputing costs neither memory nor an extra memory pass.

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

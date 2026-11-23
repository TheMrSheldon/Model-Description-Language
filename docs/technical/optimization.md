# Optimization patterns

This page describes the general optimizations mdlc applies between the compute graph and the generated code. None of
them is specific to a model, and almost all of them are target-independent. Target-specific parts are noted.
For the CUDA kernels themselves see [cuda_kernels.md](cuda_kernels.md); for kernel packages see
[../kernels.md](../kernels.md).

The guiding rule is to recognize *structure* rather than specific models or layers. Each pass looks for a general
shape in the graph (an elementwise chain, a reduction, operations reading the same tensor) and is safe whenever that
shape occurs.

## Overview

| Stage | Pattern | Where |
|---|---|---|
| Graph (frontend) | Attention fusion | `src/mdl/passes/attention.cpp` |
| Graph (frontend) | Masking: element vs suffix masks | `src/mdl/passes/masking.cpp` |
| Graph (frontend) | Gradients: what to save, what to recompute | `src/mdl/passes/autodiff.cpp` |
| Relax | Canonicalization, CSE, permutation folding | `details/pipeline.cpp`, `details/permutations.cpp` |
| Relax | Kernel selection (library calls) in rounds | `details/kernel_match.cpp`, `tvm.cpp` |
| Relax | Context pruning | `details/contexts.cpp` |
| Relax | In-place accumulation | `tvm.cpp`, `details/aot.cpp` |
| Relax | Producer–consumer fusion (TVM FuseOps/FuseTIR) | TVM |
| Relax | Row fusion | `details/fuse_rows.cpp` |
| Relax | Sibling fusion | `details/fuse_siblings.cpp` |
| TIR | Flattening of injective kernels | `details/flatten_injective.cpp` |
| TIR | Schedule rules (per target) | `targets/` |
| AOT | Workspace arena, kernel CSE | `details/aot.cpp`, `details/pipeline.cpp` |

## Graph level

### Attention fusion

`softmax(Q·Kᵀ·scale + bias) · V` is rewritten to an `ATTENTION` builtin.

- **Scale:** the pass collects the constant factors around the product.
- **Bias:** optional. A masked softmax's fill is also recognized as a mask bias.
- **Lengths:** optional; they come from a suffix mask.

The pass runs to a fixed point, *before* autodiff. Its gradient is therefore `ATTENTION_GRAD`, which needs only the
log-sum-exp statistics (`ATTENTION_LSE`), not the S×S probabilities.

If no kernel package implements the fused op for a target, the backend expands it back into matmul + softmax. The
rewrite is therefore never a pessimization.

### Element masks and suffix masks

The masking pass turns a masked function into an unmasked one with explicit fills. The kind is chosen at compile time
(`--mask f=element|suffix`); the model code is the same for both.

- **Element mask:** arbitrary booleans per element. The most general form.
- **Suffix mask:** per-sequence lengths. The structure is known: everything after the length is masked. This allows
  cheaper forms, such as cuDNN's padding mode instead of a materialized `-inf` bias, and later unpadding (see TODO).

### Gradients: save or recompute

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

## Relax level

### Canonicalization, CSE and permutation folding

After translation, the module is normalized: constants folded, common subexpressions eliminated, scalar constants
inlined. Permutations (transposes) next to matmuls and attention are folded into the operation's view arguments:

- `mdl.matmul` carries `perm_a`, `perm_b` and `perm_out`.
- The attention ops take strided views of Q, K, V and O.

A transpose is therefore usually just strides, not a copy.

### Kernel selection in rounds

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

### Context pruning

After selection, the context buffer between forward and backward is pruned to the values the backward pass still
reads. For example, once attention and its gradient are library calls, only the statistics remain, not the
probabilities. Unused context outputs are removed from the producer and unused parameters from the consumer
(`pruneContexts`), followed by dead-code elimination (`afterPruning`).

### In-place gradient accumulation

Gradient outputs are accumulated: `grad = previous + result`. The graph says this with an ordinary `add(previous, result)`.
The hidden `previous` parameter aliases the output buffer (`EntrySignature::aliases`), so:

- a matmul whose result is added matches the `addend` of the matmul pattern, and cuBLAS or oneDNN writes
  `C += A·B` directly into the gradient;
- an elementwise producer fuses the add into its kernel.

There is no separate gradient buffer and no copy, and no special case in the backend: it is pattern matching on the
graph.

### Producer–consumer fusion

TVM's FuseOps/FuseTIR fuse chains along data edges: elementwise ops into each other, and into the reductions they
feed or follow (e.g. a cast into a sum). The resulting kernel's name lists the fused ops (e.g.
`fused_rsqrt_multiply1`).

### Sibling fusion

FuseOps fuses only along producer–consumer edges. `fuseSiblings` merges independent `call_tir`s of injective kernels
when all of these hold:

- their results have the **same shape**;
- they **read at least one common tensor**;
- **neither depends on the other**. Every argument of the later call is computed before the earlier one, so the merged
  call can take the earlier one's place.

At most 8 kernels are merged into a group. The merged kernel:

- loops once over the shared iteration space, with one block containing one store per result;
- passes each shared argument once, so it is read once;
- uses the first member's symbolic dimensions for every member's buffers (they have results of the same shape), so
  later passes see one consistent layout;
- returns a tuple, whose elements rebind the original result variables, so the rest of the function is unchanged.

Typical siblings:

- a recomputed forward value and its derivative, e.g. `gelu(h)` for a weight gradient and `gelu_backward(h, dy)`;
- several elementwise results from the same activations, e.g. in normalization gradients.

Reductions are handled by row fusion (below), not by sibling fusion.

### Row fusion

FuseOps puts at most one reduction into a group. A normalization or its gradient therefore becomes many kernels that
read the same rows again and again. For example, LayerNorm backward,
`dx = rstd·(g − mean(g) − x̂·mean(g·x̂))`, took five kernels.

`fuseRows` runs between FuseOps and FuseTIR. It groups the fused calls that work on the same rows:

- a reduction over the trailing dimensions L of `[rows…, L…]`, with result `[rows…]`;
- further reductions over the same rows that read the group's inputs or results;
- per-row computations on the results (shape `[rows…]`, up to unit dimensions);
- full-shape elementwise computations of shape `[rows…, L…]`, e.g. `x − mean` between the mean and the variance.

A group is the longest prefix that ends with a full-shape epilogue:

- **Escaping results:** intermediate results used outside the group become further results, if they are elementwise.
  Reductions stay internal. Each escaping result is computed again at the group's end, so all results are written by
  trailing epilogues.
- **Unit-dimension reshapes:** TVM's reshape kernel divides by symbolic sizes and is opaque. Inside a group it is
  replaced by an elementwise kernel that maps the indices directly.

FuseTIR then produces one kernel per group. On GPUs, the RowReduction rule schedules it with a warp per row: all
reductions first, then the epilogues, whose loops are merged so that they share the row loop.

### Patterns must not duplicate work

A pattern that replaces a subgraph by one operation (e.g. `layer_norm`) checks that the subgraph's intermediate values
have no other users (`patternUses`). In training, the mean, `x − mean`, `rstd` and `x̂` are saved for the backward
pass. Replacing only the output with `layer_norm` would compute the normalization twice. There, the decomposed form
stays, and row fusion turns it into one kernel that writes the output and the saved values.

## TIR level

### Flattening injective kernels

Elementwise kernels, casts, reshape copies and trailing broadcasts are rewritten into one loop over the flat result
index `n`:

- operands with the result's layout are read at `n`;
- trailing broadcasts (e.g. a bias) are read at `n % P`.

This removes the 64-bit divisions that fusing and splitting loops over symbolic dimensions would otherwise cost per
element. Merged sibling kernels, with several stores, are flattened as well.

### Schedule rules

Each target schedules the remaining kernels with an ordered list of rules. Our rules are written for symbolic
dimensions (batch, sequence):

- **CUDA:** warp-per-row reductions, two-pass column reductions, vectorized elementwise kernels; see
  [cuda_kernels.md](cuda_kernels.md).
- **CPU:** `applyDefaultCPUSchedule`.

## AOT level

- **Workspace arena:** all temporaries of a call are offsets in one arena, sized from the symbolic dimensions.
  Offsets are bound as variables, so the size computation stays cheap.
- **Kernel CSE:** common subexpressions are eliminated inside the generated kernels (TIR CSE). It runs on the kernels
  only, not on the host code with its arena bindings.

## Adding an optimization

- Express it as a structural property of the graph (elementwise, reduction over an axis, a shared input, a known mask
  structure), not as a model-specific rewrite.
- Prefer target-independent passes. Put target-specific choices in the target (schedule rules, legalization
  overrides) or in kernel packages (patterns).
- A rewrite must be safe whenever it applies. If a fused form may be unsupported somewhere, keep a fallback, like the
  attention expansion.
- Verify with the test suite (`ctest`) and with the benchmarks against PyTorch (`examples/02_transformer`,
  `scripts/benchmark_torch.py`).

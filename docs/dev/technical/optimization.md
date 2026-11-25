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
| Relax | Producer–consumer fusion (TVM FuseOps/FuseTIR) | TVM |
| Relax | Stacked sibling sums | `details/stack_sums.cpp` |
| Relax | Row fusion | `details/fuse_rows.cpp` |
| Relax | Sibling fusion | `details/fuse_siblings.cpp` |
| TIR | Flattening of injective kernels | `details/flatten_injective.cpp` |
| TIR | Schedule rules (per target) | `targets/` |
| AOT | Workspace arena, kernel CSE | `details/aot.cpp`, `details/pipeline.cpp` |

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

## Producer–consumer fusion

TVM's FuseOps/FuseTIR fuse chains along data edges: elementwise ops into each other, and into the reductions they
feed or follow (e.g. a cast into a sum). The resulting kernel's name lists the fused ops (e.g.
`fused_rsqrt_multiply1`).

## Sibling fusion

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

## Stacked sibling sums

Column sums to the same shape of tensors computed from a common one are rewritten as one sum over a stack:
`collapse_sum_to(x_i, S)` becomes `collapse_sum_to(stack(x_1, …, x_k, axis=-1), S + [k])`, followed by a `take` per
result. Example: a normalization's `Σdy` and `Σ(dy·x̂)`.

- The inputs must share a tensor: one is the other's argument, or their elementwise producers have a common argument.
- The stacked axis is innermost, so adjacent threads sum the k columns. The producers fuse into the reduction, which
  then reads `dy` once.
- Bindings that use an earlier result move after the group, which is safe because dataflow bindings are pure.

## Row fusion

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

## Patterns must not duplicate work

`applyPatterns` does not rewrite a match whose intermediate values have users outside the match
(`sharesIntermediates`). Intermediate values are matched operations other than the root. The replacing operation
would compute the shared value again. This holds for every pattern:

- **LayerNorm:** in training, the mean, `x − mean`, `rstd` and `x̂` are saved for the backward pass.
- **GELU:** its gradient shares `erf(x/√2)` with it, which the graph's common-subexpression elimination already
  shares.

In such cases the decomposed form stays, and the fusion passes turn it into as few kernels as before. Inference, where
nothing shares the intermediates, keeps the single operations.

## Recomputing shared elementwise values

FuseOps does not duplicate a producer with several consumers, so a shared elementwise value (e.g. the `erf` above)
would be written to memory and read back. Before row fusion, an elementwise value whose uses are all elementwise
computations of its shape is computed in each of them. Sibling fusion then puts the consumers into one kernel, where
the duplicate computation disappears (TIR CSE, or the CUDA compiler).

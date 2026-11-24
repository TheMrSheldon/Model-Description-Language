# Einops Expressions {#spec_einops}

MDL supports einops-style tensor rearrangements via the `[lhs -> rhs | bindings]` syntax.
These expressions map directly to `PERMUTE`, `RESHAPE`, and `EXPAND` operations in the compute graph.

## Syntax

```
tensor[lhs -> rhs]
tensor[lhs -> rhs | key=value, ...]
```

`lhs` describes the current layout of the tensor; `rhs` describes the desired layout. An empty `rhs` makes a scalar
of a tensor whose dimensions all have size 1 (e.g. `x[1 -> ]`, `x[... 1 -> ...]` to drop a trailing size-1 dimension).
`key=value` pairs supply the sizes of named dimensions that cannot be inferred from the tensor shape alone (e.g. individual dims inside a merged group).

## Atoms

| Token | Meaning |
|-------|---------|
| `name` | A named dimension. The same name in LHS and RHS refers to the same dimension. |
| `_` | An anonymous (unnamed) dimension. Passes through without reordering. |
| `...` | An ellipsis representing zero or more batch dimensions of unknown rank. |
| `N` (integer) | A literal dimension of size N. In the RHS this introduces a new dimension. |
| `/N` | A chunk size — split a dimension into chunks of size N. |

## Parentheses (groups)

Parentheses merge or split dimensions:

- `(a b)` in the **LHS** means that dimensions `a` and `b` are stored merged as a single dimension in the input tensor. They will be split apart before further processing.
- `(a b)` in the **RHS** means that dimensions `a` and `b` should be merged into a single dimension in the output tensor.

## Examples

### Permute

<div class="mdl">

```mdl
x[b h w c -> b c h w]      // NCHW layout from NHWC
```
</div>

The compiler emits a single `PERMUTE` node.

### Merge dimensions

<div class="mdl">

```mdl
x[a b -> (a b)]             // flatten last two dims
```
</div>

Emits a `RESHAPE` node with output shape `[a*b]`.

<div class="mdl">

```mdl
x[... -> (...)]             // flatten all dimensions (e.g. mean_flat(x) = mean(x[... -> (...)]))
```
</div>

### Split a dimension

<div class="mdl">

```mdl
x[(a b) -> a b | a=4, b=8]  // split dim 0 into 4×8
```
</div>

Emits a `RESHAPE` node.  The sizes of `a` and `b` must be supplied via bindings.

### Unsqueeze (add a size-1 dimension)

<div class="mdl">

```mdl
x[a b -> a 1 b]             // insert a size-1 dim
```
</div>

Emits a `RESHAPE` node.

### Batch-agnostic permute with ellipsis

<div class="mdl">

```mdl
x[... h w -> ... w h]       // swap last two dims regardless of batch rank
```
</div>

Emits a `PERMUTE` node using negative indices for post-batch dims so the backend can handle tensors of unknown rank.

### Merge and permute together

<div class="mdl">

```mdl
x[b h w c -> b (h w) c]     // NHWC → N(HW)C
```
</div>

Emits a `RESHAPE` node (permutation is identity, so no `PERMUTE` is needed).

<div class="mdl">

```mdl
x[b h w c -> b c (h w)]     // NHWC → NC(HW)
```
</div>

Emits a `PERMUTE` followed by a `RESHAPE`.

## How bindings work

When a named dimension appears inside a parenthesized **LHS** group its size cannot be inferred from the tensor shape alone (only the product is known).  Provide the sizes via key-value pairs:

<div class="mdl">

```mdl
x[(h w) c -> h w c | h=16, w=16]
```
</div>

Bindings may also be runtime expressions, not just integer literals:

<div class="mdl">

```mdl
x[(h w) c -> h w c | h=height, w=width]
```
</div>

where `height` and `width` are in-scope variables.

## Emitted nodes

| Situation | Node emitted |
|-----------|-------------|
| Dimension order changes | `PERMUTE` |
| Grouped LHS (split merged dims) | `RESHAPE` |
| Grouped RHS or literal dims | `RESHAPE` |
| Anonymous dim broadcast to literal N | `EXPAND` |

Multiple transformations are composed in the order: un-merge → permute → expand → re-merge.

# Language specification

A model file (`.dlm`) contains imports, structs, functions, and modules. Comments are `// ...` and `/* ... */`; the
standard library documents its definitions with `/** ... */` comments.

How a model is compiled (hyperparameter values, `...` rank, float type, gradients, masking, quantization) is not part of
the language: these are build settings, see [build-config.md](build-config.md).

# Imports
```dlm
import std;              // all files of the standard library (directory std/)
import std.activations;  // a single file (std/activations.dlm)
import mylib.layers;     // mylib/layers.dlm, searched next to the importing file and in the -i directories
```
Imported definitions are visible in the importing file. Only the definitions of the compiled file itself are
exported; every file is translated once, even if it is imported several times.

# Modules
- Compile-time arguments ("hyperparameters"). E.g., the following is a linear layer with three hyperparameters: the
  input and output dimensions, and whether a bias is added.
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool = true] {
    /* ... */
  }
  ```
- Member variables (the weights):
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool = true] {
    W: float[outdim, indim]
    b: float[outdim]
    /* ... */
  }
  ```
- Member functions:
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool = true] {
    W: float[outdim, indim]
    b: float[outdim]

    float[..., outdim] forward(x: float[..., indim]) => x @ W.T() + b;
  }
  ```
- Instances of other modules (submodules); calling an instance calls its `forward`:
  ```dlm
  module Block[dim: uint] {
    linear: Linear[dim, dim]
    float[..., dim] forward(x: float[..., dim]) => linear(x) + x;
  }
  ```
  Every instance has its own weights, identified by their path (e.g. `linear.W`, or `layers.3.attn.W` for nested
  modules); calling the same instance twice uses the same weights. Other member functions of an instance are called on
  it (`norm.gradient(x, dy)`) and use its weights.
- Arrays of instances, one per index (each instance may get different hyperparameters):
  ```dlm
  layers: [Layer[d] for i in 0..n]
  stages: [Stage[conf.sizes[i], conf.sizes[i + 1]] for i in 0..3]
  ```
  `layers[i]` with a compile-time index is the instance `i` (weights `layers.<i>.…`).

A module is compiled if the argument values have an entry for it; each of its member functions becomes an exported
function.

## Hyperparameters
Hyperparameters may have default values (which may use the earlier hyperparameters). Values can be given by name
(designated-initializer style) after the positional ones; positional values fill the hyperparameters without default
in declaration order, and those with a default are only set by name (so that adding or reordering them never changes
the meaning of existing instantiations):
```dlm
module Linear[indim: uint, outdim: uint, bias: bool = true] { ... }

module Head[dim: uint] {
    proj: Linear[dim, dim]                  // bias = true (the default)
    out: Linear[dim, 3, .bias=false]
}
```
Missing values without default, unknown names, values given twice, and positional values for hyperparameters with a
default are errors:
```
error: bias of Linear has a default value and is only set by name
 3 |     l: Linear[4, 3, false]
   |                     ^^^^^
hint: write `.bias=false`
```
In the argument values (YAML) of a compiled module, left-out hyperparameters take their defaults.

Besides numbers, booleans, and structs, a hyperparameter can be a compile-time array `array[T; n]` (not a tensor), e.g.
the layer sizes of a network, given as a list in YAML (`sizes: [8, 16, 16, 4]`) or as a literal (`[8, 16, 16, 4]`),
and indexed with compile-time indices:
```dlm
struct MlpConf { sizes: array[uint; 4]  biases: array[bool; 3] }
module MLP[conf: MlpConf] {
    layers: [Linear[conf.sizes[i], conf.sizes[i + 1], .bias=conf.biases[i]] for i in 0..3]
    ...
}
```
The number of values must match the declared size. The generated C++ header has the hyperparameters a module was
compiled with as a constexpr struct (`Model::hyperparameters`), with arrays as `std::array`.

## Compile-time conditionals and annotations
`cond ? a : b` chooses at compile time: `cond` must be a compile-time value (e.g. a hyperparameter), and only the chosen
branch is elaborated, so it may use members that only exist in that case.

Annotations (`@name` or `@name(args)`) attach information to a member. `@conditional(c)` makes a member exist only if the
compile-time condition `c` holds; otherwise the instance has no such member (no weight in the ABI), and using it is an
error:
```dlm
module Linear[indim: uint, outdim: uint, bias: bool = true] {
    W: float[outdim, indim]
    @conditional(bias)
    b: float[outdim]
    float[..., outdim] forward(x: float[..., indim]) => bias ? x @ W.T() + b : x @ W.T();
}
```
Unknown annotations are a warning (`-Wunknown-annotation`); `-Werror=unknown-annotation` makes them errors and
`-Wno-unknown-annotation` ignores them.

# Functions
```dlm
float[..., N] myfunc(x: float[..., N, M], y: float[M]) {
  auto z = x @ y;
  return z;
}
```

Shorthand:
```dlm
float[..., N] myfunc(x: float[..., N, M], y: float[M]) => x @ y;
```

Multiple return values (only from a block body, not with the `=>` shorthand):
```dlm
(float[...], float[...]) split(x: float[...]) {
  return x + 1, x * 2;
}

float[...] f(x: float[...]) {
  auto a, b = split(x);
  return a - b;
}
```

Sizes that a function only needs to name can be template parameters; they are inferred from the arguments' shapes at
each call:
```dlm
template<N: uint>
float[...] sum(x: float[..., N]) => __builtin_sum(x);
```

## Variables
`auto x = expr;` (or a type instead of `auto`) declares a variable; variables are immutable values, and every block
(function body, loop body) is a scope of its own. `x = expr;` rebinds a variable declared in an enclosing scope, e.g. to
carry state through a loop (see [Loops](#loops)). A tensor declared without initializer has slots that are assigned one
by one (see [Loops](#loops)).

## Signatures and `...`
The declared types of the arguments and return values define the signature of compiled functions. `...` stands for any
number of leading dimensions: within the language a function is rank-polymorphic (the rank of `...` is inferred per
call), and only a compiled signature fixes it. All `...` of an exported function denote the same runtime dimensions;
how many is chosen when compiling (`mdlc --ellipsis-rank N`, default 1). E.g., `float[..., 768] forward(x: float[...,
768])` compiled with `--ellipsis-rank 2` takes `[batch, seq, 768]`. `--ellipsis-rank NAME=K` gives an input only the
leading K of these dimensions (e.g. labels `[batch]` next to token ids `[batch, seq]`), and outputs get their exact
shapes from shape inference (e.g. a pooled `[batch, dim]`).

## Type checks
Types are checked when compiling: the shape of every expression follows from the rules of the builtins (NumPy-style
broadcasting and matrix products; the same rules the backends implement), and arguments that do not match the declared
parameter types, results that do not match the declared return types, and operands that do not fit (e.g. the inner
dimensions of `@`) are errors. Only definite mismatches count: `...` and sizes that are not known yet match anything.
```
error: Argument 1 of forward is float[..., 512], but the parameter x is float[..., 768]
```
Errors point to their location in the source, with the line and the marked expression.

Sizes that are different compile-time expressions, e.g. two hyperparameters, may be equal in one instance and differ in
another. Where equal sizes are needed, they are a warning (`-Wpossible-mismatch`), and an error in strict mode
(`mdlc --strict`, or `-Werror=possible-mismatch`):
```dlm
module Test[input: uint, output: uint] {
    W: float[input, output]
    b: float[output]
    float[output] forward(x: float[input]) => W @ x + b;
}
```
```
warning: Operator @: the inner dimensions of the matrix product may differ ([input, output] @ [input]) [-Wpossible-mismatch]
warning: Operator +: cannot broadcast [input] and [output] (the sizes may differ) [-Wpossible-mismatch]
```

# Loops
`for (i in lo..hi) { ... }` repeats its body for `i = lo, ..., hi - 1`; the bounds are compile-time integers, and the
loop is expanded when compiling (there is no loop at runtime). The body is a scope of its own; an assignment `h = ...`
rebinds a variable declared outside, which carries state from one iteration to the next:
```dlm
module Encoder[n: uint, d: uint] {
    layers: [Layer[d] for i in 0..n]          // an array of module instances
    float[..., d] forward(x: float[..., d]) {
        auto h = x;
        for (i in 0..n) { h = layers[i](h); }
        return h;
    }
}
```

A tensor declared without initializer has slots along its first dimension that are assigned one by one, e.g. to keep
the output of every layer. Its first dimension (the number of slots) is a compile-time size, and the indices are
compile-time integers (negative ones count from the end):
```dlm
float[n, ..., d] hiddenStates(x: float[..., d]) {
    float[n, ..., d] hs;
    auto h = x;
    for (i in 0..n) {
        h = layers[i](h);
        hs[i] = h;
    }
    return hs;                                 // [n, ..., d]
}
```
`hs[k]` is the value assigned to slot k (no copy); `hs` as a whole is the slots stacked. A slot must be assigned before
it is read, and all slots before `hs` is used as a whole (errors otherwise). The assigned values must have the slot type
(the declared type without its first dimension).

# Custom gradients
A member function named `gradient` is the gradient of `forward` (for another member function `f`: `f_gradient`). It
is an ordinary function; only the gradient pass (`mdlc --grad`) gives it this meaning. If it is defined, it is used
instead of differentiating `forward`; otherwise the gradient is derived automatically.

Signature: the arguments of `forward` followed by one cotangent per output of `forward`; the results are one
cotangent per argument followed by one gradient per weight of the module in declaration order (including the weights
of submodules, depth first). A mismatch is a compile error once the gradient is needed.
```dlm
module LayerNorm[dim: uint] {
    beta: float[dim]
    gamma: float[dim]
    float[..., dim] forward(x: float[..., dim]) { ... }
    (float[..., dim], float[dim], float[dim]) gradient(x: float[..., dim], dy: float[..., dim]) {
        auto mu = mean(x)[... -> ... 1];  // recomputed; shared with forward's identical node
        ...
        return dx, sum_like(dy, beta), sum_like(dy * xhat, gamma);
    }
}
```
`gradient` cannot see the local variables of `forward`; intermediates are recomputed, and identical subexpressions are
shared with the forward pass (the computegraph interns nodes), so this costs nothing. Results that are not needed
(frozen weights, inputs without gradient) are dropped. `std.autodiff` provides `sum_like`, `broadcast_like` and
`nograd` (stops gradients).

A free function `f` gets a custom gradient by a free function `f_gradient` with the same convention (no weights). Like
every name, `f_gradient` must be declared before it is used: it applies to the calls of `f` that follow it.
```dlm
float[...] softmax_gradient(x: float[...], dy: float[...]) {
    auto y = softmax(x);  // the forward pass's node
    return y * (dy - sum(dy * y));
}
```
Custom gradients that need only values the forward pass keeps anyway (like softmax's result) keep the forward pass of a
gradient build as fast as inference: it does not compute intermediates just to save them for the backward pass.
Calls whose computation the masking pass changed (e.g. `softmax` of a masked tensor) are differentiated automatically.

# Overloads and masked tensors
Functions can be overloaded on whether their tensor parameters are `masked`. A masked tensor (`masked float[..., D]`)
has dense values plus a mask of the valid elements in the last dimension of `...` (like torch.masked); operations on it
yield masked tensors. A call with a masked argument uses the `masked` overload if there is one; otherwise a plain
parameter takes the masked tensor, and the function is elaborated again for it, so that the calls in its body see the
masked type as well. A `masked` parameter does not take plain tensors.
```dlm
float[...] average(x: float[...]) => mean(x);
float[...] average(x: masked float[...]) => mean(x.data) / mean(x.mask);
```
`x.data` is x with zeros at the masked elements, `x.mask` is 1.0 at valid and 0.0 at masked elements (broadcastable to
x). A compiled function with masked parameters (or `mdlc --mask f`) takes an additional input `mask: bool[...]`; the
compiler inserts the masking where masked elements would mix with valid ones (reductions, softmax, contractions), and
the outputs are zero at masked positions.

# Types
## Atomic types
- `uint`, `int`, `bool`
- `float32`, `bfloat16`, `float16`
- `float`: the default float type, chosen when compiling (`mdlc --float f32|bf16|f16`, default f32)

## Tensors
`<atomic type>[<shape>]`, e.g. `float[10, 10, 32]` for a $10\times 10\times 32$ tensor or `int[...]` for token ids. A
dimension is a number, a compile-time expression (`conf.dims.model`, `heads * d`), or `...` (see
[Signatures and `...`](#signatures-and-)). `masked float[...]` is a masked tensor (see
[Overloads and masked tensors](#overloads-and-masked-tensors)). `int` tensors are int32 at runtime and get no gradients.

## Structs
Structs group hyperparameters; their members are atomics, arrays, or structs (also inline, `struct { ... }`):
```dlm
struct TransformerConf {
    dims: struct { model: uint  key: uint  value: uint  inner: uint }
    numEncoders: uint
    numAttnHeads: uint
}
```
In YAML, a struct value is a map (`conf: { dims: { model: 768, ... }, numEncoders: 12, numAttnHeads: 12 }`).

## Arrays
`array[T; n]`: n compile-time values of type T (hyperparameters, see [Hyperparameters](#hyperparameters)); indexed with
compile-time indices.

## Float types
Models are usually written with `float`, so that the same model compiles for float32, bfloat16, or float16; the explicit
types are not affected by `--float` (e.g. float32 weights in a bfloat16 model).

Conversions between float types are never implicit: narrowing loses precision, widening costs memory and time. Operands
of different float types, an argument of another float type than its parameter, or a result of another type than the
declared one are errors; a conversion is written with the type's name:
```dlm
float32[..., 4] f(x: float32[..., 4], w: bfloat16[4]) => x * float32(w);
```
Implicit conversions can be enabled like g++ diagnostics, separately for widening (bfloat16/float16 to float32) and
narrowing (float32 to 16 bits, or between bfloat16 and float16, which is lossy in both directions): `-Werror=<name>`
(the default) reports an error, `-W<name>` a warning and inserts the conversion, `-Wno-<name>` inserts it silently, with
the names `float-widening` and `float-narrowing`. Operands of different types are then both converted to the wider type
(float32 for bfloat16 with float16), arguments to their parameter's type, and results to the declared type.
```
warning: Operator *: implicit widening from bfloat16 to float32 [-Wfloat-widening]
```
With `--float bf16`, `float` *is* bfloat16, so `bfloat16` and `float` values combine without conversion. Literals have
no precision of their own: they take the type of the tensor they are combined with.

The 16-bit types are storage types: reductions and the vendor kernels (matrix products, softmax, layer norm) accumulate
in float32 internally, and float16 computes in float32 on CPUs without float16 arithmetic. Weights can additionally be
stored quantized (int8/int4); this is a build setting and does not change the model (see
[build-config.md](build-config.md#quantization)).

# Expressions
- Arithmetic: `a + b`, `a - b`, `a * b`, `a / b` (elementwise, NumPy-style broadcasting), `a @ b` (matrix product),
  `-a`. Addition is assumed by the optimizer to be commutative and associative.
- Transpose of the last two dimensions: `W.T()`.
- Indexing like NumPy: `h[..., 0, :]` (integers select and remove an axis, `:` keeps one, `...` stands for the axes
  that are not listed); these indices are compile-time integers. An integer tensor as the only index selects rows:
  `table[ids]` with `table: float[num, dim]` and `ids: int[...]` is `float[..., dim]` (e.g. an embedding lookup).
- Einops: `x[... (h d) -> h ... d | h=heads]` (see [einops.md](einops.md)).
- Compile-time conditional: `cond ? a : b`.
- Float conversion: `float32(x)`, `bfloat16(x)`, `float16(x)`, `float(x)`.
- Calls: `f(x)`, `instance(x)` (its `forward`), `instance.member(x)`.
- Members: `conf.dims.model` (struct), `x.data` / `x.mask` (masked tensor).
- Literals: `1`, `2.5`, `1e-12`, `true`, arrays `[8, 16, 4]`. Identifiers may contain digits (`layer1`).

# Standard library
`import std;` (directory `std/`):

| File | Contents |
|---|---|
| `arithmetic` | `exp`, `log`, `sqrt`, `rsqrt`, `erf`, `tanh`, `sin`, `cos`, `tan` |
| `activations` | `softmax` (also for masked tensors), `gelu`, `sigmoid`, `silu` (with custom gradients for softmax and GELU) |
| `statistics` | reductions over the last dimension `sum`, `max`, `min`, `mean`, `var`; `sum_all`, `mean_all` (to a scalar) |
| `nn` | modules `Linear[indim, outdim, .bias]`, `LayerNorm[dim]` (custom gradient), `Embedding[num, dim]` |
| `losses` | `mse`, `cross_entropy` (integer labels), `logsumexp`, `one_hot_like` |
| `autodiff` | `sum_like`, `broadcast_like`, `nograd` |

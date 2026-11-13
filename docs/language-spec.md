# Imports
```dlm
import std;              // all files of the standard library (directory std/)
import std.activations;  // a single file (std/activations.dlm)
import mylib.layers;     // mylib/layers.dlm, searched next to the importing file and in the -i directories
```
Imported definitions are visible in the importing file. Only the definitions of the compiled file itself are
exported; every file is translated once, even if it is imported several times.

# Modules
- Compiletime arguments ("hyperparameters"). E.g., the following is a linear layer with three hyperparameters: the input and output dimensions, and the boolean if a bias should be added.
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool] {
    /* ... */
  }
  ```
- Can contain member variables (typically used for "parameters", i.e., weights)
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool] {
    W: float[outdim, indim]
    b: float[outdim]
    /* ... */
  }
  ```
- Can contain member functions
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool] {
    W: float[outdim, indim]
    b: float[outdim]

    float[..., outdim] forward(x: float[..., indim]) => x @ W.T() + b;
  }
  ```
- Instantiation as a submodule (calling a submodule calls its `forward`):
  ```dlm
  module Block[dim: uint] {
    linear: Linear[dim, dim, true]
    float[..., dim] forward(x: float[..., dim]) => linear(x) + x;
  }
  ```
  Every instance has its own weights, identified by their path (e.g. `linear.W`, or `layers.3.attn.W` for nested
  modules); calling the same instance twice uses the same weights.

Planned: default values and named arguments (`Linear[indim, outdim, .bias=true]`), and switch expressions on
hyperparameters.

## Functions
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

Multiple return values:
```dlm
(float[...], float[...]) split(x: float[...]) {
  return x + 1, x * 2;
}

float[...] f(x: float[...]) {
  auto a, b = split(x);
  return a - b;
}
```

The declared types of the arguments and return values define the signature of compiled functions. All `...` in a
signature denote the same runtime dimensions; how many is chosen when compiling (`mdlc --ellipsis-rank N`, default 1).
E.g., `float[..., 768] forward(x: float[..., 768])` compiled with `--ellipsis-rank 2` takes `[batch, seq, 768]`.

Multiple return values can only be returned from a block body (`return a, b;`), not with the `=>` shorthand.

Types are checked when compiling: the shape of every expression follows from the rules of the builtins (NumPy-style
broadcasting and matrix products; the same rules the backends implement), and arguments that do not match the declared
parameter types, results that do not match the declared return types, and operands that do not fit (e.g. the inner
dimensions of `@`) are errors. Only definite mismatches count: `...` and sizes that are not known yet match anything.
```
error: Argument 1 of forward is float[..., 512], but the parameter x is float[..., 768]
```

Member functions of a submodule instance are called on the instance (`norm.gradient(x, dy)`) and use its weights.

## Custom gradients
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


## Overloads and masked tensors
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
x). A compiled function with masked parameters (or `mdlc --mask f`) takes an additional input `mask: bool[...]`.


# Structs
- Represent POD datatypes: structs contain members that are either themselves structs or atomics.


# Atomic Datatypes
- uint
- int
- bool
- float32, bfloat16, float16
- float: the default float type, chosen when compiling (`mdlc --float f32|bf16|f16`, default f32)

These datatypes can also be stored as tensors: `<datatype>[<shape>]`, e.g., `float[10, 10, 32]` for a $10\times 10\times 32$ tensor.

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
in float32 internally, and float16 computes in float32 on CPUs without float16 arithmetic.

# Expressions
- Arithmetic: `a + b`, `a - b`, `a * b`, `a / b` (elementwise, broadcasting), `a @ b` (matrix product), `-a`.
- Einops: `x[... (h d) -> h ... d | h=heads]` (see [einops.md](einops.md)).
- Literals: `1`, `2.5`, `1e-12`, `true`. Identifiers may contain digits (`layer1`).

# Operators
- Addition (+) is assumed by the optimizer to be commutative and associative

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
- float

These datatypes can also be stored as tensors: `<datatype>[<shape>]`, e.g., `float[10, 10, 32]` for a $10\times 10\times 32$ tensor.

# Expressions
- Arithmetic: `a + b`, `a - b`, `a * b`, `a / b` (elementwise, broadcasting), `a @ b` (matrix product), `-a`.
- Einops: `x[... (h d) -> h ... d | h=heads]` (see [einops.md](einops.md)).
- Literals: `1`, `2.5`, `1e-12`, `true`. Identifiers may contain digits (`layer1`).

# Operators
- Addition (+) is assumed by the optimizer to be commutative and associative

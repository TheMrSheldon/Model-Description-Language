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
signature currently denote the same (single) runtime dimension.


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

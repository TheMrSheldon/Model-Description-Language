# Modules
- Compiletime arguments ("hyperparameters"). E.g., the following is a linear layer with three hyperparameters: the input and output dimensions, and the boolean if a bias should be added.
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool = false] {
    /* ... */
  }
  ```
- Can contain member variables (typically used for "parameters")
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool = false] {
    W: float[outdim, indim]
    b: float[outdim]
    /* ... */
  }
  ```
- Can contain member functions
  ```dlm
  module Linear[indim: uint, outdim: uint, bias: bool = false] {
    W: float[outdim, indim]
    b: float[outdim]
    
    float[..., outdim] forward(x: float[..., indim]) =>
        bias?
            no => x @ W.T
            yes => x @ W.T + b;
  }
  ```
- Instantiation as a submodule:
  ```dlm
  module Module[indim: uint] {
    linear: Linear[indim, indim, .bias=true]
  }
  ```

## Functions
```dlm
float[..., N] myfunc(x: float[..., N, M], y: float[M]) {
  let z = x @ y;
  return z;
}
```

Shorthand:
```dlm
float[..., N, N] myfunc(x: float[..., N, M], y: float[M]) => x @ y;
```


# Structs
- Represent POD datatypes: structs contain members that are either themselves structs or atomics.


# Atomic Datatypes
- uint
- int
- bool
- float
- bfloat16
- float16
- float32
- float64

These datatypes can also be stored as tensors: `<datatype>[<shape>]`, e.g., `float[10, 10, 32]` for a $10\times 10\times 32$ tensor.

# Operators
- Addition (+) is assumed by the optimizer to be commutative and associative
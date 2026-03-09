# Using MDL inside a CMake Project

```cmake
enable_language(MDL)

# Build a library with a model definition
add_library(mylib
    model.dlm
)

# Or build it directly into your executable
add_executable(myexe
    model.dlm
)
```
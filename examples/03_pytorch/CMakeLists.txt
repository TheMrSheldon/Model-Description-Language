enable_language(MDL)

add_library(03_pytorch SHARED
    model.mdl
)

# This is usually not required but since we are also responsible for compiling the compiler here, we will add it as a dependency
add_dependencies(03_pytorch mdlc)
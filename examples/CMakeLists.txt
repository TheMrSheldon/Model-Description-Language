# The examples double as tests of the compiler: check its shape inference against the backend while compiling them.
set(MDL_VERIFY_SHAPES ON)
add_subdirectory(01_ffn)
add_subdirectory(02_transformer)
add_subdirectory(03_pytorch)
add_subdirectory(04_gradients)
add_subdirectory(05_classification)
add_subdirectory(06_masking)

enable_language(MDL)

add_executable(01_ffn
    main.cpp
    model.mdl
)

target_mdl_argvals(01_ffn ${CMAKE_CURRENT_LIST_DIR}/model.yml)

# This is usually not required but since we are also responsible for compiling the compiler here, we will add it as a dependency
add_dependencies(01_ffn mdlc)

# Link against TVM FFI to communicate with the binary
target_link_libraries(01_ffn PUBLIC tvm_ffi_shared)

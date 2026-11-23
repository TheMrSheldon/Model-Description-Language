# Libraries for programs that call compiled models (not linked into the compiler).

# ── mdl_runtime ──────────────────────────────────────────────────────────────
# Header-only helpers (include/mdl/runtime.hpp) plus the TVM runtime that compiled models depend on (thread pool,
# workspace allocation, FFI); nlohmann/json parses the safetensors headers.
add_library(mdl_runtime INTERFACE)
target_include_directories(mdl_runtime INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_runtime INTERFACE cxx_std_20)
target_link_libraries(mdl_runtime INTERFACE tvm_runtime nlohmann_json::nlohmann_json)

# ── mdl_profile ──────────────────────────────────────────────────────────────
# The profiling hooks that models compiled with `mdlc --profile` call (mdl/runtime/profile.h).
add_library(mdl_profile STATIC kernels/profile.cpp)
target_include_directories(mdl_profile PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_profile PUBLIC cxx_std_20)
set_property(TARGET mdl_profile PROPERTY POSITION_INDEPENDENT_CODE ON)

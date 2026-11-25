# Libraries for programs that call compiled models (not linked into the compiler).

# ── mdl_runtime ──────────────────────────────────────────────────────────────
# Header-only helpers (include/mdl/runtime.hpp) plus the TVM runtime that compiled models depend on (thread pool,
# workspace allocation, FFI); nlohmann/json parses the safetensors headers.
add_library(mdl_runtime INTERFACE)
target_include_directories(mdl_runtime INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_runtime INTERFACE cxx_std_20)
target_link_libraries(mdl_runtime INTERFACE tvm_runtime nlohmann_json::nlohmann_json)

# ── mdl_target_<kind> ────────────────────────────────────────────────────────
# The runtime of a compilation target besides the CPU (mdl_add_model(... TARGET <kind> ...) links it): CUDA needs TVM's
# CUDA runtime (device API, loading the embedded PTX).
if(MDL_ENABLE_CUDA)
	add_library(mdl_target_cuda INTERFACE)
	# (It only registers itself when loaded, no symbols are referenced: keep it despite --as-needed.)
	if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
		target_link_libraries(mdl_target_cuda INTERFACE "-Wl,--push-state,--no-as-needed" tvm_runtime_cuda "-Wl,--pop-state")
	else()
		target_link_libraries(mdl_target_cuda INTERFACE tvm_runtime_cuda)
	endif()
	# TVM's runtime API (tvm/runtime/tensor.h, device_api.h) to allocate and copy device memory.
	target_include_directories(mdl_target_cuda INTERFACE ${tvm_SOURCE_DIR}/include)
endif()

# ── mdl_profile ──────────────────────────────────────────────────────────────
# The profiling hooks that models compiled with `mdlc --profile` call (mdl/runtime/profile.h).
add_library(mdl_profile STATIC kernels/profile.cpp)
target_include_directories(mdl_profile PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_profile PUBLIC cxx_std_20)
set_property(TARGET mdl_profile PROPERTY POSITION_INDEPENDENT_CODE ON)

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
	# The CUDA sessions (a stream and CUDA graphs per session, see docs/technical/runtime.md): registers TVM FFI
	# functions when loaded, which the bindings look up.
	find_package(CUDAToolkit REQUIRED)
	add_library(mdl_session_cuda SHARED sessions/cuda.cpp)
	target_compile_features(mdl_session_cuda PRIVATE cxx_std_20)
	target_link_libraries(mdl_session_cuda PRIVATE tvm_ffi_header tvm_runtime_cuda CUDA::cudart)

	add_library(mdl_target_cuda INTERFACE)
	# (Both only register themselves when loaded, no symbols are referenced: keep them despite --as-needed. One group:
	# CMake drops a repeated flag.)
	if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
		target_link_libraries(mdl_target_cuda INTERFACE
			"-Wl,--push-state,--no-as-needed" tvm_runtime_cuda mdl_session_cuda "-Wl,--pop-state")
	else()
		target_link_libraries(mdl_target_cuda INTERFACE tvm_runtime_cuda mdl_session_cuda)
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

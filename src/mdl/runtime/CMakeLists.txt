# Libraries for programs that call compiled models (not linked into the compiler).

# ── mdl_runtime ──────────────────────────────────────────────────────────────
# Header-only helpers (include/mdl/runtime.hpp) plus the TVM runtime that compiled models depend on (thread pool,
# workspace allocation, FFI); nlohmann/json parses the safetensors headers.
add_library(mdl_runtime INTERFACE)
target_include_directories(mdl_runtime INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_runtime INTERFACE cxx_std_20)
target_link_libraries(mdl_runtime INTERFACE tvm_runtime nlohmann_json::nlohmann_json)

# ── mdl_profile ──────────────────────────────────────────────────────────────
# The profiling hooks that models compiled with `mdlc --profile` call (mdl_profile_begin/end, kernels.h).
add_library(mdl_profile STATIC kernels/profile.cpp)
target_include_directories(mdl_profile PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_profile PUBLIC cxx_std_20)
set_property(TARGET mdl_profile PROPERTY POSITION_INDEPENDENT_CODE ON)

# ── mdl_kernels ──────────────────────────────────────────────────────────────
# The oneDNN kernels (include/mdl/runtime/kernels.h): the library of the kernel package kernels/onednn.
option(MDL_ENABLE_ONEDNN "Build the oneDNN kernels (mdl_kernels, the kernel package onednn)" ON)
if(MDL_ENABLE_ONEDNN)
	add_library(mdl_kernels STATIC kernels/onednn.cpp kernels/layer_norm_backward.cpp)
	target_include_directories(mdl_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
	target_compile_features(mdl_kernels PUBLIC cxx_std_20)
	set_property(TARGET mdl_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
	# Only build the primitive kinds that mdl_kernels uses (forward and backward) to keep the build short.
	set(DNNL_ENABLE_PRIMITIVE "MATMUL;SOFTMAX;ELTWISE;LAYER_NORMALIZATION;REORDER" CACHE STRING "" FORCE)
	CPMAddPackage(
		NAME oneDNN
		GITHUB_REPOSITORY uxlfoundation/oneDNN
		GIT_TAG v3.13.4
		OPTIONS
			"DNNL_LIBRARY_TYPE STATIC"
			"DNNL_CPU_RUNTIME OMP"
			"DNNL_BUILD_TESTS OFF"
			"DNNL_BUILD_EXAMPLES OFF"
			"ONEDNN_BUILD_GRAPH OFF"
			"DNNL_ENABLE_JIT_PROFILING OFF"
		EXCLUDE_FROM_ALL YES
	)
	# The static oneDNN library uses OpenMP (DNNL_CPU_RUNTIME OMP) but does not propagate it to consumers.
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	target_link_libraries(mdl_kernels PUBLIC dnnl OpenMP::OpenMP_CXX)
endif()

# Libraries for programs that call compiled models (not linked into the compiler).

# ── mdl_runtime ──────────────────────────────────────────────────────────────
# Header-only helpers (include/mdl/runtime.hpp) plus the TVM runtime that compiled models depend on (thread pool,
# workspace allocation, FFI); nlohmann/json parses the safetensors headers.
add_library(mdl_runtime INTERFACE)
target_include_directories(mdl_runtime INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_runtime INTERFACE cxx_std_20)
target_link_libraries(mdl_runtime INTERFACE tvm_runtime nlohmann_json::nlohmann_json)

# ── mdl_kernels ──────────────────────────────────────────────────────────────
# The extern kernels that the TVM backend emits calls to (see include/mdl/runtime/kernels.h).
option(MDL_ENABLE_ONEDNN "Build the oneDNN-based vendor kernels (mdl_kernels)" ON)
add_library(mdl_kernels STATIC kernels/profile.cpp)
target_include_directories(mdl_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mdl_kernels PUBLIC cxx_std_20)
set_property(TARGET mdl_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
if(MDL_ENABLE_ONEDNN)
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
	target_sources(mdl_kernels PRIVATE kernels/onednn.cpp)
	# The static oneDNN library uses OpenMP (DNNL_CPU_RUNTIME OMP) but does not propagate it to consumers.
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	target_link_libraries(mdl_kernels PUBLIC dnnl OpenMP::OpenMP_CXX)
	target_compile_definitions(mdl_kernels PUBLIC MDL_KERNELS_ONEDNN)
else()
	# Without vendor kernels the backend lowers everything to TVM-generated kernels; the library stays empty.
	target_sources(mdl_kernels PRIVATE kernels/none.cpp)
endif()

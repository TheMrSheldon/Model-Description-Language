# The kernel package onednn: oneDNN kernels behind a C ABI (include/mdl_onednn.h), packaged by package.yml.in.

option(MDL_ENABLE_ONEDNN "Build the oneDNN kernels (the kernel package onednn)" ON)
if(NOT MDL_ENABLE_ONEDNN)
	return()
endif()

# Only build the primitive kinds that the kernels use (forward and backward) to keep the build short.
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

# (libdnnl.a is installed with the compiler: its license, and that of the software that it contains, go with it.)
mdl_third_party(oneDNN COMPONENTS compiler VERSION 3.13.4 SPDX Apache-2.0 URL https://github.com/uxlfoundation/oneDNN
	LICENSE_FILES "${oneDNN_SOURCE_DIR}/LICENSE" "${oneDNN_SOURCE_DIR}/THIRD-PARTY-PROGRAMS")

add_library(mdl_onednn STATIC src/onednn.cpp src/layer_norm_backward.cpp)
target_include_directories(mdl_onednn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mdl_onednn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include) # (mdl_strided.hpp)
target_compile_features(mdl_onednn PUBLIC cxx_std_20)
set_property(TARGET mdl_onednn PROPERTY POSITION_INDEPENDENT_CODE ON)
# The static oneDNN library uses OpenMP (DNNL_CPU_RUNTIME OMP) but does not propagate it to consumers.
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(mdl_onednn PUBLIC dnnl OpenMP::OpenMP_CXX)

# In the build tree, the manifest refers to the built libraries; installed, the package is share/mdl/kernels/onednn
# with the libraries in its lib directory.
mdl_kernel_package(onednn MANIFEST package.yml.in TARGETS mdl_onednn)
file(READ "${CMAKE_CURRENT_BINARY_DIR}/onednn.package.yml.in" manifest)
string(REGEX REPLACE "\\$<TARGET_FILE:([a-z_]+)>" "lib/$<TARGET_FILE_NAME:\\1>" manifest "${manifest}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/onednn.install.yml" CONTENT "${manifest}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/onednn.install.yml" DESTINATION share/mdl/kernels/onednn
	RENAME package.yml COMPONENT compiler)
install(FILES $<TARGET_FILE:mdl_onednn> $<TARGET_FILE:dnnl> DESTINATION share/mdl/kernels/onednn/lib COMPONENT compiler)

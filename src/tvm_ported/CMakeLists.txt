# Parts of TVM's Python package translated to C++ (see README.md): the default GPU schedules (DLight), the CUDA tensor
# intrinsics, and the callbacks of TVM's CUDA C code generation. Depends on TVM and the CUDA toolkit (NVRTC) only.
add_library(mdl_tvm_ported STATIC
	backend/cuda/codegen/header.cpp
	backend/cuda/codegen/registry.cpp
	s_tir/dlight/analysis/common_analysis.cpp
	s_tir/dlight/base/common_schedules.cpp
	s_tir/dlight/base/transform.cpp
	s_tir/dlight/base/utils.cpp
	s_tir/dlight/cpu/reduction.cpp
	s_tir/dlight/gpu/fallback.cpp
	s_tir/dlight/gpu/general_reduction.cpp
	s_tir/dlight/gpu/matmul.cpp
	s_tir/dlight/gpu/reduction.cpp
	s_tir/dlight/gpu/transpose.cpp
	s_tir/tensor_intrin/cuda.cpp
)
target_compile_features(mdl_tvm_ported PUBLIC cxx_std_23)
target_link_libraries(mdl_tvm_ported PUBLIC tvm_compiler)
# The CUDA C compilation with NVRTC and the registration of the callbacks: only with the CUDA target.
if(MDL_ENABLE_CUDA)
	find_package(CUDAToolkit REQUIRED)
	target_sources(mdl_tvm_ported PRIVATE globals.cpp support/nvcc.cpp)
	target_link_libraries(mdl_tvm_ported PRIVATE CUDA::nvrtc)
endif()
# tvm_compiler uses directory-scoped include_directories internally, which do not propagate to consumers.
target_include_directories(mdl_tvm_ported PUBLIC ${tvm_SOURCE_DIR}/include ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include)
# Users include the headers as <tvm_ported/...>.
target_include_directories(mdl_tvm_ported INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(TARGET mdl_tvm_ported PROPERTY POSITION_INDEPENDENT_CODE ON)

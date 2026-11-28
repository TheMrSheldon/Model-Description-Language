# Parts of TVM's Python package translated to C++ (see README.md): the default GPU schedules (DLight), the CUDA tensor
# intrinsics, and the callbacks of TVM's CUDA C code generation. Depends on TVM and the CUDA toolkit's headers (NVRTC is loaded at run time, see support/nvcc.cpp).
find_package(CUDAToolkit REQUIRED)
add_library(mdl_tvm_ported STATIC
	globals.cpp
	backend/cuda/codegen/header.cpp
	backend/cuda/codegen/registry.cpp
	s_tir/dlight/analysis/common_analysis.cpp
	s_tir/dlight/base/common_schedules.cpp
	s_tir/dlight/base/transform.cpp
	s_tir/dlight/base/utils.cpp
	s_tir/dlight/gpu/fallback.cpp
	s_tir/dlight/gpu/general_reduction.cpp
	s_tir/dlight/gpu/matmul.cpp
	s_tir/dlight/gpu/reduction.cpp
	s_tir/dlight/gpu/transpose.cpp
	s_tir/tensor_intrin/cuda.cpp
	support/nvcc.cpp
)
target_compile_features(mdl_tvm_ported PUBLIC cxx_std_23)
# (CUDA::toolkit: the headers only; NVRTC is not linked so that mdlc starts without the toolkit.)
target_link_libraries(mdl_tvm_ported PUBLIC tvm_compiler PRIVATE CUDA::toolkit ${CMAKE_DL_LIBS})
# tvm_compiler uses directory-scoped include_directories internally, which do not propagate to consumers.
target_include_directories(mdl_tvm_ported PUBLIC ${tvm_SOURCE_DIR}/include ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include)
# Users include the headers as <tvm_ported/...>.
target_include_directories(mdl_tvm_ported INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(TARGET mdl_tvm_ported PROPERTY POSITION_INDEPENDENT_CODE ON)

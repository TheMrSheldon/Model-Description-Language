# The kernel package cublas: cuBLAS kernels behind a C ABI (include/mdl_cublas.h), packaged by package.yml.in. Only
# selected for CUDA targets (the artifact has a device-arch).

if(NOT MDL_ENABLE_CUDA)
	return()
endif()
option(MDL_ENABLE_CUBLAS "Build the cuBLAS kernels (the kernel package cublas)" ON)
if(NOT MDL_ENABLE_CUBLAS)
	return()
endif()

# The GPU architectures of the library's own kernels (the epilogue), plus PTX of the newest for later GPUs.
set(MDL_CUBLAS_ARCHITECTURES "70-real;75-real;80-real;86-real;89-real;90" CACHE STRING
	"CUDA architectures of the cublas kernel package")
list(GET MDL_CUBLAS_ARCHITECTURES 0 _oldest)
string(REGEX REPLACE "-.*" "" _oldest "${_oldest}")
set(MDL_CUBLAS_DEVICE_ARCH "sm_${_oldest}")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(MDL_CUBLAS_VERSION "${CUDAToolkit_VERSION}")

add_library(mdl_cublas STATIC src/cublas.cu)
target_include_directories(mdl_cublas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mdl_cublas PUBLIC cxx_std_17)
target_compile_options(mdl_cublas PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
set_target_properties(mdl_cublas PROPERTIES CUDA_ARCHITECTURES "${MDL_CUBLAS_ARCHITECTURES}"
	POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF)
# TVMFFIEnvGetStream (the stream of the model's other kernels): the TVM FFI that compiled models link anyway.
target_link_libraries(mdl_cublas PUBLIC CUDA::cublasLt CUDA::cublas CUDA::cudart PRIVATE tvm_ffi_header)

# In the build tree, the manifest refers to the built library; installed, the package is share/mdl/kernels/cublas
# with the library in its lib directory.
mdl_kernel_package(cublas MANIFEST package.yml.in TARGETS mdl_cublas)
file(READ "${CMAKE_CURRENT_BINARY_DIR}/cublas.package.yml.in" manifest)
string(REGEX REPLACE "\\$<TARGET_FILE:([a-z_]+)>" "lib/$<TARGET_FILE_NAME:\\1>" manifest "${manifest}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cublas.install.yml" CONTENT "${manifest}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cublas.install.yml" DESTINATION share/mdl/kernels/cublas
	RENAME package.yml COMPONENT kernels)
install(FILES $<TARGET_FILE:mdl_cublas> DESTINATION share/mdl/kernels/cublas/lib COMPONENT kernels)

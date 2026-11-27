# The kernel package cudnn: cuDNN kernels behind a C ABI (include/mdl_cudnn.h), packaged by package.yml.in. Only selected
# for CUDA targets (the artifact has a device-arch). Needs cuDNN 9 (CUDNN_ROOT, e.g. the pip package nvidia-cudnn-cu12)
# and the header-only cuDNN frontend (fetched).

if(NOT MDL_ENABLE_CUDA)
	return()
endif()
option(MDL_ENABLE_CUDNN "Build the cuDNN kernels (the kernel package cudnn)" ON)
if(NOT MDL_ENABLE_CUDNN)
	return()
endif()
find_path(MDL_CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} PATH_SUFFIXES include)
find_library(MDL_CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} PATH_SUFFIXES lib lib64)
if(NOT MDL_CUDNN_INCLUDE_DIR OR NOT MDL_CUDNN_LIBRARY)
	message(STATUS "[mdl] cuDNN not found (set CUDNN_ROOT): the kernel package cudnn is not built")
	return()
endif()
file(STRINGS "${MDL_CUDNN_INCLUDE_DIR}/cudnn_version.h" _v REGEX "define CUDNN_(MAJOR|MINOR|PATCHLEVEL) ")
string(REGEX REPLACE ".*MAJOR ([0-9]+).*MINOR ([0-9]+).*PATCHLEVEL ([0-9]+).*" "\\1.\\2.\\3" MDL_CUDNN_VERSION "${_v}")
message(STATUS "[mdl] cuDNN ${MDL_CUDNN_VERSION}: ${MDL_CUDNN_LIBRARY}")

CPMAddPackage(NAME cudnn_frontend GITHUB_REPOSITORY NVIDIA/cudnn-frontend GIT_TAG v1.12.0 DOWNLOAD_ONLY YES)
mdl_third_party(cudnn-frontend COMPONENTS cuda VERSION 1.12.0 SPDX MIT URL https://github.com/NVIDIA/cudnn-frontend
	LICENSE_FILES "${cudnn_frontend_SOURCE_DIR}/LICENSE*")
find_package(CUDAToolkit REQUIRED)

add_library(mdl_cudnn STATIC src/attention.cpp)
target_include_directories(mdl_cudnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mdl_cudnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include) # (mdl_strided.hpp)
target_include_directories(mdl_cudnn PRIVATE ${cudnn_frontend_SOURCE_DIR}/include ${MDL_CUDNN_INCLUDE_DIR})
target_compile_features(mdl_cudnn PUBLIC cxx_std_20)
set_property(TARGET mdl_cudnn PROPERTY POSITION_INDEPENDENT_CODE ON)
# TVMFFIEnvGetStream (the stream of the model's other kernels): the TVM FFI that compiled models link anyway.
target_link_libraries(mdl_cudnn PUBLIC ${MDL_CUDNN_LIBRARY} CUDA::cudart PRIVATE tvm_ffi_header)

# In the build tree, the manifest refers to the built library; installed, the package is share/mdl/kernels/cudnn with
# the library in its lib directory.
mdl_kernel_package(cudnn MANIFEST package.yml.in TARGETS mdl_cudnn)
file(READ "${CMAKE_CURRENT_BINARY_DIR}/cudnn.package.yml.in" manifest)
string(REGEX REPLACE "\\$<TARGET_FILE:([a-z_]+)>" "lib/$<TARGET_FILE_NAME:\\1>" manifest "${manifest}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cudnn.install.yml" CONTENT "${manifest}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cudnn.install.yml" DESTINATION share/mdl/kernels/cudnn
	RENAME package.yml COMPONENT cuda)
install(FILES $<TARGET_FILE:mdl_cudnn> DESTINATION share/mdl/kernels/cudnn/lib COMPONENT cuda)

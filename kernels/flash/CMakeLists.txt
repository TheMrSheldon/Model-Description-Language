# The kernel package flash: FlashAttention-2 (github.com/Dao-AILab/flash-attention, BSD-3-Clause) behind the C ABI of the
# cuDNN package's attention (include/mdl_flash.h), packaged by package.yml.in. Only selected for CUDA targets. Its
# sources are fetched; only the instantiations we call are compiled (head dimensions 32, 64, 128; bfloat16 and float16;
# not causal), for sm_80 (and newer through PTX). FlashAttention's PyTorch includes are replaced by shim/.

if(NOT MDL_ENABLE_CUDA)
	return()
endif()
option(MDL_ENABLE_FLASH "Build the FlashAttention-2 kernels (the kernel package flash; a long build)" ON)
if(NOT MDL_ENABLE_FLASH)
	return()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
# (pinned by tag; CUTLASS is FlashAttention's submodule at the commit it pins)
CPMAddPackage(NAME flash_attention GITHUB_REPOSITORY Dao-AILab/flash-attention GIT_TAG v2.7.4.post1 DOWNLOAD_ONLY YES
	GIT_SUBMODULES "csrc/cutlass")
set(MDL_FLASH_VERSION "2.7.4")
set(_src "${flash_attention_SOURCE_DIR}/csrc/flash_attn/src")

set(_instantiations)
foreach(dim IN ITEMS 32 64 128)
	foreach(type IN ITEMS bf16 fp16)
		list(APPEND _instantiations "${_src}/flash_fwd_hdim${dim}_${type}_sm80.cu" "${_src}/flash_bwd_hdim${dim}_${type}_sm80.cu")
	endforeach()
endforeach()

add_library(mdl_flash STATIC src/flash.cu ${_instantiations})
target_include_directories(mdl_flash PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mdl_flash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shim ${_src}
	${flash_attention_SOURCE_DIR}/csrc/cutlass/include)
target_compile_features(mdl_flash PUBLIC cxx_std_17)
# (FlashAttention's build flags, without the features we do not use)
target_compile_definitions(mdl_flash PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASHATTENTION_DISABLE_ALIBI
	FLASHATTENTION_DISABLE_SOFTCAP FLASHATTENTION_DISABLE_LOCAL FLASHATTENTION_DISABLE_UNEVEN_K)
target_compile_options(mdl_flash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -U__CUDA_NO_HALF_OPERATORS__
	-U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
	-Wno-deprecated-gpu-targets>)
set_target_properties(mdl_flash PROPERTIES CUDA_ARCHITECTURES "80" POSITION_INDEPENDENT_CODE ON
	CUDA_SEPARABLE_COMPILATION OFF)
# TVMFFIEnvGetStream (the stream of the model's other kernels): the TVM FFI that compiled models link anyway.
target_link_libraries(mdl_flash PUBLIC CUDA::cudart PRIVATE tvm_ffi_header)

# In the build tree, the manifest refers to the built library; installed, the package is share/mdl/kernels/flash with
# the library in its lib directory.
mdl_kernel_package(flash MANIFEST package.yml.in TARGETS mdl_flash)
file(READ "${CMAKE_CURRENT_BINARY_DIR}/flash.package.yml.in" manifest)
string(REGEX REPLACE "\\$<TARGET_FILE:([a-z_]+)>" "lib/$<TARGET_FILE_NAME:\\1>" manifest "${manifest}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/flash.install.yml" CONTENT "${manifest}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/flash.install.yml" DESTINATION share/mdl/kernels/flash
	RENAME package.yml COMPONENT kernels)
install(FILES $<TARGET_FILE:mdl_flash> DESTINATION share/mdl/kernels/flash/lib COMPONENT kernels)

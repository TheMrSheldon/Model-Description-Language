# The CUDA target (MDL_ENABLE_CUDA): models compiled with --target cuda, called with device tensors. The tests that run
# compiled code need a GPU of architecture MDL_CUDA_ARCH or newer.
set(_target "cuda -arch=${MDL_CUDA_ARCH}")
set(_examples "${PROJECT_SOURCE_DIR}/examples")

# The FFN of examples/01_ffn, unchanged: the model does not depend on the target.
mdl_add_model(cuda_ffn_model SOURCE ${_examples}/01_ffn/model.dlm CONFIG ${_examples}/01_ffn/model.yml TARGET "${_target}")
add_executable(cuda_ffn ffn.cpp)
target_link_libraries(cuda_ffn PRIVATE cuda_ffn_model)
add_test(NAME cuda_ffn COMMAND cuda_ffn)

# Targets: an unknown kind is reported with the available ones; cuda needs an architecture.
add_test(NAME mdlc_target_unknown COMMAND mdlc ${_examples}/01_ffn/model.dlm -a ${_examples}/01_ffn/model.yml --target foo -o unused.o)
set_tests_properties(mdlc_target_unknown PROPERTIES WILL_FAIL TRUE)
add_test(NAME mdlc_target_unknown_message COMMAND mdlc ${_examples}/01_ffn/model.dlm -a ${_examples}/01_ffn/model.yml --target foo -o unused.o)
set_tests_properties(mdlc_target_unknown_message PROPERTIES PASS_REGULAR_EXPRESSION "target 'foo' not found \\(available: cuda, llvm\\)")
add_test(NAME mdlc_target_cuda_arch COMMAND mdlc ${_examples}/01_ffn/model.dlm -a ${_examples}/01_ffn/model.yml --target cuda -o unused.o)
set_tests_properties(mdlc_target_cuda_arch PROPERTIES PASS_REGULAR_EXPRESSION "the GPU architecture is required")

# Kernel packages: CPU artifacts (oneDNN) are never selected for the CUDA target, but are for the CPU.
if(TARGET mdl_kernel_package_onednn)
	set(_bert ${_examples}/02_transformer)
	add_test(NAME cuda_no_cpu_kernels COMMAND mdl-kernels --kernels-dir "${MDL_KERNELS_BUILD_DIR}" check
		${_bert}/model.dlm -a ${_bert}/model.yml --kernel-package onednn --target "${_target}")
	set_tests_properties(cuda_no_cpu_kernels PROPERTIES PASS_REGULAR_EXPRESSION " 0 selected")
	add_test(NAME cpu_kernels COMMAND mdl-kernels --kernels-dir "${MDL_KERNELS_BUILD_DIR}" check
		${_bert}/model.dlm -a ${_bert}/model.yml --kernel-package onednn)
	set_tests_properties(cpu_kernels PROPERTIES FAIL_REGULAR_EXPRESSION " 0 selected")
endif()

# BERT (examples/02_transformer, unchanged) on the GPU against Hugging Face's BertModel: float32, bfloat16, and padded
# sequences with a mask (bert_check.cpp with MDL_BERT_DEVICE).
set(_bert_dir ${_examples}/02_transformer)
foreach(variant IN ITEMS "f32;;1e-4;" "bf16;bfloat16;2e-2;" "masked;;1e-4;--mask;forward;--mask;pooled")
	list(GET variant 0 name)
	list(GET variant 1 type)
	list(GET variant 2 tolerance)
	list(SUBLIST variant 3 -1 extra)
	set(options --ellipsis-rank 2 ${extra})
	set(reference ${_bert_dir}/data/bert_tiny.safetensors)
	if(name STREQUAL "bf16")
		list(APPEND options --float bf16)
	elseif(name STREQUAL "masked")
		set(reference ${_bert_dir}/data/bert_tiny_padded.safetensors)
	endif()
	mdl_add_model(cuda_bert_${name}_model SOURCE ${_bert_dir}/model.dlm CONFIG ${_bert_dir}/model_bert_tiny.yml
		TARGET "${_target}" OPTIONS ${options})
	add_executable(cuda_bert_${name} ${_bert_dir}/bert_check.cpp)
	target_link_libraries(cuda_bert_${name} PRIVATE cuda_bert_${name}_model)
	target_compile_definitions(cuda_bert_${name} PRIVATE MDL_BERT_DEVICE=kDLCUDA MDL_BERT_REFERENCE="${reference}"
		MDL_BERT_TOLERANCE=${tolerance})
	if(type)
		target_compile_definitions(cuda_bert_${name} PRIVATE MDL_BERT_FLOAT=${type})
	endif()
	if(name STREQUAL "masked")
		target_compile_definitions(cuda_bert_${name} PRIVATE MDL_BERT_MASKED)
	endif()
	add_test(NAME cuda_bert_${name} COMMAND cuda_bert_${name})
endforeach()

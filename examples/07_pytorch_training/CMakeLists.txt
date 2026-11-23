# Trains the tiny BERT classifier of 02_transformer from PyTorch: a masked gradient build as a shared library whose
# generated bindings (model_pytorch.py) wrap `logits` + `logits_backward` in a torch.autograd.Function.
#   python train.py <build dir>/examples/07_pytorch_training
mdl_add_model(07_pytorch_training SOURCE ../02_transformer/model.dlm CONFIG ../02_transformer/model_cls_tiny.yml
	BUILD autograd SHARED BINDINGS python-torch)

# The test runs only with a Python that has torch, transformers and tvm_ffi (e.g. -DMDL_PYTHON=<venv>/bin/python).
find_package(Python3 COMPONENTS Interpreter QUIET)
set(MDL_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python for the PyTorch examples")
if(MDL_PYTHON)
	execute_process(COMMAND "${MDL_PYTHON}" -c "import torch, transformers, tvm_ffi"
		RESULT_VARIABLE _missing OUTPUT_QUIET ERROR_QUIET)
	if(_missing EQUAL 0)
		add_test(NAME example_07_pytorch_training
			COMMAND "${MDL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/train.py" "${CMAKE_CURRENT_BINARY_DIR}")
	else()
		message(STATUS "example_07_pytorch_training: ${MDL_PYTHON} lacks torch/transformers/tvm_ffi, test disabled")
	endif()
endif()

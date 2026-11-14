# An MLP with quantized weight matrices (int8 and int4, see model.yml) against a reference computed from the same
# quantized weights; the weights are quantized from float32 weights while loading. Each entry: build;element
# type;tolerance (16-bit activations are rounded).
foreach(variant IN ITEMS "int8;float;1e-4" "int8-grouped;float;1e-4" "int4;float;1e-4" "int4-asym;float;1e-4"
		"mixed;float;1e-4" "int4-bf16;mdl::runtime::bfloat16;5e-2" "int8-f16;mdl::runtime::float16;1e-2")
	list(GET variant 0 build)
	list(GET variant 1 element)
	list(GET variant 2 tolerance)
	string(REPLACE "-" "_" name ${build})
	mdl_add_model(quantized_${name}_model SOURCE model.dlm CONFIG model.yml BUILD ${build})
	add_executable(09_quantization_${name} main.cpp)
	target_link_libraries(09_quantization_${name} PRIVATE quantized_${name}_model)
	target_compile_definitions(09_quantization_${name} PRIVATE MDL_ELEMENT=${element} MDL_TOLERANCE=${tolerance})
	add_test(NAME example_09_quantization_${name} COMMAND 09_quantization_${name})
endforeach()

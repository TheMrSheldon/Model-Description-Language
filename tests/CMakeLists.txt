include(CTest)

CPMAddPackage(URI "gh:catchorg/Catch2@3.8.0" EXCLUDE_FROM_ALL YES)

add_executable(tests
	# Add test files here
    mdl/computegraph/function.cpp
    mdl/computegraph/object.cpp
    mdl/computegraph/einops.cpp
    mdl/computegraph/graph.cpp
    mdl/parser/parser.cpp
    mdl/parser/overloads.cpp
    mdl/parser/imports.cpp
    mdl/backends/tvm_patterns.cpp
    mdl/utils/pathpattern.cpp
    mdl/passes/autodiff.cpp
    mdl/passes/shapes.cpp
    mdl/passes/masking.cpp
    mdl/runtime/safetensors.cpp
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain mdl mdl_runtime)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_features(tests PRIVATE cxx_std_23)

add_test(NAME unit_tests COMMAND tests
	#WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/testfiles
)
# The build settings of argument-value files (mdlc --arg-vals, see docs/build-config.md): the command line takes
# precedence, unknown keys and builds are errors.
set(_cfg "${CMAKE_CURRENT_LIST_DIR}/configs")
set(_out "${CMAKE_CURRENT_BINARY_DIR}/configs")
add_test(NAME mdlc_config_applied COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/override.yml" -o "${_out}/a.o")
set_tests_properties(mdlc_config_applied PROPERTIES WILL_FAIL TRUE)
add_test(NAME mdlc_config_overridden COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/override.yml" --float f32 -o "${_out}/b.o")
add_test(NAME mdlc_config_unknown_key COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/typo.yml" -o "${_out}/c.o")
set_tests_properties(mdlc_config_unknown_key PROPERTIES WILL_FAIL TRUE)
add_test(NAME mdlc_config_build COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/builds.yml" --build train -o "${_out}/d.o")
add_test(NAME mdlc_config_unknown_build COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/builds.yml" --build fast -o "${_out}/e.o")
set_tests_properties(mdlc_config_unknown_build PROPERTIES WILL_FAIL TRUE)
# Quantization (--quantize, `quantize:`): quantized weights are frozen; invalid group sizes are errors.
add_test(NAME mdlc_quantize COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" -o "${_out}/q1.o")
add_test(NAME mdlc_quantize_cli COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" --quantize int8=W -o "${_out}/q2.o")
add_test(NAME mdlc_quantize_frozen COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" --build train -o "${_out}/q3.o")
add_test(NAME mdlc_quantize_trainable COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" --build train-all -o "${_out}/q4.o")
add_test(NAME mdlc_quantize_group COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" --build bad-group -o "${_out}/q5.o")
add_test(NAME mdlc_quantize_vector COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" --build vector -o "${_out}/q6.o")
add_test(NAME mdlc_quantize_typo COMMAND mdlc "${_cfg}/model.dlm" -a "${_cfg}/quantize.yml" --build typo -o "${_out}/q7.o")
set_tests_properties(mdlc_quantize_trainable mdlc_quantize_group mdlc_quantize_vector PROPERTIES WILL_FAIL TRUE)
set_tests_properties(mdlc_quantize_typo PROPERTIES PASS_REGULAR_EXPRESSION "pattern 'V' selects no weight matrix")

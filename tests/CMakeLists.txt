include(CTest)

CPMAddPackage(URI "gh:catchorg/Catch2@3.8.0" EXCLUDE_FROM_ALL YES)

# Frontend tests (parser, type checks, computegraph): no backend needed.
add_executable(frontend_tests
    mdl/computegraph/function.cpp
    mdl/computegraph/object.cpp
    mdl/computegraph/graph.cpp
    mdl/parser/parser.cpp
    mdl/parser/overloads.cpp
    mdl/parser/imports.cpp
    mdl/parser/diagnostics.cpp
    mdl/parser/index.cpp
    mdl/utils/pathpattern.cpp
    mdl/passes/shapes.cpp
)
target_sources(frontend_tests PRIVATE mdl/format/format.cpp)
target_link_libraries(frontend_tests PRIVATE Catch2::Catch2WithMain mdl_frontend mdl_format)
target_compile_definitions(frontend_tests PRIVATE MDL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_include_directories(frontend_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME frontend_tests COMMAND frontend_tests)

# The language server (protocol-level tests).
add_executable(mdls_tests mdls/server.cpp)
target_link_libraries(mdls_tests PRIVATE Catch2::Catch2WithMain mdls_lib)
target_include_directories(mdls_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_definitions(mdls_tests PRIVATE MDL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME mdls_tests COMMAND mdls_tests)

# mdl-doc: the Doxygen pages of a test package (golden files, see tests/docs/check_mdl_doc.cmake).
add_test(NAME mdl_doc_golden COMMAND ${CMAKE_COMMAND} -DMDL_DOC=$<TARGET_FILE:mdl-doc>
	"-DSOURCE=${CMAKE_CURRENT_LIST_DIR}/docs" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/mdl-doc" -P "${CMAKE_CURRENT_LIST_DIR}/docs/check_mdl_doc.cmake")

if(NOT MDL_ENABLE_COMPILER)
	return()
endif()

# The installation (cmake --install, the CPack packages): a program built against it with find_package(MDL). Needs the
# examples' sources and the whole project built.
if(MDL_BUILD_EXAMPLES)
	add_test(NAME install_consumer COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR}
		-DCONSUMER=${CMAKE_CURRENT_LIST_DIR}/install -DEXAMPLE=${PROJECT_SOURCE_DIR}/examples/01_ffn
		-DWORK=${CMAKE_CURRENT_BINARY_DIR}/install_work -DGENERATOR=${CMAKE_GENERATOR} -DCUDA=${MDL_ENABLE_CUDA}
		-DCUDA_ARCH=${MDL_CUDA_ARCH}
		-P ${CMAKE_CURRENT_LIST_DIR}/install/check_install.cmake)
	set_tests_properties(install_consumer PROPERTIES TIMEOUT 1200)
endif()

# Compiler tests (passes, exports, backend, runtime).
add_executable(tests
    mdl/computegraph/einops.cpp
    mdl/backends/tvm_patterns.cpp
    mdl/kernels/packages.cpp
    mdl/passes/autodiff.cpp
    mdl/passes/masking.cpp
    mdl/runtime/safetensors.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain mdl mdl_runtime)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_features(tests PRIVATE cxx_std_23)
add_test(NAME unit_tests COMMAND tests)
target_compile_definitions(tests PRIVATE MDL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# mdl-kernels list: valid packages (exit 0), a broken package (exit 1, with its problems).
set(_kernels "${CMAKE_CURRENT_LIST_DIR}/kernels")
add_test(NAME mdl_kernels_list COMMAND mdl-kernels --kernels-dir "${_kernels}/valid" list)
set_tests_properties(mdl_kernels_list PROPERTIES PASS_REGULAR_EXPRESSION "cubin, device sm_80.*scaled_add 1.0")
add_test(NAME mdl_kernels_list_problems COMMAND mdl-kernels --kernels-dir "${_kernels}/broken" list)
set_tests_properties(mdl_kernels_list_problems PROPERTIES WILL_FAIL TRUE)
add_subdirectory(kernels/e2e)
add_subdirectory(statistics)
if(MDL_ENABLE_CUDA)
	add_subdirectory(cuda)
endif()

# mdl-kernels check/dump: the selection among competing implementations (tests/kernels/matching), the verdicts, and the
# Relax module at the matching boundary.
set(_model "${_kernels}/models/scaled_add.dlm")
# The package of the selection tests names the platform of this build (it would not be eligible on another).
mdl_kernel_host_platform(MDL_KERNEL_OS MDL_KERNEL_ARCH MDL_KERNEL_ABI)
set(_matching "${CMAKE_CURRENT_BINARY_DIR}/kernels/matching")
configure_file("${_kernels}/matching/competing/package.yml.in" "${_matching}/competing/package.yml" @ONLY)
configure_file("${_kernels}/matching/competing/lib/libcompeting.placeholder" "${_matching}/competing/lib/libcompeting.placeholder" COPYONLY)
add_test(NAME mdl_kernels_check COMMAND mdl-kernels --kernels-dir "${_matching}" check "${_model}" --kernel-package competing)
set_tests_properties(mdl_kernels_check PROPERTIES PASS_REGULAR_EXPRESSION
	"M__forward: competing/scaled-add-aligned at v4 \\(with a runtime guard\\).*M__other: competing/add-f32.*20 candidate\\(s\\), 5 selected")
add_test(NAME mdl_kernels_explain COMMAND mdl-kernels --kernels-dir "${_matching}" check --explain "${_model}" --kernel-package competing)
set_tests_properties(mdl_kernels_explain PROPERTIES PASS_REGULAR_EXPRESSION
	"guard    where \\$x.shape\\[0\\] % 16 == 0 \\(checked at runtime: B % 16 == 0\\).*overlaps competing/scaled-add-aligned.*selected with a runtime guard.*it is float32.*the constant is 2 \\(it is 3\\)")
# mdl-kernels deps/link-args: the packages' link requirements (CMake IMPORTED targets), and linker arguments from the
# kernel report of the end-to-end test.
add_test(NAME mdl_kernels_deps COMMAND mdl-kernels --kernels-dir "${_matching}" deps --kernel-package competing)
set_tests_properties(mdl_kernels_deps PROPERTIES PASS_REGULAR_EXPRESSION
	"add_library\\(mdl_kernel_artifact_competing_host STATIC IMPORTED\\).*MDL_KERNEL_PACKAGE_BUILD_competing IMPORTED")
add_test(NAME mdl_kernels_deps_unknown COMMAND mdl-kernels deps --kernel-package nonexistent)
set_tests_properties(mdl_kernels_deps_unknown PROPERTIES WILL_FAIL TRUE)
add_test(NAME mdl_kernels_link_args COMMAND mdl-kernels link-args "${CMAKE_CURRENT_BINARY_DIR}/kernels/e2e/kernel-report.json")
set_tests_properties(mdl_kernels_link_args PROPERTIES PASS_REGULAR_EXPRESSION "scaled_add_c")
# check reads the argument values like mdlc (-a, --build): quantized weights select the quantized oneDNN kernel.
if(TARGET mdl_kernel_package_onednn)
	set(_quantized "${CMAKE_CURRENT_LIST_DIR}/../examples/09_quantization")
	add_test(NAME mdl_kernels_check_build COMMAND mdl-kernels --kernels-dir "${MDL_KERNELS_BUILD_DIR}" check
		"${_quantized}/model.dlm" -a "${_quantized}/model.yml" --build mixed --kernel-package onednn)
	set_tests_properties(mdl_kernels_check_build PROPERTIES PASS_REGULAR_EXPRESSION
		"onednn/matmul-quantized-f32.*2 selected")
endif()
add_test(NAME mdl_kernels_dump COMMAND mdl-kernels dump "${_model}")
set_tests_properties(mdl_kernels_dump PROPERTIES PASS_REGULAR_EXPRESSION "R.multiply\\(x, T.float32\\(2.0\\)\\)")

# The build settings of argument-value files (mdlc --arg-vals, see docs/user/build-config.md): the command line takes
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
# Exported functions: the public member functions; --export selects some (only public ones; differentiated ones must be
# exported).
add_test(NAME mdlc_export_default COMMAND mdlc "${_cfg}/exports.dlm" -a "${_cfg}/exports.yml" -o "${_out}/x1.o")
add_test(NAME mdlc_export_selected COMMAND mdlc "${_cfg}/exports.dlm" -a "${_cfg}/exports.yml" --export forward --export Ex.loss -o "${_out}/x2.o")
add_test(NAME mdlc_export_not_public COMMAND mdlc "${_cfg}/exports.dlm" -a "${_cfg}/exports.yml" --export helper -o "${_out}/x3.o")
set_tests_properties(mdlc_export_not_public PROPERTIES PASS_REGULAR_EXPRESSION "--export helper: Ex.helper is not public")
add_test(NAME mdlc_export_missing COMMAND mdlc "${_cfg}/exports.dlm" -a "${_cfg}/exports.yml" --export nope -o "${_out}/x4.o")
set_tests_properties(mdlc_export_missing PROPERTIES PASS_REGULAR_EXPRESSION "no compiled module has a public function nope")
add_test(NAME mdlc_export_grad_not_exported COMMAND mdlc "${_cfg}/exports.dlm" -a "${_cfg}/exports.yml" --export forward --grad loss -o "${_out}/x5.o")
set_tests_properties(mdlc_export_grad_not_exported PROPERTIES PASS_REGULAR_EXPRESSION "cannot differentiate 'loss': Ex.loss is not exported")
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

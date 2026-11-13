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
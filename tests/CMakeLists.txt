include(CTest)

CPMAddPackage(URI "gh:catchorg/Catch2@3.8.0" EXCLUDE_FROM_ALL YES)

add_executable(tests
	# Add test files here
    mdl/computegraph/function.cpp
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain mdl)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_features(tests PRIVATE cxx_std_20)

add_test(NAME tirextrackertest COMMAND tests
	#WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/testfiles
)
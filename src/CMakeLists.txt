# The Model Library
add_library(mdl)

target_sources(mdl
	PUBLIC
		FILE_SET mdl_headers
		TYPE HEADERS
		BASE_DIRS
			${CMAKE_CURRENT_LIST_DIR}/../include
			${CMAKE_CURRENT_BINARY_DIR}/
		FILES
			${CMAKE_CURRENT_LIST_DIR}/../include/mdl/computegraph.hpp
	PRIVATE
		mdl/computegraph.cpp
		mdl/parser.cpp
		mdl/parser/definitions.cpp
)

# The Model Compiler
add_executable(mdlc)

target_sources(
    mdlc
    PRIVATE
        main.cpp
		mdl/logging.cpp
)

target_link_libraries(mdlc mdl)
target_compile_features(mdlc PUBLIC cxx_std_23)


##########################################################################################
# Libraries
##########################################################################################
# ANTLR4
set(ANTLR_VERSION 4.13.1)
# Download the jar file
FetchContent_Declare(antlrjar URL https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(antlrjar)
set(ANTLR_EXECUTABLE ${antlrjar_SOURCE_DIR}/antlr-${ANTLR_VERSION}-complete.jar)
# required if linking to static library
add_definitions(-DANTLR4CPP_STATIC)
# using /MD flag for antlr4_runtime (for Visual C++ compilers only)
set(ANTLR4_WITH_STATIC_CRT OFF)
set(ANTLR_BUILD_CPP_TESTS OFF)
FetchContent_Declare(
    antlr
    GIT_REPOSITORY https://github.com/antlr/antlr4
    GIT_TAG ${ANTLR_VERSION}
    SOURCE_SUBDIR "runtime/Cpp"
)
FetchContent_MakeAvailable(antlr)
set(ANTLR4CPP_INCLUDE_DIRS ${antlr_SOURCE_DIR}/runtime/Cpp/runtime/src)
include(${antlr_SOURCE_DIR}/runtime/Cpp/cmake/FindANTLR.cmake)

antlr_target(mdlparser ${CMAKE_CURRENT_LIST_DIR}/mdl/mdl.g4 LEXER PARSER VISITOR LISTENER PACKAGE antlr4)

target_link_libraries(mdl PUBLIC antlr4_static)
target_include_directories(mdl PUBLIC ${ANTLR4CPP_INCLUDE_DIRS} ${ANTLR_mdlparser_OUTPUT_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_sources(mdl PRIVATE ${ANTLR_mdlparser_CXX_OUTPUTS})

# TVM
CPMAddPackage(
	URI "gh:apache/tvm@0.22.0"
	OPTIONS "USE_LLVM ON" "USE_CUDA ON"
	EXCLUDE_FROM_ALL YES
)
target_link_libraries(mdl PUBLIC tvm)

get_target_property(LIBA_INCLUDES tvm INCLUDE_DIRECTORIES)
target_include_directories(mdl PUBLIC ${LIBA_INCLUDES}) # < I don't know why this is necessary


# NVRTC
find_package(CUDAToolkit REQUIRED)
target_link_libraries(mdl PRIVATE CUDA::nvrtc)

#-------------------#
# Compiler Command  #
#-------------------#
# spdlog for logging
CPMAddPackage(
	URI "gh:gabime/spdlog@1.15.3"
	# OPTIONS "SPDLOG_FMT_EXTERNAL ON"
	EXCLUDE_FROM_ALL YES
)
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)  # Is this needed?
target_link_libraries(mdlc spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

# CLI11
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(mdlc CLI11::CLI11)
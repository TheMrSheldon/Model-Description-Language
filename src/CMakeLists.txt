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
	PRIVATE
		mdl/argvals.cpp
		mdl/backends/tvm/tvm.cpp
		mdl/backends/tvm/details/pattern_registry.cpp
		mdl/bindings/generator.cpp
		mdl/compiler.cpp
		mdl/computegraph/datatype.cpp
		mdl/computegraph/graph.cpp
		mdl/computegraph/node.cpp
		mdl/computegraph/object.cpp
		mdl/namespace.cpp
		mdl/parser.cpp
		mdl/parser/errorlistener.cpp
		mdl/utils/utils.cpp
		mdl/utils/einops.cpp
)

# The Model Compiler
add_executable(mdlc)
target_compile_features(mdlc PUBLIC cxx_std_23)

target_sources(
    mdlc
    PRIVATE
        main.cpp
		mdl/logging.cpp
)

target_link_libraries(mdlc mdl)
target_compile_features(mdl PUBLIC cxx_std_23)

# The mdl-info command
add_executable(mdl-info)
target_sources(
    mdl-info
    PRIVATE
        commands/mdl-info.cpp
		mdl/logging.cpp
)
target_link_libraries(mdl-info mdl)



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
if (NOT BUILD_SHARED_LIBS)
	add_definitions(-DANTLR4CPP_STATIC)
endif()
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
# Pick the best available BLAS backend for this platform.
# Preference order: Intel MKL (AVX-512, cblas_sgemm_batch_strided) > OpenBLAS > none.
# macOS always uses Accelerate.
set(_MDL_MKL_HINT "/opt/intel/oneapi/mkl/latest")
if(APPLE)
	find_package(BLAS QUIET)
	# USE_BLAS=apple selects Accelerate; no extra option needed.
	set(MDL_TVM_BLAS_OPTS "USE_BLAS apple")
	set(MDL_BLAS_LINK_LIBS ${BLAS_LIBRARIES})
elseif(EXISTS "${_MDL_MKL_HINT}/include/mkl.h")
	message(STATUS "[mdl] Using Intel MKL at ${_MDL_MKL_HINT}")
	# USE_MKL takes the root path directly; do NOT also set USE_BLAS=mkl (deprecated
	# and causes TVM to overwrite USE_MKL before it's checked as a directory path).
	set(MDL_TVM_BLAS_OPTS "USE_MKL ${_MDL_MKL_HINT}")
	set(MDL_BLAS_LINK_LIBS "${_MDL_MKL_HINT}/lib/libmkl_rt.so")
else()
	find_package(BLAS QUIET)
	if(BLAS_FOUND)
		set(MDL_TVM_BLAS_OPTS "USE_BLAS openblas")
		set(MDL_BLAS_LINK_LIBS ${BLAS_LIBRARIES})
	else()
		set(MDL_TVM_BLAS_OPTS "USE_BLAS none")
		set(MDL_BLAS_LINK_LIBS "")
	endif()
endif()

CPMAddPackage(
	URI "gh:apache/tvm@0.24.0"
	OPTIONS "USE_LLVM ON" "USE_CUDA OFF" ${MDL_TVM_BLAS_OPTS}
	EXCLUDE_FROM_ALL YES
)
if(MDL_BLAS_LINK_LIBS)
	target_link_libraries(mdl PUBLIC tvm_compiler "${MDL_BLAS_LINK_LIBS}")
else()
	target_link_libraries(mdl PUBLIC tvm_compiler)
endif()

# tvm_compiler uses directory-scoped include_directories internally,
# which does not propagate to consumers — add both include roots explicitly.
target_include_directories(mdl PUBLIC
    ${tvm_SOURCE_DIR}/include
    ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include
)


# NVRTC
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(mdl PRIVATE CUDA::nvrtc)


# spdlog for logging
CPMAddPackage(
	URI "gh:gabime/spdlog@1.15.3"
	# OPTIONS "SPDLOG_FMT_EXTERNAL ON"
	EXCLUDE_FROM_ALL YES
)
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)  # Is this needed?
target_link_libraries(mdl PUBLIC spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

# yaml-cpp
CPMAddPackage(URI "gh:jbeder/yaml-cpp#0.8.0" EXCLUDE_FROM_ALL YES)
target_link_libraries(mdl PUBLIC yaml-cpp::yaml-cpp)

#-------------------#
# Compiler Command  #
#-------------------#
# CLI11
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(mdlc CLI11::CLI11)

# termcolor
CPMAddPackage(URI "gh:ikalnytskyi/termcolor@2.1.0" EXCLUDE_FROM_ALL YES)
target_link_libraries(mdl PRIVATE termcolor::termcolor)

#-------------------#
# mdl-info Command  #
#-------------------#
# CLI11
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(mdl-info CLI11::CLI11)
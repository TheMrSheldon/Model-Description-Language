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
		mdl/backends/tvm/details/aot.cpp
		mdl/backends/tvm/details/legalize.cpp
		mdl/backends/tvm/details/pattern_registry.cpp
		mdl/backends/tvm/details/patterns.cpp
		mdl/backends/tvm/details/schedule.cpp
		mdl/backends/tvm/details/pipeline.cpp
		mdl/bindings/generator.cpp
		mdl/compiler.cpp
		mdl/exports.cpp
		mdl/passes/autodiff.cpp
		mdl/passes/shapes.cpp
		mdl/passes/masking.cpp
		mdl/logging.cpp
		mdl/computegraph/datatype.cpp
		mdl/computegraph/graph.cpp
		mdl/computegraph/node.cpp
		mdl/computegraph/object.cpp
		mdl/namespace.cpp
		mdl/parser.cpp
		mdl/parser/errorlistener.cpp
		mdl/utils/utils.cpp
		mdl/utils/einops.cpp
		mdl/utils/pathpattern.cpp
)

# The Model Compiler
add_executable(mdlc)
target_compile_features(mdlc PUBLIC cxx_std_23)

target_sources(
    mdlc
    PRIVATE
        main.cpp
)

target_link_libraries(mdlc mdl)
target_compile_features(mdl PUBLIC cxx_std_23)
# Location of the standard library (`import std;`) when running from the build tree.
target_compile_definitions(mdl PRIVATE MDL_DEFAULT_STDLIB_DIR="${PROJECT_SOURCE_DIR}/std")

# The mdl-info command
add_executable(mdl-info)
target_sources(
    mdl-info
    PRIVATE
        commands/mdl-info.cpp
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
# - Vendor kernels (GEMM, softmax, layer norm, ...) are provided by oneDNN through the mdl_kernels library (see below),
#   so TVM's BLAS contrib is not needed.
# - TVM's parallel loops use OpenMP so that TVM-generated kernels and oneDNN share one thread pool (no oversubscription).
# - Pinned to 0.24: TVM 0.25+ replaces Relax StructInfo by a new type system and removes TIR buffer maps, which requires
#   porting the whole backend (planned as a separate upgrade).
CPMAddPackage(
	URI "gh:apache/tvm@0.24.0"
	# The backend uses TVM's C++ compiler APIs (pattern language, ExprMutator, passes), many of which are not marked
	# TVM_DLL; hiding private symbols (TVM's default for non-Debug builds) would make them unavailable.
	OPTIONS "USE_LLVM ON" "USE_CUDA OFF" "USE_BLAS none" "USE_OPENMP gnu" "HIDE_PRIVATE_SYMBOLS OFF"
	EXCLUDE_FROM_ALL YES
)
target_link_libraries(mdl PUBLIC tvm_compiler)

# tvm_compiler uses directory-scoped include_directories internally,
# which does not propagate to consumers — add both include roots explicitly.
target_include_directories(mdl PUBLIC
    ${tvm_SOURCE_DIR}/include
    ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include
)


# ── Runtime helpers (mdl_runtime) ────────────────────────────────────────────
# Header-only helpers for programs that call compiled models (include/mdl/runtime.hpp) plus the TVM runtime that
# compiled models depend on (thread pool, workspace allocation, FFI).
# nlohmann/json parses the safetensors headers (header-only).
CPMAddPackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF" EXCLUDE_FROM_ALL YES)
add_library(mdl_runtime INTERFACE)
target_include_directories(mdl_runtime INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(mdl_runtime INTERFACE cxx_std_20)
target_link_libraries(mdl_runtime INTERFACE tvm_runtime nlohmann_json::nlohmann_json)

# ── Vendor kernels (mdl_kernels) ─────────────────────────────────────────────
# Runtime library that compiled models link against: it implements the extern kernels the TVM backend emits calls to
# (see include/mdl/runtime/kernels.h). It is not linked into the compiler itself.
option(MDL_ENABLE_ONEDNN "Build the oneDNN-based vendor kernels (mdl_kernels)" ON)
add_library(mdl_kernels STATIC)
target_include_directories(mdl_kernels PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(mdl_kernels PUBLIC cxx_std_20)
set_property(TARGET mdl_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
target_sources(mdl_kernels PRIVATE mdl/runtime/kernels/profile.cpp)
if(MDL_ENABLE_ONEDNN)
	# Only build the primitive kinds that mdl_kernels uses (forward and backward) to keep the build short.
	set(DNNL_ENABLE_PRIMITIVE "MATMUL;SOFTMAX;ELTWISE;LAYER_NORMALIZATION;REORDER" CACHE STRING "" FORCE)
	CPMAddPackage(
		NAME oneDNN
		GITHUB_REPOSITORY uxlfoundation/oneDNN
		GIT_TAG v3.13.4
		OPTIONS
			"DNNL_LIBRARY_TYPE STATIC"
			"DNNL_CPU_RUNTIME OMP"
			"DNNL_BUILD_TESTS OFF"
			"DNNL_BUILD_EXAMPLES OFF"
			"ONEDNN_BUILD_GRAPH OFF"
			"DNNL_ENABLE_JIT_PROFILING OFF"
		EXCLUDE_FROM_ALL YES
	)
	target_sources(mdl_kernels PRIVATE mdl/runtime/kernels/onednn.cpp)
	# The static oneDNN library uses OpenMP (DNNL_CPU_RUNTIME OMP) but does not propagate it to consumers.
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	target_link_libraries(mdl_kernels PUBLIC dnnl OpenMP::OpenMP_CXX)
	target_compile_definitions(mdl_kernels PUBLIC MDL_KERNELS_ONEDNN)
else()
	# Without vendor kernels the backend lowers everything to TVM-generated kernels; the library stays empty.
	target_sources(mdl_kernels PRIVATE mdl/runtime/kernels/none.cpp)
endif()

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
CPMAddPackage(URI "gh:jbeder/yaml-cpp#yaml-cpp-0.9.0" EXCLUDE_FROM_ALL YES)
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

# ── IREE backend (optional) ───────────────────────────────────────────────────
# Enable with: cmake -DMDL_ENABLE_IREE=ON
#
# IREE is located in one of two ways (in preference order):
#   1. find_package — set MDL_IREE_INSTALL_DIR or IREE_INSTALL_DIR (env) to the
#      prefix used when installing IREE with `cmake --install`.
#   2. Automatic download — the pre-built release tar for Linux x86_64 is fetched
#      from GitHub.  Override MDL_IREE_DIST_URL to target a different release or
#      platform (see https://github.com/iree-org/iree/releases).
#
# The downloaded distribution contains libIREECompiler.so and the C API headers;
# it is placed in the CMake build directory and does NOT require compiling LLVM.
option(MDL_ENABLE_IREE "Build the IREE backend" OFF)
set(MDL_IREE_INSTALL_DIR "" CACHE PATH
    "Path to an installed IREE prefix (sets the find_package hint).")
set(MDL_IREE_DIST_URL "" CACHE STRING
    "URL of a pre-built IREE distribution tar. Leave empty to use the default release.")

if(MDL_ENABLE_IREE)
    # The IREE backend invokes iree-compile as a subprocess to avoid an LLVM
    # version conflict: libIREECompiler.so embeds LLVM 20 while TVM links
    # system LLVM 19, which causes a segfault when both are loaded in-process.
    # No link-time dependency on IREE is needed; iree-compile must be on PATH
    # (or pointed to via the IREE_COMPILE environment variable at runtime).

    target_sources(mdl PRIVATE mdl/backends/iree/iree_backend.cpp)
    target_compile_definitions(mdl PUBLIC MDL_IREE_ENABLED)
    message(STATUS "[mdl] IREE backend enabled (iree-compile subprocess)")
endif()

#-------------------#
# mdl-info Command  #
#-------------------#
# CLI11
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(mdl-info CLI11::CLI11)
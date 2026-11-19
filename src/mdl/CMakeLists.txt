# The libraries of the language: the frontend and the formatter (always), the compiler (with MDL_ENABLE_COMPILER).

##########################################################################################
# mdl_frontend: parsing, type checking, diagnostics, analysis for tools; no backend dependencies
##########################################################################################
antlr_target(mdlparser ${CMAKE_CURRENT_LIST_DIR}/mdl.g4 LEXER PARSER VISITOR LISTENER PACKAGE antlr4)

add_library(mdl_frontend STATIC
	analysis/index.cpp
	analysis/unused.cpp
	computegraph/datatype.cpp
	computegraph/graph.cpp
	computegraph/node.cpp
	computegraph/object.cpp
	diagnostics.cpp
	namespace.cpp
	parser.cpp
	parser/errorlistener.cpp
	passes/shapes.cpp
	utils/utils.cpp
	utils/einops.cpp
	utils/pathpattern.cpp
	utils/platform.cpp
	${ANTLR_mdlparser_CXX_OUTPUTS}
)
target_compile_features(mdl_frontend PUBLIC cxx_std_23)
# The version (from the latest git tag, see the top-level CMakeLists.txt).
configure_file(${PROJECT_SOURCE_DIR}/include/mdl/version.hpp.in ${PROJECT_BINARY_DIR}/generated/include/mdl/version.hpp @ONLY)
target_include_directories(mdl_frontend PUBLIC
	${PROJECT_BINARY_DIR}/generated/include
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/src # internal headers (e.g. mdl/argvals.hpp) for the compiler and the tests
	${ANTLR4CPP_INCLUDE_DIRS}
	${ANTLR_mdlparser_OUTPUT_DIR}
)
target_link_libraries(mdl_frontend PUBLIC antlr4_static PRIVATE termcolor::termcolor)
set_property(TARGET mdl_frontend PROPERTY POSITION_INDEPENDENT_CODE ON)
# The standard library when running from the build tree (see parser::defaultStdlibDirs).
target_compile_definitions(mdl_frontend PRIVATE MDL_DEFAULT_STDLIB_DIR="${PROJECT_SOURCE_DIR}/std")

##########################################################################################
# mdl_format: the formatter
##########################################################################################
add_library(mdl_format STATIC format/format.cpp)
target_link_libraries(mdl_format PUBLIC mdl_frontend)
set_property(TARGET mdl_format PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT MDL_ENABLE_COMPILER)
	return()
endif()

##########################################################################################
# mdl: the compiler (passes, exports, bindings, backends)
##########################################################################################
add_library(mdl
	argvals.cpp
	backends/tvm/tvm.cpp
	backends/tvm/details/aot.cpp
	backends/tvm/details/kernel_match.cpp
	backends/tvm/details/legalize.cpp
	backends/tvm/details/pattern_registry.cpp
	backends/tvm/details/patterns.cpp
	backends/tvm/details/schedule.cpp
	backends/tvm/details/pipeline.cpp
	backends/tvm/details/flatten_injective.cpp
	backends/tvm/targets/llvm.cpp
	backends/tvm/targets/registry.cpp
	bindings/generator.cpp
	compiler.cpp
	exports.cpp
	kernels/discovery.cpp
	kernels/expr.cpp
	kernels/package.cpp
	passes/autodiff.cpp
	passes/masking.cpp
	logging.cpp
)
target_compile_features(mdl PUBLIC cxx_std_23)
target_link_libraries(mdl PUBLIC
	mdl_frontend
	tvm_compiler
	spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>
	yaml-cpp::yaml-cpp
)
# The kernel packages built by this project, when running from the build tree (see kernels::kernelDirs).
# (The packages of the source tree's kernels/ are generated there, with their libraries; see kernels/CMakeLists.txt.)
target_compile_definitions(mdl PRIVATE MDL_DEFAULT_KERNELS_DIR="${CMAKE_BINARY_DIR}/mdl-kernels")
# The targets of the TVM backend (backends/tvm/targets/): CUDA only with MDL_ENABLE_CUDA (else unsupported). The CUDA
# target combines host and device code into one object file with the linker (ld -r).
if(MDL_ENABLE_CUDA)
	# The GPU schedules, the tensor intrinsics, and the CUDA C code generation callbacks: parts of TVM's Python package
	# translated to C++, kept separate in src/tvm_ported (see its README.md).
	target_link_libraries(mdl PRIVATE mdl_tvm_ported)
	target_sources(mdl PRIVATE backends/tvm/targets/cuda.cpp backends/tvm/targets/gpu/row_reduction.cpp)
	# The CUDA target uses TVM internals (the CUDA module of the code generation; see targets/cuda.cpp).
	set_source_files_properties(backends/tvm/targets/cuda.cpp PROPERTIES INCLUDE_DIRECTORIES "${tvm_SOURCE_DIR}/src")
	target_compile_definitions(mdl PRIVATE MDL_WITH_CUDA=1 MDL_LINKER="${CMAKE_LINKER}")
endif()
# tvm_compiler uses directory-scoped include_directories internally, which do not propagate to consumers.
target_include_directories(mdl PUBLIC ${tvm_SOURCE_DIR}/include ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include)

# ── IREE backend (optional) ───────────────────────────────────────────────────
# The IREE backend invokes iree-compile as a subprocess to avoid an LLVM version conflict: libIREECompiler.so embeds
# LLVM 20 while TVM links system LLVM 19, which causes a segfault when both are loaded in-process. No link-time
# dependency on IREE is needed; iree-compile must be on PATH (or given by the IREE_COMPILE environment variable).
option(MDL_ENABLE_IREE "Build the IREE backend" OFF)
if(MDL_ENABLE_IREE)
	target_sources(mdl PRIVATE backends/iree/iree_backend.cpp)
	target_compile_definitions(mdl PUBLIC MDL_IREE_ENABLED)
	message(STATUS "[mdl] IREE backend enabled (iree-compile subprocess)")
endif()

add_subdirectory(runtime) # mdl_runtime, mdl_profile

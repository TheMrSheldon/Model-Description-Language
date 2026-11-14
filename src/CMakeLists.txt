# The sources: the libraries (mdl/, mdls/) and the commands (commands/). This file sets up the dependencies.
#
# The compiler (mdlc, the TVM backend, the runtime libraries) is optional: without it, only the frontend and the tooling
# (mdl-format, mdls) are built, which needs neither TVM/LLVM nor oneDNN.
option(MDL_ENABLE_COMPILER "Build the compiler (mdlc) with the TVM backend and the runtime libraries" ON)

include(GNUInstallDirs)

##########################################################################################
# Dependencies of the frontend and the tooling
##########################################################################################
# ANTLR4 (parser generator: the jar at build time, the C++ runtime)
set(ANTLR_VERSION 4.13.2)
FetchContent_Declare(antlrjar URL https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(antlrjar)
set(ANTLR_EXECUTABLE ${antlrjar_SOURCE_DIR}/antlr-${ANTLR_VERSION}-complete.jar)
# required if linking to static library
if (NOT BUILD_SHARED_LIBS)
	add_definitions(-DANTLR4CPP_STATIC)
endif()
# The C runtime of antlr4_runtime (Visual C++ only): static (/MT) if the project uses the static runtime (release builds
# of the tools, -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded), else /MD.
# (A cache variable: ANTLR's option() would ignore a normal variable; its default is ON.)
if(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "^MultiThreaded(Debug)?$")
	set(WITH_STATIC_CRT ON CACHE BOOL "" FORCE)
else()
	set(WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
endif()
set(ANTLR_BUILD_CPP_TESTS OFF)
FetchContent_Declare(
    antlr
    GIT_REPOSITORY https://github.com/antlr/antlr4
    GIT_TAG ${ANTLR_VERSION}
    SOURCE_SUBDIR "runtime/Cpp"
)
FetchContent_MakeAvailable(antlr)
set_target_properties(antlr4_shared PROPERTIES EXCLUDE_FROM_ALL ON) # only the static runtime is used
if(MSVC)
	# The runtime uses std::chrono without including <chrono> (ProfilingATNSimulator.cpp); libstdc++ and libc++ include it
	# indirectly, MSVC's standard library does not.
	target_compile_options(antlr4_static PRIVATE /FIchrono)
endif()
set(ANTLR4CPP_INCLUDE_DIRS ${antlr_SOURCE_DIR}/runtime/Cpp/runtime/src)
include(${antlr_SOURCE_DIR}/runtime/Cpp/cmake/FindANTLR.cmake)

# termcolor (terminal colors of the diagnostics)
CPMAddPackage(URI "gh:ikalnytskyi/termcolor@2.1.0" EXCLUDE_FROM_ALL YES)
# nlohmann/json (the language server's protocol, safetensors headers in mdl_runtime; header-only)
CPMAddPackage(URI "gh:nlohmann/json@3.12.0" OPTIONS "JSON_BuildTests OFF" EXCLUDE_FROM_ALL YES)
# CLI11 (command lines of the commands)
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)

##########################################################################################
# Dependencies of the compiler
##########################################################################################
if(MDL_ENABLE_COMPILER)
	# TVM
	# - Vendor kernels (GEMM, softmax, layer norm, ...) are provided by oneDNN through the mdl_kernels library, so TVM's
	#   BLAS contrib is not needed.
	# - TVM's parallel loops use OpenMP so that TVM-generated kernels and oneDNN share one thread pool (no
	#   oversubscription).
	# - Pinned to 0.24: TVM 0.25+ replaces Relax StructInfo by a new type system and removes TIR buffer maps, which
	#   requires porting the whole backend (planned as a separate upgrade).
	CPMAddPackage(
		URI "gh:apache/tvm@0.24.0"
		# The backend uses TVM's C++ compiler APIs (pattern language, ExprMutator, passes), many of which are not marked
		# TVM_DLL; hiding private symbols (TVM's default for non-Debug builds) would make them unavailable.
		OPTIONS "USE_LLVM ON" "USE_CUDA OFF" "USE_BLAS none" "USE_OPENMP gnu" "HIDE_PRIVATE_SYMBOLS OFF"
		EXCLUDE_FROM_ALL YES
	)
	# spdlog (logging)
	CPMAddPackage(URI "gh:gabime/spdlog@1.15.3" EXCLUDE_FROM_ALL YES)
	set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)
	# yaml-cpp (argument values and build configurations)
	CPMAddPackage(URI "gh:jbeder/yaml-cpp#yaml-cpp-0.9.0" EXCLUDE_FROM_ALL YES)
endif()

##########################################################################################
# Libraries and commands
##########################################################################################
add_subdirectory(mdl)      # mdl_frontend, mdl_format; mdl, mdl_runtime, mdl_kernels (compiler)
add_subdirectory(mdls)     # mdls_lib (the language server)
add_subdirectory(commands) # mdl-format, mdls; mdlc, mdl-info (compiler)

# The standard library (`import std;`): an installation has it in <prefix>/share/mdl/std next to <prefix>/bin (found
# relative to the executable). Install the commands and the standard library only (not the dependencies):
# `cmake --install <build> --component tools`.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/std/ DESTINATION share/mdl/std COMPONENT tools FILES_MATCHING PATTERN "*.dlm")

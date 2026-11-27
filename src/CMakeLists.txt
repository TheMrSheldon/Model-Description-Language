# The sources: the libraries (mdl/, mdls/) and the commands (commands/). This file sets up the dependencies.
#
# The compiler (mdlc, the TVM backend, the runtime libraries) is optional: without it, only the frontend and the tooling
# (mdl-format, mdls) are built, which needs neither TVM/LLVM nor oneDNN.
option(MDL_ENABLE_COMPILER "Build the compiler (mdlc) with the TVM backend and the runtime libraries" ON)

include(GNUInstallDirs)
# The private libraries of an installation (TVM's shared libraries, mdl_profile): <prefix>/lib/mdl. The commands find the
# shared ones relative to themselves (RPATH, see mdl_add_command), programs that use the SDK through find_package(MDL).
set(MDL_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/mdl")
# The headers of the SDK's dependencies (TVM, DLPack, nlohmann/json): <prefix>/include/mdl/third_party, so that they
# neither clash with other installations of these libraries nor leak into the include path of other programs.
set(MDL_INSTALL_THIRD_PARTY_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/mdl/third_party")

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
    EXCLUDE_FROM_ALL
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
	# CUDA target (MDL_ENABLE_CUDA, a package feature, see the top-level CMakeLists.txt): TVM's CUDA codegen and runtime.
	# Without it, the cuda target is registered as unsupported (see backends/tvm/targets/).
	if(MDL_ENABLE_CUDA)
		set(MDL_TVM_USE_CUDA ON)
	else()
		set(MDL_TVM_USE_CUDA OFF)
	endif()
	# TVM
	# - Optimized kernels (GEMM, softmax, layer norm, ...) come from kernel packages (oneDNN: kernels/onednn), so TVM's
	#   BLAS contrib is not needed.
	# - TVM's parallel loops use OpenMP so that TVM-generated kernels and oneDNN share one thread pool (no
	#   oversubscription).
	# - Pinned to v0.27.0.post1 (= v0.27.0 + one CUDA fix) by commit, so that the exact revision is reproducible.
	CPMAddPackage(
		NAME tvm
		VERSION 0.27.0
		GITHUB_REPOSITORY apache/tvm
		GIT_TAG 7b5acb32bd271342a24130f4ef439ff6c2e52c6c # v0.27.0.post1
		# The backend uses TVM's C++ compiler APIs (pattern language, ExprMutator, passes), many of which are not marked
		# TVM_DLL; hiding private symbols (TVM's default for non-Debug builds) would make them unavailable.
		OPTIONS "USE_LLVM ON" "USE_CUDA ${MDL_TVM_USE_CUDA}" "USE_BLAS none" "USE_OPENMP gnu" "HIDE_PRIVATE_SYMBOLS OFF"
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
# Parts of TVM's Python package translated to C++ (mdl_tvm_ported, used by the CUDA target; see tvm_ported/README.md).
if(MDL_ENABLE_COMPILER AND MDL_ENABLE_CUDA)
	add_subdirectory(tvm_ported)
endif()
add_subdirectory(mdl)      # mdl_frontend, mdl_format; mdl, mdl_runtime, mdl_profile (compiler)
add_subdirectory(mdls)     # mdls_lib (the language server)
add_subdirectory(commands) # mdl-format, mdls; mdlc, mdl-info (compiler)

# The standard library (`import std;`): an installation has it in <prefix>/share/mdl/std next to <prefix>/bin (found
# relative to the executable). Install the commands and the standard library only (not the dependencies):
# `cmake --install <build> --component tools`.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/std/ DESTINATION share/mdl/std COMPONENT tools FILES_MATCHING PATTERN "*.dlm")

# ── The installation of the compiler and the SDK ─────────────────────────────────────────────
# Components (see docs/dev/technical/packaging.md): tools (above), runtime (TVM's runtime libraries: needed by the compiler
# and by programs that call models), compiler (mdlc, mdl-info, mdl-kernels, TVM's compiler library, the CPU kernel
# packages), cuda (optional: the CUDA runtime and the kernel packages for CUDA targets), sdk (headers, libraries, and the
# CMake package of find_package(MDL)). TVM and the other dependencies are added EXCLUDE_FROM_ALL, so their own install
# rules do not apply.
if(MDL_ENABLE_COMPILER)
	# The libraries find each other next to themselves ($ORIGIN); tvm_compiler also keeps the library path of its
	# LLVM (a system LLVM is not on the default search path; with a statically linked LLVM there is nothing to keep).
	if(APPLE)
		set(_origin "@loader_path")
	else()
		set(_origin "$ORIGIN")
	endif()
	set_target_properties(tvm_runtime tvm_ffi_shared tvm_compiler PROPERTIES INSTALL_RPATH "${_origin}")
	set_target_properties(tvm_compiler PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
	install(TARGETS tvm_ffi_shared tvm_runtime COMPONENT runtime
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} LIBRARY DESTINATION ${MDL_INSTALL_LIBDIR}
		ARCHIVE DESTINATION ${MDL_INSTALL_LIBDIR})
	install(TARGETS tvm_compiler COMPONENT compiler
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} LIBRARY DESTINATION ${MDL_INSTALL_LIBDIR}
		ARCHIVE DESTINATION ${MDL_INSTALL_LIBDIR})

	# The CUDA runtime (device API, loading the embedded cubin) of models for CUDA targets; part of the optional cuda
	# component with the kernel packages for CUDA (kernels/cublas, kernels/cudnn).
	if(MDL_ENABLE_CUDA)
		set_target_properties(tvm_runtime_cuda PROPERTIES INSTALL_RPATH "${_origin}")
		install(TARGETS tvm_runtime_cuda COMPONENT cuda
			RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} LIBRARY DESTINATION ${MDL_INSTALL_LIBDIR}
			ARCHIVE DESTINATION ${MDL_INSTALL_LIBDIR})
	endif()

	# The SDK's headers: those of the runtime (include/mdl) and of the dependencies that they include.
	install(FILES ${PROJECT_SOURCE_DIR}/include/mdl/runtime.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mdl COMPONENT sdk)
	install(FILES ${PROJECT_SOURCE_DIR}/include/mdl/runtime/profile.h
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mdl/runtime COMPONENT sdk)
	install(DIRECTORY ${tvm_SOURCE_DIR}/include/tvm/runtime ${tvm_SOURCE_DIR}/include/tvm/support
		DESTINATION ${MDL_INSTALL_THIRD_PARTY_INCLUDEDIR}/tvm COMPONENT sdk FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include/tvm/ffi
		DESTINATION ${MDL_INSTALL_THIRD_PARTY_INCLUDEDIR}/tvm COMPONENT sdk)
	install(DIRECTORY ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack
		DESTINATION ${MDL_INSTALL_THIRD_PARTY_INCLUDEDIR} COMPONENT sdk)
	install(DIRECTORY ${json_SOURCE_DIR}/include/nlohmann DESTINATION ${MDL_INSTALL_THIRD_PARTY_INCLUDEDIR} COMPONENT sdk)

	# The CMake package: find_package(MDL) defines mdl::runtime, mdl::profile and mdl_add_model().
	include(CMakePackageConfigHelpers)
	set(MDL_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/MDL")
	file(RELATIVE_PATH MDL_CONFIG_TO_PREFIX "/${MDL_CMAKE_INSTALL_DIR}" "/")
	configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/MDLConfig.cmake.in
		${CMAKE_CURRENT_BINARY_DIR}/MDLConfig.cmake INSTALL_DESTINATION ${MDL_CMAKE_INSTALL_DIR})
	write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/MDLConfigVersion.cmake
		VERSION ${MDL_VERSION} COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MDLConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/MDLConfigVersion.cmake
		${PROJECT_SOURCE_DIR}/cmake/MDL.cmake DESTINATION ${MDL_CMAKE_INSTALL_DIR} COMPONENT sdk)
endif()

# ── The licenses of the third-party software in the binaries (see cmake/ThirdParty.cmake) ──────
# The frontend and the tooling (statically linked into mdl-format, mdls, mdl-doc, and mdlc).
mdl_third_party(ANTLR COMPONENTS tools compiler VERSION ${ANTLR_VERSION} SPDX BSD-3-Clause URL https://www.antlr.org
	LICENSE_FILES "${antlr_SOURCE_DIR}/LICENSE.txt")
mdl_third_party(CLI11 COMPONENTS tools compiler VERSION 2.4.2 SPDX BSD-3-Clause URL https://github.com/CLIUtils/CLI11
	LICENSE_FILES "${CLI11_SOURCE_DIR}/LICENSE")
mdl_third_party(termcolor COMPONENTS tools compiler VERSION 2.1.0 SPDX BSD-3-Clause
	URL https://github.com/ikalnytskyi/termcolor LICENSE_FILES "${termcolor_SOURCE_DIR}/LICENSE")
# (also the headers of the SDK: mdl/runtime.hpp includes it)
mdl_third_party(nlohmann_json COMPONENTS tools compiler sdk VERSION 3.12.0 SPDX MIT URL https://github.com/nlohmann/json
	LICENSE_FILES "${json_SOURCE_DIR}/LICENSE.MIT")
if(CMAKE_EXE_LINKER_FLAGS MATCHES "static-libstdc\\+\\+")
	mdl_third_party("GCC runtime libraries (libstdc++, libgcc)" COMPONENTS tools compiler
		SPDX "GPL-3.0-or-later WITH GCC-exception-3.1" URL https://gcc.gnu.org
		NOTE "Linked statically. The GCC Runtime Library Exception permits distributing programs that contain them under terms of your choice; their source code is at https://gcc.gnu.org.")
endif()

if(MDL_ENABLE_COMPILER)
	mdl_third_party(spdlog COMPONENTS compiler VERSION 1.15.3 SPDX MIT URL https://github.com/gabime/spdlog
		LICENSE_FILES "${spdlog_SOURCE_DIR}/LICENSE")
	mdl_third_party(yaml-cpp COMPONENTS compiler VERSION 0.9.0 SPDX MIT URL https://github.com/jbeder/yaml-cpp
		LICENSE_FILES "${YAML_CPP_SOURCE_DIR}/LICENSE")
	# TVM: the compiler (libtvm_compiler), the runtime (libtvm_runtime, libtvm_ffi), and the headers of the SDK. Its
	# bundled software that these libraries contain: DLPack, libbacktrace (tvm-ffi), the half-precision conversions.
	set(_tvm_components runtime compiler sdk)
	if(MDL_ENABLE_CUDA AND NOT WIN32)
		list(APPEND _tvm_components cuda)
	endif()
	mdl_third_party(TVM COMPONENTS ${_tvm_components} VERSION "0.27.0" SPDX Apache-2.0 URL https://tvm.apache.org
		LICENSE_FILES "${tvm_SOURCE_DIR}/LICENSE" "${tvm_SOURCE_DIR}/NOTICE" "${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/LICENSE"
		"${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/NOTICE" "${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/licenses/*"
		"${tvm_SOURCE_DIR}/licenses/LICENSE.builtin_fp16.txt")
	mdl_third_party(DLPack COMPONENTS runtime compiler sdk SPDX Apache-2.0 URL https://github.com/dmlc/dlpack
		LICENSE_FILES "${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE")
	# LLVM (Apache-2.0 with LLVM exceptions): linked into the compiler. The license file is where the distribution
	# of the LLVM that TVM uses puts it.
	mdl_find_license_file(_llvm_license
		"${LLVM_DIR}/../../../share/doc/llvm*/LICENSE.TXT" "/usr/share/licenses/llvm*/LICENSE.TXT"
		"/usr/share/doc/llvm*/LICENSE.TXT" "/usr/share/doc/llvm-*/copyright" "/usr/share/doc/llvm/copyright")
	mdl_third_party(LLVM COMPONENTS compiler SPDX "Apache-2.0 WITH LLVM-exception" URL https://llvm.org
		LICENSE_FILES "${_llvm_license}" NOTE "Used by TVM's code generation for CPU targets.")
endif()

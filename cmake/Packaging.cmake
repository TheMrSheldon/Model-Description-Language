# The packages of the project (CPack; see docs/dev/technical/packaging.md): `cpack` in the build directory.
#
#   Linux    TGZ (everything), DEB and RPM (one package per component)
#   macOS    TGZ (everything), productbuild (.pkg with a choice of components)
#   Windows  ZIP (everything), NSIS (installer with a choice of components)
#
# Components (the install rules assign them): the formats without a choice install everything.
#   mdl       the license; in the DEB/RPM packages the meta package `mdl` that depends on tools, compiler and sdk
#   tools     mdl-format, mdls, mdl-doc, and the standard library (what an editor needs)
#   runtime   TVM's runtime libraries, shared by the compiler and the SDK
#   compiler  mdlc, mdl-info, mdl-kernels, the compiler library, and the CPU kernel packages (oneDNN)
#   sdk       headers, libraries and the CMake package (find_package(MDL)) to use compiled models in C++ programs
#   cuda      the CUDA runtime and the kernel packages for CUDA targets (cuBLAS, cuDNN); needs MDL_ENABLE_CUDA
#
# The generator-specific settings (install prefix, ...) are in MDLCPackOptions.cmake, which CPack reads for each generator.
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
	return() # used as a subproject: the including project packages
endif()

##########################################################################################
# Platform and version
##########################################################################################
# The platform of the file names, as in the release (linux-x64, linux-arm64, darwin-arm64, win32-x64, win32-arm64).
if(NOT MDL_PACKAGE_PLATFORM)
	string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
	if(_arch MATCHES "^(x86_64|amd64|x64)$")
		set(_arch x64)
	elseif(_arch MATCHES "^(aarch64|arm64)$")
		set(_arch arm64)
	endif()
	if(WIN32)
		set(_os win32)
	elseif(APPLE)
		set(_os darwin)
	else()
		set(_os linux)
	endif()
	set(MDL_PACKAGE_PLATFORM "${_os}-${_arch}")
endif()

# The version: the latest tag (MDL_VERSION) and what git describe adds to it (MDL_VER_META):
#   - a prerelease ("-rc1": from the tag vX.Y.Z-rc1) sorts before the release (DEB: 1.2.3~rc1, RPM release 0.rc1),
#   - a snapshot after the tag ("-17-gabc123": 17 commits after, as the nightly build) sorts after it (1.2.3+17.gabc123).
set(CPACK_PACKAGE_VERSION "${MDL_VERSION}${MDL_VER_META}")
if(MDL_VER_META STREQUAL "")
	set(_deb_version "${MDL_VERSION}")
	set(_rpm_version "${MDL_VERSION}")
	set(_rpm_release "1")
elseif(MDL_VER_META MATCHES "^-([0-9]+)-g([0-9a-f]+)(-dirty)?$")
	set(_deb_version "${MDL_VERSION}+${CMAKE_MATCH_1}.g${CMAKE_MATCH_2}")
	set(_rpm_version "${MDL_VERSION}")
	set(_rpm_release "1.${CMAKE_MATCH_1}.g${CMAKE_MATCH_2}")
else()
	string(REGEX REPLACE "^-" "" _pre "${MDL_VER_META}")
	string(REGEX REPLACE "[^A-Za-z0-9.+~]" "." _pre "${_pre}")
	set(_deb_version "${MDL_VERSION}~${_pre}")
	set(_rpm_version "${MDL_VERSION}")
	set(_rpm_release "0.${_pre}")
endif()

##########################################################################################
# Components
##########################################################################################
set(_components mdl tools)
if(MDL_ENABLE_COMPILER)
	list(APPEND _components runtime compiler sdk)
	if(MDL_ENABLE_CUDA AND NOT WIN32) # (the CUDA target is not ported to Windows)
		list(APPEND _components cuda)
	endif()
endif()
set(CPACK_COMPONENTS_ALL ${_components})

# The license (Apache-2.0) and the notice accompany every package; the meta package also has the README.
foreach(_component IN LISTS _components)
	if(_component STREQUAL "mdl")
		set(_doc_dir share/doc/mdl)
	else()
		set(_doc_dir share/doc/mdl-${_component})
	endif()
	install(FILES ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/NOTICE DESTINATION ${_doc_dir} COMPONENT ${_component})
endforeach()
install(FILES ${PROJECT_SOURCE_DIR}/README.md DESTINATION share/doc/mdl COMPONENT mdl)

# The licenses of the third-party software of each component (cmake/ThirdParty.cmake).
mdl_install_third_party_notices(${_components})

##########################################################################################
# General
##########################################################################################
set(MDL_PACKAGE_CONTACT "MDL maintainers <noreply@example.local>" CACHE STRING "The maintainer of the DEB and RPM packages")
set(CPACK_PACKAGE_NAME "mdl")
set(CPACK_PACKAGE_VENDOR "MDL")
set(CPACK_PACKAGE_CONTACT "${MDL_PACKAGE_CONTACT}")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/TheMrSheldon/Model-Description-Language")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Model Description Language: a language for neural networks and its compiler")
set(CPACK_PACKAGE_DESCRIPTION
	"MDL is a language for describing neural networks and a compiler (mdlc) that turns a model into a linkable object file plus bindings (C++ header, Python).")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "MDL")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_FILE_NAME "mdl-${CPACK_PACKAGE_VERSION}-${MDL_PACKAGE_PLATFORM}")
set(CPACK_STRIP_FILES ON)
set(CPACK_SOURCE_GENERATOR "") # (source archives are the GitHub release's)

if(WIN32)
	set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
	set(CPACK_GENERATOR "TGZ;productbuild")
else()
	set(CPACK_GENERATOR "TGZ;DEB;RPM")
endif()

# For MDLCPackOptions.cmake (CPack passes the CPACK_ variables on to it).
set(CPACK_MDL_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(CPACK_MDL_PLATFORM "${MDL_PACKAGE_PLATFORM}")
set(CPACK_MDL_COMPONENTS "${_components}")
set(CPACK_MDL_DEB_VERSION "${_deb_version}")
set(CPACK_MDL_RPM_VERSION "${_rpm_version}")
set(CPACK_MDL_RPM_RELEASE "${_rpm_release}")
set(CPACK_PROJECT_CONFIG_FILE "${PROJECT_SOURCE_DIR}/cmake/MDLCPackOptions.cmake")

##########################################################################################
# The components of the installers with a choice (NSIS, productbuild)
##########################################################################################
include(CPackComponent)
cpack_add_install_type(full DISPLAY_NAME "Full (compiler, SDK, and tools)")
cpack_add_install_type(compiler DISPLAY_NAME "Compiler (compiler and tools)")
cpack_add_install_type(tools DISPLAY_NAME "Tools only (formatter and language server)")

cpack_add_component(mdl DISPLAY_NAME "MDL" DESCRIPTION "The license" REQUIRED HIDDEN
	INSTALL_TYPES full compiler tools)
cpack_add_component(tools DISPLAY_NAME "Tools"
	DESCRIPTION "The formatter (mdl-format), the language server (mdls), the documentation generator (mdl-doc), and the standard library"
	INSTALL_TYPES full compiler tools)
if(MDL_ENABLE_COMPILER)
	cpack_add_component(runtime DISPLAY_NAME "Runtime libraries" DESCRIPTION "TVM's runtime libraries"
		HIDDEN INSTALL_TYPES full compiler)
	cpack_add_component(compiler DISPLAY_NAME "Compiler"
		DESCRIPTION "The compiler (mdlc) with the CPU kernel libraries (oneDNN)"
		DEPENDS tools runtime INSTALL_TYPES full compiler)
	cpack_add_component(sdk DISPLAY_NAME "SDK"
		DESCRIPTION "Headers, libraries and the CMake package (find_package(MDL)) to use compiled models in C++ programs"
		DEPENDS runtime INSTALL_TYPES full)
	if("cuda" IN_LIST _components)
		cpack_add_component(cuda DISPLAY_NAME "CUDA"
			DESCRIPTION "Compile for NVIDIA GPUs: the CUDA runtime and the cuBLAS and cuDNN kernels (needs the CUDA toolkit)"
			DEPENDS compiler sdk INSTALL_TYPES full)
	endif()
endif()

include(CPack)

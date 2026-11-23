#[=======================================================================[.rst:
MDL
---

Compile MDL models into object files that can be linked into C++ programs.

.. command:: mdl_add_model

  ::

    mdl_add_model(<target>
                  SOURCE <model.dlm>
                  [CONFIG <model.yml>]
                  [BUILD <name>]
                  [SHARED]
                  [KERNELS packages|none]
                  [KERNEL_PACKAGES <name>...]
                  [IMPORT_DIRS <dir>...]
                  [BINDINGS <type>...]
                  [OPTIONS <mdlc option>...])

  Compiles ``SOURCE`` with ``mdlc`` and creates a library target ``<target>`` that contains the compiled model and
  a generated C++ header ``<name>.hpp`` (``<name>`` is the file name of ``SOURCE`` without extension). Linking against
  ``<target>`` adds the header's directory to the include path and links the runtime dependencies, so a program only
  needs::

    #include "<name>.hpp"

  ``CONFIG``       hyperparameter values and build settings (``mdlc --arg-vals``, see docs/build-config.md).
  ``BUILD``        use the build settings ``builds.<name>`` of ``CONFIG`` (``mdlc --build``).
  ``SHARED``       build a shared library (e.g., to load it from Python with ``tvm_ffi.load_module``) instead of a
                   static one.
  ``KERNELS``      ``packages`` (default: the kernel packages, including the oneDNN package if it is built) or ``none`` (TVM-generated kernels only). (``onednn`` and ``tvm`` are former
                   names.)
  ``KERNEL_PACKAGES`` kernel packages built with ``mdl_kernel_package`` that the model may use (see docs/kernels.md);
                   they are linked into ``<target>``. Without it, the model uses no kernel packages.
  ``IMPORT_DIRS``  additional search paths for ``import`` statements.
  ``BINDINGS``     additional binding types to generate next to the object file (e.g. ``python-torch``).
  ``OPTIONS``      additional options passed to ``mdlc`` (they take precedence over the settings of ``CONFIG``).

  Imported files are tracked through a depfile, so the model is recompiled when any of them changes.
.. command:: mdl_kernel_package

  ::

    mdl_kernel_package(<name> MANIFEST <package.yml.in> SOURCES <source>... [LINK <library>...])

  Builds the kernel package ``<name>`` (see docs/kernels.md) from ``SOURCES`` into ``MDL_KERNELS_BUILD_DIR/<name>``:
  the library (target ``mdl_kernel_package_<name>``, linking ``LINK``) and the manifest, in which ``@MDL_KERNEL_ARTIFACT@``
  is the library's path and ``@MDL_KERNEL_OS@``, ``@MDL_KERNEL_ARCH@``, ``@MDL_KERNEL_ABI@`` the platform. With
  ``TARGETS <target>...`` instead of ``SOURCES``, the libraries are built by these targets and the manifest refers to
  them with generator expressions (``$<TARGET_FILE:...>``). Models use it with
  ``mdl_add_model(... KERNEL_PACKAGES <name>)``.
#]=======================================================================]

# The kernel directory of the kernel packages built by this project (see mdl_kernel_package).
if(NOT MDL_KERNELS_BUILD_DIR)
	set(MDL_KERNELS_BUILD_DIR "${CMAKE_BINARY_DIR}/mdl-kernels")
endif()

function(mdl_kernel_package name)
	cmake_parse_arguments(PARSE_ARGV 1 ARG "" "MANIFEST" "SOURCES;LINK;TARGETS")
	if(NOT ARG_MANIFEST OR (NOT ARG_SOURCES AND NOT ARG_TARGETS))
		message(FATAL_ERROR "mdl_kernel_package(${name}): MANIFEST and SOURCES (or TARGETS) are required")
	endif()
	set(dir "${MDL_KERNELS_BUILD_DIR}/${name}")
	# The manifest refers to the library as @MDL_KERNEL_ARTIFACT@ (its path relative to the package).
	set(MDL_KERNEL_ARTIFACT "lib/${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
	# ... and the platform it is built for as @MDL_KERNEL_OS@, @MDL_KERNEL_ARCH@, @MDL_KERNEL_ABI@.
	if(WIN32)
		set(MDL_KERNEL_OS windows)
	elseif(APPLE)
		set(MDL_KERNEL_OS darwin)
	else()
		set(MDL_KERNEL_OS linux)
	endif()
	string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" MDL_KERNEL_ARCH)
	if(MDL_KERNEL_ARCH MATCHES "^(amd64|x64)$")
		set(MDL_KERNEL_ARCH x86_64)
	elseif(MDL_KERNEL_ARCH MATCHES "^(arm64)$")
		set(MDL_KERNEL_ARCH aarch64)
	endif()
	if(MSVC)
		set(MDL_KERNEL_ABI msvc)
	elseif(APPLE)
		set(MDL_KERNEL_ABI darwin)
	else()
		set(MDL_KERNEL_ABI gnu)
	endif()
	set(target mdl_kernel_package_${name})
	if(ARG_TARGETS)
		# The libraries are built by these targets: the manifest refers to them as $<TARGET_FILE:...>.
		configure_file("${ARG_MANIFEST}" "${CMAKE_CURRENT_BINARY_DIR}/${name}.package.yml.in" @ONLY)
		file(GENERATE OUTPUT "${dir}/package.yml" INPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}.package.yml.in")
		add_library(${target} INTERFACE)
		target_link_libraries(${target} INTERFACE ${ARG_TARGETS})
		set_property(GLOBAL PROPERTY MDL_KERNEL_PACKAGE_BUILD_${name} ${ARG_TARGETS})
		return()
	endif()
	configure_file("${ARG_MANIFEST}" "${dir}/package.yml" @ONLY)
	set_property(GLOBAL PROPERTY MDL_KERNEL_PACKAGE_BUILD_${name} ${target})
	add_library(${target} STATIC ${ARG_SOURCES})
	set_target_properties(${target} PROPERTIES
		OUTPUT_NAME ${name} ARCHIVE_OUTPUT_DIRECTORY "${dir}/lib" POSITION_INDEPENDENT_CODE ON)
	foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
		string(TOUPPER "${config}" config)
		set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${config} "${dir}/lib")
	endforeach()
	if(ARG_LINK)
		target_link_libraries(${target} PUBLIC ${ARG_LINK})
	endif()
endfunction()

# A kernel package of the kernel directories (e.g. installed): its libraries as IMPORTED targets, from mdl-kernels deps.
function(mdl_import_kernel_package package)
	find_program(MDL_KERNELS_EXECUTABLE mdl-kernels)
	if(NOT MDL_KERNELS_EXECUTABLE)
		message(FATAL_ERROR "kernel package ${package}: not built by this project (mdl_kernel_package), and mdl-kernels "
			"was not found to import it")
	endif()
	set(file "${CMAKE_BINARY_DIR}/mdl-kernels-deps/${package}.cmake")
	execute_process(COMMAND "${MDL_KERNELS_EXECUTABLE}" deps --kernel-package ${package} OUTPUT_FILE "${file}"
		RESULT_VARIABLE result ERROR_VARIABLE error)
	if(NOT result EQUAL 0)
		message(FATAL_ERROR "kernel package ${package}: ${error}")
	endif()
	include("${file}")
endfunction()

function(mdl_add_model target)
	cmake_parse_arguments(PARSE_ARGV 1 ARG "SHARED" "SOURCE;CONFIG;KERNELS;BUILD" "IMPORT_DIRS;BINDINGS;OPTIONS;KERNEL_PACKAGES")
	if(NOT ARG_SOURCE)
		message(FATAL_ERROR "mdl_add_model(${target}): SOURCE is required")
	endif()

	get_filename_component(source "${ARG_SOURCE}" ABSOLUTE)
	get_filename_component(name "${ARG_SOURCE}" NAME_WE)
	set(outdir "${CMAKE_CURRENT_BINARY_DIR}/mdl/${target}")
	set(object "${outdir}/${name}${CMAKE_C_OUTPUT_EXTENSION}")
	set(header "${outdir}/${name}.hpp")
	file(MAKE_DIRECTORY "${outdir}")

	set(args "${source}" -o "${object}" --binding "cpp=${header}" --depfile "${object}.d")
	set(depends "${source}")
	if(ARG_CONFIG)
		get_filename_component(config "${ARG_CONFIG}" ABSOLUTE)
		list(APPEND args --arg-vals "${config}")
		list(APPEND depends "${config}")
	endif()
	if(ARG_BUILD)
		list(APPEND args --build "${ARG_BUILD}")
	endif()
	# The kernels: as requested; the kernel packages by default (with the oneDNN package if it is built).
	set(_none OFF)
	if(ARG_KERNELS STREQUAL "none" OR ARG_KERNELS STREQUAL "tvm")
		set(_none ON)
	endif()
	# The oneDNN kernels are a kernel package (kernels/onednn): used whenever it is built.
	if(NOT _none AND TARGET mdl_kernel_package_onednn AND NOT "onednn" IN_LIST ARG_KERNEL_PACKAGES)
		list(APPEND ARG_KERNEL_PACKAGES onednn)
	endif()
	# No package to link: none may be used (an unlisted package found elsewhere could not be linked).
	if(NOT ARG_KERNEL_PACKAGES)
		set(_none ON)
	endif()
	if(_none)
		list(APPEND args --kernels none)
	endif()
	# Kernel packages: only those that are linked (an unlisted package found elsewhere could not be linked).
	if(ARG_KERNEL_PACKAGES)
		list(APPEND args --kernels-dir "${MDL_KERNELS_BUILD_DIR}")
		foreach(package IN LISTS ARG_KERNEL_PACKAGES)
			list(APPEND args --kernel-package ${package})
			# (The libraries must exist when mdlc selects implementations.)
			get_property(build GLOBAL PROPERTY MDL_KERNEL_PACKAGE_BUILD_${package})
			if(NOT build)
				mdl_import_kernel_package(${package})
				get_property(build GLOBAL PROPERTY MDL_KERNEL_PACKAGE_BUILD_${package})
			endif()
			if(NOT build STREQUAL "IMPORTED")
				list(APPEND depends ${build})
			endif()
		endforeach()
	else()
		list(APPEND args --no-kernel-packages)
	endif()
	foreach(dir IN LISTS ARG_IMPORT_DIRS)
		get_filename_component(dir "${dir}" ABSOLUTE)
		list(APPEND args -i "${dir}")
	endforeach()
	foreach(binding IN LISTS ARG_BINDINGS)
		list(APPEND args --binding "${binding}")
	endforeach()
	list(APPEND args ${ARG_OPTIONS})

	if(TARGET mdlc)
		set(compiler "$<TARGET_FILE:mdlc>")
		list(APPEND depends mdlc)
	else()
		find_program(MDLC_EXECUTABLE mdlc REQUIRED)
		set(compiler "${MDLC_EXECUTABLE}")
	endif()

	# Development aid: check the compiler's shape inference against the backend (see mdl::shapes::infer).
	set(launcher)
	if(MDL_VERIFY_SHAPES)
		set(launcher ${CMAKE_COMMAND} -E env MDL_VERIFY_SHAPES=1)
	endif()

	add_custom_command(
		OUTPUT "${object}" "${header}"
		COMMAND ${launcher} ${compiler} ${args}
		DEPENDS ${depends}
		DEPFILE "${object}.d"
		COMMENT "Compiling MDL model ${ARG_SOURCE}"
		VERBATIM
	)
	set_source_files_properties("${object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

	if(ARG_SHARED)
		add_library(${target} SHARED "${object}" "${header}")
	else()
		add_library(${target} STATIC "${object}" "${header}")
	endif()
	set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)
	target_include_directories(${target} PUBLIC "${outdir}")

	# Runtime dependencies of compiled models: the TVM runtime (thread pool, workspace allocation; it links the TVM FFI
	# for the calling convention and errors), the profiling hooks of --profile, and the kernel packages.
	target_link_libraries(${target} PUBLIC mdl_runtime)
	if(TARGET mdl_profile)
		target_link_libraries(${target} PUBLIC mdl_profile)
	endif()
	foreach(package IN LISTS ARG_KERNEL_PACKAGES)
		target_link_libraries(${target} PUBLIC mdl_kernel_package_${package})
	endforeach()
endfunction()

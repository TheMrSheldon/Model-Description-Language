#[=======================================================================[.rst:
MDL
---

Compile MDL models into object files that can be linked into C++ programs.

.. command:: mdl_add_model

  ::

    mdl_add_model(<target>
                  SOURCE <model.dlm>
                  [CONFIG <model.yml>]
                  [SHARED]
                  [KERNELS onednn|tvm]
                  [IMPORT_DIRS <dir>...]
                  [BINDINGS <type>...]
                  [OPTIONS <mdlc option>...])

  Compiles ``SOURCE`` with ``mdlc`` and creates a library target ``<target>`` that contains the compiled model and
  a generated C++ header ``<name>.hpp`` (``<name>`` is the file name of ``SOURCE`` without extension). Linking against
  ``<target>`` adds the header's directory to the include path and links the runtime dependencies, so a program only
  needs::

    #include "<name>.hpp"

  ``CONFIG``       hyperparameter values (``mdlc --arg-vals``).
  ``SHARED``       build a shared library (e.g., to load it from Python with ``tvm_ffi.load_module``) instead of a
                   static one.
  ``KERNELS``      ``onednn`` (default if ``mdl_kernels`` was built with oneDNN) or ``tvm`` (no vendor kernels).
  ``IMPORT_DIRS``  additional search paths for ``import`` statements.
  ``BINDINGS``     additional binding types to generate next to the object file (e.g. ``python-torch``).
  ``OPTIONS``      additional options passed to ``mdlc``.

  Imported files are tracked through a depfile, so the model is recompiled when any of them changes.
#]=======================================================================]

function(mdl_add_model target)
	cmake_parse_arguments(PARSE_ARGV 1 ARG "SHARED" "SOURCE;CONFIG;KERNELS" "IMPORT_DIRS;BINDINGS;OPTIONS")
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
	if(NOT ARG_KERNELS)
		if(TARGET mdl_kernels)
			get_target_property(_defs mdl_kernels INTERFACE_COMPILE_DEFINITIONS)
			if(_defs AND "MDL_KERNELS_ONEDNN" IN_LIST _defs)
				set(ARG_KERNELS onednn)
			endif()
		endif()
		if(NOT ARG_KERNELS)
			set(ARG_KERNELS tvm)
		endif()
	endif()
	list(APPEND args --kernels "${ARG_KERNELS}")
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
	# for the calling convention and errors) and the vendor kernels.
	target_link_libraries(${target} PUBLIC mdl_runtime)
	if(ARG_KERNELS STREQUAL "onednn")
		target_link_libraries(${target} PUBLIC mdl_kernels)
	endif()
endfunction()

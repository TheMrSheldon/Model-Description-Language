# The commands: command-line parsing around the libraries.
# mdl_add_command(<name> <source> <libraries>...): installed in the component MDL_COMMAND_COMPONENT (default: tools).
function(mdl_add_command name source)
	add_executable(${name} ${source})
	target_link_libraries(${name} PRIVATE ${ARGN} CLI11::CLI11)
	if(NOT MDL_COMMAND_COMPONENT)
		set(MDL_COMMAND_COMPONENT tools)
	endif()
	install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MDL_COMMAND_COMPONENT})
endfunction()

mdl_add_command(mdl-format mdl-format.cpp mdl_format) # the formatter
mdl_add_command(mdls mdls.cpp mdls_lib)               # the language server
mdl_add_command(mdl-doc mdl-doc.cpp mdl_frontend)     # the API documentation of .dlm files (Doxygen pages)

if(MDL_ENABLE_COMPILER)
	# The compiler commands (component compiler) find TVM's libraries in <prefix>/lib/mdl.
	function(mdl_add_compiler_command name source)
		set(MDL_COMMAND_COMPONENT compiler)
		mdl_add_command(${name} ${source} ${ARGN})
		if(APPLE)
			set(origin "@loader_path")
		else()
			set(origin "$ORIGIN")
		endif()
		file(RELATIVE_PATH to_lib "/${CMAKE_INSTALL_BINDIR}" "/${MDL_INSTALL_LIBDIR}")
		set_target_properties(${name} PROPERTIES INSTALL_RPATH "${origin}/${to_lib}")
	endfunction()

	mdl_add_compiler_command(mdlc mdlc.cpp mdl nlohmann_json::nlohmann_json) # the compiler
	mdl_add_compiler_command(mdl-info mdl-info.cpp mdl) # information about TVM targets
	mdl_add_compiler_command(mdl-kernels mdl-kernels.cpp mdl nlohmann_json::nlohmann_json) # inspects kernel packages
endif()

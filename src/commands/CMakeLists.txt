# The commands: command-line parsing around the libraries.
function(mdl_add_command name source)
	add_executable(${name} ${source})
	target_link_libraries(${name} PRIVATE ${ARGN} CLI11::CLI11)
	install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT tools)
endfunction()

mdl_add_command(mdl-format mdl-format.cpp mdl_format) # the formatter
mdl_add_command(mdls mdls.cpp mdls_lib)               # the language server

if(MDL_ENABLE_COMPILER)
	mdl_add_command(mdlc mdlc.cpp mdl nlohmann_json::nlohmann_json) # the compiler
	mdl_add_command(mdl-info mdl-info.cpp mdl) # information about TVM targets
	mdl_add_command(mdl-kernels mdl-kernels.cpp mdl nlohmann_json::nlohmann_json) # inspects kernel packages
endif()

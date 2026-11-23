# mdls_lib: the language server (protocol, documents, one file per feature); the command is commands/mdls.cpp.
add_library(mdls_lib STATIC
	protocol.cpp
	document.cpp
	server.cpp
	semantic_tokens.cpp
	hover.cpp
	navigation.cpp
	completion.cpp
	formatting.cpp
	tokens.cpp
	partial.cpp
	models.cpp
	structure.cpp
	signature.cpp
	hints.cpp
	references.cpp
	folding.cpp
	actions.cpp
)
target_link_libraries(mdls_lib PUBLIC mdl_frontend mdl_format nlohmann_json::nlohmann_json)

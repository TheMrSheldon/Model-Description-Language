# mdls_lib: the language server (protocol, documents, features); the command is commands/mdls.cpp.
add_library(mdls_lib STATIC protocol.cpp semantic_tokens.cpp server.cpp)
target_link_libraries(mdls_lib PUBLIC mdl_frontend mdl_format nlohmann_json::nlohmann_json)

# Compile model.dlm into a library with the generated header model.hpp.
mdl_add_model(ffn_model SOURCE model.dlm CONFIG model.yml)

add_executable(01_ffn main.cpp)
target_link_libraries(01_ffn PRIVATE ffn_model)

add_test(NAME example_01_ffn COMMAND 01_ffn)

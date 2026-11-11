# Compile the BERT encoder layer (model.dlm, hyperparameters in model.yml) with the generated header model.hpp.
mdl_add_model(transformer_model SOURCE model.dlm CONFIG model.yml)

add_executable(02_transformer main.cpp)
target_link_libraries(02_transformer PRIVATE transformer_model)

# Correctness check against the reference implementation in main.cpp (few iterations).
add_test(NAME example_02_transformer COMMAND 02_transformer --seq 16 --iters 2)

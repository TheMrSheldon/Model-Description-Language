# An MLP configured by array hyperparameters (YAML lists), with a layer without bias and a default hyperparameter.
mdl_add_model(mlp_model SOURCE model.dlm CONFIG model.yml)
add_executable(08_hyperparameters main.cpp)
target_link_libraries(08_hyperparameters PRIVATE mlp_model)
add_test(NAME example_08_hyperparameters COMMAND 08_hyperparameters)

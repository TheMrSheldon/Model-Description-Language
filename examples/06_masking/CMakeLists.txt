# Masked tensors in the language: functions with `masked` parameters are compiled as masked functions (an additional
# input mask: bool[...]), and a masked argument selects the `masked` overload (here written with `x.data`/`x.mask`).
mdl_add_model(masking_model SOURCE model.dlm CONFIG model.yml)
add_executable(06_masking main.cpp)
target_link_libraries(06_masking PRIVATE masking_model)
add_test(NAME example_06_masking COMMAND 06_masking)

# std.var against a direct computation (population and sample variance over the last axis), and its gradient (the
# gradient rule of the builtin) against central finite differences.
mdl_add_model(statistics_model SOURCE model.dlm OPTIONS --ellipsis-rank 2)
mdl_add_model(statistics_grad_model SOURCE model.dlm OPTIONS --ellipsis-rank 2 --grad sample)
add_executable(statistics_var var.cpp)
target_link_libraries(statistics_var PRIVATE statistics_model)
add_test(NAME statistics_var COMMAND statistics_var)
add_executable(statistics_var_grad var_grad.cpp)
target_link_libraries(statistics_var_grad PRIVATE statistics_grad_model)
add_test(NAME statistics_var_grad COMMAND statistics_var_grad)

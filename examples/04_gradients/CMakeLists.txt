# Gradient builds of the same model (compile without --grad for inference):
#  - `forward` saves the residuals in a context buffer, `forward_backward(d_output)` accumulates the gradients,
#    and the objective `loss(x, target)` returns the loss and accumulates its gradients in one call;
#  - with --grad-fused, `forward(x, d_output)` returns the output and the gradients in one call.
# Both are checked against central finite differences.
mdl_add_model(gradients_model SOURCE model.dlm CONFIG model.yml OPTIONS --grad forward --grad loss --input-grads)
add_executable(04_gradients main.cpp)
target_link_libraries(04_gradients PRIVATE gradients_model)
add_test(NAME example_04_gradients COMMAND 04_gradients)

mdl_add_model(gradients_model_fused SOURCE model.dlm CONFIG model.yml OPTIONS --grad forward --input-grads --grad-fused)
add_executable(04_gradients_fused main.cpp)
target_compile_definitions(04_gradients_fused PRIVATE MDL_FUSED)
target_link_libraries(04_gradients_fused PRIVATE gradients_model_fused)
add_test(NAME example_04_gradients_fused COMMAND 04_gradients_fused)

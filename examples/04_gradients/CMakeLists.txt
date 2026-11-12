# Compiles `forward` and its gradient `forward_vjp` (all weights trainable) and checks the gradients against finite
# differences of `forward`.
mdl_add_model(gradients_model SOURCE model.dlm CONFIG model.yml OPTIONS --grad forward --input-grads)

add_executable(04_gradients main.cpp)
target_link_libraries(04_gradients PRIVATE gradients_model)
add_test(NAME example_04_gradients COMMAND 04_gradients)

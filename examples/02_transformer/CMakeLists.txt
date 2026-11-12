# Compile the BERT encoder (model.dlm; BERT-base hyperparameters in model.yml) with the generated header model.hpp.
mdl_add_model(transformer_model SOURCE model.dlm CONFIG model.yml)

add_executable(02_transformer main.cpp)
target_link_libraries(02_transformer PRIVATE transformer_model)

# Correctness check of the full 12-layer encoder against the reference implementation in main.cpp.
add_test(NAME example_02_transformer COMMAND 02_transformer --seq 16 --iters 1)

# A 2-layer model with TVM-generated kernels only (no oneDNN): tests the TVM legalizations and CPU schedules.
mdl_add_model(transformer_model_tvm SOURCE model.dlm CONFIG model_test.yml KERNELS tvm)
add_executable(02_transformer_tvm main.cpp)
target_link_libraries(02_transformer_tvm PRIVATE transformer_model_tvm)
add_test(NAME example_02_transformer_tvm COMMAND 02_transformer_tvm --seq 16 --iters 2)

# Batched 2-layer model: `...` stands for [batch, seq] (the model is unchanged).
mdl_add_model(transformer_model_batched SOURCE model.dlm CONFIG model_test.yml OPTIONS --ellipsis-rank 2)
add_executable(02_transformer_batched main.cpp)
target_link_libraries(02_transformer_batched PRIVATE transformer_model_batched)
add_test(NAME example_02_transformer_batched COMMAND 02_transformer_batched --batch 3 --seq 16 --iters 2)

# Gradient build of a tiny batched encoder: forward (saves into a context buffer) + forward_backward against finite
# differences.
mdl_add_model(transformer_model_grad SOURCE model.dlm CONFIG model_grad.yml OPTIONS --ellipsis-rank 2 --grad forward)
add_executable(02_transformer_grad grad_check.cpp)
target_link_libraries(02_transformer_grad PRIVATE transformer_model_grad)
add_test(NAME example_02_transformer_grad COMMAND 02_transformer_grad)

# Compile the BERT encoder (model.dlm; BERT-base hyperparameters in model.yml) with the generated header model.hpp.
mdl_add_model(transformer_model SOURCE model.dlm CONFIG model.yml)

add_executable(02_transformer main.cpp)
target_link_libraries(02_transformer PRIVATE transformer_model)

# Correctness check of the full 12-layer encoder against the reference implementation in main.cpp.
add_test(NAME example_02_transformer COMMAND 02_transformer --seq 16 --iters 1)

# A 2-layer model with TVM-generated kernels only (no oneDNN): tests the TVM legalizations and CPU schedules.
mdl_add_model(transformer_model_tvm SOURCE model.dlm CONFIG model_test.yml KERNELS none)
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

# The full model `Bert` (embeddings + encoder) against Hugging Face's BertModel (reference data written by
# export_bert_reference.py; `...` = [batch, seq]).
mdl_add_model(bert_model SOURCE model.dlm CONFIG model_bert_tiny.yml OPTIONS --ellipsis-rank 2)
add_executable(02_transformer_bert bert_check.cpp)
target_link_libraries(02_transformer_bert PRIVATE bert_model)
target_compile_definitions(02_transformer_bert PRIVATE MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny.safetensors")
add_test(NAME example_02_transformer_bert COMMAND 02_transformer_bert)

# Padded sequences: `Bert` as a masked function (`--mask forward`, an additional input mask: bool[batch, seq]) against
# Hugging Face's BertModel with the same attention_mask.
mdl_add_model(bert_model_masked SOURCE model.dlm CONFIG model_bert_tiny.yml
	OPTIONS --ellipsis-rank 2 --mask forward --mask pooled)
add_executable(02_transformer_bert_masked bert_check.cpp)
target_link_libraries(02_transformer_bert_masked PRIVATE bert_model_masked)
target_compile_definitions(02_transformer_bert_masked PRIVATE MDL_BERT_MASKED
	MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny_padded.safetensors")
add_test(NAME example_02_transformer_bert_masked COMMAND 02_transformer_bert_masked)

# The same gradient check for the masked encoder (`--mask forward`) with padded sequences.
mdl_add_model(transformer_model_grad_masked SOURCE model.dlm CONFIG model_grad.yml
	OPTIONS --ellipsis-rank 2 --mask forward --grad forward)
add_executable(02_transformer_grad_masked grad_check.cpp)
target_compile_definitions(02_transformer_grad_masked PRIVATE MDL_GRAD_MASKED)
target_link_libraries(02_transformer_grad_masked PRIVATE transformer_model_grad_masked)
add_test(NAME example_02_transformer_grad_masked COMMAND 02_transformer_grad_masked)

# Sequence classification (BERT + pooler + classifier) on padded sequences against Hugging Face's
# BertForSequenceClassification.
mdl_add_model(bert_cls_model SOURCE model.dlm CONFIG model_cls_tiny.yml BUILD inference)
add_executable(02_transformer_cls cls_check.cpp)
target_link_libraries(02_transformer_cls PRIVATE bert_cls_model)
target_compile_definitions(02_transformer_cls PRIVATE
	MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny_cls.safetensors")
add_test(NAME example_02_transformer_cls COMMAND 02_transformer_cls)

# Quantized weight matrices (int8, and int4 in groups of 16), quantized from the float checkpoint while loading: the
# logits within the quantization error of Hugging Face's (relative to the largest logit).
foreach(format int8 int4)
	mdl_add_model(bert_cls_${format}_model SOURCE model.dlm CONFIG model_cls_tiny.yml BUILD ${format})
	add_executable(02_transformer_cls_${format} cls_check.cpp)
	target_link_libraries(02_transformer_cls_${format} PRIVATE bert_cls_${format}_model)
	add_test(NAME example_02_transformer_cls_${format} COMMAND 02_transformer_cls_${format})
endforeach()
target_compile_definitions(02_transformer_cls_int8 PRIVATE MDL_QUANTIZED=0.1
	MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny_cls.safetensors")
target_compile_definitions(02_transformer_cls_int4 PRIVATE MDL_QUANTIZED=0.5
	MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny_cls.safetensors")

# Training the classifier: the objective `loss(ids, types, positions, labels)` as a masked gradient build, with labels
# [batch] next to the ids [batch, seq]; its value against Hugging Face's loss and its gradients against finite
# differences.
mdl_add_model(bert_cls_train_model SOURCE model.dlm CONFIG model_cls_tiny.yml BUILD train)
add_executable(02_transformer_cls_train cls_check.cpp)
target_link_libraries(02_transformer_cls_train PRIVATE bert_cls_train_model)
target_compile_definitions(02_transformer_cls_train PRIVATE MDL_CLS_TRAIN
	MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny_cls.safetensors")
add_test(NAME example_02_transformer_cls_train COMMAND 02_transformer_cls_train)

# The same with gradient checkpointing: the encoder layers save only their inputs, the backward pass recomputes them.
mdl_add_model(transformer_model_grad_checkpoint SOURCE model.dlm CONFIG model_grad.yml
	OPTIONS --ellipsis-rank 2 --mask forward --grad forward --checkpoint "layers.*")
add_executable(02_transformer_grad_checkpoint grad_check.cpp)
target_compile_definitions(02_transformer_grad_checkpoint PRIVATE MDL_GRAD_MASKED)
target_link_libraries(02_transformer_grad_checkpoint PRIVATE transformer_model_grad_checkpoint)
add_test(NAME example_02_transformer_grad_checkpoint COMMAND 02_transformer_grad_checkpoint)

# `float` as bfloat16 and float16 (`--float bf16|f16`): `Bert` against Hugging Face's float32 BertModel (the float32
# reference weights are converted when loading), with tolerances for the 16-bit types (relative to the largest output).
foreach(variant IN ITEMS "bf16;bfloat16;2e-2" "f16;float16;5e-3")
	list(GET variant 0 flag)
	list(GET variant 1 type)
	list(GET variant 2 tolerance)
	mdl_add_model(bert_model_${flag} SOURCE model.dlm CONFIG model_bert_tiny.yml OPTIONS --ellipsis-rank 2 --float ${flag})
	add_executable(02_transformer_bert_${flag} bert_check.cpp)
	target_link_libraries(02_transformer_bert_${flag} PRIVATE bert_model_${flag})
	target_compile_definitions(02_transformer_bert_${flag} PRIVATE MDL_BERT_FLOAT=${type} MDL_BERT_TOLERANCE=${tolerance}
		MDL_BERT_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/bert_tiny.safetensors")
	add_test(NAME example_02_transformer_bert_${flag} COMMAND 02_transformer_bert_${flag})
endforeach()

# Benchmarks for scripts/benchmark.sh (-DMDL_BENCHMARKS=ON; BERT-base takes a while to compile): the batched encoder,
# inference and training step, float32 and bfloat16, BERT-base (model.yml) and tiny (model_grad.yml).
if(MDL_BENCHMARKS)
	foreach(size IN ITEMS "base;model.yml" "tiny;model_grad.yml")
		list(GET size 0 sizeName)
		list(GET size 1 config)
		foreach(precision IN ITEMS "f32;" "bf16;bfloat16")
			list(GET precision 0 flag)
			list(GET precision 1 type)
			foreach(mode IN ITEMS infer train)
				set(name bench_${sizeName}_${mode}_${flag})
				set(options --ellipsis-rank 2 --float ${flag})
				if(mode STREQUAL "train")
					list(APPEND options --grad forward)
				endif()
				mdl_add_model(${name}_model SOURCE model.dlm CONFIG ${config} OPTIONS ${options})
				add_executable(${name} bench.cpp)
				target_link_libraries(${name} PRIVATE ${name}_model)
				if(mode STREQUAL "train")
					target_compile_definitions(${name} PRIVATE MDL_BENCH_TRAIN)
				endif()
				if(type)
					target_compile_definitions(${name} PRIVATE MDL_BENCH_FLOAT=${type})
				endif()
			endforeach()
		endforeach()
	endforeach()
endif()

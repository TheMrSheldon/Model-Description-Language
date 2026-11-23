# A text classifier with integer inputs (token ids, labels), an embedding table, and a cross-entropy objective.
# The gradient build replaces the objective `loss` by `loss(ids, labels) -> loss, d_<weight>...`; `forward` is
# unchanged. Checked against a reference cross-entropy and central finite differences.
mdl_add_model(classifier_model SOURCE model.dlm CONFIG model.yml OPTIONS --grad loss)
add_executable(05_classification main.cpp)
target_link_libraries(05_classification PRIVATE classifier_model)
add_test(NAME example_05_classification COMMAND 05_classification)

# End to end: a kernel package built from C (mdl_kernel_package) used by a compiled model (mdl_add_model).
mdl_kernel_package(scaled_add_c MANIFEST scaled_add_c/package.yml.in SOURCES scaled_add_c/scaled_add.c)
mdl_add_model(kernels_e2e_model SOURCE ../models/scaled_add.dlm KERNEL_PACKAGES scaled_add_c
	OPTIONS --kernel-report ${CMAKE_CURRENT_BINARY_DIR}/kernel-report.json)
add_executable(kernels_e2e main.cpp)
target_link_libraries(kernels_e2e PRIVATE kernels_e2e_model)
add_test(NAME kernels_e2e COMMAND kernels_e2e)

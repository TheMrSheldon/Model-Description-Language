# The kernel packages of this project (see docs/kernels.md), generated into MDL_KERNELS_BUILD_DIR with their libraries.

# oneDNN (mdl_kernels): the kernels that are already a package; the others are still called by the legalizer.
if(TARGET mdl_kernels AND MDL_ENABLE_ONEDNN)
	mdl_kernel_package(onednn MANIFEST onednn/package.yml.in TARGETS mdl_kernels)
endif()

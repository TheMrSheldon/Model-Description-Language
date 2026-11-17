# The kernel packages of this project (see docs/kernels.md), generated into MDL_KERNELS_BUILD_DIR with their libraries.

# oneDNN (mdl_kernels): in the build tree, the manifest refers to the built libraries; installed, the package is
# share/mdl/kernels/onednn with the libraries in its lib directory.
if(TARGET mdl_kernels)
	mdl_kernel_package(onednn MANIFEST onednn/package.yml.in TARGETS mdl_kernels)
	# The installed manifest: the configured one (onednn.package.yml.in) with the libraries relative to the package.
	file(READ "${CMAKE_CURRENT_BINARY_DIR}/onednn.package.yml.in" manifest)
	string(REGEX REPLACE "\\$<TARGET_FILE:([a-z_]+)>" "lib/$<TARGET_FILE_NAME:\\1>" manifest "${manifest}")
	file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/onednn.install.yml" CONTENT "${manifest}")
	install(FILES "${CMAKE_CURRENT_BINARY_DIR}/onednn.install.yml" DESTINATION share/mdl/kernels/onednn
		RENAME package.yml COMPONENT kernels)
	install(FILES $<TARGET_FILE:mdl_kernels> $<TARGET_FILE:dnnl> DESTINATION share/mdl/kernels/onednn/lib
		COMPONENT kernels)
endif()

# The kernel packages of this project (see docs/dev/spec/kernel-packages.md): each subdirectory builds its library and generates its
# manifest into MDL_KERNELS_BUILD_DIR. They are separate from the compiler, which only reads their manifests.

add_subdirectory(onednn) # oneDNN (MDL_ENABLE_ONEDNN)

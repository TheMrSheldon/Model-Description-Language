# Installs a build tree into a temporary prefix and builds and runs a program against the installation:
#   cmake -DBUILD_DIR=<build> -DCONSUMER=<tests/install> -DEXAMPLE=<examples/01_ffn> -DWORK=<dir> [-DCONFIG=<Release>]
#         [-DGENERATOR=<generator>] -P check_install.cmake
# (The installation is relocated after the install: it must not refer to the build tree.)
foreach(var BUILD_DIR CONSUMER EXAMPLE WORK)
	if(NOT ${var})
		message(FATAL_ERROR "${var} is required")
	endif()
endforeach()

function(run description)
	execute_process(COMMAND ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
	if(NOT result EQUAL 0)
		message(FATAL_ERROR "${description} failed (${result}):\n${output}")
	endif()
	message(STATUS "${description}: ok")
	set(output "${output}" PARENT_SCOPE)
endfunction()

file(REMOVE_RECURSE "${WORK}")
set(config)
if(CONFIG)
	set(config --config ${CONFIG})
endif()

run("install" ${CMAKE_COMMAND} --install "${BUILD_DIR}" ${config} --prefix "${WORK}/prefix")
# Nothing but ours: a dependency whose own install rules are not excluded (add it EXCLUDE_FROM_ALL) would put its files
# next to ours and into every package.
file(GLOB_RECURSE installed RELATIVE "${WORK}/prefix" "${WORK}/prefix/*")
foreach(file IN LISTS installed)
	if(NOT file MATCHES "^(bin/|include/mdl/|lib/mdl/|lib/cmake/MDL/|share/mdl/|share/doc/mdl(-[a-z]+)?/)")
		message(FATAL_ERROR "unexpected file in the installation: ${file}")
	endif()
endforeach()
# Move the installation: it has to work from anywhere.
file(RENAME "${WORK}/prefix" "${WORK}/moved")
set(prefix "${WORK}/moved")

foreach(tool mdlc mdl-kernels mdl-format mdls)
	find_program(exe_${tool} ${tool} PATHS "${prefix}/bin" NO_DEFAULT_PATH)
	if(NOT exe_${tool})
		message(FATAL_ERROR "the installation has no ${tool}")
	endif()
endforeach()
# The compiler must start without the CUDA toolkit and driver: NVRTC is loaded when a CUDA target is compiled.
find_program(READELF readelf)
if(UNIX AND READELF)
	foreach(file "${exe_mdlc}" "${prefix}/lib/mdl/libtvm_compiler.so" "${prefix}/lib/mdl/libtvm_runtime.so")
		if(EXISTS "${file}")
			run("dynamic section of ${file}" ${READELF} -d "${file}")
			if(output MATCHES "NEEDED[^\n]*(libnvrtc|libcuda|libcudart|libcublas|libcudnn)")
				message(FATAL_ERROR "${file} depends on a CUDA library:\n${output}")
			endif()
		endif()
	endforeach()
endif()
run("mdlc --help" "${exe_mdlc}" --help)
run("mdl-kernels list" "${exe_mdl-kernels}" list)

set(generator)
if(GENERATOR)
	set(generator -G "${GENERATOR}")
endif()
run("configure the consumer" ${CMAKE_COMMAND} -S "${CONSUMER}" -B "${WORK}/consumer" ${generator}
	"-DCMAKE_PREFIX_PATH=${prefix}" "-DEXAMPLE_DIR=${EXAMPLE}" -DCMAKE_BUILD_TYPE=Release)
run("build the consumer" ${CMAKE_COMMAND} --build "${WORK}/consumer" ${config})
find_program(consumer consumer PATHS "${WORK}/consumer" "${WORK}/consumer/Release" NO_DEFAULT_PATH)
run("run the consumer" "${consumer}")

# CUDA (a build with MDL_ENABLE_CUDA): the compiler targets it without a GPU (it needs the toolkit's NVRTC and headers).
if(CUDA)
	run("compile for CUDA" "${exe_mdlc}" "${EXAMPLE}/model.dlm" --arg-vals "${EXAMPLE}/model.yml"
		--target "cuda -arch=${CUDA_ARCH}" -o "${WORK}/cuda_model.o")
	if(NOT EXISTS "${WORK}/cuda_model.o")
		message(FATAL_ERROR "mdlc --target cuda did not write an object file")
	endif()
	# The runtime of CUDA targets is part of the installation (the cuda component).
	# (In script mode CMAKE_SHARED_LIBRARY_PREFIX/SUFFIX are not set, so match the name.)
	file(GLOB cuda_runtime "${prefix}/lib/mdl/*tvm_runtime_cuda*" "${prefix}/bin/*tvm_runtime_cuda*")
	if(NOT cuda_runtime)
		message(FATAL_ERROR "the installation has no CUDA runtime")
	endif()
endif()

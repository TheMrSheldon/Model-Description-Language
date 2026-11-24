# Runs mdl-doc on the test package tests/docs/mylib and compares the pages with tests/docs/expected (golden files).
# Usage: cmake -DMDL_DOC=<mdl-doc> -DSOURCE=<tests/docs> -DOUTPUT=<dir> -P check_mdl_doc.cmake
file(REMOVE_RECURSE "${OUTPUT}")
execute_process(COMMAND "${MDL_DOC}" "${SOURCE}/mylib" --output "${OUTPUT}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
	message(FATAL_ERROR "mdl-doc failed (${result})")
endif()
file(GLOB expected RELATIVE "${SOURCE}/expected" "${SOURCE}/expected/*.md")
file(GLOB actual RELATIVE "${OUTPUT}" "${OUTPUT}/*.md")
if(NOT expected STREQUAL actual)
	message(FATAL_ERROR "Expected the pages ${expected}, got ${actual}")
endif()
foreach(page IN LISTS expected)
	execute_process(
		COMMAND ${CMAKE_COMMAND} -E compare_files "${SOURCE}/expected/${page}" "${OUTPUT}/${page}"
		RESULT_VARIABLE different
	)
	if(different)
		file(READ "${OUTPUT}/${page}" content)
		message(FATAL_ERROR "${page} differs from the expected page; it is:\n${content}")
	endif()
endforeach()
